import torch
import numpy as np
from typing import Dict, Any

from cogbias.model_adapter.transformers_adapter import TransformersAdapter
from cogbias.lce.atlas.atlas import LatentAtlas

class LatentController:
    """
    Declarative API for Latent Concept Engineering.
    Abstracts away tensor math and manages concept steering based on the LatentAtlas.
    """
    def __init__(self, adapter: TransformersAdapter, atlas: LatentAtlas):
        self.adapter = adapter
        self.atlas = atlas
        self.target_state: Dict[str, float] = {}
        self.active_hooks = []
        
    def set(self, **kwargs):
        """
        Declaratively set concept strengths.
        Example: controller.set(Authority=0.8, Helpfulness=0.6, Uncertainty=-0.2)
        """
        for concept_name, strength in kwargs.items():
            if concept_name not in self.atlas.concepts:
                raise ValueError(f"Concept '{concept_name}' not found in LatentAtlas.")
            self.target_state[concept_name] = float(strength)
            
    def simulate(self) -> Dict[str, Any]:
        """
        Simulates the intervention, predicting interference and saturation.
        """
        report = {
            "requested_state": self.target_state,
            "warnings": [],
            "expected_shifts": {}
        }
        
        # Check for antagonistic collisions
        names = list(self.target_state.keys())
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                c1, c2 = names[i], names[j]
                v1 = self.atlas.concepts[c1].geometry.mean_direction
                v2 = self.atlas.concepts[c2].geometry.mean_direction
                
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 == 0 or n2 == 0: continue
                
                cos_sim = np.dot(v1/n1, v2/n2)
                
                # If they are highly antagonistic and both have positive strengths
                if cos_sim < -0.4 and self.target_state[c1] > 0 and self.target_state[c2] > 0:
                    report["warnings"].append(
                        f"Collision Warning: {c1} and {c2} are highly antagonistic (cosine: {cos_sim:.2f}). "
                        f"Applying both positively may cause representation collapse or cancellation."
                    )
                    
        return report
        
    def apply(self, layer_idx: int = -1):
        """
        Compiles the target state into a composite tensor and injects it into the model.
        """
        if not self.target_state:
            return
            
        # 1. Compile Composite Vector
        # We assume dim is known from the first concept
        first_concept = next(iter(self.atlas.concepts.values()))
        dim = first_concept.geometry.mean_direction.shape[0]
        
        composite_vector = np.zeros(dim)
        
        for name, strength in self.target_state.items():
            concept = self.atlas.concepts[name]
            v = concept.geometry.mean_direction
            # Normalize vector to unit length to ensure strength multiplier is consistent
            v_norm = v / (np.linalg.norm(v) + 1e-9)
            composite_vector += (v_norm * strength)
            
        composite_tensor = torch.tensor(composite_vector, dtype=self.adapter.model.dtype, device=self.adapter.model.device)
        
        # 2. Inject
        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
                
            # Add to all tokens in the sequence
            hidden_states = hidden_states + composite_tensor
            
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            return hidden_states

        target_layer = self.adapter.model.model.layers[layer_idx]
        hook_handle = target_layer.register_forward_hook(steering_hook)
        self.active_hooks.append(hook_handle)
        
    def clear(self):
        """Removes all active steering hooks."""
        for hook in self.active_hooks:
            hook.remove()
        self.active_hooks = []
        self.target_state = {}
