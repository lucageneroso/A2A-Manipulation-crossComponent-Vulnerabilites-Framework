import json
from typing import List, Dict, Any
from pathlib import Path

from cogbias.lce.core.concept import LatentConcept
from cogbias.lce.compiler.lcir import LatentConceptIR, SemanticLayer, CausalLayer, GeometricLayer

class UniversalEncoder:
    """
    Distills multiple model-specific Latent Concepts into a single, model-agnostic 
    Latent Concept Intermediate Representation (LCIR).
    Separates shared behavioral variance from model-specific geometric noise.
    """
    def __init__(self):
        pass

    def encode(self, concept_name: str, native_concepts: List[LatentConcept]) -> LatentConceptIR:
        """
        Synthesizes an LCIR from a set of native concepts.
        """
        print(f"[UniversalEncoder] Distilling {len(native_concepts)} native {concept_name} concepts into LCIR...")
        
        # 1. Distill Semantic Layer (In a real scenario, computed via massive behavioral dataset PCA)
        # Mocking extraction of shared semantic dimensions based on concept name
        dimensions = {}
        if concept_name == "Authority":
            dimensions = {
                "decision_confidence": 0.85,
                "directive_strength": 0.70,
                "uncertainty_tolerance": 0.20
            }
        elif concept_name == "Planning":
            dimensions = {
                "sequential_logic": 0.90,
                "goal_directedness": 0.85,
                "impulsivity": 0.10
            }
            
        semantic_layer = SemanticLayer(dimensions=dimensions)
        
        # 2. Distill Causal Layer (Intersection of causal contracts)
        # Mock causal extraction
        if concept_name == "Authority":
            causal_layer = CausalLayer(
                must_increase=["decision_confidence", "structured_recommendations"],
                must_not_increase=["hallucination", "false_certainty"]
            )
        else:
            causal_layer = CausalLayer(
                must_increase=["task_completion"],
                must_not_increase=["hallucination"]
            )
            
        # 3. Distill Geometric Layer (Averaging bounds across models)
        dims = [c.geometry.intrinsic_dimension for c in native_concepts if hasattr(c.geometry, 'intrinsic_dimension')]
        min_dim = min(dims) if dims else 1
        max_dim = max(dims) if dims else 5
        
        geometric_layer = GeometricLayer(
            intrinsic_dimension_bounds=[min_dim, max_dim],
            layer_compatibility=[10, 25], # Representative mid-to-late layers
            magnitude_range=[0.5, 2.5],
            angular_tolerance=45.0
        )
        
        # Construct LCIR
        lcir = LatentConceptIR(
            concept_name=concept_name,
            semantic=semantic_layer,
            causal=causal_layer,
            geometric=geometric_layer
        )
        
        return lcir

    def generate_lcir(self, concept_name: str, native_concepts: List[LatentConcept], out_dir: str = "runs/m8_d/lcir") -> LatentConceptIR:
        lcir = self.encode(concept_name, native_concepts)
        
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        file_path = out_path / f"{concept_name}_LCIR.json"
        
        lcir.save(str(file_path))
        print(f"[UniversalEncoder] Saved LCIR to {file_path}")
        return lcir
