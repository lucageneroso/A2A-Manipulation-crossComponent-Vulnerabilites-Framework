import torch
import torch.nn.functional as F
from typing import Tuple

from cogbias.core.schemas import Representation, PerturbationTrace
from cogbias.stages.perturbation.strategies.base import PerturbationStrategy

class RandomPerturbation(PerturbationStrategy):
    """
    Aggiunge un vettore di rumore casuale alla rappresentazione latente.
    La formula utilizzata è: x' = x + alpha * epsilon
    dove ||epsilon||_2 = ||x||_2. 
    Quindi, l'intensità della perturbazione ||delta x||_2 è esattamente alpha * ||x||_2.
    """
    def __init__(self, alpha: float, seed: int = None):
        self.alpha = alpha
        self.seed = seed

    def apply(self, representation: Representation) -> Tuple[Representation, PerturbationTrace]:
        # Copia profonda dei dati
        new_data = dict(representation.data)
        
        inputs_embeds = new_data["inputs_embeds"].clone()
        original_norm = torch.linalg.norm(inputs_embeds).item()
        
        # Genera rumore casuale (distribuzione normale)
        # Assicuriamoci che abbia lo stesso dtype di inputs_embeds se possibile
        # ma torch.randn potrebbe richiedere float32
        if self.seed is not None:
            generator = torch.Generator(device=inputs_embeds.device)
            generator.manual_seed(self.seed)
            epsilon_raw = torch.randn(*inputs_embeds.shape, dtype=torch.float32, device=inputs_embeds.device, generator=generator)
        else:
            epsilon_raw = torch.randn_like(inputs_embeds, dtype=torch.float32)
            
        epsilon_raw_norm = torch.linalg.norm(epsilon_raw).item()
        
        # Normalizza epsilon per avere la stessa norma di x
        if epsilon_raw_norm > 0:
            epsilon = epsilon_raw * (original_norm / epsilon_raw_norm)
        else:
            epsilon = epsilon_raw
            
        # Scala la perturbazione per alpha e converti al dtype originale
        perturbation = (self.alpha * epsilon).to(inputs_embeds.dtype)
        perturbation_norm = torch.linalg.norm(perturbation).item()
        
        # Applica perturbazione
        perturbed_embeds = inputs_embeds + perturbation
        new_data["inputs_embeds"] = perturbed_embeds
        
        # Calculate cosine delta
        flat_original = inputs_embeds.view(1, -1).to(torch.float32)
        flat_perturbed = perturbed_embeds.view(1, -1).to(torch.float32)
        if torch.linalg.norm(flat_original) > 0 and torch.linalg.norm(flat_perturbed) > 0:
            cosine_delta = F.cosine_similarity(flat_original, flat_perturbed).item()
        else:
            cosine_delta = 1.0
            
        relative_delta_norm = perturbation_norm / original_norm if original_norm > 0 else 0.0
        
        trace = PerturbationTrace(
            type="random",
            alpha=self.alpha,
            original_norm=original_norm,
            perturbation_norm=perturbation_norm,
            relative_delta_norm=relative_delta_norm,
            cosine_delta=cosine_delta,
            target_layer=representation.embedding_source_layer or "unknown"
        )
        
        # Crea una nuova representation
        new_representation = Representation(
            type=representation.type,
            data=new_data,
            dimension=representation.dimension,
            norm=torch.linalg.norm(perturbed_embeds).item(),
            mean=perturbed_embeds.mean().item(),
            std=perturbed_embeds.std().item(),
            tensor_shape=list(perturbed_embeds.shape),
            dtype=str(perturbed_embeds.dtype),
            requires_position_ids=representation.requires_position_ids,
            requires_attention_mask=representation.requires_attention_mask,
            embedding_source_layer=representation.embedding_source_layer,
            source_hash=representation.source_hash
        )
        
        return new_representation, trace
