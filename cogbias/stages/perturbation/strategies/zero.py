import torch
import torch.nn.functional as F
from typing import Tuple

from cogbias.core.schemas import Representation, PerturbationTrace
from cogbias.stages.perturbation.strategies.base import PerturbationStrategy

class ZeroPerturbation(PerturbationStrategy):
    """
    Aggiunge un vettore zero. Utilizzato per testare la neutralità matematica dell'infrastruttura di perturbazione.
    """
    def apply(self, representation: Representation) -> Tuple[Representation, PerturbationTrace]:
        # Copia profonda dei dati
        new_data = dict(representation.data)
        
        inputs_embeds = new_data["inputs_embeds"].clone()
        original_norm = torch.linalg.norm(inputs_embeds).item()
        
        # Aggiungi zero (in realtà non fa nulla, ma simula il processo)
        perturbation = torch.zeros_like(inputs_embeds)
        perturbed_embeds = inputs_embeds + perturbation
        
        new_data["inputs_embeds"] = perturbed_embeds
        
        # Calculate cosine delta
        flat_original = inputs_embeds.view(1, -1)
        flat_perturbed = perturbed_embeds.view(1, -1)
        # Handle zero vectors which might cause NaNs in cosine similarity
        if torch.linalg.norm(flat_original) > 0 and torch.linalg.norm(flat_perturbed) > 0:
            cosine_delta = F.cosine_similarity(flat_original, flat_perturbed).item()
        else:
            cosine_delta = 1.0
        
        trace = PerturbationTrace(
            type="zero",
            alpha=0.0,
            original_norm=original_norm,
            perturbation_norm=0.0,
            relative_delta_norm=0.0,
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
