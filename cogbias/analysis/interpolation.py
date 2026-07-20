from typing import Any, Tuple
import torch
from cogbias.core.schemas import Representation

class SemanticInterpolation:
    def __init__(self):
        pass
        
    def interpolate(self, rep1: Representation, rep2: Representation, alpha: float) -> Representation:
        """
        Interpolate between two latent representations using alpha.
        rep_interp = (1 - alpha) * rep1 + alpha * rep2
        """
        if rep1.type != "latent" or rep2.type != "latent":
            raise ValueError("Interpolation only supported for latent representations.")
            
        new_data = {}
        for k in rep1.data:
            v1 = rep1.data[k]
            v2 = rep2.data.get(k, None)
            
            if v1 is not None and v2 is not None and isinstance(v1, torch.Tensor) and k == "inputs_embeds":
                # Ensure same shape
                if v1.shape != v2.shape:
                    # Pad or truncate if sequence lengths differ
                    min_seq_len = min(v1.shape[1], v2.shape[1])
                    v1_adj = v1[:, :min_seq_len, :]
                    v2_adj = v2[:, :min_seq_len, :]
                else:
                    v1_adj = v1
                    v2_adj = v2
                    
                interp_v = (1.0 - alpha) * v1_adj + alpha * v2_adj
                new_data[k] = interp_v
            else:
                # For non-embedding tensors (like attention_mask), fallback to rep1
                if isinstance(v1, torch.Tensor) and "inputs_embeds" in new_data:
                    seq_len = new_data["inputs_embeds"].shape[1]
                    if v1.dim() > 1:
                        new_data[k] = v1[:, :seq_len]
                    else:
                        new_data[k] = v1
                else:
                    new_data[k] = v1
                    
        return Representation(
            type="latent",
            data=new_data,
            dimension=rep1.dimension,
            norm=torch.linalg.norm(new_data["inputs_embeds"]).item() if "inputs_embeds" in new_data else 0.0,
            embedding_source_layer=rep1.embedding_source_layer,
            source_hash=f"interp_{alpha}_{rep1.source_hash[:8]}_{rep2.source_hash[:8]}"
        )
