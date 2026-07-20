import hashlib
import torch
from typing import Any

from cogbias.core.schemas import Representation, Payload
from cogbias.stages.representation.stage import RepresentationStrategy
from cogbias.model_adapter.transformers_adapter import TransformersAdapter

class LatentRepresentation(RepresentationStrategy):
    """
    Converte il prompt in una rappresentazione latente (es. token ids, embeddings), 
    calcolando la norma e mantenendo la traccia.
    Richiede l'accesso al modello (encoder) per la conversione.
    """
    def __init__(self, adapter: TransformersAdapter):
        self.adapter = adapter

    def encode(self, formatted_prompt: str, original_payload: Payload) -> Representation:
        source_hash = hashlib.sha256(formatted_prompt.encode("utf-8")).hexdigest()
        
        # In this initial latent representation, we use input_ids.
        # Alternatively, we could compute embeddings explicitly (inputs_embeds).
        # We will extract input_ids as the latent representation for now, 
        # or we can extract the embeddings explicitly if requested.
        # For M5.1, we start by providing token_ids as "latent" dimension
        # or actual embeddings if we want a continuous representation.
        
        # Let's extract token ids as basic latent data, 
        # and compute the norm if we extract embeddings.
        tokenizer = self.adapter.tokenizer
        model = self.adapter.model
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        
        # Calculate position_ids explicitly
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        else:
            position_ids = torch.arange(input_ids.shape[1], dtype=torch.long, device=model.device).unsqueeze(0)
            
        with torch.no_grad():
            if hasattr(model, "get_input_embeddings"):
                embed_layer = model.get_input_embeddings()
                embeddings = embed_layer(input_ids)
                embedding_source_layer = embed_layer.__class__.__name__
            else:
                embeddings = input_ids # fallback
                embedding_source_layer = "none"
                
            norm = torch.linalg.norm(embeddings).item()
            mean = embeddings.mean().item()
            std = embeddings.std().item()
            shape = list(embeddings.shape)
            dtype_str = str(embeddings.dtype)
                
        # Store all components in data so Adapter can prepare the input precisely
        data = {
            "inputs_embeds": embeddings.cpu(),
            "attention_mask": attention_mask.cpu() if attention_mask is not None else None,
            "position_ids": position_ids.cpu()
        }
        
        return Representation(
            type="latent",
            data=data,
            dimension=f"{embeddings.shape[-1]}", # Embedding dimension
            norm=norm,
            mean=mean,
            std=std,
            tensor_shape=shape,
            dtype=dtype_str,
            requires_position_ids=True,
            requires_attention_mask=True,
            embedding_source_layer=embedding_source_layer,
            source_hash=source_hash
        )
