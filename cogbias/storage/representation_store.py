import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple
import datetime
from safetensors.torch import save_file, load_file
import torch

from cogbias.core.schemas import Representation, RepresentationArtifact

class RepresentationStore:
    """
    Gestisce la serializzazione e deserializzazione sicura di Representation
    verso disco usando .safetensors.
    """
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.tensors_dir = self.base_dir / "tensors"
        self.metadata_dir = self.base_dir / "metadata"
        
        self.tensors_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def _hash_tensors(self, tensors: Dict[str, torch.Tensor]) -> str:
        """Calcola un hash robusto del dizionario di tensori."""
        hasher = hashlib.sha256()
        # Sort keys per determinismo
        for k in sorted(tensors.keys()):
            t = tensors[k]
            if t is not None:
                hasher.update(k.encode("utf-8"))
                # Hashing a CPU numpy view (view as uint8 to handle bfloat16)
                hasher.update(t.cpu().view(torch.uint8).numpy().tobytes())
        return hasher.hexdigest()

    def save(self, representation: Representation, rep_id: str, model_id: str, tokenizer_id: str) -> RepresentationArtifact:
        """
        Salva la representation su disco.
        """
        if representation.type != "latent":
            raise ValueError("Only latent representations can be stored using safetensors")
            
        data_dict = representation.data
        if not isinstance(data_dict, dict) or "inputs_embeds" not in data_dict:
            raise ValueError("Representation data must be a dictionary with 'inputs_embeds'")

        # Rimuoviamo i None perché safetensors accetta solo tensori effettivi
        safe_tensors = {k: v.contiguous() for k, v in data_dict.items() if v is not None}
        
        # Hash per integrity check
        tensor_hash = self._hash_tensors(safe_tensors)
        
        tensor_path = self.tensors_dir / f"{rep_id}.safetensors"
        metadata_path = self.metadata_dir / f"{rep_id}.json"
        
        # Salva tensori
        save_file(safe_tensors, tensor_path)
        
        # Crea artefatto
        artifact = RepresentationArtifact(
            id=rep_id,
            tensor_path=str(tensor_path),
            sha256=tensor_hash,
            dtype=representation.dtype,
            shape=representation.tensor_shape,
            source_prompt_hash=representation.source_hash,
            created_at=datetime.datetime.utcnow().isoformat() + "Z",
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            embedding_layer=representation.embedding_source_layer,
            metadata={
                "norm": representation.norm,
                "mean": representation.mean,
                "std": representation.std,
                "requires_position_ids": representation.requires_position_ids,
                "requires_attention_mask": representation.requires_attention_mask,
                "dimension": representation.dimension
            }
        )
        
        # Salva metadata
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(artifact.model_dump_json(indent=4))
            
        return artifact

    def load(self, artifact_path: str) -> Tuple[RepresentationArtifact, Representation]:
        """
        Carica un artefatto di rappresentazione dal JSON dei metadati.
        """
        with open(artifact_path, "r", encoding="utf-8") as f:
            metadata_dict = json.load(f)
            
        artifact = RepresentationArtifact(**metadata_dict)
        
        # Load tensors
        tensors = load_file(artifact.tensor_path)
        
        # Re-verify hash
        current_hash = self._hash_tensors(tensors)
        if current_hash != artifact.sha256:
            raise ValueError(f"Hash integrity check failed for {artifact.id}")
            
        # Reconstruct Representation object
        representation = Representation(
            type="latent",
            data=tensors,
            dimension=artifact.metadata["dimension"],
            norm=artifact.metadata["norm"],
            mean=artifact.metadata["mean"],
            std=artifact.metadata["std"],
            tensor_shape=artifact.shape,
            dtype=artifact.dtype,
            requires_position_ids=artifact.metadata["requires_position_ids"],
            requires_attention_mask=artifact.metadata["requires_attention_mask"],
            embedding_source_layer=artifact.embedding_layer,
            source_hash=artifact.source_prompt_hash
        )
        
        return artifact, representation

    def verify_hash(self, artifact: RepresentationArtifact) -> bool:
        """
        Verifica che i tensori sul disco corrispondano all'hash nell'artefatto.
        """
        try:
            tensors = load_file(artifact.tensor_path)
            return self._hash_tensors(tensors) == artifact.sha256
        except Exception:
            return False
