import json
from typing import Dict, Any

class ActivationRecorder:
    """
    Registra metadati e statistiche sulle attivazioni (mean, std, norm) 
    senza salvare l'intero tensore HDF5 per preservare spazio su disco e VRAM.
    """
    def __init__(self):
        self.records = []
        
    def record(self, hook_name: str, tensor: Any) -> Dict[str, Any]:
        import torch
        if not torch.is_tensor(tensor):
            raise ValueError("ActivationRecorder expected a torch.Tensor")
            
        tensor_float = tensor.float()
        
        stat = {
            "hook_name": hook_name,
            "shape": list(tensor.shape),
            "mean": float(torch.mean(tensor_float).item()),
            "std": float(torch.std(tensor_float).item()),
            "norm": float(torch.norm(tensor_float).item()),
            "cosine_reference": None,
            "storage_ref": None
        }
        self.records.append(stat)
        return stat
