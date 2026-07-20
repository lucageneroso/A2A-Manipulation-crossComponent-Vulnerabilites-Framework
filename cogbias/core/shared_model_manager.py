from typing import Any, Callable, Dict

class SharedModelManager:
    """
    Gestore delle istanze dei modelli. Assicura che la stessa istanza
    possa essere riutilizzata senza duplicazioni in VRAM, supportando 
    la quantizzazione e l'offloading in futuro.
    """
    def __init__(self):
        self._shared_instances: Dict[str, Any] = {}
        
    def load(self, model_id: str, provider_func: Callable[[], Any]):
        """
        Carica un modello tramite una funzione provider se non è già caricato.
        """
        if model_id not in self._shared_instances:
            self._shared_instances[model_id] = provider_func()
            
    def get(self, model_id: str) -> Any:
        """
        Restituisce l'istanza condivisa del modello.
        """
        if model_id not in self._shared_instances:
            raise ValueError(f"Model '{model_id}' is not loaded. Call load() first.")
        return self._shared_instances[model_id]
        
    def release(self, model_id: str):
        """
        Rilascia la memoria associata al modello.
        """
        if model_id in self._shared_instances:
            del self._shared_instances[model_id]
