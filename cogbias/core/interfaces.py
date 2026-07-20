from abc import ABC, abstractmethod
from typing import Any, Dict, List
from .schemas import ExperimentContext

class Stage(ABC):
    """
    Componente base della Pipeline.
    Un orchestratore cieco passa il contesto allo Stage, che lo processa e lo restituisce.
    """
    @abstractmethod
    def execute(self, context: ExperimentContext) -> ExperimentContext:
        pass

class ProtocolInterface(ABC):
    """
    Un Protocollo definisce cosa vogliamo testare. 
    Il suo unico scopo è generare una lista di istanze concrete (Run) tramite la permutazione
    di configurazioni, seed, scenari e payload. Non esegue nulla.
    """
    @abstractmethod
    def generate_runs(self) -> List[ExperimentContext]:
        pass

class ModelInterface(ABC):
    """
    Interfaccia pura del Modello per isolare il Receiver dall'implementazione (es. Transformers).
    """
    @abstractmethod
    def tokenize(self, text: str) -> Any:
        pass

    @abstractmethod
    def prepare_input(self, representation: Any) -> Dict[str, Any]:
        pass

    @abstractmethod
    def generate(self, inputs: Any, config: Dict[str, Any]) -> Any:
        pass
