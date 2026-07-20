import yaml
from typing import Dict, Any

class ScenarioLoader:
    """
    Carica e parsa gli scenari sperimentali YAML garantendo un'interfaccia 
    strutturata indipendente dal modello.
    """
    @staticmethod
    def load(filepath: str) -> Dict[str, Any]:
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
