"""
experiment_configs.py — Configurazioni per la campagna statistica e Ablation Study.
"""

from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class ExperimentConfig:
    name: str
    model: str
    n_trials: int = 30
    temperature: float = 0.2
    parameters: Dict[str, Any] = field(default_factory=dict)

CONFIGS = [
    # Golden Baseline (Replica esatta del PoC originale)
    ExperimentConfig(
        name="golden_baseline_poc",
        model="ollama/llama3.1:8b",
        n_trials=30,
        temperature=0.2,
        parameters={
            "prompt_style": "original", # Usa i prompt lunghi
            "tool_mode": "react",       # Disabilita Native Tool Calling
            "payload_type": "long",     # Usa il payload originale in index.html
            "defense_level": "strong"   # Usa la backstory originale (paranoica) dell'Executor
        }
    ),
    
    # Eventuali configurazioni future per Ablation Study
    # verranno aggiunte qui in Fase 2.
]
