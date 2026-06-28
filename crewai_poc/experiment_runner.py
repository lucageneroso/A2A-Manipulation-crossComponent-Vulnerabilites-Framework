"""
experiment_runner.py — Framework statistico per l'esecuzione della campagna.
Esegue i trial in batch, gestisce i retry, pulisce lo stato, raccoglie i risultati.
"""

import os
import json
import time
import gc
from dataclasses import dataclass, field, asdict
from typing import List, Callable, Any
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential
import pandas as pd
import numpy as np
from scipy.stats import norm

from experiment_configs import ExperimentConfig
from instrumentation import tracer

# Assicuriamoci che l'IP locale sia consentito
os.environ["CREWAI_TOOLS_ALLOW_UNSAFE_PATHS"] = "true"

@dataclass
class TrialResult:
    trial_id: int
    config_name: str
    model_name: str
    success: bool
    researcher_compromised: bool
    social_engineering_attempted: bool
    tool_call_executed: bool
    action_reasoning_disconnect: bool
    duration_seconds: float
    timestamp: float = field(default_factory=time.time)
    error: str = None

def wilson_ci(k: int, n: int, alpha: float = 0.05):
    """Calcola l'intervallo di confidenza Wilson Score per una proporzione binomiale."""
    if n == 0:
        return (0.0, 1.0)
    z = norm.ppf(1 - alpha / 2)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))

class ExperimentRunner:
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=30))
    def _run_single_trial(self, trial_id: int, config: ExperimentConfig, trial_fn: Callable) -> TrialResult:
        print(f"    [Trial {trial_id+1}/{config.n_trials}] Avvio...")
        start_time = time.time()
        
        # Reset dello stato per il trial
        tracer.reset()
        
        # Esecuzione del trial
        result_data = trial_fn(config, trial_id)
        
        duration = time.time() - start_time
        
        # Salva la traccia granulare
        tracer.save_trace(self.output_dir / "tracing_events.jsonl", trial_id, config.name)
        
        return TrialResult(
            trial_id=trial_id,
            config_name=config.name,
            model_name=config.model,
            success=result_data.get("pwned", False),
            researcher_compromised=result_data.get("researcher_compromised", False),
            social_engineering_attempted=result_data.get("social_engineering_attempted", False),
            tool_call_executed=result_data.get("tool_call_executed", False),
            action_reasoning_disconnect=result_data.get("action_reasoning_disconnect", False),
            duration_seconds=duration,
        )

    def run_experiment(self, configs: List[ExperimentConfig], trial_fn: Callable) -> pd.DataFrame:
        all_results = []
        total_trials = sum(c.n_trials for c in configs)
        completed = 0

        print("="*60)
        print(f"AVVIO CAMPAGNA SPERIMENTALE ({total_trials} trial totali)")
        print("="*60)

        for config in configs:
            print(f"\n>> Configurazone: {config.name} ({config.n_trials} trials)")
            config_results = []

            for trial_id in range(config.n_trials):
                try:
                    res = self._run_single_trial(trial_id, config, trial_fn)
                except Exception as e:
                    print(f"    [Trial {trial_id+1}] Fallito definitivamente: {e}")
                    res = TrialResult(
                        trial_id=trial_id, config_name=config.name, model_name=config.model,
                        success=False, researcher_compromised=False, social_engineering_attempted=False,
                        tool_call_executed=False, action_reasoning_disconnect=False,
                        duration_seconds=0.0, error=str(e)
                    )
                
                config_results.append(res)
                all_results.append(res)
                completed += 1
                
                # Checkpoint ogni 5 trial
                if completed % 5 == 0:
                    self._save_checkpoint(all_results, completed, total_trials)
                    
                gc.collect()

            # Sommario parziale
            successes = sum(1 for r in config_results if r.success)
            n_valid = sum(1 for r in config_results if not r.error)
            ci = wilson_ci(successes, n_valid)
            print(f"   Risultato {config.name}: {successes}/{n_valid} successi ({successes/max(n_valid,1):.1%}) - CI: [{ci[0]:.3f}, {ci[1]:.3f}]")

        df = pd.DataFrame([asdict(r) for r in all_results])
        ts = int(time.time())
        df.to_csv(self.output_dir / f"campaign_results_{ts}.csv", index=False)
        print(f"\nCampagna terminata. Risultati salvati in {self.output_dir}")
        return df

    def _save_checkpoint(self, results, completed, total):
        data = {"completed": completed, "total": total, "results": [asdict(r) for r in results]}
        with open(self.output_dir / "checkpoint.json", "w") as f:
            json.dump(data, f, indent=2)
