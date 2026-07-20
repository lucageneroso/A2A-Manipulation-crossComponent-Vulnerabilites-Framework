import json
import os
from pathlib import Path
from datetime import datetime
from cogbias.core.schemas import Outcome

class DatasetDumper:
    """
    Produzione del layer "Experiment Log" sotto forma di file json.
    Supporta dataset complessi con metadati e ModelTrace completo.
    """
    def __init__(self, base_dir: str = "runs"):
        self.base_dir = Path(base_dir)
        self.today = datetime.now().strftime("%Y-%m-%d")
        
    def dump_outcome(self, condition_name: str, outcome: Outcome):
        target_dir = self.base_dir / self.today / condition_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = target_dir / "result.json"
        
        data = {
            "run_id": outcome.run_id,
            "scenario": outcome.scenario_id,
            "condition": condition_name,
            "outcome": {
                "tool_called": outcome.execution_result.tool_called,
                "tool_name": outcome.execution_result.tool_name,
                "policy_violated": outcome.policy_violated,
                "success": outcome.success
            }
        }
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def dump_baseline_run(self, dataset_name: str, outcome: Outcome):
        target_dir = self.base_dir / dataset_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = target_dir / f"{outcome.run_id}.json"
        
        data = {
            "run_id": outcome.run_id,
            "scenario": outcome.scenario_id,
            "condition": outcome.execution_result.metadata.get("condition_id", "unknown"),
            "outcome": {
                "tool_called": outcome.execution_result.tool_called,
                "tool_name": outcome.execution_result.tool_name,
                "policy_violated": outcome.policy_violated,
                "success": outcome.success
            },
            "model_trace": outcome.execution_result.metadata.get("model_trace", {})
        }
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
    def write_manifest(self, dataset_name: str, manifest_data: dict):
        target_dir = self.base_dir / dataset_name
        target_dir.mkdir(parents=True, exist_ok=True)
        with open(target_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
