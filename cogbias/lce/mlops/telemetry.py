import json
import time
from pathlib import Path
from typing import Dict, Any

class ConceptTelemetry:
    """
    Observability logger for Latent Concept interventions in production.
    Logs every steering event, including the composite state applied and any saturation warnings.
    """
    def __init__(self, log_dir: str = "runs/mlops/telemetry"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "intervention_log.jsonl"
        
    def log_intervention(self, prompt: str, target_state: Dict[str, float], warnings: list, timestamp: float = None):
        if timestamp is None:
            timestamp = time.time()
            
        record = {
            "timestamp": timestamp,
            "prompt_length": len(prompt),
            "target_state": target_state,
            "warnings": warnings,
            "status": "APPLIED"
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
            
    def get_statistics(self) -> Dict[str, Any]:
        """Parses the telemetry log to return usage statistics."""
        if not self.log_file.exists():
            return {"total_interventions": 0}
            
        stats = {"total_interventions": 0, "concept_usage": {}, "total_warnings": 0}
        with open(self.log_file, "r") as f:
            for line in f:
                record = json.loads(line)
                stats["total_interventions"] += 1
                stats["total_warnings"] += len(record.get("warnings", []))
                
                for concept in record.get("target_state", {}).keys():
                    stats["concept_usage"][concept] = stats["concept_usage"].get(concept, 0) + 1
                    
        return stats
