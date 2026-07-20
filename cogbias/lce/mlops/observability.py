import json
import time
from pathlib import Path
from typing import Dict, Any, List

class LatentObservability:
    """
    Monitors LCE interventions at runtime.
    Logs telemetry for prompt, active concepts, weights, activation magnitudes, and saturation warnings.
    Produces latent_runtime_metrics.json.
    """
    def __init__(self, log_dir: str = "runs/m8_telemetry"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "latent_runtime_metrics.json"
        
        # Load existing if present
        self.metrics = []
        if self.log_file.exists():
            try:
                with open(self.log_file, "r") as f:
                    self.metrics = json.load(f)
            except json.JSONDecodeError:
                pass
                
    def log_inference(
        self, 
        prompt: str, 
        active_concepts: Dict[str, float], 
        activation_magnitude: float,
        saturation_warning: bool,
        interference_score: float,
        composition_risk: str
    ):
        record = {
            "timestamp": time.time(),
            "input": {
                "prompt_length": len(prompt),
                "active_concepts": active_concepts
            },
            "runtime": {
                "activation_magnitude": activation_magnitude,
                "saturation_warning": saturation_warning,
                "interference_score": interference_score,
                "composition_risk": composition_risk
            }
        }
        
        self.metrics.append(record)
        
        with open(self.log_file, "w") as f:
            json.dump(self.metrics, f, indent=2)
            
        if saturation_warning:
            print(f"[Observability Warning] Saturation detected for {active_concepts}")
            
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Calculates operational drift statistics for the drift detector."""
        total = len(self.metrics)
        if total == 0:
            return {"total_interventions": 0, "total_warnings": 0}
            
        warnings = sum(1 for m in self.metrics if m["runtime"]["saturation_warning"])
        
        return {
            "total_interventions": total,
            "total_warnings": warnings,
            "warning_rate": warnings / total
        }
