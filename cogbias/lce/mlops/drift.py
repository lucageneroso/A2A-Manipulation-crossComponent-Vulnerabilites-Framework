import json
import numpy as np
from typing import Dict, Any, List
from pathlib import Path

from cogbias.lce.core.concept import LatentConcept

class ConceptDriftDetector:
    """
    Advanced 4-Level Concept Drift Detection.
    Measures Geometric, Semantic, Causal, and Operational drift of Latent Software Components.
    """
    def __init__(self, baseline_concept: LatentConcept):
        self.baseline = baseline_concept
        
    def _measure_geometric_drift(self, new_concept: LatentConcept) -> float:
        v_base = self.baseline.geometry.mean_direction
        v_new = new_concept.geometry.mean_direction
        
        if v_base.shape != v_new.shape:
            return 1.0 # Max drift if dims mismatch and no projection exists
            
        n_b = np.linalg.norm(v_base)
        n_n = np.linalg.norm(v_new)
        
        if n_b == 0 or n_n == 0: return 1.0
        cos_sim = float(np.dot(v_base/n_b, v_new/n_n))
        return float(np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))) / 90.0 # Normalized [0, 1] (0 to 90 deg)

    def _measure_semantic_drift(self, output_similarity_score: float) -> float:
        # 1.0 means identical semantics. Drift is 1 - similarity.
        return max(0.0, 1.0 - output_similarity_score)

    def _measure_causal_drift(self, new_concept: LatentConcept) -> float:
        es_base = self.baseline.causality.effect_size
        es_new = new_concept.causality.effect_size
        if es_base == 0: return 1.0
        return min(1.0, abs(es_base - es_new) / es_base)

    def _measure_operational_drift(self, operational_metrics: Dict[str, Any]) -> float:
        # Mock operational drift based on saturation warnings in production telemetry
        total_uses = operational_metrics.get("total_interventions", 1)
        warnings = operational_metrics.get("total_warnings", 0)
        return min(1.0, warnings / total_uses)

    def detect_drift(self, 
                     new_concept: LatentConcept, 
                     semantic_similarity_score: float = 0.95,
                     production_telemetry: Dict[str, Any] = None) -> Dict[str, Any]:
        """Compares the new concept against the baseline across 4 dimensions."""
        
        geom_drift = self._measure_geometric_drift(new_concept)
        sem_drift = self._measure_semantic_drift(semantic_similarity_score)
        causal_drift = self._measure_causal_drift(new_concept)
        
        op_drift = 0.0
        if production_telemetry:
            op_drift = self._measure_operational_drift(production_telemetry)
            
        status = "STABLE"
        if geom_drift > 0.5 or sem_drift > 0.2 or causal_drift > 0.3 or op_drift > 0.3:
            status = "UNSTABLE"
        if geom_drift > 0.8 or causal_drift > 0.5:
            status = "SEVERE_DRIFT"
            
        report = {
            "baseline_version": self.baseline.identity.model_hash,
            "new_version": new_concept.identity.model_hash,
            "geometric_drift": round(geom_drift, 4),
            "semantic_drift": round(sem_drift, 4),
            "causal_drift": round(causal_drift, 4),
            "operational_drift": round(op_drift, 4),
            "status": status
        }
        return report

    def generate_report(self, new_concept: LatentConcept, output_path: str = "runs/m8_drift/concept_drift_report.json"):
        report = self.detect_drift(new_concept, production_telemetry={"total_interventions": 100, "total_warnings": 5})
        
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_file, "w") as f:
            json.dump(report, f, indent=2)
            
        return report
