import json
from dataclasses import dataclass
from typing import Dict, Any, List

from cogbias.lce.core.concept import LatentConcept

@dataclass
class ContractThresholds:
    min_bootstrap_stability: float = 0.80
    max_falsification_cosine: float = 0.25
    max_intrinsic_dimension: int = 20
    max_drift_angle: float = 45.0 # M8-B Latent Platform requirement

class ConceptContract:
    """
    CI/CD Quality Gate for Latent Concepts.
    Asserts that a LatentConcept artifact adheres to engineering and scientific standards
    before it can be merged or deployed to production.
    """
    def __init__(self, thresholds: ContractThresholds = None):
        self.thresholds = thresholds or ContractThresholds()
        
    def evaluate(self, concept: LatentConcept) -> Dict[str, Any]:
        report = {
            "concept_name": concept.identity.name,
            "version": concept.identity.version,
            "passed": True,
            "checks": []
        }
        
        # 1. Stability Check
        stability = concept.validation.bootstrap_stability
        if stability >= self.thresholds.min_bootstrap_stability:
            report["checks"].append({"rule": "Stability", "status": "PASS", "value": stability})
        else:
            report["checks"].append({"rule": "Stability", "status": "FAIL", "value": stability})
            report["passed"] = False
            
        # 2. Falsification Check
        falsification = concept.validation.falsification_score
        if falsification <= self.thresholds.max_falsification_cosine:
            report["checks"].append({"rule": "Falsification", "status": "PASS", "value": falsification})
        else:
            report["checks"].append({"rule": "Falsification", "status": "FAIL", "value": falsification})
            report["passed"] = False
            
        # 3. Geometry/Drift Check (if historical baseline exists)
        # Assuming drift_angle is populated during MLOps pipeline
        if hasattr(concept, 'mlops_drift_angle'):
            drift = concept.mlops_drift_angle
            if drift <= self.thresholds.max_drift_angle:
                report["checks"].append({"rule": "Drift", "status": "PASS", "value": drift})
            else:
                report["checks"].append({"rule": "Drift", "status": "FAIL", "value": drift})
                report["passed"] = False
                
        return report
        
    def enforce(self, concept: LatentConcept):
        """Raises an exception if the contract is violated (used in CI pipelines)."""
        evaluation = self.evaluate(concept)
        if not evaluation["passed"]:
            failed_checks = [c for c in evaluation["checks"] if c["status"] == "FAIL"]
            raise ValueError(f"ConceptContract Violation for {concept.identity.name}: {failed_checks}")
        return True
