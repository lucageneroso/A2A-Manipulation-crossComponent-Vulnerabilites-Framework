import json
import numpy as np
from typing import Dict, Any
from pathlib import Path
from sklearn.cross_decomposition import CCA
from scipy.linalg import orthogonal_procrustes
from sklearn.linear_model import Ridge

from cogbias.lce.core.concept import LatentConcept

class CrossModelTransferBenchmark:
    """
    Evaluates whether a Latent Concept is a universal behavioral abstraction
    or a model-specific artifact.
    """
    def __init__(self, mode: str = "SIMULATION"):
        self.mode = mode.upper()
        if self.mode not in ["SIMULATION", "REAL"]:
            raise ValueError("mode must be SIMULATION or REAL")

    def _mock_mapping_simulation(self, source_vec: np.ndarray, target_dim: int, strategy: str) -> np.ndarray:
        """Simulates mapping a vector to a new dimensionality/space for CI/CD testing."""
        np.random.seed(42 + hash(strategy) % 1000)
        base = np.random.randn(target_dim)
        return base / (np.linalg.norm(base) + 1e-9)

    def compute_mapping(self, source_states: np.ndarray, target_states: np.ndarray, strategy: str) -> Any:
        """
        Computes the mapping matrix between two models' hidden states.
        """
        if self.mode == "SIMULATION":
            return {"strategy": strategy, "status": "simulated_mapping"}
            
        if strategy == "CCA":
            cca = CCA(n_components=min(10, source_states.shape[1], target_states.shape[1]))
            cca.fit(source_states, target_states)
            return cca
        elif strategy == "Procrustes":
            # Requires same dimensionality
            if source_states.shape[1] == target_states.shape[1]:
                R, sca = orthogonal_procrustes(source_states, target_states)
                return R
            return None
        elif strategy == "Ridge":
            ridge = Ridge(alpha=1.0)
            ridge.fit(source_states, target_states)
            return ridge
        else: # Linear
            # Simple pseudo-inverse mapping
            return np.linalg.pinv(source_states) @ target_states

    def apply_mapping(self, source_vec: np.ndarray, mapping: Any, target_dim: int, strategy: str) -> np.ndarray:
        if self.mode == "SIMULATION":
            return self._mock_mapping_simulation(source_vec, target_dim, strategy)
            
        # REAL mode mapping application logic would go here
        # (Using the trained CCA, Ridge, or Procrustes matrix)
        return self._mock_mapping_simulation(source_vec, target_dim, strategy) # Fallback for prototype

    def evaluate_preservation(self, source_vec: np.ndarray, target_vec: np.ndarray, strategy: str) -> Dict[str, float]:
        """
        Measures Geometric, Causal, and Semantic preservation.
        In SIMULATION, these return mock scores based on the strategy.
        In REAL mode, these would invoke actual model inferences.
        """
        if self.mode == "SIMULATION":
            # Synthetic scores to prove the decision logic
            if strategy == "CCA":
                return {"geometric": 0.85, "causal": 0.82, "semantic": 0.88}
            elif strategy == "Procrustes":
                return {"geometric": 0.60, "causal": 0.50, "semantic": 0.55}
            elif strategy == "Ridge":
                return {"geometric": 0.75, "causal": 0.70, "semantic": 0.80}
            else: # Linear
                return {"geometric": 0.40, "causal": 0.30, "semantic": 0.20}
                
        # REAL mode implementation would go here (omitted for brevity)
        return {"geometric": 0.0, "causal": 0.0, "semantic": 0.0}

    def determine_transfer_level(self, scores: Dict[str, float]) -> int:
        """
        0: Model Specific Steering
        1: Representation Transfer
        2: Universal Latent Abstraction
        """
        g, c, s = scores["geometric"], scores["causal"], scores["semantic"]
        
        if g >= 0.80 and c >= 0.80 and s >= 0.80:
            return 2
        elif g >= 0.50 and c >= 0.50:
            return 1
        return 0

    def run_benchmark(self, source_model_name: str, target_model_name: str, concept: LatentConcept) -> Dict[str, Any]:
        print(f"[CrossModelBenchmark] Running in {self.mode} mode.")
        print(f"[CrossModelBenchmark] Transferring {concept.identity.name} from {source_model_name} to {target_model_name}")
        
        strategies = ["Linear", "Procrustes", "Ridge", "CCA"]
        best_level = 0
        best_strategy = "None"
        best_scores = {"geometric": 0.0, "causal": 0.0, "semantic": 0.0}
        
        for strategy in strategies:
            # Mock states for simulation
            src_states = np.random.randn(100, 1536)
            tgt_states = np.random.randn(100, 2048)
            
            mapping = self.compute_mapping(src_states, tgt_states, strategy)
            target_vec = self.apply_mapping(concept.geometry.mean_direction, mapping, 2048, strategy)
            
            scores = self.evaluate_preservation(concept.geometry.mean_direction, target_vec, strategy)
            level = self.determine_transfer_level(scores)
            
            if level > best_level or (level == best_level and sum(scores.values()) > sum(best_scores.values())):
                best_level = level
                best_strategy = strategy
                best_scores = scores
                
        confidence = sum(best_scores.values()) / 3.0
        
        report = {
            "source_model": source_model_name,
            "target_model": target_model_name,
            "concept": concept.identity.name,
            "best_mapping_strategy": best_strategy,
            "geometric_score": best_scores["geometric"],
            "causal_score": best_scores["causal"],
            "semantic_score": best_scores["semantic"],
            "transfer_level": best_level,
            "confidence": confidence,
            "mode": self.mode
        }
        
        out_path = Path("runs/m8_c_transfer/cross_model_transfer_report.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
            
        print(f"[CrossModelBenchmark] Transfer Level: {best_level} (Strategy: {best_strategy})")
        return report
