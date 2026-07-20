import numpy as np
from typing import List, Dict, Any, Tuple

class StatisticalValidationSuite:
    """
    Rigorously validates the causal efficacy of Latent Concepts for publication.
    Provides bootstrapping, effect size (Cohen's d), and permutation tests.
    """
    def __init__(self, n_iterations: int = 1000):
        self.n_iterations = n_iterations

    def compute_cohens_d(self, control: List[float], treatment: List[float]) -> float:
        """Calculates Cohen's d effect size."""
        n1, n2 = len(control), len(treatment)
        var1, var2 = np.var(control, ddof=1), np.var(treatment, ddof=1)
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        
        # Guard against zero variance
        if pooled_var == 0:
            return 0.0
            
        pooled_std = np.sqrt(pooled_var)
        return (np.mean(treatment) - np.mean(control)) / pooled_std

    def bootstrap_ci(self, data: List[float], ci: float = 95.0) -> Tuple[float, float]:
        """Calculates bootstrap confidence intervals for the mean."""
        n = len(data)
        means = np.empty(self.n_iterations)
        data_arr = np.array(data)
        for i in range(self.n_iterations):
            sample = np.random.choice(data_arr, size=n, replace=True)
            means[i] = np.mean(sample)
        
        lower_bound = np.percentile(means, (100 - ci) / 2.0)
        upper_bound = np.percentile(means, 100 - (100 - ci) / 2.0)
        return float(lower_bound), float(upper_bound)

    def permutation_test(self, control: List[float], treatment: List[float]) -> float:
        """Calculates p-value using a permutation test."""
        observed_diff = np.mean(treatment) - np.mean(control)
        combined = np.concatenate([control, treatment])
        n_control = len(control)
        
        count_extreme = 0
        for _ in range(self.n_iterations):
            np.random.shuffle(combined)
            perm_control = combined[:n_control]
            perm_treatment = combined[n_control:]
            perm_diff = np.mean(perm_treatment) - np.mean(perm_control)
            if perm_diff >= observed_diff:
                count_extreme += 1
                
        p_value = count_extreme / self.n_iterations
        return float(p_value)

    def generate_validation_report(
        self, 
        control_scores: List[float], 
        treatment_scores: List[float],
        random_baseline_scores: List[float],
        native_upper_bound_scores: List[float]
    ) -> Dict[str, Any]:
        """
        Generates a comprehensive scientific validation report.
        """
        # Treatment vs Control
        d_treatment = self.compute_cohens_d(control_scores, treatment_scores)
        ci_treatment = self.bootstrap_ci(treatment_scores)
        p_treatment = self.permutation_test(control_scores, treatment_scores)
        
        # Random vs Control
        d_random = self.compute_cohens_d(control_scores, random_baseline_scores)
        p_random = self.permutation_test(control_scores, random_baseline_scores)
        
        # Native vs Control
        d_native = self.compute_cohens_d(control_scores, native_upper_bound_scores)
        
        return {
            "control_mean": float(np.mean(control_scores)),
            "compiled_treatment": {
                "mean": float(np.mean(treatment_scores)),
                "95_ci": ci_treatment,
                "cohens_d": float(d_treatment),
                "p_value": float(p_treatment),
                "significant": bool(p_treatment < 0.05)
            },
            "random_baseline": {
                "mean": float(np.mean(random_baseline_scores)),
                "cohens_d": float(d_random),
                "p_value": float(p_random),
                "significant": bool(p_random < 0.05)
            },
            "native_upper_bound": {
                "mean": float(np.mean(native_upper_bound_scores)),
                "cohens_d": float(d_native)
            },
            "transfer_efficiency": float(d_treatment / d_native) if d_native > 0 else 0.0
        }
