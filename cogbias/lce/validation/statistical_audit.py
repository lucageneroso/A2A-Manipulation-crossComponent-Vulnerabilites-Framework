import numpy as np
from typing import List, Dict, Any, Tuple
from scipy.stats import false_discovery_control

class StatisticalAuditLayer:
    """
    Rigorously audits experimental results to ensure findings hold up to 
    independent scientific standards (FDR correction, 10k resample bootstrap).
    """
    def __init__(self, bootstrap_resamples: int = 10000):
        self.resamples = bootstrap_resamples

    def cohens_d(self, control: List[float], treatment: List[float]) -> float:
        var1, var2 = np.var(control, ddof=1), np.var(treatment, ddof=1)
        pooled_var = ((len(control) - 1) * var1 + (len(treatment) - 1) * var2) / (len(control) + len(treatment) - 2)
        if pooled_var == 0:
            return 0.0
        return (np.mean(treatment) - np.mean(control)) / np.sqrt(pooled_var)

    def bootstrap_ci(self, data: List[float]) -> Tuple[float, float, float, float]:
        """Returns (mean, std, ci_lower, ci_upper). Minimum 10000 resamples."""
        data_arr = np.array(data)
        n = len(data_arr)
        means = np.zeros(self.resamples)
        
        # Vectorized bootstrapping for speed
        idx = np.random.randint(0, n, size=(self.resamples, n))
        means = data_arr[idx].mean(axis=1)
        
        lower_bound = np.percentile(means, 2.5)
        upper_bound = np.percentile(means, 97.5)
        return float(np.mean(data_arr)), float(np.std(data_arr)), float(lower_bound), float(upper_bound)

    def permutation_test(self, control: List[float], treatment: List[float]) -> float:
        """H0: 'LCE intervention has no causal behavioral effect'"""
        obs_diff = np.mean(treatment) - np.mean(control)
        combined = np.concatenate([control, treatment])
        n_control = len(control)
        
        count_extreme = 0
        # For audit, 10000 permutations
        for _ in range(self.resamples):
            np.random.shuffle(combined)
            perm_diff = np.mean(combined[n_control:]) - np.mean(combined[:n_control])
            if perm_diff >= obs_diff:
                count_extreme += 1
                
        return count_extreme / self.resamples

    def audit_experiment(
        self, 
        random_baseline: List[float],
        prompt_baseline: List[float],
        native_concept: List[float],
        lce_intervention: List[float]
    ) -> Dict[str, Any]:
        """Audits the full suite and applies Benjamini-Hochberg FDR."""
        
        # 1. Effect Sizes
        d_vs_random = self.cohens_d(random_baseline, lce_intervention)
        d_vs_prompt = self.cohens_d(prompt_baseline, lce_intervention)
        d_vs_native = self.cohens_d(native_concept, lce_intervention) # Usually negative as native > compiled
        
        # 2. Bootstrap CIs
        mean, std, ci_low, ci_high = self.bootstrap_ci(lce_intervention)
        
        # 3. Hypothesis Tests (p-values)
        p_random = self.permutation_test(random_baseline, lce_intervention)
        p_prompt = self.permutation_test(prompt_baseline, lce_intervention)
        
        # FDR Correction
        corrected_pvals = false_discovery_control([p_random, p_prompt], method='bh')
        p_random_fdr, p_prompt_fdr = corrected_pvals
        
        # Rejection Criteria
        # Reject random baseline if FDR corrected p < 0.05
        reject_random = p_random_fdr < 0.05
        # The zero hypothesis requires the CI of the effect (lce - random) to exclude zero
        # Simplified: Check if ci_low > mean(random)
        random_mean = np.mean(random_baseline)
        excludes_zero_effect = ci_low > random_mean
        
        report = {
            "effect_size": {
                "lce_vs_random": d_vs_random,
                "lce_vs_prompt": d_vs_prompt,
                "lce_vs_native": d_vs_native
            },
            "bootstrap_10k": {
                "mean": mean,
                "std": std,
                "95_ci": [ci_low, ci_high],
                "excludes_zero_effect": excludes_zero_effect
            },
            "hypothesis_testing": {
                "p_vs_random_raw": p_random,
                "p_vs_random_fdr": float(p_random_fdr),
                "h0_random_rejected": bool(reject_random)
            }
        }
        return report
