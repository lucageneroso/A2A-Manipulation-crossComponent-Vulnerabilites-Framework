import numpy as np
import random
from typing import List, Tuple, Dict

from cogbias.model_adapter.transformers_adapter import TransformersAdapter
from cogbias.lce.core.concept import LatentConcept
from cogbias.lce.core.state import ConceptState
from cogbias.lce.discovery.extractor import ConceptExtractor

class ConceptValidator:
    """
    Automated Falsification Suite (The Gauntlet).
    Validates a LatentConcept using statistical rigor.
    """
    def __init__(self, adapter: TransformersAdapter):
        self.adapter = adapter
        self.extractor = ConceptExtractor(adapter)
        
    def validate(
        self, 
        concept: LatentConcept, 
        fake_pairs: Dict[str, List[Tuple[str, str]]],
        layer_idx: int = -1,
        n_bootstraps: int = 50,
        cosine_threshold: float = 0.5
    ) -> Tuple[LatentConcept, Dict]:
        """
        Runs the falsification suite.
        Transitions the concept to VALIDATED if it passes all checks.
        Returns a tuple: (LatentConcept, ValidationReport)
        """
        if concept.state.value < ConceptState.DISCOVERED.value:
            raise ValueError("Concept must be at least DISCOVERED to be validated.")
            
        print(f"\n[LCE Validator] Validating concept: {concept.identity.name} v{concept.identity.version}")
        
        real_v = concept.geometry.mean_direction
        real_norm = np.linalg.norm(real_v)
        
        report = {
            "concept": concept.identity.name,
            "status": "PENDING",
            "reasons": []
        }
        
        # 1. Falsification (Negative Controls)
        print("  Running Falsification Controls...")
        negative_controls_results = {}
        passed_falsification = True
        
        for fake_name, pairs in fake_pairs.items():
            fake_v, _ = self.extractor._extract_mean_difference(pairs, layer_idx)
            cos_sim = np.dot(real_v / real_norm, fake_v / np.linalg.norm(fake_v))
            negative_controls_results[fake_name] = float(cos_sim)
            
            # If a fake concept aligns too strongly with our concept, the concept is rejected
            if abs(cos_sim) > 0.3:
                print(f"    [FAIL] Fake concept '{fake_name}' aligned too strongly (cosine: {cos_sim:.3f})")
                passed_falsification = False
                
                interpretation = "Concept may represent positive affect or a generic correlation."
                if fake_name == "Label_Shuffle":
                    interpretation = "Positive and negative examples have high semantic overlap or lexical leakage. The concept is not contrastive enough."
                elif fake_name == "Happiness" or fake_name == "Vacation":
                    interpretation = "Concept is entangled with positive emotion/sentiment."
                    
                report["reasons"].append({
                    "test": "negative_control",
                    "failure": f"{fake_name} correlation",
                    "severity": "high",
                    "interpretation": interpretation
                })
            else:
                print(f"    [PASS] Fake concept '{fake_name}' orthogonal (cosine: {cos_sim:.3f})")
                
        concept.validation.negative_controls = negative_controls_results
        
        # 2. Bootstrap Stability
        print("  Running Bootstrap Stability...")
        base_pairs = list(zip(concept.semantics.negative_examples, concept.semantics.positive_examples))
        cosines = []
        
        for i in range(n_bootstraps):
            # Resample with replacement
            sampled_pairs = [random.choice(base_pairs) for _ in range(len(base_pairs))]
            boot_v, _ = self.extractor._extract_mean_difference(sampled_pairs, layer_idx)
            
            cos_sim = np.dot(real_v / real_norm, boot_v / np.linalg.norm(boot_v))
            cosines.append(float(cos_sim))
            
        mean_cos = float(np.mean(cosines))
        std_cos = float(np.std(cosines))
        
        print(f"    Bootstrap Mean Cosine: {mean_cos:.3f} ± {std_cos:.3f}")
        concept.validation.bootstrap_stability = mean_cos
        concept.validation.confidence_intervals = {
            "bootstrap_cosine": [mean_cos - 1.96*std_cos, mean_cos + 1.96*std_cos]
        }
        
        passed_bootstrap = mean_cos >= cosine_threshold
        
        if not passed_bootstrap:
            report["reasons"].append({
                "test": "bootstrap_stability",
                "failure": f"Mean cosine {mean_cos:.3f} below threshold {cosine_threshold}",
                "severity": "high",
                "interpretation": "The dataset is too small or noisy. The geometric direction changes heavily depending on which examples are sampled."
            })
        
        if passed_bootstrap and passed_falsification:
            print("[LCE Validator] Concept PASSED validation gauntlet.")
            concept.transition_state(ConceptState.VALIDATED)
            report["status"] = "PASSED"
        else:
            print("[LCE Validator] Concept FAILED validation gauntlet. State remains uncertified.")
            report["status"] = "FAILED"
            
        return concept, report
