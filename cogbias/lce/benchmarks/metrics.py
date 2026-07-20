import numpy as np
from typing import Dict, Any, List

class ConceptMetrics:
    """
    Computes LCE-specific metrics for benchmark evaluation.
    """
    @staticmethod
    def compute_concept_alignment_score(target_state: Dict[str, float], actual_scores: Dict[str, float]) -> float:
        """
        quanto l'output mantiene il profilo concettuale richiesto.
        Returns a normalized score [0, 1].
        """
        if not target_state:
            return 1.0
            
        errors = []
        for concept, expected in target_state.items():
            actual = actual_scores.get(concept, 0.0)
            errors.append(abs(expected - actual))
            
        return max(0.0, 1.0 - np.mean(errors))

    @staticmethod
    def compute_concept_leakage(target_state: Dict[str, float], actual_scores: Dict[str, float]) -> float:
        """
        se un concetto attiva dimensioni indesiderate.
        Returns the sum of activations for concepts NOT in the target state.
        """
        leakage = 0.0
        for concept, actual in actual_scores.items():
            if concept not in target_state:
                leakage += abs(actual)
        return leakage

    @staticmethod
    def compute_semantic_stability(embeddings: List[np.ndarray]) -> float:
        """
        variazione del profilo latente tra generazioni diverse.
        Returns the average cosine similarity between multiple generations of the same prompt.
        """
        if len(embeddings) < 2:
            return 1.0
            
        sims = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                v1 = embeddings[i]
                v2 = embeddings[j]
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 > 0 and n2 > 0:
                    sim = np.dot(v1/n1, v2/n2)
                    sims.append(float(sim))
        
        return float(np.mean(sims)) if sims else 1.0

    @staticmethod
    def compute_interference_score(concept_a: str, concept_b: str, interaction_matrix: Dict[str, Dict[str, Any]]) -> float:
        """
        misura conflitti tra concetti combinati.
        """
        try:
            cos_sim = interaction_matrix[concept_a][concept_b]["cosine_similarity"]
            # High negative cosine means high interference
            if cos_sim < -0.3:
                return abs(cos_sim)
            return 0.0
        except KeyError:
            return 0.0
