import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple

from cogbias.model_adapter.transformers_adapter import TransformersAdapter
from cogbias.lce.atlas.atlas import LatentAtlas

class ConceptInteractionAnalyzer:
    """
    Analyzes what happens when concepts are combined (composition).
    """
    def __init__(self, adapter: TransformersAdapter, atlas: LatentAtlas):
        self.adapter = adapter
        self.atlas = atlas

    def analyze_pair(self, concept_a: str, concept_b: str) -> Dict[str, Any]:
        """
        Analyzes the geometric and theoretical interaction of two concepts.
        (Behavioral shift analysis via text generation is deferred to a full pipeline).
        """
        if concept_a not in self.atlas.concepts or concept_b not in self.atlas.concepts:
            raise ValueError(f"Concepts {concept_a} or {concept_b} not found in Atlas.")
            
        ca = self.atlas.concepts[concept_a]
        cb = self.atlas.concepts[concept_b]
        
        va = ca.geometry.mean_direction
        vb = cb.geometry.mean_direction
        
        norm_a = np.linalg.norm(va)
        norm_b = np.linalg.norm(vb)
        
        cos_sim = float(np.dot(va / norm_a, vb / norm_b))
        
        # Determine theoretical interaction based on geometry
        # This acts as a proxy before runtime behavioral validation
        if cos_sim > 0.4:
            interaction_class = "Synergistic (High Overlap)"
        elif cos_sim < -0.4:
            interaction_class = "Antagonistic (High Interference)"
        elif -0.1 <= cos_sim <= 0.1:
            interaction_class = "Independent (Orthogonal)"
        else:
            interaction_class = "Complex (Partial Interference)"
            
        return {
            "concept_a": concept_a,
            "concept_b": concept_b,
            "cosine_similarity": cos_sim,
            "interaction_class": interaction_class
        }

    def generate_interaction_matrix(self, output_path: str):
        """Generates the concept_interaction_matrix.json report."""
        names = list(self.atlas.concepts.keys())
        matrix = {name: {} for name in names}
        
        for name_a in names:
            for name_b in names:
                if name_a == name_b:
                    matrix[name_a][name_b] = {"interaction_class": "Identity", "cosine_similarity": 1.0}
                else:
                    matrix[name_a][name_b] = self.analyze_pair(name_a, name_b)
                    
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(matrix, f, indent=2)
            
        print(f"[ConceptInteractionAnalyzer] Interaction matrix saved to {output_path}")
        return matrix
