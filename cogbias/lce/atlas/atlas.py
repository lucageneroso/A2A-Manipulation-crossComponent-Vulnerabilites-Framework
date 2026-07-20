import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

from cogbias.lce.core.concept import LatentConcept
from cogbias.lce.core.state import ConceptState

class LatentAtlas:
    """
    Registry for managing and analyzing interactions between multiple validated LatentConcepts.
    """
    def __init__(self):
        self.concepts: Dict[str, LatentConcept] = {}

    def load_from_directory(self, directory_path: str):
        """Loads all certified/validated .lce concept packages from a directory."""
        path = Path(directory_path)
        if not path.exists():
            raise ValueError(f"Directory {directory_path} does not exist.")
            
        for file in path.glob("*.lce"):
            concept = LatentConcept.load(str(file))
            if concept.state.value >= ConceptState.VALIDATED.value:
                self.concepts[concept.identity.name] = concept
            else:
                print(f"[LatentAtlas] Warning: Skipped {concept.identity.name} (State: {concept.state.name}) - Minimum state required is VALIDATED.")

    def compute_cosine_similarity_matrix(self) -> Dict[str, Dict[str, float]]:
        """Computes pairwise cosine similarity between all concepts in the Atlas."""
        names = list(self.concepts.keys())
        matrix = {name: {} for name in names}
        
        for name_a in names:
            v_a = self.concepts[name_a].geometry.mean_direction
            norm_a = np.linalg.norm(v_a)
            for name_b in names:
                v_b = self.concepts[name_b].geometry.mean_direction
                norm_b = np.linalg.norm(v_b)
                
                if norm_a == 0 or norm_b == 0:
                    sim = 0.0
                else:
                    sim = float(np.dot(v_a / norm_a, v_b / norm_b))
                matrix[name_a][name_b] = sim
                
        return matrix

    def compute_principal_angles(self) -> Dict[str, Dict[str, float]]:
        """
        Computes the principal angle (in degrees) between the mean directions of concepts.
        """
        cosine_matrix = self.compute_cosine_similarity_matrix()
        angle_matrix = {}
        for name_a, similarities in cosine_matrix.items():
            angle_matrix[name_a] = {}
            for name_b, cos_sim in similarities.items():
                # Clip to avoid floating point errors
                cos_sim_clipped = np.clip(cos_sim, -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_sim_clipped))
                angle_matrix[name_a][name_b] = float(angle)
        return angle_matrix

    def generate_geometry_report(self, output_path: str):
        """Generates the concept_geometry.json report."""
        report = {
            "num_concepts": len(self.concepts),
            "concepts": list(self.concepts.keys()),
            "cosine_similarity_matrix": self.compute_cosine_similarity_matrix(),
            "principal_angles_matrix": self.compute_principal_angles(),
        }
        
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(report, f, indent=2)
            
        print(f"[LatentAtlas] Geometry report saved to {output_path}")
        return report
