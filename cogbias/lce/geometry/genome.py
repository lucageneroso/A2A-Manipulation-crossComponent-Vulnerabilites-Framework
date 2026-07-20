import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from sklearn.decomposition import FastICA

from cogbias.lce.atlas.atlas import LatentAtlas

class ConceptGenome:
    """
    Decomposes certified latent concepts into foundational geometric primitives
    using Independent Component Analysis (ICA). This acts as the "Conceptual DNA".
    """
    def __init__(self, atlas: LatentAtlas):
        self.atlas = atlas
        
    def decompose(self, n_primitives: int = 5) -> Dict[str, Any]:
        """
        Extracts foundational primitives from the aggregated concept subspace.
        """
        # 1. Aggregate all certified concept vectors
        concept_names = list(self.atlas.concepts.keys())
        if len(concept_names) < 2:
            raise ValueError("Genome decomposition requires at least 2 concepts in the Atlas.")
            
        vectors = []
        for name in concept_names:
            v = self.atlas.concepts[name].geometry.mean_direction
            # Normalize to ensure fair contribution
            vectors.append(v / (np.linalg.norm(v) + 1e-9))
            
        matrix = np.array(vectors) # Shape: (num_concepts, hidden_dim)
        
        # 2. Perform ICA to find independent primitives
        # We find components across the feature dimension
        n_components = min(n_primitives, matrix.shape[0])
        ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
        
        # We fit ICA on the concept vectors.
        # This isolates statistically independent source vectors (primitives).
        primitives_matrix = ica.fit_transform(matrix.T).T # Shape: (n_components, hidden_dim)
        
        # 3. Calculate loadings (how much each concept relies on each primitive)
        # mixing_ matrix shape: (hidden_dim, n_components), but we fit on transposed.
        # Let's project concepts onto primitives directly.
        
        genome = {
            "primitives": [],
            "concepts": []
        }
        
        # Build primitive definitions
        for i in range(n_components):
            p_vec = primitives_matrix[i]
            p_norm = p_vec / (np.linalg.norm(p_vec) + 1e-9)
            
            # Find which concepts strongly activate this primitive
            related = []
            for j, name in enumerate(concept_names):
                c_vec = matrix[j]
                overlap = float(np.dot(c_vec, p_norm))
                if abs(overlap) > 0.3: # Threshold for significant relation
                    related.append({"concept": name, "loading": overlap})
                    
            genome["primitives"].append({
                "id": f"PRIMITIVE_{i+1}",
                "vector_norm": float(np.linalg.norm(p_vec)), # Actual vector omitted from JSON for size
                "related_concepts": sorted(related, key=lambda x: abs(x["loading"]), reverse=True)
            })
            
        # Build concept decomposition maps
        for i, name in enumerate(concept_names):
            c_vec = matrix[i]
            decomposition = {}
            for j in range(n_components):
                p_vec = primitives_matrix[j]
                p_norm = p_vec / (np.linalg.norm(p_vec) + 1e-9)
                loading = float(np.dot(c_vec, p_norm))
                decomposition[f"PRIMITIVE_{j+1}"] = loading
                
            genome["concepts"].append({
                "name": name,
                "decomposition": decomposition
            })
            
        return genome

    def generate_genome_report(self, output_dir: str = "runs/m8_genome"):
        genome = self.decompose()
        
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        file_path = out_path / "concept_genome.json"
        
        with open(file_path, "w") as f:
            json.dump(genome, f, indent=2)
            
        print(f"[ConceptGenome] Genome report saved to {file_path}")
        return genome
