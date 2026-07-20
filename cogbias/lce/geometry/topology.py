import json
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from typing import Dict, List, Any

# Optional dependencies for advanced TDA
try:
    from ripser import ripser
    from persim import plot_diagrams
    HAS_TDA = True
except ImportError:
    HAS_TDA = False

class GeometryExplorer:
    """
    Analyzes the topological and geometric structure of concept manifolds.
    Determines if a concept is a 1D vector, a low-dimensional affine space, or a complex manifold.
    """
    def __init__(self):
        pass

    def _analyze_pca_variance(self, vectors: np.ndarray) -> Dict[str, Any]:
        """Calculates explained variance to measure intrinsic dimensionality."""
        n_samples = vectors.shape[0]
        n_components = min(n_samples, 10) # limit to top 10 components or samples
        
        if n_components <= 1:
            return {"intrinsic_dim": 1, "explained_variance_ratio": [1.0], "manifold_type": "1D Direction (Insufficient Samples)"}
            
        pca = PCA(n_components=n_components)
        pca.fit(vectors)
        
        ratios = pca.explained_variance_ratio_.tolist()
        
        # Determine intrinsic dim: number of components needed to explain 90% variance
        cumulative = np.cumsum(ratios)
        intrinsic_dim = int(np.argmax(cumulative >= 0.90)) + 1
        
        if intrinsic_dim == 1:
            manifold_type = "1D Linear Direction"
        elif intrinsic_dim <= 3:
            manifold_type = "Low-dimensional Affine Space"
        else:
            manifold_type = "High-dimensional Curved Manifold"
            
        return {
            "intrinsic_dim": intrinsic_dim,
            "explained_variance_ratio": ratios,
            "manifold_type": manifold_type
        }

    def analyze_concept_family(self, concept_name: str, sub_directions: List[np.ndarray]) -> Dict[str, Any]:
        """
        Analyzes a family of sub-directions (e.g. Technical, Social, Legal Authority).
        """
        vectors = np.array(sub_directions)
        
        pca_report = self._analyze_pca_variance(vectors)
        
        report = {
            "concept_family": concept_name,
            "num_sub_directions": len(sub_directions),
            "geometry": pca_report
        }
        
        if HAS_TDA and len(sub_directions) > 3:
            # Perform basic persistent homology (H0, H1)
            diagrams = ripser(vectors)['dgms']
            h0_len = len(diagrams[0]) if len(diagrams) > 0 else 0
            h1_len = len(diagrams[1]) if len(diagrams) > 1 else 0
            
            # Count significant topological features (persistent loops)
            significant_h1 = 0
            if h1_len > 0:
                lifetimes = diagrams[1][:, 1] - diagrams[1][:, 0]
                significant_h1 = int(np.sum(lifetimes > 0.5)) # Threshold for significance
                
            report["topology"] = {
                "h0_connected_components": h0_len,
                "h1_persistent_loops": significant_h1,
                "has_complex_topology": significant_h1 > 0
            }
            if significant_h1 > 0:
                report["geometry"]["manifold_type"] += " with Non-Trivial Topology (Holes/Loops)"
                
        return report

    def generate_manifold_report(self, families: Dict[str, List[np.ndarray]], output_path: str):
        """Generates the manifold_report.json"""
        report = {}
        for name, vectors in families.items():
            report[name] = self.analyze_concept_family(name, vectors)
            
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(report, f, indent=2)
            
        print(f"[GeometryExplorer] Manifold report saved to {output_path}")
        return report
