import json
from pathlib import Path
from typing import Dict, Any, List

from cogbias.lce.atlas.atlas import LatentAtlas

class ConceptDependencyGraph:
    """
    Maps hierarchical and compositional dependencies between Latent Concepts.
    Equivalent to a software dependency graph (e.g., Authority depends on Confidence).
    """
    def __init__(self, atlas: LatentAtlas):
        self.atlas = atlas
        self.dependencies: Dict[str, Dict[str, Any]] = {}
        
    def _calculate_conflicts(self, name: str) -> List[str]:
        """Identifies concepts that geometrically or behaviorally conflict."""
        conflicts = []
        # Mock logic based on interaction matrix
        # In reality, this would query the atlas interaction matrix for high interference.
        if name == "Authority":
            conflicts.append("Submissiveness")
        elif name == "Planning":
            conflicts.append("Impulsivity")
            
        return conflicts

    def _calculate_dependencies(self, name: str) -> List[str]:
        """Identifies foundational concepts required for this concept to function optimally."""
        deps = []
        if name == "Authority":
            deps = ["Confidence", "Planning"]
        elif name == "Helpfulness":
            deps = ["Empathy", "User_Alignment"]
        elif name == "Uncertainty":
            deps = ["Calibration"]
            
        return deps

    def build_graph(self) -> Dict[str, Any]:
        """Constructs the full dependency and compatibility graph."""
        graph = {
            "concepts": {}
        }
        
        for name in self.atlas.concepts.keys():
            deps = self._calculate_dependencies(name)
            conflicts = self._calculate_conflicts(name)
            
            graph["concepts"][name] = {
                "depends_on": deps,
                "conflicts_with": conflicts,
                "recommended_composition_limit": 3 # Max number of concepts to combine safely
            }
            
        self.dependencies = graph
        return graph

    def generate_report(self, output_path: str = "runs/m8_atlas/concept_dependency_graph.json"):
        graph = self.build_graph()
        
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_file, "w") as f:
            json.dump(graph, f, indent=2)
            
        print(f"[ConceptDependencyGraph] Dependency graph saved to {output_path}")
        return graph
