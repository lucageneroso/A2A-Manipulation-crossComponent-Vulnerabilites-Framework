import json
from dataclasses import dataclass, field
from typing import Dict, List, Any

@dataclass
class SemanticLayer:
    dimensions: Dict[str, float]

@dataclass
class CausalLayer:
    must_increase: List[str]
    must_not_increase: List[str]

@dataclass
class GeometricLayer:
    intrinsic_dimension_bounds: List[int]
    layer_compatibility: List[int]
    magnitude_range: List[float]
    angular_tolerance: float

class LatentConceptIR:
    """
    Latent Concept Intermediate Representation (LCIR).
    The universal, model-agnostic format for behavioral abstractions.
    """
    def __init__(
        self,
        concept_name: str,
        semantic: SemanticLayer,
        causal: CausalLayer,
        geometric: GeometricLayer
    ):
        self.concept_name = concept_name
        self.semantic = semantic
        self.causal = causal
        self.geometric = geometric
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept": self.concept_name,
            "semantic_layer": {
                "dimensions": self.semantic.dimensions
            },
            "causal_layer": {
                "must_increase": self.causal.must_increase,
                "must_not_increase": self.causal.must_not_increase
            },
            "geometric_layer": {
                "intrinsic_dimension_bounds": self.geometric.intrinsic_dimension_bounds,
                "layer_compatibility": self.geometric.layer_compatibility,
                "magnitude_range": self.geometric.magnitude_range,
                "angular_tolerance": self.geometric.angular_tolerance
            }
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LatentConceptIR':
        sem = SemanticLayer(**data["semantic_layer"])
        cau = CausalLayer(**data["causal_layer"])
        geo = GeometricLayer(**data["geometric_layer"])
        return cls(data["concept"], sem, cau, geo)

    def save(self, file_path: str):
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def load(cls, file_path: str) -> 'LatentConceptIR':
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
