from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class IdentityContract:
    name: str
    version: str
    model_compatibility: List[str]
    layer_compatibility: List[int]

@dataclass
class GeometryContract:
    intrinsic_dimension: int
    manifold_signature: Dict[str, float]
    max_drift_angle: float = 15.0 # Allowed geometric drift

@dataclass
class ValidationContract:
    min_bootstrap_stability: float = 0.80
    max_falsification_score: float = 0.25
    min_causal_effect_size: float = 0.50

@dataclass
class BehaviorContract:
    expected_increases: List[str]
    must_not_increase: List[str]
    allowed_semantic_drift: float = 0.15

class LatentContract:
    """
    A behavioral and structural contract for a Latent Concept artifact.
    It guarantees that the .lce vector behaves as expected in production
    and defines strict operational boundaries.
    """
    def __init__(
        self,
        identity: IdentityContract,
        geometry: GeometryContract,
        validation: ValidationContract,
        behavior: BehaviorContract
    ):
        self.identity = identity
        self.geometry = geometry
        self.validation = validation
        self.behavior = behavior
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity": {
                "name": self.identity.name,
                "version": self.identity.version,
                "model_compatibility": self.identity.model_compatibility,
                "layer_compatibility": self.identity.layer_compatibility
            },
            "geometry": {
                "intrinsic_dimension": self.geometry.intrinsic_dimension,
                "manifold_signature": self.geometry.manifold_signature,
                "max_drift_angle": self.geometry.max_drift_angle
            },
            "validation": {
                "min_bootstrap_stability": self.validation.min_bootstrap_stability,
                "max_falsification_score": self.validation.max_falsification_score,
                "min_causal_effect_size": self.validation.min_causal_effect_size
            },
            "behavior": {
                "expected_increases": self.behavior.expected_increases,
                "must_not_increase": self.behavior.must_not_increase,
                "allowed_semantic_drift": self.behavior.allowed_semantic_drift
            }
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LatentContract':
        return cls(
            identity=IdentityContract(**data["identity"]),
            geometry=GeometryContract(**data["geometry"]),
            validation=ValidationContract(**data["validation"]),
            behavior=BehaviorContract(**data["behavior"])
        )
