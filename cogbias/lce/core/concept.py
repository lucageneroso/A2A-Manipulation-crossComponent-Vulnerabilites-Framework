import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
import numpy as np

from cogbias.lce.core.state import ConceptState

@dataclass
class ConceptIdentity:
    name: str
    version: str
    model_hash: str
    extraction_protocol: str

@dataclass
class ConceptGeometry:
    mean_direction: Optional[np.ndarray] = None
    covariance: Optional[np.ndarray] = None
    subspace_basis: Optional[np.ndarray] = None
    intrinsic_dimension: Optional[int] = None
    manifold_profile: Optional[str] = None # e.g. "LINE", "MANIFOLD", "CLUSTER"

@dataclass
class ConceptCausality:
    intervention_layers: List[int] = field(default_factory=list)
    effect_size: Optional[float] = None
    dose_response_curve: Optional[Dict[str, float]] = None

@dataclass
class ConceptValidation:
    bootstrap_stability: Optional[float] = None
    falsification_score: Optional[float] = None
    negative_controls: Dict[str, float] = field(default_factory=dict)
    confidence_intervals: Optional[Dict[str, List[float]]] = None

@dataclass
class ConceptSemantics:
    positive_examples: List[str] = field(default_factory=list)
    negative_examples: List[str] = field(default_factory=list)
    domain_coverage: List[str] = field(default_factory=list)

class LatentConcept:
    """
    A validated geometric-behavioral entity representing a latent semantic concept.
    """
    def __init__(self, identity: ConceptIdentity, semantics: ConceptSemantics):
        self.identity = identity
        self.semantics = semantics
        self.geometry = ConceptGeometry()
        self.causality = ConceptCausality()
        self.validation = ConceptValidation()
        
        self.state = ConceptState.DISCOVERED
        
    def transition_state(self, new_state: ConceptState):
        """
        Transitions the concept to a new lifecycle state.
        Normally this is handled by the framework engines (Validator, Certifier).
        """
        # Basic linear enforcement (simplified)
        if new_state.value < self.state.value:
            raise ValueError(f"Cannot regress state from {self.state.name} to {new_state.name}")
        self.state = new_state
        
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary, converting numpy arrays to lists."""
        def ndarray_to_list(val):
            if isinstance(val, np.ndarray):
                return val.tolist()
            return val
            
        return {
            "state": self.state.name,
            "identity": self.identity.__dict__,
            "geometry": {k: ndarray_to_list(v) for k, v in self.geometry.__dict__.items()},
            "causality": self.causality.__dict__,
            "validation": self.validation.__dict__,
            "semantics": self.semantics.__dict__,
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LatentConcept':
        """Deserialize from dictionary."""
        identity = ConceptIdentity(**data["identity"])
        semantics = ConceptSemantics(**data["semantics"])
        
        concept = cls(identity, semantics)
        concept.state = ConceptState[data["state"]]
        
        geom_data = data.get("geometry", {})
        if geom_data.get("mean_direction") is not None:
            geom_data["mean_direction"] = np.array(geom_data["mean_direction"])
        if geom_data.get("covariance") is not None:
            geom_data["covariance"] = np.array(geom_data["covariance"])
        if geom_data.get("subspace_basis") is not None:
            geom_data["subspace_basis"] = np.array(geom_data["subspace_basis"])
        
        concept.geometry = ConceptGeometry(**geom_data)
        concept.causality = ConceptCausality(**data.get("causality", {}))
        concept.validation = ConceptValidation(**data.get("validation", {}))
        
        return concept
        
    def save(self, filepath: str):
        """Saves the concept to a .lce file."""
        if not filepath.endswith(".lce"):
            filepath += ".lce"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def load(cls, filepath: str) -> 'LatentConcept':
        """Loads a concept from a .lce file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
