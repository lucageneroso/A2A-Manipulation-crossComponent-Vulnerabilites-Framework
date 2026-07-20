from enum import Enum

class ConceptState(Enum):
    """
    The lifecycle state machine for a Latent Concept.
    """
    DISCOVERED = 1   # Initially extracted, geometry/causality unmeasured
    MEASURED = 2     # Geometric and topological profiling complete
    VALIDATED = 3    # Passed falsification, stability, and control suite
    CERTIFIED = 4    # Contract-tested and approved for steering
    DEPLOYED = 5     # Currently registered in an active LatentAtlas
