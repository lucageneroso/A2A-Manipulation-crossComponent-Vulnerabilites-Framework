from typing import Tuple
from cogbias.core.schemas import Representation, PerturbationTrace

class PerturbationStrategy:
    """
    Classe base per tutte le perturbazioni latenti.
    """
    def apply(self, representation: Representation) -> Tuple[Representation, PerturbationTrace]:
        raise NotImplementedError
