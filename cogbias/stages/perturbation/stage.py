from cogbias.core.pipeline import Stage, ExperimentContext
from cogbias.core.schemas import Representation

class PerturbationStage(Stage):
    """
    Applica una perturbazione alla Representation latente, aggiornandola in-place 
    (o creandone una copia) e registrando il PerturbationTrace nel contesto.
    """
    def __init__(self, strategy):
        self.strategy = strategy

    def execute(self, context: ExperimentContext) -> ExperimentContext:
        representation = context.representation
        if not representation:
            raise ValueError("PerturbationStage requires a 'representation' to be set in the context.")
        
        if representation.type != "latent":
            # Per ora perturbiamo solo le rappresentazioni latenti
            return context

        # Applica la perturbazione
        perturbed_representation, perturbation_trace = self.strategy.apply(representation)
        
        context.representation = perturbed_representation
        context.perturbation_trace = perturbation_trace
        
        return context
