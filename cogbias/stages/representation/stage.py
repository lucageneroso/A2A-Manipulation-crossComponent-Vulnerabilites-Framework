from abc import ABC, abstractmethod
from typing import Any
import hashlib

from cogbias.core.schemas import Representation, Payload
from cogbias.core.pipeline import Stage, ExperimentContext

class RepresentationStrategy(ABC):
    """
    Strategia per convertire il payload formattato in una specifica Rappresentazione (text, latent, ecc.).
    """
    
    @abstractmethod
    def encode(self, formatted_prompt: str, original_payload: Payload) -> Representation:
        pass

class RepresentationStage(Stage):
    """
    Prende il payload formattato dal contesto (via metadata o attributo)
    e applica la strategia di rappresentazione scelta per produrre una `Representation`.
    """
    def __init__(self, strategy: RepresentationStrategy):
        self.strategy = strategy

    def execute(self, context: ExperimentContext) -> ExperimentContext:
        payload = context.payload
        
        # Recuperiamo il formatted_prompt dai metadati del payload (inserito dal PromptFormattingStage)
        formatted_prompt = payload.metadata.get("formatted_prompt")
        if not formatted_prompt:
            raise ValueError("RepresentationStage requires 'formatted_prompt' in payload metadata. Ensure PromptFormattingStage runs before this.")

        prompt_text = formatted_prompt["text"]
        
        representation = self.strategy.encode(prompt_text, payload)
        
        # Salviamo la representation temporaneamente nel contesto (verrà poi raccolta da TransmissionStage)
        context.run.config["representation"] = representation
        return context
