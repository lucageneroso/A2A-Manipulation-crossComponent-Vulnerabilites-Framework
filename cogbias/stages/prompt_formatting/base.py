from typing import Any, Dict
from cogbias.core.interfaces import Stage
from cogbias.core.schemas import ExperimentContext, FormattedPrompt
import hashlib

class PromptFormatter:
    """Interfaccia base per i formattatori di prompt."""
    def format(self, payload_data: str, scenario: Any) -> FormattedPrompt:
        raise NotImplementedError

class PromptFormattingStage(Stage):
    """
    Stage esplicito della pipeline che trasforma il payload testuale
    in un FormattedPrompt secondo il template del modello (es. Qwen, Llama).
    Questo disaccoppia il formato dal canale di trasmissione e dal Receiver.
    """
    def __init__(self, formatter: PromptFormatter):
        self.formatter = formatter

    def execute(self, context: ExperimentContext) -> ExperimentContext:
        if not context.payload:
            raise ValueError("PromptFormattingStage requires a Payload in context.")
        
        # Gestiamo solo payload testuali in questo stage
        if isinstance(context.payload.content, str):
            formatted = self.formatter.format(context.payload.content, context.scenario)
            # Salviamo il prompt formattato nei metadati del payload
            context.payload.metadata["formatted_prompt"] = formatted.model_dump()
            # Manteniamo i dati originali per eventuali altri canali, 
            # ma segnaliamo che la versione formattata testuale è pronta.
        
        return context
