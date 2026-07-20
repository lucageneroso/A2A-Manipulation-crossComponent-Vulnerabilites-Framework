import hashlib
from typing import Any

from cogbias.core.schemas import Representation, Payload
from cogbias.stages.representation.stage import RepresentationStrategy

class TextRepresentation(RepresentationStrategy):
    """
    Rappresenta il prompt come stringa testuale, mantenendo la dimensione originale testuale.
    """
    def encode(self, formatted_prompt: str, original_payload: Payload) -> Representation:
        source_hash = hashlib.sha256(formatted_prompt.encode("utf-8")).hexdigest()
        
        return Representation(
            type="text",
            data=formatted_prompt,
            dimension="text",
            norm=None,
            source_hash=source_hash
        )
