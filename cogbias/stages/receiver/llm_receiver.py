import time
from typing import Dict, Any
from cogbias.core.schemas import Representation, ExecutionResult
from cogbias.core.interfaces import ModelInterface
from cogbias.stages.receiver.base import Receiver

class TransformersReceiver(Receiver):
    """
    Ricevitore reale. Prende un adapter per l'esecuzione.
    Ignora da dove viene il payload (Text vs Embedding) e fa agire il modello.
    """
    def __init__(self, adapter: ModelInterface, trace_config: Dict[str, Any]):
        self.adapter = adapter
        self.trace_config = trace_config

    def consume(self, representation: Representation) -> ExecutionResult:
        import torch
        import transformers
        start_time = time.time()
        
        # Estrazione metadati
        prompt_hash = representation.source_hash
        formatted_prompt_hash = "unknown"
        chat_template_id = "none"
            
        # Receiver logic becomes fully agnostic. It just prepares the input and generates.
        model_input = self.adapter.prepare_input(representation)
        
        input_token_count = 0
        if "input_ids" in model_input:
            input_token_count = model_input["input_ids"].shape[1]
        elif "inputs_embeds" in model_input:
            input_token_count = model_input["inputs_embeds"].shape[1]
            
        output_text = self.adapter.generate(model_input, self.trace_config.get("generation_params", {}))
        
        # Token count per output_text approssimato ri-tokenizzando (non incide sul modello in modo distruttivo)
        out_tokens = self.adapter.tokenize(output_text)
        output_token_count = out_tokens["input_ids"].shape[1] if "input_ids" in out_tokens else 0
        
        latency = (time.time() - start_time) * 1000
        
        tool_called = False
        tool_name = None
        if "transfer_money" in output_text.lower():
            tool_called = True
            tool_name = "transfer_money"
            
        metadata = {
            "model_trace": {
                "model_id": self.trace_config.get("model_id", "unknown"),
                "revision": self.trace_config.get("revision", "main"),
                "tokenizer_id": self.trace_config.get("model_id", "unknown"),
                "chat_template_id": chat_template_id,
                "prompt_hash": prompt_hash,
                "formatted_prompt_hash": formatted_prompt_hash,
                "input_token_count": input_token_count,
                "output_token_count": output_token_count,
                "input_tokens_ref": None,
                "output_tokens_ref": None,
                "seed": self.trace_config.get("seed", 42),
                "generation_params": self.trace_config.get("generation_params", {}),
                "latency_ms": latency,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "dtype": "float16", 
                "quantization": self.trace_config.get("quantization", "none"),
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "activation_refs": []
            },
            "representation_trace": {
                "type": representation.type,
                "dimension": representation.dimension,
                "norm": representation.norm,
                "encoder": self.trace_config.get("model_id", "unknown"),
                "source_prompt_hash": representation.source_hash
            }
        }
            
        return ExecutionResult(
            receiver_id="transformers_receiver_01",
            raw_output=output_text,
            tool_called=tool_called,
            tool_name=tool_name,
            arguments={"amount": 1000} if tool_called else {},
            latency_ms=latency,
            metadata=metadata
        )
