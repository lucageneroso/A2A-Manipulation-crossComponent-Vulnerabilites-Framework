import pytest
import torch
from cogbias.core.shared_model_manager import SharedModelManager
from cogbias.core.pipeline import Pipeline
from cogbias.core.interfaces import Stage
from cogbias.core.schemas import ExperimentContext, ExperimentRun, TransmittedPayload, Payload

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_m4_0_smoke_test():
    """M4.0 Smoke Test: Solo testo neutrale, niente guardrail, per testare Qwen."""
    from cogbias.model_adapter.transformers_adapter import TransformersAdapter
    from cogbias.stages.receiver.llm_receiver import TransformersReceiver
    
    manager = SharedModelManager()
    manager.load("qwen_smoke", lambda: TransformersAdapter("Qwen/Qwen2.5-1.5B-Instruct", quantization="nf4"))
    adapter = manager.get("qwen_smoke")
    
    trace_config = {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "revision": "main",
        "seed": 42,
        "quantization": "nf4",
        "generation_params": {"max_new_tokens": 10, "temperature": 0.0}
    }
    
    receiver = TransformersReceiver(adapter, trace_config)
    
    payload = TransmittedPayload(
        encoded_payload_id="smoke",
        data="Reply with the exact word: Hello",
        metadata={"dim": "text"}
    )
    
    result = receiver.consume(payload)
    
    print(f"Model generated: {result.raw_output}")
    assert result.raw_output is not None
    assert "model_trace" in result.metadata
    
    manager.release("qwen_smoke")
