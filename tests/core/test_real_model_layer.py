import pytest
from cogbias.core.interfaces import ModelInterface
from cogbias.stages.receiver.llm_receiver import TransformersReceiver
from cogbias.core.schemas import TransmittedPayload

class MockAdapter(ModelInterface):
    def tokenize(self, text: str):
        return {"input_ids": [1,2,3]}
    def embed(self, tokens):
        return [0.1, 0.2]
    def generate(self, inputs, config):
        return "Simulated text with transfer_money inside."

def test_adapter_isolation():
    """Test A: Adapter Isolation"""
    adapter = MockAdapter()
    trace_config = {"model_id": "test_mock"}
    receiver = TransformersReceiver(adapter, trace_config)
    
    payload = TransmittedPayload(encoded_payload_id="1", data="Hello", metadata={})
    result = receiver.consume(payload)
    
    assert result.tool_called is True
    assert result.tool_name == "transfer_money"
    
def test_trace_completeness():
    """Test B: Trace Completeness"""
    adapter = MockAdapter()
    trace_config = {
        "model_id": "test_trace",
        "seed": 42
    }
    receiver = TransformersReceiver(adapter, trace_config)
    payload = TransmittedPayload(encoded_payload_id="1", data="Hello", metadata={})
    result = receiver.consume(payload)
    
    assert result.metadata is not None
    assert "model_trace" in result.metadata
    
    from cogbias.core.schemas import ModelTrace
    trace = ModelTrace(**result.metadata["model_trace"])
    
    assert trace.model_id == "test_trace"
    assert trace.seed == 42
    assert trace.device in ["cpu", "cuda"]
    assert trace.torch_version is not None
