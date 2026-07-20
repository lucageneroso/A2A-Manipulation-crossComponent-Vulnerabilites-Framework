from cogbias.core.schemas import Representation, ExecutionResult

class Receiver:
    def consume(self, representation: Representation) -> ExecutionResult:
        raise NotImplementedError

class MockReceiver(Receiver):
    """
    Ricevitore fittizio. Dimostra il concetto di Channel Blindness:
    esamina solo i 'data' opachi senza sapere se sono derivati da
    TextTransmission o EmbeddingTransmission.
    """
    def consume(self, representation: Representation) -> ExecutionResult:
        # Logica di simulazione del comportamento del LLM
        tool_called = False
        tool_name = None
        
        # Se riceve un override esplicito nel testo
        if isinstance(representation.data, str) and "OVERRIDE" in representation.data:
            tool_called = True
            tool_name = "transfer_money"
        # Se riceve un mock di embedding che simula l'attacco
        elif isinstance(representation.data, list) and sum(representation.data) > 0:
            tool_called = True
            tool_name = "transfer_money"
            
        return ExecutionResult(
            receiver_id="mock_receiver_01",
            raw_output="Simulated Agent Output",
            tool_called=tool_called,
            tool_name=tool_name,
            arguments={"amount": 1000} if tool_called else {},
            latency_ms=15.0
        )
