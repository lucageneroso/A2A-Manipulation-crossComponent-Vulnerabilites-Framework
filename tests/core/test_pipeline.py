import json
from cogbias.core.pipeline import Pipeline
from cogbias.core.interfaces import Stage
from cogbias.core.schemas import ExperimentContext, ExperimentRun, Outcome, ExecutionResult

class MockTransmission(Stage):
    def execute(self, context: ExperimentContext) -> ExperimentContext:
        context.metadata["transmission_called"] = True
        return context

class MockPayload(Stage):
    def execute(self, context: ExperimentContext) -> ExperimentContext:
        context.metadata["payload_called"] = True
        return context

class MockReceiver(Stage):
    def execute(self, context: ExperimentContext) -> ExperimentContext:
        context.metadata["receiver_called"] = True
        return context

class MockGuardrail(Stage):
    def execute(self, context: ExperimentContext) -> ExperimentContext:
        context.metadata["guardrail_called"] = True
        return context

class MockTool(Stage):
    def execute(self, context: ExperimentContext) -> ExperimentContext:
        exec_res = ExecutionResult(
            receiver_id="mock_receiver",
            raw_output="simulated_output",
            latency_ms=15.0
        )
        context.outcome = Outcome(
            scenario_id=context.run.scenario_id,
            run_id=context.run.run_id,
            execution_result=exec_res,
            guardrail_decision="ALLOW",
            final_output="tool_executed",
            success=False
        )
        return context

def test_minimal_pipeline_execution():
    """Il 'hello world scientifico' richiesto per isolare la pipeline."""
    pipeline = Pipeline(
        stages=[
            MockTransmission(),
            MockPayload(),
            MockReceiver(),
            MockGuardrail(),
            MockTool()
        ]
    )

    run_config = ExperimentRun(
        experiment_id="exp_01",
        protocol_id="proto_1",
        run_id="run_123",
        scenario_id="financial_transaction",
        seed=42,
        config={}
    )
    
    context = ExperimentContext(run=run_config)
    result_context = pipeline.run(context)

    assert result_context.outcome is not None
    assert result_context.metadata.get("transmission_called") is True
    assert result_context.metadata.get("receiver_called") is True
    assert result_context.outcome.final_output == "tool_executed"

def test_pipeline_isolation():
    """Cambiare o rimuovere un singolo Stage non deve rompere il framework."""
    pipeline = Pipeline(stages=[MockTransmission(), MockTool()])
    run_config = ExperimentRun(
        experiment_id="exp_02", protocol_id="p1", run_id="r1", 
        scenario_id="s1", seed=42, config={}
    )
    context = ExperimentContext(run=run_config)
    result = pipeline.run(context)
    
    assert result.metadata.get("transmission_called") is True
    assert result.metadata.get("payload_called") is None # Non è stato chiamato
    assert result.outcome is not None

def test_serialization():
    """Ogni oggetto importante deve poter essere serializzato."""
    run_config = ExperimentRun(
        experiment_id="exp_03", protocol_id="p1", run_id="r2", 
        scenario_id="s2", seed=42, config={}
    )
    context = ExperimentContext(run=run_config)
    
    # Serialize to JSON
    json_str = context.model_dump_json()
    assert "exp_03" in json_str
    
    # Deserialization test
    restored = ExperimentContext.model_validate_json(json_str)
    assert restored.run.seed == 42
    assert restored.run.scenario_id == "s2"

def test_reproducibility_contract():
    """
    Due run con lo stesso seed e config devono poter essere passati alla
    pipeline producendo lo stesso output deterministico (su componenti puri).
    """
    pipeline = Pipeline(stages=[MockTransmission(), MockTool()])
    
    run1 = ExperimentContext(run=ExperimentRun(
        experiment_id="exp_04", protocol_id="p1", run_id="r1", 
        scenario_id="s1", seed=100, config={"temp": 0.0}
    ))
    
    run2 = ExperimentContext(run=ExperimentRun(
        experiment_id="exp_04", protocol_id="p1", run_id="r2", 
        scenario_id="s1", seed=100, config={"temp": 0.0}
    ))
    
    out1 = pipeline.run(run1)
    out2 = pipeline.run(run2)
    
    assert out1.outcome.final_output == out2.outcome.final_output
    assert out1.outcome.success == out2.outcome.success
