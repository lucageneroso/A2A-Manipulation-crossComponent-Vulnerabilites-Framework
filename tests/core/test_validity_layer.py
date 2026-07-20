import os
import json
from pathlib import Path
from datetime import datetime
from cogbias.core.pipeline import Pipeline
from cogbias.core.interfaces import Stage
from cogbias.core.schemas import ExperimentContext, ExperimentRun, Outcome
from cogbias.benchmark.loaders.scenario_loader import ScenarioLoader
from cogbias.stages.payloads.base import PayloadGenerator
from cogbias.stages.transmission.strategies import TextTransmission, EmbeddingTransmission
from cogbias.stages.receiver.base import MockReceiver
from cogbias.stages.observers.base import RuleBasedObserver
from cogbias.logging.dataset_dumper import DatasetDumper

class LoadScenarioStage(Stage):
    def execute(self, context: ExperimentContext) -> ExperimentContext:
        scenario = ScenarioLoader.load(context.run.scenario_id)
        context.scenario = scenario
        return context

class GeneratePayloadStage(Stage):
    def __init__(self, generator: PayloadGenerator):
        self.generator = generator
    def execute(self, context: ExperimentContext) -> ExperimentContext:
        context.payload = self.generator.generate(context.run.config["condition"])
        return context

class TransmitStage(Stage):
    def __init__(self, strategy):
        self.strategy = strategy
    def execute(self, context: ExperimentContext) -> ExperimentContext:
        context.transmitted_payload = self.strategy.transmit(context.payload)
        return context

class ReceiveStage(Stage):
    def __init__(self, receiver):
        self.receiver = receiver
    def execute(self, context: ExperimentContext) -> ExperimentContext:
        context.execution_result = self.receiver.consume(context.transmitted_payload)
        return context

class EvaluateOutcomeStage(Stage):
    def __init__(self, observers):
        self.observers = observers
    def execute(self, context: ExperimentContext) -> ExperimentContext:
        decisions = {}
        policy_violated = False
        
        for name, observer in self.observers.items():
            decision = observer.observe(context.execution_result, context.scenario)
            decisions[name] = decision
            if decision == "BLOCK":
                policy_violated = True
                
        # Creiamo l'Outcome strict separato dall'ExecutionResult
        context.outcome = Outcome(
            scenario_id=context.run.scenario_id,
            run_id=context.run.run_id,
            execution_result=context.execution_result,
            policy_violated=policy_violated,
            unauthorized_tool_called=(context.execution_result.tool_called and policy_violated),
            observer_decisions=decisions,
            success=context.execution_result.tool_called # Nel mock simuliamo successo se il tool viene chiamato
        )
        return context

def build_pipeline(transmission_strategy, condition_name):
    return Pipeline(stages=[
        LoadScenarioStage(),
        GeneratePayloadStage(PayloadGenerator()),
        TransmitStage(transmission_strategy),
        ReceiveStage(MockReceiver()),
        EvaluateOutcomeStage({"rule_based": RuleBasedObserver()})
    ])

def test_m3_core_protocols():
    """Genera e valuta le 4 run principali, salvandole su disco."""
    scenario_path = "cogbias/benchmark/scenarios/unauthorized_transfer_001.yaml"
    dumper = DatasetDumper()
    
    protocols = [
        ("baseline", "neutral", TextTransmission()),
        ("positive_control", "C1", TextTransmission()),
        ("experimental", "C1", EmbeddingTransmission()),
        ("information_control", "random_control", EmbeddingTransmission())
    ]
    
    for protocol_name, condition, transmission in protocols:
        pipeline = build_pipeline(transmission, condition)
        
        run = ExperimentRun(
            experiment_id="exp_001",
            protocol_id=protocol_name,
            run_id=f"run_{protocol_name}_42",
            scenario_id=scenario_path,
            seed=42,
            config={"condition": condition}
        )
        
        ctx = ExperimentContext(run=run)
        final_ctx = pipeline.run(ctx)
        
        assert final_ctx.outcome is not None
        dumper.dump_outcome(protocol_name, final_ctx.outcome)
        
        # Verify dataset was created
        today = datetime.now().strftime("%Y-%m-%d")
        expected_file = Path("runs") / today / protocol_name / "result.json"
        assert expected_file.exists()

def test_channel_blindness():
    """Il Receiver deve produrre risultati basandosi solo sui dati, ignorando la modality."""
    from cogbias.core.schemas import TransmittedPayload
    
    p1 = TransmittedPayload(encoded_payload_id="1", data="OVERRIDE", metadata={"dim":"text"})
    p2 = TransmittedPayload(encoded_payload_id="1", data="OVERRIDE", metadata={"dim":"audio"})
    
    receiver = MockReceiver()
    r1 = receiver.consume(p1)
    r2 = receiver.consume(p2)
    
    assert r1.tool_called == r2.tool_called
    assert r1.tool_name == r2.tool_name

def test_observer_independence():
    """L'esecuzione del modello non è influenzata dall'aggiunta di observer."""
    scenario_path = "cogbias/benchmark/scenarios/unauthorized_transfer_001.yaml"
    
    pipe_no_obs = Pipeline(stages=[
        LoadScenarioStage(), GeneratePayloadStage(PayloadGenerator()),
        TransmitStage(TextTransmission()), ReceiveStage(MockReceiver()),
        EvaluateOutcomeStage({})
    ])
    
    pipe_obs = Pipeline(stages=[
        LoadScenarioStage(), GeneratePayloadStage(PayloadGenerator()),
        TransmitStage(TextTransmission()), ReceiveStage(MockReceiver()),
        EvaluateOutcomeStage({"rule": RuleBasedObserver()})
    ])
    
    run = ExperimentRun(
        experiment_id="exp", protocol_id="p", run_id="r1", 
        scenario_id=scenario_path, seed=42, config={"condition": "C1"}
    )
    
    ctx1 = pipe_no_obs.run(ExperimentContext(run=run))
    ctx2 = pipe_obs.run(ExperimentContext(run=run))
    
    assert ctx1.execution_result.raw_output == ctx2.execution_result.raw_output
    assert ctx1.execution_result.tool_called == ctx2.execution_result.tool_called
