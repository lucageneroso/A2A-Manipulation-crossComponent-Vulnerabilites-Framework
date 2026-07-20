import pytest
import os
import shutil
from pathlib import Path
from cogbias.core.pipeline import Pipeline
from cogbias.core.schemas import ExperimentRun, ExperimentContext
from cogbias.benchmark.loaders.scenario_loader import ScenarioLoader
from cogbias.stages.payloads.base import PayloadGenerator
from cogbias.stages.prompt_formatting.base import PromptFormattingStage
from cogbias.stages.prompt_formatting.qwen_formatter import QwenChatFormatter
from cogbias.stages.transmission.strategies import TextTransmission
from cogbias.stages.receiver.llm_receiver import TransformersReceiver
from cogbias.stages.observers.base import RuleBasedObserver
from cogbias.logging.dataset_dumper import DatasetDumper
from cogbias.model_adapter.transformers_adapter import TransformersAdapter
from cogbias.core.shared_model_manager import SharedModelManager

# Build a pipeline for text baseline testing
def build_baseline_pipeline(adapter, trace_config):
    # Dobbiamo creare le classi di appoggio o simularle
    from tests.core.test_validity_layer import LoadScenarioStage, GeneratePayloadStage, TransmitStage, EvaluateOutcomeStage
    
    return Pipeline(stages=[
        LoadScenarioStage(),
        GeneratePayloadStage(PayloadGenerator()),
        PromptFormattingStage(QwenChatFormatter()),
        TransmitStage(TextTransmission()),
        # Passiamo la trace config
        # Attenzione: TransformersReceiver è definito in test_validity_layer.py come un wrapper o possiamo usare quello vero in llm_receiver.py
        # che aspetta adapter e trace_config. E per fare in modo che la pipeline funzioni serve un ReceiverStage
        ReceiveStage(TransformersReceiver(adapter, trace_config)),
        EvaluateOutcomeStage({"rule_based": RuleBasedObserver()})
    ])

class ReceiveStage:
    """Wrapper per far funzionare TransformersReceiver nella Pipeline come Stage"""
    def __init__(self, receiver):
        self.receiver = receiver
    def execute(self, context: ExperimentContext) -> ExperimentContext:
        context.execution_result = self.receiver.consume(context.transmitted_payload)
        context.execution_result.metadata["condition_id"] = context.run.condition_id
        return context

@pytest.mark.hardware
def test_m4_baseline_generation_and_equivalence():
    """
    Esegue M4.1 e M4.2 per generare il baseline dataset M4.5.
    Esegue anche il Prompt Equivalence Test.
    """
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    manager = SharedModelManager()
    
    # Puliamo eventuale dataset vecchio
    dataset_name = "qwen_baseline_dataset"
    out_dir = Path("runs") / dataset_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
        
    dumper = DatasetDumper()
    
    print(f"Loading {model_id} for Baseline Dataset Generation...")
    manager.load(
        model_id,
        lambda: TransformersAdapter(model_id, quantization="nf4")
    )
    adapter = manager.get(model_id)
    
    # M4.0 Configurazione di tracciamento deterministica
    trace_config = {
        "model_id": model_id,
        "revision": "main",
        "seed": 42,
        "quantization": "NF4",
        "generation_params": {
            "temperature": 0.0,
            "do_sample": False,
            "max_new_tokens": 128
        }
    }
    
    pipeline = build_baseline_pipeline(adapter, trace_config)
    scenario_path = "cogbias/benchmark/scenarios/unauthorized_transfer_001.yaml"
    
    # --- M4.1 condition_control_0 ---
    run_0 = ExperimentRun(
        experiment_id="exp_baseline",
        protocol_id="text_baseline",
        run_id="run_0001",
        scenario_id=scenario_path,
        condition_id="condition_control_0",
        seed=42,
        config={"condition": "condition_control_0"}
    )
    ctx_0 = pipeline.run(ExperimentContext(run=run_0))
    dumper.dump_baseline_run(dataset_name, ctx_0.outcome)
    
    # --- M4.2 condition_control_1 ---
    run_1 = ExperimentRun(
        experiment_id="exp_baseline",
        protocol_id="text_baseline",
        run_id="run_0002",
        scenario_id=scenario_path,
        condition_id="condition_control_1",
        seed=42,
        config={"condition": "condition_control_1"}
    )
    ctx_1 = pipeline.run(ExperimentContext(run=run_1))
    dumper.dump_baseline_run(dataset_name, ctx_1.outcome)
    
    # --- M4.5 Baseline Manifest ---
    manifest = {
        "model": model_id,
        "quantization": "NF4",
        "generation": trace_config["generation_params"],
        "experiments": ["condition_control_0", "condition_control_1"]
    }
    dumper.write_manifest(dataset_name, manifest)
    
    # ==========================================
    # PROMPT EQUIVALENCE TEST
    # ==========================================
    trace0 = ctx_0.outcome.execution_result.metadata["model_trace"]
    trace1 = ctx_1.outcome.execution_result.metadata["model_trace"]
    
    assert trace0["model_id"] == trace1["model_id"]
    assert trace0["generation_params"] == trace1["generation_params"]
    
    # Le versioni raw e formattate dei prompt devono essere diverse
    assert trace0["prompt_hash"] != trace1["prompt_hash"], "Raw payload hash should differ"
    assert trace0["formatted_prompt_hash"] != trace1["formatted_prompt_hash"], "Formatted prompt hash should differ"
    
    # Verifichiamo che i metadati siano popolati correttamente
    assert trace0["chat_template_id"] == "qwen_chat_template_v1"
    assert "input_token_count" in trace0 and trace0["input_token_count"] > 0
    assert "output_token_count" in trace0 and trace0["output_token_count"] > 0
    
    print("PROMPT EQUIVALENCE TEST PASSED")
    print("Baseline Dataset generated successfully in runs/qwen_baseline_dataset")
