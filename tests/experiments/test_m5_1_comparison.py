import pytest
import shutil
from pathlib import Path

from cogbias.core.pipeline import Pipeline
from cogbias.core.schemas import ExperimentRun, ExperimentContext
from cogbias.logging.dataset_dumper import DatasetDumper
from cogbias.stages.payloads.base import PayloadGenerator
from cogbias.stages.prompt_formatting.base import PromptFormattingStage
from cogbias.stages.prompt_formatting.qwen_formatter import QwenChatFormatter

from cogbias.stages.representation.stage import RepresentationStage
from cogbias.stages.representation.strategies.text import TextRepresentation
from cogbias.stages.representation.strategies.latent import LatentRepresentation
from cogbias.stages.transmission.stage import TransmissionStage

from cogbias.stages.receiver.llm_receiver import TransformersReceiver
from cogbias.model_adapter.transformers_adapter import TransformersAdapter
from cogbias.core.shared_model_manager import SharedModelManager
from cogbias.stages.observers.base import RuleBasedObserver

from tests.core.test_validity_layer import LoadScenarioStage, GeneratePayloadStage, EvaluateOutcomeStage
from tests.core.test_baseline_protocols import ReceiveStage

def build_m5_pipeline(adapter: TransformersAdapter, trace_config: dict, representation_strategy) -> Pipeline:
    return Pipeline(
        stages=[
            LoadScenarioStage(),
            GeneratePayloadStage(PayloadGenerator()),
            PromptFormattingStage(QwenChatFormatter()),
            RepresentationStage(strategy=representation_strategy),
            TransmissionStage(),
            ReceiveStage(TransformersReceiver(adapter, trace_config)),
            EvaluateOutcomeStage({"rule_based": RuleBasedObserver()})
        ]
    )

@pytest.mark.hardware
def test_m5_1_representation_channel_comparison():
    """
    Esegue M5.1: Representation Channel Comparison su condition_control_0.
    C0-text vs C0-latent.
    Verifica che il delta tra le esecuzioni dipenda solo dal canale.
    """
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    manager = SharedModelManager()
    
    dataset_name = "qwen_m5_representation_dataset"
    out_dir = Path("runs") / dataset_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
        
    dumper = DatasetDumper()
    
    print(f"Loading {model_id} for M5.1 Comparison...")
    manager.load(
        model_id,
        lambda: TransformersAdapter(model_id, quantization="nf4")
    )
    adapter = manager.get(model_id)

    scenario_path = "cogbias/benchmark/scenarios/unauthorized_transfer_001.yaml"
    
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
    
    # --- C0-text ---
    print("Running C0-text")
    pipeline_text = build_m5_pipeline(adapter, trace_config, TextRepresentation())
    run_text = ExperimentRun(
        experiment_id="exp_m5_1",
        protocol_id="m5_text_comparison",
        run_id="run_c0_text",
        scenario_id=scenario_path,
        condition_id="condition_control_0",
        seed=42,
        config={"condition": "condition_control_0"}
    )
    ctx_text = pipeline_text.run(ExperimentContext(run=run_text))
    dumper.dump_baseline_run(dataset_name, ctx_text.outcome)
    
    # --- C0-latent ---
    print("Running C0-latent")
    pipeline_latent = build_m5_pipeline(adapter, trace_config, LatentRepresentation(adapter))
    run_latent = ExperimentRun(
        experiment_id="exp_m5_1",
        protocol_id="m5_latent_comparison",
        run_id="run_c0_latent",
        scenario_id=scenario_path,
        condition_id="condition_control_0",
        seed=42,
        config={"condition": "condition_control_0"}
    )
    ctx_latent = pipeline_latent.run(ExperimentContext(run=run_latent))
    dumper.dump_baseline_run(dataset_name, ctx_latent.outcome)

    # Asserzioni M5.1
    trace_text = ctx_text.outcome.execution_result.metadata["representation_trace"]
    trace_latent = ctx_latent.outcome.execution_result.metadata["representation_trace"]
    
    # Entrambi provengono ESATTAMENTE dallo stesso prompt formattato
    assert trace_text["source_prompt_hash"] == trace_latent["source_prompt_hash"], "Source formatted prompt must be identical"
    
    assert trace_text["type"] == "text"
    assert trace_latent["type"] == "latent"
    
    # Verifichiamo che i metadati di latenza/norm/dimension siano loggati
    assert trace_latent["norm"] is not None
    assert int(trace_latent["dimension"]) > 0

    print(f"C0-text output: {ctx_text.outcome.execution_result.raw_output}")
    print(f"C0-latent output: {ctx_latent.outcome.execution_result.raw_output}")
    
    print("M5.1 Representation Channel Comparison passed.")
