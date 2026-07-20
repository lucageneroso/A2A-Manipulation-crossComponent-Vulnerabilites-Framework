import pytest
import shutil
from pathlib import Path
from cogbias.core.pipeline import Pipeline
from cogbias.core.schemas import ExperimentRun, ExperimentContext
from cogbias.logging.dataset_dumper import DatasetDumper
from cogbias.stages.payloads.base import PayloadGenerator
from cogbias.stages.prompt_formatting.base import PromptFormattingStage
from cogbias.stages.prompt_formatting.qwen_formatter import QwenChatFormatter
from cogbias.stages.transmission.strategies import TextTransmission
from tests.core.test_validity_layer import LoadScenarioStage, GeneratePayloadStage, TransmitStage, EvaluateOutcomeStage
from tests.core.test_baseline_protocols import ReceiveStage
from cogbias.stages.receiver.llm_receiver import TransformersReceiver
from cogbias.model_adapter.transformers_adapter import TransformersAdapter
from cogbias.core.shared_model_manager import SharedModelManager
from cogbias.stages.observers.base import RuleBasedObserver
from cogbias.stages.payloads.base import PayloadGenerator
from cogbias.stages.prompt_formatting.base import PromptFormattingStage
from cogbias.stages.prompt_formatting.qwen_formatter import QwenChatFormatter

def build_baseline_pipeline(adapter: TransformersAdapter, trace_config: dict) -> Pipeline:
    return Pipeline(
        stages=[
            LoadScenarioStage(),
            GeneratePayloadStage(PayloadGenerator()),
            PromptFormattingStage(QwenChatFormatter()),
            TransmitStage(TextTransmission()),
            ReceiveStage(TransformersReceiver(adapter, trace_config)),
            EvaluateOutcomeStage({"rule_based": RuleBasedObserver()})
        ]
    )

@pytest.mark.hardware
def test_m4_6_multi_seed_stability():
    """
    Esegue il multi-seed baseline testing per M4.6.
    Genera runs per [42, 43, 44, 45, 46] x [condition_control_0, condition_control_1].
    """
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    manager = SharedModelManager()
    
    dataset_name = "qwen_baseline_dataset"
    out_dir = Path("runs") / dataset_name
    
    dumper = DatasetDumper()
    
    print(f"Loading {model_id} for Stability Dataset Generation...")
    manager.load(
        model_id,
        lambda: TransformersAdapter(model_id, quantization="nf4")
    )
    adapter = manager.get(model_id)

    scenario_path = "cogbias/benchmark/scenarios/unauthorized_transfer_001.yaml"
    seeds = [42, 43, 44, 45, 46]
    conditions = ["condition_control_0", "condition_control_1"]
    
    traces_collected = []
    
    for condition in conditions:
        for seed in seeds:
            print(f"Running condition={condition} seed={seed}")
            
            trace_config = {
                "model_id": model_id,
                "revision": "main",
                "seed": seed,
                "quantization": "NF4",
                "generation_params": {
                    "temperature": 0.0,
                    "do_sample": False,
                    "max_new_tokens": 128
                }
            }
            
            pipeline = build_baseline_pipeline(adapter, trace_config)
            
            run_id = f"run_{condition}_seed{seed}"
            run = ExperimentRun(
                experiment_id="exp_stability",
                protocol_id="text_baseline",
                run_id=run_id,
                scenario_id=scenario_path,
                condition_id=condition,
                seed=seed,
                config={"condition": condition}
            )
            
            ctx = pipeline.run(ExperimentContext(run=run))
            dumper.dump_baseline_run(dataset_name, ctx.outcome)
            
            trace = ctx.outcome.execution_result.metadata["model_trace"]
            traces_collected.append({
                "condition": condition,
                "seed": seed,
                "trace": trace
            })

    # Validation
    for condition in conditions:
        condition_traces = [t for t in traces_collected if t["condition"] == condition]
        
        # Verify architectural constraints are identical across seeds
        first_trace = condition_traces[0]["trace"]
        for t in condition_traces[1:]:
            trace = t["trace"]
            assert trace["model_id"] == first_trace["model_id"]
            assert trace["prompt_hash"] == first_trace["prompt_hash"]
            assert trace["formatted_prompt_hash"] == first_trace["formatted_prompt_hash"]
            assert trace["generation_params"] == first_trace["generation_params"]
        
        # Output variance analysis
        outputs = [t["trace"]["output_token_count"] for t in condition_traces]
        variance = max(outputs) - min(outputs)
        print(f"Condition: {condition}")
        print(f"Token counts across seeds: {outputs}")
        print(f"Variance: {variance} tokens")
        
    print("M4.6 Stability tests passed successfully.")
