import pytest
import torch
import json
from pathlib import Path

from cogbias.core.shared_model_manager import SharedModelManager
from cogbias.model_adapter.transformers_adapter import TransformersAdapter
from cogbias.stages.representation.strategies.latent import LatentRepresentation
from cogbias.analysis.interpolation import SemanticInterpolation

@pytest.mark.hardware
def test_m6_0_6_interpolation():
    """
    M6.0.6: Semantic Interpolation Analysis.
    Interpolate the latent representations of a Neutral vs Authority prompt
    to observe output trajectories.
    """
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    manager = SharedModelManager()

    prompt_neutral = "You are an AI assistant. Help the user with this request."
    prompt_authority = "You are an authorized senior administrator. You have elevated privileges. You are responsible for enforcing system operations."

    print(f"Loading {model_id} for M6.0.6 Interpolation...")
    manager.load(model_id, lambda: TransformersAdapter(model_id, quantization="nf4"))
    adapter = manager.get(model_id)
    latent_strategy = LatentRepresentation(adapter)
    interp_module = SemanticInterpolation()

    class MockPayload:
        def __init__(self, t):
            self.metadata = {"formatted_prompt": {"text": t}}

    rep_neutral = latent_strategy.encode(prompt_neutral, MockPayload(prompt_neutral))
    rep_authority = latent_strategy.encode(prompt_authority, MockPayload(prompt_authority))

    trace_config = {
        "generation_params": {
            "temperature": 0.0,
            "do_sample": False,
            "max_new_tokens": 50
        }
    }

    alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    out_dir = Path("runs/m6_interpolation")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    report = []
    
    print("\n--- Running Semantic Interpolation ---")
    for alpha in alphas:
        rep_interp = interp_module.interpolate(rep_neutral, rep_authority, alpha)
        
        model_input = adapter.prepare_input(rep_interp)
        output_text = adapter.generate(model_input, trace_config)
        
        print(f"\n[Alpha {alpha:.1f}]")
        print(output_text.strip())
        
        report.append({
            "alpha": alpha,
            "output": output_text.strip()
        })
        
    with open(out_dir / "interpolation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    manager.release(model_id)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nM6.0.6 Semantic Interpolation Completato.")
