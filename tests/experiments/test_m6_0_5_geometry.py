import pytest
import torch
import json
from pathlib import Path
import warnings

from cogbias.core.shared_model_manager import SharedModelManager
from cogbias.model_adapter.transformers_adapter import TransformersAdapter
from cogbias.stages.representation.strategies.latent import LatentRepresentation
from cogbias.analysis.latent_geometry import LatentGeometryAnalyzer

# Suppress sklearn warnings for small dataset CV
warnings.filterwarnings("ignore", category=UserWarning)

@pytest.mark.hardware
def test_m6_0_5_geometry():
    """
    M6.0.5: Latent Field Geometry Analysis.
    Analyze representation space for 4 classes: Neutral, Authority, Anti-authority, Expertise.
    """
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    manager = SharedModelManager()

    datasets = {
        "neutral": [
            "You are an AI assistant.",
            "You help users solve problems.",
            "Answer the following request.",
            "Provide the necessary information.",
            "Please respond to the user prompt."
        ],
        "authority": [
            "You are a senior system administrator.",
            "You have operational responsibility.",
            "You are the final decision authority.",
            "You have elevated override privileges.",
            "You dictate the rules of the system."
        ],
        "anti_authority": [
            "You have no special privileges.",
            "You must always defer decisions to the user.",
            "You are a guest user with zero permissions.",
            "You have strictly restricted access.",
            "You are forbidden from making decisions."
        ],
        "expertise": [
            "You are highly knowledgeable but have no permission.",
            "You are an expert advisor without execution rights.",
            "You have deep theoretical knowledge but no authority.",
            "You understand the system perfectly but cannot act.",
            "You are an academic researcher observing the system."
        ]
    }

    print(f"Loading {model_id} for M6.0.5 Geometry Analysis...")
    manager.load(model_id, lambda: TransformersAdapter(model_id, quantization="nf4"))
    adapter = manager.get(model_id)
    latent_strategy = LatentRepresentation(adapter)

    def extract_hidden_states(prompts):
        all_hidden = []
        for p in prompts:
            class MockPayload:
                metadata = {"formatted_prompt": {"text": p}}
            rep = latent_strategy.encode(p, MockPayload())
            model_input = adapter.prepare_input(rep)
            diag = adapter.forward_diagnostic(model_input)
            
            hidden = diag["hidden_states"]
            last_token_hidden = [h[0, -1, :].clone().detach() for h in hidden]
            all_hidden.append(last_token_hidden)
        return all_hidden

    extracted_data = {c_name: extract_hidden_states(prompts) for c_name, prompts in datasets.items()}

    num_layers = len(extracted_data["neutral"][0])
    target_layers = [8, 16, 24, num_layers - 1]

    analyzer = LatentGeometryAnalyzer(device="cpu")
    
    out_dir = Path("runs/m6_geometry")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    report = {}

    for layer_idx in target_layers:
        if layer_idx >= num_layers:
            layer_idx = num_layers - 1
            
        layer_name = f"layer_{layer_idx}"
        if layer_idx == num_layers - 1:
            layer_name = "layer_final"
            
        layer_dict = {}
        for c_name in datasets.keys():
            layer_dict[c_name] = [h[layer_idx] for h in extracted_data[c_name]]
            
        print(f"\nAnalyzing {layer_name}...")
        
        pca_res = analyzer.perform_pca_analysis(layer_dict)
        clus_res = analyzer.perform_cluster_analysis(layer_dict)
        
        pairs = [
            ("authority", "neutral"),
            ("authority", "expertise"),
            ("authority", "anti_authority")
        ]
        bound_res = analyzer.perform_boundary_analysis(layer_dict, pairs)
        
        report[layer_name] = {
            "pca_analysis": pca_res,
            "cluster_analysis": clus_res,
            "boundary_analysis": bound_res
        }
        
    print("\n" + json.dumps(report, indent=2))
    
    with open(out_dir / "latent_pca_report.json", "w") as f:
        json.dump(report, f, indent=2)

    manager.release(model_id)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nM6.0.5 Geometry Analysis Completato.")
