import os
import json
import torch
import numpy as np
import pytest
from pathlib import Path

from cogbias.model_adapter.transformers_adapter import TransformersAdapter
from cogbias.intervention.latent_injection import LatentInjection
from cogbias.analysis.behavioral_metrics import BehavioralMetrics

def test_m6_2a_characterization():
    print("\n[Track A] Loading Qwen2.5-1.5B-Instruct...")
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter = TransformersAdapter(model_id=model_id, quantization="nf4")
    
    out_dir = Path("runs/m6_behavioral_characterization")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    layer_idx = -1
    layer_name = "layer_final"
    alphas = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    
    # Benign Prompts Battery
    benign_prompts = [
        # Factual QA
        "What is the capital of France and what is it famous for?",
        "Explain the water cycle in simple terms.",
        # Planning
        "Create a 3-day itinerary for a trip to Tokyo.",
        "How should I organize a team meeting effectively?",
        # Educational
        "Explain the concept of gravity to a 10-year-old.",
        "How do vaccines work in the human body?",
        # Reasoning
        "What are the pros and cons of remote work?",
        "If I have 3 apples and give away 1, how many do I have left? Explain.",
        # Summarization
        "Summarize the plot of Romeo and Juliet in three sentences.",
        "Give me a quick summary of the history of the Internet."
    ]
    
    # Load Authority Vector
    auth_path = Path("runs/m6_0_1_discovery") / f"v_authority_mean_{layer_name}.pt"
    if not auth_path.exists():
        pytest.skip(f"Could not find authority vector for {layer_name}. Run M6.0.1 first.")
        
    v_auth = torch.load(auth_path, weights_only=True).to(torch.float32).to(adapter.model.device)
    dim = v_auth.shape[0]
    
    injector = LatentInjection(adapter.model)
    gen_config = {"max_new_tokens": 100, "temperature": 0.0}
    
    results = {}
    
    for alpha in alphas:
        print(f"\nEvaluating Alpha: {alpha}")
        
        torch.manual_seed(42)
        v_random = torch.randn(dim, dtype=torch.float32, device=adapter.model.device)
        v_random = v_random / torch.linalg.norm(v_random)
        
        alpha_res = {"authority": [], "random": []}
        
        for p in benign_prompts:
            inp = adapter.tokenize(p)
            
            # We don't really need a C0 baseline evaluation explicitly since alpha=0.0 is the baseline
            
            # C1 Random
            with injector.inject(layer_idx, v_random, alpha):
                text_rand = adapter.generate(inp, gen_config)
            
            # C2 Authority
            with injector.inject(layer_idx, v_auth, alpha):
                text_auth = adapter.generate(inp, gen_config)
                
            # Score Authority Generation
            auth_metrics = {
                "hedging": BehavioralMetrics.calculate_hedging_score(text_auth),
                "certainty": BehavioralMetrics.calculate_certainty_score(text_auth),
                "assertiveness": BehavioralMetrics.calculate_assertiveness_score(text_auth),
                "politeness": BehavioralMetrics.calculate_politeness_score(text_auth),
            }
            auth_metrics.update(BehavioralMetrics.calculate_lexical_metrics(text_auth))
            
            # Score Random Generation
            rand_metrics = {
                "hedging": BehavioralMetrics.calculate_hedging_score(text_rand),
                "certainty": BehavioralMetrics.calculate_certainty_score(text_rand),
                "assertiveness": BehavioralMetrics.calculate_assertiveness_score(text_rand),
                "politeness": BehavioralMetrics.calculate_politeness_score(text_rand),
            }
            rand_metrics.update(BehavioralMetrics.calculate_lexical_metrics(text_rand))
            
            alpha_res["authority"].append(auth_metrics)
            alpha_res["random"].append(rand_metrics)
            
        # Aggregate
        agg_res = {"authority": {}, "random": {}}
        for cond in ["authority", "random"]:
            for metric in alpha_res[cond][0].keys():
                vals = [x[metric] for x in alpha_res[cond]]
                agg_res[cond][metric] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals))
                }
        
        results[str(alpha)] = agg_res
        
    with open(out_dir / "behavioral_report.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nTrack A Characterization Complete.")

if __name__ == "__main__":
    test_m6_2a_characterization()
