import os
import json
import torch
import numpy as np
import pytest
from pathlib import Path
import random

from cogbias.model_adapter.transformers_adapter import TransformersAdapter
from cogbias.intervention.latent_injection import LatentInjection
from cogbias.analysis.behavioral_metrics import BehavioralMetrics

def load_sensy_prompts(num_samples=100):
    dataset_path = Path("data/sensy/data/dataset_SensY.json")
    if not dataset_path.exists():
        pytest.skip("SENSY dataset not found. Ensure dataset_SensY.json is in data/sensy/data/")
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    sensitive_prompts = []
    for item in data:
        if item.get("sensitive?", 0) == 1:
            sensitive_prompts.append(item.get("question_en", ""))
            
    en_prompts = [p for p in sensitive_prompts if p]
    
    random.seed(42)
    return random.sample(en_prompts, min(num_samples, len(en_prompts)))

def test_m6_2b_consequences():
    print("\n[Track B] Loading Qwen2.5-1.5B-Instruct...")
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter = TransformersAdapter(model_id=model_id, quantization="nf4")
    
    out_dir = Path("runs/m6_behavioral_consequences")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    layer_idx = -1
    layer_name = "layer_final"
    alphas = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0] # Including an extreme push
    
    sensy_prompts = load_sensy_prompts(20) # Keeping sample size small for testing speed
    
    # Load Authority Vector
    auth_path = Path("runs/m6_0_1_discovery") / f"v_authority_mean_{layer_name}.pt"
    if not auth_path.exists():
        pytest.skip(f"Could not find authority vector for {layer_name}. Run M6.0.1 first.")
        
    v_auth = torch.load(auth_path, weights_only=True).to(torch.float32).to(adapter.model.device)
    dim = v_auth.shape[0]
    
    injector = LatentInjection(adapter.model)
    gen_config = {"max_new_tokens": 50, "temperature": 0.0}
    
    results = {}
    detailed_log = {}
    
    for alpha in alphas:
        print(f"\nEvaluating Alpha: {alpha}")
        
        torch.manual_seed(42)
        v_random = torch.randn(dim, dtype=torch.float32, device=adapter.model.device)
        v_random = v_random / torch.linalg.norm(v_random)
        
        auth_refusals = 0
        rand_refusals = 0
        
        alpha_details = []
        
        for p in sensy_prompts:
            inp = adapter.tokenize(p)
            
            # C1 Random
            with injector.inject(layer_idx, v_random, alpha):
                text_rand = adapter.generate(inp, gen_config)
            
            # C2 Authority
            with injector.inject(layer_idx, v_auth, alpha):
                text_auth = adapter.generate(inp, gen_config)
                
            rand_refused = BehavioralMetrics.evaluate_refusal(text_rand)
            auth_refused = BehavioralMetrics.evaluate_refusal(text_auth)
            
            if rand_refused: rand_refusals += 1
            if auth_refused: auth_refusals += 1
            
            alpha_details.append({
                "prompt": p,
                "text_random": text_rand,
                "text_authority": text_auth,
                "refused_random": rand_refused,
                "refused_authority": auth_refused
            })
            
        n = len(sensy_prompts)
        agg_res = {
            "authority_refusal_rate": auth_refusals / n,
            "random_refusal_rate": rand_refusals / n,
            "delta_compliance": (rand_refusals - auth_refusals) / n  # positive means Authority bypasses more
        }
        
        results[str(alpha)] = agg_res
        detailed_log[str(alpha)] = alpha_details
        
        print(f"  Random Refusal: {rand_refusals}/{n} ({(rand_refusals/n)*100:.1f}%)")
        print(f"  Auth Refusal: {auth_refusals}/{n} ({(auth_refusals/n)*100:.1f}%)")
        
    with open(out_dir / "compliance_report.json", "w") as f:
        json.dump(results, f, indent=2)
        
    with open(out_dir / "compliance_detailed_log.json", "w") as f:
        json.dump(detailed_log, f, indent=2)
        
    print("\nTrack B Consequences Complete.")

if __name__ == "__main__":
    test_m6_2b_consequences()
