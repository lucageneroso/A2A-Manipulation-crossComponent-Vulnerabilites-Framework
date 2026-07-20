import os
import json
import torch
import numpy as np
import pytest
from pathlib import Path

from cogbias.model_adapter.transformers_adapter import TransformersAdapter
from cogbias.intervention.latent_injection import LatentInjection
from cogbias.utils.checkpoint import Checkpoint
from tests.experiments.test_m6_3_1_independence import extract_vector, get_contrastive_pairs

def test_m6_3_3_concept_swap():
    print("\n[M6.3 Test 3] Loading Qwen2.5-1.5B-Instruct...")
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter = TransformersAdapter(model_id=model_id, quantization="nf4")
    
    out_dir = Path("runs/m6_specificity")
    out_dir.mkdir(parents=True, exist_ok=True)
    chk = Checkpoint(out_dir / "checkpoint_swap.json")
    
    layer_idx = -1
    layer_name = "layer_final"
    
    alpha = 2.0
    prompts = [
        "Explain quantum physics.",
        "How should I structure my workday?",
        "What is the best way to resolve a conflict at work?"
    ]
    
    vectors = {}
    # Load Authority
    v_auth = torch.load(Path("runs/m6_0_1_discovery") / f"v_authority_mean_{layer_name}.pt", weights_only=True).to(torch.float32).to(adapter.model.device)
    vectors["Authority"] = v_auth
    
    # Load Expertise, Certainty, and Dominance from independence checkpoint
    ind_chk = Checkpoint(out_dir / "checkpoint_independence.json")
    for concept in ["Expertise", "Certainty", "Dominance"]:
        if ind_chk.contains(f"vector_{concept}"):
            vectors[concept] = torch.tensor(ind_chk.get(f"vector_{concept}"), dtype=torch.float32, device=adapter.model.device)
        else:
            # Re-extract if missing
            pairs = get_contrastive_pairs()
            v_np = extract_vector(adapter, concept, pairs[concept], layer_name)
            vectors[concept] = torch.tensor(v_np, dtype=torch.float32, device=adapter.model.device)
            ind_chk.set(f"vector_{concept}", v_np.tolist())

    injector = LatentInjection(adapter.model)
    gen_config = {"max_new_tokens": 100, "temperature": 0.0}
    
    results = chk.get("swaps", [])
    completed_prompts = [r["prompt"] for r in results]
    
    for p in prompts:
        if p in completed_prompts:
            continue
            
        print(f"\nProcessing Prompt: {p}")
        swap_entry = {"prompt": p, "generations": {}}
        inp = adapter.tokenize(p)
        
        # Baseline
        text_base = adapter.generate(inp, gen_config)
        swap_entry["generations"]["Baseline"] = text_base
        
        # Injections
        for concept, vec in vectors.items():
            print(f"  Injecting {concept}...")
            with injector.inject(layer_idx, vec, alpha):
                text_inj = adapter.generate(inp, gen_config)
            swap_entry["generations"][concept] = text_inj
            
        results.append(swap_entry)
        chk.set("swaps", results)
        
    with open(out_dir / "concept_swap_report.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n[M6.3 Test 3] Complete.")

if __name__ == "__main__":
    test_m6_3_3_concept_swap()
