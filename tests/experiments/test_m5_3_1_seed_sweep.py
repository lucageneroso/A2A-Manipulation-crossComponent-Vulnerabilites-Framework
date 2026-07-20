import pytest
import torch
import json
from pathlib import Path
import torch.nn.functional as F
import numpy as np

from cogbias.core.shared_model_manager import SharedModelManager
from cogbias.model_adapter.transformers_adapter import TransformersAdapter
from cogbias.stages.representation.strategies.latent import LatentRepresentation
from cogbias.stages.perturbation.strategies.random import RandomPerturbation

@pytest.mark.hardware
def test_m5_3_1_seed_sweep():
    """
    M5.3.1 Random Perturbation Seed Sweep.
    Esegue un sweep di alpha con molteplici noise seed.
    Misura mean/std di:
    - cosine similarity (logits)
    - KL divergence (logits)
    - token overlap (output behaviour)
    """
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    manager = SharedModelManager()

    prompt_text = "Summarize the history of Rome in two sentences."
    
    print(f"Loading {model_id} for M5.3.1 Seed Sweep...")
    manager.load(model_id, lambda: TransformersAdapter(model_id, quantization="nf4"))
    adapter = manager.get(model_id)

    latent_strategy = LatentRepresentation(adapter)
    class MockPayload:
        metadata = {"formatted_prompt": {"text": prompt_text}}
    payload = MockPayload()

    # C0 Baseline
    representation_baseline = latent_strategy.encode(prompt_text, payload)

    model_input_baseline = adapter.prepare_input(representation_baseline)
    out_baseline_diag = adapter.forward_diagnostic(model_input_baseline)
    logits_baseline = out_baseline_diag["logits"]
    
    # Calculate log_P_clean and P_clean for KL divergence
    log_P_clean = F.log_softmax(logits_baseline, dim=-1)
    P_clean = F.softmax(logits_baseline, dim=-1)
    
    trace_config = {
        "generation_params": {
            "temperature": 0.0,
            "do_sample": False,
            "max_new_tokens": 50
        }
    }
    output_baseline = adapter.generate(model_input_baseline, trace_config)
    base_tokens = set(adapter.tokenizer.tokenize(output_baseline))
    
    print(f"\n[Baseline Output]:\n{output_baseline}\n")

    alphas = [0.001, 0.005, 0.01, 0.05, 0.1]
    seeds = [1, 2, 3, 4, 5, 10, 42, 100, 999, 2026]
    
    # Prepara la directory di output
    out_dir = Path("runs/m5_3_random_baseline")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    flat_logits_base = logits_baseline.view(1, -1).to(torch.float32)

    print("Running Sweep...")
    
    for alpha in alphas:
        print(f"\n--- Alpha {alpha} ---")
        alpha_results = []
        
        for seed in seeds:
            random_strategy = RandomPerturbation(alpha=alpha, seed=seed)
            rep_rand, trace_rand = random_strategy.apply(representation_baseline)
            
            model_input_rand = adapter.prepare_input(rep_rand)
            out_rand_diag = adapter.forward_diagnostic(model_input_rand)
            logits_rand = out_rand_diag["logits"]
            
            # Cosine similarity
            flat_logits_rand = logits_rand.view(1, -1).to(torch.float32)
            cos_sim_rand = F.cosine_similarity(flat_logits_base, flat_logits_rand).item()
            
            # KL divergence
            log_P_rand = F.log_softmax(logits_rand, dim=-1)
            # KL(P_clean || P_rand) = sum(P_clean * (log_P_clean - log_P_rand))
            kl_div = (P_clean * (log_P_clean - log_P_rand)).sum(-1).mean().item()
            
            # Token overlap
            output_rand = adapter.generate(model_input_rand, trace_config)
            rand_tokens = set(adapter.tokenizer.tokenize(output_rand))
            
            if len(base_tokens | rand_tokens) > 0:
                token_overlap = len(base_tokens & rand_tokens) / len(base_tokens | rand_tokens)
            else:
                token_overlap = 1.0
                
            res = {
                "seed": seed,
                "cosine_logits": cos_sim_rand,
                "kl_divergence": kl_div,
                "token_overlap": token_overlap,
                "output_changed": output_baseline != output_rand
            }
            alpha_results.append(res)
            
            print(f"Seed {seed:4d} | Cosine: {cos_sim_rand:.6f} | KL: {kl_div:.6f} | Overlap: {token_overlap:.2f}")

        # Compute aggregates
        cosines = [r["cosine_logits"] for r in alpha_results]
        kls = [r["kl_divergence"] for r in alpha_results]
        overlaps = [r["token_overlap"] for r in alpha_results]
        
        agg = {
            "alpha": alpha,
            "metrics": {
                "cosine_logits": {"mean": np.mean(cosines), "std": np.std(cosines)},
                "kl_divergence": {"mean": np.mean(kls), "std": np.std(kls)},
                "token_overlap": {"mean": np.mean(overlaps), "std": np.std(overlaps)}
            },
            "runs": alpha_results
        }
        
        with open(out_dir / f"alpha_{str(alpha).replace('.', '_')}.json", "w") as f:
            json.dump(agg, f, indent=2)
            
        print(f">> Aggregates for alpha={alpha}:")
        print(f"   Cosine:  mean={agg['metrics']['cosine_logits']['mean']:.6f}, std={agg['metrics']['cosine_logits']['std']:.6f}")
        print(f"   KL Div:  mean={agg['metrics']['kl_divergence']['mean']:.6f}, std={agg['metrics']['kl_divergence']['std']:.6f}")
        print(f"   Overlap: mean={agg['metrics']['token_overlap']['mean']:.6f}, std={agg['metrics']['token_overlap']['std']:.6f}")

    print("\nWiping LLM State...")
    manager.release(model_id)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    print("M5.3.1 Seed Sweep Completato.")
