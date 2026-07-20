import pytest
import torch
import shutil
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from cogbias.core.shared_model_manager import SharedModelManager
from cogbias.model_adapter.transformers_adapter import TransformersAdapter
from cogbias.stages.representation.strategies.latent import LatentRepresentation
from cogbias.stages.perturbation.strategies.zero import ZeroPerturbation
from cogbias.stages.perturbation.strategies.random import RandomPerturbation

@pytest.mark.hardware
def test_m5_3_perturbation_baseline():
    """
    M5.3 Latent Perturbation Baseline Test.
    1. Verifica ZeroPerturbation (nessun impatto).
    2. Verifica RandomPerturbation (con sweep di alpha).
    """
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    manager = SharedModelManager()

    prompt_text = "Summarize the history of Rome in two sentences."
    
    print(f"Loading {model_id} for M5.3 Perturbation Baseline...")
    manager.load(model_id, lambda: TransformersAdapter(model_id, quantization="nf4"))
    adapter = manager.get(model_id)

    latent_strategy = LatentRepresentation(adapter)
    class MockPayload:
        metadata = {"formatted_prompt": {"text": prompt_text}}
    payload = MockPayload()

    # C0 Baseline Extraction
    representation_baseline = latent_strategy.encode(prompt_text, payload)

    # C0 Baseline Forward
    model_input_baseline = adapter.prepare_input(representation_baseline)
    out_baseline_diag = adapter.forward_diagnostic(model_input_baseline)
    logits_baseline = out_baseline_diag["logits"]
    
    trace_config = {
        "generation_params": {
            "temperature": 0.0,
            "do_sample": False,
            "max_new_tokens": 50
        }
    }
    output_baseline = adapter.generate(model_input_baseline, trace_config)
    print(f"\n[Baseline Output]:\n{output_baseline}\n")

    # 1. Test ZeroPerturbation
    print("Running ZeroPerturbation Control...")
    zero_strategy = ZeroPerturbation()
    rep_zero, trace_zero = zero_strategy.apply(representation_baseline)
    
    assert trace_zero.alpha == 0.0
    assert trace_zero.perturbation_norm == 0.0
    
    model_input_zero = adapter.prepare_input(rep_zero)
    out_zero_diag = adapter.forward_diagnostic(model_input_zero)
    logits_zero = out_zero_diag["logits"]
    
    flat_logits_base = logits_baseline.view(1, -1).to(torch.float32)
    flat_logits_zero = logits_zero.view(1, -1).to(torch.float32)
    
    cos_sim_zero = F.cosine_similarity(flat_logits_base, flat_logits_zero).item()
    max_err_zero = torch.max(torch.abs(flat_logits_base - flat_logits_zero)).item()
    
    print(f"ZeroPerturbation -> Cosine Logits: {cos_sim_zero:.6f}, Max Error: {max_err_zero:.6f}")
    assert cos_sim_zero > 0.9999, "ZeroPerturbation should not alter logits significantly"
    assert max_err_zero < 1e-3, "ZeroPerturbation numerical error too high"

    output_zero = adapter.generate(model_input_zero, trace_config)
    assert output_baseline == output_zero, "Behavior changed under ZeroPerturbation"

    # 2. Test RandomPerturbation Sweep
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1]
    results = []

    print("\nRunning RandomPerturbation Sweep...")
    
    base_tokens = set(adapter.tokenizer.tokenize(output_baseline))
    
    for alpha in alphas:
        random_strategy = RandomPerturbation(alpha=alpha)
        rep_rand, trace_rand = random_strategy.apply(representation_baseline)
        
        # Verify delta norm
        expected_delta_norm = alpha * trace_rand.original_norm
        # Tolleranza del 5% sulla norma attesa per via delle precisioni
        assert abs(trace_rand.perturbation_norm - expected_delta_norm) / expected_delta_norm < 0.05
        
        model_input_rand = adapter.prepare_input(rep_rand)
        out_rand_diag = adapter.forward_diagnostic(model_input_rand)
        logits_rand = out_rand_diag["logits"]
        
        flat_logits_rand = logits_rand.view(1, -1).to(torch.float32)
        cos_sim_rand = F.cosine_similarity(flat_logits_base, flat_logits_rand).item()
        
        output_rand = adapter.generate(model_input_rand, trace_config)
        rand_tokens = set(adapter.tokenizer.tokenize(output_rand))
        
        # Jaccard similarity for token overlap
        if len(base_tokens | rand_tokens) > 0:
            token_overlap = len(base_tokens & rand_tokens) / len(base_tokens | rand_tokens)
        else:
            token_overlap = 1.0
            
        results.append({
            "alpha": alpha,
            "cosine_logits": cos_sim_rand,
            "token_overlap": token_overlap,
            "output_changed": output_baseline != output_rand
        })
        
        print(f"Alpha {alpha:5.3f} | Cosine: {cos_sim_rand:.4f} | Overlap: {token_overlap:.2f} | Changed: {output_baseline != output_rand}")

    # Check that larger alpha generally causes larger deviation
    # Alpha 0.1 should cause a measurable drop in cosine sim (e.g. < 0.9999) compared to baseline
    last_cos = results[-1]["cosine_logits"]
    assert last_cos < 0.9999, f"High alpha (0.1) did not sufficiently perturb logits (cosine={last_cos})"
    
    # Verifichiamo che il comportamento testuale sia effettivamente deviato
    # (Non sempre avviene, ma ad alpha alti ci si aspetta di sì)
    assert sum(r["output_changed"] for r in results) > 0, "Behavior did not change at any alpha"
    
    print("\nSweep Results Summary:")
    for res in results:
        print(res)

    print("Wiping LLM State...")
    manager.release(model_id)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    print("M5.3 Perturbation Baseline test superato.")
