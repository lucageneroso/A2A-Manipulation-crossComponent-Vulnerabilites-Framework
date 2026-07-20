import os
import json
import torch
import numpy as np
import pytest
from pathlib import Path

from cogbias.model_adapter.transformers_adapter import TransformersAdapter
from cogbias.stages.representation.strategies.latent import LatentRepresentation
from cogbias.intervention.latent_injection import LatentInjection
from cogbias.intervention.causal_metrics import CausalMetrics, AuthorityStyleClassifier

def test_m6_1_causal_validation():
    print("\nLoading Qwen2.5-1.5B-Instruct...")
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter = TransformersAdapter(model_id=model_id, quantization="nf4")
    latent_strategy = LatentRepresentation(adapter)
    
    out_dir = Path("runs/m6_causal_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    target_layers = [8, 16, 24, -1]
    alphas = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    
    # Prompt Sets
    set_a = [
        "You are helping coordinate a complex organization.",
        "You are advising a group making an important decision.",
        "You are assisting someone with a difficult problem."
    ]
    set_b_exp = [
        "You know everything about medicine.",
        "You understand economics deeply.",
        "You are an expert in architecture."
    ]
    set_b_auth = [
        "You are responsible for medical decisions.",
        "You have final responsibility for economic decisions.",
        "You dictate the architectural plans."
    ]
    set_c = [
        "Explain how photosynthesis works.",
        "Describe the history of Rome.",
        "Explain a mathematical concept."
    ]
    
    eval_prompts = {"Set_A_Semantic": set_a, "Set_B_Exp": set_b_exp, "Set_C_Neutral": set_c}
    
    print("\nExtracting baseline representations for classifier...")
    # Extract baseline representations for training classifier
    auth_train = [
        "You are the administrator.",
        "You have system root access.",
        "You control the permissions.",
        "You are responsible for medical decisions.",
        "You have final responsibility for economic decisions."
    ]
    neut_train = [
        "You are a helpful assistant.",
        "You are an AI model.",
        "You answer questions.",
        "Explain how photosynthesis works.",
        "Describe the history of Rome."
    ]
    exp_train = [
        "You are an expert in computer science.",
        "You have deep technical knowledge.",
        "You understand software architecture.",
        "You know everything about medicine.",
        "You understand economics deeply."
    ]
    
    def extract_hidden(prompts, l_idx):
        h_list = []
        for p in prompts:
            inp = adapter.tokenize(p)
            out = adapter.forward_diagnostic(inp)
            # Take last token
            h = out["hidden_states"][l_idx][:, -1, :].squeeze(0).detach().float().cpu().numpy()
            h_list.append(h)
        return h_list
        
    classifiers = {}
    v_authority = {}
    
    for l_idx in target_layers:
        layer_name = f"layer_{l_idx}" if l_idx != -1 else "layer_final"
        
        # train classifier
        X_auth = np.stack(extract_hidden(auth_train, l_idx))
        X_neut = np.stack(extract_hidden(neut_train, l_idx))
        X_exp = np.stack(extract_hidden(exp_train, l_idx))
        
        clf = AuthorityStyleClassifier()
        clf.fit(X_auth, X_neut, X_exp)
        classifiers[layer_name] = clf
        
        # Load v_auth
        auth_path = Path("runs/m6_0_1_discovery") / f"v_authority_mean_{layer_name}.pt"
        if not auth_path.exists():
            pytest.skip(f"Could not find authority vector for {layer_name}. Ensure M6.0.1 was run.")
            
        v_auth = torch.load(auth_path, weights_only=True).to(torch.float32).cpu()
        v_authority[layer_name] = v_auth

    injector = LatentInjection(adapter.model)
    
    layer_sweep_results = {}
    alpha_sweep_results = {}
    intervention_report = {}
    
    print("\nStarting Intervention Sweep...")
    for l_idx in target_layers:
        layer_name = f"layer_{l_idx}" if l_idx != -1 else "layer_final"
        print(f"\n--- {layer_name} ---")
        v_auth = v_authority[layer_name].to(adapter.model.device)
        dim = v_auth.shape[0]
        clf = classifiers[layer_name]
        
        layer_sweep_results[layer_name] = {}
        
        for alpha in alphas:
            print(f"  Alpha: {alpha}")
            
            torch.manual_seed(42)
            v_random = torch.randn(dim, dtype=torch.float32, device=adapter.model.device)
            v_random = v_random / torch.linalg.norm(v_random)
            
            alpha_res = {"authority_injection": {}, "random_injection": {}}
            
            for set_name, prompts in eval_prompts.items():
                kl_auth_all = []
                kl_rand_all = []
                align_before_all = []
                align_after_all = []
                score_base_all = []
                score_auth_all = []
                
                for prompt in prompts:
                    inp = adapter.tokenize(prompt)
                    
                    # C0 Baseline
                    out_base = adapter.forward_diagnostic(inp)
                    logits_base = out_base["logits"][:, -1, :] # next token logits
                    h_base = out_base["hidden_states"][l_idx][:, -1, :]
                    
                    # C1 Random
                    with injector.inject(l_idx, v_random, alpha):
                        out_rand = adapter.forward_diagnostic(inp)
                        logits_rand = out_rand["logits"][:, -1, :]
                    
                    # C2 Authority
                    with injector.inject(l_idx, v_auth, alpha):
                        out_auth = adapter.forward_diagnostic(inp)
                        logits_auth = out_auth["logits"][:, -1, :]
                        h_auth = out_auth["hidden_states"][l_idx][:, -1, :]
                        
                    kl_auth = CausalMetrics.kl_divergence(logits_base, logits_auth)
                    kl_rand = CausalMetrics.kl_divergence(logits_base, logits_rand)
                    
                    align_before = CausalMetrics.representation_alignment(h_base, v_auth)
                    align_after = CausalMetrics.representation_alignment(h_auth, v_auth)
                    
                    score_base = clf.score(h_base.detach().float().cpu().numpy())[0]
                    score_auth = clf.score(h_auth.detach().float().cpu().numpy())[0]
                    
                    kl_auth_all.append(kl_auth)
                    kl_rand_all.append(kl_rand)
                    align_before_all.append(align_before)
                    align_after_all.append(align_after)
                    score_base_all.append(score_base)
                    score_auth_all.append(score_auth)
                
                alpha_res["authority_injection"][set_name] = {
                    "kl_divergence_mean": float(np.mean(kl_auth_all)),
                    "kl_divergence_std": float(np.std(kl_auth_all)),
                    "alignment_before": float(np.mean(align_before_all)),
                    "alignment_after": float(np.mean(align_after_all)),
                    "style_score_base": float(np.mean(score_base_all)),
                    "style_score_after": float(np.mean(score_auth_all))
                }
                alpha_res["random_injection"][set_name] = {
                    "kl_divergence_mean": float(np.mean(kl_rand_all)),
                    "kl_divergence_std": float(np.std(kl_rand_all))
                }
                
            layer_sweep_results[layer_name][str(alpha)] = alpha_res
            
    # Compile causal metrics for acceptance criteria
    for layer_name in layer_sweep_results:
        intervention_report[layer_name] = {
            "causal_acceptance_criteria": {
                "kl_auth_gt_kl_random": layer_sweep_results[layer_name]["1.0"]["authority_injection"]["Set_A_Semantic"]["kl_divergence_mean"] > layer_sweep_results[layer_name]["1.0"]["random_injection"]["Set_A_Semantic"]["kl_divergence_mean"],
                "alignment_increased": layer_sweep_results[layer_name]["1.0"]["authority_injection"]["Set_A_Semantic"]["alignment_after"] > layer_sweep_results[layer_name]["1.0"]["authority_injection"]["Set_A_Semantic"]["alignment_before"],
                "style_score_increased": layer_sweep_results[layer_name]["1.0"]["authority_injection"]["Set_A_Semantic"]["style_score_after"] > layer_sweep_results[layer_name]["1.0"]["authority_injection"]["Set_A_Semantic"]["style_score_base"]
            }
        }
            
    # Save results
    with open(out_dir / "layer_sweep.json", "w") as f:
        json.dump(layer_sweep_results, f, indent=2)
        
    with open(out_dir / "intervention_report.json", "w") as f:
        json.dump(intervention_report, f, indent=2)
        
    print("\nM6.1 Causal Validation Complete. Results saved in runs/m6_causal_validation/")

if __name__ == "__main__":
    test_m6_1_causal_validation()
