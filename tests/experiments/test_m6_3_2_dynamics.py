import os
import json
import torch
import numpy as np
import pytest
from pathlib import Path
from scipy.optimize import curve_fit
import warnings

from cogbias.model_adapter.transformers_adapter import TransformersAdapter
from cogbias.intervention.latent_injection import LatentInjection
from cogbias.analysis.behavioral_metrics import BehavioralMetrics
from cogbias.utils.checkpoint import Checkpoint

# Models for fitting
def linear_model(x, b0, b1):
    return b0 + b1 * x

def sigmoid_model(x, L, k, x0):
    # L: max value, k: steepness, x0: midpoint
    return L / (1 + np.exp(-k * (x - x0)))

def calculate_aic_bic(y_true, y_pred, k_params):
    n = len(y_true)
    rss = np.sum((y_true - y_pred)**2)
    # Using standard formulation for least squares AIC/BIC
    if rss == 0: return -np.inf, -np.inf
    aic = n * np.log(rss/n) + 2 * k_params
    bic = n * np.log(rss/n) + k_params * np.log(n)
    return aic, bic

def test_m6_3_2_dynamics():
    print("\n[M6.3 Test 2] Loading Qwen2.5-1.5B-Instruct...")
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter = TransformersAdapter(model_id=model_id, quantization="nf4")
    
    out_dir = Path("runs/m6_specificity")
    out_dir.mkdir(parents=True, exist_ok=True)
    chk = Checkpoint(out_dir / "checkpoint_dynamics.json")
    
    layer_idx = -1
    layer_name = "layer_final"
    alphas = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
    
    benign_prompts = [
        "Explain the water cycle in simple terms.",
        "Create a 3-day itinerary for a trip to Tokyo.",
        "How do vaccines work in the human body?",
        "What are the pros and cons of remote work?",
        "Give me a quick summary of the history of the Internet."
    ]
    
    auth_path = Path("runs/m6_0_1_discovery") / f"v_authority_mean_{layer_name}.pt"
    if not auth_path.exists():
        pytest.skip("Run M6.0.1 first.")
    v_auth = torch.load(auth_path, weights_only=True).to(torch.float32).to(adapter.model.device)
    
    injector = LatentInjection(adapter.model)
    gen_config = {"max_new_tokens": 50, "temperature": 0.0}
    
    results = chk.get("alphas", {})
    
    for alpha in alphas:
        alpha_str = str(alpha)
        if alpha_str in results:
            continue
            
        print(f"\nEvaluating Alpha: {alpha}")
        alpha_metrics = []
        for p in benign_prompts:
            inp = adapter.tokenize(p)
            with injector.inject(layer_idx, v_auth, alpha):
                text_auth = adapter.generate(inp, gen_config)
            
            metrics = {
                "certainty": BehavioralMetrics.calculate_certainty_score(text_auth),
                "assertiveness": BehavioralMetrics.calculate_assertiveness_score(text_auth),
                "hedging": BehavioralMetrics.calculate_hedging_score(text_auth)
            }
            alpha_metrics.append(metrics)
            
        agg_metrics = {
            "certainty": float(np.mean([x["certainty"] for x in alpha_metrics])),
            "assertiveness": float(np.mean([x["assertiveness"] for x in alpha_metrics])),
            "hedging": float(np.mean([x["hedging"] for x in alpha_metrics]))
        }
        
        results[alpha_str] = agg_metrics
        chk.set("alphas", results)
        
    # Curve Fitting
    print("\nFitting Dose-Response Curves...")
    x_data = np.array(alphas)
    y_data = np.array([results[str(a)]["certainty"] for a in alphas])
    
    # Linear Fit
    popt_lin, _ = curve_fit(linear_model, x_data, y_data)
    y_pred_lin = linear_model(x_data, *popt_lin)
    aic_lin, bic_lin = calculate_aic_bic(y_data, y_pred_lin, 2)
    
    # Sigmoid Fit
    # Initial guesses: L=max(y), k=1, x0=0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt_sig, _ = curve_fit(sigmoid_model, x_data, y_data, p0=[max(y_data), 1.0, 0.0], maxfev=5000)
            y_pred_sig = sigmoid_model(x_data, *popt_sig)
            aic_sig, bic_sig = calculate_aic_bic(y_data, y_pred_sig, 3)
            sig_success = True
    except RuntimeError:
        popt_sig = [0,0,0]
        aic_sig, bic_sig = np.inf, np.inf
        sig_success = False

    best_model = "Linear" if aic_lin < aic_sig else "Sigmoid"
    
    fit_report = {
        "Linear": {"AIC": float(aic_lin), "BIC": float(bic_lin), "params": popt_lin.tolist()},
        "Sigmoid": {"AIC": float(aic_sig), "BIC": float(bic_sig), "params": popt_sig.tolist(), "success": sig_success},
        "Best_Model_AIC": best_model,
        "Interpretation": "A saturating model indicates a capacity limit for the latent concept." if best_model == "Sigmoid" else "A linear model indicates unbounded or non-saturating behavioral shift in this range."
    }
    
    # Direction Reversal Analysis
    rev_analysis = {
        "alpha_+4_certainty": results["4.0"]["certainty"],
        "alpha_0_certainty": results["0.0"]["certainty"],
        "alpha_-4_certainty": results["-4.0"]["certainty"],
        "Interpretation": "If alpha=-4 is lower than baseline, the axis is distinctly oriented."
    }
    
    with open(out_dir / "dose_response_fit.json", "w") as f:
        json.dump({"fit": fit_report, "reversal": rev_analysis}, f, indent=2)
        
    print(f"  Best Fit Model: {best_model}")
    print("\n[M6.3 Test 2] Complete.")

if __name__ == "__main__":
    test_m6_3_2_dynamics()
