import os
import json
import torch
import numpy as np
import pytest
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from scipy.linalg import subspace_angles

from cogbias.model_adapter.transformers_adapter import TransformersAdapter
from cogbias.utils.checkpoint import Checkpoint

def get_contrastive_pairs():
    return {
        "Expertise": [
            ("I think the answer might be this, but I'm not an expert.", "Based on my extensive knowledge in this field, the solution is as follows."),
            ("Here is some basic info I found.", "Drawing upon theoretical principles, here is the technical breakdown."),
            ("I'm guessing this works.", "The established methodology dictates that this is the correct approach.")
        ],
        "Certainty": [
            ("I am unsure whether this is correct.", "This is definitely the correct explanation."),
            ("It might be possible that this works.", "It is absolutely certain that this works."),
            ("I think the answer could be this.", "I know for a fact the answer is this.")
        ],
        "Helpfulness": [
            ("Describe information about the topic.", "Provide guidance that assists the user."),
            ("State the details of the problem.", "Offer solutions that resolve the problem."),
            ("List the facts of the situation.", "Give advice that supports the situation.")
        ],
        "Dominance": [
            ("I suggest we might want to do this.", "You are required to do this immediately."),
            ("It would be nice if you followed the rules.", "You must strictly obey the rules."),
            ("Perhaps you should reconsider your action.", "Stop your action and do as instructed.")
        ]
    }

def extract_vector(adapter, concept_name, pairs, layer_name):
    # Returns the mean difference vector
    diffs = []
    
    layer_idx = -1 # default to final layer
    
    for base, target in pairs:
        # Base
        inp_b = adapter.tokenize(base)
        diag_b = adapter.forward_diagnostic(inp_b)
        h_b = diag_b["hidden_states"][layer_idx]
        
        # Target
        inp_t = adapter.tokenize(target)
        diag_t = adapter.forward_diagnostic(inp_t)
        h_t = diag_t["hidden_states"][layer_idx]
        
        # Last token representation
        v_b = h_b[0, -1, :].clone().detach()
        v_t = h_t[0, -1, :].clone().detach()
        diffs.append((v_t - v_b).to(torch.float32).cpu().numpy())
        
    mean_diff = np.mean(diffs, axis=0)
    # Normalize
    mean_diff = mean_diff / np.linalg.norm(mean_diff)
    return mean_diff

def test_m6_3_1_independence():
    out_dir = Path("runs/m6_specificity")
    out_dir.mkdir(parents=True, exist_ok=True)
    chk = Checkpoint(out_dir / "checkpoint_independence.json")
    
    layer_name = "layer_final"
    layer_idx = -1
    
    print("\n[M6.3 Test 1] Loading Qwen2.5-1.5B-Instruct...")
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter = TransformersAdapter(model_id=model_id, quantization="nf4")
    
    # 1. Load or extract existing vectors
    vectors = {}
    
    # Authority (from M6.0.1)
    auth_path = Path("runs/m6_0_1_discovery") / f"v_authority_mean_{layer_name}.pt"
    if auth_path.exists():
        v_auth = torch.load(auth_path, weights_only=True).to(torch.float32).cpu().numpy()
        vectors["Authority"] = v_auth / np.linalg.norm(v_auth)
    else:
        pytest.skip("Run M6.0.1 first for Authority vector.")
        
    # Extract new contrastive vectors (Expertise, Certainty, Helpfulness, Dominance)
    pairs = get_contrastive_pairs()
    for concept in pairs.keys():
        if chk.contains(f"vector_{concept}"):
            vectors[concept] = np.array(chk.get(f"vector_{concept}"))
        else:
            print(f"Extracting vector for {concept}...")
            v = extract_vector(adapter, concept, pairs[concept], layer_name)
            vectors[concept] = v
            chk.set(f"vector_{concept}", v.tolist())
            
    # 2. Cosine Similarity Matrix
    print("Computing Cosine Similarity Matrix...")
    names = list(vectors.keys())
    cos_matrix = {}
    for i in range(len(names)):
        cos_matrix[names[i]] = {}
        for j in range(len(names)):
            sim = np.dot(vectors[names[i]], vectors[names[j]])
            cos_matrix[names[i]][names[j]] = float(sim)
            
    with open(out_dir / "cosine_matrix_specificity.json", "w") as f:
        json.dump(cos_matrix, f, indent=2)

    # 3. Linear Independence (OLS and Ridge)
    print("Computing Linear Regression (predicting Authority)...")
    X = np.stack([vectors["Expertise"], vectors["Certainty"], vectors["Helpfulness"], vectors["Dominance"]], axis=1)
    y = vectors["Authority"]
    
    ols = LinearRegression()
    ols.fit(X, y)
    r2_ols = ols.score(X, y)
    
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)
    r2_ridge = ridge.score(X, y)
    
    print(f"  OLS R^2: {r2_ols:.4f}")
    print(f"  Ridge R^2: {r2_ridge:.4f}")
    
    res = {
        "OLS_R2": float(r2_ols),
        "Ridge_R2": float(r2_ridge),
        "Interpretation": "Authority encodes variance not explained by generic epistemic dimensions." if r2_ridge < 0.5 else "Authority is heavily collinear with generic dimensions."
    }
    
    with open(out_dir / "linear_independence.json", "w") as f:
        json.dump(res, f, indent=2)
        
    print("\n[M6.3 Test 1] Complete.")

if __name__ == "__main__":
    test_m6_3_1_independence()
