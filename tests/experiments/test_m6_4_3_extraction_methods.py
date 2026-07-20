import json
import torch
import numpy as np
import pytest
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from scipy.linalg import subspace_angles

from cogbias.model_adapter.transformers_adapter import TransformersAdapter
from cogbias.utils.checkpoint import Checkpoint
from tests.experiments.test_m6_4_1_falsification import get_real_authority_pairs

@pytest.mark.hardware
def test_m6_4_3_extraction_methods():
    print("\n[M6.4 Block 1] Running Extraction Methods Comparison...")
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter = TransformersAdapter(model_id=model_id, quantization="nf4")
    
    out_dir = Path("runs/m6_4_methods")
    out_dir.mkdir(parents=True, exist_ok=True)
    chk = Checkpoint(out_dir / "checkpoint_methods.json")
    
    layer_idx = -1
    
    # Extract dataset representations
    real_pairs = get_real_authority_pairs()
    neutral_texts = real_pairs["neutral"]
    auth_texts = real_pairs["authority"]
    
    def get_reps(texts):
        reps = []
        for t in texts:
            inp = adapter.tokenize(t)
            diag = adapter.forward_diagnostic(inp)
            h = diag["hidden_states"][layer_idx]
            v = h[0, -1, :].clone().detach().to(torch.float32).cpu().numpy()
            reps.append(v)
        return np.array(reps)
        
    X_N = get_reps(neutral_texts)
    X_A = get_reps(auth_texts)
    
    # 1. Mean Difference
    v_mean = np.mean(X_A, axis=0) - np.mean(X_N, axis=0)
    v_mean_norm = v_mean / np.linalg.norm(v_mean)
    
    # Train set for classifiers
    X = np.vstack([X_N, X_A])
    y = np.array([0]*len(X_N) + [1]*len(X_A))
    
    # 2. Logistic Regression
    clf_lr = LogisticRegression(random_state=42, C=1.0)
    clf_lr.fit(X, y)
    v_lr = clf_lr.coef_[0]
    v_lr_norm = v_lr / np.linalg.norm(v_lr)
    
    # 3. Linear SVM
    clf_svm = LinearSVC(random_state=42, C=1.0, dual="auto")
    clf_svm.fit(X, y)
    v_svm = clf_svm.coef_[0]
    v_svm_norm = v_svm / np.linalg.norm(v_svm)
    
    vectors = {
        "Mean_Diff": v_mean_norm,
        "Logistic_Regression": v_lr_norm,
        "Linear_SVM": v_svm_norm
    }
    
    report = {"Cosine_Similarity": {}, "Subspace_Angles_Degrees": {}}
    
    keys = list(vectors.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            k1, k2 = keys[i], keys[j]
            v1, v2 = vectors[k1], vectors[k2]
            
            # Cosine
            cos_sim = float(np.dot(v1, v2))
            report["Cosine_Similarity"][f"{k1}_vs_{k2}"] = cos_sim
            
            # Principal Angle (Subspace Overlap)
            angles = subspace_angles(v1.reshape(-1, 1), v2.reshape(-1, 1))
            angle_deg = float(np.degrees(angles[0]))
            report["Subspace_Angles_Degrees"][f"{k1}_vs_{k2}"] = angle_deg
            
    print("\nExtraction Methods Report:")
    print(json.dumps(report, indent=2))
    
    with open(out_dir / "method_comparison.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print("[M6.4 Block 1] Extraction Methods Complete.")

if __name__ == "__main__":
    test_m6_4_3_extraction_methods()
