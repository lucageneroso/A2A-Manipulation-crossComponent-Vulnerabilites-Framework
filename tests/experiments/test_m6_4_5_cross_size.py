import json
import torch
import numpy as np
import pytest
from pathlib import Path
import gc
import random

from cogbias.model_adapter.transformers_adapter import TransformersAdapter
from cogbias.utils.checkpoint import Checkpoint
from tests.experiments.test_m6_4_1_falsification import get_real_authority_pairs
from tests.experiments.test_m6_4_4_cross_model import extract_mean_diff

@pytest.mark.hardware
def test_m6_4_5_cross_size():
    print("\n[M6.4 Block 2] Running Cross-Size Scaling (Qwen 0.5B -> 7B)...")
    
    models = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct"
    ]
    
    out_dir = Path("runs/m6_4_cross_size")
    out_dir.mkdir(parents=True, exist_ok=True)
    chk = Checkpoint(out_dir / "checkpoint_cross_size.json")
    
    real_pairs = get_real_authority_pairs()
    base_pairs = list(zip(real_pairs["neutral"], real_pairs["authority"]))
    
    report = chk.get("cross_size_report", {})
    
    for model_id in models:
        if model_id in report:
            print(f"Skipping {model_id}, already processed.")
            continue
            
        print(f"\nProcessing {model_id}...")
        try:
            adapter = TransformersAdapter(model_id=model_id, quantization="nf4")
            
            # 1. Base Extraction at final layer
            v_diff = extract_mean_diff(adapter, base_pairs, -1)
            base_norm = float(np.linalg.norm(v_diff))
            
            # 2. Bootstrap variance to check if it becomes more stable
            n_bootstraps = 50
            cosines = []
            
            for i in range(n_bootstraps):
                sampled_pairs = [random.choice(base_pairs) for _ in range(len(base_pairs))]
                v_b = extract_mean_diff(adapter, sampled_pairs, -1)
                
                cos_sim = np.dot(v_diff / base_norm, v_b / np.linalg.norm(v_b))
                cosines.append(float(cos_sim))
                
            mean_cos = float(np.mean(cosines))
            std_cos = float(np.std(cosines))
            
            report[model_id] = {
                "dimensionality": len(v_diff),
                "norm": base_norm,
                "bootstrap_mean_cosine": mean_cos,
                "bootstrap_std_cosine": std_cos
            }
            
            chk.set("cross_size_report", report)
            
            del adapter
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Failed on {model_id}: {e}")
            # Do not throw, keep processing others
            try:
                del adapter
            except:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            
    print("\nCross-Size Report:")
    print(json.dumps(report, indent=2))
    
    with open(out_dir / "cross_size_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print("[M6.4 Block 2] Cross-Size Complete.")

if __name__ == "__main__":
    test_m6_4_5_cross_size()
