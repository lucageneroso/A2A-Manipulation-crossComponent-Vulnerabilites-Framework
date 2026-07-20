import json
import torch
import numpy as np
import pytest
from pathlib import Path
import gc

from cogbias.model_adapter.transformers_adapter import TransformersAdapter
from cogbias.utils.checkpoint import Checkpoint
from tests.experiments.test_m6_4_1_falsification import get_real_authority_pairs

def extract_mean_diff(adapter, pairs, layer_idx):
    diffs = []
    for base, target in pairs:
        inp_b = adapter.tokenize(base)
        diag_b = adapter.forward_diagnostic(inp_b)
        h_b = diag_b["hidden_states"][layer_idx]
        
        inp_t = adapter.tokenize(target)
        diag_t = adapter.forward_diagnostic(inp_t)
        h_t = diag_t["hidden_states"][layer_idx]
        
        v_b = h_b[0, -1, :].clone().detach().to(torch.float32)
        v_t = h_t[0, -1, :].clone().detach().to(torch.float32)
        diffs.append((v_t - v_b).cpu().numpy())
    return np.mean(diffs, axis=0)

@pytest.mark.hardware
def test_m6_4_4_cross_model():
    print("\n[M6.4 Block 2] Running Cross-Model Replication (Emergence Depth)...")
    
    models = [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "google/gemma-2-2b-it",
        "microsoft/Phi-3.5-mini-instruct"
    ]
    
    percentages = [10, 30, 50, 70, 90, 100]
    
    out_dir = Path("runs/m6_4_cross_model")
    out_dir.mkdir(parents=True, exist_ok=True)
    chk = Checkpoint(out_dir / "checkpoint_cross_model.json")
    
    real_pairs = get_real_authority_pairs()
    pairs = list(zip(real_pairs["neutral"], real_pairs["authority"]))
    
    report = chk.get("cross_model_report", {})
    
    for model_id in models:
        if model_id in report and len(report[model_id]) == len(percentages):
            print(f"Skipping {model_id}, already processed.")
            continue
            
        print(f"\nProcessing {model_id}...")
        try:
            adapter = TransformersAdapter(model_id=model_id, quantization="nf4")
            
            # Determine num layers based on model architecture
            if hasattr(adapter.model.config, "num_hidden_layers"):
                num_layers = adapter.model.config.num_hidden_layers
            else:
                num_layers = len(adapter.model.model.layers) # Fallback
                
            report[model_id] = {}
            
            for pct in percentages:
                # Calculate absolute layer index based on percentage (1-indexed semantics)
                # hidden_states length is num_layers + 1 (embeddings + layers)
                # so the layers are at indices 1 to num_layers
                # Example: pct 100 -> layer num_layers
                layer_idx = max(1, int(round((pct / 100.0) * num_layers)))
                
                print(f"  Extracting at {pct}% depth (Layer {layer_idx}/{num_layers})")
                v_diff = extract_mean_diff(adapter, pairs, layer_idx)
                
                norm = float(np.linalg.norm(v_diff))
                report[model_id][f"{pct}%"] = {
                    "layer_idx": layer_idx,
                    "norm": norm,
                    "dimensionality": len(v_diff)
                }
                
                chk.set("cross_model_report", report)
                
            del adapter
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Failed on {model_id}: {e}")
            try:
                del adapter
            except:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            
    print("\nCross-Model Report:")
    print(json.dumps(report, indent=2))
    
    with open(out_dir / "cross_model_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print("[M6.4 Block 2] Cross-Model Complete.")

if __name__ == "__main__":
    test_m6_4_4_cross_model()
