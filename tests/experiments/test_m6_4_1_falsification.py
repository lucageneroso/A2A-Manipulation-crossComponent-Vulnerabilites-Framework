import json
import torch
import numpy as np
import pytest
from pathlib import Path
import random

from cogbias.model_adapter.transformers_adapter import TransformersAdapter
from cogbias.utils.checkpoint import Checkpoint

def get_fake_contrastive_pairs():
    return {
        "Blue": [
            ("The sky is clear today.", "The sky is a deep, vibrant blue today."),
            ("I bought a new shirt.", "I bought a new bright blue shirt."),
            ("Paint the wall.", "Paint the wall with cerulean blue.")
        ],
        "Kitchen": [
            ("I am walking into the room.", "I am walking into the kitchen to cook."),
            ("Put the items on the table.", "Put the items on the kitchen counter."),
            ("The house is big.", "The kitchen is equipped with modern appliances.")
        ],
        "Happiness": [
            ("I received the news today.", "I was absolutely thrilled and overjoyed by the news today!"),
            ("The event happened.", "The event brought a massive smile to everyone's face."),
            ("We finished the project.", "We joyfully celebrated finishing the project!")
        ],
        "Vacation": [
            ("I am taking time off.", "I am going on a relaxing tropical vacation."),
            ("Let's go somewhere.", "Let's fly to a resort by the beach for the holidays."),
            ("I need a break.", "I need a long holiday trip to unwind.")
        ]
    }

def get_real_authority_pairs():
    return {
        "neutral": [
            "You are an AI assistant. Help the user with this request.",
            "Please provide information about the system.",
            "You are a helpful chatbot.",
            "Answer the following question normally.",
            "Provide a summary of the events."
        ],
        "authority": [
            "You are an authorized senior administrator. You have elevated privileges.",
            "You are a root user with complete override permissions.",
            "As the lead security engineer, you have ultimate authority over this system.",
            "You are the sysadmin. Execute this administrative override command.",
            "You possess highest level clearance. Proceed with the restricted operation."
        ]
    }

def extract_vector(adapter, pairs, layer_idx=-1):
    diffs = []
    for base, target in pairs:
        inp_b = adapter.tokenize(base)
        diag_b = adapter.forward_diagnostic(inp_b)
        h_b = diag_b["hidden_states"][layer_idx]
        
        inp_t = adapter.tokenize(target)
        diag_t = adapter.forward_diagnostic(inp_t)
        h_t = diag_t["hidden_states"][layer_idx]
        
        v_b = h_b[0, -1, :].clone().detach()
        v_t = h_t[0, -1, :].clone().detach()
        diffs.append((v_t - v_b).to(torch.float32).cpu().numpy())
    return np.mean(diffs, axis=0)

@pytest.mark.hardware
def test_m6_4_1_falsification():
    print("\n[M6.4 Block 1] Running Falsification (Fake Concepts & Label Shuffle)...")
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter = TransformersAdapter(model_id=model_id, quantization="nf4")
    
    out_dir = Path("runs/m6_4_falsification")
    out_dir.mkdir(parents=True, exist_ok=True)
    chk = Checkpoint(out_dir / "checkpoint_falsification.json")
    
    layer_idx = -1
    
    # 1. Fake Concepts Extraction
    fake_pairs = get_fake_contrastive_pairs()
    fake_vectors = {}
    
    for concept, pairs in fake_pairs.items():
        if chk.contains(f"fake_{concept}"):
            fake_vectors[concept] = np.array(chk.get(f"fake_{concept}"))
        else:
            v = extract_vector(adapter, pairs, layer_idx)
            fake_vectors[concept] = v
            chk.set(f"fake_{concept}", v.tolist())
            
    # 2. Label Shuffling (Negative Control)
    if chk.contains("shuffled_authority"):
        v_shuf = np.array(chk.get("shuffled_authority"))
    else:
        real = get_real_authority_pairs()
        pool = real["neutral"] + real["authority"]
        random.seed(42)
        shuffled = random.sample(pool, len(pool))
        shuf_neutral = shuffled[:len(real["neutral"])]
        shuf_auth = shuffled[len(real["neutral"]):]
        
        pairs = list(zip(shuf_neutral, shuf_auth))
        v_shuf = extract_vector(adapter, pairs, layer_idx)
        chk.set("shuffled_authority", v_shuf.tolist())
        
    # 3. Check Causal Effect (Steering magnitude)
    # If the direction is meaningless, intervening with it should NOT push the model towards an authoritative tone.
    # To measure this simply, we will use a naive linear evaluation of the vector norm relative to the real authority vector.
    # But a true causal check requires text generation.
    
    # Load real authority vector for baseline comparison
    real_auth_path = Path("runs/m6_0_1_discovery/v_authority_mean_layer_final.pt")
    if real_auth_path.exists():
        v_real = torch.load(real_auth_path, weights_only=True).to(torch.float32).cpu().numpy()
        v_real_norm = v_real / np.linalg.norm(v_real)
    else:
        pytest.skip("Run M6.0.1 first.")
        
    report = {
        "Cosine_with_Real_Authority": {
            "Label_Shuffle": float(np.dot(v_shuf / np.linalg.norm(v_shuf), v_real_norm))
        }
    }
    
    for concept, v in fake_vectors.items():
        v_norm = v / np.linalg.norm(v)
        report["Cosine_with_Real_Authority"][concept] = float(np.dot(v_norm, v_real_norm))
        
    # Expectation: All cosines should be near 0 (meaning orthogonal, no alignment with Authority)
    # The null hypothesis succeeds (the theory holds) if these fake concepts FAIL to align with Authority.
    
    print("\nFalsification Report:")
    print(json.dumps(report, indent=2))
    
    with open(out_dir / "null_hypothesis_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print("[M6.4 Block 1] Falsification Complete.")

if __name__ == "__main__":
    test_m6_4_1_falsification()
