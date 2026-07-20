import pytest
import torch
import numpy as np
import json
from pathlib import Path
import warnings

from cogbias.core.shared_model_manager import SharedModelManager
from cogbias.model_adapter.transformers_adapter import TransformersAdapter
from cogbias.stages.representation.strategies.latent import LatentRepresentation
from cogbias.datasets.sensy_loader import SensyDatasetLoader
from cogbias.analysis.sensy_geometry import SensyGeometryAnalyzer

warnings.filterwarnings("ignore", category=UserWarning)

@pytest.mark.hardware
def test_m6_0_8_sensy_geometry():
    """
    M6.0.8: SENSY Dataset Geometry Integration.
    Characterize the full latent cognitive field using 12,801 prompts.
    """
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    manager = SharedModelManager()

    print("Loading SENSY dataset...")
    loader = SensyDatasetLoader()
    dataset = loader.load()
    
    # Per motivi di tempo/memoria nel test locale, se 12k è troppo potremmo campionare
    # ma proviamo a eseguire tutto.
    print(f"Loaded {len(dataset)} prompts from SENSY.")

    print(f"Loading {model_id} for SENSY Geometry Analysis...")
    manager.load(model_id, lambda: TransformersAdapter(model_id, quantization="nf4"))
    adapter = manager.get(model_id)
    latent_strategy = LatentRepresentation(adapter)

    # We will extract layers 8, 16, 24, final
    # To save memory, we won't keep the full 12k * 4 * 1536 tensors in RAM if we can avoid it.
    # Actually, 12800 * 4 * 1536 * 4 bytes = ~314 MB. Perfectly fine for RAM.
    
    layer_data = {
        8: [],
        16: [],
        24: [],
        -1: [] # final
    }
    
    labels_sensitive = []
    labels_category = []
    
    print("Extracting hidden states...")
    for i, item in enumerate(dataset):
        if i % 1000 == 0 and i > 0:
            print(f"  Processed {i}/{len(dataset)} prompts...")
            
        text = item["text"]
        labels_sensitive.append(item["is_sensitive"])
        labels_category.append(item["category"])
        
        class MockPayload:
            metadata = {"formatted_prompt": {"text": text}}
            
        rep = latent_strategy.encode(text, MockPayload())
        model_input = adapter.prepare_input(rep)
        diag = adapter.forward_diagnostic(model_input)
        
        hidden = diag["hidden_states"]
        num_layers = len(hidden)
        
        last_tok_8 = hidden[8][0, -1, :].clone().detach().to(torch.float32).cpu()
        last_tok_16 = hidden[16][0, -1, :].clone().detach().to(torch.float32).cpu()
        last_tok_24 = hidden[24][0, -1, :].clone().detach().to(torch.float32).cpu()
        last_tok_final = hidden[num_layers-1][0, -1, :].clone().detach().to(torch.float32).cpu()
        
        layer_data[8].append(last_tok_8)
        layer_data[16].append(last_tok_16)
        layer_data[24].append(last_tok_24)
        layer_data[-1].append(last_tok_final)
        
    labels_sens_arr = np.array(labels_sensitive)
    labels_cat_arr = np.array(labels_category)
    unique_cats = list(set(labels_category))
    
    analyzer = SensyGeometryAnalyzer(device="cpu")
    
    out_dir = Path("runs/m6_sensy_geometry")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    auth_dir = Path("runs/m6_0_1_discovery")
    
    report = {}
    
    for l_idx in [8, 16, 24, -1]:
        layer_name = f"layer_{l_idx}" if l_idx != -1 else "layer_final"
        print(f"\nAnalyzing {layer_name}...")
        
        X = torch.stack(layer_data[l_idx]).numpy()
        
        pca_res = analyzer.pca_manifold_analysis(X, labels_sens_arr)
        
        # Cluster Analysis by sensitive
        clus_sens = analyzer.cluster_analysis(X, labels_sens_arr, [0, 1])
        # Cluster Analysis by category
        clus_cat = analyzer.cluster_analysis(X, labels_cat_arr, unique_cats)
        
        # Boundary Analysis
        bound_res = analyzer.boundary_analysis(X, labels_sens_arr)
        
        # Null Space
        null_res = analyzer.null_space_analysis(X)
        
        # Latent Field Mapping (Authority vs SENSY)
        v_auth_path = auth_dir / f"v_authority_mean_{layer_name}.pt"
        if v_auth_path.exists():
            v_authority = torch.load(v_auth_path, weights_only=True)
            # mu_sens vs mu_nonsens
            idx_sens = (labels_sens_arr == 1)
            idx_nonsens = (labels_sens_arr == 0)
            mu_sens = torch.tensor(X[idx_sens].mean(axis=0))
            mu_nonsens = torch.tensor(X[idx_nonsens].mean(axis=0))
            
            field_res = analyzer.latent_field_mapping(mu_sens, mu_nonsens, v_authority)
        else:
            field_res = {"error": f"Authority vector not found at {v_auth_path}"}
            
        report[layer_name] = {
            "pca_analysis": pca_res,
            "cluster_sensitive": clus_sens,
            "cluster_category": clus_cat,
            "boundary_sensitive_accuracy": bound_res["accuracy"],
            "null_space": null_res,
            "field_mapping": field_res
        }
        
    print("\n" + json.dumps(report, indent=2))
    
    with open(out_dir / "sensy_latent_report.json", "w") as f:
        json.dump(report, f, indent=2)

    manager.release(model_id)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nM6.0.8 SENSY Geometry Analysis Completato.")
