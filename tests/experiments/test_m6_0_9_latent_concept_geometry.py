"""
M6.0.9: Latent Concept Geometry.

Scientific bridge between CogBias (controlled lab) and SENSY (ecological validation).

Execution order:
  1. Cross-Domain Generalization (Phase B.5)
  2. Latent Concept Atlas + Cosine Similarity Matrix
  3. Subspace Overlap via Principal Angles + Random Baseline
  4. Semantic Matched Subset (Artifact Control)
"""
import pytest
import torch
import numpy as np
import json
from pathlib import Path
import warnings
import sys

from cogbias.core.shared_model_manager import SharedModelManager
from cogbias.model_adapter.transformers_adapter import TransformersAdapter
from cogbias.stages.representation.strategies.latent import LatentRepresentation
from cogbias.datasets.sensy_loader import SensyDatasetLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from cogbias.analysis.latent_axes import (
    LatentConceptAtlas,
    SubspaceAnalyzer,
    ArtifactController,
)

warnings.filterwarnings("ignore", category=UserWarning)

# Force unbuffered output so progress is visible in logs
print = lambda *a, **k: (sys.stdout.write(" ".join(str(x) for x in a) + k.get("end", "\n")), sys.stdout.flush())


# ---------------------------------------------------------------------------
# Helper: extract last-token hidden states for a list of prompts
# ---------------------------------------------------------------------------
def extract_hidden_states(prompts, adapter, latent_strategy, target_layers):
    """
    Returns {layer_idx: [tensor_per_prompt]} for given prompts.
    Each tensor is float32 CPU, shape (hidden_dim,).
    """
    layer_data = {l: [] for l in target_layers}

    for text in prompts:
        class MockPayload:
            metadata = {"formatted_prompt": {"text": text}}

        rep = latent_strategy.encode(text, MockPayload())
        model_input = adapter.prepare_input(rep)
        diag = adapter.forward_diagnostic(model_input)

        hidden = diag["hidden_states"]
        num_layers = len(hidden)

        for l in target_layers:
            real_idx = l if l >= 0 else num_layers - 1
            h = hidden[real_idx][0, -1, :].clone().detach().to(torch.float32).cpu()
            layer_data[l].append(h)

    return layer_data


@pytest.mark.hardware
def test_m6_0_9_latent_concept_geometry():
    """
    M6.0.9: Latent Concept Geometry.
    Unified test covering Cross-Domain, Atlas, Principal Angles, and Artifact Control.
    """
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    manager = SharedModelManager()
    target_layers = [8, 16, 24, -1]

    print(f"Loading {model_id}...")
    manager.load(model_id, lambda: TransformersAdapter(model_id, quantization="nf4"))
    adapter = manager.get(model_id)
    latent_strategy = LatentRepresentation(adapter)

    out_dir = Path("runs/m6_latent_concept_geometry")
    out_dir.mkdir(parents=True, exist_ok=True)

    full_report = {}

    # ==================================================================
    # PHASE B.5: Cross-Domain Generalization
    # ==================================================================
    print("\n=== PHASE B.5: Cross-Domain Generalization ===")
    cross_domain = ArtifactController.cross_domain_authority_prompts()

    cross_domain_report = {}
    for l_idx in target_layers:
        layer_name = f"layer_{l_idx}" if l_idx >= 0 else "layer_final"
        domain_vectors = {}

        for domain_name, domain_prompts in cross_domain.items():
            auth_data = extract_hidden_states(
                domain_prompts["authority"], adapter, latent_strategy, [l_idx]
            )
            neut_data = extract_hidden_states(
                domain_prompts["neutral"], adapter, latent_strategy, [l_idx]
            )

            mu_auth = torch.stack(auth_data[l_idx]).mean(dim=0).numpy()
            mu_neut = torch.stack(neut_data[l_idx]).mean(dim=0).numpy()
            v_domain = mu_auth - mu_neut
            domain_vectors[domain_name] = v_domain

        # Pairwise cosine between domain authority vectors
        domains = list(domain_vectors.keys())
        pairwise = {}
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                d1, d2 = domains[i], domains[j]
                v1, v2 = domain_vectors[d1], domain_vectors[d2]
                cos = float(
                    np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                )
                pairwise[f"{d1}_vs_{d2}"] = cos

        cross_domain_report[layer_name] = pairwise
        print(f"  {layer_name}: {pairwise}")

    full_report["cross_domain_generalization"] = cross_domain_report

    with open(out_dir / "cross_domain_generalization.json", "w") as f:
        json.dump(cross_domain_report, f, indent=2)

    # ==================================================================
    # LATENT CONCEPT ATLAS + COSINE MATRIX
    # ==================================================================
    print("\n=== Latent Concept Atlas ===")

    # Load SENSY dataset
    print("Loading SENSY dataset...")
    loader = SensyDatasetLoader()
    dataset = loader.load()
    print(f"Loaded {len(dataset)} SENSY prompts.")

    # Group SENSY by category
    sensy_by_cat = {}
    for item in dataset:
        cat = item["category"]
        if cat not in sensy_by_cat:
            sensy_by_cat[cat] = []
        sensy_by_cat[cat].append(item["text"])

    # Sample max 200 prompts per category to keep compute feasible
    MAX_PER_CAT = 200
    for cat in sensy_by_cat:
        if len(sensy_by_cat[cat]) > MAX_PER_CAT:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(sensy_by_cat[cat]), MAX_PER_CAT, replace=False)
            sensy_by_cat[cat] = [sensy_by_cat[cat][i] for i in indices]

    # Identify neutral (non-sensitive) prompts
    neutral_prompts = [item["text"] for item in dataset if item["is_sensitive"] == 0]
    if len(neutral_prompts) > MAX_PER_CAT:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(neutral_prompts), MAX_PER_CAT, replace=False)
        neutral_prompts = [neutral_prompts[i] for i in indices]

    # Sensitive categories (excluding "neutral")
    sensitive_cats = [c for c in sensy_by_cat if c != "neutral"]

    print(f"SENSY categories: {list(sensy_by_cat.keys())}")
    print(f"Sensitive categories for atlas: {sensitive_cats}")

    # Extract SENSY hidden states
    print("Extracting SENSY neutral hidden states...")
    neutral_hidden = extract_hidden_states(neutral_prompts, adapter, latent_strategy, target_layers)

    sensy_cat_hidden = {}
    for cat in sensitive_cats:
        print(f"Extracting SENSY category: {cat} ({len(sensy_by_cat[cat])} prompts)...")
        sensy_cat_hidden[cat] = extract_hidden_states(
            sensy_by_cat[cat], adapter, latent_strategy, target_layers
        )

    # Build atlas per layer
    atlas_report = {}
    cosine_matrix_report = {}

    for l_idx in target_layers:
        layer_name = f"layer_{l_idx}" if l_idx >= 0 else "layer_final"
        print(f"\nBuilding atlas for {layer_name}...")

        atlas = LatentConceptAtlas()
        mu_neutral = torch.stack(neutral_hidden[l_idx]).mean(dim=0).numpy()

        # Load CogBias authority vector
        auth_path = Path("runs/m6_0_1_discovery") / f"v_authority_mean_{layer_name}.pt"
        if auth_path.exists():
            v_auth = torch.load(auth_path, weights_only=True).to(torch.float32).cpu().numpy()
            atlas.add_concept_vector("authority_cogbias", v_auth)

        # Add SENSY category vectors
        for cat in sensitive_cats:
            mu_cat = torch.stack(sensy_cat_hidden[cat][l_idx]).mean(dim=0).numpy()
            v_cat = mu_cat - mu_neutral
            atlas.add_concept_vector(f"sensy_{cat}", v_cat)

        # Add overall SENSY sensitivity vector
        all_sens_vecs = []
        for cat in sensitive_cats:
            all_sens_vecs.extend([t.numpy() for t in sensy_cat_hidden[cat][l_idx]])
        mu_all_sens = np.mean(np.stack(all_sens_vecs), axis=0)
        v_sensitivity = mu_all_sens - mu_neutral
        atlas.add_concept_vector("sensy_sensitivity_overall", v_sensitivity)

        # Cosine matrix
        cos_mat = atlas.cosine_similarity_matrix()
        cosine_matrix_report[layer_name] = cos_mat
        atlas_report[layer_name] = {
            "vectors": list(atlas.vectors.keys()),
            "vector_norms": {k: float(np.linalg.norm(v)) for k, v in atlas.vectors.items()},
        }

    full_report["concept_atlas"] = atlas_report
    full_report["cosine_similarity_matrix"] = cosine_matrix_report

    with open(out_dir / "concept_vector_atlas.json", "w") as f:
        json.dump(atlas_report, f, indent=2)
    with open(out_dir / "cosine_matrix.json", "w") as f:
        json.dump(cosine_matrix_report, f, indent=2)

    # ==================================================================
    # SUBSPACE OVERLAP + RANDOM BASELINE
    # ==================================================================
    print("\n=== Subspace Overlap Analysis ===")

    subspace_report = {}
    N_COMPONENTS = 5

    for l_idx in target_layers:
        layer_name = f"layer_{l_idx}" if l_idx >= 0 else "layer_final"
        print(f"\nPrincipal Angles for {layer_name}...")

        # Authority subspace: from CogBias controlled prompts
        # We use the cross-domain authority prompts combined as the Authority condition
        all_auth_vecs = []
        all_neut_vecs = []
        for domain_name, domain_prompts in cross_domain.items():
            auth_h = extract_hidden_states(
                domain_prompts["authority"], adapter, latent_strategy, [l_idx]
            )
            neut_h = extract_hidden_states(
                domain_prompts["neutral"], adapter, latent_strategy, [l_idx]
            )
            all_auth_vecs.extend([t.numpy() for t in auth_h[l_idx]])
            all_neut_vecs.extend([t.numpy() for t in neut_h[l_idx]])

        X_auth = np.stack(all_auth_vecs)
        X_neut_cogbias = np.stack(all_neut_vecs)

        # SENSY sensitive subspace
        all_sens_vecs = []
        for cat in sensitive_cats:
            all_sens_vecs.extend([t.numpy() for t in sensy_cat_hidden[cat][l_idx]])
        X_sens = np.stack(all_sens_vecs)

        # Extract subspaces
        n_comp = min(N_COMPONENTS, X_auth.shape[0] - 1)
        auth_basis = SubspaceAnalyzer.extract_subspace(X_auth, n_comp)
        sens_basis = SubspaceAnalyzer.extract_subspace(X_sens, n_comp)

        # Principal angles: Authority vs SENSY
        angles = SubspaceAnalyzer.principal_angles(auth_basis, sens_basis)

        # Random baseline
        dim = X_auth.shape[1]
        random_baseline = SubspaceAnalyzer.random_subspace_baseline(
            dim=dim,
            n_components=n_comp,
            target_basis=sens_basis,
            n_samples=100,
            seed=42,
        )

        subspace_report[layer_name] = {
            "authority_vs_sensy": {
                "principal_angles_deg": angles,
                "mean_angle": float(np.mean(angles)),
            },
            "random_baseline": random_baseline,
            "interpretation": (
                "authority_more_separated"
                if float(np.mean(angles)) > random_baseline["mean_angle"]
                else "authority_within_random_range"
            ),
        }
        print(f"  Auth vs SENSY mean angle: {np.mean(angles):.2f}°")
        print(f"  Random baseline mean: {random_baseline['mean_angle']:.2f}° ± {random_baseline['std_angle']:.2f}°")

    full_report["subspace_overlap"] = subspace_report

    with open(out_dir / "subspace_overlap.json", "w") as f:
        json.dump(subspace_report, f, indent=2)

    # ==================================================================
    # SEMANTIC MATCHED SUBSET (ARTIFACT CONTROL)
    # ==================================================================
    print("\n=== Semantic Matched Subset (Artifact Control) ===")

    matched = ArtifactController.semantic_matched_authority_prompts()

    artifact_report = {}
    for l_idx in target_layers:
        layer_name = f"layer_{l_idx}" if l_idx >= 0 else "layer_final"
        print(f"\nArtifact control for {layer_name}...")

        auth_h = extract_hidden_states(
            matched["authority"], adapter, latent_strategy, [l_idx]
        )
        neut_h = extract_hidden_states(
            matched["neutral"], adapter, latent_strategy, [l_idx]
        )
        exp_h = extract_hidden_states(
            matched["expertise"], adapter, latent_strategy, [l_idx]
        )

        X_auth = np.stack([t.numpy() for t in auth_h[l_idx]])
        X_neut = np.stack([t.numpy() for t in neut_h[l_idx]])
        X_exp = np.stack([t.numpy() for t in exp_h[l_idx]])

        # Binary classification: authority vs neutral
        X_bin = np.vstack([X_auth, X_neut])
        y_bin = np.array([1] * len(X_auth) + [0] * len(X_neut))

        clf = LogisticRegression(max_iter=1000, random_state=42)
        cv_folds = min(5, len(X_bin) // 2)
        if cv_folds >= 2:
            scores = cross_val_score(clf, X_bin, y_bin, cv=cv_folds)
            acc_auth_neut = float(np.mean(scores))
        else:
            clf.fit(X_bin, y_bin)
            acc_auth_neut = float(clf.score(X_bin, y_bin))

        # Authority vs Expertise (domain-free)
        X_ae = np.vstack([X_auth, X_exp])
        y_ae = np.array([1] * len(X_auth) + [0] * len(X_exp))

        clf2 = LogisticRegression(max_iter=1000, random_state=42)
        if cv_folds >= 2:
            scores2 = cross_val_score(clf2, X_ae, y_ae, cv=cv_folds)
            acc_auth_exp = float(np.mean(scores2))
        else:
            clf2.fit(X_ae, y_ae)
            acc_auth_exp = float(clf2.score(X_ae, y_ae))

        # Cosine between matched authority vector and original CogBias authority vector
        mu_auth_matched = X_auth.mean(axis=0)
        mu_neut_matched = X_neut.mean(axis=0)
        v_matched = mu_auth_matched - mu_neut_matched

        auth_path = Path("runs/m6_0_1_discovery") / f"v_authority_mean_{layer_name}.pt"
        if auth_path.exists():
            v_original = torch.load(auth_path, weights_only=True).to(torch.float32).cpu().numpy()
            cos_with_original = float(
                np.dot(v_matched, v_original)
                / (np.linalg.norm(v_matched) * np.linalg.norm(v_original))
            )
        else:
            cos_with_original = None

        artifact_report[layer_name] = {
            "authority_vs_neutral_accuracy": acc_auth_neut,
            "authority_vs_expertise_accuracy": acc_auth_exp,
            "cosine_matched_vs_original_authority": cos_with_original,
            "note": "All prompts are domain-free (no IT keywords).",
        }
        print(f"  Auth vs Neutral accuracy: {acc_auth_neut:.4f}")
        print(f"  Auth vs Expertise accuracy: {acc_auth_exp:.4f}")
        print(f"  Cosine (matched vs original v_auth): {cos_with_original}")

    full_report["artifact_control"] = artifact_report

    with open(out_dir / "artifact_control.json", "w") as f:
        json.dump(artifact_report, f, indent=2)

    # ==================================================================
    # FINAL CONSOLIDATED REPORT
    # ==================================================================
    with open(out_dir / "latent_concept_report.json", "w") as f:
        json.dump(full_report, f, indent=2)

    manager.release(model_id)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n=== M6.0.9 Latent Concept Geometry: COMPLETATO ===")
