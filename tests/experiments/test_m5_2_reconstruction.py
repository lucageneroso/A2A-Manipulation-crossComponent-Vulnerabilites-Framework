import pytest
import torch
import torch.nn.functional as F
from pathlib import Path
import shutil
import gc

from cogbias.model_adapter.transformers_adapter import TransformersAdapter
from cogbias.core.shared_model_manager import SharedModelManager
from cogbias.stages.representation.strategies.latent import LatentRepresentation
from cogbias.storage.representation_store import RepresentationStore
from safetensors.torch import load_file, save_file

@pytest.mark.hardware
def test_m5_2_representation_reconstruction():
    """
    M5.2 Representation Reconstruction Test.
    Dimostra che la rappresentazione latente è un artefatto sperimentale riproducibile,
    indipendentemente dal processo che l'ha generato.
    """
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    manager = SharedModelManager()
    
    out_dir = Path("runs/representations/m5_2")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    store = RepresentationStore(out_dir)
    
    prompt_text = "You are a helpful AI assistant.\n\nPlease process the request accordingly."
    
    # ---------------------------------------------------------
    # RUN A: Extraction
    # ---------------------------------------------------------
    print(f"Loading {model_id} for RUN A (Extraction)...")
    manager.load(model_id, lambda: TransformersAdapter(model_id, quantization="nf4"))
    adapter_A = manager.get(model_id)
    
    latent_strategy = LatentRepresentation(adapter_A)
    # Mocked payload solo per formattazione
    class MockPayload:
        metadata = {"formatted_prompt": {"text": prompt_text}}
    payload = MockPayload()
    
    representation_A = latent_strategy.encode(prompt_text, payload)
    
    # Salviamo l'artefatto
    artifact = store.save(
        representation=representation_A,
        rep_id="rep_test_m5_2",
        model_id=model_id,
        tokenizer_id=model_id
    )
    
    # Estraiamo logits e output di Run A come Ground Truth
    model_input_A = adapter_A.prepare_input(representation_A)
    out_A_diag = adapter_A.forward_diagnostic(model_input_A)
    logits_A = out_A_diag["logits"]
    
    trace_config = {
        "generation_params": {
            "temperature": 0.0,
            "do_sample": False,
            "max_new_tokens": 30
        }
    }
    
    output_A = adapter_A.generate(model_input_A, trace_config)
    print(f"Run A Output: {output_A}")

    # ---------------------------------------------------------
    # WIPE LLM STATE
    # ---------------------------------------------------------
    print("Wiping LLM State...")
    manager.release(model_id)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

    # ---------------------------------------------------------
    # RUN B: Replay
    # ---------------------------------------------------------
    print(f"Loading {model_id} for RUN B (Replay)...")
    manager.load(model_id, lambda: TransformersAdapter(model_id, quantization="nf4"))
    adapter_B = manager.get(model_id)
    
    artifact_path = store.metadata_dir / f"{artifact.id}.json"
    loaded_artifact, representation_B = store.load(str(artifact_path))
    
    # Assert Model Matching
    assert loaded_artifact.model_id == model_id, "Model mismatch"
    
    # Assert Tensor Identity
    assert store.verify_hash(loaded_artifact), "Hash mismatch in Run B"
    
    model_input_B = adapter_B.prepare_input(representation_B)
    out_B_diag = adapter_B.forward_diagnostic(model_input_B)
    logits_B = out_B_diag["logits"]
    
    # Model Fidelity: Logits Cosine Similarity
    flat_logits_A = logits_A.view(1, -1)
    flat_logits_B = logits_B.view(1, -1)
    cos_sim_logits = F.cosine_similarity(flat_logits_A, flat_logits_B).item()
    print(f"Run B - Logits Cosine Sim with Run A: {cos_sim_logits}")
    assert cos_sim_logits > 0.99, "Logits fidelity fallita"

    # Behavioral Fidelity: Output String Equality
    output_B = adapter_B.generate(model_input_B, trace_config)
    print(f"Run B Output: {output_B}")
    assert output_A == output_B, "Behavioral fidelity fallita: output diversi"

    # ---------------------------------------------------------
    # NEGATIVE TEST: Hash Integrity Failure
    # ---------------------------------------------------------
    print("Running Negative Integrity Test...")
    # Corrompiamo il file safetensors
    tensors = load_file(loaded_artifact.tensor_path)
    # Aggiungiamo rumore al primo elemento del tensore inputs_embeds
    tensors["inputs_embeds"][0][0][0] += 0.001
    save_file(tensors, loaded_artifact.tensor_path)
    
    # verify_hash dovrebbe fallire
    assert store.verify_hash(loaded_artifact) is False, "Hash verification dovrebbe fallire dopo corruzione"
    
    print("M5.2 Representation Reconstruction Test superato.")
