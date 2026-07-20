import pytest
import torch
from cogbias.core.shared_model_manager import SharedModelManager

MAX_MODEL_MEMORY_MB = 2800

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for VRAM testing")
def test_model_load_under_vram_limit():
    from cogbias.model_adapter.transformers_adapter import TransformersAdapter
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    initial_alloc = torch.cuda.memory_allocated() / (1024**2)
    
    manager = SharedModelManager()
    
    # Load Qwen in 4-bit nf4
    manager.load("qwen", lambda: TransformersAdapter("Qwen/Qwen2.5-1.5B-Instruct", quantization="nf4"))
    
    alloc_mb = torch.cuda.memory_allocated() / (1024**2)
    reserved_mb = torch.cuda.memory_reserved() / (1024**2)
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    
    vram_stats = {
        "allocated_mb": alloc_mb,
        "reserved_mb": reserved_mb,
        "peak_mb": peak_mb
    }
    
    print(f"VRAM Stats: {vram_stats}")
    
    # Validation against budget
    assert peak_mb <= MAX_MODEL_MEMORY_MB, f"Peak memory {peak_mb} exceeded {MAX_MODEL_MEMORY_MB} MB"
    
    # Cleanup verification
    manager.release("qwen")
    
    # Rilasciamo esplicitamente il GC e la cache
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    final_alloc = torch.cuda.memory_allocated() / (1024**2)
    # Tollera 100MB di overhead non raccolto dal CUDA allocator
    assert final_alloc <= initial_alloc + 100, "VRAM was not properly cleaned up"
