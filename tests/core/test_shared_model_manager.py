from cogbias.core.shared_model_manager import SharedModelManager

class MockModel:
    pass

def test_model_is_shared():
    manager = SharedModelManager()
    
    # Definiamo un provider factory
    def provider():
        return MockModel()
        
    manager.load("qwen2.5", provider)
    
    model_a = manager.get("qwen2.5")
    model_b = manager.get("qwen2.5")
    
    # Asserzione critica: i due modelli recuperati devono essere 
    # ESATTAMENTE la stessa istanza in memoria per evitare OOM su 4GB VRAM.
    assert model_a is model_b
    
def test_model_release():
    manager = SharedModelManager()
    manager.load("test_model", lambda: MockModel())
    assert manager.get("test_model") is not None
    
    manager.release("test_model")
    
    # Ora dovrebbe sollevare eccezione
    import pytest
    with pytest.raises(ValueError):
        manager.get("test_model")
