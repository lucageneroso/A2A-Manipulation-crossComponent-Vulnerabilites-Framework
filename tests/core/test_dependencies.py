import sys

def test_core_has_no_ml_dependencies():
    """
    Assicura che l'importazione dei moduli core non inneschi
    il caricamento di librerie ML pesanti come torch o transformers.
    """
    # Importiamo tutto il core
    import cogbias.core.interfaces
    import cogbias.core.schemas
    import cogbias.core.pipeline
    import cogbias.core.shared_model_manager

    forbidden_modules = ['torch', 'transformers', 'bitsandbytes', 'accelerate']
    
    for module in forbidden_modules:
        assert module not in sys.modules, f"Errore critico: il core ha importato {module}!"
