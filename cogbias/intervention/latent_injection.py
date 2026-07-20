from contextlib import contextmanager
from cogbias.intervention.activation_hook import ActivationHook

class LatentInjection:
    def __init__(self, model):
        self.model = model
        
    @contextmanager
    def inject(self, layer_idx, direction, alpha):
        if alpha == 0.0:
            yield
            return
            
        hook = ActivationHook(self.model, layer_idx, direction, alpha)
        hook.register()
        try:
            yield
        finally:
            hook.remove()
