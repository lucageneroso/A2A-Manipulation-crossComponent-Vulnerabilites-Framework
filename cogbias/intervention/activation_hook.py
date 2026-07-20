import torch

class ActivationHook:
    def __init__(self, model, layer_idx, direction, alpha):
        self.model = model
        self.layer_idx = layer_idx
        if isinstance(direction, torch.Tensor):
            self.direction = direction
        else:
            self.direction = torch.tensor(direction, dtype=torch.float32)
        self.alpha = alpha
        self.hook_handle = None
        
        # normalize direction
        norm = torch.linalg.norm(self.direction)
        if norm > 0:
            self.direction = self.direction / norm

    def _hook_fn(self, module, input, output):
        # output is a tuple (hidden_states, ...)
        hidden_states = output[0]
        
        # direction shape is (D,)
        # hidden_states shape is (B, L, D)
        direction_reshaped = self.direction.view(1, 1, -1).to(hidden_states.device, dtype=hidden_states.dtype)
        
        # Inject at all tokens
        modified_hidden_states = hidden_states + self.alpha * direction_reshaped
        
        if isinstance(output, tuple):
            return (modified_hidden_states,) + output[1:]
        else:
            return modified_hidden_states

    def register(self):
        layer_module = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layer_module = self.model.model.layers[self.layer_idx]
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            layer_module = self.model.transformer.h[self.layer_idx]
        else:
            raise ValueError("Unsupported model architecture for hooking")
            
        self.hook_handle = layer_module.register_forward_hook(self._hook_fn)

    def remove(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
