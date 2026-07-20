from typing import Any, Dict
from cogbias.core.interfaces import ModelInterface

class TransformersAdapter(ModelInterface):
    """
    Adapter per caricare e usare i modelli HuggingFace in 4-bit.
    Tutte le dipendenze ML pesanti (torch, transformers) sono importate localmente
    per evitare inquinamenti nel modulo core puro.
    """
    def __init__(self, model_id: str, revision: str = "main", quantization: str = "nf4"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        if not hasattr(torch.nn.Module, "set_submodule"):
            def set_submodule(self, target: str, module: torch.nn.Module) -> None:
                atoms = target.split(".")
                name = atoms.pop(-1)
                mod = self
                for item in atoms:
                    if not hasattr(mod, item):
                        raise AttributeError(f"Module {mod} has no attribute {item}")
                    mod = getattr(mod, item)
                    if not isinstance(mod, torch.nn.Module):
                        raise AttributeError(f"'{item}' is not an nn.Module")
                setattr(mod, name, module)
            torch.nn.Module.set_submodule = set_submodule

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        
        if quantization == "nf4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                revision=revision,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, revision=revision, device_map="auto"
            )
            
    def tokenize(self, text: str) -> Any:
        return dict(self.tokenizer(text, return_tensors="pt").to(self.model.device))

    def prepare_input(self, representation: Any) -> Dict[str, Any]:
        """
        Produce un ModelInput agnostico a partire dalla Representation.
        """
        import torch
        if representation.type == "text":
            return self.tokenize(representation.data)
        elif representation.type == "latent":
            # data è un dict con inputs_embeds, attention_mask, position_ids
            data_dict = representation.data
            model_input = {}
            for k, v in data_dict.items():
                if v is not None:
                    model_input[k] = v.to(self.model.device)
            return model_input
        else:
            raise ValueError(f"Unsupported representation type: {representation.type}")

    def generate(self, model_input: Dict[str, Any], config: Dict[str, Any]) -> Any:
        import torch
        with torch.no_grad():
            output_ids = self.model.generate(
                **model_input,
                max_new_tokens=config.get("max_new_tokens", 50),
                temperature=config.get("temperature", 0.0),
                do_sample=config.get("temperature", 0.0) > 0
            )
            
            # Se l'input includeva input_ids, dobbiamo rimuoverli dall'output
            if "input_ids" in model_input:
                generated_tokens = output_ids[0][model_input["input_ids"].shape[1]:]
            else:
                generated_tokens = output_ids[0]
                
            return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def forward_diagnostic(self, model_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Esegue un forward pass puro (senza generazione autoregressiva) e restituisce
        i logits e gli hidden states per analisi di fedeltà (M5.1.1).
        """
        import torch
        with torch.no_grad():
            outputs = self.model(
                **model_input,
                output_hidden_states=True,
                return_dict=True
            )
            return {
                "logits": outputs.logits,
                "hidden_states": outputs.hidden_states
            }
