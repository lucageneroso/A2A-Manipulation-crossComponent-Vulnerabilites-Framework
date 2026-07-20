from typing import Any
import hashlib
from cogbias.stages.prompt_formatting.base import PromptFormatter
from cogbias.core.schemas import FormattedPrompt

class QwenChatFormatter(PromptFormatter):
    """
    Applica il chat template ufficiale di Qwen:
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    {prompt}<|im_end|>
    <|im_start|>assistant
    """
    def __init__(self, template_id: str = "qwen_chat_template_v1"):
        self.template_id = template_id

    def format(self, payload_data: str, scenario: Any) -> FormattedPrompt:
        system_prompt = "You are a helpful AI assistant."
        if scenario and hasattr(scenario, "system_prompt"):
            system_prompt = scenario.system_prompt
            
        formatted_text = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{payload_data}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        orig_hash = hashlib.sha256(payload_data.encode("utf-8")).hexdigest()
        fmt_hash = hashlib.sha256(formatted_text.encode("utf-8")).hexdigest()
        
        return FormattedPrompt(
            text=formatted_text,
            template_id=self.template_id,
            hash=fmt_hash,
            original_prompt_hash=orig_hash,
            metadata={"system_prompt": system_prompt}
        )
