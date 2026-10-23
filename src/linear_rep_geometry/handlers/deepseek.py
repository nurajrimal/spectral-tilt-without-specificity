import torch
import transformers
from .base import ModelHandler
from typing import Optional

class DeepSeekHandler(ModelHandler):
    """Handler for DeepSeek AI models.
    
    Covers: DeepSeek-V2-Lite, deepseek-moe-16b-base, etc.
    
    Architecture notes:
    - Custom architectures requiring trust_remote_code=True
    - DeepSeek-V2-Lite: MoE with Multi-head Latent Attention (MLA)
      15.7B total / 2.4B active, hidden_size=2048
    - DeepSeek-MoE-16B: Fine-grained MoE with shared experts
      16.4B total / 2.7B active, hidden_size=4096
    - Both use custom modeling code from DeepSeek
    """

    def load(self, device_map: str = "auto"):
        print(f"Loading DeepSeek: {self.model_path}")
        print(f"  → trust_remote_code=True (custom DeepSeek architecture)")
        
        # Patch missing flash_attn utility that DeepSeek's custom code tries to import
        # from newer transformers versions (removed in recent releases)
        import transformers.utils as _tu
        if not hasattr(_tu, 'is_flash_attn_greater_or_equal_2_10'):
            _tu.is_flash_attn_greater_or_equal_2_10 = lambda: False
        if not hasattr(_tu, 'is_flash_attn_greater_or_equal'):
            _tu.is_flash_attn_greater_or_equal = lambda v: False
        
        # Load config first to fix RoPE integer bug
        config = transformers.AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if hasattr(config, 'rope_scaling') and isinstance(config.rope_scaling, dict):
            for key in ['beta_fast', 'beta_slow', 'factor', 'mscale', 'mscale_all_dim']:
                if key in config.rope_scaling and isinstance(config.rope_scaling[key], int):
                    config.rope_scaling[key] = float(config.rope_scaling[key])
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=config,
            low_cpu_mem_usage=True,
            device_map=device_map,
            torch_dtype="auto",
            trust_remote_code=True,
            attn_implementation="eager"  # avoid flash_attn dependency
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def format_input(self, text: str, source_lang: str = "en", target_lang_code: str = "fr", tokenizer: Optional[transformers.PreTrainedTokenizer] = None) -> str:
        """DeepSeek base models: raw text. Chat models: DeepSeek template."""
        if tokenizer is None:
            return text
        
        if "chat" in self.model_path.lower() or "instruct" in self.model_path.lower():
            messages = [{"role": "user", "content": text}]
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                return text
        
        return text

    def get_norm_layer(self, model: torch.nn.Module) -> Optional[torch.nn.Module]:
        # DeepSeek standard path
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            return model.model.norm
        # DeepSeek V2 alternate path
        elif hasattr(model, "model") and hasattr(model.model, "final_norm"):
            return model.model.final_norm
        return None
