"""Local-only Transformers generator used by evaluation."""

from __future__ import annotations


class TransformersGenerator:
    """Single-device generator with optional QLoRA adapter for 8 GB class GPUs."""

    def __init__(self, model_path: str, *, adapter_path: str | None = None,
                 quantization: str | None = "4bit") -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        kwargs = {"local_files_only": True, "device_map": "auto"}
        if quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path,
                                                   local_files_only=True)
        self.model.eval()

    def generate(self, prompts: list[str], max_new_tokens: int, *, do_sample: bool = False,
                 temperature: float = 0.6, top_p: float = 0.95) -> list[str]:
        import torch
        output: list[str] = []
        for prompt in prompts:
            if getattr(self.tokenizer, "chat_template", None):
                prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False,
                    add_generation_prompt=True)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                generation = {"max_new_tokens": max_new_tokens, "do_sample": do_sample,
                              "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id}
                if do_sample:
                    generation.update(temperature=temperature, top_p=top_p)
                tokens = self.model.generate(**inputs, **generation)
            output.append(self.tokenizer.decode(tokens[0][inputs["input_ids"].shape[1]:],
                                                skip_special_tokens=True))
        return output
