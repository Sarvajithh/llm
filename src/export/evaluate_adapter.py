"""Local-only Transformers generator used by evaluation."""

from __future__ import annotations


class TransformersGenerator:
    """Minimal batched text generator, loading only local model files."""

    def __init__(self, model_path: str) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                                                          device_map="auto")

    def generate(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        import torch
        output: list[str] = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                tokens = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            output.append(self.tokenizer.decode(tokens[0][inputs["input_ids"].shape[1]:],
                                                skip_special_tokens=True))
        return output
