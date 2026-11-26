"""Ollama Modelfile generation and explicit model-create command helper."""

from __future__ import annotations

import subprocess
from pathlib import Path


def write_modelfile(gguf_path: str, output_path: str, context_length: int,
                    system_prompt: str = "You are a helpful assistant.") -> Path:
    """Write an Ollama Modelfile referencing a GGUF artifact."""
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = f'FROM {gguf_path}\nPARAMETER num_ctx {context_length}\nSYSTEM """{system_prompt}"""\n'
    target.write_text(content, encoding="utf-8")
    return target


def create_ollama_model(modelfile: str, model_name: str) -> None:
    """Create an Ollama model when explicitly requested by the user."""
    subprocess.run(["ollama", "create", model_name, "-f", modelfile], check=True)
