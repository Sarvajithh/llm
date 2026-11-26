"""Explicit adapter merge, GGUF conversion and quantization wrappers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any


def merge_adapter(base_model_path: str, adapter_path: str, output_path: str) -> None:
    """Merge a PEFT adapter into its local base and save Hugging Face format."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(base_model_path, local_files_only=True,
                                                device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_path, local_files_only=True)
    merged = model.merge_and_unload()
    target = Path(output_path)
    target.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(target, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_model_path, local_files_only=True).save_pretrained(target)


def export_gguf(llama_cpp_path: str, hf_path: str, gguf_path: str, quantized_path: str,
                quantization: str) -> None:
    """Invoke llama.cpp's converter and quantizer for a local merged model."""
    root = Path(llama_cpp_path)
    converter = root / "convert_hf_to_gguf.py"
    quantizer = root / "build" / "bin" / "llama-quantize"
    if sys.platform == "win32" and not quantizer.with_suffix(".exe").is_file():
        quantizer = quantizer.with_suffix(".exe")
    if not converter.is_file() or not quantizer.is_file():
        raise FileNotFoundError("llama.cpp converter or llama-quantize binary was not found")
    Path(gguf_path).parent.mkdir(parents=True, exist_ok=True)
    Path(quantized_path).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(converter), hf_path, "--outfile", gguf_path,
                    "--outtype", "f16"], check=True)
    subprocess.run([str(quantizer), gguf_path, quantized_path, quantization], check=True)
