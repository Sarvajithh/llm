"""Tests for export configuration and Modelfile generation."""

from src.config.loader import validate_export
from src.export.ollama import write_modelfile


def test_export_config_accepts_configured_paths() -> None:
    validate_export({"base_model_path": "base", "adapter_path": "adapter",
                     "merged_output_path": "merged",
                     "llama_cpp_path": "llama.cpp", "gguf_output_path": "model.gguf",
                     "quantized_gguf_path": "model-q4.gguf", "ollama_model_name": "model",
                     "ollama_modelfile_path": "Modelfile",
                     "gguf_quantization": "Q4_K_M", "context_length": 4096})


def test_modelfile_contents(tmp_path) -> None:
    path = write_modelfile("model.gguf", str(tmp_path / "Modelfile"), 2048)
    assert "FROM model.gguf" in path.read_text(encoding="utf-8")
