"""YAML configuration loading with lightweight validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is missing required or valid values."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML mapping from ``path`` and reject empty or non-mapping documents."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_path}")
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"Expected a YAML mapping in {config_path}")
    return value


def require(config: dict[str, Any], *keys: str) -> None:
    """Require non-empty top-level keys in a config mapping."""
    missing = [key for key in keys if key not in config or config[key] in (None, "")]
    if missing:
        raise ConfigError(f"Missing required configuration values: {', '.join(missing)}")


def validate_split(split: dict[str, float]) -> None:
    """Validate train/validation/test proportions, allowing omitted splits as zero."""
    allowed = {"train", "validation", "test"}
    if set(split) - allowed:
        raise ConfigError(f"Unsupported split names: {sorted(set(split) - allowed)}")
    if any(value < 0 or value > 1 for value in split.values()):
        raise ConfigError("Split proportions must be between 0 and 1")
    if abs(sum(split.values()) - 1.0) > 1e-8:
        raise ConfigError("Split proportions must sum to 1")


def validate_training(config: dict[str, Any], algorithm: str) -> None:
    """Validate common trainer fields and algorithm-specific positive settings."""
    require(config, "model", "dataset_config", "dataset_group", "output_dir")
    for key in ("seed", "per_device_train_batch_size", "gradient_accumulation_steps",
                "num_train_epochs"):
        if key in config and config[key] <= 0:
            raise ConfigError(f"{key} must be positive")
    if algorithm in {"dpo", "orpo", "simpo"} and config.get("beta", 0) <= 0:
        raise ConfigError("beta must be positive for preference optimization")
    if algorithm not in {"sft", "dpo", "orpo", "simpo", "grpo"}:
        raise ConfigError(f"Unsupported training algorithm: {algorithm}")


def validate_export(config: dict[str, Any]) -> None:
    """Validate the paths and identifiers required by export workflows."""
    require(config, "base_model_path", "adapter_path", "merged_output_path", "context_length",
            "llama_cpp_path", "gguf_output_path", "quantized_gguf_path",
            "ollama_modelfile_path", "ollama_model_name", "gguf_quantization")
    if config.get("context_length", 0) <= 0:
        raise ConfigError("context_length must be a positive integer")
    if config.get("gguf_quantization") not in {"Q4_0", "Q4_K_M", "Q5_K_M", "Q8_0"}:
        raise ConfigError("Unsupported GGUF quantization; choose Q4_0, Q4_K_M, Q5_K_M, or Q8_0")
