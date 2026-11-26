"""Shared training setup helpers."""

from __future__ import annotations

import logging
import random
from typing import Any

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Set Python and (when installed) framework random seeds."""
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            logger.info("CUDA memory: allocated=%d reserved=%d", torch.cuda.memory_allocated(),
                        torch.cuda.memory_reserved())
    except ImportError:
        logger.debug("PyTorch is not installed; seeded Python random only")


def load_training_components(model_path: str, *, quantization: str | None,
                             lora: dict[str, Any]) -> tuple[Any, Any]:
    """Load a local model/tokenizer and apply optional QLoRA; never downloads implicitly."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    kwargs: dict[str, Any] = {"local_files_only": True, "device_map": "auto"}
    if quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
                                                           bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype="bfloat16",
                                                           bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("Post-load CUDA memory: allocated=%d reserved=%d",
                        torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
    except ImportError:
        pass
    if lora.get("enabled", True):
        if quantization == "4bit":
            model = prepare_model_for_kbit_training(model)
        config = LoraConfig(r=int(lora.get("r", 16)), lora_alpha=int(lora.get("alpha", 32)),
                            lora_dropout=float(lora.get("dropout", 0.05)),
                            target_modules=lora.get("target_modules"), task_type="CAUSAL_LM")
        model = get_peft_model(model, config)
    return model, tokenizer


def training_arguments(config: dict[str, Any], output_dir: str) -> Any:
    """Create shared Transformers training arguments with reproducible settings."""
    from transformers import TrainingArguments

    return TrainingArguments(output_dir=output_dir, seed=int(config.get("seed", 42)),
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
        learning_rate=float(config.get("learning_rate", 2e-5)),
        num_train_epochs=float(config.get("num_train_epochs", 1)),
        logging_steps=int(config.get("logging_steps", 10)), save_steps=int(config.get("save_steps", 250)),
        bf16=config.get("precision", "bf16") == "bf16", report_to="none")
