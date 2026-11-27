"""Shared training setup helpers."""

from __future__ import annotations

import logging
import random
import inspect
import os
from typing import Any

# Must be set before PyTorch initializes the CUDA allocator.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logger = logging.getLogger(__name__)


def supported_kwargs(callable_obj: Any, values: dict[str, Any]) -> dict[str, Any]:
    """Drop optional arguments absent from the installed Transformers/TRL API."""
    parameters = inspect.signature(callable_obj).parameters
    return {key: value for key, value in values.items() if key in parameters}


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
                             lora: dict[str, Any],
                             adapter_path: str | None = None,
                             attention_implementation: str = "sdpa") -> tuple[Any, Any]:
    """Load a local model/tokenizer and apply optional QLoRA; never downloads implicitly."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    kwargs: dict[str, Any] = {"local_files_only": True, "device_map": "auto",
                              "attn_implementation": attention_implementation}
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
    if adapter_path:
        from peft import PeftModel
        if quantization == "4bit":
            model = prepare_model_for_kbit_training(model)
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    elif lora.get("enabled", True):
        if quantization == "4bit":
            model = prepare_model_for_kbit_training(model)
        config = LoraConfig(r=int(lora.get("r", 16)), lora_alpha=int(lora.get("alpha", 32)),
                            lora_dropout=float(lora.get("dropout", 0.0)),
                            target_modules=lora.get("target_modules"), task_type="CAUSAL_LM")
        model = get_peft_model(model, config)
    if hasattr(model, "config"):
        model.config.use_cache = False
    return model, tokenizer


def training_arguments(config: dict[str, Any], output_dir: str,
                       args_class: Any = None, extra: dict[str, Any] | None = None) -> Any:
    """Create shared Transformers training arguments with reproducible settings."""
    if args_class is None:
        from transformers import TrainingArguments
        args_class = TrainingArguments

    values = dict(output_dir=output_dir, seed=int(config.get("seed", 42)),
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
        learning_rate=float(config.get("learning_rate", 2e-5)),
        num_train_epochs=float(config.get("num_train_epochs", 1)),
        logging_steps=int(config.get("logging_steps", 10)), save_steps=int(config.get("save_steps", 250)),
        bf16=config.get("precision", "bf16") == "bf16",
        fp16=config.get("precision", "bf16") == "fp16",
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=config.get("optim", "paged_adamw_8bit"),
        eval_strategy="steps" if config.get("do_eval", False) else "no",
        evaluation_strategy="steps" if config.get("do_eval", False) else "no",
        eval_steps=int(config.get("eval_steps", config.get("logging_steps", 10))),
        report_to="none")
    values.update(extra or {})
    values["use_liger_kernel"] = liger_enabled(config)
    return args_class(**supported_kwargs(args_class, values))


def liger_enabled(config: dict[str, Any]) -> bool:
    """Enable Liger only when requested and installed; otherwise use native kernels."""
    if not config.get("use_liger_kernel", False):
        return False
    try:
        import liger_kernel  # noqa: F401
    except ImportError:
        logger.warning("Liger requested but not installed; continuing with native kernels")
        return False
    return True


def memory_monitor_callbacks(config: dict[str, Any]) -> list[Any]:
    """Return a callback that logs peak allocated VRAM after warmup training steps."""
    warmup_steps = int(config.get("memory_check_steps", 20))
    budget_gb = float(config.get("memory_budget_gb", 7.2))
    if warmup_steps <= 0:
        return []
    from transformers import TrainerCallback

    class PeakCudaMemoryCallback(TrainerCallback):
        reported = False

        def on_train_begin(self, args, state, control, **kwargs):
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step >= warmup_steps:
                self.report()
            return control

        def on_train_end(self, args, state, control, **kwargs):
            self.report()
            return control

        def report(self):
            if self.reported:
                return
            import torch
            if not torch.cuda.is_available():
                return
            peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
            logger.info("CUDA peak memory after warmup: allocated=%.2f GiB reserved=%.2f GiB "
                        "(allocated budget %.2f GiB)", peak_gb, reserved_gb, budget_gb)
            if peak_gb > budget_gb:
                logger.warning("Peak VRAM exceeded configured headroom budget")
            self.reported = True

    return [PeakCudaMemoryCallback()]
