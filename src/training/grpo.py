"""GRPO training with caller-supplied verifiable reward functions."""

from __future__ import annotations

from typing import Any, Callable

from src.training.common import (liger_enabled, load_training_components, memory_monitor_callbacks,
                                 seed_everything, supported_kwargs)


class GRPOTrainer:
    """Run TRL GRPO with explicit reward functions and local-only model loading."""

    def train(self, config: dict[str, Any], model_path: str, dataset: Any,
              rewards: list[Callable[..., float]]) -> Any:
        from trl import GRPOConfig, GRPOTrainer as TRLGRPOTrainer

        if not rewards:
            raise ValueError("At least one reward function is required")
        seed_everything(int(config.get("seed", 42)))
        model, tokenizer = load_training_components(model_path, quantization=config.get("quantization"),
                                                     lora=config.get("lora", {}),
                                                     adapter_path=config.get("initial_adapter"),
                                                     attention_implementation=config.get(
                                                         "attention_implementation", "sdpa"))
        if getattr(tokenizer, "chat_template", None):
            def format_prompt(row: dict[str, Any]) -> dict[str, Any]:
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["prompt"]}], tokenize=False,
                    add_generation_prompt=True)
                return {"prompt": rendered}
            dataset = dataset.map(format_prompt)
        use_liger = liger_enabled(config)
        args_values = dict(output_dir=config["output_dir"], seed=int(config.get("seed", 42)),
            per_device_train_batch_size=int(config.get("per_device_train_batch_size", 1)),
            gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
            learning_rate=float(config.get("learning_rate", 1e-6)),
            num_train_epochs=float(config.get("num_train_epochs", 1)),
            max_prompt_length=int(config.get("max_prompt_length", 1024)),
            max_completion_length=int(config.get("max_completion_length", 2048)),
            num_generations=int(config.get("num_generations", 2)),
            beta=float(config.get("beta", 0.04)),
            loss_type=config.get("loss_type", "dr_grpo"),
            mask_truncated_completions=bool(config.get("mask_truncated_completions", True)),
            optim=config.get("optim", "paged_adamw_8bit"),
            use_liger_kernel=use_liger,
            use_liger_loss=use_liger,
            generation_kwargs={"use_cache": True},
            bf16=config.get("precision", "bf16") == "bf16",
            fp16=config.get("precision", "bf16") == "fp16",
            gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
            report_to="none")
        args = GRPOConfig(**supported_kwargs(GRPOConfig, args_values))
        trainer_values = dict(model=model, args=args, train_dataset=dataset,
                              reward_funcs=rewards, processing_class=tokenizer,
                              tokenizer=tokenizer, callbacks=memory_monitor_callbacks(config))
        trainer = TRLGRPOTrainer(**supported_kwargs(TRLGRPOTrainer, trainer_values))
        result = trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
        trainer.save_model(config["output_dir"])
        tokenizer.save_pretrained(config["output_dir"])
        return result
