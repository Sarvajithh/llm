"""GRPO training with caller-supplied verifiable reward functions."""

from __future__ import annotations

from typing import Any, Callable

from src.training.common import load_training_components, seed_everything


class GRPOTrainer:
    """Run TRL GRPO with explicit reward functions and local-only model loading."""

    def train(self, config: dict[str, Any], model_path: str, dataset: Any,
              rewards: list[Callable[..., float]]) -> Any:
        from trl import GRPOConfig, GRPOTrainer as TRLGRPOTrainer

        if not rewards:
            raise ValueError("At least one reward function is required")
        seed_everything(int(config.get("seed", 42)))
        model, tokenizer = load_training_components(model_path, quantization=config.get("quantization"),
                                                     lora=config.get("lora", {}))
        args = GRPOConfig(output_dir=config["output_dir"], seed=int(config.get("seed", 42)),
            per_device_train_batch_size=int(config.get("per_device_train_batch_size", 1)),
            gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
            learning_rate=float(config.get("learning_rate", 1e-6)),
            num_train_epochs=float(config.get("num_train_epochs", 1)),
            max_prompt_length=int(config.get("max_prompt_length", 1024)),
            max_completion_length=int(config.get("max_completion_length", 2048)),
            num_generations=int(config.get("num_generations", 4)), report_to="none")
        trainer = TRLGRPOTrainer(model=model, args=args, train_dataset=dataset,
                                 reward_funcs=rewards, processing_class=tokenizer)
        return trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
