"""Supervised fine-tuning / response-level distillation."""

from __future__ import annotations

from typing import Any

from src.training.common import load_training_components, seed_everything, training_arguments


class SFTTrainer:
    """Run response-level SFT using TRL's trainer implementation."""

    def train(self, config: dict[str, Any], model_path: str, dataset: Any) -> Any:
        from trl import SFTTrainer as TRLSFTTrainer

        seed_everything(int(config.get("seed", 42)))
        model, tokenizer = load_training_components(model_path, quantization=config.get("quantization"),
                                                     lora=config.get("lora", {}))
        trainer = TRLSFTTrainer(model=model, args=training_arguments(config, config["output_dir"]),
                                train_dataset=dataset, processing_class=tokenizer,
                                max_seq_length=int(config.get("max_seq_length", 4096)))
        return trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
