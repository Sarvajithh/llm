"""Supervised fine-tuning / response-level distillation."""

from __future__ import annotations

from typing import Any

from src.training.common import load_training_components, seed_everything, supported_kwargs, training_arguments


class SFTTrainer:
    """Run response-level SFT using TRL's trainer implementation."""

    def train(self, config: dict[str, Any], model_path: str, dataset: Any,
              eval_dataset: Any | None = None) -> Any:
        from trl import SFTTrainer as TRLSFTTrainer

        seed_everything(int(config.get("seed", 42)))
        model, tokenizer = load_training_components(model_path, quantization=config.get("quantization"),
                                                     lora=config.get("lora", {}),
                                                     adapter_path=config.get("initial_adapter"))
        def format_rows(rows: Any) -> Any:
            def render(row: dict[str, str]) -> dict[str, str]:
                prompt, response = row["prompt"], row["response"]
                if getattr(tokenizer, "chat_template", None):
                    messages = [{"role": "user", "content": prompt},
                                {"role": "assistant", "content": response}]
                    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                         add_generation_prompt=False)
                else:
                    text = f"User:\n{prompt}\n\nAssistant:\n{response}{tokenizer.eos_token or ''}"
                return {"text": text}
            return rows.map(render, remove_columns=rows.column_names)

        train_dataset = format_rows(dataset)
        validation = format_rows(eval_dataset) if eval_dataset is not None and len(eval_dataset) else None
        trainer_values = dict(model=model, args=training_arguments(config, config["output_dir"]),
                              train_dataset=train_dataset, eval_dataset=validation,
                              processing_class=tokenizer, tokenizer=tokenizer,
                              max_seq_length=int(config.get("max_seq_length", 2048)),
                              max_length=int(config.get("max_seq_length", 2048)),
                              dataset_text_field="text")
        trainer = TRLSFTTrainer(**supported_kwargs(TRLSFTTrainer, trainer_values))
        result = trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
        trainer.save_model(config["output_dir"])
        tokenizer.save_pretrained(config["output_dir"])
        return result
