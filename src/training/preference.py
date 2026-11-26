"""DPO-family preference optimization through TRL."""

from __future__ import annotations

from typing import Any

from src.training.common import load_training_components, seed_everything


class PreferenceTrainer:
    """Select TRL DPO, ORPO or CPO (SimPO loss) from configuration."""

    def train(self, config: dict[str, Any], model_path: str, dataset: Any) -> Any:
        from transformers import Trainer, TrainingArguments
        from trl import DPOConfig, DPOTrainer, ORPOConfig, ORPOTrainer

        seed_everything(int(config.get("seed", 42)))
        algorithm = config.get("algorithm", "dpo").lower()
        model, tokenizer = load_training_components(model_path, quantization=config.get("quantization"),
                                                     lora=config.get("lora", {}))
        common = dict(output_dir=config["output_dir"], seed=int(config.get("seed", 42)),
                      per_device_train_batch_size=int(config.get("per_device_train_batch_size", 1)),
                      gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
                      learning_rate=float(config.get("learning_rate", 5e-6)),
                      num_train_epochs=float(config.get("num_train_epochs", 1)), report_to="none")
        if algorithm == "simpo":
            import torch
            import torch.nn.functional as functional

            tokenizer.padding_side = "right"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token

            class PairCollator:
                """Tokenize prompt/response pairs and mask prompt tokens in labels."""

                def __call__(self, features: list[dict[str, str]]) -> dict[str, torch.Tensor]:
                    batch: dict[str, list[list[int]]] = {"chosen_input_ids": [], "chosen_labels": [],
                        "chosen_attention_mask": [], "rejected_input_ids": [], "rejected_labels": [],
                        "rejected_attention_mask": []}
                    for row in features:
                        prompt = tokenizer(row["prompt"], add_special_tokens=True)["input_ids"]
                        for key in ("chosen", "rejected"):
                            answer = tokenizer(row[key], add_special_tokens=False)["input_ids"]
                            ids = (prompt + answer + [tokenizer.eos_token_id])[:int(config.get("max_length", 4096))]
                            labels = ([-100] * len(prompt) + answer + [tokenizer.eos_token_id])[:len(ids)]
                            batch[f"{key}_input_ids"].append(ids)
                            batch[f"{key}_labels"].append(labels)
                            batch[f"{key}_attention_mask"].append([1] * len(ids))
                    result: dict[str, torch.Tensor] = {}
                    for key, rows in batch.items():
                        pad = tokenizer.pad_token_id if "input_ids" in key else -100
                        if "attention_mask" in key:
                            pad = 0
                        max_len = max(map(len, rows))
                        result[key] = torch.tensor([row + [pad] * (max_len - len(row)) for row in rows])
                    return result

            class SimPOTrainer(Trainer):
                """Reference-free SimPO objective over length-normalized response log-probabilities."""

                def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                    def sequence_score(ids_key: str, labels_key: str) -> torch.Tensor:
                        ids = inputs[ids_key].to(model.device)
                        labels = inputs[labels_key].to(model.device)
                        attention = inputs[f"{ids_key.replace('input_ids', 'attention_mask')}"].to(model.device)
                        logits = model(input_ids=ids, attention_mask=attention).logits[:, :-1]
                        target = labels[:, 1:]
                        log_probs = functional.log_softmax(logits, dim=-1)
                        safe_target = target.masked_fill(target == -100, 0)
                        token_log_probs = log_probs.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
                        token_log_probs = token_log_probs.masked_fill(target == -100, 0)
                        counts = (target != -100).sum(dim=-1).clamp_min(1)
                        return token_log_probs.sum(dim=-1) / counts

                    chosen = sequence_score("chosen_input_ids", "chosen_labels")
                    rejected = sequence_score("rejected_input_ids", "rejected_labels")
                    beta = float(config.get("beta", 2.0))
                    gamma = float(config.get("gamma_beta_ratio", 0.5)) * beta
                    loss = -functional.logsigmoid(beta * (chosen - rejected) - gamma).mean()
                    return (loss, {"chosen_score": chosen, "rejected_score": rejected}) if return_outputs else loss

            args = TrainingArguments(output_dir=config["output_dir"], seed=int(config.get("seed", 42)),
                per_device_train_batch_size=int(config.get("per_device_train_batch_size", 1)),
                gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
                learning_rate=float(config.get("learning_rate", 5e-6)),
                num_train_epochs=float(config.get("num_train_epochs", 1)), report_to="none",
                remove_unused_columns=False)
            trainer = SimPOTrainer(model=model, args=args, train_dataset=dataset,
                                   data_collator=PairCollator())
            return trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
        if algorithm == "orpo":
            trainer_cls, args_cls = ORPOTrainer, ORPOConfig
            args = args_cls(**common, beta=float(config.get("beta", 0.1)),
                            max_length=int(config.get("max_length", 4096)))
        elif algorithm == "dpo":
            trainer_cls, args_cls = DPOTrainer, DPOConfig
            args = args_cls(**common, beta=float(config.get("beta", 0.1)),
                            max_length=int(config.get("max_length", 4096)),
                            max_prompt_length=int(config.get("max_prompt_length", 2048)),
                            loss_type="sigmoid")
        else:
            raise ValueError("algorithm must be dpo, orpo, or simpo")
        trainer = trainer_cls(model=model, args=args, train_dataset=dataset,
                              processing_class=tokenizer)
        return trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
