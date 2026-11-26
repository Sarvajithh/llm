"""Command line entry points for data, training, evaluation and export workflows."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
from pathlib import Path
from typing import Any

from src.config.loader import load_yaml, validate_export, validate_split, validate_training

logger = logging.getLogger("local_llm_training")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _dataset(config_path: str, group: str, split: str = "train") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from src.data.prepare import prepare_dataset

    config = load_yaml(config_path)
    validate_split(config.get("split", {"train": 1.0}))
    path = Path("data/processed") / group
    expected = path / f"{group}_{split}.jsonl"
    if not expected.is_file():
        prepare_dataset(config, group, str(path))
    if not expected.is_file():
        raise FileNotFoundError(f"Prepared split is empty or missing: {expected}")
    rows = _read_rows(expected)
    if not rows and split == "train":
        raise ValueError(f"Prepared split contains no usable records: {expected}")
    return rows, config


def _model_path(model_key: str) -> str:
    models = load_yaml("configs/models.yaml").get("models", {})
    if model_key not in models:
        raise ValueError(f"Unknown model key {model_key!r}; configure it in configs/models.yaml")
    model = models[model_key]
    local_path = model.get("local_path")
    if local_path and Path(local_path).exists():
        return str(local_path)
    hf_model_id = model.get("hf_model_id")
    if hf_model_id:
        return str(hf_model_id)  # loaders are local_files_only; cached weights work offline
    raise FileNotFoundError(f"Neither local_path nor hf_model_id is configured for {model_key}")


def _download_model(model_key: str) -> None:
    """Explicitly download one configured Hugging Face model to its local directory."""
    models = load_yaml("configs/models.yaml").get("models", {})
    if model_key not in models:
        raise ValueError(f"Unknown model key {model_key!r}; configure it in configs/models.yaml")
    model = models[model_key]
    model_id, local_path = model.get("hf_model_id"), model.get("local_path")
    if not model_id or not local_path or "YOUR_" in str(model_id) + str(local_path):
        raise ValueError(f"{model_key} needs hf_model_id and local_path in configs/models.yaml")
    from huggingface_hub import snapshot_download

    target = Path(local_path)
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model_id, local_dir=str(target))
    logger.info("Downloaded %s to %s", model_id, target)


def _train_sft(config_path: str, dataset_group: str | None = None) -> None:
    from datasets import Dataset
    from src.training.sft import SFTTrainer

    config = load_yaml(config_path)
    validate_training(config, "sft")
    group = dataset_group or config["dataset_group"]
    rows, _ = _dataset(config["dataset_config"], group)
    validation_rows, _ = _dataset(config["dataset_config"], group, "validation")
    SFTTrainer().train(config, _model_path(config["model"]), Dataset.from_list(rows),
                       eval_dataset=Dataset.from_list(validation_rows) if validation_rows else None)


def _train_preference(config_path: str) -> None:
    from datasets import Dataset
    from src.training.preference import PreferenceTrainer

    config = load_yaml(config_path)
    validate_training(config, config.get("algorithm", "dpo"))
    rows, _ = _dataset(config["dataset_config"], config["dataset_group"])
    validation_rows, _ = _dataset(config["dataset_config"], config["dataset_group"], "validation")
    PreferenceTrainer().train(config, _model_path(config["model"]), Dataset.from_list(rows),
                              eval_dataset=Dataset.from_list(validation_rows)
                              if validation_rows else None)


def _train_grpo(config_path: str) -> None:
    from datasets import Dataset
    from src.rewards.registry import resolve_rewards
    from src.training.grpo import GRPOTrainer

    config = load_yaml(config_path)
    validate_training(config, "grpo")
    rows, _ = _dataset(config["dataset_config"], config["dataset_group"])
    reward_names = config.get("reward", [])
    sandbox = None
    if {"code_compilation", "unit_tests"} & set(reward_names):
        sandbox_config = config.get("sandbox", {})
        if sandbox_config.get("network_access", True):
            raise ValueError("Code reward sandbox network_access must be false")
        factory_path = sandbox_config.get("backend", "")
        if ":" not in factory_path or "YOUR_" in factory_path:
            raise ValueError("Configure sandbox.backend as 'python.module:factory' in the GRPO YAML")
        module_name, factory_name = factory_path.split(":", 1)
        factory = getattr(importlib.import_module(module_name), factory_name)
        sandbox = factory(sandbox_config)
    rewards = resolve_rewards(reward_names, sandbox=sandbox,
                              timeout_seconds=int(config.get("sandbox", {}).get("timeout_seconds", 10)))
    GRPOTrainer().train(config, _model_path(config["model"]), Dataset.from_list(rows), rewards)


def _evaluate(config_path: str) -> None:
    from src.evaluation.runner import evaluate_jsonl
    from src.export.evaluate_adapter import TransformersGenerator

    config = load_yaml(config_path)
    dataset_config = load_yaml(config["dataset_config"])
    group = config.get("dataset_group", "sft")
    specs = dataset_config.get("datasets", {}).get(group, [])
    if not specs:
        raise ValueError(f"No data source configured for evaluation group {group!r}")
    # Evaluation files should be prepared using the standard prepare-data command.
    data_path = Path("data/processed") / group / f"{group}_test.jsonl"
    if not data_path.is_file():
        raise FileNotFoundError(f"Run prepare-data first; missing {data_path}")
    model_configs = load_yaml("configs/models.yaml").get("models", {})
    model_paths = config.get("models_to_compare") or [config["model"]]
    comparisons = {}
    for model_ref in model_paths:
        if isinstance(model_ref, dict):
            name = model_ref["name"]
            base_ref = model_ref.get("model", config["model"])
            base_path = _model_path(base_ref) if base_ref in model_configs else base_ref
            adapter = model_ref.get("adapter")
            if adapter and not Path(adapter).is_dir():
                comparisons[name] = {"status": "skipped", "reason": f"adapter not found: {adapter}"}
                logger.info("Skipping %s; train its adapter first (%s)", name, adapter)
                continue
        else:
            name = str(model_ref)
            base_path = _model_path(model_ref) if model_ref in model_configs else model_ref
            adapter = None
        slug = name.replace("/", "_")
        report = evaluate_jsonl(TransformersGenerator(base_path, adapter_path=adapter,
                                                       quantization="4bit"), str(data_path),
            str(Path(config["output_dir"]) / slug / "predictions.jsonl"),
            int(config.get("max_new_tokens", 2048)),
            samples_per_prompt=max(config.get("pass_k", [1])), pass_k=config.get("pass_k", [1]),
            metric=config.get("metric", "exact"))
        comparisons[name] = report
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    logger.info("Evaluation metrics: %s", comparisons)


def _export(config_path: str, create_ollama: bool = False) -> None:
    from src.export.convert import export_gguf, merge_adapter
    from src.export.ollama import create_ollama_model, write_modelfile

    config = load_yaml(config_path)
    validate_export(config)
    merge_adapter(config["base_model_path"], config["adapter_path"], config["merged_output_path"])
    export_gguf(config["llama_cpp_path"], config["merged_output_path"], config["gguf_output_path"],
                config["quantized_gguf_path"], config["gguf_quantization"])
    modelfile = write_modelfile(config["quantized_gguf_path"], config["ollama_modelfile_path"],
                                int(config["context_length"]))
    if create_ollama:
        create_ollama_model(str(modelfile), config["ollama_model_name"])
    else:
        logger.info("Wrote Ollama Modelfile to %s; create explicitly with ollama create %s -f %s",
                    modelfile, config["ollama_model_name"], modelfile)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser; commands are inert until explicitly invoked."""
    parser = argparse.ArgumentParser(prog="python -m src.cli",
                                     description="Modular local LLM post-training pipeline")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-data")
    prepare.add_argument("--config", default="configs/datasets.yaml")
    prepare.add_argument("--group", choices=["sft", "preferences", "grpo", "evaluation"], default="sft")
    download = sub.add_parser("download-model", help="download a configured model from Hugging Face")
    download.add_argument("--model", choices=["deepseek_r1_distill_qwen_7b", "qwen2_5_coder_7b"],
                          required=True, help="model key from configs/models.yaml")
    for command, default in [("train-sft", "configs/sft.yaml"),
                             ("train-dpo", "configs/dpo.yaml"),
                             ("train-grpo", "configs/grpo_math.yaml"),
                             ("evaluate", "configs/evaluation.yaml"),
                             ("merge-adapter", "configs/export.yaml"),
                             ("export-gguf", "configs/export.yaml"),
                             ("export-ollama", "configs/export.yaml")]:
        child = sub.add_parser(command)
        child.add_argument("--config", default=None if command == "train-grpo" else default)
        if command == "train-sft":
            child.add_argument("--dataset-group", default=None,
                               help="override dataset group (for separate source/mixture runs)")
        if command == "train-grpo":
            child.add_argument("--task", choices=["math", "code"], default="math")
        if command == "export-ollama":
            child.add_argument("--create", action="store_true",
                               help="also invoke `ollama create` after writing the Modelfile")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI dispatcher."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.command == "prepare-data":
        from src.data.prepare import prepare_dataset
        config = load_yaml(args.config)
        validate_split(config.get("split", {"train": 1.0}))
        report = prepare_dataset(config, args.group, str(Path("data/processed") / args.group))
        logger.info("Preprocessing report: %s", report)
    elif args.command == "download-model":
        _download_model(args.model)
    elif args.command == "train-sft":
        _train_sft(args.config, args.dataset_group)
    elif args.command == "train-dpo":
        _train_preference(args.config)
    elif args.command == "train-grpo":
        config_path = args.config or ("configs/grpo_math.yaml" if args.task == "math"
                                      else "configs/grpo_code.yaml")
        _train_grpo(config_path)
    elif args.command == "evaluate":
        _evaluate(args.config)
    elif args.command in {"merge-adapter", "export-gguf", "export-ollama"}:
        # Export is a complete chained workflow; explicit CLI invocation is required.
        if args.command == "merge-adapter":
            from src.export.convert import merge_adapter
            config = load_yaml(args.config)
            merge_adapter(config["base_model_path"], config["adapter_path"], config["merged_output_path"])
        elif args.command == "export-gguf":
            from src.export.convert import export_gguf
            config = load_yaml(args.config)
            export_gguf(config["llama_cpp_path"], config["merged_output_path"],
                        config["gguf_output_path"], config["quantized_gguf_path"],
                        config["gguf_quantization"])
        else:
            _export(args.config, args.create)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
