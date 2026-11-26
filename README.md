# Local LLM Training Pipeline

Modular, configuration-driven post-training workflows for a reasoning model
(DeepSeek-R1-Distill-Qwen-7B) and a coding model (Qwen2.5-Coder-7B). The code
supports response-level SFT/distillation, DPO, ORPO, GRPO rewards, local data
preparation, evaluation, adapter merging, GGUF conversion and Ollama packaging.
Model weights and datasets are never bundled here.

## Architecture

```text
configs/  -> validated YAML inputs
src/data/ -> local JSONL/Parquet and explicitly requested Hugging Face loading,
             normalization, filtering, deduplication, splitting, reports
src/training/ -> SFT, preference optimization (DPO/ORPO/SimPO selection), GRPO
src/rewards/ -> numeric/symbolic checks, constraints, isolated sandbox interface
src/evaluation/ -> shared generator contract and JSONL metrics/reports
src/export/ -> PEFT merge, llama.cpp conversion/quantization, Ollama Modelfile
```

The training loader sets `local_files_only=True`; it will not fetch model files.
Dataset loading from a Hugging Face identifier is an explicit `prepare-data`
operation and may access the network when the user later runs it. All file paths
are relative to the current working directory unless configured otherwise.

## Installation

Create a Python 3.10+ environment and install only the extras needed for the
workflow you intend to run. These commands are for the user to run manually:

```bash
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -e '.[data,training,dev]'
# Optional symbolic algebra rewards:
python -m pip install -e '.[math]'
# Optional standalone model download command:
python -m pip install -e '.[model-download]'
```

Install a PyTorch build matching your CUDA/ROCm/CPU environment by following the
official PyTorch instructions. `bitsandbytes` support depends on the platform.
The application does not install dependencies automatically.

## Configuration

1. Copy `.env.example` to `.env` only if you need a credential-backed optional
   integration. Replace placeholders locally; never commit `.env`.
2. The configured math model is DeepSeek-R1-Distill-Qwen-7B. The coder model
   is Qwen2.5-Coder-7B-Instruct. Place model files in the configured `models/`
   directories or cache those exact Hugging Face model IDs before training.
3. The selected math sources are Nemotron-Math-v2 plus the math domain from
   Superior-Reasoning Stage 1 for SFT, PRM800K solution preferences for DPO,
   and Nemotron problem/answer rows for GRPO. Preparation is streamed and
   capped by `configs/datasets.yaml`.
4. Tune the workflow-specific batch sizes, sequence limits, learning rates,
   checkpoint locations, LoRA rank and quantization in the respective YAML.
5. Set export paths and the Ollama name in `configs/export.yaml`.

Training uses local model files or existing Hugging Face cache entries; it
does not download model weights implicitly. Hugging Face dataset loading is
explicitly started by `prepare-data` and requires network access.

To explicitly download a configured model to its `local_path`, run one of:

```bash
python -m src.cli download-model --model deepseek_r1_distill_qwen_7b
python -m src.cli download-model --model qwen2_5_coder_7b
```

The downloads are large and require Hugging Face network access. If a model
requires authentication, log in with the Hugging Face CLI before downloading.

## Dataset preparation

Supported local formats are JSONL and Parquet. Hugging Face Dataset identifiers
are supported as a configured source. Canonical schemas are `prompt_response`,
`messages`, `prompt_chosen_rejected`, and `prompt_reference`; `auto` detects
common record shapes. Preparation filters malformed/length-constrained rows,
deduplicates exact normalized records, shuffles with a fixed seed, writes train,
validation and test JSONL files under `data/processed/`, and produces a JSON
summary.

```bash
python -m src.cli prepare-data --config configs/datasets.yaml --group sft
python -m src.cli prepare-data --config configs/datasets.yaml --group preferences
python -m src.cli prepare-data --config configs/datasets.yaml --group grpo
```

Dataset entries use their configured weights during deterministic sampling
without replacement. The held-out math evaluation split is generated from the
SFT group, with all rows for the same normalized problem kept in one split.

## SFT / response-level distillation

Prepare prompt/response examples with teacher responses already present, then
configure the model and training values in `configs/sft.yaml`:

```bash
python -m src.cli train-sft --config configs/sft.yaml
```

SFT uses TRL and PEFT; `quantization: "4bit"` prepares a QLoRA run, while
`lora.enabled: false` selects full fine-tuning. Set `resume_from_checkpoint`
to an existing checkpoint path to resume.

## DPO, ORPO and SimPO

Preference rows use `prompt`, `chosen` and `rejected` fields. Select
`algorithm: dpo`, `orpo`, or `simpo` in `configs/dpo.yaml` and run:

```bash
python -m src.cli train-dpo --config configs/dpo.yaml
```

The preference module delegates DPO and ORPO to their TRL trainers. SimPO uses
a small custom Transformers trainer with a reference-free, length-normalized
chosen-versus-rejected log-probability objective and configurable target margin
(`gamma_beta_ratio`). Pin compatible framework versions for reproducibility.

## GRPO

Math rewards include numeric answer matching, optional SymPy equivalence, and
constraint checks. Coding rewards use the `IsolatedSandbox` protocol. Supply a
backend that runs in a dedicated, resource-limited environment with no network
and no host credentials; there is intentionally no local subprocess fallback.
Configure the backend integration in your deployment before enabling code
rewards.

```bash
python -m src.cli train-grpo --task math --config configs/grpo_math.yaml
python -m src.cli train-grpo --task code --config configs/grpo_code.yaml
```

For math GRPO, make sure source records include the answer fields expected by
the selected reward. For code GRPO, map unit tests and any sandbox context into
the reward call's batch fields. The generic reward registry is a starting point;
adapt the record/reward field mapping to your dataset schema.

## Evaluation

Evaluation uses a common generator interface, writes prediction JSONL, and
reports exact match, pass@1, and configured empirical pass@k metrics. It loops
through `models_to_compare` and writes a comparison JSON report. Extend
`src/evaluation/runner.py` for benchmark-specific parsers. No benchmark is
downloaded automatically.

```bash
python -m src.cli evaluate --config configs/evaluation.yaml
```

`configs/evaluation.yaml` compares the base model and each trained adapter on
the same held-out math prompts. Adapters that have not been trained yet are
reported as skipped, so evaluation can be run before training and after each
stage. The math metric extracts final answers and compares numeric answers with
a small tolerance; other answers use normalized exact match.

## Selected math run

Prepare all three groups once, then run baseline evaluation and the stages in
order. DPO loads the saved SFT adapter; GRPO loads the saved DPO adapter.

```bash
python -m src.cli prepare-data --group sft
python -m src.cli prepare-data --group preferences
python -m src.cli prepare-data --group grpo
python -m src.cli evaluate --config configs/evaluation.yaml
python -m src.cli train-sft --config configs/sft.yaml
python -m src.cli evaluate --config configs/evaluation.yaml
python -m src.cli train-dpo --config configs/dpo.yaml
python -m src.cli evaluate --config configs/evaluation.yaml
python -m src.cli train-grpo --task math --config configs/grpo_math.yaml
python -m src.cli evaluate --config configs/evaluation.yaml
```

## GGUF export

Install/build llama.cpp separately and set `llama_cpp_path`. The CLI chains
adapter merging, Hugging Face saving, FP16 GGUF conversion and quantization:

```bash
python -m src.cli merge-adapter --config configs/export.yaml
python -m src.cli export-gguf --config configs/export.yaml
```

`merge-adapter` performs just the PEFT merge. `export-gguf` expects the merged
model directory to exist. `export-ollama` runs the full export chain and writes
an Ollama Modelfile.

## Ollama deployment

Review the generated Modelfile and GGUF path first. The default command writes
the file but does not create an Ollama model. Add `--create` only when you
explicitly want the CLI to invoke `ollama create`:

```bash
python -m src.cli export-ollama --config configs/export.yaml
python -m src.cli export-ollama --config configs/export.yaml --create
# Or manually after reviewing the file:
ollama create YOUR_OLLAMA_MODEL_NAME -f outputs/ollama/Modelfile
```

## Hardware limitations

7B models require substantial memory, especially for full fine-tuning, long
sequences, or multiple GRPO generations. QLoRA lowers weight memory but still
needs memory for activations, optimizer state, and temporary buffers. Begin
with short sequences and small per-device batches, then increase accumulation
or use multi-GPU Accelerate after validating your hardware. `src.training.common`
logs CUDA allocated/reserved memory when PyTorch is present. Actual capacity
depends on the model revision, context length, kernels, and GPU.

## Tests

Tests are included for normalization, configuration, rewards, mixture sampling,
export configuration, mocked training interfaces and an offline preparation
integration path. Run only in an environment with the optional test/data
dependencies installed:

```bash
python -m pip install -e '.[dev,data]'
python -m pytest
```

The integration test uses a temporary local JSONL file. No test is run as part
of this implementation task. Tests requiring GPU, external models/data, APIs,
or code-execution services should be added as separately marked opt-in tests.

## Commands to run manually

After placing model files in the configured directories and installing the
matching dependencies, use the workflow commands above. Training, evaluation,
export, model creation, and remote dataset access are explicit user actions;
the repository does not start them automatically.
