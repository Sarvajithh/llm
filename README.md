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
```

Install a PyTorch build matching your CUDA/ROCm/CPU environment by following the
official PyTorch instructions. `bitsandbytes` support depends on the platform.
The application does not install dependencies automatically.

## Configuration

1. Copy `.env.example` to `.env` only if you need a credential-backed optional
   integration. Replace placeholders locally; never commit `.env`.
2. Set model identifiers and local model directories in `configs/models.yaml`.
   The expected model roles are DeepSeek-R1-Distill-Qwen-7B for reasoning/math
   and Qwen2.5-Coder-7B for coding.
3. Set dataset sources, format, schemas and splits in `configs/datasets.yaml`.
4. Tune the workflow-specific batch sizes, sequence limits, learning rates,
   checkpoint locations, LoRA rank and quantization in the respective YAML.
5. Set export paths and the Ollama name in `configs/export.yaml`.

The YAML files intentionally contain placeholders, not usable user-specific
paths. Training expects model files to already exist locally. Hugging Face
dataset IDs are supported only when the user explicitly runs preparation.

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
python -m src.cli prepare-data --config configs/datasets.yaml --group evaluation
```

Dataset entries use their configured `weight` values during preparation. The
seeded weighted sampler is deterministic and samples with replacement; the
sample count equals the total size of all non-empty input datasets before
deduplication.

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
python -m src.cli prepare-data --config configs/datasets.yaml --group evaluation
python -m src.cli evaluate --config configs/evaluation.yaml
```

For fair base/fine-tuned comparison, point the model config at each local model
in turn and preserve each report under a distinct output directory.

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

After setting all placeholders and installing compatible dependencies, use the
workflow commands above. Training, evaluation, export, model creation, and any
remote dataset access are explicit user actions; this repository does not start
them automatically.
