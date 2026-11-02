# Dispatch: Worker Milestone 3 (CLI Training Pipeline & GPU Scripts)

## Mission
Implement the complete CLI training and execution infrastructure for SemEval-2027 Task 2 (DiCo-NLI):
1. `src/training/__init__.py`
2. `src/training/utils.py`
3. `src/training/trainer.py`
4. `experiments/train.py`
5. `scripts/run_gpu_training.sh` (make executable: `chmod +x`)
6. Minor AMP dtype fix in `src/models/losses.py` (line 263: cast `self.P` to `probs.dtype`)
7. Comprehensive unit and integration test suite in `tests/test_training.py`

## Mandatory Reference Documents
- Read `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/ORIGINAL_REQUEST.md` (Authoritative user requirements §R3)
- Read `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/PROJECT.md`
- Read `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m3_1/handoff.md` (Blueprint for `experiments/train.py`)
- Read `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m3_2/handoff.md` (Blueprint for `src/training/trainer.py` and `src/training/utils.py`)
- Read `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m3_3/handoff.md` (Blueprint for `scripts/run_gpu_training.sh`)

## MANDATORY INTEGRITY WARNING
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

## Detailed Requirements

### 1. `src/training/utils.py`
- `set_seed(seed: int = 42, deterministic: bool = True)`: seeds `random`, `os.environ["PYTHONHASHSEED"]`, `np.random`, and `torch` (CPU, CUDA, MPS) if available.
- `get_device(preferred: Optional[str] = None)`: detects and returns optimal compute device (`cuda`, `mps`, `cpu`).
- `get_device_info(device=None) -> Dict[str, Any]`: returns hardware diagnostics (device type, VRAM, capability, bf16 support).
- `can_use_amp(device=None) -> bool`: returns True if device supports AMP.
- `get_autocast_dtype(fp16: bool = False, bf16: bool = False)`: resolves `torch.bfloat16`, `torch.float16`, or None.
- `get_autocast_context(device=None, enabled=True, dtype=None)`: context manager for `torch.amp.autocast` (or no-op fallback).
- `JsonlLogger(filepath: str)`: appends structured JSON dictionaries to a `.jsonl` file.

### 2. `src/training/trainer.py`
- Dual execution engine:
  - Real PyTorch mode when `HAS_TORCH=True` and model is not mock: AdamW with decoupled weight decay (exclude bias and LayerNorm/RMSNorm), linear warmup scheduler, AMP `torch.amp.autocast` with `GradScaler` (unscaling before gradient clipping `torch.nn.utils.clip_grad_norm_`), evaluation after each epoch on validation set, saving `best_model.pt` based on validation weighted F1, saving `checkpoint_last.pt`, structured JSONL metrics.
  - Fast CPU Mock mode when `is_mock=True` or `not HAS_TORCH`: simulates training across epochs in pure Python (<0.5s execution), updates mock engine, evaluates official metrics, outputs identical checkpoint metadata and JSONL logs.
- Support both Cross-Encoder dataset (`baseline`, `bsa`, `sd`) and Paired Siamese CRL dataset (`crl`, `crl_sd`).
- Predict and evaluate methods: computes official `Weighted F1`, `SoftCons`, `HardCons`.

### 3. `src/training/__init__.py`
- Export `Trainer`, `set_seed`, `get_device`, `get_device_info`, `can_use_amp`, `JsonlLogger`, etc.

### 4. `experiments/train.py`
- Standalone executable CLI script (`#!/usr/bin/env python3`).
- CLI flags:
  - `--track` (`1`, `2`, `en`, `es`)
  - `--model` (defaults: `microsoft/deberta-v3-base` for Track 1, `dccuchile/bert-base-spanish-wwm-cased` for Track 2, `mock` if `--test_run`)
  - `--method` (`baseline`, `bsa`, `crl`, `sd`, `crl_sd`)
  - `--batch_size`, `--learning_rate` / `--lr`, `--epochs`, `--warmup_ratio`, `--weight_decay`, `--seed`
  - `--fp16`, `--bf16`, `--device`
  - `--eval_only`, `--predict_only`, `--test_run` / `--fast_dev_run`
  - `--output_dir`, `--checkpoint_dir`, `--log_dir`, `--predictions_dir`, `--checkpoint_path`
- Complete pipeline: parse args -> set seed -> load data (`load_phrasis_split`) -> build model, tokenizer, loss (`build_model`, `build_tokenizer`, `build_loss`) -> train with `Trainer` -> evaluate on dev -> export official Codabench submission file (`results/<run_name>_submission.csv`) -> validate with `validate_codabench_submission` -> save summary JSON in `logs/` -> print formatted summary.

### 5. `scripts/run_gpu_training.sh`
- POSIX/Bash 3.2+ compatible shell script. Make executable (`chmod +x scripts/run_gpu_training.sh`).
- Strict bash safety (`set -euo pipefail`).
- Remote NVIDIA GPU environment setup (`PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`, `TOKENIZERS_PARALLELISM="false"`, `PYTHONUNBUFFERED=1`).
- Hardware and dependency probe.
- Creates `checkpoints/`, `logs/`, `results/`, `data/`.
- 8-experiment benchmark matrix (Track 1 and Track 2 for all 4 modeling families).
- Configurable flags (`--track`, `--method`, `--quick-test`, `--parallel`, `--dry-run`, `--batch-size`, `--epochs`, `--lr`, `--fp16`, `--bf16`, `--gpu`, `--allow-cpu`).
- Codabench validation check on generated CSV submissions.
- Print summary markdown table of results.

### 6. Minor AMP Fix in `src/models/losses.py`
- In `TranspositionOperator.forward`: cast `self.P` to `probs.dtype` to avoid FP16/FP32 mismatch under AMP:
  `return torch.matmul(probs, self.P.to(dtype=probs.dtype))`

### 7. Comprehensive Test Suite `tests/test_training.py`
- Unit tests for `utils.py` (seeding, devices, AMP detection, JsonlLogger).
- Unit tests for `trainer.py` (mock training, evaluation, checkpointing, metric logging).
- Integration test for `experiments/train.py` (CLI `--test_run` execution for both tracks and all methods, verifying exit code 0, checkpoint creation, and Codabench CSV creation/validation).
- Integration test for `scripts/run_gpu_training.sh` (`bash -n` syntax validation, `--dry-run`, and `--quick-test` execution).
- Ensure all 83 existing tests + new training tests pass in <60 seconds on CPU.

## Completion Criteria
- Run full test suite: `python3 -m unittest discover -s tests -p "test_*.py" -v`
- Verify 100% test pass rate with zero errors and runtime <60s on CPU.
- Write detailed handoff report to `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/worker_m3/handoff.md`.
- Notify orchestrator via `send_message`.


## 2026-11-02T02:20:42Z
You are the Worker for Milestone 3 (CLI Training Pipeline & GPU Scripts).
Your working directory is: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/worker_m3/
Please read your detailed assignment in: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/worker_m3/DISPATCH.md
Read the authoritative requirements in: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/ORIGINAL_REQUEST.md
Read the project architecture in: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/PROJECT.md
Read the detailed technical blueprints in:
- /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m3_1/handoff.md
- /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m3_2/handoff.md
- /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m3_3/handoff.md

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Implement:
1. `src/training/__init__.py`
2. `src/training/utils.py`
3. `src/training/trainer.py`
4. `experiments/train.py` (chmod +x)
5. `scripts/run_gpu_training.sh` (chmod +x)
6. Minor AMP fix in `src/models/losses.py` (cast `self.P` to `probs.dtype` in `TranspositionOperator.forward`)
7. Comprehensive test suite in `tests/test_training.py`

Run the test suite and verify 100% pass rate in <60 seconds on CPU.
Write your complete handoff report to /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/worker_m3/handoff.md and notify orchestrator via send_message when done.
