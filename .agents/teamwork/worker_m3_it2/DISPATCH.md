# Dispatch: Worker Milestone 3 Iteration 2 (Remediation)

## Mission
Apply the remediation blueprints created by Explorers 1, 2, and 3 to resolve all defects identified during Milestone 3 Iteration 1 review and stress testing.

## Reference Blueprints & Patch Files
- Read `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/ORIGINAL_REQUEST.md` (Authoritative user requirements)
- Read `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m3_it2_1/handoff.md` (Blueprint for `src/training/trainer.py`)
- Read `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m3_it2_2/handoff.md` (Blueprint for `src/models/factory.py`)
- Read `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m3_it2_3/handoff.md` (Blueprint for `experiments/train.py`, `src/training/utils.py`, `scripts/run_gpu_training.sh`)

## MANDATORY INTEGRITY WARNING
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

## Remediation Tasks

### 1. `src/models/factory.py` (`SymmetrizedModelWrapper`)
- Add transparent delegation methods to `SymmetrizedModelWrapper`:
  - `parameters(self, recurse: bool = True)`
  - `named_parameters(self, prefix: str = "", recurse: bool = True)`
  - `state_dict(self, *args, **kwargs)`
  - `load_state_dict(self, state_dict, strict: bool = True)`
  - `to(self, *args, **kwargs)`
  - `eval(self)` and `train(self, mode: bool = True)`
  - `buffers(self, recurse: bool = True)` and `named_buffers(...)`
  - `zero_grad(self, set_to_none: bool = True)`
  - `save_pretrained(self, save_directory, **kwargs)`
  - `__getattr__(self, name)` fallback to `self.base_model`

### 2. `src/training/trainer.py`
- In `Trainer._prepare_dataset()` and `evaluate()`: detect if data is already a list of `ReciprocalPair` objects (or instantiated `Dataset`), preventing `pair_instances(records)` from receiving `ReciprocalPair` objects and returning empty lists.
- Delete the 22-line orphaned duplicate `def evaluate(...)` stub (lines 494–515) preceding `_predict_pair_indices`.
- In `CrossEncoderDataset.__getitem__`, `evaluate`, and `predict`: sanitize missing, `None`, or `NaN` text fields via `_clean_str()`.
- Implement `_create_grad_scaler()` supporting PyTorch >= 2.3, PyTorch < 2.3, and `_DummyGradScaler` for CPU/mock runs.
- Defensively unwrap `getattr(self.model, "base_model", self.model)` in `_train_pytorch()` and `_save_checkpoint()`.

### 3. `experiments/train.py`
- Validate `--checkpoint_path`: In `--eval_only` or `--predict_only`, verify file exists; if not, raise `FileNotFoundError` or exit 1 with clear error.
- Enforce mutual exclusion: `--fp16` and `--bf16` cannot both be active.

### 4. `src/training/utils.py`
- In `JsonlLogger.log()`: use `json.dumps(record, default=str)`.

### 5. `scripts/run_gpu_training.sh`
- Under `--continue-on-error`: track failure count and exit with non-zero exit code if any experiment failed.
- In benchmark matrix: support `--method crl_sd` or map appropriately.

### 6. Tests in `tests/test_training.py` & `tests/test_models.py`
- Add regression tests covering all remediated edge cases.
- Run full test suite: `python3 -m unittest discover -s tests -p "test_*.py" -v`.
- Verify 100% test pass rate in <60s on CPU.
- Verify `bash scripts/run_gpu_training.sh --quick-test` completes with 100% success.

Write your report to `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/worker_m3_it2/handoff.md` and notify orchestrator via `send_message`.

## 2026-11-02T02:49:20Z
Received dispatch from parent `df6ce377-6407-4a2d-bd18-829b36e20607`:
Apply remediation blueprints to:
1. `src/models/factory.py` (SymmetrizedModelWrapper delegation)
2. `src/training/trainer.py` (ReciprocalPair pre-pairing, dead stub removal, _clean_str, GradScaler compatibility, model unwrapping)
3. `experiments/train.py` (checkpoint path validation in eval_only, fp16/bf16 mutual exclusion)
4. `src/training/utils.py` (JsonlLogger default=str)
5. `scripts/run_gpu_training.sh` (continue-on-error exit code, crl_sd matrix mapping)
6. Add regression tests to `tests/test_training.py` and `tests/test_models.py`
Verify 100% test pass rate in <60s on CPU.
Verify quick-test completes.
Write handoff report to `handoff.md`.
