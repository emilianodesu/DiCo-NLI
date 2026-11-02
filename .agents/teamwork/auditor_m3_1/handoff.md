# Forensic Audit Report: Milestone 3 (CLI Training Pipeline & GPU Scripts)

**Auditor:** Forensic Auditor M3 (`auditor_m3_1`)  
**Target:** Milestone 3 (`src/training/`, `experiments/train.py`, `scripts/run_gpu_training.sh`, `tests/test_training.py`)  
**Integrity Mode:** Development (per `ORIGINAL_REQUEST.md:8`)  
**Date:** 2026-11-02T02:35:00Z  

---

## Forensic Audit Report

**Work Product**: `src/training/`, `experiments/train.py`, `scripts/run_gpu_training.sh`, `tests/test_training.py`  
**Profile**: General Project (Integrity Forensics)  
**Verdict**: **CLEAN**  

### Phase Results
- **Hardcoded Output Detection**: **PASS** — Source code contains no canned test results, hardcoded metrics, or predetermined predictions.
- **Facade Detection**: **PASS** — `Trainer`, `CrossEncoderDataset`, and `experiments/train.py` genuinely execute training steps, tokenization, device allocation, and metric evaluations.
- **Pre-populated Artifact Detection**: **PASS** — Identified artifacts in `logs/` and `results/` match verifiable outputs generated dynamically during runtime execution.
- **Behavioral Verification (Build & Test)**: **PASS** — Full test suite executed with 104/104 tests passing in 0.862s; bash orchestrator quick-test runs 8 matrix experiments in 1s.
- **Output & Format Verification**: **PASS** — All generated submission files strictly comply with Codabench two-column specifications (`instance_id,label`) and pass programmatic validation.
- **Execution Delegation / Dependency Audit**: **PASS** — No unauthorized external solvers or mock circumventions. Fast mock pipeline is fully self-contained and computes authentic metrics.

---

## 1. Observation

Direct empirical observations from source analysis, tool runs, and stress-tests:

1. **Source Code Inspection**:
   - `src/training/__init__.py`: Clean exports of `Trainer`, `TrainerConfig`, `TrainingArguments`, `CrossEncoderDataset`, and utility functions (`set_seed`, `get_device`, `get_device_info`, `can_use_amp`, `get_autocast_dtype`, `get_autocast_context`, `JsonlLogger`).
   - `src/training/utils.py`: Deterministic seed setting across Python/NumPy/PyTorch (lines 36-66); device probe hierarchy (CUDA -> MPS -> CPU) with `torch.cuda.get_device_properties` (lines 68-143); AMP context management (lines 176-200); and append-only thread-safe JSONL logger (lines 202-231).
   - `src/training/trainer.py`: Dual-mode execution engine. `_train_pytorch` (lines 345-492) implements genuine AdamW optimization with differential weight decay, linear warmup scheduler, mixed precision with `torch.amp.GradScaler`, gradient clipping, validation evaluation, and PyTorch checkpoint serialization (`best_model.pt`). `_train_mock` (lines 277-343) performs fast deterministic CPU state updates and computes actual validation metrics via `self.evaluate()`.
   - `src/training/trainer.py:494-515`: Contains a preliminary `evaluate` method signature that was superseded by `def _predict_pair_indices` (line 516) and the full `evaluate` implementation at line 539. In Python class dictionaries, the second definition cleanly overwrites the first without adverse side effects.
   - `experiments/train.py`: Unified executable CLI (`chmod +x`, 460 lines). Parses flags (`--track`, `--method`, `--model`, `--epochs`, `--batch_size`, `--lr`, `--fp16`, `--bf16`, `--test_run`, `--predictions_file`, etc.). Dispatches BSA augmentation via `build_reversal_dataset`, Siamese pairing via `pair_instances`, and SD wrapping via `build_model`. Validates Codabench submission files programmatically on line 414.
   - `scripts/run_gpu_training.sh`: POSIX/Bash 3.2+ master shell orchestrator (`chmod +x`, 626 lines). Implements `set -euo pipefail`, hardware probe (`PROBE_RESULT`, line 301), 8-experiment benchmark matrix (lines 358-367), subprocess execution of `experiments/train.py`, Codabench format verification (lines 485-502), and results aggregation (`logs/matrix_summary.md` and `results/summary.json`).

2. **Automated Test Suite Execution**:
   - Command:
     ```bash
     python3 -m unittest discover -s tests -p "test_*.py" -v
     ```
   - Result:
     ```
     Ran 104 tests in 0.862s
     OK
     ```
   - All 21 tests in `tests/test_training.py` passed (testing seed reproducibility, device resolution, JSONL logging, Trainer mock training, checkpointing, metric evaluation, SD 100% SoftCons guarantee, Codabench export, CLI invocation, bash syntax, dry-run, and quick-test).

3. **Bash Orchestrator Validation**:
   - `bash -n scripts/run_gpu_training.sh`: Exit code 0 (clean syntax).
   - `bash scripts/run_gpu_training.sh --dry-run`: Exit code 0 (valid argument construction).
   - `bash scripts/run_gpu_training.sh --quick-test`:
     - Executed all 8 benchmark runs across Track 1 and Track 2 in `00m01s` on CPU.
     - Symmetrized Decoding produced `SoftCons = 1.000000` on both Track 1 (`t1_sd_deberta`) and Track 2 (`t2_sd_beto`).
     - Verified all 8 prediction CSVs via `src.data.codabench.validate_codabench_submission`: 100% PASSED.
     - Generated `logs/matrix_summary.md`, `results/summary.json`, and `results/codabench_submissions.zip`.

4. **Adversarial Stress-Testing**:
   - Invalid track (`python3 experiments/train.py --track 3 --test_run`): Gracefully raised `ValueError: Unsupported track identifier '3'. Must be track 1 (English) or 2 (Spanish).` with exit code 1.
   - Invalid method (`python3 experiments/train.py --method magic --test_run`): Argument parser rejected input immediately with exit code 2.
   - Mixed precision on CPU (`python3 experiments/train.py --test_run --fp16 --device cpu`): Handled gracefully via `can_use_amp("cpu") == False` without crashing.
   - Nested output directories (`--predictions_file tmp_test/deep/sub/dir/preds.csv`): Created intermediate directories automatically and exported valid Codabench file.

---

## 2. Logic Chain

1. **Integrity Mode Derivation**:
   - `ORIGINAL_REQUEST.md` specifies `Integrity mode: development`. Under Development Mode, standard library and framework usage are permitted; prohibited patterns are hardcoded test results, facade implementations, and fabricated outputs.
2. **Evaluation of Hardcoded Outputs**:
   - Inspection of `trainer.py` lines 539-595 reveals that `evaluate()` iterates through all reciprocal pairs, calls `_predict_pair_indices` or `predict()`, collects predictions, and passes them to `compute_official_metrics()`. At no point are metric values hardcoded or intercepted.
3. **Evaluation of Facade Implementations**:
   - Both PyTorch and CPU mock paths are functional. The PyTorch path contains the complete loss calculation, optimizer step, gradient clipping, AMP scaling, and checkpointing. The mock path executes model state steps and runs the full evaluation pipeline, generating authentic Codabench submission files.
4. **Authenticity of Verification Outputs**:
   - Pre-existing files in `results/` and `logs/` were matched against freshly regenerated runs via `scripts/run_gpu_training.sh --quick-test`. Fresh execution produced identical valid output structures in real-time.
5. **Deduction**:
   - Since no prohibited patterns exist, all 104 tests pass independently, and behavioral verification confirms strict adherence to R3 requirements, the work product is authentic.

---

## 3. Caveats

- **Redundant Method Stub**: In `src/training/trainer.py` (lines 494-515), a partial `evaluate()` method stub exists above `_predict_pair_indices` and is immediately shadowed by the complete `evaluate()` method at line 539. This does not impair execution or test passing, but represents dead code that should be cleaned up during general maintenance.
- **Physical GPU Testing**: Remote CUDA execution was verified via static code analysis of the AMP/GradScaler loop, PyTorch CUDA capability checks, and dry-run CLI argument parsing; actual NVIDIA hardware execution depends on the user's remote cluster environment.

---

## 4. Conclusion

**Verdict: CLEAN**

Milestone 3 deliverables (`src/training/`, `experiments/train.py`, `scripts/run_gpu_training.sh`, and `tests/test_training.py`) fully satisfy Requirement §R3 and Acceptance Criteria without any integrity violations, facade implementations, or hardcoded shortcuts.

---

## 5. Verification Method

To reproduce the findings:

1. **Execute Complete Test Suite**:
   ```bash
   python3 -m unittest discover -s tests -p "test_*.py" -v
   ```
   *Expected*: 104 tests pass in < 2 seconds with exit code 0.

2. **Run GPU Orchestrator Quick-Test**:
   ```bash
   bash scripts/run_gpu_training.sh --quick-test
   ```
   *Expected*: 8/8 benchmark experiments complete with status `SUCCESS` and Codabench validation `PASSED` in ~1-2 seconds.

3. **Verify Adversarial CLI Resilience**:
   ```bash
   python3 experiments/train.py --test_run --fp16 --device cpu
   python3 experiments/train.py --track 2 --method crl_sd --test_run
   ```
   *Expected*: Both commands exit with code 0 and output valid submission files.

4. **Invalidation Conditions**:
   - Any failure in the 104-test suite.
   - Non-zero exit code during `bash scripts/run_gpu_training.sh --quick-test`.
   - Any generated submission failing `validate_codabench_submission`.
