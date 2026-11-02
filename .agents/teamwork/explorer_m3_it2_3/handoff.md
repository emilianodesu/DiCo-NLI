# Remediation Blueprint & Handoff Report: Explorer 3 (Milestone 3 Iteration 2)

**Author:** Explorer 3 (`explorer_m3_it2_3`)  
**Scope:** CLI Training Engine (`experiments/train.py`), Training Utils (`src/training/utils.py`), and Master GPU Launcher (`scripts/run_gpu_training.sh`)  
**Date:** 2026-11-02T02:52:00Z  
**Recipient:** Orchestrator (`df6ce377-6407-4a2d-bd18-829b36e20607`) / Worker M3  
**Status:** READY FOR IMPLEMENTATION  

---

## 1. Observation

All defects reported in `DISPATCH.md`, `challenger_m3_1/handoff.md`, and `reviewer_m3_2/handoff.md` were directly reproduced and confirmed via local execution:

### Observation 1: Silent Execution on Non-Existent Checkpoint (`experiments/train.py:359`)
- **Command:** `python3 experiments/train.py --eval_only --checkpoint_path /path/to/nonexistent.pt --test_run`
- **Output & Exit Code:** Exited with code `0`.
  ```
  2026-11-01 20:44:27 | INFO | dico_nli.train | Mode: eval_only | Test Run: True
  ...
  2026-11-01 20:44:28 | INFO | dico_nli.train | OFFICIAL EVALUATION METRICS:
  2026-11-01 20:44:28 | INFO | dico_nli.train |   Weighted F1 : 0.320000
  ```
- **Code Inspection (`experiments/train.py:359-363`):**
  ```python
  if args.checkpoint_path is not None and os.path.isfile(args.checkpoint_path):
      logger.info(f"Loading checkpoint weights from: {args.checkpoint_path}")
      if hasattr(model, "load_state_dict") and HAS_TORCH:
          model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
  ```
  `os.path.isfile(args.checkpoint_path)` evaluates to `False`. The branch silently bypasses checkpoint loading, evaluates uninitialized/mock weights, and generates fake submission files without any error or warning.

### Observation 2: Unchecked Conflicting Mixed Precision Flags (`experiments/train.py:222`)
- **Command:** `python3 experiments/train.py --fp16 --bf16 --test_run`
- **Output & Exit Code:** Exited with code `0`.
  ```
  2026-11-01 20:44:31 | INFO | dico_nli.train | Device: auto | Mixed Precision: fp16=True, bf16=True
  ```
- **Code Inspection (`experiments/train.py:142-143, 222-232`):**
  `--fp16` and `--bf16` are independent boolean action flags. Neither `argparse` nor `parse_args()` verifies mutual exclusion. In PyTorch execution, passing both activates `GradScaler("cuda", enabled=True)` alongside `torch.bfloat16`, which is invalid and can degrade training stability.

### Observation 3: `JsonlLogger` Crash on Non-Serializable Types (`src/training/utils.py:217`)
- **Command:**
  ```python
  from src.training.utils import JsonlLogger
  from pathlib import Path
  logger = JsonlLogger(Path('/tmp/test.jsonl'))
  logger.log({'set_val': {1, 2, 3}, 'path_val': Path('/tmp')})
  ```
- **Result:** Crashed with unhandled exception:
  ```
  TypeError: Object of type set is not JSON serializable
  File "src/training/utils.py", line 217, in log:
      f.write(json.dumps(record_with_time) + "\n")
  ```
  `json.dumps` lacks `default=str`, causing sudden failure when metric dictionaries contain sets, Path objects, or tensor references. Furthermore, `JsonlLogger` lacks concurrency locking despite being advertised as "Thread-safe".

### Observation 4: Exit Code Masking under `--continue-on-error` (`scripts/run_gpu_training.sh:466, 624`)
- **Command:** `bash scripts/run_gpu_training.sh --python "/usr/bin/false" --continue-on-error --track 1`
- **Output & Exit Code:** Exited with code `0`.
  ```
  ======================================================================
   Benchmark Matrix Execution Summary
  ======================================================================
  RUN NAME               TRACK    METHOD     STATUS         TIME     W-F1     SOFTCONS   HARDCONS   CODABENCH 
  ----------------------------------------------------------------------------------------------------------
  t1_baseline_deberta    track1   baseline   FAILED         00m00s   -        -          -          SKIPPED   
  t1_bsa_deberta         track1   bsa        FAILED         00m00s   -        -          -          SKIPPED   
  t1_crl_deberta         track1   crl        FAILED         00m00s   -        -          -          SKIPPED   
  t1_sd_deberta          track1   sd         FAILED         00m00s   -        -          -          SKIPPED   
  ----------------------------------------------------------------------------------------------------------
  ...
  [SUCCESS] All benchmark runs completed successfully.
  ```
- **Code Inspection (`scripts/run_gpu_training.sh:624`):**
  When `CONTINUE_ON_ERROR=1`, the loop skips aborting on individual failed runs. At line 624, the script unconditionally prints `[SUCCESS] All benchmark runs completed successfully.` and exits with `exit 0`, masking batch failures in CI/CD and remote slurm pipelines.

### Observation 5: Missing `crl_sd` in Shell Benchmark Matrix (`scripts/run_gpu_training.sh:358-386`)
- **Command:** `bash scripts/run_gpu_training.sh --method crl_sd --dry-run`
- **Output:**
  ```
  [WARN] No experiments matched the specified filters: --track all --method crl_sd
  ```
- **Code Inspection (`scripts/run_gpu_training.sh:358-367`):**
  The argument parser permits `--method crl_sd` (lines 112, 226), but `EXPERIMENT_MATRIX` only defined 8 items (`baseline`, `bsa`, `crl`, `sd`), omitting `crl_sd` from both Track 1 and Track 2.

### Observation 6: Reviewer Hardening Points in Shell Script
- **F-M3-05 (Zip artifact pollution):** Line 620 packages `results/*.csv` indiscriminately, archiving stale test submissions.
- **F-M3-06 (Validation output regex):** Line 503 relies on `[[ "$VAL_OUTPUT" == *"PASSED"* ]]`, ignoring Python process exit code.
- **F-M3-02 (Inert parallel flag):** `--parallel` flag sets `PARALLEL_TRACKS=1` without checking GPU count.

---

## 2. Logic Chain

1. **Defensive Checkpoint Validation (Observation 1)**:
   - When a user explicitly specifies `--checkpoint_path <path>`, they assert that weights exist at that location.
   - If `os.path.isfile(args.checkpoint_path)` is false, raising `FileNotFoundError` halts execution immediately.
   - For `eval_only` and `predict_only` modes without `--checkpoint_path` in production runs, emitting a warning alerts users to the uninitialized state.
   - Checkpoints saved by `Trainer._save_checkpoint` store weights under key `"model_state_dict"`; the loader must unwrap this dictionary before calling `model.load_state_dict()`.

2. **Parser-Level Precision Mutual Exclusion (Observation 2)**:
   - PyTorch AMP requires selecting either FP16 or BF16, never both.
   - Validating `if parsed.fp16 and parsed.bf16: parser.error(...)` within `parse_args()` provides standard CLI error behavior (exit code 2) with clear error diagnostics.

3. **Robust Structured Serialization (Observation 3)**:
   - Adding `default=str` to `json.dumps()` in `JsonlLogger.log()` guarantees that any non-native JSON object (Python `set`, `pathlib.Path`, numpy/torch scalars) serializes gracefully.
   - Adding `threading.Lock()` guarantees atomic file operations in multi-threaded data loader environments.

4. **Batch Failure Accounting (Observation 4)**:
   - Introducing `FAILED_COUNT=0` allows tracking failed runs across the execution loop.
   - Even when `--continue-on-error` is active, the script must exit with `exit 1` if `FAILED_COUNT > 0`, logging an explicit failure summary instead of claiming success.

5. **Complete 10-Experiment Matrix Coverage (Observation 5)**:
   - `crl_sd` is supported by `experiments/train.py` and represents the optimal combined mitigation strategy.
   - Adding `track1|crl_sd|...` and `track2|crl_sd|...` to `EXPERIMENT_MATRIX` creates a complete 10-experiment suite (5 methods × 2 tracks), resolving the filter discrepancy.

6. **Shell Pipeline Hardening (Observation 6)**:
   - Tracking `ACTIVE_SUBMISSION_FILES` ensures `codabench_submissions.zip` contains only files produced during the current session.
   - Capturing the exit code of the Python validation command (`VAL_EXIT`) prevents false positives from substring matching.

---

## 3. Caveats

- **Mock Checkpoints in Unit Tests**: Mock checkpoints written by `_save_mock_checkpoint()` contain JSON metadata rather than pickled PyTorch state dicts. The checkpoint loader must handle mock mode gracefully by verifying file existence without crashing on `torch.load()`.
- **Parallel Multi-GPU Execution**: Running experiments in parallel requires multiple discrete CUDA devices with sufficient VRAM. The blueprint validates `DEV_COUNT >= 2` before enabling parallel tracks, safely falling back to sequential execution when unavailable.

---

## 4. Conclusion & Actionable Blueprints

The exact modifications required for Worker M3 are detailed below.

### Blueprint A: `experiments/train.py`

#### Edit A1: Precision Mutual Exclusion in `parse_args()`
**File:** `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/experiments/train.py`  
**Target Range:** Lines 222-233  

**Original Content:**
```python
    # Normalize precision flag
    if parsed.precision == "fp16":
        parsed.fp16 = True
        parsed.bf16 = False
    elif parsed.precision == "bf16":
        parsed.bf16 = True
        parsed.fp16 = False
    elif parsed.precision == "fp32":
        parsed.fp16 = False
        parsed.bf16 = False

    return parsed
```

**Replacement Content:**
```python
    # Validate mutual exclusivity of precision flags
    if parsed.fp16 and parsed.bf16:
        parser.error("Cannot specify both --fp16 and --bf16. Please select one.")

    # Normalize precision flag if specified
    if parsed.precision == "fp16":
        if parsed.bf16:
            parser.error("Cannot specify both --fp16 and --bf16. Please select one.")
        parsed.fp16 = True
        parsed.bf16 = False
    elif parsed.precision == "bf16":
        if parsed.fp16:
            parser.error("Cannot specify both --fp16 and --bf16. Please select one.")
        parsed.bf16 = True
        parsed.fp16 = False
    elif parsed.precision == "fp32":
        if parsed.fp16 or parsed.bf16:
            parser.error("Cannot specify --fp16 or --bf16 when --precision fp32 is set.")
        parsed.fp16 = False
        parsed.bf16 = False

    return parsed
```

---

#### Edit A2: Checkpoint Validation and Unwrapping
**File:** `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/experiments/train.py`  
**Target Range:** Lines 358-363  

**Original Content:**
```python
    # Load existing checkpoint if requested
    if args.checkpoint_path is not None and os.path.isfile(args.checkpoint_path):
        logger.info(f"Loading checkpoint weights from: {args.checkpoint_path}")
        if hasattr(model, "load_state_dict") and HAS_TORCH:
            model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
```

**Replacement Content:**
```python
    # Load existing checkpoint if requested
    if args.checkpoint_path is not None:
        if not os.path.isfile(args.checkpoint_path):
            error_msg = f"Checkpoint file not found: {args.checkpoint_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        logger.info(f"Loading checkpoint weights from: {args.checkpoint_path}")
        if hasattr(model, "load_state_dict") and HAS_TORCH:
            try:
                state_dict = torch.load(args.checkpoint_path, map_location=device)
                if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                    model.load_state_dict(state_dict["model_state_dict"])
                elif isinstance(state_dict, dict):
                    model.load_state_dict(state_dict)
                logger.info(f"Successfully loaded checkpoint weights from {args.checkpoint_path}")
            except Exception as exc:
                if is_mock:
                    logger.info(f"Mock execution mode: validated mock checkpoint at {args.checkpoint_path}")
                else:
                    logger.error(f"Failed to load checkpoint weights: {exc}")
                    raise
        elif is_mock:
            logger.info(f"Mock execution mode: validated mock checkpoint at {args.checkpoint_path}")
    elif args.mode in ["eval_only", "predict_only"] and not args.test_run and not is_mock:
        logger.warning(
            "Running in '%s' mode without --checkpoint_path. Evaluating initial model weights.",
            args.mode,
        )
```

---

### Blueprint B: `src/training/utils.py`

#### Edit B1: Concurrency and Serialization Hardening in `JsonlLogger`
**File:** `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/src/training/utils.py`  
**Target Range:** Lines 10-18 (imports) and 202-219 (`JsonlLogger`)  

**Original Imports (Lines 10-18):**
```python
import contextlib
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Union
```

**Replacement Imports:**
```python
import contextlib
import json
import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Union
```

**Original `JsonlLogger` (Lines 202-219):**
```python
class JsonlLogger:
    """
    Thread-safe append-only structured JSONL metrics logger.
    """

    def __init__(self, filepath: Union[str, Path]) -> None:
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        """Appends a dictionary record as a JSON line and flushes to disk."""
        record_with_time = dict(record)
        if "timestamp" not in record_with_time:
            record_with_time["timestamp"] = time.time()
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record_with_time) + "\n")
            f.flush()
```

**Replacement `JsonlLogger`:**
```python
class JsonlLogger:
    """
    Thread-safe append-only structured JSONL metrics logger.
    """

    def __init__(self, filepath: Union[str, Path]) -> None:
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, record: Dict[str, Any]) -> None:
        """Appends a dictionary record as a JSON line and flushes to disk."""
        record_with_time = dict(record)
        if "timestamp" not in record_with_time:
            record_with_time["timestamp"] = time.time()
        serialized = json.dumps(record_with_time, default=str) + "\n"
        with self._lock:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(serialized)
                f.flush()
```

---

### Blueprint C: `scripts/run_gpu_training.sh`

#### Edit C1: State Variables Initialization
**File:** `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/scripts/run_gpu_training.sh`  
**Target Range:** Lines 30-34  

**Original Content:**
```bash
# Global Execution Tracking
START_WALL_TIME=$(date +%s)
SUMMARY_RECORDS=()
CLEANUP_PIDS=()
```

**Replacement Content:**
```bash
# Global Execution Tracking
START_WALL_TIME=$(date +%s)
SUMMARY_RECORDS=()
CLEANUP_PIDS=()
FAILED_COUNT=0
ACTIVE_SUBMISSION_FILES=()
```

---

#### Edit C2: Multi-GPU Parallel Check
**File:** `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/scripts/run_gpu_training.sh`  
**Target Range:** After Line 346 (before nvidia-smi probe)  

**Insertion Content:**
```bash
# Check parallel execution capabilities
if [ "$PARALLEL_TRACKS" -eq 1 ]; then
    if [ "$DEV_COUNT" -lt 2 ]; then
        log_warn "Multi-GPU parallel execution requested (--parallel), but only ${DEV_COUNT} device(s) detected."
        log_warn "Falling back to sequential execution queue to prevent resource contention."
        PARALLEL_TRACKS=0
    else
        log_info "Multi-device environment detected (${DEV_COUNT} devices). Parallel tracks enabled."
    fi
fi
```

---

#### Edit C3: Add `crl_sd` to Matrix
**File:** `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/scripts/run_gpu_training.sh`  
**Target Range:** Lines 355-367  

**Original Content:**
```bash
# ------------------------------------------------------------------------------
# 7. Define 8-Experiment Benchmark Matrix
# ------------------------------------------------------------------------------
# Format: TRACK|METHOD|BACKBONE|RUN_NAME
EXPERIMENT_MATRIX=(
    "track1|baseline|${MODEL_T1}|t1_baseline_deberta"
    "track1|bsa|${MODEL_T1}|t1_bsa_deberta"
    "track1|crl|${MODEL_T1}|t1_crl_deberta"
    "track1|sd|${MODEL_T1}|t1_sd_deberta"
    "track2|baseline|${MODEL_T2}|t2_baseline_beto"
    "track2|bsa|${MODEL_T2}|t2_bsa_beto"
    "track2|crl|${MODEL_T2}|t2_crl_beto"
    "track2|sd|${MODEL_T2}|t2_sd_beto"
)
```

**Replacement Content:**
```bash
# ------------------------------------------------------------------------------
# 7. Define Benchmark Matrix (10 Experiments)
# ------------------------------------------------------------------------------
# Format: TRACK|METHOD|BACKBONE|RUN_NAME
EXPERIMENT_MATRIX=(
    "track1|baseline|${MODEL_T1}|t1_baseline_deberta"
    "track1|bsa|${MODEL_T1}|t1_bsa_deberta"
    "track1|crl|${MODEL_T1}|t1_crl_deberta"
    "track1|sd|${MODEL_T1}|t1_sd_deberta"
    "track1|crl_sd|${MODEL_T1}|t1_crl_sd_deberta"
    "track2|baseline|${MODEL_T2}|t2_baseline_beto"
    "track2|bsa|${MODEL_T2}|t2_bsa_beto"
    "track2|crl|${MODEL_T2}|t2_crl_beto"
    "track2|sd|${MODEL_T2}|t2_sd_beto"
    "track2|crl_sd|${MODEL_T2}|t2_crl_sd_beto"
)
```

---

#### Edit C4: Execution Failure Tracking and Strict Codabench Verification
**File:** `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/scripts/run_gpu_training.sh`  
**Target Range:** Lines 466-512  

**Original Content:**
```bash
    if [ $RUN_EXIT_CODE -ne 0 ]; then
        log_error "Training failed for ${exp_name} with exit code ${RUN_EXIT_CODE}."
        RUN_STATUS="FAILED"
        if [ "$CONTINUE_ON_ERROR" -eq 0 ]; then
            log_error "Aborting matrix execution. (Use --continue-on-error to skip failing runs)."
            exit $RUN_EXIT_CODE
        fi
    fi
    
    # Codabench Export Verification Step
    VALID_STATUS="SKIPPED"
    if [ "$RUN_STATUS" = "SUCCESS" ]; then
        log_info "Verifying official Codabench submission file: ${PRED_FILE}"
        if [ ! -f "${PRED_FILE}" ]; then
            log_error "Expected prediction file ${PRED_FILE} was not created!"
            RUN_STATUS="INVALID_OUTPUT"
            VALID_STATUS="MISSING_FILE"
        else
            # Run programmatic validation via src.data.codabench
            VAL_OUTPUT=$(${PYTHON} -c "
import sys
from pathlib import Path
from src.data.codabench import validate_codabench_submission
from src.data.loader import load_phrasis_split

sub_path = '${PRED_FILE}'
track = '${exp_track}'
try:
    df_dev = load_phrasis_split(track, 'dev')
    raw_ids = df_dev['instance_id']
    expected_ids = set(raw_ids.tolist() if hasattr(raw_ids, 'tolist') else raw_ids)
    validate_codabench_submission(sub_path, expected_ids=expected_ids)
    print('PASSED')
except Exception as e:
    print(f'FAILED: {e}')
" 2>&1 || echo "FAILED")

            if [[ "$VAL_OUTPUT" == *"PASSED"* ]]; then
                log_success "Codabench submission format verification: PASSED"
                VALID_STATUS="PASSED"
            else
                log_error "Codabench submission format verification FAILED: ${VAL_OUTPUT}"
                RUN_STATUS="INVALID_OUTPUT"
                VALID_STATUS="FAILED"
            fi
        fi
    fi
```

**Replacement Content:**
```bash
    if [ $RUN_EXIT_CODE -ne 0 ]; then
        log_error "Training failed for ${exp_name} with exit code ${RUN_EXIT_CODE}."
        RUN_STATUS="FAILED"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        if [ "$CONTINUE_ON_ERROR" -eq 0 ]; then
            log_error "Aborting matrix execution. (Use --continue-on-error to skip failing runs)."
            exit $RUN_EXIT_CODE
        fi
    fi
    
    # Codabench Export Verification Step
    VALID_STATUS="SKIPPED"
    if [ "$RUN_STATUS" = "SUCCESS" ]; then
        log_info "Verifying official Codabench submission file: ${PRED_FILE}"
        if [ ! -f "${PRED_FILE}" ]; then
            log_error "Expected prediction file ${PRED_FILE} was not created!"
            RUN_STATUS="INVALID_OUTPUT"
            VALID_STATUS="MISSING_FILE"
            FAILED_COUNT=$((FAILED_COUNT + 1))
            if [ "$CONTINUE_ON_ERROR" -eq 0 ]; then
                log_error "Aborting matrix execution due to missing submission file in ${exp_name}."
                exit 1
            fi
        else
            # Run programmatic validation via src.data.codabench with strict exit code check
            VAL_EXIT=0
            VAL_OUTPUT=$(${PYTHON} -c "
import sys
from pathlib import Path
from src.data.codabench import validate_codabench_submission
from src.data.loader import load_phrasis_split

sub_path = '${PRED_FILE}'
track = '${exp_track}'
try:
    df_dev = load_phrasis_split(track, 'dev')
    raw_ids = df_dev['instance_id']
    expected_ids = set(raw_ids.tolist() if hasattr(raw_ids, 'tolist') else raw_ids)
    validate_codabench_submission(sub_path, expected_ids=expected_ids, raise_on_error=True)
    print('VALIDATION_PASSED')
except Exception as e:
    print(f'VALIDATION_FAILED: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1) || VAL_EXIT=$?

            if [ $VAL_EXIT -eq 0 ] && [[ "$VAL_OUTPUT" == *"VALIDATION_PASSED"* ]]; then
                log_success "Codabench submission format verification: PASSED"
                VALID_STATUS="PASSED"
                ACTIVE_SUBMISSION_FILES+=("${exp_name}_submission.csv")
            else
                log_error "Codabench submission format verification FAILED: ${VAL_OUTPUT}"
                RUN_STATUS="INVALID_OUTPUT"
                VALID_STATUS="FAILED"
                FAILED_COUNT=$((FAILED_COUNT + 1))
                if [ "$CONTINUE_ON_ERROR" -eq 0 ]; then
                    log_error "Aborting matrix execution due to invalid Codabench submission in ${exp_name}."
                    exit 1
                fi
            fi
        fi
    fi
```

---

#### Edit C5: Active Zip Archiving & Exit Status Accounting
**File:** `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/scripts/run_gpu_training.sh`  
**Target Range:** Lines 618-626  

**Original Content:**
```bash
# Optional: Zip all submissions for Codabench upload
if command -v zip &>/dev/null; then
    (cd results && zip -q codabench_submissions.zip *_submission.csv 2>/dev/null || true)
    log_info "Packaged all predictions to results/codabench_submissions.zip"
fi

log_success "All benchmark runs completed successfully."
exit 0
```

**Replacement Content:**
```bash
# Package only active submissions from current run for Codabench upload
if command -v zip &>/dev/null && [ ${#ACTIVE_SUBMISSION_FILES[@]} -gt 0 ]; then
    (cd results && rm -f codabench_submissions.zip && zip -q codabench_submissions.zip "${ACTIVE_SUBMISSION_FILES[@]}" 2>/dev/null || true)
    log_info "Packaged ${#ACTIVE_SUBMISSION_FILES[@]} active predictions to results/codabench_submissions.zip"
fi

# Exit with appropriate status code based on failures
if [ "$FAILED_COUNT" -gt 0 ]; then
    log_error "Benchmark matrix finished with ${FAILED_COUNT} failed or invalid experiment(s)."
    exit 1
fi

log_success "All benchmark runs completed successfully."
exit 0
```

---

### Blueprint D: Supplementary Tests for `tests/test_training.py`

To prevent regressions, Worker M3 should append the following test cases to `tests/test_training.py`:

```python
    # Under TestTrainingUtils:
    def test_jsonl_logger_non_serializable_objects(self) -> None:
        log_file = Path(self.temp_dir) / "metrics_non_serializable.jsonl"
        logger = JsonlLogger(log_file)
        # Logging non-serializable objects (set, Path) must not raise TypeError
        logger.log({"set_val": {"a", "b"}, "path_val": Path("/tmp/model.pt")})
        records = logger.read_all()
        self.assertEqual(len(records), 1)
        self.assertIn("set_val", records[0])
        self.assertIn("path_val", records[0])

    # Under TestCLIIntegration:
    def test_cli_nonexistent_checkpoint_error(self) -> None:
        res = subprocess.run(
            [
                self.python_bin,
                "experiments/train.py",
                "--eval_only",
                "--checkpoint_path", "/nonexistent/path/model.pt",
                "--test_run",
            ],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("FileNotFoundError", res.stderr + res.stdout)

    def test_cli_conflicting_precision_flags(self) -> None:
        res = subprocess.run(
            [
                self.python_bin,
                "experiments/train.py",
                "--fp16",
                "--bf16",
                "--test_run",
            ],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(res.returncode, 2)
        self.assertIn("Cannot specify both --fp16 and --bf16", res.stderr)

    # Under TestGPUScript:
    def test_gpu_script_continue_on_error_fails(self) -> None:
        res = subprocess.run(
            [
                "bash",
                "scripts/run_gpu_training.sh",
                "--python", "/usr/bin/false",
                "--continue-on-error",
                "--track", "1",
            ],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("failed", res.stderr + res.stdout)

    def test_gpu_script_crl_sd_filter(self) -> None:
        res = subprocess.run(
            [
                "bash",
                "scripts/run_gpu_training.sh",
                "--method", "crl_sd",
                "--dry-run",
            ],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("t1_crl_sd_deberta", res.stdout)
        self.assertIn("t2_crl_sd_beto", res.stdout)
```

---

## 5. Verification Method

Once implemented by Worker M3, the following commands independently verify complete resolution:

```bash
# 1. Existing unit test suite passes 100%
python3 -m unittest discover -s tests -p "test_*.py" -v

# 2. Verify non-existent checkpoint raises FileNotFoundError and exits non-zero (exit 1)
python3 experiments/train.py --eval_only --checkpoint_path /path/to/nonexistent.pt --test_run
# Expected: Exit code 1, FileNotFoundError logged

# 3. Verify conflicting precision flags are rejected by parser (exit 2)
python3 experiments/train.py --fp16 --bf16 --test_run
# Expected: Exit code 2, error: Cannot specify both --fp16 and --bf16. Please select one.

# 4. Verify JsonlLogger serializes non-native types without raising TypeError
python3 -c "
from src.training.utils import JsonlLogger
from pathlib import Path
import tempfile
with tempfile.TemporaryDirectory() as td:
    l = JsonlLogger(Path(td)/'test.jsonl')
    l.log({'set': {1, 2}, 'path': Path('/tmp')})
    print('SUCCESS')
"
# Expected: SUCCESS (exit 0)

# 5. Verify --continue-on-error exits with code 1 when runs fail
bash scripts/run_gpu_training.sh --python "/usr/bin/false" --continue-on-error --track 1
# Expected: Exit code 1, logs failure count summary

# 6. Verify --method crl_sd matches planned experiments
bash scripts/run_gpu_training.sh --method crl_sd --dry-run
# Expected: Exit code 0, lists t1_crl_sd_deberta and t2_crl_sd_beto

# 7. Run full benchmark quick-test (10 experiments)
bash scripts/run_gpu_training.sh --quick-test
# Expected: All 10 experiments succeed in <5s, Codabench format verified
```
