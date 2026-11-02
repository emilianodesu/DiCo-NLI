# Handoff Report — Test Blueprint for Evaluation CLI Path Validation

**Agent**: `explorer_m4_it2_3` (Role: Explorer 3 M4 It2 Regression Tests)  
**Milestone**: M4 Iteration 2  
**Date**: 2026-11-02  
**Target File**: `tests/test_evaluation.py`  
**Referenced Implementation File**: `experiments/evaluate.py`  

---

## 1. Observation

### 1.1 Current Baseline Test Suite Status
Running the project test discovery suite:
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```
Result:
- **150 tests passed** in 1.180s across 10 test modules (`test_codabench`: 11, `test_consistency`: 3, `test_data_loader`: 7, `test_evaluation`: 36, `test_metrics`: 11, `test_models`: 29, `test_pipeline`: 3, `test_reversal_logic`: 15, `test_symmetrized_decoding`: 5, `test_training`: 30).
- Running `tests/test_evaluation.py` standalone: `Ran 36 tests in 0.169s ... OK`.

### 1.2 Reproduction of Silent Exit Code 0 on Nonexistent Paths
Direct execution commands on `experiments/evaluate.py`:

1. **Nonexistent Checkpoint**:
   ```bash
   python3 experiments/evaluate.py --checkpoint_path /path/to/nonexistent.pt
   ```
   *Exit code*: `0`  
   *Log snippet*:
   ```text
   2026-11-01 21:43:30 | INFO | dico_nli.evaluate | Dynamic Evaluation: track=track1, method=baseline, model=DeBERTa-v3, device=auto
   2026-11-01 21:43:30 | INFO | src.models.factory | Instantiating mock model for method=baseline on device=auto
   ...
   2026-11-01 21:43:30 | INFO | dico_nli.evaluate | Evaluation completed successfully in 0.010s
   ```
   *Root Cause in `experiments/evaluate.py` (lines 319-334)*:
   ```python
   if checkpoint_path and os.path.isfile(checkpoint_path):
       logger.info(f"Loading checkpoint weights from: {checkpoint_path}")
   ```
   Because `/path/to/nonexistent.pt` is not a file, it silently skips loading checkpoint weights, proceeds to evaluate default uninitialized mock model weights, prints a benchmark table, and exits with code 0.

2. **Nonexistent Prediction File**:
   ```bash
   python3 experiments/evaluate.py --prediction_file /path/to/nonexistent.csv
   ```
   *Exit code*: `0`  
   *Log snippet*:
   ```text
   2026-11-01 21:43:33 | INFO | dico_nli.evaluate | Scanning 'results' and 'logs' for existing benchmark results...
   2026-11-01 21:43:33 | INFO | dico_nli.evaluate | Exporting comparison tables across all formats...
   ```
   *Root Cause in `experiments/evaluate.py` (lines 561-597)*:
   ```python
   if args.prediction_file and os.path.isfile(args.prediction_file):
       ...
   else:
       logger.info(f"Scanning '{args.results_dir}' and '{args.logs_dir}' for existing benchmark results...")
   ```
   When `--prediction_file` is specified but missing, the controller silently falls back to scanning `results/` and exits with code 0.

### 1.3 Existing Reference Implementation in `tests/test_training.py`
In `tests/test_training.py` lines 383-399:
```python
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
```
And in `experiments/train.py` lines 370-374:
```python
if args.checkpoint_path is not None:
    if not os.path.isfile(args.checkpoint_path):
        error_msg = f"Checkpoint file not found: {args.checkpoint_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
```

### 1.4 Existing Structure of `tests/test_evaluation.py`
Lines 779-908 contain `TestEvaluateCLI`:
- `setUp` initializes a clean temporary directory (`self.temp_dir = tempfile.mkdtemp(prefix="test_eval_cli_")`).
- `tearDown` cleans up `self.temp_dir`.
- Current tests:
  - `test_cli_argument_parsing_defaults` (lines 789-798)
  - `test_cli_argument_parsing_custom_flags` (lines 799-817)
  - `test_cli_test_run_execution_speed_and_artifacts` (lines 818-883)
  - `test_cli_single_prediction_file_evaluation` (lines 884-907) — already tests a valid existing prediction CSV.

---

## 2. Logic Chain

1. **Vulnerability Assessment**:
   - Specifying `--checkpoint_path` or `--prediction_file` is an explicit user intent to evaluate specific weight checkpoints or prediction outputs.
   - Silently ignoring nonexistent files and producing benchmark outputs from uninitialized mock models or stale `results/` files compromises benchmark provenance and hides user configuration errors.
   - Therefore, passing a nonexistent path to `--checkpoint_path` or `--prediction_file` must explicitly raise `FileNotFoundError` and cause the CLI to terminate with non-zero exit code (specifically exit code 1 via `main()`'s exception handler).

2. **Test Design Strategy**:
   - The test suite must cover both integration invocation (via `subprocess.run`, checking `returncode == 1` and `stderr` containing `FileNotFoundError`) and programmatic invocation (via `run_evaluation(args)` with `assertRaises(FileNotFoundError)`).
   - Valid existing paths must be verified to proceed normally (exit code 0, generating artifacts).
   - In `test_evaluation.py`, `TestEvaluateCLI` already has `self.temp_dir`, `PROJECT_ROOT`, and standard `subprocess.run` fixtures.
   - Adding 4 focused methods to `TestEvaluateCLI` will raise the test count from 36 to 40 in `test_evaluation.py`, and total project tests from 150 to 154, executing in $<0.15$s without regressions.

---

## 3. Caveats

- **Mock Execution Mode vs CUDA**: Local unit tests use `--test_run` or mock models. In mock mode, valid checkpoint files do not need real PyTorch weights (a plain text file or state dict suffices because `run_dynamic_model_evaluation` handles mock checkpoints gracefully).
- **Read-Only Scope**: This agent is strictly read-only. No edits have been made to `tests/test_evaluation.py` or `experiments/evaluate.py`. The blueprint below provides exact code snippets ready for implementation by the worker agent.

---

## 4. Conclusion & Test Blueprint

### 4.1 Concrete Test Blueprint for `tests/test_evaluation.py`

Add the following 4 test methods to class `TestEvaluateCLI` in `tests/test_evaluation.py` (after line 907):

```python
    def test_cli_nonexistent_checkpoint_error(self) -> None:
        """Verifies CLI exits with non-zero code (exit 1) and FileNotFoundError when checkpoint is missing."""
        script_path = PROJECT_ROOT / "experiments" / "evaluate.py"
        missing_ckpt = Path(self.temp_dir) / "nonexistent_model.pt"

        cmd = [
            sys.executable,
            str(script_path),
            "--checkpoint_path", str(missing_ckpt),
            "--output_dir", str(Path(self.temp_dir) / "out_missing_ckpt"),
            "--test_run",
        ]

        res = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertNotEqual(
            res.returncode,
            0,
            f"Expected CLI failure on missing checkpoint, but got returncode 0.\nSTDOUT:\n{res.stdout}",
        )
        self.assertEqual(res.returncode, 1, f"Expected returncode 1, got {res.returncode}")
        combined_err = res.stderr + res.stdout
        self.assertIn("FileNotFoundError", combined_err)
        self.assertIn(str(missing_ckpt), combined_err)

    def test_cli_nonexistent_prediction_file_error(self) -> None:
        """Verifies CLI exits with non-zero code (exit 1) and FileNotFoundError when prediction file is missing."""
        script_path = PROJECT_ROOT / "experiments" / "evaluate.py"
        missing_pred = Path(self.temp_dir) / "nonexistent_predictions.csv"

        cmd = [
            sys.executable,
            str(script_path),
            "--prediction_file", str(missing_pred),
            "--output_dir", str(Path(self.temp_dir) / "out_missing_pred"),
            "--test_run",
        ]

        res = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertNotEqual(
            res.returncode,
            0,
            f"Expected CLI failure on missing prediction file, but got returncode 0.\nSTDOUT:\n{res.stdout}",
        )
        self.assertEqual(res.returncode, 1, f"Expected returncode 1, got {res.returncode}")
        combined_err = res.stderr + res.stdout
        self.assertIn("FileNotFoundError", combined_err)
        self.assertIn(str(missing_pred), combined_err)

    def test_cli_valid_checkpoint_path_execution(self) -> None:
        """Verifies CLI succeeds (exit code 0) and generates outputs when a valid checkpoint exists."""
        valid_ckpt = Path(self.temp_dir) / "valid_model.pt"
        valid_ckpt.write_text("mock checkpoint content for evaluation")

        out_dir = Path(self.temp_dir) / "valid_ckpt_eval"
        script_path = PROJECT_ROOT / "experiments" / "evaluate.py"

        cmd = [
            sys.executable,
            str(script_path),
            "--checkpoint_path", str(valid_ckpt),
            "--output_dir", str(out_dir),
            "--track", "1",
            "--method", "baseline",
            "--test_run",
        ]

        res = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertEqual(
            res.returncode,
            0,
            f"CLI failed for valid checkpoint with returncode {res.returncode}:\n{res.stderr}",
        )
        self.assertTrue((out_dir / "comparison_table.md").is_file())
        self.assertTrue((out_dir / "evaluation_summary.json").is_file())

        with open(out_dir / "evaluation_summary.json", "r", encoding="utf-8") as f:
            summary = json.load(f)
        self.assertEqual(summary["status"], "success")
        self.assertEqual(summary["total_evaluated"], 1)

    def test_run_evaluation_path_validation_programmatic(self) -> None:
        """Verifies run_evaluation raises FileNotFoundError programmatically for missing files."""
        missing_ckpt = Path(self.temp_dir) / "missing_checkpoint_prog.pt"
        args_ckpt = parse_args([
            "--checkpoint_path", str(missing_ckpt),
            "--output_dir", str(Path(self.temp_dir) / "out_prog_ckpt"),
            "--test_run",
        ])
        with self.assertRaises(FileNotFoundError) as ctx_ckpt:
            run_evaluation(args_ckpt)
        self.assertIn("Checkpoint file not found", str(ctx_ckpt.exception))

        missing_pred = Path(self.temp_dir) / "missing_pred_prog.csv"
        args_pred = parse_args([
            "--prediction_file", str(missing_pred),
            "--output_dir", str(Path(self.temp_dir) / "out_prog_pred"),
            "--test_run",
        ])
        with self.assertRaises(FileNotFoundError) as ctx_pred:
            run_evaluation(args_pred)
        self.assertIn("Prediction file not found", str(ctx_pred.exception))
```

### 4.2 Companion Fix Required in `experiments/evaluate.py`

To satisfy the test blueprint, the worker agent must add path validation in `experiments/evaluate.py`:

1. **Inside `run_evaluation(args)` (around line 548, before controller branches)**:
```python
    # Path existence validation for explicit file arguments
    if args.checkpoint_path is not None and not os.path.isfile(args.checkpoint_path):
        error_msg = f"Checkpoint file not found: {args.checkpoint_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    if args.prediction_file is not None and not os.path.isfile(args.prediction_file):
        error_msg = f"Prediction file not found: {args.prediction_file}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
```

2. **Inside `run_dynamic_model_evaluation` (around line 319)**:
```python
    if checkpoint_path is not None:
        if not os.path.isfile(checkpoint_path):
            error_msg = f"Checkpoint file not found: {checkpoint_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
```

---

## 5. Verification Method

To verify this blueprint after the worker applies the fix:

1. **Run standalone evaluation tests**:
   ```bash
   python3 -m unittest -v tests/test_evaluation.py
   ```
   *Expected*: `Ran 40 tests in ~0.25s ... OK` (36 original + 4 new tests passing).

2. **Direct CLI regression validation**:
   ```bash
   # Nonexistent checkpoint must fail with exit code 1
   python3 experiments/evaluate.py --checkpoint_path /path/to/nonexistent.pt
   echo "Exit code: $?" # Expected: 1

   # Nonexistent prediction file must fail with exit code 1
   python3 experiments/evaluate.py --prediction_file /path/to/nonexistent.csv
   echo "Exit code: $?" # Expected: 1

   # Valid test run must succeed with exit code 0
   python3 experiments/evaluate.py --test_run --output_dir /tmp/eval_verify
   echo "Exit code: $?" # Expected: 0
   ```

3. **Full project test suite**:
   ```bash
   python3 -m unittest discover -s tests -p "test_*.py" -v
   ```
   *Expected*: `Ran 154 tests in <2.0s ... OK` (100% pass rate).

4. **GPU automation quick test**:
   ```bash
   bash scripts/run_gpu_training.sh --quick-test
   ```
   *Expected*: 10/10 mock experiments exit 0 in $\le 5$s.
