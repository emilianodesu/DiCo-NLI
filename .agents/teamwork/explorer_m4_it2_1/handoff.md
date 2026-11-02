# CLI File Validation Investigation & Fix Blueprint — Milestone 4 Iteration 2

**Author**: `explorer_m4_it2_1` (Role: Explorer 1 M4 It2 CLI File Validation)  
**Date**: 2026-11-02  
**Target Scope**: `experiments/evaluate.py`, `tests/test_evaluation.py`  
**Milestone**: M4 Iteration 2 (Gate Failure Remediation)  
**Status**: COMPLETE (Investigation & Fix Blueprint Ready for Implementer `worker_m4_it2`)

---

## 1. Observation

Direct empirical and code-level investigation revealed three distinct vulnerabilities in `experiments/evaluate.py` causing silent failures (exit code 0) and unrequested fallback to mock models when invalid or non-existent file/directory paths are supplied to the CLI:

### 1.1 Empirical Verification of Vulnerabilities (Commands and Outputs)

1. **Non-existent Checkpoint Path (`--checkpoint_path`)**:
   - Command:
     ```bash
     python3 experiments/evaluate.py --checkpoint_path /path/to/nonexistent.pt
     ```
   - Observed Output:
     ```text
     2026-11-01 21:44:57 | INFO | dico_nli.evaluate | SemEval-2027 Task 2 (DiCo-NLI) — Evaluation CLI Engine
     2026-11-01 21:44:57 | INFO | dico_nli.evaluate | Tracks: ['track1', 'track2'] | Methods: ['baseline', 'bsa', 'crl', 'sd', 'crl_sd'] | Split: dev
     2026-11-01 21:44:57 | INFO | src.models.factory | Instantiating mock model for method=baseline on device=auto
     ...
     Evaluation completed successfully in 0.009s
     ```
   - **Exit Code**: **`0`** (FAIL: Should terminate with non-zero exit code and raise `FileNotFoundError`).

2. **Non-existent Prediction File (`--prediction_file`)**:
   - Command:
     ```bash
     python3 experiments/evaluate.py --prediction_file /path/to/nonexistent.csv
     ```
   - Observed Output:
     ```text
     2026-11-01 21:45:00 | INFO | dico_nli.evaluate | Scanning 'results' and 'logs' for existing benchmark results...
     ## Experimental Evaluation & Reversal Mitigation Benchmark
     ...
     Evaluation completed successfully in 0.006s
     ```
   - **Exit Code**: **`0`** (FAIL: Silently ignored argument and fell back to scanning `results/`, outputting unrelated existing benchmark tables).

3. **Non-existent Results Directory (`--results_dir`)**:
   - Command:
     ```bash
     python3 experiments/evaluate.py --results_dir /path/to/nonexistent_dir --logs_dir /path/to/nonexistent_logs
     ```
   - Observed Output:
     ```text
     2026-11-01 21:45:12 | INFO | dico_nli.evaluate | Scanning '/path/to/nonexistent_dir' and '/path/to/nonexistent_logs' for existing benchmark results...
     2026-11-01 21:45:12 | INFO | dico_nli.evaluate | Executing fast mock benchmark evaluation across all target tracks and methods...
     ...
     Evaluation completed successfully in 0.010s
     ```
   - **Exit Code**: **`0`** (FAIL: Silently fabricated synthetic mock evaluations without the user passing `--test_run` or `--model_name mock`).

4. **Reference Implementation in `experiments/train.py`**:
   - Command:
     ```bash
     python3 experiments/train.py --eval_only --checkpoint_path /nonexistent/model.pt --test_run
     ```
   - Observed Output:
     ```text
     2026-11-01 21:48:26 | ERROR | dico_nli.train | Checkpoint file not found: /nonexistent/model.pt
     2026-11-01 21:48:26 | ERROR | dico_nli.train | Pipeline execution failed: Checkpoint file not found: /nonexistent/model.pt
     FileNotFoundError: Checkpoint file not found: /nonexistent/model.pt
     ```
   - **Exit Code**: **`1`** (PASS: Explicit `FileNotFoundError`, logged to stderr, process terminated with exit code 1).

---

### 1.2 Line-by-Line Code Inspection of `experiments/evaluate.py`

- **Line 186**:
  ```python
  group_io.add_argument(
      "--results_dir",
      type=str,
      default="results",
      help="Directory containing pre-computed Codabench prediction CSVs.",
  )
  ```
  *Issue*: Default `"results"` makes it ambiguous whether `--results_dir` was explicitly passed by the user or was defaulted.

- **Lines 319–334 (`run_dynamic_model_evaluation`)**:
  ```python
  if checkpoint_path and os.path.isfile(checkpoint_path):
      logger.info(f"Loading checkpoint weights from: {checkpoint_path}")
      try:
          import torch
          state_dict = torch.load(checkpoint_path, map_location=target_device)
          ...
      except Exception as exc:
          ...
  ```
  *Issue*: If `checkpoint_path` is passed but `os.path.isfile(checkpoint_path)` is False, the entire weight-loading block is skipped silently without error or warning. Execution continues to line 340, performing inference with uninitialized or mock model weights and returning exit code 0.

- **Lines 560–572 (`run_evaluation`)**:
  ```python
  # 1. Direct Single Prediction File Evaluation
  if args.prediction_file and os.path.isfile(args.prediction_file):
      logger.info(f"Evaluating single prediction CSV: {args.prediction_file}")
      ...
      table = ComparisonTable([single_result])
  ```
  *Issue*: If `--prediction_file` is specified but does not exist, `args.prediction_file and os.path.isfile(args.prediction_file)` evaluates to `False`. The branch is skipped and execution drops into branch 3 (scanning `results/`), completely ignoring the user's explicit parameter and printing unrelated metrics.

- **Lines 594–622 (`run_evaluation`)**:
  ```python
  # 3. Static Evaluation of Existing Results Directory (or Fast Mock under test_run)
  else:
      logger.info(f"Scanning '{args.results_dir}' and '{args.logs_dir}' for existing benchmark results...")
      table = aggregate_results_from_dir(...)

      # If results_dir had no matching files or in test_run mode, generate fast mock evaluations
      if len(table.results) == 0 or args.test_run:
          logger.info("Executing fast mock benchmark evaluation across all target tracks and methods...")
          ...
  ```
  *Issue*:
  1. `results_dir` is not checked for existence before calling `aggregate_results_from_dir`.
  2. If `len(table.results) == 0`, even when `args.test_run` is `False`, the script silently runs mock evaluations and prints synthetic results instead of alerting the user that no valid prediction files were found in the scanned directory.

- **Lines 690–697 (`main`)**:
  ```python
  def main() -> None:
      args = parse_args()
      try:
          run_evaluation(args)
      except Exception as exc:
          logger.exception(f"Evaluation CLI failed: {exc}")
          sys.exit(1)
  ```
  *Observation*: `main()` already contains a `try...except Exception as exc:` block that calls `sys.exit(1)` and logs the exception with traceback. Thus, raising `FileNotFoundError` inside `validate_cli_args` or `run_evaluation` guarantees immediate termination with exit code 1.

---

## 2. Logic Chain

1. **Requirement Alignment**:
   - According to `GATE_STATUS.md` and `challenger_m4_2/handoff.md`, the CLI evaluation tool must not silently succeed when a user provides paths to non-existent assets.
   - Silently ignoring missing checkpoints or prediction files compromises benchmark scientific integrity: a practitioner who typos `--checkpoint_path checkpoints/best.pt` receives numbers from uninitialized mock weights and could mistakenly report them as valid experimental findings.

2. **Differentiating Input vs Output Directories**:
   - In `experiments/evaluate.py`, `--output_dir` (default: `"results/evaluation"`) is the directory where generated tables, matrices, and summaries are written. Line 548 correctly ensures `output_dir.mkdir(parents=True, exist_ok=True)`.
   - In contrast, `--results_dir` is strictly an **input** directory containing pre-computed Codabench prediction CSVs to be evaluated.
   - Creating an empty directory for `--results_dir` when it does not exist is counter-productive: an empty directory contains no prediction CSVs, so static evaluation cannot proceed.
   - Therefore, when scanning for existing files, if `--results_dir` does not exist, the script must **error** (raise `FileNotFoundError`), not create the directory.

3. **Fail-Fast Architecture**:
   - Validation must occur upfront in a dedicated function `validate_cli_args(args)` called at the very beginning of `run_evaluation(args)`.
   - Upfront validation prevents any unnecessary side effects (e.g. creating `output_dir`, loading datasets, or instantiating models) before failing.
   - Modular validation should also remain in `run_dynamic_model_evaluation` (line 319) to guard programmatic calls from external Python code.

4. **Preserving Fast Test Run Budget (<2.0s)**:
   - When `--test_run` is enabled (without `--prediction_file` and without `--checkpoint_path`), the user desires a rapid self-contained mock benchmark on CPU.
   - Under `--test_run`, `results/` is not required to exist beforehand. The script should immediately generate synthetic mock evaluations without scanning `results_dir`.
   - If a user explicitly specifies `--checkpoint_path` or `--prediction_file` or `--results_dir`, path existence must be enforced even if `--test_run` is also passed.

---

## 3. Caveats

- **Read-Only Explorer Scope**: This report provides the architectural diagnosis and exact code implementation blueprint. Source code files (`experiments/evaluate.py` and `tests/test_evaluation.py`) are not modified directly by this agent.
- **`data_dir` Behavior**: `load_phrasis_split` has built-in `allow_mock_fallback=True` for offline unit testing without raw PhrasIS CSVs. This mock fallback is legitimate and specified in Milestone 1. The gate failure is strictly about `--checkpoint_path`, `--prediction_file`, and `--results_dir`.
- **Exit Code Conventions**: `argparse` built-in choices/parsing errors exit with code 2; application runtime validation (`FileNotFoundError`) caught in `main()` exits with code 1 via `sys.exit(1)`. Both satisfy "non-zero exit code" requirements.

---

## 4. Conclusion & Complete Fix Blueprint

### 4.1 Summary of Recommended Fixes

1. **Implement `validate_cli_args(args: argparse.Namespace) -> None`** in `experiments/evaluate.py` to validate:
   - `--checkpoint_path`: Must exist and be a file if passed (`os.path.isfile`).
   - `--prediction_file`: Must exist and be a file if passed (`os.path.isfile`).
   - `--results_dir`: If explicitly passed, must exist and be a directory (`os.path.isdir`). If in static evaluation mode, the resolved results directory must exist.
   - `--logs_dir`: If explicitly passed, must exist and be a directory (`os.path.isdir`).
2. **Call `validate_cli_args(args)`** at line 1 of `run_evaluation(args)` before directory creation or dataset loading.
3. **Update `run_dynamic_model_evaluation`** (line 319) to raise `FileNotFoundError` if `checkpoint_path` is provided but missing.
4. **Update `run_evaluation` static evaluation** (line 561 & 606) to avoid silent mock generation when `len(table.results) == 0` outside of `--test_run`.
5. **Adjust `parse_args` defaults** for `--results_dir` and `--logs_dir` to `None` (resolving to `"results"` and `"logs"`) so explicit flags can be unambiguously identified.
6. **Add 3 CLI integration tests and 4 unit tests** in `tests/test_evaluation.py`.

---

### 4.2 Exact Code Blueprint for `experiments/evaluate.py`

#### Edit 1: Adjust `parse_args()` parameter defaults (lines 184–197)

```python
<<<< BEFORE (experiments/evaluate.py:184-197)
    # --- Directories & Inputs (Static Evaluation) ---
    group_io = parser.add_argument_group("Directories & Input Artifacts")
    group_io.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing pre-computed Codabench prediction CSVs.",
    )
    group_io.add_argument(
        "--logs_dir",
        type=str,
        default="logs",
        help="Directory containing training run summary JSON logs.",
    )
==== AFTER
    # --- Directories & Inputs (Static Evaluation) ---
    group_io = parser.add_argument_group("Directories & Input Artifacts")
    group_io.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory containing pre-computed Codabench prediction CSVs (default: 'results').",
    )
    group_io.add_argument(
        "--logs_dir",
        type=str,
        default=None,
        help="Directory containing training run summary JSON logs (default: 'logs').",
    )
>>>>
```

Also polish `--no_tables` (line 220):
```python
<<<< BEFORE (experiments/evaluate.py:220-224)
    group_artifacts.add_argument(
        "--no_tables",
        action="store_false",
        dest="generate_tables",
        help="Disable table generation.",
    )
==== AFTER
    group_artifacts.add_argument(
        "--no_tables",
        action="store_false",
        dest="generate_tables",
        default=argparse.SUPPRESS,
        help="Disable table generation.",
    )
>>>>
```

---

#### Edit 2: Add `validate_cli_args` function (insert after line 252)

```python
def validate_cli_args(args: argparse.Namespace) -> None:
    """Validates CLI argument existence and integrity before pipeline execution.

    Raises:
        FileNotFoundError: If an explicitly supplied checkpoint_path, prediction_file,
            results_dir, or required static evaluation directory does not exist.
        NotADirectoryError: If results_dir or logs_dir points to a non-directory.
    """
    # 1. Validate Checkpoint Path
    if args.checkpoint_path is not None:
        if not os.path.isfile(args.checkpoint_path):
            error_msg = f"Checkpoint file not found: {args.checkpoint_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

    # 2. Validate Prediction File
    if args.prediction_file is not None:
        if not os.path.isfile(args.prediction_file):
            error_msg = f"Prediction file not found: {args.prediction_file}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

    # 3. Validate Results Directory (if explicitly passed)
    if args.results_dir is not None:
        if not os.path.isdir(args.results_dir):
            error_msg = f"Results directory not found: {args.results_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

    # 4. Validate Logs Directory (if explicitly passed)
    if args.logs_dir is not None:
        if not os.path.isdir(args.logs_dir):
            error_msg = f"Logs directory not found: {args.logs_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

    # 5. Validate Static Evaluation Directory Requirements
    is_static_eval = (
        args.prediction_file is None
        and args.checkpoint_path is None
        and not (args.model_name and not args.test_run)
        and not args.test_run
    )
    if is_static_eval:
        target_results = args.results_dir or "results"
        if not os.path.isdir(target_results):
            error_msg = f"Results directory not found for static evaluation: {target_results}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
```

---

#### Edit 3: Update `run_dynamic_model_evaluation` (lines 319–334)

```python
<<<< BEFORE (experiments/evaluate.py:319-334)
    if checkpoint_path and os.path.isfile(checkpoint_path):
        logger.info(f"Loading checkpoint weights from: {checkpoint_path}")
        try:
            import torch
            state_dict = torch.load(checkpoint_path, map_location=target_device)
            if "model_state_dict" in state_dict:
                model.load_state_dict(state_dict["model_state_dict"])
            else:
                model.load_state_dict(state_dict)
        except Exception as exc:
            if is_mock:
                logger.info("Mock mode: validated checkpoint structure.")
            else:
                logger.error(f"Failed loading checkpoint: {exc}")
                raise
==== AFTER
    if checkpoint_path is not None:
        if not os.path.isfile(checkpoint_path):
            error_msg = f"Checkpoint file not found: {checkpoint_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        logger.info(f"Loading checkpoint weights from: {checkpoint_path}")
        try:
            import torch
            state_dict = torch.load(checkpoint_path, map_location=target_device)
            if "model_state_dict" in state_dict:
                model.load_state_dict(state_dict["model_state_dict"])
            else:
                model.load_state_dict(state_dict)
        except Exception as exc:
            if is_mock:
                logger.info("Mock mode: validated checkpoint structure.")
            else:
                logger.error(f"Failed loading checkpoint: {exc}")
                raise
>>>>
```

---

#### Edit 4: Update `run_evaluation()` upfront validation and branches (lines 542–622)

```python
<<<< BEFORE (experiments/evaluate.py:542-622)
def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """Master controller executing static or dynamic evaluation and generating comparison tables."""
    t0 = time.time()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracks = resolve_target_tracks(args.track)
    methods = resolve_target_methods(args.method)

    logger.info("=" * 70)
    logger.info("SemEval-2027 Task 2 (DiCo-NLI) — Evaluation CLI Engine")
    logger.info(f"Tracks: {tracks} | Methods: {methods} | Split: {args.split}")
    logger.info(f"Results Dir: {args.results_dir} | Output Dir: {output_dir}")
    logger.info(f"Test Run: {args.test_run} | Tables: {args.generate_tables}")
    logger.info("=" * 70)

    # 1. Direct Single Prediction File Evaluation
    if args.prediction_file and os.path.isfile(args.prediction_file):
        logger.info(f"Evaluating single prediction CSV: {args.prediction_file}")
        trk = tracks[0] if tracks else "track1"
        gold_df = load_phrasis_split(track=trk, split=args.split, data_dir=args.data_dir)
        single_result = evaluate_prediction_file(
            pred_csv_path=args.prediction_file,
            gold_df_or_records=gold_df,
            track=trk,
            split=args.split,
        )
        table = ComparisonTable([single_result])

    # 2. Dynamic Model Evaluation (When checkpoint or specific model requested)
    elif args.checkpoint_path or (args.model_name and not args.test_run):
        ...
    # 3. Static Evaluation of Existing Results Directory (or Fast Mock under test_run)
    else:
        logger.info(f"Scanning '{args.results_dir}' and '{args.logs_dir}' for existing benchmark results...")
        table = aggregate_results_from_dir(
            results_dir=args.results_dir,
            logs_dir=args.logs_dir,
            data_dir=args.data_dir,
            split=args.split,
            tracks=tracks,
        )

        # If results_dir had no matching files or in test_run mode, generate fast mock evaluations
        if len(table.results) == 0 or args.test_run:
            logger.info("Executing fast mock benchmark evaluation across all target tracks and methods...")
            ...
==== AFTER
def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """Master controller executing static or dynamic evaluation and generating comparison tables."""
    # 0. Upfront CLI File and Directory Validation
    validate_cli_args(args)

    t0 = time.time()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracks = resolve_target_tracks(args.track)
    methods = resolve_target_methods(args.method)
    results_dir = args.results_dir or "results"
    logs_dir = args.logs_dir or "logs"

    logger.info("=" * 70)
    logger.info("SemEval-2027 Task 2 (DiCo-NLI) — Evaluation CLI Engine")
    logger.info(f"Tracks: {tracks} | Methods: {methods} | Split: {args.split}")
    logger.info(f"Results Dir: {results_dir} | Output Dir: {output_dir}")
    logger.info(f"Test Run: {args.test_run} | Tables: {args.generate_tables}")
    logger.info("=" * 70)

    # 1. Direct Single Prediction File Evaluation
    if args.prediction_file:
        logger.info(f"Evaluating single prediction CSV: {args.prediction_file}")
        trk = tracks[0] if tracks else "track1"
        gold_df = load_phrasis_split(track=trk, split=args.split, data_dir=args.data_dir)
        single_result = evaluate_prediction_file(
            pred_csv_path=args.prediction_file,
            gold_df_or_records=gold_df,
            track=trk,
            split=args.split,
        )
        table = ComparisonTable([single_result])

    # 2. Dynamic Model Evaluation (When checkpoint or specific model requested)
    elif args.checkpoint_path or (args.model_name and not args.test_run):
        results_list = []
        for trk in tracks:
            for mth in methods:
                mdl = args.model_name or ("DeBERTa-v3" if trk == "track1" else "BETO")
                res = run_dynamic_model_evaluation(
                    track=trk,
                    method=mth,
                    model_name=mdl,
                    checkpoint_path=args.checkpoint_path,
                    split=args.split,
                    data_dir=args.data_dir,
                    device=args.device,
                    batch_size=args.batch_size,
                    output_dir=str(output_dir),
                    test_run=args.test_run,
                )
                results_list.append(res)
        table = build_comparison_table(results_list)

    # 3. Static Evaluation of Existing Results Directory (or Fast Mock under test_run)
    else:
        if args.test_run:
            logger.info("Test run mode: executing fast mock benchmark evaluation across all target tracks and methods...")
            mock_results = []
            for trk in tracks:
                for mth in methods:
                    res = run_dynamic_model_evaluation(
                        track=trk,
                        method=mth,
                        model_name="mock",
                        split=args.split,
                        data_dir=args.data_dir,
                        output_dir=str(output_dir),
                        test_run=True,
                    )
                    mock_results.append(res)
            table = build_comparison_table(mock_results)
        else:
            logger.info(f"Scanning '{results_dir}' and '{logs_dir}' for existing benchmark results...")
            table = aggregate_results_from_dir(
                results_dir=results_dir,
                logs_dir=logs_dir,
                data_dir=args.data_dir,
                split=args.split,
                tracks=tracks,
            )
            if len(table.results) == 0:
                error_msg = f"No valid evaluation results found in '{results_dir}' or '{logs_dir}'."
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
>>>>
```

---

### 4.3 Exact Test Additions for `tests/test_evaluation.py`

In `tests/test_evaluation.py`, update imports at line 79 to include `validate_cli_args`:
```python
from experiments.evaluate import (
    parse_args,
    run_dynamic_model_evaluation,
    run_evaluation,
    validate_cli_args,
)
```

In class `TestCLIIntegration` (around line 908), add the following test methods:

```python
    def test_cli_nonexistent_checkpoint_error(self) -> None:
        """Verifies CLI evaluation errors and exits with code 1 when given a missing checkpoint."""
        script_path = PROJECT_ROOT / "experiments" / "evaluate.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--checkpoint_path", "/nonexistent/checkpoint_path_model.pt",
            "--test_run",
        ]
        res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=10)
        self.assertEqual(res.returncode, 1, f"Expected returncode 1, got {res.returncode}.\nStderr: {res.stderr}")
        self.assertIn("FileNotFoundError", res.stderr + res.stdout)
        self.assertIn("Checkpoint file not found", res.stderr + res.stdout)

    def test_cli_nonexistent_prediction_file_error(self) -> None:
        """Verifies CLI evaluation errors and exits with code 1 when given a missing prediction file."""
        script_path = PROJECT_ROOT / "experiments" / "evaluate.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--prediction_file", "/nonexistent/prediction_file.csv",
        ]
        res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=10)
        self.assertEqual(res.returncode, 1, f"Expected returncode 1, got {res.returncode}.\nStderr: {res.stderr}")
        self.assertIn("FileNotFoundError", res.stderr + res.stdout)
        self.assertIn("Prediction file not found", res.stderr + res.stdout)

    def test_cli_nonexistent_results_dir_error(self) -> None:
        """Verifies CLI evaluation errors and exits with code 1 when given a missing results directory."""
        script_path = PROJECT_ROOT / "experiments" / "evaluate.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--results_dir", "/nonexistent/custom_results_directory",
        ]
        res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=10)
        self.assertEqual(res.returncode, 1, f"Expected returncode 1, got {res.returncode}.\nStderr: {res.stderr}")
        self.assertIn("FileNotFoundError", res.stderr + res.stdout)
        self.assertIn("Results directory not found", res.stderr + res.stdout)

    def test_validate_cli_args_direct_exceptions(self) -> None:
        """Direct unit testing of validate_cli_args exception raising."""
        # Missing checkpoint
        args_ckpt = parse_args(["--checkpoint_path", "/nonexistent/path/ckpt.pt"])
        with self.assertRaises(FileNotFoundError) as ctx_ckpt:
            validate_cli_args(args_ckpt)
        self.assertIn("Checkpoint file not found", str(ctx_ckpt.exception))

        # Missing prediction file
        args_pred = parse_args(["--prediction_file", "/nonexistent/path/pred.csv"])
        with self.assertRaises(FileNotFoundError) as ctx_pred:
            validate_cli_args(args_pred)
        self.assertIn("Prediction file not found", str(ctx_pred.exception))

        # Missing results dir
        args_res = parse_args(["--results_dir", "/nonexistent/path/results"])
        with self.assertRaises(FileNotFoundError) as ctx_res:
            validate_cli_args(args_res)
        self.assertIn("Results directory not found", str(ctx_res.exception))
```

---

## 5. Verification Method

To verify this implementation once applied by `worker_m4_it2`:

1. **Verify Checkpoint Validation Exit Code and Stderr**:
   ```bash
   python3 experiments/evaluate.py --checkpoint_path /path/to/nonexistent.pt
   ```
   *Expected Outcome*: Returncode **`1`**; output contains `FileNotFoundError: Checkpoint file not found: /path/to/nonexistent.pt`.

2. **Verify Prediction File Validation Exit Code and Stderr**:
   ```bash
   python3 experiments/evaluate.py --prediction_file /path/to/nonexistent.csv
   ```
   *Expected Outcome*: Returncode **`1`**; output contains `FileNotFoundError: Prediction file not found: /path/to/nonexistent.csv`.

3. **Verify Results Directory Validation Exit Code and Stderr**:
   ```bash
   python3 experiments/evaluate.py --results_dir /path/to/nonexistent_dir
   ```
   *Expected Outcome*: Returncode **`1`**; output contains `FileNotFoundError: Results directory not found: /path/to/nonexistent_dir`.

4. **Verify Fast Test Run Budget (<2.0s) Unaltered**:
   ```bash
   python3 experiments/evaluate.py --test_run --generate_tables --generate_matrices --generate_analysis --output_dir results/evaluation
   ```
   *Expected Outcome*: Returncode **`0`**; execution completes in $<0.1$s; all tables, matrices, and syntactic analysis artifacts exported.

5. **Run Evaluation Unit & Integration Test Suite**:
   ```bash
   python3 -m unittest tests/test_evaluation.py -v
   ```
   *Expected Outcome*: All test methods pass (including the 4 new validation tests).

6. **Run Full Project Test Suite**:
   ```bash
   python3 -m unittest discover -s tests -p "test_*.py" -v
   ```
   *Expected Outcome*: 154+ tests pass cleanly (100% pass rate).

7. **Run Remote GPU Training Script Quick-Test**:
   ```bash
   bash scripts/run_gpu_training.sh --quick-test
   ```
   *Expected Outcome*: 10/10 mock experiments finish in $\le 5$s with `SUCCESS` and valid Codabench outputs.
