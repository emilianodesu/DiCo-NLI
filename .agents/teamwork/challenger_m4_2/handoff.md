# Adversarial Challenge Report — Milestone 4 (CLI & Full Pipeline)

**Challenger Agent**: `challenger_m4_2`  
**Milestone**: M4 (Evaluation Tables, Matrices, Syntactic Breakdown & CLI Evaluation Engine)  
**Date**: 2026-11-02  
**Verdict**: **`REQUEST_CHANGES`**

---

## 1. Observation

### 1.1 CLI Entrypoint `experiments/evaluate.py` Flag Combinations & Performance

The standalone CLI entrypoint was tested across varied flag configurations with `--test_run`:

- **Test Combination A (Track 1 Single Method, Tables Only)**:
  ```bash
  python3 experiments/evaluate.py --track 1 --split dev --method baseline --output_dir /tmp/test_eval_1a --generate_tables --test_run
  ```
  *Output*: Exit code 0 in **0.007s**. Generated `comparison_table.md`, `comparison_table.tex`, `comparison_table.txt`, `comparison_table.json`, `evaluation_summary.json`.

- **Test Combination B (Track 2 Symmetrized Decoding, All Generators Active)**:
  ```bash
  python3 experiments/evaluate.py --track 2 --split dev --method crl_sd --output_dir /tmp/test_eval_1b --generate_tables --generate_matrices --generate_analysis --test_run
  ```
  *Output*: Exit code 0 in **0.011s**. Generated 16 artifacts in `/tmp/test_eval_1b/`, including 4x4 confusion matrices (`.txt`, `.tex`, `.json`), 4x4 directional transition matrices (`.txt`, `.tex`, `.json`), syntactic error analyses (`.md`, `.tex`, `.json`), and Codabench CSV `eval_track2_crl_sd_mock_dev.csv`.
  *Metric Invariant*: `SoftCons` = 1.0000 (100.0%) confirmed for `crl_sd`.

- **Test Combination C (All Tracks, All Methods, All Generators Active)**:
  ```bash
  python3 experiments/evaluate.py --track all --method all --split dev --output_dir /tmp/test_eval_all --generate_tables --generate_matrices --generate_analysis --test_run
  ```
  *Output*: Exit code 0 in **0.052s**. Evaluated all 10 track/method pairs and exported complete suite of artifacts.
  *Timing Constraint*: All runs completed well under the 2.0s budget ($0.007\text{s} - 0.052\text{s} \ll 2.0\text{s}$, approximately $40\times$ faster than budget).

- **Deterministic Seed Invariance**:
  Two successive executions with `--seed 123` produced bit-for-bit identical metric outputs, verified via `diff` across the generated JSON tables.

### 1.2 Invalid Argument Exit Codes & Vulnerability Discovery

CLI error handling was stress-tested against invalid inputs:

- **Invalid Track Choice**:
  `python3 experiments/evaluate.py --track 3`
  *Result*: Exit code 2 (`evaluate.py: error: argument --track: invalid choice: '3'`). **PASSED**.
- **Invalid Method Choice**:
  `python3 experiments/evaluate.py --method invalid`
  *Result*: Exit code 2 (`evaluate.py: error: argument --method: invalid choice: 'invalid'`). **PASSED**.
- **Invalid Split Choice**:
  `python3 experiments/evaluate.py --split bogus`
  *Result*: Exit code 2 (`evaluate.py: error: argument --split: invalid choice: 'bogus'`). **PASSED**.
- **Invalid Device Choice**:
  `python3 experiments/evaluate.py --device tpu`
  *Result*: Exit code 2 (`evaluate.py: error: argument --device: invalid choice: 'tpu'`). **PASSED**.

- **VULNERABILITY: Non-existent Checkpoint Path in Evaluation Mode**:
  ```bash
  python3 experiments/evaluate.py --checkpoint_path /path/to/nonexistent.pt
  ```
  *Result*: **Exit code 0** (Silent failure).
  *Observed Behavior*: In `experiments/evaluate.py` (lines 319-334), the script checks:
  ```python
  if checkpoint_path and os.path.isfile(checkpoint_path):
      ...
  ```
  Because `/path/to/nonexistent.pt` is not a file, it silently skips loading checkpoint weights, proceeds to evaluate default uninitialized mock model weights, prints a benchmark table, and exits with code 0.
  *Contrast with `experiments/train.py`*: In `experiments/train.py` (lines 370-373, enforced by `test_training.py:test_cli_nonexistent_checkpoint_error`), specifying a missing checkpoint path raises `FileNotFoundError` and causes exit code 1:
  ```python
  if args.checkpoint_path is not None:
      if not os.path.isfile(args.checkpoint_path):
          error_msg = f"Checkpoint file not found: {args.checkpoint_path}"
          logger.error(error_msg)
          raise FileNotFoundError(error_msg)
  ```

- **DEFICIENCY: Non-existent Prediction File**:
  ```bash
  python3 experiments/evaluate.py --prediction_file /path/to/nonexistent.csv
  ```
  *Result*: **Exit code 0** (Silent fallback).
  *Observed Behavior*: In `experiments/evaluate.py` line 561:
  ```python
  if args.prediction_file and os.path.isfile(args.prediction_file):
      ...
  ```
  When an explicit `--prediction_file` is specified but does not exist, the CLI silently ignores the parameter and falls back to scanning the `results/` directory, printing full tables and exiting with code 0 instead of notifying the user of the missing input file.

### 1.3 Full Pipeline Regression

- **GPU Script Quick-Test Automation**:
  `bash scripts/run_gpu_training.sh --quick-test`
  *Result*: Exit code 0 in **2.0s**.
  All 10 mock runs executed successfully:
  - `t1_baseline_deberta`: SUCCESS (W-F1: 0.3200, SoftCons: 0.6667, HardCons: 0.2667)
  - `t1_bsa_deberta`: SUCCESS (W-F1: 0.3200, SoftCons: 0.6667, HardCons: 0.2667)
  - `t1_crl_deberta`: SUCCESS (W-F1: 0.3200, SoftCons: 0.6667, HardCons: 0.2667)
  - `t1_sd_deberta`: SUCCESS (W-F1: 0.5333, SoftCons: 1.0000, HardCons: 0.5333)
  - `t1_crl_sd_deberta`: SUCCESS (W-F1: 0.5333, SoftCons: 1.0000, HardCons: 0.5333)
  - `t2_baseline_beto`: SUCCESS (W-F1: 0.2824, SoftCons: 0.7333, HardCons: 0.2667)
  - `t2_bsa_beto`: SUCCESS (W-F1: 0.2824, SoftCons: 0.7333, HardCons: 0.2667)
  - `t2_crl_beto`: SUCCESS (W-F1: 0.2824, SoftCons: 0.7333, HardCons: 0.2667)
  - `t2_sd_beto`: SUCCESS (W-F1: 0.4054, SoftCons: 1.0000, HardCons: 0.4000)
  - `t2_crl_sd_beto`: SUCCESS (W-F1: 0.4054, SoftCons: 1.0000, HardCons: 0.4000)
  - Packaged to `results/codabench_submissions.zip`.

- **Unit Test Discovery Suite**:
  `python3 -m unittest discover -s tests -p "test_*.py" -v`
  *Result*: `Ran 150 tests in 1.633s ... OK`. 100% pass rate across all 150 unit and integration tests.

### 1.4 Artifact Quality Audit in `results/`

1. **Comparison Tables (`results/evaluation/comparison_table.*`)**:
   - `comparison_table.md`: 9 columns, Markdown formatted, delta columns relative to baseline correctly computed.
   - `comparison_table.tex`: Formatted with `\begin{table*}[t]`, booktabs `\toprule`, `\cmidrule`, `\bottomrule`, multi-track headers, and bold best scores.
   - `comparison_table.json`: Valid JSON adhering to schema with total results, tracks, methods, and run provenance metadata.
2. **Transition & Confusion Matrices (`results/evaluation/transition_matrix_*.*`, `confusion_matrix_*.*`)**:
   - All 4 classes (`EQUIVALENCE`, `FORWARD_ENTAILMENT`, `BACKWARD_ENTAILMENT`, `NEGATIVE_OTHER`) represented.
   - For `sd` and `crl_sd`, SoftCons is exactly 100.0%, RCSI is 0.0%, and Entailment Loops are 0.
3. **Syntactic Analysis (`results/evaluation/*_syntactic_analysis.*`, `syntactic_error_analysis.json`)**:
   - 5 categories: nominal, verbal, prepositional, adverbial, quantificational.
   - Qualitative diagnoses correctly extract original text phrases ("Pope Francis", "pope"), gold labels, predictions, and linguistic error taxonomy (`GENERAL_MISCLASSIFICATION`, `REVERSAL_CURSE_LOOP`, `EQUIVALENCE_ASYMMETRY`).
   - Summary JSON is 107 KB, well-formed and valid.
4. **Codabench Submission CSVs**:
   - Programmatically validated all 41 `.csv` submission files in `results/` using `src.data.codabench.validate_codabench_submission`. 100% valid.

---

## 2. Logic Chain

1. **Scope Requirement Alignment**:
   - *Observation 1.1 & Dispatch Item 1*: The dispatch scope explicitly demands:
     > "Verify invalid arguments trigger appropriate exit codes (e.g. invalid track, non-existent checkpoint path in eval mode)."
   - *Observation 1.2*: When an invalid `--track 3` is passed, `argparse` triggers exit code 2. However, when `--checkpoint_path /path/to/nonexistent.pt` is passed, `evaluate.py` exits with code 0 instead of failing.
   - *Deduction*: Silently continuing evaluation when a user requests evaluation of a specific checkpoint that does not exist violates CLI ergonomics and benchmark integrity (the user receives benchmark numbers from uninitialized mock weights believing they came from their checkpoint).
   - *Inference*: `experiments/evaluate.py` must validate that if `--checkpoint_path` or `--prediction_file` is supplied, the path must exist as a file; otherwise, it must raise `FileNotFoundError` (or call `parser.error()`) to exit with a non-zero exit code (code 1 or 2).

2. **Core Functionality & Performance Robustness**:
   - *Observation 1.1*: All valid flag combinations execute properly. Under `--test_run`, evaluation executes in 0.007s - 0.052s, vastly exceeding the <2.0s timing budget.
   - *Observation 1.3*: The existing pipeline (`run_gpu_training.sh --quick-test`) and 150 unit tests pass without regression.
   - *Observation 1.4*: All output artifacts are structurally and syntactically sound.

3. **Conclusion Derivation**:
   - The implementation of Milestone 4 is exceptionally high quality across data structures, metrics, tables, matrices, and syntactic analyses.
   - However, because the orchestrator specifically included non-existent checkpoint paths as a verification criterion, and the CLI currently fails open (exiting 0) on missing checkpoint and prediction files, a targeted fix is required.

---

## 3. Caveats

- **Mock Execution Mode vs Full PyTorch CUDA**: Local tests were executed in CPU mock mode as designed for the local development workstation without NVIDIA CUDA GPU. Remote training on CUDA with actual model weights will be performed on the user's remote server.
- **Scope Limit**: As a reviewer/challenger agent under strict constraints, implementation code was not modified directly.

---

## 4. Conclusion & Required Changes

**Verdict**: **`REQUEST_CHANGES`**

### Summary of Strengths
1. Outstanding execution speed: `evaluate.py --test_run` finishes in 0.052s across all 10 benchmark configurations (38x faster than the 2.0s requirement).
2. Flawless mathematical properties: Symmetrized Decoding consistently produces 100% SoftCons in all tables, transition matrices, and syntactic breakdowns.
3. 100% regression passing: 150/150 unit tests passing; `run_gpu_training.sh --quick-test` runs 10/10 experiments in 2.0s with valid Codabench outputs.
4. Comprehensive multi-format artifact generation (Markdown, LaTeX, ASCII, JSON) for comparison tables, matrices, and linguistic error breakdowns.

### Required Changes (Target: Worker M4)
1. **Enforce Non-Zero Exit Code on Non-Existent `--checkpoint_path`**:
   In `experiments/evaluate.py` (inside `run_evaluation` or `parse_args`):
   ```python
   if args.checkpoint_path is not None and not os.path.isfile(args.checkpoint_path):
       error_msg = f"Checkpoint file not found: {args.checkpoint_path}"
       logger.error(error_msg)
       raise FileNotFoundError(error_msg)
   ```
2. **Enforce Non-Zero Exit Code on Non-Existent `--prediction_file`**:
   In `experiments/evaluate.py`:
   ```python
   if args.prediction_file is not None and not os.path.isfile(args.prediction_file):
       error_msg = f"Prediction file not found: {args.prediction_file}"
       logger.error(error_msg)
       raise FileNotFoundError(error_msg)
   ```
3. **Add CLI Unit Tests in `tests/test_evaluation.py`**:
   Add test methods `test_cli_nonexistent_checkpoint_error` and `test_cli_nonexistent_prediction_file_error` asserting non-zero exit code (`returncode != 0`) when missing paths are passed.

---

## 5. Verification Method

To verify the resolution of the requested changes:

1. **Verify Missing Checkpoint Exit Code**:
   ```bash
   python3 experiments/evaluate.py --checkpoint_path /nonexistent/model.pt
   ```
   *Expected*: Non-zero exit code (1 or 2) and `FileNotFoundError` logged to stderr.

2. **Verify Missing Prediction File Exit Code**:
   ```bash
   python3 experiments/evaluate.py --prediction_file /nonexistent/file.csv
   ```
   *Expected*: Non-zero exit code (1 or 2) and `FileNotFoundError` logged to stderr.

3. **Verify Valid Test Run Fast Execution**:
   ```bash
   python3 experiments/evaluate.py --test_run --generate_tables --generate_matrices --generate_analysis --output_dir results/evaluation
   ```
   *Expected*: Exit code 0, elapsed time $< 2.0$s.

4. **Verify Full Unit Test Suite**:
   ```bash
   python3 -m unittest discover -s tests -p "test_*.py" -v
   ```
   *Expected*: All tests pass (150+ tests).

5. **Verify GPU Training Script**:
   ```bash
   bash scripts/run_gpu_training.sh --quick-test
   ```
   *Expected*: 10/10 mock experiments finish in $\le 5$s with `SUCCESS` and valid Codabench submissions.
