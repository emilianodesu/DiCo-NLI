# Adversarial Challenge Report — Milestone 4 Iteration 2: CLI Error Paths & Exit Codes

**Agent**: `challenger_m4_it2_1` (Role: Challenger 1 M4 It2 CLI Error Paths)  
**Milestone**: Milestone 4 Iteration 2 (Gate Remediation Verification)  
**Date**: 2026-11-02T04:05:50Z  
**Verdict**: **`APPROVE`**

---

## 1. Observation

All tests were executed directly in the environment without relying on prior logs or worker assertions.

### 1.1 Non-Existent Checkpoint Path
- Command executed:
  ```bash
  python3 experiments/evaluate.py --checkpoint_path /path/to/nonexistent.pt; echo "EXIT_CODE=$?"
  ```
- Output observed:
  ```text
  2026-11-01 22:04:27 | ERROR   | dico_nli.evaluate | Checkpoint file not found: /path/to/nonexistent.pt
  2026-11-01 22:04:27 | ERROR   | dico_nli.evaluate | Evaluation CLI failed: Checkpoint file not found: /path/to/nonexistent.pt
  FileNotFoundError: Checkpoint file not found: /path/to/nonexistent.pt
  EXIT_CODE=1
  ```
- Result: Raised `FileNotFoundError`, logged to stderr, and exited with exit code `1` (NOT `0`).

### 1.2 Non-Existent Prediction File
- Command executed:
  ```bash
  python3 experiments/evaluate.py --prediction_file /path/to/nonexistent.csv; echo "EXIT_CODE=$?"
  ```
- Output observed:
  ```text
  2026-11-01 22:04:29 | ERROR   | dico_nli.evaluate | Prediction file not found: /path/to/nonexistent.csv
  2026-11-01 22:04:29 | ERROR   | dico_nli.evaluate | Evaluation CLI failed: Prediction file not found: /path/to/nonexistent.csv
  FileNotFoundError: Prediction file not found: /path/to/nonexistent.csv
  EXIT_CODE=1
  ```
- Result: Raised `FileNotFoundError`, logged to stderr, and exited with exit code `1` (NOT `0`).

### 1.3 Non-Existent Results Directory
- Command executed:
  ```bash
  python3 experiments/evaluate.py --results_dir /path/to/nonexistent_dir; echo "EXIT_CODE=$?"
  ```
- Output observed:
  ```text
  2026-11-01 22:04:32 | ERROR   | dico_nli.evaluate | Results directory not found: /path/to/nonexistent_dir
  2026-11-01 22:04:32 | ERROR   | dico_nli.evaluate | Evaluation CLI failed: Results directory not found: /path/to/nonexistent_dir
  FileNotFoundError: Results directory not found: /path/to/nonexistent_dir
  EXIT_CODE=1
  ```
- Result: Raised `FileNotFoundError`, logged to stderr, and exited with exit code `1` (NOT `0`).

### 1.4 Valid Fast Test Run Mode
- Command executed:
  ```bash
  python3 experiments/evaluate.py --test_run --generate_tables --generate_matrices --generate_analysis; echo "EXIT_CODE=$?"
  ```
- Output observed:
  ```text
  2026-11-01 22:04:35 | INFO    | dico_nli.evaluate | Evaluation completed successfully in 0.070s
  2026-11-01 22:04:35 | INFO    | dico_nli.evaluate | Artifacts saved to: results/evaluation
  EXIT_CODE=0
  ```
- Result: Completed in `0.070s` (<0.1s budget), generated Markdown/LaTeX/ASCII tables, confusion and transition matrices, and syntactic error analysis JSON/LaTeX reports, and exited with exit code `0`.

### 1.5 Adversarial Edge-Case Stress Testing
- **Non-existent logs directory**:
  ```bash
  python3 experiments/evaluate.py --logs_dir /path/to/nonexistent_logs; echo "EXIT_CODE=$?"
  ```
  Result: Exited with code `1` and `FileNotFoundError: Logs directory not found: /path/to/nonexistent_logs`.
- **Invalid checkpoint passed with `--test_run` flag**:
  ```bash
  python3 experiments/evaluate.py --test_run --checkpoint_path /path/to/nonexistent.pt; echo "EXIT_CODE=$?"
  ```
  Result: Exited with code `1` and `FileNotFoundError: Checkpoint file not found: /path/to/nonexistent.pt`. Test run flag does not mask invalid explicit paths.
- **Invalid prediction file passed with `--test_run` flag**:
  ```bash
  python3 experiments/evaluate.py --test_run --prediction_file /path/to/nonexistent.csv; echo "EXIT_CODE=$?"
  ```
  Result: Exited with code `1` and `FileNotFoundError: Prediction file not found: /path/to/nonexistent.csv`.

### 1.6 Automated Test Suite Execution
- `python3 -m unittest tests/test_evaluation.py -v`:
  - `Ran 46 tests in 0.476s ... OK`
- `python3 -m unittest discover -s tests -p "test_*.py"`:
  - `Ran 160 tests in 1.548s ... OK`
- `bash scripts/run_gpu_training.sh --quick-test`:
  - 10/10 mock benchmark runs completed with `SUCCESS` in `00m01s`, Codabench submissions verified, exit code `0`.

---

## 2. Logic Chain

1. **Defect Remediation Verification**:
   - In Iteration 1, the CLI silently bypassed missing paths and exited with status 0, masking missing checkpoints or prediction files.
   - Observations 1.1, 1.2, and 1.3 confirm that `validate_cli_args()` now intercepts missing `--checkpoint_path`, `--prediction_file`, and `--results_dir` upfront.
   - Lines 757-760 of `experiments/evaluate.py` intercept `FileNotFoundError`, emit the formatted error to `sys.stderr`, and call `sys.exit(1)`. All three invalid paths deterministically returned `EXIT_CODE=1`.
2. **Robustness Under Flag Composition**:
   - Observation 1.5 confirms that combining `--test_run` with an invalid explicit file path does not erroneously suppress the error; validation occurs prior to pipeline execution.
3. **Execution Performance & Functional Integrity**:
   - Observation 1.4 confirms that valid invocations complete in `0.070s`, well under both the strict `<0.1s` target and the `<2.0s` smoke test requirement.
   - Observation 1.6 demonstrates zero regressions across all 160 unit/integration tests and GPU training scripts.

---

## 3. Caveats

No caveats. All four mandated test scenarios and additional adversarial edge cases were empirically tested and confirmed.

---

## 4. Conclusion

The CLI error paths in `experiments/evaluate.py` have been completely and robustly remediated. Non-existent checkpoint paths, prediction files, and results directories consistently raise `FileNotFoundError`, output error logs to stderr, and exit with status code 1. Valid test runs execute cleanly in 0.070s with status code 0.

Final Verdict: **`APPROVE`**

---

## 5. Verification Method

To re-verify independently:
```bash
# 1. Nonexistent checkpoint (must exit 1)
python3 experiments/evaluate.py --checkpoint_path /path/to/nonexistent.pt
test $? -eq 1 && echo "Checkpoint validation PASSED"

# 2. Nonexistent prediction file (must exit 1)
python3 experiments/evaluate.py --prediction_file /path/to/nonexistent.csv
test $? -eq 1 && echo "Prediction file validation PASSED"

# 3. Nonexistent results dir (must exit 1)
python3 experiments/evaluate.py --results_dir /path/to/nonexistent_dir
test $? -eq 1 && echo "Results dir validation PASSED"

# 4. Valid test run mode (must exit 0 in <0.1s)
python3 experiments/evaluate.py --test_run --generate_tables --generate_matrices --generate_analysis
test $? -eq 0 && echo "Valid test run PASSED"

# 5. Full test suite (160 tests, all passing)
python3 -m unittest discover -s tests -p "test_*.py"
```

### Invalidation Conditions
- Any of commands 1–3 exits with code 0 or fails to raise `FileNotFoundError`.
- Command 4 exits with non-zero exit code or exceeds execution time budget.
- Any regression occurs in the 160 unit tests.
