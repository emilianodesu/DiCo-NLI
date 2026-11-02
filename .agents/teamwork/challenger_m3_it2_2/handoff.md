# Adversarial Re-Test Handoff Report: Milestone 3 Iteration 2 (GPU Script & Matrix Re-Testing)

**Agent**: Challenger 2 (`challenger_m3_it2_2`)  
**Role**: EMPIRICAL CHALLENGER (critic, specialist)  
**Recipient**: Orchestrator (`df6ce377-6407-4a2d-bd18-829b36e20607`)  
**Date**: 2026-11-02T03:04:00Z  
**Verdict**: **APPROVE**  

---

## Challenge Summary

**Overall risk assessment**: **LOW** (All acceptance criteria empirically verified; 0 defects found)

### Challenges Evaluated

1. **10-Experiment Benchmark Matrix Completeness**:
   - *Hypothesis*: `scripts/run_gpu_training.sh` might omit `crl_sd` or misconfigure track bindings.
   - *Stress test*: Inspected `EXPERIMENT_MATRIX` in `scripts/run_gpu_training.sh` (lines 371–382), executed `--dry-run`, and ran full matrix execution.
   - *Result*: **PASS**. Exactly 10 experiments are defined (5 for Track 1, 5 for Track 2), including `crl_sd` for both tracks.

2. **Failure Accounting under `--continue-on-error`**:
   - *Hypothesis*: `--continue-on-error` might mask execution failures by returning exit code 0.
   - *Stress test*: Injected failing python interpreter (`/usr/bin/false`) and corrupted Codabench CSV files.
   - *Result*: **PASS**. `FAILED_COUNT` tracks failures; script exits with exit code 1 in both command failure and validation failure cases.

3. **Execution Runtime Performance**:
   - *Hypothesis*: Quick-test might exceed the 60s CPU threshold or fail on CPU-only machines.
   - *Stress test*: Executed `bash scripts/run_gpu_training.sh --quick-test` on macOS CPU host.
   - *Result*: **PASS**. All 10 experiments completed in **2 seconds** (well within the <60s limit).

4. **Codabench Submission File Integrity**:
   - *Hypothesis*: Predictions exported by the script might contain invalid headers, empty rows, or ID mismatches.
   - *Stress test*: Programmatic audit using `src.data.codabench.validate_codabench_submission` against all 39 CSVs in `results/` and the packaged `results/codabench_submissions.zip`.
   - *Result*: **PASS**. 0 malformed files found; all 10 active runs packaged properly in the zip archive.

5. **Full Unit Test Suite Regression**:
   - *Hypothesis*: Worker edits might have broken existing test assertions in other modules.
   - *Stress test*: Executed `python3 -m unittest discover -s tests -p "test_*.py" -v`.
   - *Result*: **PASS**. 114/114 tests passed in 1.355s.

---

## 1. Observation

### 1.1 10-Experiment Benchmark Matrix (`scripts/run_gpu_training.sh:371-382`)
Lines 371–382 define the complete matrix:
```bash
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
- Command: `bash scripts/run_gpu_training.sh --dry-run`
  - Output: `Planned Benchmark Matrix (10 Experiments)`
  - Confirmed 10 experiments planned with correct backbones (`microsoft/deberta-v3-base` for Track 1, `dccuchile/bert-base-spanish-wwm-cased` for Track 2).
  - Exit code: `0`.

### 1.2 Failure Accounting under `--continue-on-error`
- Test 1 (Execution failure tracking):
  - Command: `bash scripts/run_gpu_training.sh --python "/usr/bin/false" --continue-on-error --track 1`
  - Observed output:
    ```
    [ERROR] Training failed for t1_baseline_deberta with exit code 1.
    [ERROR] Training failed for t1_bsa_deberta with exit code 1.
    [ERROR] Training failed for t1_crl_deberta with exit code 1.
    [ERROR] Training failed for t1_sd_deberta with exit code 1.
    [ERROR] Training failed for t1_crl_sd_deberta with exit code 1.
    ...
    [ERROR] Benchmark matrix finished with 5 failed or invalid experiment(s).
    [ERROR] Pipeline failed with exit code 1.
    ```
  - Exit code: `1`.
- Test 2 (Immediate abort without continue-on-error):
  - Command: `bash scripts/run_gpu_training.sh --python "/usr/bin/false" --track 1`
  - Observed output: Aborted immediately after experiment 1 with `[ERROR] Aborting matrix execution. (Use --continue-on-error to skip failing runs).`
  - Exit code: `1`.
- Test 3 (Codabench validation failure tracking):
  - Command: Invoked `run_gpu_training.sh` with wrapper corrupting the output CSV header to `MALFORMED_HEADER,EXTRA`.
  - Observed output:
    ```
    [ERROR] Codabench submission format verification FAILED: VALIDATION_FAILED: Invalid header ['MALFORMED_HEADER', 'EXTRA']. Expected exactly ['instance_id', 'label']
    RUN NAME: t1_sd_deberta | STATUS: INVALID_OUTPUT | CODABENCH: FAILED
    [ERROR] Benchmark matrix finished with 1 failed or invalid experiment(s).
    [ERROR] Pipeline failed with exit code 1.
    ```
  - Exit code: `1`.

### 1.3 Quick-Test Execution Performance
- Command: `bash scripts/run_gpu_training.sh --quick-test`
- Execution time: **00m02s** (2 seconds wall clock time).
- 10/10 experiments completed with status `SUCCESS`.
- Summary outputs generated:
  - `logs/matrix_summary.md` (Markdown table with all 10 experiments, duration, and metrics).
  - `results/summary.json` (Structured JSON containing runtime and all 10 experiment entries).
  - `results/codabench_submissions.zip` (Zip containing exactly 10 active CSV files).
- Symmetrized decoding verified:
  - `t1_sd_deberta`: `SoftCons = 1.0000`
  - `t1_crl_sd_deberta`: `SoftCons = 1.0000`
  - `t2_sd_beto`: `SoftCons = 1.0000`
  - `t2_crl_sd_beto`: `SoftCons = 1.0000`

### 1.4 Codabench Submission CSV Audit
- Command: Executed Python audit invoking `src.data.codabench.validate_codabench_submission` across all 39 CSV files in `results/`.
- Observed result:
  - 10 matrix submission files (`t1_*` and `t2_*`): all 31 rows validated, matching dev split instance IDs, valid class labels (`EQUIVALENCE`, `FORWARD_ENTAILMENT`, `BACKWARD_ENTAILMENT`, `NEGATIVE_OTHER`).
  - Total malformed files: **0**.
  - Confirmed `results/codabench_submissions.zip` contains exactly the 10 active matrix CSVs:
    1. `t1_baseline_deberta_submission.csv`
    2. `t1_bsa_deberta_submission.csv`
    3. `t1_crl_deberta_submission.csv`
    4. `t1_sd_deberta_submission.csv`
    5. `t1_crl_sd_deberta_submission.csv`
    6. `t2_baseline_beto_submission.csv`
    7. `t2_bsa_beto_submission.csv`
    8. `t2_crl_beto_submission.csv`
    9. `t2_sd_beto_submission.csv`
    10. `t2_crl_sd_beto_submission.csv`

### 1.5 Unit Test Suite
- Command: `python3 -m unittest discover -s tests -p "test_*.py" -v`
- Result: **Ran 114 tests in 1.355s — OK**.
- 100% pass rate, 0 failures, 0 errors.

### 1.6 Additional CLI Edge Case Verification
- `bash scripts/run_gpu_training.sh --track 99` -> Exits 1 with clear validation error.
- `bash scripts/run_gpu_training.sh --method nonesuch` -> Exits 1 with clear validation error.
- `python3 experiments/train.py --fp16 --bf16 --test_run` -> Exits 2 with parser error (`Cannot specify both --fp16 and --bf16`).
- `python3 experiments/train.py --eval_only --checkpoint_path /tmp/nonexistent_ckpt.pt --test_run` -> Exits 1 with `FileNotFoundError`.

---

## 2. Logic Chain

1. Observations in §1.1 demonstrate that the matrix is expanded to all 10 required combinations across both tracks, satisfying requirement R3 and the benchmark specification.
2. Observations in §1.2 confirm that `scripts/run_gpu_training.sh` does not mask errors under `--continue-on-error`: `FAILED_COUNT` is incremented on subprocess error codes as well as validation errors, and the script terminates with exit code 1 whenever `FAILED_COUNT > 0`.
3. Observations in §1.3 demonstrate that the complete 10-experiment suite executes in 2 seconds on CPU in `--quick-test` mode, comfortably beating the <60s performance budget.
4. Observations in §1.4 confirm that all submission files strictly adhere to the official Codabench 2-column format without nulls, empty lines, or corrupt IDs, and that `results/codabench_submissions.zip` correctly bundles active predictions.
5. Observations in §1.5 and §1.6 demonstrate that no regressions were introduced to the test suite and CLI parameter validations enforce strict constraints.

---

## 3. Caveats

- Tests were run in a CPU environment with mock backbones. Full NVIDIA GPU multi-node training requires remote CUDA hardware, though CUDA detection, precision branching, and device allocation logic were verified statically and through unit tests (`tests/test_training.py`).

---

## 4. Conclusion

The implementation and remediation in `scripts/run_gpu_training.sh`, `experiments/train.py`, and related modules fully satisfy all requirements with 0 defects identified.

**Verdict**: **APPROVE**

---

## 5. Verification Method

To reproduce the verification results:

```bash
# 1. Run complete unit test suite (114 tests)
python3 -m unittest discover -s tests -p "test_*.py" -v

# 2. Test dry-run output (10 experiments)
bash scripts/run_gpu_training.sh --dry-run

# 3. Test quick-test execution (<60s CPU benchmark)
bash scripts/run_gpu_training.sh --quick-test

# 4. Verify Codabench submissions
python3 -c "
import glob
from src.data.codabench import validate_codabench_submission
for f in sorted(glob.glob('results/*_submission.csv')):
    validate_codabench_submission(f, raise_on_error=True)
print('All submissions valid.')
"

# 5. Verify continue-on-error failure accounting exits 1
bash scripts/run_gpu_training.sh --python "/usr/bin/false" --continue-on-error --track 1
echo "Exit code: $?"  # Must be 1
```
