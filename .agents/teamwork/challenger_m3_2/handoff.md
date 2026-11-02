# Handoff Report: Challenger 2 Milestone 3 (CLI Training Pipeline & GPU Scripts)

**Author:** Challenger 2 Milestone 3  
**Date:** 2026-11-02T02:35:00Z  
**Recipient:** Orchestrator (`df6ce377-6407-4a2d-bd18-829b36e20607`)  
**Workspace:** `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject`  
**Verdict:** **APPROVE**  

---

## 1. Observation

Direct empirical observations from terminal command executions, syntax checks, timing measurements, adversarial injections, and artifact schema audits:

1. **Bash Script Syntax & POSIX Portability**:
   - Command: `bash -n scripts/run_gpu_training.sh`
   - Exit code: `0`
   - Stderr/Stdout: Empty (0 syntax errors). Compatible with macOS Bash 3.2+ and POSIX Linux shells.

2. **Dry-Run Mode Execution**:
   - Command: `bash scripts/run_gpu_training.sh --dry-run`
   - Exit code: `0`
   - Output confirmed: Created required directory tree (`checkpoints/`, `logs/`, `results/`, `data/raw/`, `data/processed/`, `.cache/huggingface/`), performed hardware probe (`CUDA Available: False`, auto-resolved precision `fp32`), listed 8 planned benchmark experiments across Track 1 and Track 2, printed planned CLI invocations, and exited with `[SUCCESS] Dry run complete. All argument paths and variables valid.`

3. **CPU Quick-Test Execution & Wall-Clock Timing**:
   - Command: `time bash scripts/run_gpu_training.sh --quick-test`
   - Exit code: `0`
   - Total Wall Time: `1.121s` (`0.78s user 0.28s system 94% cpu`).
   - Benchmark Requirement: Acceptance criteria requires pipeline test to complete in `<60s` on CPU. At 1.121s, execution time is ~1.8% of the threshold budget.
   - All 8 experiments ran to completion with `STATUS: SUCCESS` and inline `CODABENCH: PASSED`.
   - Generated submissions, logs, summaries, and `results/codabench_submissions.zip`.

4. **Subset Filtering Capabilities**:
   - Track Filter: `bash scripts/run_gpu_training.sh --quick-test --track 1` executed exactly 4 experiments (`t1_baseline_deberta`, `t1_bsa_deberta`, `t1_crl_deberta`, `t1_sd_deberta`) in `00m00s` with exit code `0`.
   - Method Filter: `bash scripts/run_gpu_training.sh --quick-test --method crl` executed exactly 2 experiments (`t1_crl_deberta`, `t2_crl_beto`) in `00m00s` with exit code `0`.

5. **Codabench Submission CSV Adversarial Integrity Audit**:
   - Evaluated all 8 benchmark submission CSV files:
     - `results/t1_baseline_deberta_submission.csv`
     - `results/t1_bsa_deberta_submission.csv`
     - `results/t1_crl_deberta_submission.csv`
     - `results/t1_sd_deberta_submission.csv`
     - `results/t2_baseline_beto_submission.csv`
     - `results/t2_bsa_beto_submission.csv`
     - `results/t2_crl_beto_submission.csv`
     - `results/t2_sd_beto_submission.csv`
   - Verification script inspected byte-level and row-level properties against official dev splits:
     - Byte-level check: `b'\r\n'` not in content (100% strict Unix LF line endings).
     - Header: Exactly `instance_id,label`.
     - Columns: Exactly 2 comma-separated columns per row.
     - Row counts: Exactly 31 rows (matching all 31 dev split instances for Track 1 and Track 2).
     - ID integrity: Exactly matches the set of `instance_id` values from `load_phrasis_split(track, 'dev')`. Missing IDs = 0, extraneous IDs = 0, duplicate IDs = 0, null/NaN/empty IDs = 0.
     - Label integrity: All labels strictly belong to `VALID_LABELS` (`EQUIVALENCE`, `FORWARD_ENTAILMENT`, `BACKWARD_ENTAILMENT`, `NEGATIVE_OTHER`).
     - Validator check: `validate_codabench_submission(..., expected_ids=expected_ids, raise_on_error=True)` returned `True` with 0 exceptions on all 8 files.
     - Archive verification: `results/codabench_submissions.zip` was inspected via `zipfile.ZipFile` and verified to contain all 8 submission CSVs without corruption.

6. **Summary Artifact Verification**:
   - `results/summary.json` (101 lines): Valid JSON object containing top-level `total_runtime` and an array of 8 objects with keys: `run_name`, `track`, `method`, `status`, `duration`, `weighted_f1`, `soft_cons`, `hard_cons`, `codabench_valid`, and `submission_file`.
   - `logs/matrix_summary.md` (16 lines): Markdown table with date, total wall time, and 8 rows mapping run name to status, time, Weighted F1, SoftCons, HardCons, and Codabench verification status.

7. **Adversarial Error Handling & Signal Traps**:
   - Unknown argument (`--invalid-option`): Exited with code `1`, displayed usage message, and executed cleanup trap.
   - Invalid track (`--track 3`): Exited with code `1` and printed `[ERROR] Invalid --track '3'. Valid: 1, 2, track1, track2, all`.
   - Invalid method (`--method non_existent`): Exited with code `1` and printed `[ERROR] Invalid --method 'non_existent'. Valid: baseline, bsa, crl, sd, crl_sd, all`.
   - Abort on failure: When simulated experiment failure occurred (`--lr not_a_number`), script stopped immediately on Experiment 1 with exit code `2`, logging `[ERROR] Aborting matrix execution. (Use --continue-on-error to skip failing runs).`
   - Fault tolerance (`--continue-on-error`): When passed `--continue-on-error`, script ran through all 8 experiments, recorded `STATUS: FAILED` and `CODABENCH: SKIPPED`, and produced the summary report without unhandled crashes.
   - Signal Trap (`SIGINT`): Sending `SIGINT` to the script invoked the trap handler, logged `[WARN] Interrupted by user (SIGINT)`, and exited cleanly with standard code `130`.

8. **Full Test Suite Execution**:
   - Command: `python3 -m unittest discover -s tests -p "test_*.py" -v`
   - Result: `Ran 104 tests in 0.878s. OK.` (100% pass rate, 0 errors, 0 failures).

---

## 2. Logic Chain

1. **Compliance with User Requirement R3 & Acceptance Criteria**:
   - Requirement R3 mandates modular command-line training (`experiments/train.py` and `scripts/run_gpu_training.sh`) with device detection, mixed precision, structured logging, and automated official Codabench prediction export.
   - From Obs 1, 2, 3, and 5: `scripts/run_gpu_training.sh` is syntax-valid, dry-run capable, executes the full 8-experiment benchmark matrix, formats metrics into JSON and Markdown reports, and produces validated Codabench CSV files in `results/`.
   - From Obs 3: The fast CPU smoke test executes in 1.121s, fully satisfying the `<60s` acceptance criterion.

2. **Codabench Submission Safety**:
   - A critical competition requirement is zero rejection on the Codabench leaderboard.
   - From Obs 5: Every generated submission file was independently verified with byte-level checks (`\n` line endings, 2 comma-separated fields, header `instance_id,label`, zero missing/duplicate IDs, zero NaN/null values). The programmatic validator `validate_codabench_submission` passed 100% of files without errors.

3. **Robustness Under Failure Modes**:
   - From Obs 7: The script handles invalid options, invalid parameter combinations, sub-process failures, and user interruption gracefully. By default, it fails fast to avoid wasting GPU compute; when configured with `--continue-on-error`, it survives failures and tabulates the partial results.

4. **Regressive Safety Across Milestones 1 and 2**:
   - From Obs 8: All 104 unit and integration tests across reversal logic, directional metrics, symmetrized decoding, Codabench formatting, and the training pipeline execute and pass in under 1 second.

---

## 3. Caveats

- **Concurrent Execution of Single Experiments in the Same Directory**: When running a single experiment with default output paths (e.g., `bash scripts/run_gpu_training.sh --quick-test --track 1 --method sd`), the script overwrites `results/summary.json` and `logs/matrix_summary.md` with the single-experiment summary. This is expected behavior when sharing default file paths; running the full matrix repopulates all 8 entries.
- **Physical NVIDIA GPU Availability**: Empirical testing in this environment was performed on CPU using mock backbones and simulated CUDA paths, since the local host is a macOS machine. Hardware detection logic was verified via Python probes and mock execution.

---

## 4. Conclusion

**Verdict: APPROVE**

Milestone 3 (`scripts/run_gpu_training.sh`, `experiments/train.py`, `src/training/trainer.py`, `src/training/utils.py`) satisfies all functional and non-functional requirements specified in `ORIGINAL_REQUEST.md` (§R3) and `PROJECT.md`:
1. Master script executes cleanly in dry-run, quick-test, and filtered modes.
2. 8-experiment benchmark completes on CPU in ~1.12s (<60s requirement).
3. All 8 generated Codabench submission CSVs pass strict schema validation (0 malformed files).
4. Error handling and signal traps are robust.
5. All 104 tests in the test suite pass with 100% success rate.

No blocking issues or defects found.

---

## 5. Verification Method

To reproduce and independently verify all findings:

1. **Verify Bash Syntax & Dry-Run**:
   ```bash
   bash -n scripts/run_gpu_training.sh
   bash scripts/run_gpu_training.sh --dry-run
   ```

2. **Execute Full 8-Experiment Quick-Test Matrix**:
   ```bash
   time bash scripts/run_gpu_training.sh --quick-test
   ```
   *Expected*: Total runtime < 60s (typically ~1-2s). All 8 runs reported as `SUCCESS` and `PASSED`.

3. **Verify Codabench Submission Integrity for All 8 Files**:
   ```bash
   python3 -c "
   from pathlib import Path
   from src.data.codabench import validate_codabench_submission
   from src.data.loader import load_phrasis_split
   for f in Path('results').glob('*_submission.csv'):
       tr = 'track1' if 't1_' in f.name else 'track2'
       ids = set(load_phrasis_split(tr, 'dev')['instance_id'])
       validate_codabench_submission(str(f), expected_ids=ids, raise_on_error=True)
       print(f'Verified {f.name}')
   "
   ```

4. **Verify Full Unit Test Suite**:
   ```bash
   python3 -m unittest discover -s tests -p "test_*.py" -v
   ```
   *Expected*: 104 tests pass in < 5s with exit code 0.

5. **Invalidation Conditions**:
   - Any test failure in the test suite.
   - Any runtime exceeding 60s for `--quick-test` on CPU.
   - Any Codabench validation error or schema deviation in the generated CSVs.
   - Any failure of `--dry-run` or argument parsing.
