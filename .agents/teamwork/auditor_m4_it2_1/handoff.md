# Handoff Report — Forensic Integrity Audit: Milestone 4 Iteration 2

**Agent**: `auditor_m4_it2_1` (Role: Forensic Auditor M4 It2)  
**Milestone**: M4 Iteration 2 Gate Remediation  
**Integrity Mode**: DEVELOPMENT  
**Target Scope**:
- `experiments/evaluate.py`
- `src/evaluation/tables.py`
- `tests/test_evaluation.py`

---

## Forensic Audit Report

**Work Product**: Milestone 4 Iteration 2 (`experiments/evaluate.py`, `src/evaluation/tables.py`, `tests/test_evaluation.py`)  
**Profile**: General Project (Development Mode)  
**Verdict**: CLEAN

### Phase Results
- **Hardcoded Test Results**: PASS — No hardcoded test outputs, synthetic mock bypasses, or fixed return strings in production pathways.
- **Facade Implementations**: PASS — Upfront validation functions (`validate_cli_args`), LaTeX formatting/escaping (`_latex_escape`), and multi-model table renderers contain genuine implementations.
- **Fabricated Verification Outputs**: PASS — All artifacts and predictions are dynamically generated at runtime.
- **Path Validation & Error Handling**: PASS — Non-existent checkpoints, prediction files, and missing result directories deterministically raise `FileNotFoundError` and exit with code 1.
- **Multi-Model Data Integrity**: PASS — Side-by-side LaTeX tables dynamically inspect model counts per track and fall back to stacked layout when multiple models are present, avoiding row collisions and data loss.
- **Automated Test Suite**: PASS — 160/160 tests passing in 1.65s on CPU.
- **End-to-End GPU Script Quick Test**: PASS — 10/10 mock experiments execute and pass Codabench format verification.

---

## 1. Observation

### 1.1 Line-by-Line Static Code Inspection
1. **`experiments/evaluate.py`**:
   - Lines 256–304: `validate_cli_args(args: argparse.Namespace) -> None` inspects `args.checkpoint_path`, `args.prediction_file`, `args.results_dir`, and `args.logs_dir` using `os.path.exists()`, `os.path.isfile()`, and `os.path.isdir()`. If any specified path does not exist, it logs an error and raises `FileNotFoundError`.
   - Lines 370–374: `run_dynamic_model_evaluation` checks `if checkpoint_path is not None: if not os.path.isfile(checkpoint_path): raise FileNotFoundError(...)`.
   - Lines 681–685: Under static evaluation mode (`--test_run` False), if `len(table.results) == 0`, raises `FileNotFoundError` rather than silently substituting mock models.
   - Lines 757–764: `main()` explicitly catches `FileNotFoundError`, logs the failure, writes `FileNotFoundError: ...` to `sys.stderr`, and exits with code 1.
2. **`src/evaluation/tables.py`**:
   - Lines 665–682: `_latex_escape(text: str) -> str` replaces LaTeX reserved characters (`&`, `%`, `$`, `#`, `_`, `{`, `}`, `~`, `^`).
   - Lines 144–150: `EvaluationResult.__post_init__` enforces range bounds `[0.0, 1.0]` across `weighted_f1`, `soft_cons`, and `hard_cons`.
   - Lines 859–862 & 886: `generate_latex_table` checks `has_single_model_per_track = len(t1_models) <= 1 and len(t2_models) <= 1`. Side-by-side mode is only enabled if `has_single_model_per_track` is `True` and exactly two tracks are present; otherwise, it cleanly falls back to stacked table layout.
   - Lines 892–893: Dynamically derives LaTeX column headers from actual model names in evaluated results (`t1_name` and `t2_name`).
3. **`tests/test_evaluation.py`**:
   - Lines 286–344: Includes unit tests for multi-model side-by-side fallback (`test_multi_model_side_by_side_fallback`), dynamic model headers (`test_dynamic_model_headers_in_side_by_side_latex`), model-grouped sorting (`test_multi_model_sorting_groups_by_model`), and LaTeX special character escaping (`test_latex_special_character_escaping`).
   - Lines 968–1124: Includes CLI integration tests verifying `exit code 1` and `FileNotFoundError` for missing checkpoints, missing prediction files, missing results directories, as well as programmatic exception checking.

### 1.2 Independent Dynamic Execution Results
1. **Non-Existent Checkpoint Path Verification**:
   - Command: `python3 experiments/evaluate.py --checkpoint_path /path/to/nonexistent.pt`
   - Exit code: `1`
   - Stderr / Log output:
     ```text
     2026-11-01 22:05:22 | ERROR   | dico_nli.evaluate | Checkpoint file not found: /path/to/nonexistent.pt
     2026-11-01 22:05:22 | ERROR   | dico_nli.evaluate | Evaluation CLI failed: Checkpoint file not found: /path/to/nonexistent.pt
     FileNotFoundError: Checkpoint file not found: /path/to/nonexistent.pt
     ```
   - Arbitrary nonexistent path command: `python3 experiments/evaluate.py --checkpoint_path arbitrary_foobar_nonexistent.pt`
   - Exit code: `1`
   - Output: `FileNotFoundError: Checkpoint file not found: arbitrary_foobar_nonexistent.pt` (proves path checking is dynamic and not hardcoded).

2. **Non-Existent Prediction File Verification**:
   - Command: `python3 experiments/evaluate.py --prediction_file /path/to/nonexistent.csv`
   - Exit code: `1`
   - Stderr / Log output:
     ```text
     2026-11-01 22:05:32 | ERROR   | dico_nli.evaluate | Prediction file not found: /path/to/nonexistent.csv
     2026-11-01 22:05:32 | ERROR   | dico_nli.evaluate | Evaluation CLI failed: Prediction file not found: /path/to/nonexistent.csv
     FileNotFoundError: Prediction file not found: /path/to/nonexistent.csv
     ```
   - Arbitrary nonexistent path command: `python3 experiments/evaluate.py --prediction_file arbitrary_foobar_nonexistent.csv`
   - Exit code: `1`
   - Output: `FileNotFoundError: Prediction file not found: arbitrary_foobar_nonexistent.csv`.

3. **Full Automated Test Suite Execution**:
   - Command: `python3 -m unittest discover -s tests -p "test_*.py" -v`
   - Result:
     ```text
     Ran 160 tests in 1.650s
     OK
     ```
   - Zero test failures, zero test errors.

4. **GPU Script Automated Quick-Test**:
   - Command: `bash scripts/run_gpu_training.sh --quick-test`
   - Result: All 10 experiments completed with `SUCCESS` in 2 seconds; Codabench format checks passed 10/10.

5. **Dynamic Fast Evaluation Run**:
   - Command: `python3 experiments/evaluate.py --test_run --generate_tables --generate_matrices --generate_analysis`
   - Result: Completed in `0.064s` (<2.0s budget), generating Markdown, LaTeX, ASCII, and JSON comparison tables, confusion & transition matrices, and syntactic chunk error analyses.

6. **Static Evaluation & Multi-Model Table Rendering**:
   - Command: `python3 experiments/evaluate.py --results_dir results --logs_dir logs --generate_tables`
   - Result: Output table preserved all evaluated models across tracks (`DeBERTa-v3`, `Mock`, `non_existent_model_xyz` on Track 1; `BETO`, `Mock` on Track 2) without row collision or data overwrite. LaTeX table (`results/evaluation/comparison_table.tex`) cleanly rendered stacked tabular layout with `\_` escaping and per-model dividers.

---

## 2. Logic Chain

1. **Path Validation Robustness**:
   - Observation 1.1 showed that `validate_cli_args()` executes at step 0 of `run_evaluation()`, and `run_dynamic_model_evaluation()` verifies `os.path.isfile(checkpoint_path)`.
   - Observation 1.2 items 1 & 2 demonstrated empirically that any missing checkpoint, prediction file, or results dir causes immediate failure with exit code 1 and `FileNotFoundError`.
   - Because arbitrary path strings (`arbitrary_foobar_nonexistent.pt` and `arbitrary_foobar_nonexistent.csv`) triggered identical exit-1 behavior, there are no hardcoded string traps or bypasses.
2. **Elimination of Mock Leakage**:
   - Observation 1.1 confirmed that when `--test_run` is False, static evaluation raises `FileNotFoundError` if results are empty, and dynamic evaluation strictly loads genuine model weights or raises `FileNotFoundError`.
   - Thus, mock models are exclusively constrained to explicit `--test_run` or `--model mock` flags.
3. **Multi-Model Data Preservation**:
   - Observation 1.1 confirmed that `has_single_model_per_track` conditions the side-by-side rendering in `generate_latex_table()`.
   - Observation 1.2 item 6 proved that multi-model results are fully preserved in both Markdown and LaTeX output, with dynamic escaping of LaTeX characters.
4. **Verification Integrity**:
   - All 160 tests across the test suite passed independently in 1.65s, confirming zero regressions across all milestones.
   - All empirical checks requested in the dispatch passed with verified raw tool outputs.

---

## 3. Caveats

No caveats. All checks were verified independently through direct tool executions and static code inspections.

---

## 4. Conclusion

The work product for Milestone 4 Iteration 2 is completely compliant with all project requirements and Development Mode integrity standards.
All gate failure causes from Iteration 1 have been remediated:
- Robust, dynamic path validation and deterministic exit code 1 on missing files.
- No silent mock fallback in production evaluation paths.
- Multi-model deduplication and dynamic LaTeX headers implemented and tested.
- 100% test pass rate (160/160 tests in 1.65s).

Unconditional binary verdict: **`CLEAN`**.

---

## 5. Verification Method

To independently reproduce this verification:

```bash
# 1. Full automated unit & integration test suite (160 tests, <2.0s)
python3 -m unittest discover -s tests -p "test_*.py" -v

# 2. Verify non-existent checkpoint error path (must exit 1 with FileNotFoundError)
python3 experiments/evaluate.py --checkpoint_path /path/to/nonexistent.pt
echo "Exit code: $?"  # Expected: 1

# 3. Verify non-existent prediction file error path (must exit 1 with FileNotFoundError)
python3 experiments/evaluate.py --prediction_file /path/to/nonexistent.csv
echo "Exit code: $?"  # Expected: 1

# 4. Verify GPU automation script quick-test (10/10 mock experiments)
bash scripts/run_gpu_training.sh --quick-test
echo "Exit code: $?"  # Expected: 0

# 5. Verify dynamic fast evaluation execution budget (<2.0s)
python3 experiments/evaluate.py --test_run --generate_tables --generate_matrices --generate_analysis
echo "Exit code: $?"  # Expected: 0
```

### Invalidation Conditions
- Any test in `tests/` fails.
- `python3 experiments/evaluate.py --checkpoint_path ...` or `--prediction_file ...` returns exit code 0 when given a non-existent path.
- Evaluation on multiple models for a track silently drops rows or collides in LaTeX output.
- Any hardcoded test bypass or dummy mock exception is found.
