# Dispatch: Explorer 3 Milestone 3 Iteration 2 (CLI, Utils & Script Hardening)

## Mission
Analyze defects in `experiments/train.py`, `src/training/utils.py`, and `scripts/run_gpu_training.sh` and provide an exact code blueprint for the remediation Worker.

## Key Defect Items to Solve
1. **Checkpoint Path Validation in `eval_only` (`experiments/train.py:359`)**:
   If `--eval_only` (or `--predict_only`) is specified with `--checkpoint_path`, verify that the path exists. If it does not exist, log an error and raise `FileNotFoundError` (or exit code 1) instead of silently evaluating an uninitialized model.
2. **Mutual Exclusion of `--fp16` and `--bf16`**:
   In `experiments/train.py`, validate that `--fp16` and `--bf16` cannot both be active simultaneously.
3. **`JsonlLogger` Serialization (`src/training/utils.py`)**:
   In `JsonlLogger.log()`, use `json.dumps(record, default=str)` so non-serializable objects (sets, tensors, paths) do not cause a crash.
4. **`scripts/run_gpu_training.sh` Hardening**:
   - In `--continue-on-error`, if any run failed, track failure count and exit with non-zero exit code at the very end unless explicitly suppressed.
   - Support `--method crl_sd` or map it cleanly in the shell matrix.

## Reference Reports
- Read `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/challenger_m3_1/handoff.md`
- Read `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/reviewer_m3_2/handoff.md`

Write your findings and exact patch specifications to `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m3_it2_3/handoff.md`.
