# Dispatch: Explorer 1 Milestone 3 (CLI, Arguments & Pipeline Architecture)

## Mission
Analyze requirements and existing code for `experiments/train.py`, argument parsing, CLI flags, configuration handling, and integration with `src/data/`, `src/models/`, and `src/metrics/`.

## Key Documents
- Read `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/ORIGINAL_REQUEST.md` (Authoritative user requirements §R3)
- Read `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/PROJECT.md` (Project blueprint and interface contracts)
- Inspect `src/models/factory.py`, `src/data/loader.py`, `src/data/codabench.py`, `src/metrics/metrics.py`

## Scope
1. Define the complete CLI interface for `experiments/train.py` covering:
   - `--track` (`1`, `2`, `en`, `es`)
   - `--model` (HuggingFace backbone or `mock`)
   - `--method` (`baseline`, `bsa`, `crl`, `sd`, `crl_sd`)
   - `--batch_size`, `--learning_rate` / `--lr`, `--epochs`, `--warmup_ratio`, `--weight_decay`, `--seed`
   - `--fp16`, `--bf16`, `--device` (auto-detection: `cuda`, `mps`, `cpu`)
   - `--eval_only`, `--predict_only`, `--test_run` / `--fast_dev_run` (for <60s local testing)
   - `--output_dir`, `--checkpoint_dir`, `--log_dir`, `--predictions_dir`
2. Design the workflow from CLI arguments to data loading, model building, training/evaluation, and saving official Codabench prediction files to `results/`.
3. Provide concrete code architecture, file paths, and interface specifications for the implementation Worker.
4. Write your findings to `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m3_1/handoff.md`.


## 2026-11-01T23:32:21Z
You are Explorer 1 for Milestone 3 (CLI Training Pipeline & GPU Scripts).
Your working directory is: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m3_1/
Please read your assignment in: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m3_1/DISPATCH.md
Read the authoritative requirements in: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/ORIGINAL_REQUEST.md
Read the project architecture in: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/PROJECT.md

Investigate requirements and technical design for `experiments/train.py`, CLI arguments, execution modes, and integration with data loader, model factory, and Codabench export. Write a comprehensive analysis and implementation blueprint to /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m3_1/handoff.md following the Handoff Protocol. Notify the orchestrator via send_message when done.
