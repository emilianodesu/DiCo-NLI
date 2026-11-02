# Dispatch: Explorer 3 Milestone 3 (GPU Execution Script & Server Environment)

## Mission
Analyze requirements and technical architecture for `scripts/run_gpu_training.sh` and production execution environment on NVIDIA GPU servers.

## Key Documents
- Read `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/ORIGINAL_REQUEST.md` (Authoritative user requirements §R3)
- Read `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/PROJECT.md` (Project blueprint and interface contracts)

## Scope
1. Design `scripts/run_gpu_training.sh`:
   - Environment setup: CUDA environment variable configuration (`CUDA_VISIBLE_DEVICES`, PyTorch caching, thread tuning).
   - Dependency checks: Python environment, PyTorch with CUDA, Hugging Face `transformers`, `datasets`, `accelerate`.
   - Directory hierarchy creation: ensure `checkpoints/`, `logs/`, `results/`, `data/` exist.
   - Comprehensive matrix execution: Support running all 4 modeling families across both Track 1 (English) and Track 2 (Spanish) — 8 core experiments minimum:
     1. Track 1 + Baseline (`deberta-v3-base`)
     2. Track 1 + BSA (`deberta-v3-base`)
     3. Track 1 + CRL (`deberta-v3-base`)
     4. Track 1 + SD (`deberta-v3-base`) / CRL+SD
     5. Track 2 + Baseline (`BETO` / `roberta-bne`)
     6. Track 2 + BSA (`BETO` / `roberta-bne`)
     7. Track 2 + CRL (`BETO` / `roberta-bne`)
     8. Track 2 + SD (`BETO` / `roberta-bne`) / CRL+SD
   - Configurable flags in bash: `--track`, `--method`, `--quick-test`, `--parallel`, `--dry-run`.
   - Robust error handling: exit traps, logging stdout/stderr to `logs/`, timing each experiment, summary table of completed runs.
   - Codabench export verification: verify that each run outputs a valid 2-column CSV in `results/` passing Codabench validation.
2. Outline test strategy for verifying the shell script (syntax checks with `bash -n`, dry-run mode, mock execution on CPU).

## 2026-11-01T23:32:21Z
Received from orchestrator (df6ce377-6407-4a2d-bd18-829b36e20607):
You are Explorer 3 for Milestone 3 (CLI Training Pipeline & GPU Scripts).
Your working directory is: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m3_3/
Please read your assignment in: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m3_3/DISPATCH.md
Read the authoritative requirements in: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/ORIGINAL_REQUEST.md
Read the project architecture in: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/PROJECT.md

Investigate requirements and technical design for `scripts/run_gpu_training.sh` and production execution on remote NVIDIA GPU servers. Cover environment variables, dependency checks, directory creation, 8-experiment benchmark matrix execution, error handling, logging, and Codabench export verification. Write a comprehensive analysis and implementation blueprint to /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m3_3/handoff.md following the Handoff Protocol. Notify the orchestrator via send_message when done.
