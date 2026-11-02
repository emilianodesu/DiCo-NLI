# BRIEFING — 2026-11-01T23:36:30Z

## Mission
Analyze requirements and technical design for `scripts/run_gpu_training.sh` and production execution on remote NVIDIA GPU servers.

## 🔒 My Identity
- Archetype: explorer
- Roles: investigation, synthesis
- Working directory: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m3_3/
- Original parent: df6ce377-6407-4a2d-bd18-829b36e20607
- Milestone: Milestone 3 (CLI Training Pipeline & GPU Scripts)

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- Cover environment variables, dependency checks, directory creation, 8-experiment benchmark matrix execution, error handling, logging, and Codabench export verification
- Follow Handoff Protocol (handoff.md) and notify orchestrator via send_message

## Current Parent
- Conversation ID: df6ce377-6407-4a2d-bd18-829b36e20607
- Updated: 2026-11-01T23:36:30Z

## Investigation State
- **Explored paths**: `ORIGINAL_REQUEST.md`, `PROJECT.md`, `requirements.txt`, `src/data/loader.py`, `src/data/codabench.py`, `src/metrics/metrics.py`, `src/models/factory.py`, `tests/test_pipeline.py`, environment probes (Python 3.14.3, Bash 3.2 on macOS).
- **Key findings**:
  1. Bash compatibility: Must maintain compatibility with Bash 3.2 (macOS default) while exploiting Linux server capabilities without relying on Bash 4+ `declare -A`.
  2. GPU Server Environment: Needs `CUDA_VISIBLE_DEVICES`, `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`, `TOKENIZERS_PARALLELISM="false"`, cache dirs (`HF_HOME`), and OMP/MKL thread limits.
  3. 8-Experiment Benchmark Matrix: 2 tracks (Track 1 EN DeBERTa-v3, Track 2 ES BETO/RoBERTa-bne) x 4 methods (Baseline, BSA, CRL, SD/CRL+SD).
  4. Dual-Mode Operation: Full GPU heavy training mode + fast CPU/mock smoke test mode (`--quick-test`) executing in <60s.
  5. Codabench Verification: Each completed run must be verified immediately against `src.data.codabench.validate_codabench_submission` with expected IDs.
  6. Deliverable Artifacts: Per-run log files, JSONL metric histories, Codabench submission CSVs, markdown summary table, and machine-readable JSON summary.
- **Unexplored areas**: None. Full architecture and test strategy specified.

## Key Decisions Made
- Chose delimited string arrays rather than `declare -A` to ensure dual Bash 3.2 and Bash 5 compatibility.
- Designed automated device and precision probe defaulting to BF16 on Ampere+, FP16 on Volta/Turing, and FP32 on CPU.
- Standardized Codabench verification hook directly in bash using inline Python call to `validate_codabench_submission`.

## Artifact Index
- DISPATCH.md — Task assignment and incoming messages
- BRIEFING.md — Situational awareness and working memory
- progress.md — Liveness heartbeat
- handoff.md — 5-component handoff report (complete implementation blueprint)
