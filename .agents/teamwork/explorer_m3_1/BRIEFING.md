# BRIEFING — 2026-11-01T23:35:30Z

## Mission
Investigate requirements and technical design for `experiments/train.py`, CLI arguments, execution modes, and integration with data loader, model factory, metrics, and Codabench export.

## 🔒 My Identity
- Archetype: Explorer
- Roles: Investigation, Synthesis
- Working directory: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m3_1
- Original parent: df6ce377-6407-4a2d-bd18-829b36e20607
- Milestone: Milestone 3 (CLI Training Pipeline & GPU Scripts)

## 🔒 Key Constraints
- Read-only investigation — do NOT implement production code
- Analyze requirements from ORIGINAL_REQUEST.md (§R3) and PROJECT.md
- Produce comprehensive handoff report at .agents/teamwork/explorer_m3_1/handoff.md following 5 components
- Keep progress.md updated for heartbeat

## Current Parent
- Conversation ID: df6ce377-6407-4a2d-bd18-829b36e20607
- Updated: not yet

## Investigation State
- **Explored paths**:
  - `ORIGINAL_REQUEST.md` (§R1, §R2, §R3)
  - `PROJECT.md` (Architecture, contracts, milestones, file layout)
  - `src/data/loader.py` (`load_phrasis_split`, `normalize_track`, `normalize_split`, `InstanceRecord`)
  - `src/data/reversal.py` (`build_reversal_dataset`, `pair_instances`, `reverse_label`, `PairedReversalDataset`)
  - `src/data/codabench.py` (`export_codabench_submission`, `validate_codabench_submission`)
  - `src/models/factory.py` (`build_model`, `build_loss`, `build_tokenizer`, `resolve_device`, `resolve_torch_dtype`, `MockTransformerModel`, `SymmetrizedModelWrapper`)
  - `src/models/cross_encoder.py`, `siamese_crl.py`, `decoding.py`, `losses.py`
  - `src/metrics/metrics.py` (`compute_official_metrics`, `compute_official_evaluation`, `compute_weighted_f1`)
  - `tests/test_pipeline.py` (Fast CPU pipeline reference test)
  - DISPATCH files for Explorer 2 and Explorer 3
- **Key findings**:
  1. `experiments/train.py` serves as the central command-line interface for the entire research project.
  2. Needs to seamlessly support 5 methods: `baseline`, `bsa`, `crl`, `sd`, `crl_sd`.
  3. Supports both live GPU execution on remote servers (NVIDIA CUDA, FP16/BF16, Hugging Face transformers) and fast CPU deterministic execution (`--test_run` / `--fast_dev_run` in <10s).
  4. Integration contracts with `src/data/`, `src/models/`, `src/training/` (Explorer 2), and `src/metrics/` are clearly decoupled and compatible.
  5. Codabench submission export to `results/` with automatic validation (`validate_codabench_submission`) ensures 100% compliance with SemEval-2027 Task 2 official rules.
- **Unexplored areas**: None for CLI scope. Complete technical blueprint ready.

## Key Decisions Made
- Unified CLI architecture with robust `argparse` handling groups: Task/Track, Model/Backbone, Method/Reversal, Hyperparameters, Hardware/Precision, Execution Modes, Output/Logging paths.
- Track-specific model defaults: Track 1 -> `microsoft/deberta-v3-base`; Track 2 -> `dccuchile/bert-base-spanish-wwm-cased`; `--test_run` -> `mock`.
- Modular execution pipeline cleanly separating argument parsing, dataset preparation, model instantiation, training/eval orchestration, and Codabench export.

## Artifact Index
- DISPATCH.md — Task assignment from orchestrator
- BRIEFING.md — Working memory and context tracking
- progress.md — Liveness heartbeat and step tracking
- handoff.md — 5-component handoff report for worker
