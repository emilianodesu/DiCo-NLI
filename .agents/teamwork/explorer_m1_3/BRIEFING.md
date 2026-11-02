# BRIEFING — 2026-11-01T21:55:30Z

## Mission
Design the concrete implementation blueprint for SemEval-2027 Task 2 (DiCo-NLI) consistency metrics, official parity (Weighted F1, SoftCons, HardCons, mathematical invariants), and directional transition matrix/inconsistency taxonomy for `src/metrics/metrics.py` and `src/metrics/consistency.py`.

## 🔒 My Identity
- Archetype: explorer
- Roles: investigation, synthesis
- Working directory: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m1_3
- Original parent: df6ce377-6407-4a2d-bd18-829b36e20607
- Milestone: M1 (Exploration & Design for Consistency Metrics & Official Parity)

## 🔒 Key Constraints
- Read-only investigation — do NOT implement in `src/` directly; produce comprehensive design reports and blueprints in `.agents/teamwork/explorer_m1_3/`
- Ensure mathematical invariants: HardCons <= SoftCons <= 1.0
- Exact parity with SemEval-2027 official evaluation functions and rules
- Comprehensive edge cases (unpaired samples, missing predictions, tie-breaking, multiclass weighted F1 definitions)

## Current Parent
- Conversation ID: df6ce377-6407-4a2d-bd18-829b36e20607
- Updated: not yet

## Investigation State
- **Explored paths**:
  - `ORIGINAL_REQUEST.md`, `PROJECT.md`, `survey_miner/report.md`, `survey_miner/handoff.md`, `survey_modeling/report.md`
  - Official SemEval-2027 Task 2 evaluation repo fetched from GitHub: `metrics.py`, `labels.py`, `schemas.py`, `validation.py`
- **Key findings**:
  - Official evaluation implements exact Weighted F1, Macro F1, Accuracy, and per-label confusion matrix over 4 classes.
  - Official SoftCons and HardCons evaluate strictly over `ReversiblePair` records built by deduplicating reciprocal instances via `frozenset((record.instance_id, reverse_record.instance_id))`.
  - Predictions with `NEGATIVE_OTHER` on reversible pairs receive `soft_ok = False` (since `NEGATIVE_OTHER not in REVERSIBLE_LABELS`).
  - Strict mathematical proof that `h_i = 1 ==> s_i = 1`, enforcing `HardCons <= SoftCons <= 1.0`.
  - Defined 4x4 Directional Transition Matrix with 16-cell taxonomy partitioning all transitions into Consistent (3 cells), Entailment Loop (2 cells), Equivalence Asymmetry (6 cells), and Semantic Dropout/Noise Collapse (5 cells).
- **Unexplored areas**: None. Design parameters fully mapped out.

## Key Decisions Made
- Dual-API design for `src/metrics/metrics.py`:
  1. Record/Dictionary-based API (`compute_official_evaluation`): Direct drop-in parity with official `evaluation_functions`.
  2. Vectorized/Array-based API (`compute_metrics_from_arrays`): High-throughput evaluation for PyTorch training loops without dictionary overhead.
- In `src/metrics/consistency.py`: Formulated full Directional Transition Matrix, normalized variants (row-normalized transition probabilities and joint distribution), and granular Inconsistency Taxonomy breakdown with LaTeX/JSON export capabilities.

## Artifact Index
- DISPATCH.md — incoming instructions record
- BRIEFING.md — persistent situational awareness
- progress.md — liveness heartbeat
- report.md — comprehensive blueprint for metrics and consistency analysis
- handoff.md — 5-component handoff report
