# BRIEFING — 2026-11-01T22:23:00Z

## Mission
Investigate Data Contracts & BSA Ingestion Fixes for M1 Iteration 2: resolve build_reversal_dataset row extraction bugs across pandas/SimpleDataFrame/list-of-dicts, unify InstanceRecord between loader.py and reversal.py, and design exact code patch blueprints.

## 🔒 My Identity
- Archetype: explorer
- Roles: investigation, synthesis
- Working directory: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m1_it2_1/
- Original parent: df6ce377-6407-4a2d-bd18-829b36e20607
- Milestone: M1 Iteration 2

## 🔒 Key Constraints
- Read-only investigation — do NOT modify src/ or tests/ directly
- Deliver actionable analysis report and handoff report
- Deliver exact code patch blueprints for Worker

## Current Parent
- Conversation ID: df6ce377-6407-4a2d-bd18-829b36e20607
- Updated: 2026-11-01T22:23:00Z

## Investigation State
- **Explored paths**:
  - `src/data/loader.py` (InstanceRecord, SimpleDataFrame, extract_syntactic_category)
  - `src/data/reversal.py` (build_reversal_dataset, pair_instances, ReciprocalPair)
  - `src/data/__init__.py`
  - `src/metrics/metrics.py` (compute_official_metrics, compute_consistency_report)
  - `tests/test_pipeline.py` (SRC_AVAILABLE import bypass)
  - Reviewer 1 & 2 handoffs, Challenger 1 handoff, GATE_STATUS.md
- **Key findings**:
  1. `build_reversal_dataset`: Missing upfront row extraction caused column names to be returned when `augment_bsa=False` and `dict('instance_id')` `ValueError` when `augment_bsa=True`.
  2. `InstanceRecord` duplication: Duplicate classes in `loader.py` and `reversal.py` caused `isinstance` checks to fail and sequence indexing `TypeError` on dict conversion.
  3. `NEGATIVE_OTHER` reversibility: Defaulting `is_reversible=True` broke the mathematical invariant `HardCons <= SoftCons`.
  4. Suffix false positives: Naive `-ly` check in `extract_syntactic_category` misclassified nouns like "family", "butterfly" as adverbial.
  5. Decoupled test imports: Splitting M1 imports in `tests/test_pipeline.py` fixes `SRC_AVAILABLE: True`.
- **Unexplored areas**: None for M1 data contracts.

## Key Decisions Made
- Canonical single source of truth for `InstanceRecord` is `src/data/loader.py`, providing full mapping protocol (`keys()`, `__iter__()`, `__contains__()`, `to_dict()`) and `transposed()`.
- `build_reversal_dataset` uses polymorphic row extraction supporting DataFrames, sequences of records, and dicts.
- `pair_instances` explicitly checks label reversibility, enforces `strict_reversal`, and decouples pairing passes.

## Artifact Index
- `DISPATCH.md` — incoming dispatch message log
- `progress.md` — liveness heartbeat
- `report.md` — comprehensive analysis report with exact code patch blueprints
- `handoff.md` — self-contained handoff report conforming to the 5-component protocol
