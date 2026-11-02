# BRIEFING — 2026-11-02T03:18:00Z

## Mission
Investigate and design `src/evaluation/matrices.py` for multiclass confusion matrix and directional transition matrix analysis, integration with prediction DataFrames/Codabench CSVs, and unit test plan.

## 🔒 My Identity
- Archetype: explorer
- Roles: M4 Confusion & Transition Matrices Explorer
- Working directory: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m4_2
- Original parent: df6ce377-6407-4a2d-bd18-829b36e20607
- Milestone: Milestone 4 (R4: Experimental Evaluation & Error Analysis)

## 🔒 Key Constraints
- Read-only investigation — do NOT implement directly in src/ (design and propose only)
- Follow Handoff Protocol (Observation, Logic Chain, Caveats, Conclusion, Verification Method)
- .agents/teamwork/ holds only metadata

## Current Parent
- Conversation ID: df6ce377-6407-4a2d-bd18-829b36e20607
- Updated: 2026-11-02T03:10:23Z

## Investigation State
- **Explored paths**:
  - `ORIGINAL_REQUEST.md` (§R4.2)
  - `PROJECT.md` (§Architecture, §Feature Inventory, §Code Layout)
  - `src/metrics/consistency.py` & `src/metrics/metrics.py` (existing transition and classification logic)
  - `src/data/loader.py`, `src/data/reversal.py`, `src/data/codabench.py` (data structures, pairing logic, SimpleDataFrame, CSV export/import)
  - `src/models/decoding.py` (involution matrix math and Symmetrized Decoding)
  - `tests/test_consistency.py`, `tests/test_metrics.py`
- **Key findings**:
  - Canonical label sequence for evaluation/reporting is `("EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER")`.
  - In this canonical order, the ideal involution matrix is $P = [[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]$.
  - Inconsistency diagnostics formulated: Violation Rate, Frobenius Distance (raw and normalized in $[0,1]$), Mean Total Variation Distance (TVD), Mean Shannon Transition Entropy (bits), Reversal Curse Sensitivity Index (RCSI), and 16-cell taxonomy.
  - Exporters designed: ASCII tables, LaTeX tabular format conforming to SemEval/ACL template with bolding on involution cells and underline on loops, and JSON serialization.
  - Integration designed: `load_predictions` supports Codabench 2-column CSVs, Pandas and SimpleDataFrames, and dict mappings. `extract_reciprocal_pairs_from_dataset` extracts direct/inverse pairs. `evaluate_matrices_from_data` unifies the pipeline.
  - Created and empirically verified `proposed_matrices.py` and `test_proposed_matrices.py` (11 tests pass in 0.006s).
- **Unexplored areas**: None for M4 matrices scope.

## Key Decisions Made
- Canonical presentation order is EQUIVALENCE, FORWARD_ENTAILMENT, BACKWARD_ENTAILMENT, NEGATIVE_OTHER; dynamic `get_ideal_involution_matrix(labels)` also supports custom orders like BWD-first.
- All code strictly supports pure-Python standard library with optional numpy/pandas acceleration and `SimpleDataFrame` support.
- Fully written and tested reference implementation placed in `proposed_matrices.py` for worker consumption.

## Artifact Index
- `DISPATCH.md` — Initial dispatch message
- `BRIEFING.md` — Working memory
- `progress.md` — Heartbeat and status
- `proposed_matrices.py` — Complete drop-in code candidate for `src/evaluation/matrices.py`
- `test_proposed_matrices.py` — 11-test verification suite (passes 100%)
- `handoff.md` — Comprehensive handoff report
