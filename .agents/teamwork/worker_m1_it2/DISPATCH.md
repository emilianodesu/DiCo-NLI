## 2026-11-01T22:27:59Z
You are the Implementation Worker for Milestone 1 Iteration 2 (Data Pipeline & Consistency Metrics Rework).
Your working directory is: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/worker_m1_it2/
Orchestrator conversation ID: df6ce377-6407-4a2d-bd18-829b36e20607

Read:
- /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/ORIGINAL_REQUEST.md
- /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/PROJECT.md
- /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/orchestrator_1/GATE_STATUS.md
- /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m1_it2_1/report.md
- /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m1_it2_2/report.md
- /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m1_it2_3/report.md

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

File write ownership (you own exclusively):
- src/data/reversal.py
- src/data/loader.py
- src/metrics/metrics.py
- tests/test_reversal_logic.py
- tests/test_pipeline.py

Implement the concrete blueprints from the 3 Iteration 2 Explorers:
1. `src/data/reversal.py`:
   - Import `InstanceRecord` from `src.data.loader` (eliminating duplicate class definition).
   - In `build_reversal_dataset`: cleanly extract rows from DataFrame, SimpleDataFrame, or list of dicts. When `augment_bsa=False`, return genuine dataset records (not column names). When `augment_bsa=True`, create reciprocal inverted records with mapped labels and unique `instance_id` (e.g. `__bsa`).
   - In `ReciprocalPair.is_reversible`: set to `False` if either first_label or second_label is `NEGATIVE_OTHER` (or not in `REVERSIBLE_LABELS`). Add duck-typing properties (`first_id`, `second_id`, `first_label`, `second_label`) matching `compute_consistency_report`.
   - In `pair_instances`: make `strict_reversal` active: if `strict_reversal=True`, validate that label pairs follow `reverse_label` algebra, raising `ValueError` on contradictions. If `False`, mark contradictory pairs as `is_reversible=False`. Decouple fallback pairing so missing `reverse_pair_id` rows are safely paired by `pair_id` and text swap.
2. `src/metrics/metrics.py`:
   - In `compute_official_metrics`: when `is_reversible_mask` contains 0 True values (no reversible pairs in slice), return `soft_cons = 0.0` and `hard_cons = 0.0` instead of raising `ValueError`. Add length checks and universal pair duck-typing.
3. `tests/test_pipeline.py`:
   - Ensure `SRC_AVAILABLE` evaluates to `True` for live M1 components (`src.data`, `src.metrics`), while cleanly handling unwritten M2 models without failing imports.
4. Execute test suite:
   `python3 -m unittest discover -s tests -p "test_*.py" -v`
   Verify 100% test pass rate.
5. Write your handoff report to:
   /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/worker_m1_it2/handoff.md
   and message orchestrator (df6ce377-6407-4a2d-bd18-829b36e20607) when complete.
