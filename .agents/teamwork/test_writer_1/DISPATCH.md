# Task Assignment: E2E Testing Track (Tiers 1-4 Test Suite)

- Agent: test_writer_1 (teamwork_preview_test_writer)
- Working directory: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/test_writer_1/
- Request reference: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/ORIGINAL_REQUEST.md
- Project reference: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/orchestrator_1/PROJECT.md
- Survey reference: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/survey_miner/report.md
- Survey reference: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/survey_modeling/report.md
- Survey reference: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/survey_env_deliverables/report.md
- Orchestrator: df6ce377-6407-4a2d-bd18-829b36e20607

## 2026-11-01T21:52:33Z
You are the E2E Test Suite Writer for SemEval-2027 Task 2 (DiCo-NLI).
Your working directory is: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/test_writer_1/
Orchestrator conversation ID: df6ce377-6407-4a2d-bd18-829b36e20607

Read:
- /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/ORIGINAL_REQUEST.md
- /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/orchestrator_1/PROJECT.md
- /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/survey_miner/report.md
- /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/survey_modeling/report.md
- /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/survey_env_deliverables/report.md

Your objectives:
1. Ensure /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/PROJECT.md exists at project root (replicate content from /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/orchestrator_1/PROJECT.md).
2. Create /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/TEST_INFRA.md following the E2E Test Infra template (Category-Partition, BVA, Pairwise, Real-World Scenarios).
3. Implement the comprehensive test suite in `tests/`:
   - `tests/test_reversal_logic.py`: Testing label inversion mapping, pair identification, reciprocal index pairing, BSA pair generation.
   - `tests/test_metrics.py`: Testing Weighted F1, SoftCons, HardCons, edge cases (empty, non-reversible labels, perfect consistency, total inconsistency, math bound HardCons <= SoftCons).
   - `tests/test_codabench.py`: Validating format of submission files (two columns instance_id,label, no header errors, no trailing blanks, no nulls, invalid format detection).
   - `tests/test_symmetrized_decoding.py`: Testing mathematical invariant that Symmetrized Decoding strictly produces SoftCons = 1.0 on arbitrary probability distributions.
   - `tests/test_pipeline.py`: Fast CPU end-to-end smoke test executing in <60 seconds on CPU without external downloads or network access.
   Ensure test files are compatible with standard `python3 -m unittest` and `pytest`.
4. Run the test suite and verify test results.
5. Create /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/TEST_READY.md at project root with test command, tier breakdown, and coverage summary.
6. Write your handoff report to:
   /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/test_writer_1/handoff.md
   and message the orchestrator (Recipient: df6ce377-6407-4a2d-bd18-829b36e20607).
