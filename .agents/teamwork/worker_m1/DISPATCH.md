# Task Assignment: Worker M1 (Data Pipeline & Consistency Metrics Implementation)

- Agent: worker_m1 (teamwork_preview_worker)
- Working directory: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/worker_m1/
- Request reference: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/ORIGINAL_REQUEST.md
- Project reference: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/PROJECT.md
- Explorer 1 (Loader): /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m1_1/report.md
- Explorer 2 (Reversal & Codabench): /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m1_2/report.md
- Explorer 3 (Metrics & Parity): /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m1_3/report.md
- Orchestrator: df6ce377-6407-4a2d-bd18-829b36e20607
## 2026-11-01T22:01:14Z
You are the Implementation Worker for Milestone 1 (M1: Data Pipeline & Consistency Metrics) for SemEval-2027 Task 2 (DiCo-NLI).
Your working directory is: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/worker_m1/
Orchestrator conversation ID: df6ce377-6407-4a2d-bd18-829b36e20607

Read:
- /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/ORIGINAL_REQUEST.md
- /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/PROJECT.md
- /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m1_1/report.md
- /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m1_2/report.md
- /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m1_3/report.md

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

File write ownership (you own exclusively):
- src/data/__init__.py
- src/data/loader.py
- src/data/reversal.py
- src/data/codabench.py
- src/metrics/__init__.py
- src/metrics/metrics.py
- src/metrics/consistency.py
- data/ directory (mock or synthetic generator artifacts)

Implementation goals:
1. `src/data/loader.py`:
   - Load PhrasIS dataset splits (train, dev, test) for Track 1 (English) and Track 2 (Spanish).
   - Strict parsing, multilingual syntactic category extraction (nominal, verbal, prepositional, adverbial, quantificational).
   - Mock/synthetic fallback generation for offline CPU execution and sandbox environments.
   - Zero external dependency fallback (`SimpleDataFrame`) when pandas is unavailable, but using pandas if installed.
2. `src/data/reversal.py`:
   - Bijective reversal algebra:
     EQUIVALENCE <-> EQUIVALENCE
     FORWARD_ENTAILMENT <-> BACKWARD_ENTAILMENT
     BACKWARD_ENTAILMENT <-> FORWARD_ENTAILMENT
     NEGATIVE_OTHER <-> NEGATIVE_OTHER (for BSA training)
   - Canonical pair keying and reciprocal pairing (`pair_instances`).
   - BSA dataset augmentation (`build_reversal_dataset`).
3. `src/data/codabench.py`:
   - Exporter creating strict 2-column CSV/TSV `instance_id,label` with LF endings, zero trailing blanks, zero index.
   - Format validator checking no empty lines, no NaN/null, valid label enum.
4. `src/metrics/metrics.py`:
   - Official Weighted F1 across the 4 classes.
   - Official SoftCons (directional consistency rate over unique reversible source pairs).
   - Official HardCons (strict paired correctness rate where both (A,B) and (B,A) match ground truth).
   - Strict mathematical invariant: 0.0 <= HardCons <= SoftCons <= 1.0.
5. `src/metrics/consistency.py`:
   - 4x4 Directional Transition Matrix computation.
   - Inconsistency categorization and transition statistics.
6. Run the test suite:
   `python3 -m unittest discover -s tests -p "test_*.py" -v`
   Verify 100% test pass rate.
7. Write your handoff report to:
   /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/worker_m1/handoff.md
   and send a message to orchestrator (Recipient: df6ce377-6407-4a2d-bd18-829b36e20607) when complete.
