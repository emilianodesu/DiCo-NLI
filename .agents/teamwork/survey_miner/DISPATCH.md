## 2026-11-01T21:42:59Z
You are the Data & Metrics Specification Miner for SemEval-2027 Task 2 (DiCo-NLI).
Your working directory is: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/survey_miner/
Orchestrator conversation ID: df6ce377-6407-4a2d-bd18-829b36e20607

Read the authoritative user request at:
/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/ORIGINAL_REQUEST.md
Also inspect /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/Guia Proyectos - PLN-2027.pdf and any relevant project files.

Investigate and document in full detail:
1. PhrasIS corpus structure, splits (train, dev, test), syntactic categories (nominal phrases, verb chains, prepositional phrases, adverbials), format, and instance identifier conventions for Track 1 (English) and Track 2 (Spanish).
2. Deterministic reversal logic:
   - EQUIVALENCE <-> EQUIVALENCE
   - FORWARD_ENTAILMENT <-> BACKWARD_ENTAILMENT
   - BACKWARD_ENTAILMENT <-> FORWARD_ENTAILMENT
   - NEGATIVE_OTHER <-> NEGATIVE_OTHER
   How pairs (A, B) and (B, A) are linked, matched, and indexed.
3. Precise mathematical and programmatic definitions of the 3 official evaluation metrics:
   - Weighted F1 across the 4 classes.
   - SoftCons (directional consistency rate between independent (A, B) and (B, A) predictions).
   - HardCons (directional accuracy rate where BOTH directions match ground truth).
4. Codabench submission format requirements: 2-column CSV/TSV `instance_id,label`, validation rules (no nulls, no empty rows).
5. Data module architecture recommendation for `src/data/` and `src/metrics/`.

Write your full findings and report to:
/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/survey_miner/report.md
Also write a handoff summary to /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/survey_miner/handoff.md.
When finished, send a message to the orchestrator (Recipient: df6ce377-6407-4a2d-bd18-829b36e20607).
