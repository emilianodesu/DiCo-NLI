# BRIEFING — 2026-11-01T21:52:00Z

## Mission
Mine and document the exhaustive data and metrics specifications for SemEval-2027 Task 2 (DiCo-NLI).

## 🔒 My Identity
- Archetype: Specification Miner
- Roles: Data & Metrics Specification Miner
- Working directory: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/survey_miner/
- Original parent: df6ce377-6407-4a2d-bd18-829b36e20607
- Milestone: SemEval-2027 Task 2 (DiCo-NLI) Data & Metrics Specification Mining

## 🔒 Key Constraints
- Discover and document features by probing the authoritative specification. Do NOT implement anything.
- Prioritize authoritative sources over LLM prior knowledge.
- Report table format: Features Discovered and Edge Cases.
- Handoff report in handoff.md with 5 sections: Observation, Logic Chain, Caveats, Conclusion, Verification Method.
- Send message to orchestrator when finished.

## Current Parent
- Conversation ID: df6ce377-6407-4a2d-bd18-829b36e20607
- Updated: not yet

## Task Summary
- **What to build**: Full specification report covering PhrasIS corpus structure and identifier rules (Track 1 & Track 2), deterministic reversal mapping logic and pair linking, mathematical & algorithmic definitions of Weighted F1, SoftCons, and HardCons, Codabench submission formats and validation rules, and data module architecture recommendations for `src/data/` and `src/metrics/`.
- **Success criteria**: Exhaustive, mathematically precise, code-level actionable report and handoff.
- **Interface contracts**: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/ORIGINAL_REQUEST.md
- **Code layout**: src/data/ and src/metrics/

## Loaded Skills
- None specified in dispatch prompt.

## Key Decisions Made
- Fully mined primary task website, official GitHub repository, official `evaluation_functions/` package, LaTeX task proposals, and data splits.
- Rigorously documented the distinct treatment of `NEGATIVE_OTHER` in official evaluation (excluded from SoftCons/HardCons) vs. data augmentation (symmetric mapping).
- Produced mathematical proof and programmatic formulation showing $h_i = 1 \implies s_i = 1$ ($\HardCons \le \SoftCons$).
- Formulated exact architecture for `src/data/` and `src/metrics/` including Symmetrized Decoding (SD) which provably delivers $\SoftCons = 1.0$.

## Artifact Index
- report.md — Comprehensive data and metrics specification report
- handoff.md — 5-component handoff summary
- progress.md — Liveness heartbeat
- DISPATCH.md — Log of dispatch instructions
