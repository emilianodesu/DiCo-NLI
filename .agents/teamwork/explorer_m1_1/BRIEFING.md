# BRIEFING — 2026-11-01T22:06:00Z

## Mission
Design concrete implementation blueprint for `src/data/loader.py` covering PhrasIS dataset ingestion (Track 1 & Track 2, train/dev/test), schema parsing, syntactic category extraction, fallback/mock data generation, data structures, and type hints for SemEval-2027 Task 2.

## 🔒 My Identity
- Archetype: explorer
- Roles: investigator, architect, synthesizer
- Working directory: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m1_1/
- Original parent: df6ce377-6407-4a2d-bd18-829b36e20607
- Milestone: M1 - Data Loading & Ingestion

## 🔒 Key Constraints
- Read-only investigation — do NOT implement code in src/ directly (propose concrete blueprint/patch/replacement)
- Write only to working directory: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m1_1/
- Must cover Track 1 (English) and Track 2 (Spanish) PhrasIS dataset format
- Handle train, dev, and test splits (including missing labels in test)
- Syntactic category extraction (nominal, verbal, prepositional, adverbial, quantificational)
- Fallback/mock data generation mechanism when running on CPU without full external downloads
- Adhere to PROJECT.md architectural guidelines

## Current Parent
- Conversation ID: df6ce377-6407-4a2d-bd18-829b36e20607
- Updated: 2026-11-01T22:06:00Z

## Investigation State
- **Explored paths**:
  - `ORIGINAL_REQUEST.md`, `PROJECT.md`, `survey_miner/report.md`, `survey_miner/handoff.md`, `survey_env_deliverables/report.md`.
  - Host environment (`Darwin 27.2.0`, Python 3.14.3, absence of `pandas`/`torch` in `.venv`).
  - Unit test implementations in `tests/`.
- **Key findings**:
  - Exact PhrasIS partition counts: Train Track 1 & 2 = 3,042 instances (1,282 reversible pairs, 478 negatives); Dev Track 1 & 2 = 660 instances (277 reversible pairs, 106 negatives).
  - Exact reciprocal frequency balance: $N(\text{FORWARD}) = N(\text{BACKWARD})$.
  - Schema requirements: 7 columns for participant CSV, 8 columns for reference CSV, 6 columns for test CSV.
  - Zero-dependency `SimpleDataFrame` allows seamless offline execution without `pandas` in CPU sandboxes.
  - Priority rule engine distinguishes verbal gerundival chains (`in destroying`, `al destruir`) from prepositional phrases (`in Spain`, `en España`).
- **Unexplored areas**:
  - Track 3 (Basque) and Track 4 (Mixed Multilingual) — intentionally out of scope per user requirements.

## Key Decisions Made
- Implemented `SimpleDataFrame` fallback to eliminate external library dependency for fast CPU smoke tests and local unit tests.
- Designed `extract_syntactic_category` with verbal precedence over prepositional rules to handle gerundival adjuncts correctly.
- Implemented round-robin category interleaving in `generate_mock_phrasis_data` so all 5 categories are present even in small sample batches ($N=5$).
- Created `verify_split_leakage` to audit train/dev pair isolation.

## Artifact Index
- `DISPATCH.md` — Incoming orchestrator instructions
- `BRIEFING.md` — Working memory and status
- `progress.md` — Heartbeat log
- `proposed_loader.py` — Complete, verified implementation blueprint for `src/data/loader.py`
- `test_proposed_loader.py` — Comprehensive unit and integration test suite (100% pass)
- `report.md` — Full architectural specification and blueprint report
- `handoff.md` — 5-component hard handoff report
