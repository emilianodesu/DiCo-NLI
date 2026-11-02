# Task Assignment: Environment, Pipeline & Deliverables Explorer (Survey Phase)

- Working directory: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/survey_env_deliverables/
- Request reference: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/ORIGINAL_REQUEST.md
- Orchestrator: df6ce377-6407-4a2d-bd18-829b36e20607

## 2026-11-01T21:42:59Z
You are the Pipeline, CLI & Deliverables Explorer for SemEval-2027 Task 2 (DiCo-NLI).
Your working directory is: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/survey_env_deliverables/
Orchestrator conversation ID: df6ce377-6407-4a2d-bd18-829b36e20607

Read the authoritative user request at:
/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/ORIGINAL_REQUEST.md

Investigate and document in full detail:
1. Environment and dependencies: Inspect Python environment, available packages in .venv or system, PyTorch/Transformers status, CUDA/MPS/CPU availability, and how to structure a fast CPU test pipeline (<60s) using tiny/mock models or synthetic batches.
2. CLI and Script Architecture:
   - `experiments/train.py`: CLI arguments (--track, --model, --method, --batch_size, --fp16, --lr, --epochs, --eval_only, etc.), automatic device detection (cuda vs cpu/mps), mixed precision, logging to `logs/`, checkpoint saving to `checkpoints/`, and prediction export to `results/`.
   - `scripts/run_gpu_training.sh`: Reproducible bash entrypoint that handles environment setup and runs complete training/evaluation across both tracks.
3. Evaluation outputs & artifacts:
   - Comparison tables across all methods (Weighted F1, SoftCons, HardCons).
   - Confusion matrices and Directional Transition Inconsistency matrices.
   - Syntactic category error analysis (nominal, verbal, prepositional, adverbial).
4. Academic deliverables & testing:
   - `report/paper.tex` in SemEval/ACL LaTeX format, required sections (Title, Abstract, Intro, Corpus, Methodology with rigorous math, Results, Error Analysis, Conclusions, References).
   - `presentation/presentation.pdf` (LaTeX Beamer or Marp/report format compiled to PDF) covering TOC, Intro, Corpus, Methodology, Results, Conclusions.
   - Test suite structure in `tests/`: `tests/test_reversal_logic.py`, `tests/test_metrics.py`, `tests/test_pipeline.py`.

Write your full findings and report to:
/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/survey_env_deliverables/report.md
Also write a handoff summary to /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/survey_env_deliverables/handoff.md.
When finished, send a message to the orchestrator (Recipient: df6ce377-6407-4a2d-bd18-829b36e20607).
