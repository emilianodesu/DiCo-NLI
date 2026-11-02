# Plan: SemEval-2027 Task 2 (DiCo-NLI) End-to-End Orchestration

## Phase 0: Survey & Technical Mapping
- [ ] Spawn 3 Explorers in parallel to map scope:
  - Explorer 1: Data schema, PhrasIS corpus, reversal logic, Codabench format, and exact metrics (Weighted F1, SoftCons, HardCons).
  - Explorer 2: Model architectures (DeBERTa-v3, BETO/RoBERTa-bne), training strategies (Naive, BSA, CRL with KL/JS), and mathematical formulation of Symmetrized Decoding (SD).
  - Explorer 3: Execution environment, GPU CLI training scripts, evaluation artifacts, ACL paper template & presentation requirements, test suite architecture.
- [ ] Merge findings into `PROJECT.md` (Feature Inventory, Architecture, Milestones, Interface Contracts) and `TEST_INFRA.md`.

## Phase 1: Dual Track Execution
### Track A: E2E Testing Suite (Tiers 1-4)
- [ ] Dispatch Test Writer / Worker to create E2E test suite (Category-Partition, Boundary Value Analysis, Pairwise, Real-World Scenarios).
- [ ] Verify test suite and publish `TEST_READY.md`.

### Track B: Implementation Track
- [ ] **Milestone 1 (R1: Data & Metrics)**:
  - `src/data/`: Dataset loaders, reversal pairing, deterministic label transpositions, Codabench generator/validator.
  - `src/metrics/`: Weighted F1, SoftCons, HardCons.
  - Reviewer + Challenger + Auditor gate.
- [ ] **Milestone 2 (R2: Modeling & Reversal Strategies)**:
  - `src/models/`: Cross-encoder baseline, Bi-directional Symmetrical Augmentation (BSA), Consistency Regularization Loss (CRL with KL/JS divergence), Symmetrized Decoding (SD).
  - Reviewer + Challenger + Auditor gate.
- [ ] **Milestone 3 (R3: Training Pipeline & GPU Scripts)**:
  - `experiments/train.py`: CLI training and evaluation with CUDA detection, fp16/bf16, checkpoints, logging.
  - `scripts/run_gpu_training.sh`: Reproducible bash launcher.
  - Fast CPU mode verification (<60s).
  - Reviewer + Challenger + Auditor gate.
- [ ] **Milestone 4 (R4: Experimental Analysis & Ablations)**:
  - Experimental evaluation across Track 1 (English) and Track 2 (Spanish).
  - Comparison tables, confusion & transition matrices, syntactic breakdown analysis.
  - Reviewer + Challenger + Auditor gate.
- [ ] **Milestone 5 (R5: Academic Deliverables & Documentation)**:
  - `report/paper.tex` and compiled `report/paper.pdf` in SemEval ACL format.
  - `presentation/presentation.pdf` conforming to presentation rubric.
  - `README.md` complete and verified.
  - Reviewer + Auditor gate.

## Phase 2: Final Acceptance & Hardening
- [ ] Run 100% E2E test suite (Tiers 1-4) against complete pipeline.
- [ ] Tier 5 Adversarial Coverage Hardening (Challengers stress-testing edge cases and symmetry).
- [ ] Final Forensic Audit verification.
- [ ] Deliver final synthesis to Sentinel / Parent.
