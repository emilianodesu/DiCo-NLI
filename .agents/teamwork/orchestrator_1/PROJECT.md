# Project: SemEval-2027 Task 2 (DiCo-NLI)

## Architecture
The system is structured as a modular, decoupled research and production library with clean separation of concerns:
- `src/data/`: Data ingestion, PhrasIS corpus processing, reciprocal pairing, label inversion algebra, and strict Codabench format exporting/validation.
- `src/metrics/`: Official evaluation metrics (Weighted F1, SoftCons, HardCons), confusion matrices, and directional consistency verification.
- `src/models/`: Neural architectures and reversal strategies:
  - Naive Cross-Encoder baseline (DeBERTa-v3 for English, BETO/RoBERTa-bne for Spanish).
  - Bi-directional Symmetrical Augmentation (BSA).
  - Consistency Regularization Loss (CRL) with Siamese/dual-stream encoding and involution permutation matrix $P$.
  - Symmetrized Decoding (SD) post-hoc projection guaranteeing SoftCons = 100%.
  - Abstract model factory with local mock support for fast CPU execution (<60s) and full PyTorch/CUDA support for remote GPU execution.
- `src/training/`: Reproducible training loop, mixed precision (FP16/BF16), learning rate schedules, and checkpointing.
- `src/evaluation/`: Evaluation pipelines, master comparison tables, transition matrices, and syntactic chunk error analysis (nominal, verbal, prepositional, adverbial).
- `experiments/`: CLI entrypoints (`experiments/train.py`, `experiments/evaluate.py`).
- `scripts/`: Remote GPU execution scripts (`scripts/run_gpu_training.sh`).
- `tests/`: Automated unit and integration test suite covering reversal logic, metrics, Codabench format, and fast CPU pipeline.
- `report/`: Academic paper in ACL SemEval format (`report/paper.tex` and compiled `report/paper.pdf`).
- `presentation/`: Presentation slides (`presentation/presentation.tex` and compiled `presentation/presentation.pdf`).

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Data Loading & Ingestion | Load PhrasIS train and dev splits for Track 1 (EN) and Track 2 (ES) | M1 | ORIGINAL_REQUEST §R1 |
| 2 | Deterministic Reversal Pairing | Pair (A,B) with (B,A) and apply bijective label reversal algebra | M1 | ORIGINAL_REQUEST §R1.1 |
| 3 | Codabench Exporter & Validator | Strict 2-column `instance_id,label` formatting without blank rows or nulls | M1 | ORIGINAL_REQUEST §R1.3 |
| 4 | Directional Consistency Metrics | Exact implementation of Weighted F1, SoftCons, and HardCons | M1 | ORIGINAL_REQUEST §R1.2 |
| 5 | Naive Cross-Encoder Baseline | Standard unidirectional cross-encoder fine-tuning | M2 | ORIGINAL_REQUEST §R2.1 |
| 6 | Bi-directional Symmetrical Augmentation (BSA) | Doubled training set with transposed pairs and inverted labels | M2 | ORIGINAL_REQUEST §R2.2 |
| 7 | Consistency Regularization Loss (CRL) | Siamese/dual-stream network with involution matrix P and KL/JS divergence loss | M2 | ORIGINAL_REQUEST §R2.3 |
| 8 | Symmetrized Decoding (SD) | Equivariant post-hoc projection with mathematical proof of SoftCons=100% | M2 | ORIGINAL_REQUEST §R2.4 |
| 9 | Model Factory & Architecture Abstraction | Unified interface supporting DeBERTa-v3, BETO, RoBERTa-bne, and fast CPU mocks | M2 | ORIGINAL_REQUEST §R2 |
| 10 | CLI Training Engine (`experiments/train.py`) | Configurable CLI with device detection (CUDA/CPU), mixed precision, logging | M3 | ORIGINAL_REQUEST §R3.1 |
| 11 | GPU Training Automation (`scripts/run_gpu_training.sh`) | Self-contained bash launcher for remote NVIDIA server | M3 | ORIGINAL_REQUEST §R3.3 |
| 12 | Structured Checkpoints & Logging | Organized saving in `checkpoints/`, `logs/`, and `results/` | M3 | ORIGINAL_REQUEST §R3.2 |
| 13 | Cross-Method Comparison Tables | Consolidated metrics (Weighted F1, SoftCons, HardCons) across all methods | M4 | ORIGINAL_REQUEST §R4.1 |
| 14 | Inconsistency Transition & Confusion Matrices | 4x4 directional transition and error distribution matrices | M4 | ORIGINAL_REQUEST §R4.2 |
| 15 | Syntactic Breakdown Error Analysis | Disaggregated performance across nominal, verbal, prepositional, adverbial chunks | M4 | ORIGINAL_REQUEST §R4.3 |
| 16 | ACL SemEval Research Paper (`report/paper.tex`) | 8-section academic paper in LaTeX compiled to `paper.pdf` | M5 | ORIGINAL_REQUEST §R5.1 |
| 17 | Presentation Slides (`presentation/presentation.pdf`) | 6-section presentation slides compiled to PDF | M5 | ORIGINAL_REQUEST §R5.2 |
| 18 | Repository Documentation (`README.md`) | Comprehensive setup, CLI usage, and benchmark documentation | M5 | Acceptance Criteria |
| 19 | Automated Test Suite (Tiers 1-4) | Comprehensive unit and integration tests with 100% pass rate in <60s on CPU | M_TEST | Acceptance Criteria |
| 20 | Adversarial Coverage Hardening (Tier 5) | Adversarial stress testing for symmetry and edge-case robustness | M_TEST | Dual Track Phase 2 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M_TEST | E2E Testing Suite (Tiers 1-4) | Design test harness, Tiers 1-4 test cases, publish `TEST_READY.md` | None | DONE |
| M1 | Data Pipeline & Consistency Metrics (R1) | Ingestion, reversal pairing, Codabench format, Weighted F1, SoftCons, HardCons | None | DONE |
| M2 | Models & Reversal Strategies (R2) | Baseline, BSA, CRL, SD post-hoc projection with mathematical proof, model factory | M1 | DONE |
| M3 | CLI Training Pipeline & GPU Scripts (R3) | `experiments/train.py`, `scripts/run_gpu_training.sh`, checkpoints, logging | M1, M2 | PLANNED |
| M4 | Experimental Evaluation & Error Analysis (R4) | Evaluation script, comparison tables, confusion & transition matrices, syntactic breakdown | M2, M3 | PLANNED |
| M5 | Academic Deliverables & Documentation (R5) | `report/paper.tex` & `paper.pdf`, `presentation/presentation.pdf`, `README.md` | M4 | PLANNED |
| M_FINAL | Final Acceptance & Adversarial Hardening | Pass 100% E2E tests, Tier 5 hardening, Forensic Integrity Audit | M_TEST, M5 | PLANNED |

## Interface Contracts
### `src/data/` ↔ `src/models/` & `src/training/`
- `load_phrasis_split(track: str, split: str, data_dir: str = "data") -> pd.DataFrame`:
  Returns DataFrame with columns `['instance_id', 'pair_id', 'text1_lang', 'text2_lang', 'text1', 'text2', 'label', 'category']`.
- `build_reversal_dataset(df: pd.DataFrame, augment_bsa: bool = False) -> Dataset`:
  Pairs direct instances $(A, B)$ with reverse $(B, A)$ via `pair_id`. If `augment_bsa=True`, adds synthetic inverted pairs with transposed labels.
- `reverse_label(label: str) -> str`:
  Maps `EQUIVALENCE` $\to$ `EQUIVALENCE`, `FORWARD_ENTAILMENT` $\leftrightarrow$ `BACKWARD_ENTAILMENT`, `NEGATIVE_OTHER` $\to$ `NEGATIVE_OTHER`.

### `src/models/` ↔ `src/evaluation/` & `src/metrics/`
- Class Index Encoding:
  - `0`: `BACKWARD_ENTAILMENT`
  - `1`: `EQUIVALENCE`
  - `2`: `FORWARD_ENTAILMENT`
  - `3`: `NEGATIVE_OTHER`
- Involutory Permutation Matrix $P$:
  $$P = \begin{pmatrix} 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$
- `symmetrized_decoding(probs_dir: np.ndarray, probs_rev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`:
  Computes $p_{sym,dir} = 0.5 \cdot (p_{dir} + p_{rev} P)$ and $p_{sym,rev} = 0.5 \cdot (p_{rev} + p_{dir} P)$.
  Guarantees $\operatorname{argmax}(p_{sym,rev}) = \operatorname{Rev}(\operatorname{argmax}(p_{sym,dir}))$ for all inputs.
- `compute_official_metrics(y_true_dir, y_pred_dir, y_true_rev, y_pred_rev, is_reversible_mask) -> Dict[str, float]`:
  Returns `{"weighted_f1": float, "soft_cons": float, "hard_cons": float}`.

### `src/data/codabench.py`
- `export_codabench_submission(instance_ids: List[str], predictions: List[str], output_path: str)`:
  Writes strict two-column CSV `instance_id,label` with LF line endings, no blank rows, no index, no extra whitespace.
- `validate_codabench_submission(submission_path: str, expected_ids: Set[str]) -> bool`:
  Validates against official Codabench criteria.

## Code Layout
```
data/
  raw/                  # Raw PhrasIS corpus CSVs (Track 1 EN, Track 2 ES)
  processed/            # Cleaned splits and paired instances
src/
  data/
    __init__.py
    loader.py           # PhrasIS dataset reader and split manager
    reversal.py         # Reversal algebra, pairing logic, BSA generator
    codabench.py        # Codabench export and validation
  metrics/
    __init__.py
    metrics.py          # Weighted F1, SoftCons, HardCons official implementations
    consistency.py      # Consistency checking and transition matrix generators
  models/
    __init__.py
    cross_encoder.py    # Standard cross-encoder backbone
    siamese_crl.py      # Siamese / dual-stream with Consistency Regularization Loss
    losses.py           # CrossEntropy, Symmetric KL, Jensen-Shannon losses
    decoding.py         # Symmetrized Decoding post-hoc projection
    factory.py          # Unified model loader (DeBERTa-v3, BETO, RoBERTa, Fast Mock)
  training/
    __init__.py
    trainer.py          # Unified trainer with mixed precision, device detection, checkpointing
    utils.py            # Seed setting, logging, argument parsing
  evaluation/
    __init__.py
    evaluator.py        # Master evaluation harness across models & tracks
    analysis.py         # Syntactic chunk error analysis and report generation
    matrices.py         # Confusion and transition matrix plotters
experiments/
  train.py              # CLI training and evaluation entrypoint
  evaluate.py           # CLI standalone evaluation entrypoint
scripts/
  run_gpu_training.sh   # Master bash execution script for remote NVIDIA GPU
tests/
  test_reversal_logic.py # Reversal algebra, pairing, and label inversion tests
  test_metrics.py       # Weighted F1, SoftCons, HardCons tests
  test_symmetrized_decoding.py # Mathematical proof & empirical SoftCons=100% tests
  test_codabench.py     # Format validation tests
  test_pipeline.py      # Fast CPU end-to-end smoke test (<60s)
report/
  paper.tex             # ACL SemEval format final paper
  paper.pdf             # Compiled final paper
  acl.sty / semeval.sty # Style packages
  references.bib        # Comprehensive bibliography
presentation/
  presentation.tex      # LaTeX Beamer slides conforming to rubric
  presentation.pdf      # Compiled presentation slides
README.md               # User documentation, usage, and benchmark results
```
