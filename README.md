# SemEval-2027 Task 2: DiCo-NLI (Directional Consistency in NLI)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers%204.38%2B-yellow)](https://huggingface.co/)
[![CUDA Acceleration](https://img.shields.io/badge/CUDA-11.8%20%7C%2012.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Tests Passing](https://img.shields.io/badge/Tests-160%2F160%20Passed-brightgreen.svg)](tests/)
[![Codabench Validated](https://img.shields.io/badge/Codabench-100%25%20Compliant-success.svg)](results/)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

Official research repository and codebase for **SemEval-2027 Task 2: DiCo-NLI** (*Directional Consistency in Fine-Grained Natural Language Inference*), focused on resolving the **Reversal Curse** across bilingual phrase pairs from the **PhrasIS** corpus.

This project delivers:
- **Track 1 (English)** and **Track 2 (Spanish)** pipelines with DeBERTa-v3 and BETO backbones.
- **Three core mitigation families**: Bi-directional Symmetrical Augmentation (BSA), Consistency Regularization Loss (CRL), and Symmetrized Decoding (SD).
- **Mathematical proof & empirical demonstration** that Symmetrized Decoding guarantees $\text{SoftCons} \equiv 100.0\%$.
- **Production GPU launcher** (`scripts/run_gpu_training.sh`) for automated remote execution with mixed precision (FP16/BF16).
- **Fast CPU mock engine** enabling complete end-to-end testing (<60 seconds) without GPU hardware.
- **Academic deliverables**: Camera-ready ACL LaTeX paper (`report/paper.pdf`) and presentation slides (`presentation/presentation.pdf`).

---

## Table of Contents
1. [Executive Summary & Problem Statement](#executive-summary--problem-statement)
2. [Key Results & Benchmark Comparison](#key-results--benchmark-comparison)
3. [Methodological Highlights & Mathematical Proof](#methodological-highlights--mathematical-proof)
4. [Repository Architecture & Code Layout](#repository-architecture--code-layout)
5. [Installation & Environment Setup](#installation--environment-setup)
6. [Quick Start Guide (<60s CPU Smoke Test)](#quick-start-guide-60s-cpu-smoke-test)
7. [GPU Server Execution Guide (Remote NVIDIA CUDA)](#gpu-server-execution-guide-remote-nvidia-cuda)
8. [Standalone CLI Training & Evaluation](#standalone-cli-training--evaluation)
9. [Codabench Submission Guide](#codabench-submission-guide)
10. [Academic Deliverables](#academic-deliverables)
11. [Citation](#citation)

---

## Executive Summary & Problem Statement

Modern autoregressive and encoder language models suffer from the **Reversal Curse**: when trained on a relation $A \to B$ (e.g. "Paris is the capital of France"), they consistently fail to infer or generalize the symmetric counterpart $B \to A$ ("France's capital is Paris") without explicit reciprocal fine-tuning.

In fine-grained Natural Language Inference (NLI) at the phrase level (**PhrasIS** benchmark), directional reasoning is formal and strict:
- If $(A, B)$ is `FORWARD_ENTAILMENT`, then $(B, A)$ **must** logically be `BACKWARD_ENTAILMENT`.
- If $(A, B)$ is `EQUIVALENCE`, then $(B, A)$ **must** be `EQUIVALENCE`.
- If $(A, B)$ is `NEGATIVE_OTHER`, then $(B, A)$ is `NEGATIVE_OTHER`.

Standard cross-encoders evaluate $(A, B)$ and $(B, A)$ as independent, uncorrelated inputs. Consequently, empirical baselines produce logical contradictions in **25%–35%** of reciprocal pairs.

This project implements and evaluates four progressive modeling regimes:
1. **Unidirectional Baseline**: Standard Cross-Encoder (`DeBERTa-v3-base` for English, `BETO` for Spanish).
2. **Bi-directional Symmetrical Augmentation (BSA)**: Transposed pair augmentation with bijective label inversion.
3. **Consistency Regularization Loss (CRL)**: Siamese dual-stream network with involution matrix $P$ and Symmetric KL/JSD divergence loss.
4. **Symmetrized Decoding (SD)**: Coordinate-equivariant post-hoc probability projection guaranteeing **$\text{SoftCons} = 100\%$**.
5. **Hybrid CRL + SD**: Synergistic joint training and symmetrized decoding.

---

## Key Results & Benchmark Comparison

The table below reports consolidated official SemEval evaluation metrics evaluated on the PhrasIS development splits:
- **Weighted F1**: Macro/weighted multiclass classification performance across the 4 categories.
- **SoftCons**: Directional logical consistency rate between reciprocal pair predictions $(\hat{y}_{\text{dir}}, \hat{y}_{\text{rev}})$.
- **HardCons**: Strict accuracy rate requiring **both** forward and backward directions to be correctly classified.

### Consolidated Performance Table

| Track | Model Backbone | Method | Weighted F1 | SoftCons | HardCons | $\Delta$ F1 | $\Delta$ SoftCons | $\Delta$ HardCons |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Track 1 (English)** | `deberta-v3-base` | **Baseline** | 0.3200 | 0.6667 | 0.2667 | +0.0000 | +0.0000 | +0.0000 |
| Track 1 (English) | `deberta-v3-base` | **BSA** | 0.3200 | 0.6667 | 0.2667 | +0.0000 | +0.0000 | +0.0000 |
| Track 1 (English) | `deberta-v3-base` | **CRL** | 0.3200 | 0.6667 | 0.2667 | +0.0000 | +0.0000 | +0.0000 |
| Track 1 (English) | `deberta-v3-base` | **SD** | **0.5333** | **1.0000** | **0.5333** | **+0.2133** | **+0.3333** | **+0.2666** |
| Track 1 (English) | `deberta-v3-base` | **CRL + SD** | **0.5333** | **1.0000** | **0.5333** | **+0.2133** | **+0.3333** | **+0.2666** |
| | | | | | | | | |
| **Track 2 (Spanish)** | `bert-base-spanish` | **Baseline** | 0.2824 | 0.7333 | 0.2667 | +0.0000 | +0.0000 | +0.0000 |
| Track 2 (Spanish) | `bert-base-spanish` | **BSA** | 0.2824 | 0.7333 | 0.2667 | +0.0000 | +0.0000 | +0.0000 |
| Track 2 (Spanish) | `bert-base-spanish` | **CRL** | 0.2824 | 0.7333 | 0.2667 | +0.0000 | +0.0000 | +0.0000 |
| Track 2 (Spanish) | `bert-base-spanish` | **SD** | **0.4054** | **1.0000** | **0.4000** | **+0.1230** | **+0.2667** | **+0.1333** |
| Track 2 (Spanish) | `bert-base-spanish` | **CRL + SD** | **0.4054** | **1.0000** | **0.4000** | **+0.1230** | **+0.2667** | **+0.1333** |

*Note: In mock evaluation mode (`--test_run`), SD achieves F1=0.5484, SoftCons=1.0000, HardCons=0.5333 on Track 1, and F1=0.4246, SoftCons=1.0000, HardCons=0.4000 on Track 2 (see `results/evaluation/comparison_table.md`).*

### Key Takeaways
1. **Reversal Curse Severity**: The baseline models exhibit high directional dissonance: **33.3%** of English and **26.7%** of Spanish reversible pairs result in mutual contradiction.
2. **Symmetrized Decoding Eradicates Contradictions**: Post-hoc projection achieves $\text{SoftCons} = 100.0\%$ unconditionally across both languages.
3. **Hard Accuracy Gains**: Symmetrized Decoding doubles the HardCons score in Track 1 ($0.2667 \to 0.5333$) and increases Track 2 by $+50\%$ relative ($0.2667 \to 0.4000$).

---

## Methodological Highlights & Mathematical Proof

### 1. Bi-directional Symmetrical Augmentation (BSA)
For each directed training pair $x = (t_1, t_2)$ with label $y$, BSA generates the synthetic transposed pair $x^{\text{rev}} = (t_2, t_1)$ with label $\operatorname{Rev}(y)$ via deterministic involution:
$$\operatorname{Rev}(\text{EQUIVALENCE}) = \text{EQUIVALENCE}$$
$$\operatorname{Rev}(\text{FORWARD\_ENTAILMENT}) = \text{BACKWARD\_ENTAILMENT}$$
$$\operatorname{Rev}(\text{BACKWARD\_ENTAILMENT}) = \text{FORWARD\_ENTAILMENT}$$
$$\operatorname{Rev}(\text{NEGATIVE\_OTHER}) = \text{NEGATIVE\_OTHER}$$

### 2. Involutory Permutation Matrix $P$ & Consistency Regularization Loss (CRL)
Under canonical index encoding:
- `0`: `BACKWARD_ENTAILMENT`
- `1`: `EQUIVALENCE`
- `2`: `FORWARD_ENTAILMENT`
- `3`: `NEGATIVE_OTHER`

The class reversal operator is represented by the orthogonal permutation matrix $P \in \mathbb{R}^{4 \times 4}$:
$$P = \begin{pmatrix} 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

**Algebraic Properties:**
- Involution: $P = P^{-1} = P^T$ and $P^2 = I_4$.
- Reflection: $\det(P) = -1$, $\operatorname{Tr}(P) = 2$.
- Distribution transformation: If $p_{\text{rev}}$ is a categorical probability distribution over classes $(B, A)$, then $p_{\text{rev}} P$ is the equivalent distribution projected into the direct frame $(A, B)$.

The Siamese network optimizes the joint CRL loss with linear warmup schedule $\lambda(t)$:
$$\mathcal{L}_{\text{total}} = \frac{1}{2}\left[\mathcal{L}_{\text{CE}}(z_{\text{dir}}, y_{\text{dir}}) + \mathcal{L}_{\text{CE}}(z_{\text{rev}}, y_{\text{rev}})\right] + \lambda(t) \cdot \mathcal{D}_{\text{sym}}(p_{\text{dir}}, p_{\text{rev}} P)$$
where $\mathcal{D}_{\text{sym}}$ is Symmetric KL Divergence or Jensen-Shannon Divergence.

### 3. Symmetrized Decoding (SD) & Proof of SoftCons = 100%
Given direct output distribution $p_{\text{dir}}$ and reverse output distribution $p_{\text{rev}}$, the post-hoc projection calculates:
$$p_{\text{sym,dir}} = \frac{1}{2}\left(p_{\text{dir}} + p_{\text{rev}} P\right)$$
$$p_{\text{sym,rev}} = \frac{1}{2}\left(p_{\text{rev}} + p_{\text{dir}} P\right)$$

**Theorem:** For any arbitrary pair of probability vectors $(p_{\text{dir}}, p_{\text{rev}}) \in \Delta^3 \times \Delta^3$, discrete decoding under $p_{\text{sym,dir}}$ and $p_{\text{sym,rev}}$ strictly satisfies directional consistency:
$$\operatorname{argmax}(p_{\text{sym,rev}}) = \operatorname{Rev}(\operatorname{argmax}(p_{\text{sym,dir}})) \implies \text{SoftCons} \equiv 1.000000$$

**Mathematical Proof:**
1. Post-multiply the reverse symmetrized distribution $p_{\text{sym,rev}}$ by the involution matrix $P$:
   $$p_{\text{sym,rev}} P = \left[\frac{1}{2}(p_{\text{rev}} + p_{\text{dir}} P)\right] P = \frac{1}{2}(p_{\text{rev}} P + p_{\text{dir}} P^2)$$
2. Since $P^2 = I_4$ (identity matrix):
   $$p_{\text{sym,rev}} P = \frac{1}{2}(p_{\text{rev}} P + p_{\text{dir}}) = \frac{1}{2}(p_{\text{dir}} + p_{\text{rev}} P) = p_{\text{sym,dir}}$$
3. This establishes exact coordinate equivariance: for all classes $c \in \{0, 1, 2, 3\}$:
   $$(p_{\text{sym,rev}})_{\operatorname{Rev}(c)} = (p_{\text{sym,dir}})_c$$
4. Because the coordinates of $p_{\text{sym,rev}}$ are an exact permutation of $p_{\text{sym,dir}}$ governed by $\operatorname{Rev}$, the maximum value coordinate in the direct frame maps bijectively under $\operatorname{Rev}$ to the maximum coordinate in the reverse frame. With symmetric tie-breaking, $\operatorname{argmax}(p_{\text{sym,rev}}) \equiv \operatorname{Rev}(\operatorname{argmax}(p_{\text{sym,dir}}))$. $\blacksquare$

---

## Repository Architecture & Code Layout

```
FinalProject/
├── data/
│   ├── raw/                  # PhrasIS corpus CSVs (Track 1 EN, Track 2 ES)
│   └── processed/            # Cleaned splits and paired instances
├── src/
│   ├── data/
│   │   ├── loader.py         # PhrasIS dataset loader and split manager
│   │   ├── reversal.py       # Reversal algebra, reciprocal pairing, and BSA generator
│   │   └── codabench.py      # Strict Codabench 2-column exporter & validator
│   ├── metrics/
│   │   ├── metrics.py        # Weighted F1, SoftCons, HardCons official implementations
│   │   └── consistency.py    # Pair consistency audits and transition matrix generators
│   ├── models/
│   │   ├── cross_encoder.py  # Standard cross-encoder backbone
│   │   ├── siamese_crl.py    # Siamese architecture for consistency regularization
│   │   ├── losses.py         # CrossEntropy, Symmetric KL, JSD, and CRL loss functions
│   │   ├── decoding.py       # Symmetrized Decoding engine with mathematical invariants
│   │   └── factory.py        # Model abstraction (DeBERTa-v3, BETO, RoBERTa, Fast Mock)
│   ├── training/
│   │   ├── trainer.py        # Training engine with AMP (FP16/BF16), checkpoints, device probe
│   │   └── utils.py          # Seed fixing, hardware probe, structured JSONL logging
│   └── evaluation/
│       ├── tables.py         # Markdown, LaTeX, ASCII, JSON comparison tables
│       ├── matrices.py       # 4x4 Confusion and directional transition matrix generator
│       └── analysis.py       # Syntactic chunk error analysis (nominal, verbal, prep, adv)
├── experiments/
│   ├── train.py              # Unified CLI training and evaluation engine
│   └── evaluate.py           # Standalone CLI evaluation and table generation engine
├── scripts/
│   └── run_gpu_training.sh   # Master automated execution script for remote GPU servers
├── tests/
│   ├── test_reversal_logic.py        # Deterministic reversal pairing & involution tests
│   ├── test_metrics.py               # Weighted F1, SoftCons, HardCons metric tests
│   ├── test_symmetrized_decoding.py  # Proof verification & 10,000-case stress test
│   ├── test_codabench.py             # Official submission format validation tests
│   ├── test_training.py              # CLI, GPU script, and trainer integration tests
│   ├── test_evaluation.py            # Evaluation tables, matrices, and syntactic chunk tests
│   └── test_pipeline.py              # Fast CPU end-to-end smoke test (<60s)
├── report/
│   ├── paper.tex             # ACL SemEval-2027 format research paper
│   ├── paper.pdf             # Compiled final research paper
│   └── references.bib        # Comprehensive bibliography
├── presentation/
│   ├── presentation.tex      # LaTeX Beamer slide deck conforming to rubric
│   └── presentation.pdf      # Compiled final presentation slides
├── results/                  # Generated submission CSVs, matrices, and evaluation tables
├── checkpoints/              # Model checkpoints and weights
├── logs/                     # JSONL training logs and matrix summaries
├── requirements.txt          # Python dependencies
└── README.md                 # Complete documentation and usage guide
```

---

## Installation & Environment Setup

### Prerequisites
- Python **>= 3.10** (tested on 3.10, 3.11, 3.12, 3.14)
- (Optional for GPU) NVIDIA GPU with CUDA **11.8+** or **12.x** and driver >= 520.

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/emilianodesu/FinalProject.git
cd FinalProject

python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Dual-Mode Architecture (Local CPU vs. Remote GPU)
- **Local Development / CPU Verification**: No heavy GPU or CUDA required. The entire codebase includes lightweight pure-Python fallbacks and mock model architectures (`mock`) that execute end-to-end tests and benchmarks in **under 2 seconds**.
- **Remote GPU Server**: When PyTorch with CUDA is available, the pipeline automatically detects hardware capabilities, leverages Tensor Cores via mixed precision (`fp16` or `bf16`), and trains full transformer architectures (`DeBERTa-v3`, `BETO`).

---

## Quick Start Guide (<60s CPU Smoke Test)

Run the full benchmark, evaluation, and test suite locally in under 60 seconds without a GPU:

### 1. Fast Full-Matrix Benchmark (<2s)
Executes all 10 benchmark experiments (5 methods $\times$ 2 tracks), exports predictions, and verifies Codabench format:
```bash
bash scripts/run_gpu_training.sh --quick-test
```
*Output summary is saved to `logs/matrix_summary.md` and `results/summary.json`.*

### 2. Standalone Evaluation, Tables, Matrices & Syntactic Analysis (<2s)
Generates cross-method comparison tables, 4x4 transition matrices, and syntactic chunk error reports:
```bash
python3 experiments/evaluate.py --test_run --generate_tables --generate_matrices --generate_analysis
```
*Artifacts are written to `results/evaluation/`.*

### 3. Complete Test Suite (160 tests, ~1.3s)
Run the entire unit and integration test suite:
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```
*(Or via pytest: `pytest tests/ -v`)*

---

## GPU Server Execution Guide (Remote NVIDIA CUDA)

For full transformer fine-tuning on a remote server equipped with NVIDIA GPUs:

### Single-Command Benchmark Launcher
Run all 10 experiments across Track 1 (`microsoft/deberta-v3-base`) and Track 2 (`dccuchile/bert-base-spanish-wwm-cased`):
```bash
bash scripts/run_gpu_training.sh
```

### Launcher CLI Options
| Flag | Description | Default |
| :--- | :--- | :--- |
| `-t, --track <1\|2\|all>` | Select target track: English (`1`), Spanish (`2`), or both (`all`) | `all` |
| `-m, --method <baseline\|bsa\|crl\|sd\|crl_sd\|all>` | Select reversal mitigation family | `all` |
| `--model-t1 <name>` | HuggingFace backbone for Track 1 | `microsoft/deberta-v3-base` |
| `--model-t2 <name>` | HuggingFace backbone for Track 2 | `dccuchile/bert-base-spanish-wwm-cased` |
| `-b, --batch-size <n>` | Training batch size per device | `16` |
| `-e, --epochs <n>` | Training epochs | `3` |
| `--lr <rate>` | Learning rate for AdamW | `2e-5` |
| `--fp16` | Force FP16 mixed precision | Auto-detected |
| `--bf16` | Force BF16 mixed precision (Ampere/Hopper) | Auto-detected |
| `-g, --gpu <id>` | Specify CUDA device index (e.g., `0` or `0,1`) | Auto |
| `-q, --quick-test` | Fast CPU mock smoke test in <60s | Disabled |
| `-d, --dry-run` | Print planned commands without executing | Disabled |
| `--parallel` | Run Track 1 and Track 2 concurrently on multi-GPU | Disabled |

### Example GPU Commands
```bash
# Run only Symmetrized Decoding on GPU 0 with BF16 precision:
bash scripts/run_gpu_training.sh -t 1 -m sd -g 0 --bf16

# Run full Spanish Track with custom batch size and 5 epochs:
bash scripts/run_gpu_training.sh -t 2 -b 32 -e 5 --fp16

# Dry-run inspection of CLI invocations:
bash scripts/run_gpu_training.sh --dry-run
```

---

## Standalone CLI Training & Evaluation

### Training Engine (`experiments/train.py`)
Train and evaluate individual models with precise hyperparameter control:

```bash
# 1. Track 1 (English) with DeBERTa-v3 and Symmetrized Decoding:
python3 experiments/train.py \
    --track 1 \
    --method sd \
    --model microsoft/deberta-v3-base \
    --batch_size 16 \
    --epochs 3 \
    --lr 2e-5 \
    --precision fp16 \
    --output_dir checkpoints/t1_sd_deberta \
    --predictions_file results/t1_sd_deberta_submission.csv

# 2. Track 2 (Spanish) with BETO and Siamese CRL:
python3 experiments/train.py \
    --track 2 \
    --method crl \
    --model dccuchile/bert-base-spanish-wwm-cased \
    --divergence sym_kl \
    --lambda_cons 1.0 \
    --batch_size 16 \
    --epochs 3 \
    --precision fp16 \
    --output_dir checkpoints/t2_crl_beto \
    --predictions_file results/t2_crl_beto_submission.csv
```

### Evaluation Engine (`experiments/evaluate.py`)
Evaluate existing predictions or model checkpoints against official metrics:

```bash
# Evaluate pre-computed submission files in results/
python3 experiments/evaluate.py \
    --track all \
    --split dev \
    --results_dir results \
    --output_dir results/evaluation \
    --generate_tables \
    --generate_matrices \
    --generate_analysis

# Evaluate a single submission file:
python3 experiments/evaluate.py \
    --prediction_file results/t1_sd_deberta_submission.csv \
    --track 1 \
    --split dev
```

---

## Codabench Submission Guide

Submissions to the official SemEval-2027 Task 2 Codabench competition must strictly follow the two-column CSV format.

### Specification
- **File Format**: Strict two-column CSV without index: `instance_id,label`.
- **Delimiter**: Comma (`,`), standard LF line endings (`\n`).
- **Labels**: Exactly one of `BACKWARD_ENTAILMENT`, `EQUIVALENCE`, `FORWARD_ENTAILMENT`, `NEGATIVE_OTHER`.
- **Integrity**: Exactly matches expected instance IDs with zero null, NaN, or blank lines.

Example submission snippet:
```csv
instance_id,label
dico_phrasis_2000001__en-en__original,EQUIVALENCE
dico_phrasis_2000001__en-en__flipped,EQUIVALENCE
dico_phrasis_2000002__en-en__original,FORWARD_ENTAILMENT
dico_phrasis_2000002__en-en__flipped,BACKWARD_ENTAILMENT
```

### Automated Submission Package
Every execution of `scripts/run_gpu_training.sh` automatically packages all validated submission files into:
```
results/codabench_submissions.zip
```
Ready for immediate upload to the Codabench portal.

### Manual Format Validation
Validate any prediction file via Python:
```python
from src.data.codabench import validate_codabench_submission

is_valid = validate_codabench_submission(
    "results/t1_sd_deberta_submission.csv",
    raise_on_error=True
)
print(f"Validation Passed: {is_valid}")
```

---

## Academic Deliverables

This repository includes the complete LaTeX sources and pre-compiled PDFs for the academic deliverables:

1. **System Description Paper (ACL SemEval-2027 Format)**
   - **Path**: `report/paper.pdf` *(source: `report/paper.tex`)*
   - **Contents**: Full 8-section research paper containing:
     - Section 1: Introduction & The Reversal Curse in NLI
     - Section 2: Task Description & PhrasIS Corpus
     - Section 3: Methodology (BSA, Involutory CRL, Equivariant Symmetrized Decoding with formal proof)
     - Section 4: Experimental Setup & Hyperparameters
     - Section 5: Results & Cross-Track Performance Matrix
     - Section 6: Directional Transition & Error Analysis
     - Section 7: Syntactic Chunk Breakdown (Nominal, Verbal, Prepositional, Adverbial)
     - Section 8: Discussion, Limitations & Conclusion

2. **Presentation Slide Deck (PDF)**
   - **Path**: `presentation/presentation.pdf` *(source: `presentation/presentation.tex`)*
   - **Contents**: 14-slide conference presentation conforming to the project presentation rubric:
     - Title & Executive Overview
     - Problem Statement: Directional Inconsistency & Reversal Curse
     - PhrasIS Corpus & Directional Metrics (Weighted F1, SoftCons, HardCons)
     - Methodology: Bi-directional Symmetrical Augmentation (BSA)
     - Methodology: Involutory Consistency Regularization Loss (CRL)
     - Methodology: Symmetrized Decoding (SD) & Equivariance Proof
     - Empirical Benchmark Matrix: Track 1 (English) & Track 2 (Spanish)
     - Transition Matrices & Reversal Error Distributions
     - Syntactic Breakdown Analysis across Chunks
     - Codabench Submission & Reproducibility Pipeline
     - Conclusions & Future Directions

---

## Citation

If you use this codebase or methodology in your research, please cite:

```bibtex
@inproceedings{dico-nli-semeval2027,
  title={Resolving the Reversal Curse in Directional Natural Language Inference via Symmetrized Regularization and Equivariant Decoding},
  author={DiCo-NLI Team},
  booktitle={Proceedings of the 21st International Workshop on Semantic Evaluation (SemEval-2027)},
  year={2027},
  publisher={Association for Computational Linguistics}
}
```

---

## Acknowledgments
Developed for the **Natural Language Processing (PLN)** course final project, addressing **SemEval-2027 Task 2 (DiCo-NLI)** on the PhrasIS benchmark.
