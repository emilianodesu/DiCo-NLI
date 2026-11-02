# Blueprint Report: Directional Consistency Metrics & Official Parity for SemEval-2027 Task 2 (DiCo-NLI)

**Agent:** `explorer_m1_3` (Explorer for Consistency Metrics & Official Parity)  
**Target Files:** `src/metrics/metrics.py` and `src/metrics/consistency.py`  
**Dependencies:** `src/data/` schemas and reversal algebra  
**Consumers:** `src/training/trainer.py`, `src/evaluation/evaluator.py`, `src/evaluation/matrices.py`, `experiments/train.py`, `experiments/evaluate.py`, `tests/test_metrics.py`, `report/paper.tex`

---

## 1. Executive Summary & Problem Scope

SemEval-2027 Task 2 (*Directional Consistency in Fine-Grained Natural Language Inference — DiCo-NLI*) formalizes the evaluation of Natural Language Inference models over ordered phrase pairs under directional inversion. Unlike standard sentence-level NLI tasks that evaluate isolated instances, DiCo-NLI enforces joint directional evaluation to measure the **Reversal Curse**: the empirical failure of standard language models to generalize symmetric or directional knowledge from $(A, B)$ to $(B, A)$.

Official competition scoring on CodaBench is governed by a three-metric profile:
1. **Weighted F1 ($F_w$):** Support-weighted multi-class F1 score across all 4 classes over all $N$ evaluation instances.
2. **Soft Consistency Rate ($\SoftCons$):** Logical self-consistency rate over unique reversible source pairs $\mathcal{P}$, testing whether predicted relations are mutually compatible under directional reversal, independent of gold correctness.
3. **Hard Consistency Rate ($\HardCons$):** Strict joint correctness rate over unique reversible source pairs $\mathcal{P}$, requiring both directions $(A, B)$ and $(B, A)$ to match the ground truth.

This report establishes:
- The exact mathematical formulations, invariants, and edge-case behavior of the official metrics.
- Mathematical proofs for the fundamental inequality:
  $$0.0 \le \HardCons \le \SoftCons \le 1.0$$
  and the upper bound:
  $$\HardCons \le \min\left(\text{Acc}_{\text{dir}}, \text{Acc}_{\text{rev}}\right) \le \text{Acc}_{\text{mean}}$$
- A dual-mode API for `src/metrics/metrics.py`:
  - **Drop-in Official Parity Mode:** Directly consumes record dictionaries matching the official competition evaluator (`evaluation_functions/metrics.py`) to guarantee zero scoring discrepancy on CodaBench submissions.
  - **High-Throughput Vectorized Mode:** Operates on batched PyTorch tensors and NumPy arrays for fast evaluation during training loops without Python dictionary overhead.
- A comprehensive $4 \times 4$ **Directional Transition Matrix** engine and a 16-cell **Inconsistency Taxonomy** for `src/metrics/consistency.py`, enabling rigorous error analysis for the ACL SemEval paper (`report/paper.tex`) and presentation (`presentation/presentation.pdf`).

---

## 2. Label Space & Algebraic Foundations

### 2.1 Label Inventory
The task operates over an official 4-class categorical label space $\mathcal{L}$:
$$\mathcal{L} = \{\text{BACKWARD\_ENTAILMENT}, \text{EQUIVALENCE}, \text{FORWARD\_ENTAILMENT}, \text{NEGATIVE\_OTHER}\}$$

In numerical tensors and matrices, classes are canonically indexed as:
- **Index 0:** `BACKWARD_ENTAILMENT` (Hypothesis entails Premise: $B \implies A$)
- **Index 1:** `EQUIVALENCE` (Mutual entailment: $A \iff B$)
- **Index 2:** `FORWARD_ENTAILMENT` (Premise entails Hypothesis: $A \implies B$)
- **Index 3:** `NEGATIVE_OTHER` (Contradiction, independence, or unaligned phrase noise)

### 2.2 Reversible Subset $\mathcal{R}$ and Inversion Operator $\Rev(\cdot)$
Consistency metrics evaluate only pairs exhibiting structured directional semantics. The reversible label subset is:
$$\mathcal{R} = \{\text{EQUIVALENCE}, \text{FORWARD\_ENTAILMENT}, \text{BACKWARD\_ENTAILMENT}\} \subset \mathcal{L}$$

The deterministic reversal operator $\Rev: \mathcal{R} \to \mathcal{R}$ is defined by:
$$\Rev(y) = \begin{cases}
\text{EQUIVALENCE}, & \text{if } y = \text{EQUIVALENCE} \\
\text{BACKWARD\_ENTAILMENT}, & \text{if } y = \text{FORWARD\_ENTAILMENT} \\
\text{FORWARD\_ENTAILMENT}, & \text{if } y = \text{BACKWARD\_ENTAILMENT}
\end{cases}$$

Notice that $\Rev$ is an **involution**:
$$\Rev(\Rev(y)) = y \quad \forall y \in \mathcal{R}$$

In tensor index space $\{0, 1, 2, 3\}$, the reversal operator corresponds to the permutation:
$$\pi = (0 \leftrightarrow 2)(1)(3)$$
which is represented by the symmetric, involutory permutation matrix $P \in \mathbb{R}^{4 \times 4}$:
$$P = \begin{pmatrix}
0 & 0 & 1 & 0 \\
0 & 1 & 0 & 0 \\
1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1
\end{pmatrix}$$
with $P = P^T = P^{-1}$ and $P^2 = I_4$.

### 2.3 The Critical Rule Regarding `NEGATIVE_OTHER`
In official competition evaluation (`evaluation_functions/labels.py` and `metrics.py`):
1. `NEGATIVE_OTHER` is **not** an element of $\mathcal{R}$.
2. Calling $\Rev(\text{NEGATIVE\_OTHER})$ in the official codebase raises a `ValidationError`.
3. If an evaluation model predicts `NEGATIVE_OTHER` for either instance in a reversible pair $i \in \mathcal{P}$, the pair receives:
   $$s_i = 0 \quad (\text{not soft-consistent})$$
   $$h_i = 0 \quad (\text{not hard-correct})$$
4. Consequently, reversible pairs in $\mathcal{P}$ cannot be considered logically consistent by predicting noise `NEGATIVE_OTHER`, even if both directions are labeled `NEGATIVE_OTHER`.

---

## 3. Official Evaluation Metrics (`src/metrics/metrics.py`)

### 3.1 Weighted F1 ($F_w$)

#### Mathematical Definition
Let $N$ be the total number of evaluation instances. Let $\mathcal{L} = \{0, 1, 2, 3\}$ be the label set.  
Let $C \in \mathbb{Z}_{\ge 0}^{4 \times 4}$ be the multiclass confusion matrix, where $C_{j, k}$ denotes the number of instances with ground-truth label $j$ predicted as label $k$.

For each class $\ell \in \mathcal{L}$:
- **True Positives ($TP_\ell$):**
  $$TP_\ell = C_{\ell, \ell}$$
- **False Positives ($FP_\ell$):**
  $$FP_\ell = \sum_{k \neq \ell} C_{k, \ell}$$
- **False Negatives ($FN_\ell$):**
  $$FN_\ell = \sum_{k \neq \ell} C_{\ell, k}$$
- **Support ($n_\ell$):**
  $$n_\ell = \sum_{k \in \mathcal{L}} C_{\ell, k}$$
  satisfying $\sum_{\ell \in \mathcal{L}} n_\ell = N$.

Per-class Precision ($P_\ell$), Recall ($R_\ell$), and F1 score ($F_\ell$) with zero-division handling:
$$P_\ell = \begin{cases} \frac{TP_\ell}{TP_\ell + FP_\ell}, & \text{if } TP_\ell + FP_\ell > 0 \\ 0.0, & \text{otherwise} \end{cases}$$
$$R_\ell = \begin{cases} \frac{TP_\ell}{TP_\ell + FN_\ell}, & \text{if } TP_\ell + FN_\ell > 0 \\ 0.0, & \text{otherwise} \end{cases}$$
$$F_\ell = \begin{cases} \frac{2 P_\ell R_\ell}{P_\ell + R_\ell}, & \text{if } P_\ell + R_\ell > 0 \\ 0.0, & \text{otherwise} \end{cases}$$

The overall **Weighted F1 ($F_w$)** is the support-weighted average of per-class F1 scores:
$$F_w = \sum_{\ell \in \mathcal{L}} \left(\frac{n_\ell}{N}\right) F_\ell$$

#### Diagnostic Classification Metrics
- **Macro F1 ($F_{\text{macro}}$):** Unweighted mean over all classes with non-zero support:
  $$F_{\text{macro}} = \frac{1}{|\{\ell \in \mathcal{L} : n_\ell > 0\}|} \sum_{\ell : n_\ell > 0} F_\ell$$
- **Overall Accuracy ($\text{Acc}$):**
  $$\text{Acc} = \frac{1}{N} \sum_{\ell \in \mathcal{L}} C_{\ell, \ell} = \frac{\operatorname{Tr}(C)}{N}$$

---

### 3.2 Soft Consistency Rate ($\SoftCons$)

#### Target Evaluation Set $\mathcal{P}$
Let $\mathcal{P}$ denote the set of unique reversible source pairs extracted from the gold reference.  
In the PhrasIS dataset:
- On Track 1 & Track 2 Dev splits: $N = 660$ instances, with $554$ reversible instances forming $|\mathcal{P}| = 277$ unique pairs ($106$ instances are non-reversible `NEGATIVE_OTHER`).
- On Track 1 & Track 2 Train splits: $N = 3,042$ instances, with $2,564$ reversible instances forming $|\mathcal{P}| = 1,282$ unique pairs ($478$ instances are `NEGATIVE_OTHER`).

Each pair $i \in \mathcal{P}$ consists of:
- `first_id`: Instance ID of $(A, B)$ (e.g. `...__original`)
- `second_id`: Instance ID of $(B, A)$ (e.g. `...__flipped`)
- `first_label`: Gold label $y_i^{(1)} \in \mathcal{R}$
- `second_label`: Gold label $y_i^{(2)} = \Rev(y_i^{(1)}) \in \mathcal{R}$

To prevent duplicate scoring, canonical deduplication uses unordered sets:
$$k = \operatorname{frozenset}(\text{first\_id}, \text{second\_id})$$
Each unique $k$ corresponds to exactly one evaluation entry in $\mathcal{P}$.

#### Mathematical Definition
For pair $i \in \mathcal{P}$, let $\hat{y}_i^{(1)} = \hat{y}(\text{first\_id})$ and $\hat{y}_i^{(2)} = \hat{y}(\text{second\_id})$ be the model predictions.  
The soft consistency indicator $s_i \in \{0, 1\}$ is defined as:
$$s_i = \begin{cases}
1, & \text{if } \hat{y}_i^{(1)} \in \mathcal{R} \;\land\; \hat{y}_i^{(2)} = \Rev(\hat{y}_i^{(1)}) \\
0, & \text{otherwise}
\end{cases}$$

Because $\Rev$ is an involution on $\mathcal{R}$, this condition is symmetric:
$$\hat{y}_i^{(2)} = \Rev(\hat{y}_i^{(1)}) \iff \hat{y}_i^{(1)} = \Rev(\hat{y}_i^{(2)})$$
The overall **SoftCons** metric is the empirical mean:
$$\SoftCons = \frac{1}{|\mathcal{P}|} \sum_{i \in \mathcal{P}} s_i$$

---

### 3.3 Hard Consistency Rate ($\HardCons$)

#### Mathematical Definition
The hard consistency metric evaluates strict joint correctness across both orientations of the reversible pair:
$$h_i = \begin{cases}
1, & \text{if } \hat{y}_i^{(1)} = y_i^{(1)} \;\land\; \hat{y}_i^{(2)} = y_i^{(2)} \\
0, & \text{otherwise}
\end{cases}$$

The overall **HardCons** metric is the empirical mean:
$$\HardCons = \frac{1}{|\mathcal{P}|} \sum_{i \in \mathcal{P}} h_i$$

---

## 4. Mathematical Theorems & Invariant Guarantees

### Theorem 1 (Fundamental Consistency Inequality)
For any set of model predictions on a valid DiCo-NLI evaluation set, the following strict inequality holds:
$$0.0 \le \HardCons \le \SoftCons \le 1.0$$

#### Formal Proof:
1. Let $i \in \mathcal{P}$ be any reversible pair.
2. By construction of the gold reference dataset, the gold labels satisfy $y_i^{(1)} \in \mathcal{R}$ and $y_i^{(2)} = \Rev(y_i^{(1)})$.
3. Suppose $h_i = 1$. By definition of $h_i$, this implies:
   $$\hat{y}_i^{(1)} = y_i^{(1)} \quad \text{and} \quad \hat{y}_i^{(2)} = y_i^{(2)}$$
4. Substituting the gold equality $y_i^{(2)} = \Rev(y_i^{(1)})$ into the prediction variables:
   $$\hat{y}_i^{(1)} = y_i^{(1)} \in \mathcal{R}$$
   $$\hat{y}_i^{(2)} = y_i^{(2)} = \Rev(y_i^{(1)}) = \Rev(\hat{y}_i^{(1)})$$
5. This matches the exact condition for $s_i = 1$. Therefore:
   $$h_i = 1 \implies s_i = 1$$
6. Since $h_i, s_i \in \{0, 1\}$, it follows that:
   $$h_i \le s_i \quad \forall i \in \mathcal{P}$$
7. Summing over all $|\mathcal{P}|$ pairs and dividing by $|\mathcal{P}| > 0$:
   $$\HardCons = \frac{1}{|\mathcal{P}|} \sum_{i \in \mathcal{P}} h_i \le \frac{1}{|\mathcal{P}|} \sum_{i \in \mathcal{P}} s_i = \SoftCons$$
8. Furthermore, because $0 \le h_i$ and $s_i \le 1$ for all $i$:
   $$0.0 \le \HardCons \le \SoftCons \le 1.0$$
$\blacksquare$

---

### Theorem 2 (Directional Accuracy Upper Bound on HardCons)
HardCons is strictly bounded from above by the individual directional accuracies over the reversible pair set:
$$\HardCons \le \min\left(\text{Acc}^{(1)}, \text{Acc}^{(2)}\right) \le \frac{\text{Acc}^{(1)} + \text{Acc}^{(2)}}{2} = \text{Acc}_{\text{pairs}}$$
where:
$$\text{Acc}^{(1)} = \frac{1}{|\mathcal{P}|} \sum_{i \in \mathcal{P}} \mathbf{1}_{\hat{y}_i^{(1)} = y_i^{(1)}}, \qquad \text{Acc}^{(2)} = \frac{1}{|\mathcal{P}|} \sum_{i \in \mathcal{P}} \mathbf{1}_{\hat{y}_i^{(2)} = y_i^{(2)}}$$

#### Formal Proof:
1. For any pair $i$, the joint event $(\hat{y}_i^{(1)} = y_i^{(1)} \land \hat{y}_i^{(2)} = y_i^{(2)})$ implies both individual marginal events.
2. Thus:
   $$h_i \le \mathbf{1}_{\hat{y}_i^{(1)} = y_i^{(1)}} \quad \text{and} \quad h_i \le \mathbf{1}_{\hat{y}_i^{(2)} = y_i^{(2)}}$$
3. Summing and dividing by $|\mathcal{P}|$:
   $$\HardCons \le \text{Acc}^{(1)} \quad \text{and} \quad \HardCons \le \text{Acc}^{(2)}$$
4. Hence, $\HardCons \le \min(\text{Acc}^{(1)}, \text{Acc}^{(2)})$.
5. Since $\min(a, b) \le \frac{a + b}{2}$, the arithmetic mean of directional accuracies provides an upper bound.
$\blacksquare$

*Significance for System Optimization:*  
If an unregularized baseline achieves $\text{Acc}^{(1)} = 0.85$ on forward instances but suffers a Reversal Curse on backward instances dropping to $\text{Acc}^{(2)} = 0.65$, HardCons cannot exceed $0.65$, regardless of how high forward accuracy is. Balancing both directions via BSA and CRL is mandatory to raise HardCons.

---

### Theorem 3 (Invariance of Symmetrized Decoding)
Let $p^{(1)}, p^{(2)} \in \Delta^3$ be the predicted probability distributions over classes $\{0, 1, 2, 3\}$ for $(A, B)$ and $(B, A)$.  
Let $P$ be the involution permutation matrix.  
Define the symmetrized post-hoc probabilities:
$$p_{\text{sym}}^{(1)} = \frac{1}{2}\left(p^{(1)} + p^{(2)} P\right), \qquad p_{\text{sym}}^{(2)} = \frac{1}{2}\left(p^{(2)} + p^{(1)} P\right) = p_{\text{sym}}^{(1)} P$$
If the predicted class $\hat{k}^{(1)} = \operatorname{argmax}_c (p_{\text{sym}}^{(1)})_c \in \{0, 1, 2\}$, then:
$$\hat{k}^{(2)} = \operatorname{argmax}_c (p_{\text{sym}}^{(2)})_c = \pi(\hat{k}^{(1)}) = \Rev(\hat{k}^{(1)})$$
and the pair is guaranteed to satisfy $s_i = 1$.  
Consequently, under Symmetrized Decoding, $\SoftCons \equiv 1.0$ across all pairs where $\operatorname{argmax} \neq 3$ (non-negative).

---

## 5. Directional Transition Matrix & Inconsistency Taxonomy (`src/metrics/consistency.py`)

Standard confusion matrices compare predictions against ground-truth labels. However, to diagnose the Reversal Curse, we must inspect the **cross-directional transition matrix** between direct and reverse predictions.

### 5.1 Formulation of the 4x4 Directional Transition Matrix $T$

For every unique reversible pair $i \in \mathcal{P}$, let:
- $x_i^{(dir)}$ be the canonical forward instance (e.g., variant `original` or ordered $id_1 < id_2$).
- $x_i^{(rev)}$ be the reciprocal reverse instance (e.g., variant `flipped`).
- $\hat{y}_i^{(dir)} \in \{0, 1, 2, 3\}$ be the model prediction for the direct presentation.
- $\hat{y}_i^{(rev)} \in \{0, 1, 2, 3\}$ be the model prediction for the reverse presentation.

The $4 \times 4$ Directional Transition Matrix $T \in \mathbb{Z}_{\ge 0}^{4 \times 4}$ is defined by:
$$T_{j, k} = \sum_{i \in \mathcal{P}} \mathbf{1}_{\left(\hat{y}_i^{(dir)} = j \;\land\; \hat{y}_i^{(rev)} = k\right)}$$
where row $j$ represents the direct prediction and column $k$ represents the reverse prediction:
$$\text{Classes: } [0: \text{BACKWARD}, \; 1: \text{EQUIV}, \; 2: \text{FORWARD}, \; 3: \text{NEGATIVE}]$$

$$T = \begin{pmatrix}
T_{0, 0} & T_{0, 1} & T_{0, 2} & T_{0, 3} \\
T_{1, 0} & T_{1, 1} & T_{1, 2} & T_{1, 3} \\
T_{2, 0} & T_{2, 1} & T_{2, 2} & T_{2, 3} \\
T_{3, 0} & T_{3, 1} & T_{3, 2} & T_{3, 3}
\end{pmatrix}$$

#### Matrix Representations:
1. **Raw Count Matrix ($T$):** Total counts satisfying $\sum_{j=0}^3 \sum_{k=0}^3 T_{j, k} = |\mathcal{P}|$.
2. **Row-Normalized Conditional Probability Matrix ($T^{\text{cond}}$):**
   $$T^{\text{cond}}_{j, k} = P\left(\hat{y}^{(rev)} = k \mid \hat{y}^{(dir)} = j\right) = \frac{T_{j, k}}{\sum_{m=0}^3 T_{j, m}}$$
3. **Globally Normalized Joint Probability Distribution Matrix ($T^{\text{joint}}$):**
   $$T^{\text{joint}}_{j, k} = P\left(\hat{y}^{(dir)} = j, \;\hat{y}^{(rev)} = k\right) = \frac{T_{j, k}}{|\mathcal{P}|}$$

#### Algebraic Connection to SoftCons:
Admissible logically consistent transitions correspond to entries where $k = \pi(j)$ for $j \in \{0, 1, 2\}$:
- Direct `BACKWARD` (0) $\to$ Reverse `FORWARD` (2): $T_{0, 2}$
- Direct `EQUIVALENCE` (1) $\to$ Reverse `EQUIVALENCE` (1): $T_{1, 1}$
- Direct `FORWARD` (2) $\to$ Reverse `BACKWARD` (0): $T_{2, 0}$

Using the permutation matrix $P$:
$$(T P)_{j, j} = \sum_{k=0}^3 T_{j, k} P_{k, j} = T_{j, \pi(j)}$$
Taking the matrix trace:
$$\operatorname{Tr}(T P) = T_{0, 2} + T_{1, 1} + T_{2, 0} + T_{3, 3}$$
Because $T_{3, 3}$ represents mutual prediction of `NEGATIVE_OTHER` (which is excluded from SoftCons), the exact number of soft-consistent pairs is given by the algebraic expression:
$$\text{Consistent Pairs} = \operatorname{Tr}(T P) - T_{3, 3} = T_{0, 2} + T_{1, 1} + T_{2, 0}$$
$$\SoftCons = \frac{\operatorname{Tr}(T P) - T_{3, 3}}{|\mathcal{P}|}$$

---

### 5.2 The 16-Cell Inconsistency Taxonomy

The 16 cells of the transition matrix $T$ form an exhaustive, mutually exclusive partition into 4 distinct behavioral categories:

```
                          Reverse Prediction (B, A)
                       BWD (0)     EQUIV (1)     FWD (2)      NEG (3)
       ┌─────────────┬───────────┬─────────────┬───────────┬───────────┐
BWD (0)│ Entailment  │ Equiv-Asym│ CONSISTENT  │ Noise-Drop│           │
       │ Loop (BWD)  │           │ (BWD->FWD)  │           │           │
       ├─────────────┼───────────┼─────────────┼───────────┤           │
EQU (1)│ Equiv-Asym  │CONSISTENT │ Equiv-Asym  │ Equiv-Drop│           │
       │             │(EQU->EQU) │             │           │           │
       ├─────────────┼───────────┼─────────────┼───────────┤           │
FWD (2)│ CONSISTENT  │ Equiv-Asym│ Entailment  │ Noise-Drop│           │
       │ (FWD->BWD)  │           │ Loop (FWD)  │           │           │
       ├─────────────┼───────────┼─────────────┼───────────┤           │
NEG (3)│ Noise-Rise  │ Equiv-Rise│ Noise-Rise  │ Non-Rev   │           │
       │             │           │             │ Noise (x) │           │
       └─────────────┴───────────┴─────────────┴───────────┴───────────┘
```

#### Class 1: Logically Consistent Transitions (3 cells)
- **Cell $T_{2, 0}$ (`FWD_TO_BWD`):** Direct is `FORWARD`, Reverse is `BACKWARD`. Perfect directional inversion.
- **Cell $T_{0, 2}$ (`BWD_TO_FWD`):** Direct is `BACKWARD`, Reverse is `FORWARD`. Perfect directional inversion.
- **Cell $T_{1, 1}$ (`EQUIV_TO_EQUIV`):** Direct is `EQUIVALENCE`, Reverse is `EQUIVALENCE`. Symmetric preservation.
- **Rate:** $\text{Rate}_{\text{consistent}} = \frac{T_{2, 0} + T_{0, 2} + T_{1, 1}}{|\mathcal{P}|} = \SoftCons$.

#### Class 2: Entailment Loop / Reversal Curse Trap (2 cells)
The model detects semantic entailment between the phrases but fails to invert the direction upon argument swap:
- **Cell $T_{2, 2}$ (`FWD_FWD_LOOP`):** Direct `FORWARD` and Reverse `FORWARD`. The model predicts $A \implies B$ and $B \implies A$, but classifies neither as `EQUIVALENCE`. This is the classic unidirectional bias of pretrained LLMs and cross-encoders.
- **Cell $T_{0, 0}$ (`BWD_BWD_LOOP`):** Direct `BACKWARD` and Reverse `BACKWARD`.
- **Rate:** $\text{Rate}_{\text{loop}} = \frac{T_{2, 2} + T_{0, 0}}{|\mathcal{P}|}$.

#### Class 3: Equivalence Asymmetry (6 cells)
Equivalence in natural language logic is fundamentally symmetric ($A \equiv B \iff B \equiv A$). A model predicting equivalence in one orientation but non-equivalence in the reciprocal orientation exhibits structural reasoning breakdown:
- **`EQUIV_TO_DIR` ($T_{1, 0} + T_{1, 2}$):** Predicts `EQUIVALENCE` forward, but directional entailment backward.
- **`DIR_TO_EQUIV` ($T_{0, 1} + T_{2, 1}$):** Predicts directional entailment forward, but `EQUIVALENCE` backward.
- **`EQUIV_TO_NEG` ($T_{1, 3}$):** Predicts `EQUIVALENCE` forward, but noise backward.
- **`NEG_TO_EQUIV` ($T_{3, 1}$):** Predicts noise forward, but `EQUIVALENCE` backward.
- **Rate:** $\text{Rate}_{\text{equiv\_asym}} = \frac{T_{1, 0} + T_{1, 2} + T_{0, 1} + T_{2, 1} + T_{1, 3} + T_{3, 1}}{|\mathcal{P}|}$.

#### Class 4: Semantic Dropout & Noise Collapse (5 cells)
Reversal causes semantic representation collapse into unrelated noise:
- **`DIR_TO_NEG` ($T_{0, 3} + T_{2, 3}$):** Valid directional entailment drops to `NEGATIVE_OTHER` when reversed.
- **`NEG_TO_DIR` ($T_{3, 0} + T_{3, 2}$):** Noise in forward direction inexplicably rises to directional entailment when reversed.
- **`MUTUAL_NEG` ($T_{3, 3}$):** Both directions predicted as `NEGATIVE_OTHER` on a known reversible pair.
- **Rate:** $\text{Rate}_{\text{noise}} = \frac{T_{0, 3} + T_{2, 3} + T_{3, 0} + T_{3, 2} + T_{3, 3}}{|\mathcal{P}|}$.

#### Partition Verification:
$$\text{Rate}_{\text{consistent}} + \text{Rate}_{\text{loop}} + \text{Rate}_{\text{equiv\_asym}} + \text{Rate}_{\text{noise}} = 1.000000000000$$

---

### 5.3 Diagnostic Indices & Metrics

1. **Directional Inconsistency Rate ($\operatorname{Inc}$):**
   $$\operatorname{Inc} = 1.0 - \SoftCons = \frac{\text{Loop} + \text{EquivAsym} + \text{Noise}}{|\mathcal{P}|}$$

2. **Transition Entropy ($H_T$):**
   Measures the dispersion of reverse predictions given direct predictions. For direct class $j$:
   $$H_j = - \sum_{k=0}^3 T^{\text{cond}}_{j, k} \log_2 \left(T^{\text{cond}}_{j, k} + \epsilon\right)$$
   For an ideally consistent model, $H_j = 0$ for all $j$, as all probability mass is concentrated on the unique reciprocal label $\pi(j)$.

3. **Reversal Curse Severity Index ($\operatorname{RCSI}$):**
   Specific metric quantifying the proportion of directional entailment pairs that fall into the entailment loop:
   $$\operatorname{RCSI} = \frac{T_{2, 2} + T_{0, 0}}{\sum_{k} T_{2, k} + \sum_{k} T_{0, k}}$$

---

## 6. Comprehensive Implementation Blueprint

### 6.1 `src/metrics/metrics.py` Blueprint

```python
"""
src/metrics/metrics.py

Official evaluation metrics implementation for SemEval-2027 Task 2 (DiCo-NLI).
Guarantees exact parity with official evaluation functions and CodaBench scoring:
  - Multiclass Weighted F1 across 4 labels with zero-division handling.
  - SoftCons over unique reversible source pairs.
  - HardCons over unique reversible source pairs.
  - Enforces mathematical invariant: 0.0 <= HardCons <= SoftCons <= 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union
import numpy as np


# -----------------------------------------------------------------------------
# 1. Label Inventory and Inversion Constants
# -----------------------------------------------------------------------------

class TaskLabel(str, Enum):
    """Canonical 4-class labels for SemEval-2027 Task 2."""
    BACKWARD_ENTAILMENT = "BACKWARD_ENTAILMENT"
    EQUIVALENCE = "EQUIVALENCE"
    FORWARD_ENTAILMENT = "FORWARD_ENTAILMENT"
    NEGATIVE_OTHER = "NEGATIVE_OTHER"


LABEL_TO_IDX: Dict[str, int] = {
    TaskLabel.BACKWARD_ENTAILMENT.value: 0,
    TaskLabel.EQUIVALENCE.value: 1,
    TaskLabel.FORWARD_ENTAILMENT.value: 2,
    TaskLabel.NEGATIVE_OTHER.value: 3,
}

IDX_TO_LABEL: Dict[int, str] = {idx: label for label, idx in LABEL_TO_IDX.items()}

ALL_LABELS: Tuple[str, ...] = (
    TaskLabel.BACKWARD_ENTAILMENT.value,
    TaskLabel.EQUIVALENCE.value,
    TaskLabel.FORWARD_ENTAILMENT.value,
    TaskLabel.NEGATIVE_OTHER.value,
)

REVERSIBLE_LABELS: Tuple[str, ...] = (
    TaskLabel.BACKWARD_ENTAILMENT.value,
    TaskLabel.EQUIVALENCE.value,
    TaskLabel.FORWARD_ENTAILMENT.value,
)

REVERSE_LABEL_MAP: Dict[str, str] = {
    TaskLabel.BACKWARD_ENTAILMENT.value: TaskLabel.FORWARD_ENTAILMENT.value,
    TaskLabel.EQUIVALENCE.value: TaskLabel.EQUIVALENCE.value,
    TaskLabel.FORWARD_ENTAILMENT.value: TaskLabel.BACKWARD_ENTAILMENT.value,
}

INVOLUTION_PERMUTATION_MATRIX: np.ndarray = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)


def is_reversible_label(label: str) -> bool:
    """Return True if label belongs to the closed reversible set."""
    return label in REVERSE_LABEL_MAP


def reverse_label(label: str) -> str:
    """
    Return deterministic reverse of a reversible label.
    Raises ValueError if label is not in REVERSIBLE_LABELS.
    """
    if label not in REVERSE_LABEL_MAP:
        raise ValueError(f"Label {label!r} has no deterministic reverse.")
    return REVERSE_LABEL_MAP[label]


# -----------------------------------------------------------------------------
# 2. Typed Report Structures
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PerLabelMetrics:
    """Precision, recall, F1, support and confusion statistics for a single label."""
    label: str
    support: int
    predicted: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> Dict[str, Union[str, int, float]]:
        return {
            "label": self.label,
            "support": self.support,
            "predicted": self.predicted,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass(frozen=True, slots=True)
class ClassificationReport:
    """Item-level multiclass classification metrics across all instances."""
    weighted_f1: float
    macro_f1: float
    accuracy: float
    total_instances: int
    per_label: Dict[str, PerLabelMetrics]
    confusion_matrix: Dict[str, Dict[str, int]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "weighted_f1": self.weighted_f1,
            "macro_f1": self.macro_f1,
            "accuracy": self.accuracy,
            "total_instances": self.total_instances,
            "per_label": {k: v.to_dict() for k, v in self.per_label.items()},
            "confusion_matrix": self.confusion_matrix,
        }


@dataclass(frozen=True, slots=True)
class ConsistencyReport:
    """Pair-level directional consistency metrics over reversible pairs."""
    soft_cons: float
    hard_cons: float
    reversible_pairs: int
    soft_consistent_pairs: int
    hard_correct_pairs: int
    pair_errors: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "soft_cons": self.soft_cons,
            "hard_cons": self.hard_cons,
            "reversible_pairs": self.reversible_pairs,
            "soft_consistent_pairs": self.soft_consistent_pairs,
            "hard_correct_pairs": self.hard_correct_pairs,
            "pair_errors": self.pair_errors,
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Master official evaluation report matching CodaBench output."""
    weighted_f1: float
    soft_cons: float
    hard_cons: float
    classification: ClassificationReport
    consistency: ConsistencyReport

    def to_dict(self) -> Dict[str, object]:
        return {
            "weighted_f1": self.weighted_f1,
            "soft_cons": self.soft_cons,
            "hard_cons": self.hard_cons,
            "classification": self.classification.to_dict(),
            "consistency": self.consistency.to_dict(),
        }

    def format_scores_text(self) -> str:
        """Format 12-decimal precision text identical to CodaBench scores.txt."""
        return (
            f"weighted_f1: {self.weighted_f1:.12f}\n"
            f"soft_cons: {self.soft_cons:.12f}\n"
            f"hard_cons: {self.hard_cons:.12f}\n"
        )


@dataclass(frozen=True, slots=True)
class ReversiblePairSpec:
    """Unordered reciprocal pair linking two evaluation instances."""
    first_id: str
    second_id: str
    first_label: str
    second_label: str


# -----------------------------------------------------------------------------
# 3. Official Computation Engine (Dictionary / Parity Mode)
# -----------------------------------------------------------------------------

def safe_divide(numerator: float, denominator: float) -> float:
    """Return 0.0 on zero division, maintaining exact float precision."""
    if denominator == 0.0 or denominator == 0:
        return 0.0
    return numerator / denominator


def compute_classification_report(
    gold_by_id: Mapping[str, str],
    predicted_labels: Mapping[str, str],
    labels: Tuple[str, ...] = ALL_LABELS,
) -> ClassificationReport:
    """
    Compute item-level multiclass Weighted F1, Macro F1, and Accuracy.
    Input:
        gold_by_id: Map of instance_id -> gold_label
        predicted_labels: Map of instance_id -> predicted_label
    """
    total = len(gold_by_id)
    if total == 0:
        raise ValueError("Cannot compute classification metrics for zero instances.")

    confusion: Dict[str, Dict[str, int]] = {g: {p: 0 for p in labels} for g in labels}
    correct = 0

    for instance_id, gold_label in gold_by_id.items():
        pred_label = predicted_labels[instance_id]
        if gold_label not in confusion:
            confusion[gold_label] = {p: 0 for p in labels}
        confusion[gold_label][pred_label] = confusion[gold_label].get(pred_label, 0) + 1
        if gold_label == pred_label:
            correct += 1

    per_label: Dict[str, PerLabelMetrics] = {}
    weighted_f1 = 0.0
    macro_f1_values: List[float] = []

    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        support = sum(confusion[label].values())
        predicted = sum(confusion[other][label] for other in labels)

        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)

        per_label[label] = PerLabelMetrics(
            label=label,
            support=support,
            predicted=predicted,
            true_positive=tp,
            false_positive=fp,
            false_negative=fn,
            precision=precision,
            recall=recall,
            f1=f1,
        )
        weighted_f1 += (support / total) * f1
        if support > 0:
            macro_f1_values.append(f1)

    macro_f1 = sum(macro_f1_values) / len(macro_f1_values) if macro_f1_values else 0.0
    accuracy = correct / total

    return ClassificationReport(
        weighted_f1=weighted_f1,
        macro_f1=macro_f1,
        accuracy=accuracy,
        total_instances=total,
        per_label=per_label,
        confusion_matrix=confusion,
    )


def compute_consistency_report(
    pairs: Sequence[ReversiblePairSpec],
    predicted_labels: Mapping[str, str],
    max_pair_errors: int = 50,
) -> ConsistencyReport:
    """
    Compute official SoftCons and HardCons over reversible source pairs.
    """
    total = len(pairs)
    if total == 0:
        raise ValueError("Cannot compute consistency metrics without reversible pairs.")

    soft_count = 0
    hard_count = 0
    pair_errors: List[Dict[str, str]] = []

    for pair in pairs:
        pred_1 = predicted_labels[pair.first_id]
        pred_2 = predicted_labels[pair.second_id]

        soft_ok = (
            pred_1 in REVERSIBLE_LABELS
            and pred_2 == reverse_label(pred_1)
        )
        hard_ok = (
            pred_1 == pair.first_label
            and pred_2 == pair.second_label
        )

        if soft_ok:
            soft_count += 1
        if hard_ok:
            hard_count += 1
        elif len(pair_errors) < max_pair_errors:
            pair_errors.append({
                "first_id": pair.first_id,
                "second_id": pair.second_id,
                "first_gold": pair.first_label,
                "second_gold": pair.second_label,
                "first_prediction": pred_1,
                "second_prediction": pred_2,
                "soft_consistent": str(soft_ok).lower(),
                "hard_correct": str(hard_ok).lower(),
            })

    soft_cons = soft_count / total
    hard_cons = hard_count / total

    # Enforce mathematical invariant sanity check
    assert hard_cons <= soft_cons + 1e-12, (
        f"Mathematical invariant violated: HardCons ({hard_cons}) > SoftCons ({soft_cons})"
    )

    return ConsistencyReport(
        soft_cons=soft_cons,
        hard_cons=hard_cons,
        reversible_pairs=total,
        soft_consistent_pairs=soft_count,
        hard_correct_pairs=hard_count,
        pair_errors=pair_errors,
    )


def compute_official_evaluation(
    gold_by_id: Mapping[str, str],
    predicted_labels: Mapping[str, str],
    reversible_pairs: Sequence[ReversiblePairSpec],
    max_pair_errors: int = 50,
) -> EvaluationReport:
    """
    Compute complete official evaluation profile (Weighted F1, SoftCons, HardCons).
    Matches SemEval-2027 CodaBench evaluation exactly.
    """
    cls_report = compute_classification_report(gold_by_id, predicted_labels)
    cons_report = compute_consistency_report(
        reversible_pairs, predicted_labels, max_pair_errors=max_pair_errors
    )

    return EvaluationReport(
        weighted_f1=cls_report.weighted_f1,
        soft_cons=cons_report.soft_cons,
        hard_cons=cons_report.hard_cons,
        classification=cls_report,
        consistency=cons_report,
    )


# -----------------------------------------------------------------------------
# 4. High-Throughput Vectorized API (for Training & Fast Validation)
# -----------------------------------------------------------------------------

def compute_metrics_from_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pair_indices: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Vectorized computation for GPU/CPU PyTorch training loops.
    Args:
        y_true: 1D integer array of shape (N,) with class indices in [0, 3].
        y_pred: 1D integer array of shape (N,) with class indices in [0, 3].
        pair_indices: 2D integer array of shape (P, 2) indexing the direct and
                      reverse instance row indices for each reversible pair.
    Returns:
        Dict containing: 'weighted_f1', 'macro_f1', 'accuracy',
        and optionally 'soft_cons', 'hard_cons' if pair_indices provided.
    """
    n_classes = len(ALL_LABELS)
    total = len(y_true)
    if total == 0:
        raise ValueError("Cannot evaluate empty arrays.")

    # Confusion matrix
    conf = np.bincount(
        y_true * n_classes + y_pred, minlength=n_classes * n_classes
    ).reshape(n_classes, n_classes)

    tp = np.diag(conf)
    fp = np.sum(conf, axis=0) - tp
    fn = np.sum(conf, axis=1) - tp
    support = np.sum(conf, axis=1)

    precision = np.divide(tp, tp + fp, out=np.zeros(n_classes), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros(n_classes), where=(tp + fn) != 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(n_classes), where=denom != 0)

    weighted_f1 = float(np.sum((support / total) * f1))
    valid_support = support > 0
    macro_f1 = float(np.mean(f1[valid_support])) if np.any(valid_support) else 0.0
    accuracy = float(np.sum(tp) / total)

    result = {
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
    }

    if pair_indices is not None and len(pair_indices) > 0:
        idx_1 = pair_indices[:, 0]
        idx_2 = pair_indices[:, 1]

        p1 = y_pred[idx_1]
        p2 = y_pred[idx_2]
        t1 = y_true[idx_1]
        t2 = y_true[idx_2]

        # Invert p1 in index space: 0 <-> 2, 1 -> 1, 3 -> -1
        rev_p1 = np.where(p1 == 0, 2, np.where(p1 == 2, 0, np.where(p1 == 1, 1, -1)))

        soft_ok = (p1 != 3) & (p2 == rev_p1)
        hard_ok = (p1 == t1) & (p2 == t2)

        soft_cons = float(np.mean(soft_ok))
        hard_cons = float(np.mean(hard_ok))

        result["soft_cons"] = soft_cons
        result["hard_cons"] = hard_cons

    return result
```

---

### 6.2 `src/metrics/consistency.py` Blueprint

```python
"""
src/metrics/consistency.py

Directional Transition Matrix and Inconsistency Taxonomy analysis engine.
Deconstructs model predictions across ordered pairs (A, B) vs (B, A) into:
  - 4x4 Directional Transition Matrix (raw counts, conditional probabilities, joint distribution).
  - 16-cell Inconsistency Taxonomy (Consistent, Entailment Loop, Equivalence Asymmetry, Noise).
  - Syntactic category error attribution.
  - LaTeX tabular and ASCII formatted outputs for paper and logs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np

from .metrics import (
    ALL_LABELS,
    IDX_TO_LABEL,
    INVOLUTION_PERMUTATION_MATRIX,
    LABEL_TO_IDX,
    REVERSIBLE_LABELS,
    ReversiblePairSpec,
    TaskLabel,
)


@dataclass(frozen=True, slots=True)
class InconsistencyTaxonomyBreakdown:
    """Detailed partition of all 16 transition matrix cells."""
    total_pairs: int
    # 1. Consistent transitions
    consistent_count: int
    consistent_rate: float
    fwd_to_bwd: int
    bwd_to_fwd: int
    equiv_to_equiv: int
    # 2. Entailment Loop / Reversal Curse Trap
    entailment_loop_count: int
    entailment_loop_rate: float
    fwd_fwd_bias: int
    bwd_bwd_bias: int
    # 3. Equivalence Asymmetry
    equiv_asymmetry_count: int
    equiv_asymmetry_rate: float
    equiv_to_dir: int
    dir_to_equiv: int
    equiv_to_noise: int
    noise_to_equiv: int
    # 4. Semantic Dropout / Noise Collapse
    noise_collapse_count: int
    noise_collapse_rate: float
    dir_to_noise: int
    noise_to_dir: int
    mutual_noise: int

    def to_dict(self) -> Dict[str, Union[int, float]]:
        return {
            "total_pairs": self.total_pairs,
            "consistent_count": self.consistent_count,
            "consistent_rate": self.consistent_rate,
            "fwd_to_bwd": self.fwd_to_bwd,
            "bwd_to_fwd": self.bwd_to_fwd,
            "equiv_to_equiv": self.equiv_to_equiv,
            "entailment_loop_count": self.entailment_loop_count,
            "entailment_loop_rate": self.entailment_loop_rate,
            "fwd_fwd_bias": self.fwd_fwd_bias,
            "bwd_bwd_bias": self.bwd_bwd_bias,
            "equiv_asymmetry_count": self.equiv_asymmetry_count,
            "equiv_asymmetry_rate": self.equiv_asymmetry_rate,
            "equiv_to_dir": self.equiv_to_dir,
            "dir_to_equiv": self.dir_to_equiv,
            "equiv_to_noise": self.equiv_to_noise,
            "noise_to_equiv": self.noise_to_equiv,
            "noise_collapse_count": self.noise_collapse_count,
            "noise_collapse_rate": self.noise_collapse_rate,
            "dir_to_noise": self.dir_to_noise,
            "noise_to_dir": self.noise_to_dir,
            "mutual_noise": self.mutual_noise,
        }


@dataclass(frozen=True, slots=True)
class DirectionalTransitionReport:
    """Complete diagnostic report for directional transitions."""
    raw_matrix: np.ndarray             # Shape (4, 4), dtype int64
    conditional_matrix: np.ndarray     # Shape (4, 4), dtype float64 (row-normalized)
    joint_matrix: np.ndarray           # Shape (4, 4), dtype float64 (total-normalized)
    taxonomy: InconsistencyTaxonomyBreakdown
    transition_entropy: float          # Mean conditional transition entropy (bits)
    reversal_curse_index: float        # Proportion of directional entailments in loop

    def format_ascii_table(self) -> str:
        """Render readable ASCII transition matrix table."""
        header = f"{'Direct \\ Reverse':<18} | {'BWD (0)':<9} | {'EQUIV (1)':<9} | {'FWD (2)':<9} | {'NEG (3)':<9} | {'Total':<6}"
        divider = "-" * len(header)
        lines = [header, divider]
        short_names = ["BWD (0)", "EQUIV (1)", "FWD (2)", "NEG (3)"]

        for j in range(4):
            row_vals = " | ".join(f"{self.raw_matrix[j, k]:>9d}" for k in range(4))
            row_sum = int(np.sum(self.raw_matrix[j, :]))
            lines.append(f"{short_names[j]:<18} | {row_vals} | {row_sum:>6d}")

        lines.append(divider)
        col_sums = " | ".join(f"{int(np.sum(self.raw_matrix[:, k])):>9d}" for k in range(4))
        tot = int(np.sum(self.raw_matrix))
        lines.append(f"{'Total':<18} | {col_sums} | {tot:>6d}")
        return "\n".join(lines)

    def format_latex_table(self, caption: str = "Directional Transition Matrix") -> str:
        """Render LaTeX tabular block suitable for report/paper.tex."""
        s = [
            "\\begin{table}[ht]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{l|cccc|r}",
            "\\toprule",
            "\\textbf{Direct $\\downarrow$ / Reverse $\\rightarrow$} & \\textbf{BWD} & \\textbf{EQU} & \\textbf{FWD} & \\textbf{NEG} & \\textbf{Total} \\\\",
            "\\midrule",
        ]
        labels = ["\\textbf{BWD}", "\\textbf{EQU}", "\\textbf{FWD}", "\\textbf{NEG}"]
        for j in range(4):
            # Highlight consistent cells with bold or box
            cells = []
            for k in range(4):
                val = self.raw_matrix[j, k]
                is_cons = (j == 0 and k == 2) or (j == 1 and k == 1) or (j == 2 and k == 0)
                if is_cons:
                    cells.append(f"\\textbf{{{val}}}")
                else:
                    cells.append(f"{val}")
            row_sum = int(np.sum(self.raw_matrix[j, :]))
            s.append(f"{labels[j]} & " + " & ".join(cells) + f" & {row_sum} \\\\")

        s.append("\\midrule")
        col_sums = [str(int(np.sum(self.raw_matrix[:, k]))) for k in range(4)]
        tot = int(np.sum(self.raw_matrix))
        s.append("\\textbf{Total} & " + " & ".join(col_sums) + f" & \\textbf{{{tot}}} \\\\")
        s.append("\\bottomrule")
        s.append("\\end{tabular}")
        s.append(f"\\caption{{{caption}}}")
        s.append("\\label{tab:transition_matrix}")
        s.append("\\end{table}")
        return "\n".join(s)


def compute_directional_transition_matrix(
    pairs: Sequence[ReversiblePairSpec],
    predicted_labels: Mapping[str, str],
    canonical_orientation: bool = True,
) -> DirectionalTransitionReport:
    """
    Compute 4x4 Directional Transition Matrix and Inconsistency Taxonomy.
    Args:
        pairs: Reversible pairs list.
        predicted_labels: Map of instance_id -> predicted label.
        canonical_orientation: If True, first_id is Direct and second_id is Reverse.
    """
    T = np.zeros((4, 4), dtype=np.int64)

    for pair in pairs:
        pred_dir = predicted_labels[pair.first_id]
        pred_rev = predicted_labels[pair.second_id]

        idx_dir = LABEL_TO_IDX[pred_dir]
        idx_rev = LABEL_TO_IDX[pred_rev]

        T[idx_dir, idx_rev] += 1

    total = int(np.sum(T))
    if total == 0:
        raise ValueError("Cannot compute transition matrix on 0 pairs.")

    # Conditional probability matrix (row-normalized with safe division)
    row_sums = T.sum(axis=1, keepdims=True)
    cond_T = np.divide(T, row_sums, out=np.zeros((4, 4), dtype=np.float64), where=row_sums != 0)

    # Joint probability matrix
    joint_T = T.astype(np.float64) / total

    # 1. Consistent cells
    fwd_to_bwd = int(T[2, 0])
    bwd_to_fwd = int(T[0, 2])
    equiv_to_equiv = int(T[1, 1])
    consistent_count = fwd_to_bwd + bwd_to_fwd + equiv_to_equiv

    # 2. Entailment Loop (Reversal Curse Trap)
    fwd_fwd_bias = int(T[2, 2])
    bwd_bwd_bias = int(T[0, 0])
    entailment_loop_count = fwd_fwd_bias + bwd_bwd_bias

    # 3. Equivalence Asymmetry
    equiv_to_dir = int(T[1, 0] + T[1, 2])
    dir_to_equiv = int(T[0, 1] + T[2, 1])
    equiv_to_noise = int(T[1, 3])
    noise_to_equiv = int(T[3, 1])
    equiv_asym_count = equiv_to_dir + dir_to_equiv + equiv_to_noise + noise_to_equiv

    # 4. Semantic Dropout / Noise Collapse
    dir_to_noise = int(T[0, 3] + T[2, 3])
    noise_to_dir = int(T[3, 0] + T[3, 2])
    mutual_noise = int(T[3, 3])
    noise_collapse_count = dir_to_noise + noise_to_dir + mutual_noise

    # Taxonomy breakdown
    taxonomy = InconsistencyTaxonomyBreakdown(
        total_pairs=total,
        consistent_count=consistent_count,
        consistent_rate=consistent_count / total,
        fwd_to_bwd=fwd_to_bwd,
        bwd_to_fwd=bwd_to_fwd,
        equiv_to_equiv=equiv_to_equiv,
        entailment_loop_count=entailment_loop_count,
        entailment_loop_rate=entailment_loop_count / total,
        fwd_fwd_bias=fwd_fwd_bias,
        bwd_bwd_bias=bwd_bwd_bias,
        equiv_asymmetry_count=equiv_asym_count,
        equiv_asymmetry_rate=equiv_asym_count / total,
        equiv_to_dir=equiv_to_dir,
        dir_to_equiv=dir_to_equiv,
        equiv_to_noise=equiv_to_noise,
        noise_to_equiv=noise_to_equiv,
        noise_collapse_count=noise_collapse_count,
        noise_collapse_rate=noise_collapse_count / total,
        dir_to_noise=dir_to_noise,
        noise_to_dir=noise_to_dir,
        mutual_noise=mutual_noise,
    )

    # Transition Entropy: mean Shannon entropy over rows (in bits)
    entropy_per_row = []
    for j in range(4):
        probs = cond_T[j, :]
        p_nonzero = probs[probs > 0]
        if len(p_nonzero) > 0:
            h = -float(np.sum(p_nonzero * np.log2(p_nonzero)))
            entropy_per_row.append(h)
        else:
            entropy_per_row.append(0.0)
    transition_entropy = float(np.mean(entropy_per_row))

    # Reversal Curse Index: proportion of directional entailments falling into loop
    dir_entailment_total = int(np.sum(T[2, :]) + np.sum(T[0, :]))
    rcsi = (
        float((fwd_fwd_bias + bwd_bwd_bias) / dir_entailment_total)
        if dir_entailment_total > 0
        else 0.0
    )

    return DirectionalTransitionReport(
        raw_matrix=T,
        conditional_matrix=cond_T,
        joint_matrix=joint_T,
        taxonomy=taxonomy,
        transition_entropy=transition_entropy,
        reversal_curse_index=rcsi,
    )
```

---

## 7. Syntactic Chunk Category Disaggregation

In SemEval-2027 Task 2, instances originate from five syntactic phrase structures:
1. **Nominal Phrases (`NP`):** Entities and modifiers (e.g., *"Wikileaks soldier Bradley Manning"* vs *"Bradley Manning"*).
2. **Verb Chains (`VP`):** Aspectual predicates, gerunds, modals (e.g., *"are playing"* vs *"run"*).
3. **Prepositional Phrases (`PP`):** Spatial, temporal, locative adjuncts (e.g., *"with a tennis ball"* vs *"with a ball"*).
4. **Adverbial Phrases (`ADVP`):** Manner and trajectory modifiers (e.g., *"down a snowy hill"* vs *"sledding"*).
5. **Quantificational Expressions (`QUANT`):** Generalized quantifiers and numerals (e.g., *"Nine children"* vs *"Five children"*).

### Design for `compute_consistency_by_category`
When metadata includes the syntactic category, `src/metrics/consistency.py` provides:
```python
def compute_consistency_by_category(
    pairs: Sequence[ReversiblePairSpec],
    pair_categories: Mapping[Tuple[str, str], str],
    predicted_labels: Mapping[str, str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute SoftCons, HardCons, and RCSI partitioned by syntactic category.
    Returns:
        {category_name: {'count': int, 'soft_cons': float, 'hard_cons': float, 'rcsi': float}}
    """
```
This produces the granular empirical breakdown required for Section 5 of the research paper (`report/paper.tex`).

---

## 8. Test Specifications (`tests/test_metrics.py`)

A comprehensive unit test suite in `tests/test_metrics.py` must validate all mathematical properties, edge cases, and official parity.

### Test Scenarios:

| # | Test Case | Inputs | Expected Output / Invariant |
|---|---|---|---|
| 1 | **Exact Parity with Official Package** | 10 mock instances + 4 reversible pairs | $F_w$, $\SoftCons$, $\HardCons$ match `evaluation_functions` to 12 decimal places |
| 2 | **Perfect Prediction** | Predictions match gold for all items | $F_w = 1.0$, $\SoftCons = 1.0$, $\HardCons = 1.0$ |
| 3 | **Soft-Consistent but Zero Correctness** | All predicted `EQUIVALENCE` when gold is `FORWARD`/`BACKWARD` | $\SoftCons = 1.0$, $\HardCons = 0.0$, $F_w \approx 0.0$ |
| 4 | **The Reversal Curse Trap** | Direct predicted `FORWARD`, Reverse predicted `FORWARD` | $\SoftCons = 0.0$, $\HardCons = 0.0$, $T_{2, 2} = |\mathcal{P}|$ |
| 5 | **Noise Penalty Rule** | Reversible pair predicted as `NEGATIVE_OTHER` in both directions | $s_i = 0$, $h_i = 0$, $\SoftCons = 0.0$, $\HardCons = 0.0$ |
| 6 | **Invariant Guarantee (Theorem 1)** | 100 random prediction configurations | Asserts $0.0 \le \HardCons \le \SoftCons \le 1.0$ on every iteration |
| 7 | **Directional Bound (Theorem 2)** | Asymmetric accuracy model ($\text{Acc}_1=0.9, \text{Acc}_2=0.4$) | Asserts $\HardCons \le \min(0.9, 0.4) = 0.40$ |
| 8 | **Zero Division Handling** | Label with 0 support and 0 predictions | `_safe_divide` returns $0.0$, no NaN/Inf, Macro F1 excludes label |
| 9 | **Empty Input Guard** | Empty dictionary passed to metric calculators | Raises `ValueError` with clear message |
| 10 | **Matrix Consistency Identity** | Any valid prediction set | $\operatorname{Tr}(T P) - T_{3, 3} == \text{soft\_count}$ |
| 11 | **Taxonomy Partition Completeness** | Any valid prediction set | $\text{Consistent} + \text{Loop} + \text{EquivAsym} + \text{Noise} == |\mathcal{P}|$ |
| 12 | **Vectorized API Parity** | Array input vs Dictionary input | `weighted_f1`, `soft_cons`, `hard_cons` identical to float epsilon $< 10^{-7}$ |

---

## 9. Integration with Other Modules

1. **`src/data/reversal.py`**:
   - Supplies `ReversiblePairSpec` records extracted from data splits.
   - Shares canonical label token constants and involution mapping.
2. **`src/models/decoding.py`**:
   - Post-hoc symmetrized decoding utilizes the exact same permutation matrix $P$.
   - Tested against `compute_consistency_report` to empirically verify $\SoftCons = 100\%$.
3. **`src/training/trainer.py`**:
   - Calls `compute_metrics_from_arrays` at each validation epoch for fast evaluation.
4. **`src/evaluation/matrices.py` & `report/paper.tex`**:
   - Directly consumes `DirectionalTransitionReport.format_latex_table()` to populate Table 3 of the final paper.

---

## 10. Conclusion & Action Items

The design for `src/metrics/metrics.py` and `src/metrics/consistency.py` is fully specified with mathematical rigor and exact official parity. All edge cases, invariants, data models, and matrix taxonomies are defined.

**Immediate Next Steps:**
1. Implement `src/metrics/metrics.py` according to Section 6.1.
2. Implement `src/metrics/consistency.py` according to Section 6.2.
3. Add full unit tests in `tests/test_metrics.py` implementing the 12 scenarios from Section 8.
