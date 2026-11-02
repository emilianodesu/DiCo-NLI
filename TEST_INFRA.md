# End-to-End Test Infrastructure Specification (TEST_INFRA)
## SemEval-2027 Task 2: DiCo-NLI (Directional Consistency in Fine-Grained NLI)

### 1. Executive Summary & Testing Philosophy
This document establishes the comprehensive test infrastructure, methodology, and verification harnesses for SemEval-2027 Task 2 (DiCo-NLI). The primary objective is to guarantee absolute logical correctness, mathematical invariants, and strict competition compliance across all project components:
- **Bijective Reversal Algebra & Involutions**
- **Directional Consistency Metrics (Weighted F1, SoftCons, HardCons)**
- **Strict Codabench Submission Formatting & Validation**
- **Equivariant Symmetrized Decoding Guarantee ($\text{SoftCons} \equiv 1.0000$)**
- **Sub-60-Second Fast CPU End-to-End Execution Pipeline**

#### Core Principles:
1. **Authoritative Oracle Derivation**: Every test case derives expected outputs from mathematical proofs, SemEval-2027 Task 2 official evaluation specifications (`evaluation_functions/`), or analytical invariants.
2. **Progressive Testability & Decoupling**: Tests are completely isolated, self-contained, and runnable in any standard Python 3.10+ environment without requiring external network connectivity or heavy GPU downloads.
3. **Formal Invariant Verification**: Critical mathematical properties (e.g., $P^2 = I$, $\text{HardCons} \le \text{SoftCons}$, $p_{\text{sym,rev}} = P p_{\text{sym,dir}}$) are verified both analytically and via Monte Carlo simulation across arbitrary probability simplices.
4. **Adversarial Hardening**: Edge cases, boundary values, malformed inputs, whitespace faults, and unaligned labels are rigorously asserted against defined exception contracts.

---

### 2. Test Architecture & Tier Breakdown

| Tier | Category | Scope | Test Module | Target Runtime | Invariants Tested |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Tier 1** | Unit Logic | Reversal Algebra & Pairing | `tests/test_reversal_logic.py` | $<1.0\text{s}$ | Involution $\text{Rev}(\text{Rev}(y)) = y$; Bijective mapping across $\mathcal{R}$; BSA augmentation closure. |
| **Tier 2** | Unit & Property | Evaluation Metrics | `tests/test_metrics.py` | $<2.0\text{s}$ | Support-weighted multiclass F1; $h_i = 1 \implies s_i = 1$; $\text{HardCons} \le \text{SoftCons}$; Zero-division stability. |
| **Tier 3** | Integration | Codabench Submission Integrity | `tests/test_codabench.py` | $<1.0\text{s}$ | Strict 2-column CSV; header `instance_id,label`; no blank lines; no nulls; bijection with gold ID set. |
| **Tier 4** | Mathematical | Symmetrized Decoding Invariant | `tests/test_symmetrized_decoding.py` | $<3.0\text{s}$ | $P_{\text{sym}}(B, A) = P P_{\text{sym}}(A, B)$; $\text{SoftCons} \equiv 100\%$ on arbitrary Dirichlet/random distributions. |
| **Tier 5** | System Integration | Fast CPU End-to-End Smoke Test | `tests/test_pipeline.py` | $<30.0\text{s}$ | Full pipeline (load mock $\to$ train 1 epoch $\to$ symmetrized decode $\to$ export $\to$ validate $\to$ score) in $<60$s on CPU. |

---

### 3. Test Design Methodology

#### 3.1 Category-Partition Method
The input domain is partitioned into mutually exclusive equivalence classes:

1. **Label Domain $\mathcal{L}$**:
   - Class 0: `BACKWARD_ENTAILMENT` (Reversible, maps to Class 2)
   - Class 1: `EQUIVALENCE` (Reversible, symmetric, maps to Class 1)
   - Class 2: `FORWARD_ENTAILMENT` (Reversible, maps to Class 0)
   - Class 3: `NEGATIVE_OTHER` (Non-reversible noise / negative subset)
2. **Reversibility Subset $\mathcal{R}$**:
   - $\mathcal{R} = \{\text{EQUIVALENCE}, \text{FORWARD_ENTAILMENT}, \text{BACKWARD_ENTAILMENT}\}$
   - $\text{Rev}(\cdot)$ is an automorphism on $\mathcal{R}$.
   - Evaluation of `SoftCons` and `HardCons` strictly partitions pairs based on whether gold/prediction belongs to $\mathcal{R}$.
3. **Syntactic Chunk Classes**:
   - Nominal Phrases (NP)
   - Verbal Chains (VP)
   - Prepositional Phrases (PP)
   - Adverbial Expressions (ADVP)
4. **Input Prediction Dialects**:
   - Comma-Separated Values (CSV) with CRLF / LF
   - Tab-Separated Values (TSV)
   - UTF-8 vs UTF-8 with BOM (`utf-8-sig`)

#### 3.2 Boundary Value Analysis (BVA)
1. **Probability Extremes**:
   - Uniform distribution: $p = [0.25, 0.25, 0.25, 0.25]$ (maximal entropy, strict tie condition).
   - One-hot vectors: $p = [1.0, 0.0, 0.0, 0.0]$, etc. (zero entropy).
   - Vanishing probabilities: $p_i = 10^{-15}$ to verify numerical stability against $\log(0)$ under KL/JS divergences.
2. **Support & Dataset Sizes**:
   - Empty input: $|\mathcal{P}| = 0$ (must handle division safely or raise documented `ValueError`).
   - Single pair: $|\mathcal{P}| = 1$ (minimum non-trivial batch).
   - Odd vs even counts of paired instances.
3. **File Boundaries**:
   - Trailing blank lines at EOF (must be rejected by Codabench validator).
   - Embedded commas / quotes in text phrases.
   - Whitespace in `instance_id` (must be rejected).

#### 3.3 Combinatorial / Pairwise State Space
For all pairs of predictions $(\hat{y}_{dir}, \hat{y}_{rev}) \in \mathcal{L} \times \mathcal{L}$, the full $4 \times 4 = 16$ transition space is exhaustively validated:

| $\hat{y}_{dir}$ | $\hat{y}_{rev}$ | Reversible Pair? | $\text{SoftCons}$ ($s_i$) | Category |
| :--- | :--- | :---: | :---: | :--- |
| `EQUIVALENCE` | `EQUIVALENCE` | Yes | **1** | Consistent Symmetric |
| `EQUIVALENCE` | `FORWARD_ENTAILMENT` | Yes | 0 | Inconsistent Asymmetric |
| `EQUIVALENCE` | `BACKWARD_ENTAILMENT`| Yes | 0 | Inconsistent Asymmetric |
| `EQUIVALENCE` | `NEGATIVE_OTHER` | Yes | 0 | Non-reversible Fallback |
| `FORWARD_ENTAILMENT` | `BACKWARD_ENTAILMENT`| Yes | **1** | Consistent Inverted |
| `FORWARD_ENTAILMENT` | `FORWARD_ENTAILMENT` | Yes | 0 | **Reversal Curse Failure Mode** |
| `FORWARD_ENTAILMENT` | `EQUIVALENCE` | Yes | 0 | Inconsistent Asymmetric |
| `FORWARD_ENTAILMENT` | `NEGATIVE_OTHER` | Yes | 0 | Non-reversible Fallback |
| `BACKWARD_ENTAILMENT`| `FORWARD_ENTAILMENT` | Yes | **1** | Consistent Inverted |
| `BACKWARD_ENTAILMENT`| `BACKWARD_ENTAILMENT`| Yes | 0 | **Reversal Curse Failure Mode** |
| `BACKWARD_ENTAILMENT`| `EQUIVALENCE` | Yes | 0 | Inconsistent Asymmetric |
| `BACKWARD_ENTAILMENT`| `NEGATIVE_OTHER` | Yes | 0 | Non-reversible Fallback |
| `NEGATIVE_OTHER` | Any | Yes | 0 | Excluded from SoftCons |

#### 3.4 Real-World Linguistic Scenarios
Test fixtures incorporate genuine phrase pairs modeled after PhrasIS:
- English NP: `("Pope Francis", "pope")` $\to$ `FORWARD_ENTAILMENT`
- English VP: `("is sprinting", "is running")` $\to$ `FORWARD_ENTAILMENT`
- English PP: `("in Madrid", "in Spain")` $\to$ `FORWARD_ENTAILMENT`
- English ADVP: `("extremely quickly", "very fast")` $\to$ `EQUIVALENCE`
- Spanish NP: `("el papa Francisco", "el papa")` $\to$ `FORWARD_ENTAILMENT`
- Spanish VP: `("está corriendo", "se mueve")` $\to$ `FORWARD_ENTAILMENT`
- Spanish PP: `("en una ciudad de España", "en Europa")` $\to$ `FORWARD_ENTAILMENT`
- Spanish ADVP: `("con mucha rapidez", "velozmente")` $\to$ `EQUIVALENCE`

---

### 4. Authoritative Oracle Derivations

#### 4.1 Permutation Matrix $P$ & Involutory Algebra
The permutation matrix representing the reversal operator on label vector $[y_0, y_1, y_2, y_3]^T = [\text{BWD}, \text{EQUIV}, \text{FWD}, \text{NEG}]^T$ is:
$$P = \begin{pmatrix} 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$
Test oracle asserts:
1. $P = P^T$ (Symmetry)
2. $P^2 = I$ (Involution)
3. $\det(P) = -1$ (Orthogonal reflection)
4. $\sum_{j} P_{ij} = 1$ and $\sum_{i} P_{ij} = 1$ (Doubly stochastic permutation)

#### 4.2 Symmetrized Decoding Invariant
Given probability vectors $p_{dir}, p_{rev} \in \Delta^3$:
$$p_{sym,dir} = \frac{1}{2}(p_{dir} + p_{rev} P)$$
$$p_{sym,rev} = \frac{1}{2}(p_{rev} + p_{dir} P)$$
Then:
$$p_{sym,rev} = P p_{sym,dir}$$
Since $\operatorname{argmax}_c (P v)_c = \operatorname{Rev}(\operatorname{argmax}_c v_c)$ under any consistent tie-breaking rule, $\hat{y}_{rev} \equiv \operatorname{Rev}(\hat{y}_{dir})$ holds unconditionally.
The test suite generates 10,000 randomized probability distributions across various Dirichlet concentrations $\alpha \in \{0.1, 1.0, 10.0\}$ and verifies $0$ violations ($\text{SoftCons} = 1.000000$).

#### 4.3 Official Codabench Validation Criteria
Derived directly from `evaluation_functions/io.py` and `evaluation_functions/validation.py`:
- Columns must be exactly `["instance_id", "label"]`.
- Row count must exactly match gold instances.
- Zero duplicate keys in `instance_id`.
- Zero NaN, None, or empty string values.
- Zero whitespace characters in IDs.
- Valid label enum: `{"EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"}`.

---

### 5. Test Execution Protocol

#### Test Execution Commands:
```bash
# 1. Standard library unittest discovery (zero third-party dependencies required)
python3 -m unittest discover -s tests -p "test_*.py" -v

# 2. Pytest execution (when pytest is installed)
pytest tests/ -v --durations=10
```

#### Acceptance Criteria & Pass Gates:
- **Pass Rate**: 100% (0 errors, 0 failures, 0 unexpected skips).
- **Execution Budget**: Total test suite execution on standard CPU must complete in $<60$ seconds.
- **Offline Integrity**: Zero network calls or external downloads allowed during test execution.
