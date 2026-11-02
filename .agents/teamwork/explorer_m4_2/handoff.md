# Handoff Report: Multiclass Confusion & Directional Transition Matrices

**Agent**: `explorer_m4_2` (Role: M4 Confusion & Transition Matrices Explorer)  
**Parent Caller ID**: `df6ce377-6407-4a2d-bd18-829b36e20607`  
**Milestone**: Milestone 4 (R4: Experimental Evaluation & Error Analysis)  
**Target Module**: `src/evaluation/matrices.py`  
**Proposed Reference Implementation**: `.agents/teamwork/explorer_m4_2/proposed_matrices.py`  
**Verified Unit Test Suite**: `.agents/teamwork/explorer_m4_2/test_proposed_matrices.py`

---

## 1. Observation

1. **System & Task Requirements**:
   - `ORIGINAL_REQUEST.md` line 38 (§R4.2) dictates:
     > "2. Matrices de confusión y matrices de transición de inconsistencia direccional."
   - `PROJECT.md` line 38 (Feature 14) specifies:
     > "Inconsistency Transition & Confusion Matrices: 4x4 directional transition and error distribution matrices | M4 | ORIGINAL_REQUEST §R4.2"
   - `PROJECT.md` line 116 specifies code layout:
     > "matrices.py # Confusion and transition matrix plotters"
   - Dispatch message from parent orchestrator specifies:
     > "Multiclass Confusion Matrix (4x4): Ground truth vs Predicted for EQUIVALENCE, FORWARD_ENTAILMENT, BACKWARD_ENTAILMENT, NEGATIVE_OTHER."
     > "Directional Transition Matrix (4x4): For reciprocal pairs (A -> B, B -> A), compare predicted direct label \hat{y}_{dir} against predicted reverse label \hat{y}_{rev}."
     > "Ideal deterministic involution permutation matrix P: P = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]."
     > "Format exporters: ASCII string, LaTeX tabular format for the ACL paper, and JSON."
     > "Integration with prediction DataFrames and Codabench CSV files."
     > "Unit test plan for tests/test_evaluation.py covering matrices."

2. **Existing Codebase Inventory & Label Encodings**:
   - `src/metrics/metrics.py` lines 38–43:
     ```python
     LABEL_TO_IDX: Dict[str, int] = {
         TaskLabel.BACKWARD_ENTAILMENT.value: 0,
         TaskLabel.EQUIVALENCE.value: 1,
         TaskLabel.FORWARD_ENTAILMENT.value: 2,
         TaskLabel.NEGATIVE_OTHER.value: 3,
     }
     ```
     Under this internal head indexing, $P = [[0,0,1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]]$.
   - In contrast, the natural canonical evaluation order requested for presentation in tables and paper is:
     `0: EQUIVALENCE, 1: FORWARD_ENTAILMENT, 2: BACKWARD_ENTAILMENT, 3: NEGATIVE_OTHER`.
     Under this canonical presentation ordering:
     - Row 0 (EQU) maps to EQU (col 0): `[1, 0, 0, 0]`
     - Row 1 (FWD) maps to BWD (col 2): `[0, 0, 1, 0]`
     - Row 2 (BWD) maps to FWD (col 1): `[0, 1, 0, 0]`
     - Row 3 (NEG) maps to NEG (col 3): `[0, 0, 0, 1]`
     Yielding exactly:
     $$P = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$
     To eliminate all index ambiguity, `src/evaluation/matrices.py` must support explicit string label names and dynamically compute the ideal involution matrix for any ordering via `get_ideal_involution_matrix(labels)`.

3. **Data Ingestion and Codabench Format**:
   - `src/data/codabench.py` line 41 defines `EXPECTED_HEADER = ["instance_id", "label"]`.
   - `src/data/loader.py` line 239 defines `SimpleDataFrame` as a zero-dependency fallback for offline environments where `pandas` is not installed.
   - `src/data/reversal.py` line 421 implements `ReciprocalPair(direct, inverse, pair_id=p_id)`.
   - `data/loader.py` lines 109 and 116 show that dataset records include `pair_id` and optional `reverse_pair_id`.

4. **Environment Execution**:
   - Command `python3 -m unittest discover tests` successfully ran 114 tests in 1.15 seconds (100% pass rate).
   - Python 3.14 without third-party dependencies is installed; all evaluation modules must have zero-dependency fallbacks (pure Python lists and dicts) while optionally integrating with `numpy` and `pandas` when available.

---

## 2. Logic Chain

1. **Decoupling Presentation Order from Model Head Indices**:
   - *Premise*: Different parts of the codebase or external models may order classes differently (e.g. alphabetical `BWD, EQU, FWD, NEG` vs canonical `EQU, FWD, BWD, NEG`).
   - *Deduction*: By accepting string labels everywhere and mapping them dynamically through `CANONICAL_LABELS = ("EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER")`, `matrices.py` prevents hard-coded off-by-one errors.
   - *Validation*: `get_ideal_involution_matrix(CANONICAL_LABELS)` produces $P = [[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]$ with $P^2 = I_4$. Passing BWD-first labels produces $P = [[0,0,1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]]$. Both orders are mathematically exact.

2. **Multiclass Confusion Matrix Architecture**:
   - *Requirements*: Raw counts, row-normalized percentages (Recall), col-normalized percentages (Precision), total-normalized percentages (Joint distribution), per-class TP/FP/FN/TN, Precision, Recall, F1, Specificity, Support, Accuracy, Macro F1, Weighted F1.
   - *Design*: Implement `PerClassMetrics` and `ConfusionMatrixReport`.
   - *Verification*: On a synthetic dataset with known TP/FP/FN, `ConfusionMatrixReport` correctly computed Accuracy ($0.8500$), Weighted F1 (exact parity with `src.metrics.metrics.compute_weighted_f1`), and handled zero support without zero-division errors.

3. **Directional Transition Matrix Architecture**:
   - *Requirements*: Compare predicted direct label $\hat{y}_{dir}$ against predicted reverse label $\hat{y}_{rev}$ for reciprocal pairs $(A \to B, B \to A)$.
   - *Formulations*:
     - Joint Count Matrix: $T_{i, j} = \sum_{p} \mathbb{I}(\hat{y}_{dir}^{(p)}=i, \hat{y}_{rev}^{(p)}=j)$
     - Conditional Transition Probability: $P(\hat{y}_{rev}=j \mid \hat{y}_{dir}=i) = \frac{T_{i, j}}{\sum_k T_{i, k}}$
     - Transition Inconsistency Violation Rate: $\text{Violation Rate} = 1 - \frac{\sum_{i, j} T_{i, j} P_{i, j}}{N_{pairs}}$
     - Frobenius Distance to Ideal Involution $P$: $\|C - P\|_F = \sqrt{\sum_{i,j} (C_{i, j} - P_{i, j})^2}$
     - Normalized Frobenius Distance: $\frac{\|C - P\|_F}{\sqrt{2 \cdot 4}} = \frac{\|C - P\|_F}{\sqrt{8}} \in [0.0, 1.0]$
     - Mean Total Variation Distance: $\frac{1}{4} \sum_i \frac{1}{2} \sum_j |C_{i, j} - P_{i, j}|$
     - Mean Shannon Transition Entropy: $\bar{H} = \frac{1}{4} \sum_i \left(-\sum_{j: C_{ij}>0} C_{ij} \log_2 C_{ij}\right)$ (bits)
     - Reversal Curse Index (RCSI): $\frac{T_{1, 1} + T_{2, 2}}{\sum_k (T_{1, k} + T_{2, k})}$ (entailment loop trap rate)
     - 16-cell Taxonomy: Consistent ($T_{0,0}, T_{1,2}, T_{2,1}, T_{3,3}$), Entailment Loop ($T_{1,1}, T_{2,2}$), Equiv Asymmetry ($T_{0,k}, T_{k,0}$), Noise Collapse ($T_{1,3}, T_{2,3}, T_{3,1}, T_{3,2}$).
   - *Design*: Implement `TransitionInconsistencyMetrics` and `DirectionalTransitionReport`.

4. **Exporter Compliance**:
   - *ASCII Table*: Cleanly aligned text output with legend flags (`*` for ideal involution targets, `!` for Reversal Curse traps) and summary metrics block.
   - *LaTeX Tabular*: SemEval/ACL publication-ready block featuring `booktabs` (`\toprule`, `\midrule`, `\bottomrule`), bolded target cells (`\textbf{...}`), underlined traps (`\underline{...}`), row/column totals, caption, and label.
   - *JSON*: Structured dictionary output with full precision for reproducibility and automated reporting.

5. **Universal Data Integration**:
   - `load_predictions(source)` accepts:
     - Codabench 2-column CSV files (`instance_id,label`) with automatic header detection and comma/tab delimiter sniffing.
     - Pandas `DataFrame` and `SimpleDataFrame` (detecting `instance_id` and `label`/`prediction` columns).
     - Python mappings (`dict`) and lists of tuples.
   - `extract_reciprocal_pairs_from_dataset(dataset)` extracts reciprocal pairs via explicit `reverse_pair_id` pointer pass followed by `pair_id` group pass.
   - `evaluate_matrices_from_data(dataset, predictions)` executes the full analysis pipeline in a single call.

---

## 3. Caveats

1. **Zero-Frequency Rows in Transition Matrix**:
   If a model never predicts a particular class $i$ in the direct direction ($\sum_k T_{i, k} = 0$), the conditional transition probability row is undefined ($0/0$). The implementation safely assigns $0.0$ to all probabilities in that row and computes row metrics (entropy, TVD) only over active non-empty rows.
2. **Negative Other in Reciprocal Pairs**:
   In official DiCo-NLI PhrasIS data, reversible pairs are primarily from `EQUIVALENCE`, `FORWARD_ENTAILMENT`, and `BACKWARD_ENTAILMENT`. However, some pairs are `NEGATIVE_OTHER`. The $4 \times 4$ transition matrix cleanly accommodates all 4 classes, mapping `NEGATIVE_OTHER` $\to$ `NEGATIVE_OTHER` on the involution diagonal ($P_{3,3}=1$).
3. **Pure-Python vs Numpy**:
   All core calculations are implemented using pure-Python standard library structures (`list`, `dict`, `math`). If `numpy` is installed in the runtime environment, helper methods `numpy_raw()` and `numpy_row_norm()` return `np.ndarray` without forcing numpy as a hard dependency.

---

## 4. Conclusion

The architecture, mathematical formulation, exporters, and data integration for `src/evaluation/matrices.py` are fully designed, implemented, and empirically verified.

The complete reference implementation has been placed in:
`/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m4_2/proposed_matrices.py`

The 11-test unit test suite in:
`/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m4_2/test_proposed_matrices.py`
passes with 100% success in 0.006 seconds on CPU.

### Interface Contract for `src/evaluation/matrices.py`
```python
# Constants
CANONICAL_LABELS: Tuple[str, ...]
LABEL_TO_CANONICAL_IDX: Dict[str, int]
CANONICAL_IDX_TO_LABEL: Dict[int, str]
LABEL_SHORT_NAMES: Dict[str, str]
IDEAL_INVOLUTION_MATRIX: List[List[float]]

# Data Classes
class PerClassMetrics: ...
class ConfusionMatrixReport:
    def format_ascii_table(self, show_metrics: bool = True) -> str: ...
    def format_latex_table(self, caption: str = "...", label: str = "...") -> str: ...
    def to_dict(self) -> Dict[str, Any]: ...
    def to_json(self, indent: int = 2) -> str: ...
    def save_json(self, file_path: Union[str, Path]) -> None: ...
    def numpy_raw(self) -> Any: ...
    def numpy_row_norm(self) -> Any: ...

class TransitionInconsistencyMetrics: ...
class DirectionalTransitionReport:
    def format_ascii_table(self, show_conditional: bool = True) -> str: ...
    def format_latex_table(self, caption: str = "...", label: str = "...") -> str: ...
    def to_dict(self) -> Dict[str, Any]: ...
    def to_json(self, indent: int = 2) -> str: ...
    def save_json(self, file_path: Union[str, Path]) -> None: ...
    def numpy_raw(self) -> Any: ...
    def numpy_conditional(self) -> Any: ...

# Core Engine Functions
def get_ideal_involution_matrix(labels: Sequence[str] = CANONICAL_LABELS) -> List[List[float]]: ...
def compute_confusion_matrix(gold, pred, labels=CANONICAL_LABELS) -> ConfusionMatrixReport: ...
def compute_directional_transition_matrix(pairs, predictions, labels=CANONICAL_LABELS) -> DirectionalTransitionReport: ...

# Data Integration Utilities
def load_predictions(source, id_col="instance_id", pred_col="label") -> Dict[str, str]: ...
def extract_reciprocal_pairs_from_dataset(dataset) -> List[Tuple[str, str]]: ...
def evaluate_matrices_from_data(dataset, predictions, labels=CANONICAL_LABELS) -> Tuple[ConfusionMatrixReport, DirectionalTransitionReport]: ...
```

---

## 5. Verification Method

To independently verify the proposed design and reference implementation:

1. **Run Proposed Matrices Unit Test Suite**:
   ```bash
   cd /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m4_2
   python3 test_proposed_matrices.py
   ```
   *Expected Result*: Ran 11 tests in <0.010s, `OK`.

2. **Verify Existing Project Test Suite Remains Untouched**:
   ```bash
   cd /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject
   python3 -m unittest discover tests
   ```
   *Expected Result*: Ran 114 tests, `OK`.

3. **Verify Codabench Submission Loading & End-to-End Pipeline**:
   Execute the integration test `test_data_integration_codabench_csv` and `test_evaluate_matrices_from_data_pipeline` within `test_proposed_matrices.py` to confirm CSV parsing, pair matching, and LaTeX generation.

4. **Invalidation Conditions**:
   - If `P_{1, 2} != 1.0` or `P_{2, 1} != 1.0` under canonical ordering `[EQUIVALENCE, FORWARD_ENTAILMENT, BACKWARD_ENTAILMENT, NEGATIVE_OTHER]`.
   - If `Weighted F1` calculated by `compute_confusion_matrix` deviates from `src.metrics.metrics.compute_weighted_f1`.
   - If `load_predictions` crashes on standard 2-column CSV files generated by `src.data.codabench.export_codabench_submission`.
   - If `format_latex_table` produces invalid LaTeX syntax.
