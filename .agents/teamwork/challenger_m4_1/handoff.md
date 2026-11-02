# Handoff Report — Adversarial Challenge: Milestone 4 Invariants & Stress Testing (challenger_m4_1)

## 1. Observation

1. **Unit Test Suite Execution**:
   - Command: `python3 -m unittest discover -s tests -p "test_*.py" -v`
   - Result:
     ```text
     Ran 150 tests in 3.309s
     OK
     ```
   - Command: `python3 -m unittest tests/test_evaluation.py -v`
   - Result:
     ```text
     Ran 36 tests in 0.248s
     OK
     ```

2. **Empirical Invariant Stress Testing**:
   - **Invariant 1: $0 \le \text{HardCons} \le \text{SoftCons} \le 1.0$**:
     - Tested on $N=50,000$ randomized prediction pairs:
       `HardCons = 0.0617`, `SoftCons = 0.1895`, satisfying $0 \le \text{HardCons} \le \text{SoftCons} \le 1.0$.
     - Tested on adversarial Reversal Curse loop distributions ($N=1,000$ pairs where $\hat{y}_{dir}=\text{FWD}$ and $\hat{y}_{rev}=\text{FWD}$):
       `HardCons = 0.0`, `SoftCons = 0.0`.
     - Tested on adversarial soft-consistent but 100% incorrect pairs ($\hat{y}_{dir}=\text{EQUIV}, \hat{y}_{rev}=\text{EQUIV}$ on gold $\text{FWD}/\text{BWD}$):
       `HardCons = 0.0`, `SoftCons = 1.0`.
     - Tested on adversarial 100% correct pairs:
       `HardCons = 1.0`, `SoftCons = 1.0`.
     - Tested `compute_official_metrics` with zero reversible pairs (`is_reversible_mask = [False]*N`):
       Returns `{"weighted_f1": ..., "soft_cons": 0.0, "hard_cons": 0.0}` defensively without raising exceptions.
     - In zero cases did $\text{HardCons} > \text{SoftCons}$.

   - **Invariant 2: Symmetrized Decoding Guarantees $\text{SoftCons} = 1.0$, Frobenius Distance $P = 0$, Violation Rate $= 0$**:
     - Tested `SymmetrizedDecoder(mask_negative_other=True)` on $N=20,000$ random Dirichlet distributions:
       `SoftCons = 1.000000 (100.0%)`, `consistent_pairs = 20,000/20,000`, `violation_rate = 0.000000`.
     - Tested extreme boundary and tie-breaking probability vectors:
       * Uniform ties: `[0.25, 0.25, 0.25, 0.25] -> EQUIV / EQUIV`
       * Asymmetric ties: `[0.5, 0.0, 0.5, 0.0] -> EQUIV / EQUIV`
       * Extreme logits: `[1e15, 0, 0, 0] vs [0, 0, 1e15, 0] -> BWD / FWD`
       * One-hot spikes across all 16 $(d, r)$ combinations:
         In all cases, $\hat{y}_{rev} = \text{Rev}(\hat{y}_{dir})$ strictly.
     - Directional Transition Matrix Frobenius Distance $\|C - P\|_F$:
       * On full 4-class prediction distributions where all 4 classes are predicted:
         `Frobenius distance = 0.000000`, `violation_rate = 0.000000`, `RCSI = 0.000000`, `mean_tvd = 0.000000`.
       * On 3-class reversible projection (`labels=["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT"]`):
         `Frobenius distance = 0.000000`, `violation_rate = 0.000000`.

   - **Invariant 3: Confusion Matrix Marginals**:
     - Evaluated across 100 randomized trials with variable instance counts ($N \in [1, 500]$) and random class subsets (including zero-count classes):
       * $\sum_j C_{i, j} = \text{support}(y_{true} = i)$ holds with 0 discrepancies.
       * $\sum_i C_{i, j} = \text{support}(y_{pred} = j)$ holds with 0 discrepancies.
       * $\sum_{i, j} C_{i, j} = N = \text{total\_instances}$ holds with 0 discrepancies.
       * Zero-count classes yield `precision = 0.0`, `recall = 0.0`, `f1 = 0.0` via safe division.

   - **Invariant 4: Directional Transition Matrix Marginals**:
     - Evaluated across 100 randomized trials with variable pair counts ($N \in [1, 500]$) and random class subsets:
       * $\sum_j T_{i, j} = \text{count}(\hat{y}_{dir} = i)$ holds with 0 discrepancies.
       * $\sum_i T_{i, j} = \text{count}(\hat{y}_{rev} = j)$ holds with 0 discrepancies.
       * $\sum_{i, j} T_{i, j} = N_{pairs} = \text{total\_pairs}$ holds with 0 discrepancies.

3. **Boundary Conditions & Malformed Inputs Stress Testing**:
   - **Empty Inputs**:
     * `compute_confusion_matrix([], [])` raises `ValueError: Cannot compute confusion matrix on empty sequences.`
     * `compute_directional_transition_matrix([], {})` raises `ValueError: Cannot compute transition matrix on 0 reciprocal pairs.`
     * `compute_classification_report({}, {})` raises `ValueError: Cannot compute classification metrics for zero instances.`
     * `compute_consistency_report([], {})` raises `ValueError: Cannot compute consistency metrics without reversible pairs.`
     * `analyze_syntactic_categories([], {})` handles empty input gracefully, returning `report.total_instances = 0`, `report.total_pairs = 0` without crashing.
     * `build_comparison_table([])` handles empty results gracefully and renders valid Markdown/LaTeX/ASCII tables.
   - **Zero-Count Classes**:
     * When classes have 0 support in gold and 0 predictions, all per-class metrics report 0.0 without `ZeroDivisionError`.
     * `mean_transition_entropy` and `mean_tvd` correctly filter inactive rows (`row_sum > 0`) and report 0.0 without `math.log2(0)` domain errors.
   - **NaN / Whitespace Handling**:
     * `load_predictions_from_csv` and `load_predictions` strip leading/trailing whitespace from IDs and labels (`" 1 ", " EQUIVALENCE "` -> `"1": "EQUIVALENCE"`).
     * Empty CSV files raise `ValueError: Empty prediction CSV`.
   - **Single-Pair Inputs ($N=1$)**:
     * `compute_confusion_matrix`, `compute_directional_transition_matrix`, and `compute_consistency_report` execute accurately for $N=1$.
     * `diagnose_error` correctly diagnoses individual error mechanisms (`REVERSAL_CURSE_LOOP`, `EQUIVALENCE_ASYMMETRY`, `DIRECTIONAL_FLIP`, `NOISE_COLLAPSE`, `SEMANTIC_HALLUCINATION`, `GENERAL_MISCLASSIFICATION`).
   - **Non-Reversible Pairs (`NEGATIVE_OTHER`)**:
     * `src.data.reversal.pair_instances` explicitly flags `pair.is_reversible = False` for `NEGATIVE_OTHER` pairs.
     * `src.evaluation.tables.evaluate_prediction_mapping` filters `rev_pairs = [p for p in pairs if getattr(p, "is_reversible", True)]`, ensuring non-reversible pairs never corrupt `SoftCons` or `HardCons`.
     * `src.evaluation.matrices.extract_reciprocal_pairs_from_dataset` correctly extracts all reciprocal pairs while supporting transition analysis.
   - **Missing Syntactic Categories**:
     * When evaluating datasets lacking one or more syntactic categories (e.g. only `nominal` present, `verbal`/`prepositional`/`adverbial`/`quantificational` absent), `analyze_syntactic_categories` processes existing categories cleanly without KeyError or division errors, generating valid reports.
   - **Unknown Labels / Missing Instance IDs**:
     * Unknown gold label: `ValueError: Unrecognized gold label 'INVALID_LABEL' at index 0.`
     * Unknown predicted label: `ValueError: Unrecognized pred label 'CONTRADICTION' at index 0.`
     * Missing instance ID in classification: `KeyError: "Missing prediction for instance_id: 'inst_2'"`
     * Missing reciprocal instance ID: `KeyError: 'Missing predictions for pair (d1, r1)'`

4. **CLI & GPU Pipeline Quick-Test**:
   - `python3 experiments/evaluate.py --test_run --generate_tables --generate_matrices --generate_analysis`:
     Executed in 0.091s (< 2.0s budget), correctly generated all artifacts with `SoftCons = 1.0000` on `sd` and `crl_sd`.
   - `bash scripts/run_gpu_training.sh --quick-test`:
     Executed 10/10 mock training runs in 2.0s with zero regressions and valid Codabench prediction files.

---

## 2. Logic Chain

1. **Mathematical Invariant Consistency**:
   - *Premise*: By definition, a pair is hard-correct only if both directions match ground truth, and if ground truth is reversible, correct predictions are necessarily soft-consistent ($(\hat{y}_{dir} = y_{dir}^* \land \hat{y}_{rev} = y_{rev}^*) \implies \hat{y}_{rev} = \text{Rev}(\hat{y}_{dir})$).
   - *Observation*: Across 50,000 randomized and adversarial pairs, $\text{HardCons} \le \text{SoftCons}$ held at every evaluation step.
   - *Deduction*: The invariant $0 \le \text{HardCons} \le \text{SoftCons} \le 1.0$ is mathematically sound and strictly enforced by the codebase.

2. **Symmetrized Decoding Invariant Guarantee**:
   - *Premise*: Symmetrized Decoding computes $p_{sym,dir} = \frac{1}{2}(p_{dir} + p_{rev} P)$ and $p_{sym,rev} = \frac{1}{2}(p_{rev} + p_{dir} P)$. Because $P = P^{-1} = P^T$, $p_{sym,rev} = p_{sym,dir} P$. With an involutory tie-breaking rule, $\operatorname{argmax}(p_{sym,rev}) = \operatorname{Rev}(\operatorname{argmax}(p_{sym,dir}))$.
   - *Observation*: 20,000 Dirichlet-distributed probability vectors and 16 extreme boundary/tie configurations yielded 100% SoftCons, 0.0 violation rate, and 0.0 RCSI.
   - *Deduction*: Symmetrized Decoding guarantees directional consistency under all probability regimes.

3. **Matrix Marginal Invariants**:
   - *Premise*: Confusion matrix row sums represent marginal distribution of true labels ($\sum_j C_{i, j} = N_i^{true}$); column sums represent marginal distribution of predicted labels ($\sum_i C_{i, j} = N_j^{pred}$). Transition matrix row sums represent marginal distribution of direct predictions ($\sum_j T_{i, j} = N_i^{dir}$).
   - *Observation*: All 100 random multi-class trials verified row sums and column sums identically match class counts.
   - *Deduction*: Both `compute_confusion_matrix` and `compute_directional_transition_matrix` are numerically exact.

4. **Input Defense and Error Handling**:
   - *Premise*: Real-world evaluation inputs may contain malformed data, empty datasets, missing categories, or unmapped classes.
   - *Observation*: Empty inputs, unknown labels, and missing IDs are trapped immediately with explicit `ValueError` and `KeyError` messages. Missing syntactic categories, zero-count classes, and non-reversible pairs degrade gracefully without crashing.
   - *Deduction*: The evaluation subsystem is resilient to adversarial edge cases.

---

## 3. Caveats

- **Frobenius Norm on Inactive Classes**: In an $n \times n$ transition matrix, if a class is never predicted in the direct direction ($N_i^{dir} = 0$), its conditional row in $C$ is $\mathbf{0}$. Because $P$ has row norm 1, that inactive row contributes $1.0$ to $\|C - P\|_F^2$. On reversible-only predictions where `NEGATIVE_OTHER` is masked out, the full 4x4 matrix exhibits $\|C - P\|_F = 1.0$ due to the inactive fourth row, whereas the 3x3 reversible matrix exhibits $\|C - P\|_F = 0.0$. This is mathematically correct according to the standard matrix norm definition and does not affect the violation rate or SoftCons metrics (which remain 0.0 and 1.0 respectively).

---

## 4. Conclusion

### EMPIRICAL VERDICT: `APPROVE`

Milestone 4 (Evaluation & Analysis) satisfies all mathematical invariants, boundary conditions, edge case requirements, and test suites:
- Invariant $0 \le \text{HardCons} \le \text{SoftCons} \le 1.0$ verified empirically.
- Symmetrized Decoding guarantees $\text{SoftCons} = 1.0000$ and zero violation rate across all reversible pairs.
- Confusion matrix and directional transition matrix marginal invariants hold with zero discrepancy.
- Comprehensive boundary condition resilience confirmed across empty inputs, zero-count classes, single pairs, non-reversible pairs, missing syntactic categories, and unknown labels.
- Full unit test suite passes: 150/150 tests in 3.3s.
- Subsystem test suite passes: 36/36 tests in 0.25s.
- `evaluate.py --test_run` completes in 0.091s (< 2.0s budget) with 106 verified artifacts.
- End-to-end `run_gpu_training.sh --quick-test` passes 10/10 runs in 2.0s with valid Codabench submissions.

---

## 5. Verification Method

To independently reproduce and verify this assessment:

1. **Run Full Test Suite**:
   ```bash
   python3 -m unittest discover -s tests -p "test_*.py" -v
   ```
   *Expected*: `Ran 150 tests in ~3.3s ... OK`.

2. **Run Evaluation Subsystem Tests**:
   ```bash
   python3 -m unittest tests/test_evaluation.py -v
   ```
   *Expected*: `Ran 36 tests in ~0.25s ... OK`.

3. **Run Evaluate CLI in Fast Test Mode**:
   ```bash
   python3 experiments/evaluate.py --test_run --generate_tables --generate_matrices --generate_analysis --output_dir results/test_run_artifacts
   ```
   *Expected*: Completes in $<0.2$s, exports 106 artifacts, reports SoftCons = 1.0000 for `sd` and `crl_sd`.

4. **Run GPU Training Quick-Test**:
   ```bash
   bash scripts/run_gpu_training.sh --quick-test
   ```
   *Expected*: Completes 10/10 runs in ~2.0s with zero failures.

5. **Invalidation Conditions**:
   - Any test failure in `tests/test_evaluation.py` or `tests/test_metrics.py`.
   - $\text{HardCons} > \text{SoftCons}$ on any prediction set.
   - Symmetrized Decoding yielding $\text{SoftCons} < 1.0$ on reversible pairs.
   - Unhandled exception when encountering empty datasets or zero-count classes.
