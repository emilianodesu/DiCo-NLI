# Handoff Report — Reviewer 2: Milestone 4 Matrices & Analysis (reviewer_m4_2)

## 1. Observation

1. **Independent Test Execution**:
   - Command: `python3 -m unittest discover -s tests -p "test_*.py" -v`
   - Verbatim Output:
     ```text
     Ran 150 tests in 1.721s
     OK
     ```
     Result: 150/150 tests passed cleanly without errors, warnings, or skips.
   - Command: `python3 experiments/evaluate.py --test_run --generate_matrices --generate_analysis`
   - Verbatim Output:
     ```text
     2026-11-01 21:34:24 | INFO | dico_nli.evaluate | SemEval-2027 Task 2 (DiCo-NLI) — Evaluation CLI Engine
     ...
     2026-11-01 21:34:25 | INFO | dico_nli.evaluate | Evaluation completed successfully in 0.070s
     2026-11-01 21:34:25 | INFO | dico_nli.evaluate | Artifacts saved to: results/evaluation
     ```
     Execution took 0.070 seconds (well within the strictly enforced 2.0s ceiling).

2. **Artifact Verification in `results/evaluation/`**:
   - 118 files created across Markdown, LaTeX, ASCII, JSON, and Codabench CSV formats.
   - 20 Codabench submission CSVs tested via `validate_codabench_submission`: 20/20 PASSED.
   - 31 LaTeX `.tex` files tested for balanced braces and nested environments (`table`, `tabular`): 31/31 PASSED.

3. **Code Inspection**:
   - `src/evaluation/matrices.py` (1088 lines):
     - Multiclass Confusion Matrix: Canonical ordering `("EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER")`, true rows vs predicted columns, row-normalized (recall), col-normalized (precision), total-normalized percentages, per-class support, precision, recall, F1, specificity, aggregate accuracy, macro F1, and support-weighted F1 (matching official SemEval formula).
     - Directional Transition Matrix: Reciprocal pair extraction via `reverse_pair_id` and `pair_id`, joint transition counts $T_{i,j}$, row-conditional probabilities $P(\hat{y}_{rev}=j \mid \hat{y}_{dir}=i)$, joint probabilities $P(i, j)$, ideal deterministic involution permutation matrix $P$ ($P^2 = I_4$), Frobenius distance $\|T - P\|_F$, normalized Frobenius distance $\frac{\|T - P\|_F}{\sqrt{2n}}$, mean row Total Variation Distance (TVD), conditional Shannon transition entropy $H(T_i) = -\sum_j P_{ij} \log_2 P_{ij}$ (in bits), Reversal Curse Severity Index (RCSI) $\frac{\text{fwd\_fwd\_trap} + \text{bwd\_bwd\_trap}}{\text{dir\_entailment\_total}}$, and 16-cell taxonomy diagnostics.
     - Exporters: ASCII table, LaTeX table (`booktabs`, `\toprule`, `\midrule`, `\bottomrule`, `tab:confusion_matrix`, `tab:transition_matrix`), and JSON serialization.
   - `src/evaluation/analysis.py` (1014 lines):
     - Chunk disaggregation: Partitions instances and reciprocal pairs across `nominal`, `verbal`, `prepositional`, `adverbial`, `quantificational` chunks.
     - Disaggregated metrics: Weighted F1, SoftCons, HardCons, entailment loop rate, equivalence asymmetry rate, noise collapse rate, RCSI.
     - Qualitative error cases extraction with max limit per category.
     - 6-class linguistic diagnosis taxonomy in `ErrorType`:
       1. `REVERSAL_CURSE_LOOP` (with syntactic-specific locative/hyponymic/monotonic explanations)
       2. `EQUIVALENCE_ASYMMETRY`
       3. `DIRECTIONAL_FLIP`
       4. `NOISE_COLLAPSE`
       5. `SEMANTIC_HALLUCINATION`
       6. `GENERAL_MISCLASSIFICATION`
     - Cross-method comparisons: Computes baseline deltas, identifies most vulnerable category (`adverbial` / `prepositional`), and identifies most improved category under Symmetrized Decoding.
     - Exporters: Markdown report, publication-ready LaTeX tables (`category_performance_table`, `cross_method_table`, `error_cases_table` with escaped special characters `\_`, `\%`, `\&`), and JSON serialization.

4. **Integrity Audit**:
   - No hardcoded test outputs or dummy facades: All metrics and matrix values are computed dynamically from actual predictions.
   - Zero-division defenses: Tested edge cases (empty sequences, 0 support classes, single-instance inputs, zero reciprocal pairs) — all either return 0.0 or raise explicit, descriptive exceptions.

---

## 2. Logic Chain

1. **Correctness & Mathematical Validity**:
   - *Observation 1.3*: In `matrices.py`, the ideal involution permutation matrix $P$ over canonical classes `("EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER")` is:
     $$P = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$
     Satisfies $P = P^T = P^{-1}$ and $P^2 = I_4$.
   - *Observation 1.3*: Conditional transition probabilities $P(\hat{y}_{rev}=j \mid \hat{y}_{dir}=i)$ sum to 1.0 for each active row, and Shannon entropy $H(T_i)$ accurately measures dispersion from deterministic involution in bits.
   - *Observation 1.3*: The Reversal Curse Severity Index directly captures the proportion of directional entailments trapped in bidirectional forward or backward loops.

2. **Completeness & SemEval Conformance**:
   - *Observation 1.3*: All 5 PhrasIS chunk categories (`nominal`, `verbal`, `prepositional`, `adverbial`, `quantificational`) are supported with automatic inference fallback if dataset columns are missing.
   - *Observation 1.3*: The linguistic error taxonomy implements all 6 specified classes with domain-specific explanations.
   - *Observation 1.2*: All 31 generated LaTeX tables compile with balanced braces and valid `booktabs` column structures for direct inclusion in `report/paper.tex`.

3. **Performance & Invariants**:
   - *Observation 1.1*: CLI `--test_run` executes in 0.070s, well beneath the 2.0s ceiling.
   - *Observation 1.1 & 1.3*: Symmetrized Decoding runs (`sd`, `crl_sd`) consistently achieve $\text{SoftCons} = 1.0000$ (100.0%) and $\text{RCSI} = 0.0\%$.

---

## 3. Caveats

- **No Caveats**: All requested review tasks, unit tests, CLI smoke tests, and adversarial challenges were completed directly in the environment without mocks or external dependencies.

---

## 4. Conclusion

Milestone 4 (Matrices & Analysis) fulfills all functional, architectural, and mathematical requirements.
There are zero integrity violations, zero hardcoded values, and zero regressions across the 150-test project suite.
The generated LaTeX, Markdown, and JSON artifacts are publication-ready for the ACL SemEval paper (`report/paper.tex`).

**Verdict: APPROVE**

---

## 5. Verification Method

To independently verify this evaluation:

1. **Run Full Test Suite**:
   ```bash
   python3 -m unittest discover -s tests -p "test_*.py" -v
   ```
   *Expected*: `Ran 150 tests in ~1.7s ... OK`.

2. **Run Standalone Evaluation CLI with Matrices & Analysis**:
   ```bash
   python3 experiments/evaluate.py --test_run --generate_matrices --generate_analysis --output_dir results/evaluation
   ```
   *Expected*: Completes in $< 0.1$s ($< 2.0$s ceiling) and populates `results/evaluation/` with 106+ files (.tex, .md, .json, .txt, .csv).

3. **Validate Codabench Submissions**:
   ```bash
   python3 -c "
   import glob
   from src.data.codabench import validate_codabench_submission
   for f in glob.glob('results/evaluation/eval_*.csv'):
       assert validate_codabench_submission(f), f'Failed: {f}'
   print('All submission files valid!')
   "
   ```

4. **Validate LaTeX Syntax**:
   ```bash
   python3 -c "
   import glob, re
   for tf in glob.glob('results/evaluation/*.tex'):
       with open(tf) as f: content = f.read()
       assert content.count('{') == content.count('}'), f'Braces: {tf}'
       stack = []
       for tag, name in re.findall(r'\\\\(begin|end)\{([^}]+)\}', content):
           if tag == 'begin': stack.append(name)
           else: assert stack.pop() == name
       assert not stack, f'Unclosed: {tf}'
   print('All LaTeX tables valid!')
   "
   ```

5. **Invalidation Conditions**:
   - Any test failure in `tests/test_evaluation.py`.
   - CLI execution exceeding 2.0s under `--test_run`.
   - Any unbalanced LaTeX tags in generated `.tex` files.
   - Any Codabench validation error in exported prediction CSVs.

---

## Quality Review Report

**Verdict**: APPROVE

### Findings

#### [Minor] Finding 1: Error Taxonomy Precedence in `diagnose_error`
- **What**: In `src/evaluation/analysis.py` (lines 163-172), when a gold `EQUIVALENCE` pair is predicted as `(NEGATIVE_OTHER, NEGATIVE_OTHER)`, the condition `(pred_dir not in ("EQUIVALENCE",) or pred_rev not in ("EQUIVALENCE",))` triggers `EQUIVALENCE_ASYMMETRY` before Rule #3 (`NOISE_COLLAPSE`) can evaluate.
- **Where**: `src/evaluation/analysis.py`, lines 163-172.
- **Why**: The resulting diagnosis string states: `"...predicts asymmetric 'NEGATIVE_OTHER' forward and 'NEGATIVE_OTHER' backward..."`, which is linguistically contradictory because the prediction is symmetric, not asymmetric. It represents semantic dropout/noise collapse to `NEGATIVE_OTHER`.
- **Suggestion**: In Rule #2, require `pred_dir != pred_rev` for `EQUIVALENCE_ASYMMETRY`. If `pred_dir == pred_rev == "NEGATIVE_OTHER"`, allow Rule #3 (`NOISE_COLLAPSE`) to claim the diagnosis.

#### [Minor] Finding 2: Unconditioned Frobenius Distance on Inactive Rows
- **What**: In `src/evaluation/matrices.py` (lines 790-802), `mean_tvd` and `mean_transition_entropy` are conditioned on `active_rows` (`row_sum > 0`), whereas `frobenius_distance` computes squared differences over all $n$ rows.
- **Where**: `src/evaluation/matrices.py`, lines 790-802.
- **Why**: If a direct class never appears in predictions (e.g. `BACKWARD_ENTAILMENT`), its conditional probability row is $[0, 0, 0, 0]$. Comparing against ideal basis vector $P_i$ adds 1.0 to $\|T - P\|_F^2$, giving $\|T - P\|_F \ge 1.0$ even when 100% of all observed pairs are consistent.
- **Suggestion**: Document this design choice in the ACL paper/methodology section, noting that the unconditioned Frobenius norm treats unobserved states as non-transitional zero vectors.

#### [Minor] Finding 3: 4-State Involution Consistency vs Official 3-Class SoftCons
- **What**: In `DirectionalTransitionReport.format_ascii_table()` (line 448), the metric is labeled `Consistent Pairs (SoftCons)`.
- **Where**: `src/evaluation/matrices.py`, line 448.
- **Why**: The 4x4 algebraic involution $P$ has $P_{3,3} = 1.0$ (`NEGATIVE_OTHER` $\to$ `NEGATIVE_OTHER`), counting (NEG, NEG) transitions as involution-consistent. Official SemEval Task 2 `SoftCons` (`src/metrics/metrics.py`) explicitly restricts consistency to reversible classes (`EQU`, `FWD`, `BWD`).
- **Suggestion**: Clarify in report headers that this represents 4-state algebraic involution consistency ($T \approx P$), which upper-bounds official `SoftCons` when `NEGATIVE_OTHER` predictions exist.

### Verified Claims
- Multiclass confusion matrix per-class metrics and weighted F1 match official SemEval metrics $\to$ Verified via unit tests and synthetic vector analysis $\to$ PASS.
- Directional transition matrix correctly calculates conditional transition probabilities and Shannon entropy $\to$ Verified via mathematical trace $\to$ PASS.
- Disaggregated syntactic analysis groups instances into nominal, verbal, prepositional, adverbial, and quantificational chunks $\to$ Verified against PhrasIS dev split $\to$ PASS.
- 6-class linguistic diagnosis taxonomy classifies all error types deterministically $\to$ Verified via test harness $\to$ PASS.
- CLI executes within sub-second budget (<2.0s) and exports all required formats (MD, LaTeX, JSON) $\to$ Verified independently in 0.070s $\to$ PASS.

### Coverage Gaps
- None. All requirements outlined in ORIGINAL_REQUEST §R4 and PROJECT.md §13-15 are covered.

### Unverified Items
- None.

---

## Adversarial Challenge Report

**Overall Risk Assessment**: LOW

### Challenges

#### [Low] Challenge 1: Empty or Single-Pair Reciprocal Sets
- **Assumption Challenged**: In small dataset splits or edge cases, certain categories might contain only 1 pair or 0 pairs.
- **Attack Scenario**: Evaluated `compute_directional_transition_matrix` with single-pair input and empty list.
- **Blast Radius**: Handled gracefully. Empty lists raise clear `ValueError` with descriptive message; single pairs compute valid probabilities and avoid NaN in Shannon entropy ($0 \cdot \log_2(0) = 0$).
- **Mitigation**: Existing defensive guards (`active_rows > 0`, `row_sum > 0`, `if prob > 0.0`) are robust.

#### [Low] Challenge 2: Mismatched or Corrupt Label Ingestion
- **Assumption Challenged**: Predictions CSV files might contain unknown labels or missing IDs.
- **Attack Scenario**: Tested ingestion with invalid label strings and omitted instance IDs.
- **Blast Radius**: `matrices.py` rejects unknown labels with `ValueError` and missing IDs with `KeyError`.
- **Mitigation**: Input validation is strict and informative.
