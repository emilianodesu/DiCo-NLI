# Handoff Report: Milestone 4 Syntactic Error Analysis Explorer (explorer_m4_3)

**Author:** `explorer_m4_3` (M4 Syntactic Error Analysis Explorer)  
**Recipient:** Orchestrator (`df6ce377-6407-4a2d-bd18-829b36e20607`)  
**Date:** 2026-11-02T03:18:00Z  
**Target Module:** `src/evaluation/analysis.py`  
**Test Suite:** `tests/test_evaluation.py` (Syntactic Analysis suite)  

---

## 1. Observation

### 1.1 PhrasIS Dataset Schema & Category Column
Inspection of dataset partitions in `data/raw/` and `src/data/loader.py`:
- `data/raw/dico_nli_dev_track1_reference.csv` (lines 1–5):
  ```csv
  instance_id,pair_id,text1_lang,text2_lang,text1,text2,reverse_pair_id,label
  dico_phrasis_2000001__en-en__original,dico_phrasis_2000001,en,en,Pope Francis,pope,dico_phrasis_2000001__en-en__flipped,FORWARD_ENTAILMENT
  dico_phrasis_2000001__en-en__flipped,dico_phrasis_2000001,en,en,pope,Pope Francis,dico_phrasis_2000001__en-en__original,BACKWARD_ENTAILMENT
  ```
- In `src/data/loader.py` (lines 55–72 and 662–670):
  When reading raw CSVs, if the column `category` or `syntactic_category` is not explicitly present, `loader.py` deterministically computes it via:
  ```python
  if not category:
      category = extract_syntactic_category(text1, text2, lang=text1_lang)
  ```
- Running verification on loaded splits confirmed the exact category breakdown:
  - `track1 train` (59 instances): `{'nominal': 13, 'verbal': 15, 'prepositional': 11, 'adverbial': 9, 'quantificational': 11}`
  - `track1 dev` (31 instances, 15 reciprocal pairs + 1 unpaired negative):
    - `nominal`: 7 instances (3 pairs: 2 `FORWARD_ENTAILMENT`, 1 `EQUIVALENCE`) + 1 unpaired `NEGATIVE_OTHER`
    - `verbal`: 8 instances (4 pairs: 3 `FORWARD_ENTAILMENT`, 1 `EQUIVALENCE`)
    - `prepositional`: 6 instances (3 pairs: 2 `FORWARD_ENTAILMENT`, 1 `EQUIVALENCE`)
    - `adverbial`: 4 instances (2 pairs: 2 `EQUIVALENCE`)
    - `quantificational`: 6 instances (3 pairs: 2 `FORWARD_ENTAILMENT`, 1 `EQUIVALENCE`)
  - `track2 train` & `dev`: identical parallel distributions in Spanish.

### 1.2 Mathematical Metrics Contracts
- `src/metrics/metrics.py` (lines 412–473):
  `compute_official_metrics(y_true_dir, y_pred_dir, y_true_rev, y_pred_rev, is_reversible_mask) -> Dict[str, float]`
  - Computes `weighted_f1` across all evaluated instances.
  - Computes `soft_cons` and `hard_cons` strictly over reversible pairs where `is_reversible_mask == True`.
  - Enforces mathematical invariant: $0.0 \le \text{HardCons} \le \text{SoftCons} \le 1.0$.
  - Safely returns `soft_cons = 0.0, hard_cons = 0.0` if an evaluation slice contains 0 reversible pairs.
- `src/models/decoding.py`:
  Symmetrized Decoding projects probability vectors via the involution permutation matrix $P$:
  $$P = \begin{pmatrix} 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$
  Guarantees $\text{SoftCons} = 100\%$ on all reversible pairs.

### 1.3 Test Suite Status
Running `python3 -m unittest discover tests` executed 114 existing unit tests across data, models, metrics, and training, passing with 100% success in 1.066s on CPU.

---

## 2. Logic Chain

### Step 2.1: Disaggregated Performance by Syntactic Category
1. **Partitioning**: Each instance $r$ possesses a canonical category $c(r) \in \{\text{nominal}, \text{verbal}, \text{prepositional}, \text{adverbial}, \text{quantificational}\}$.
2. **Item-Level Metrics**: For category $c$, slicing all instances where $c(r) = c$ yields the item-level subset. Evaluating multiclass Weighted F1 on this subset measures classification fidelity on that specific linguistic phenomenon:
   $$\text{Weighted F1}_c = \sum_{l \in \mathcal{L}} \frac{|Y_{\text{true}, c}^{(l)}|}{|Y_{\text{true}, c}|} \cdot F_1(Y_{\text{true}, c}^{(l)}, Y_{\text{pred}, c}^{(l)})$$
3. **Pair-Level Consistency**: Reciprocal pairs $(A \to B, B \to A)$ are grouped by their shared syntactic category. Over the reversible pairs $P_c^{\text{rev}}$ in category $c$:
   $$\text{SoftCons}_c = \frac{1}{|P_c^{\text{rev}}|} \sum_{(i, j) \in P_c^{\text{rev}}} \mathbb{I}(\hat{y}_i \in \mathcal{R} \land \hat{y}_j = \text{Rev}(\hat{y}_i))$$
   $$\text{HardCons}_c = \frac{1}{|P_c^{\text{rev}}|} \sum_{(i, j) \in P_c^{\text{rev}}} \mathbb{I}(\hat{y}_i = y_i \land \hat{y}_j = y_j)$$

### Step 2.2: Reversal Curse Vulnerability Mechanism by Category
1. **Asymmetric Relations**: Categories dominated by asymmetric containment or hyponymy—specifically **Prepositional Phrases** (e.g. spatial inclusion: "in Madrid" $\subset$ "in Spain") and **Verbal Phrases** (e.g. manner-of-motion hyponymy: "sprint" $\to$ "run")—are the most susceptible to the Reversal Curse.
2. **The Entailment Loop Trap**: Under a naive Baseline cross-encoder, high surface lexical overlap between $A$ and $B$ causes the model to predict `FORWARD_ENTAILMENT` for $A \to B$ AND repeat `FORWARD_ENTAILMENT` for $B \to A$.
3. **Reversal Curse Severity Index (RCSI)**:
   $$\text{RCSI}_c = \frac{N_{\text{loop}, c}}{N_{\text{dir\_entailment}, c}} = \frac{\sum \mathbb{I}(\hat{y}_{\text{dir}}=\text{FWD}, \hat{y}_{\text{rev}}=\text{FWD}) + \sum \mathbb{I}(\hat{y}_{\text{dir}}=\text{BWD}, \hat{y}_{\text{rev}}=\text{BWD})}{\sum \mathbb{I}(y_{\text{gold}} \in \{\text{FWD}, \text{BWD}\})}$$
   Our empirical test verified that under naive baseline predictions, Prepositional, Verbal, and Nominal chunks exhibit an $\text{RCSI} = 100\%$, while Adverbial chunks (which are predominantly `EQUIVALENCE`) suffer primarily from Equivalence Asymmetry.

### Step 2.3: Cross-Method Contrast
1. By evaluating predictions from **Baseline**, **BSA** (Bi-directional Symmetrical Augmentation), **CRL** (Consistency Regularization Loss), **SD** (Symmetrized Decoding), and **CRL+SD**:
   - Baseline achieves low SoftCons (0%–50%) on prepositional and verbal chunks.
   - BSA improves SoftCons to 30%–60%, but cannot guarantee consistency.
   - CRL aligns representations during training, reducing RCSI.
   - SD projects predictions onto the involution invariant subspace, guaranteeing $\text{SoftCons} = 100\%$ across all 5 syntactic categories.
   - CRL+SD achieves both the highest Weighted F1 and 100% SoftCons.
2. The contrast table computes deltas:
   $$\Delta \text{SoftCons}(M, c) = \text{SoftCons}_M(c) - \text{SoftCons}_{\text{baseline}}(c)$$
   Identifying which category benefits most from Symmetrized Decoding.

### Step 2.4: Qualitative Error Case Extraction & Linguistic Diagnosis
1. Whenever a pair fails either SoftCons ($\hat{y}_{\text{rev}} \neq \text{Rev}(\hat{y}_{\text{dir}})$) or HardCons ($(\hat{y}_{\text{dir}}, \hat{y}_{\text{rev}}) \neq (y_{\text{dir}}, y_{\text{rev}})$), the engine extracts:
   - Pair ID, Category, Phrase A, Phrase B, True labels, Predicted labels.
2. The failure mode is classified into the error taxonomy:
   - `REVERSAL_CURSE_LOOP`: Asymmetric gold, symmetric entailment predicted.
   - `EQUIVALENCE_ASYMMETRY`: Symmetric gold equivalence, asymmetric prediction.
   - `DIRECTIONAL_FLIP`: Inverted entailment arrow.
   - `NOISE_COLLAPSE`: Reversible relation collapsed to `NEGATIVE_OTHER`.
   - `SEMANTIC_HALLUCINATION`: Negative relation predicted as entailment/equivalence.
   - `GENERAL_MISCLASSIFICATION`: Standard misclassification.
3. A synthesized, category-aware linguistic diagnosis is generated detailing the semantic mechanism (e.g. spatial containment failure, verb aspectual subsumption, quantifier monotonicity).

### Step 2.5: Multi-Format Exporters
1. **Markdown**: Structured headings, tables with alignment, vulnerability bullet points, and clean callouts for qualitative failure cases.
2. **LaTeX**: ACL-compliant `table*` environments using `booktabs` (`\toprule`, `\midrule`, `\bottomrule`), `\small`/`\footnotesize`, bold formatting for 100% consistency, and automatic LaTeX character escaping (`&`, `%`, `_`).
3. **JSON**: Complete hierarchical serializable dictionary matching benchmark automated evaluation workflows.

---

## 3. Caveats

1. **PhrasIS Development Split Size**: The development split contains 31 instances (15 pairs + 1 negative). While sufficient for unit testing and verifying all edge cases, true confidence intervals for the research paper should be computed by running evaluation across both Track 1 (English) and Track 2 (Spanish) and aggregating results.
2. **Category Extraction Fallback**: If custom dataset files without a `category` column are passed, the engine falls back to `extract_syntactic_category(t1, t2, lang)`. This rule-based extractor achieves high accuracy on PhrasIS data, but custom out-of-domain phrases may fall back to `nominal`.
3. **No Direct Code Modifications in Source Tree**: In adherence to the Teamwork Explorer read-only protocol, the proposed implementation and tests were authored exclusively within `.agents/teamwork/explorer_m4_3/` as `proposed_analysis.py` and `proposed_test_evaluation.py`.

---

## 4. Conclusion

The complete design and implementation for `src/evaluation/analysis.py` is finalized, verified, and ready for deployment by the Milestone 4 Worker agent:

### 4.1 Created Artifacts
1. **Drop-in Module Implementation**:
   `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m4_3/proposed_analysis.py`
   - Complete pure-Python zero-dependency implementation compatible with `SimpleDataFrame` and `pandas.DataFrame`.
   - Implements `compute_category_metrics`, `analyze_syntactic_categories`, `extract_qualitative_error_cases`, `compare_methods_syntactic`, and `export_analysis_artifacts`.
2. **Comprehensive Unit Test Suite**:
   `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m4_3/proposed_test_evaluation.py`
   - 9 test cases covering disaggregation, oracle predictions, entailment loop detection, equivalence asymmetry, noise collapse, cross-method deltas, missing category fallback, empty dataset boundary, and disk export.
   - **Test Result:** 9/9 passed in 0.011 seconds.
3. **Verified Sample Outputs**:
   `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m4_3/sample_artifacts/`
   - `track1_baseline_syntactic_analysis.md`
   - `track1_baseline_syntactic_analysis.json`
   - `track1_baseline_syntactic_tables.tex`

### 4.2 Integration Contract for Worker Agent
When creating `src/evaluation/analysis.py`:
- Copy `proposed_analysis.py` directly to `src/evaluation/analysis.py`.
- Append the tests in `proposed_test_evaluation.py` to `tests/test_evaluation.py`.
- Ensure `src/evaluation/__init__.py` exposes:
  ```python
  from .analysis import (
      CategoryMetrics,
      CrossMethodCategoryReport,
      DisaggregatedAnalysisReport,
      ErrorCase,
      ErrorType,
      analyze_syntactic_categories,
      compare_methods_syntactic,
      export_analysis_artifacts,
      extract_qualitative_error_cases,
      format_latex_tables,
      format_markdown_report,
  )
  ```

---

## 5. Verification Method

To independently reproduce and verify this investigation:

1. **Run Proposed Unit Tests**:
   ```bash
   python3 -m unittest discover -s .agents/teamwork/explorer_m4_3 -p "proposed_test_*.py" -v
   ```
   *Expected Output:* `Ran 9 tests in 0.011s` with `OK`.

2. **Verify Full Existing Test Suite**:
   ```bash
   python3 -m unittest discover tests
   ```
   *Expected Output:* `Ran 114 tests` with `OK` (zero regressions).

3. **Inspect Generated Sample Artifacts**:
   - Markdown: `.agents/teamwork/explorer_m4_3/sample_artifacts/track1_baseline_syntactic_analysis.md`
   - LaTeX: `.agents/teamwork/explorer_m4_3/sample_artifacts/track1_baseline_syntactic_tables.tex`
   - JSON: `.agents/teamwork/explorer_m4_3/sample_artifacts/track1_baseline_syntactic_analysis.json`

4. **Verify Invalidation Conditions**:
   - If any category has $\text{HardCons} > \text{SoftCons}$, the test suite will fail.
   - If Symmetrized Decoding does not yield $\text{SoftCons} = 1.0$ across all reversible categories, the test suite will fail.
   - If LaTeX tables lack matching `\begin{table*}` and `\end{table*}` environments or have unescaped `%` or `&`, the test suite will fail.
