# Handoff Report — explorer_m4_1: M4 Comparison Tables & Evaluation CLI

## 1. Observation

### 1.1 Existing Codebase & Environment Inspection
1. **Repository Layout and Baseline State**:
   - `src/evaluation/` directory did not exist prior to Milestone 4.
   - `experiments/evaluate.py` did not exist; only `experiments/train.py` was present.
   - `tests/test_evaluation.py` did not exist.
   - Existing test suite (`tests/test_*.py`) executes via `python3 -m unittest discover tests` with 114 tests passing in 1.003s.
2. **Existing Run Artifacts**:
   - `results/` contains 42 files from prior Milestone 3 runs, including both live training artifacts (`t1_baseline_deberta_submission.csv`, `t1_sd_deberta_submission.csv`, `t2_baseline_beto_submission.csv`, `t2_sd_beto_submission.csv`) and mock dev runs (`test_track1_baseline_mock_*_submission.csv`).
   - `logs/` contains 126 files, including individual summary JSONs (`logs/*_summary.json`) storing metrics dictionaries:
     ```json
     "metrics": {
       "weighted_f1": 0.5333333333333333,
       "soft_cons": 1.0,
       "hard_cons": 0.5333333333333333
     }
     ```
3. **Data Schema & Reciprocal Pairing Findings**:
   - `src/data/loader.py:671-683`: `InstanceRecord` holds `['instance_id', 'pair_id', 'text1_lang', 'text2_lang', 'text1', 'text2', 'label', 'category', 'reverse_pair_id']`.
   - `src/data/reversal.py:286-318`: `ReciprocalPair` enforces that `NEGATIVE_OTHER` is strictly non-reversible (`is_reversible = False`).
   - `src/data/reversal.py:344-450`: `pair_instances(records)` extracts reciprocal pairs $(A, B)$ and $(B, A)$. Non-reversible `NEGATIVE_OTHER` instances without reciprocal counterparts are NOT paired.
   - **Crucial Edge Case Observed**: In `compute_classification_report(gold_by_id, predicted_labels)` (`src/metrics/metrics.py:275`), an exception `KeyError: "Missing prediction for instance_id: '...'"` is raised if any instance in `gold_by_id` lacks a prediction. Therefore, dynamic evaluation in `experiments/evaluate.py` must iterate over all input records and assign predictions to unpaired `NEGATIVE_OTHER` instances, in addition to evaluating paired reciprocal instances.
4. **Official Metric Invariants**:
   - `src/metrics/metrics.py:398-401`: Enforces $0.0 \le \text{HardCons} \le \text{SoftCons} \le 1.0$.
   - `src/models/decoding.py`: Symmetrized Decoding mathematically and empirically guarantees $\text{SoftCons} = 1.0000$ (100%) on all reversible pairs.
5. **Sibling Explorers Division of Labor**:
   - `explorer_m4_1` (This agent): Comparison tables (`src/evaluation/tables.py`), standalone CLI (`experiments/evaluate.py`), aggregation across models/methods/tracks, and unit test plan.
   - `explorer_m4_2`: Multiclass confusion & directional transition matrices (`src/evaluation/matrices.py`).
   - `explorer_m4_3`: Syntactic chunk error analysis (`src/evaluation/analysis.py`).

---

## 2. Logic Chain

1. **Aggregation & Data Model Strategy (`src/evaluation/tables.py`)**:
   - To compare methods systematically, we require a strongly typed record (`EvaluationResult`) storing track, method, model, split, Weighted F1, SoftCons, HardCons, relative deltas ($\Delta\text{F}_1$, $\Delta\text{SoftCons}$, $\Delta\text{HardCons}$), instance/pair counts, and submission path.
   - The container `ComparisonTable` wraps a collection of `EvaluationResult` objects and provides:
     - Canonical ordering: Track 1 before Track 2; methods ordered logically: `baseline` $\to$ `bsa` $\to$ `crl` $\to$ `sd` $\to$ `crl_sd`.
     - Delta computation against the baseline method.
     - Multiformat exporters: Markdown, publication-ready LaTeX (ACL format), ASCII plain text, and structured JSON.
   - Two entrypoints for data ingestion:
     - `aggregate_results_from_dir`: Automatically scans `results/` for Codabench submission CSVs and `logs/` for summary JSONs. If metric values are already present in JSONs, it ingests them directly; if only CSV files exist, it evaluates them against the PhrasIS gold benchmark using `evaluate_prediction_file`.
     - `build_comparison_table`: Direct builder from pre-computed metrics or dictionaries.

2. **ACL LaTeX Table Formulation**:
   - `report/paper.tex` requires a high-quality table compliant with the SemEval-2027 ACL template (`booktabs`).
   - When both Track 1 (English - DeBERTa-v3) and Track 2 (Spanish - BETO) are evaluated, `ComparisonTable.to_latex(multi_track=True)` formats a side-by-side two-column table environment (`table*`):
     - `\multicolumn{3}{c}{\textbf{Track 1: English (DeBERTa-v3)}}` and `\multicolumn{3}{c}{\textbf{Track 2: Spanish (BETO)}}`.
     - Columns: `Method`, `Strategy`, and for each track `Weighted F$_1$`, `SoftCons`, `HardCons`.
     - Automatic bolding of the best scores per track column (e.g. `\textbf{1.0000}` for SoftCons under SD and CRL+SD).
   - If a single track is evaluated, it falls back to a clean single-column `table` format.

3. **Standalone Evaluation CLI (`experiments/evaluate.py`)**:
   - Designed to support both static evaluation and dynamic model evaluation:
     - **Static Mode**: Evaluates pre-existing prediction CSV files or summary JSONs in `--results_dir`.
     - **Dynamic Mode**: Loads a model checkpoint via `--checkpoint_path` or instantiates models via `build_model`, runs inference over the requested split, exports Codabench submission CSVs, and computes metrics.
   - Flag interface:
     - `--track`: `track1`, `track2`, or `all`.
     - `--split`: `dev` (default), `train`, `test`.
     - `--method`: `baseline`, `bsa`, `crl`, `sd`, `crl_sd`, or `all`.
     - `--checkpoint_path`: path to `.pt` checkpoint file.
     - `--results_dir`: input directory (default `results`).
     - `--output_dir`: output directory (default `results/evaluation`).
     - `--generate_tables`: exports `.md`, `.tex`, `.txt`, `.json` (default `True`).
     - `--generate_matrices`: triggers 4x4 confusion and transition matrix export.
     - `--generate_analysis`: triggers syntactic category breakdown export.
     - `--test_run`: ultra-fast CPU test mode (<2s execution budget).
   - Speed Assertion: In `--test_run` mode, mock data and mock models execute all 10 combinations (2 tracks $\times$ 5 methods) in **0.020s**, far below the 2.0s limit.

4. **Integration with Sibling Explorer Modules**:
   - `proposed_evaluate.py` imports `src.evaluation.matrices` and `src.evaluation.analysis` if available, but falls back defensively to `src.metrics.consistency` (`compute_directional_transition_matrix` and `compute_consistency_by_category`). This guarantees that `evaluate.py` functions immediately and remains fully backward- and forward-compatible.

---

## 3. Caveats

1. **Pandas and PyTorch Portability**:
   - While `pandas` and `torch` are standard, all proposed modules include defensive fallbacks (`SimpleDataFrame`, standard library `csv`, `math`, `json`, and pure-Python matrix operations) so that evaluation scripts run even in minimal environments without crashing.
2. **Gold Label Availability for Test Split**:
   - When `--split test` is evaluated, ground-truth labels are absent in official competitions. In that scenario, `evaluate.py` exports submission CSVs and validates Codabench compliance, but outputs metrics as `-` (unlabeled). SoftCons can still be computed across predicted reciprocal pairs even without gold labels!

---

## 4. Conclusion & Actionable Proposals

Three complete, production-ready source artifacts have been developed and tested in `.agents/teamwork/explorer_m4_1/`:

1. **`proposed_tables.py`** $\to$ Target: `src/evaluation/tables.py`
   - Complete implementation of `EvaluationResult`, `ComparisonTable`, `aggregate_results_from_dir`, `build_comparison_table`, and formatters for Markdown, LaTeX, ASCII, and JSON.
2. **`proposed_evaluate.py`** $\to$ Target: `experiments/evaluate.py`
   - Complete standalone CLI tool supporting all requested flags, dual static/dynamic evaluation modes, fast mock execution (<0.03s), and automated table/matrix/analysis artifact export.
3. **`proposed_test_evaluation.py`** $\to$ Target: `tests/test_evaluation.py`
   - Complete 15-test test suite covering dataclasses, filtering, sorting, deltas, all 4 table formatters, metadata extraction, mathematical invariants ($0 \le \text{HardCons} \le \text{SoftCons} \le 1.0$), SD 100% SoftCons proof assertion, CLI argument parsing, single-file evaluation, and strict <2.0s execution speed assertion.

### Concrete Implementation Steps for Worker M4:
1. Copy `.agents/teamwork/explorer_m4_1/proposed_tables.py` to `src/evaluation/tables.py` (and create `src/evaluation/__init__.py`).
2. Copy `.agents/teamwork/explorer_m4_1/proposed_evaluate.py` to `experiments/evaluate.py`.
3. Copy `.agents/teamwork/explorer_m4_1/proposed_test_evaluation.py` to `tests/test_evaluation.py`.
4. Run `python3 -m unittest discover tests` to verify 100% test pass rate.

---

## 5. Verification Method

### 5.1 Verification Commands
1. **Run Unit Test Suite**:
   ```bash
   python3 .agents/teamwork/explorer_m4_1/proposed_test_evaluation.py
   ```
   *Expected Result*: `Ran 15 tests in ~0.16s ... OK`.

2. **Verify CLI Test Run Execution Speed and Artifact Generation**:
   ```bash
   python3 .agents/teamwork/explorer_m4_1/proposed_evaluate.py --test_run --generate_tables --generate_matrices --generate_analysis --output_dir results/test_verify
   ```
   *Expected Result*:
   - Completes in $< 0.05$ seconds ($< 2.0$s budget).
   - Outputs Markdown table to stdout.
   - Creates `results/test_verify/comparison_table.md`, `comparison_table.tex`, `comparison_table.txt`, `comparison_table.json`, `evaluation_summary.json`, transition matrices, and syntactic analysis JSON.

3. **Verify Table Ingestion of Real Milestone 3 Results**:
   ```bash
   python3 -c "
   import sys, importlib.util
   spec = importlib.util.spec_from_file_location('tables', '.agents/teamwork/explorer_m4_1/proposed_tables.py')
   tables = importlib.util.module_from_spec(spec)
   sys.modules['tables'] = tables
   spec.loader.exec_module(tables)

   table = tables.aggregate_results_from_dir(results_dir='results', logs_dir='logs', data_dir='data')
   print(table.to_markdown())
   "
   ```
   *Expected Result*:
   - Successfully ingests all 21 runs from `results/` and `logs/`.
   - Displays comparison table with SoftCons = 1.0000 on SD and CRL+SD runs.

4. **Verify LaTeX Output Validity**:
   Inspect `results/test_verify/comparison_table.tex`.
   Ensure `\begin{table*}[t]`, `\toprule`, `\cmidrule`, `\textbf{1.0000}`, caption, label, and `\end{table*}` are properly formatted.

### 5.2 Invalidation Conditions
- Any execution of `evaluate.py --test_run` exceeding 2.0 seconds.
- Any failure in the mathematical invariant $\text{HardCons} \le \text{SoftCons}$.
- Failure of Symmetrized Decoding rows (`sd`, `crl_sd`) to display $\text{SoftCons} = 1.0000$ (100.0%).
- Any syntax or compilation error in generated LaTeX code.
