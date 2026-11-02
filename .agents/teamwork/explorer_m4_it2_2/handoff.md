# Investigation & Architectural Blueprint: Multi-Model Disambiguation & ACL LaTeX Compatibility in `src/evaluation/tables.py`

**Agent**: `explorer_m4_it2_2` (Role: Explorer 2 M4 It2 Tables Multi-Model)  
**Date**: 2026-11-02T03:50:00Z  
**Target Scope**: `src/evaluation/tables.py`, `tests/test_evaluation.py`, ACL SemEval LaTeX integration (`report/paper.tex`)  
**Mission**: Investigate `src/evaluation/tables.py`, formulate a robust blueprint to ensure multi-model disambiguation, verify LaTeX table formatting and booktabs rules without directly modifying source code.

---

## 1. Observation

### 1.1 Direct Inspection of `src/evaluation/tables.py`

1. **Side-by-Side LaTeX Exporter Overwriting Multiple Models (Lines 838–863)**:
   ```python
   # Check if multi-track side-by-side layout can be used
   can_side_by_side = multi_track and len(tracks_present) == 2

   if can_side_by_side:
       ...
       t1_dict = {r.method: r for r in table.results if r.track == "track1"}
       t2_dict = {r.method: r for r in table.results if r.track == "track2"}
   ```
   - When a user evaluates multiple models for the same track (e.g., both `DeBERTa-v3` and `Mock`, or cross-model ablations), `t1_dict` indexes exclusively by `r.method`. If two rows share `method="baseline"`, the later entry overwrites the earlier entry. All earlier models are silently dropped and lost from the exported LaTeX table.

2. **Hardcoded Model Names in Column Headers (Lines 847–850)**:
   ```python
   "\\begin{tabular}{ll ccc ccc}",
   "\\toprule",
   " & & \\multicolumn{3}{c}{\\textbf{Track 1: English (DeBERTa-v3)}} & \\multicolumn{3}{c}{\\textbf{Track 2: Spanish (BETO)}} \\\\",
   "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}",
   ```
   - The sub-headers in side-by-side mode statically hardcode `(DeBERTa-v3)` and `(BETO)`. Even if the evaluation was run using `Mock`, `RoBERTa-bne`, or custom fine-tuned checkpoints, the rendered LaTeX table falsely claims `DeBERTa-v3` and `BETO`.

3. **Bolding Desynchronization in Side-by-Side Mode (Lines 820–835, 867–880)**:
   ```python
   for t in tracks_present:
       t_res = [r for r in table.results if r.track == t]
       if t_res:
           best_f1[t] = max(r.weighted_f1 for r in t_res)
           best_sc[t] = max(r.soft_cons for r in t_res)
           best_hc[t] = max(r.hard_cons for r in t_res)
   ```
   - `best_f1` is computed across all results for `track1` (including the dropped model). If `DeBERTa-v3` had the best F1 score (e.g. 0.85) but was overwritten in `t1_dict` by `Mock` (highest F1 0.31), the printed table displays `Mock`'s metrics, and zero cells are bolded because `0.31 != 0.85`.

4. **Interleaved Sorting in Multi-Model Sweeps (Lines 213–221)**:
   ```python
   def sort(self, by: str = "canonical") -> ComparisonTable:
       """Sort results canonically by track, then canonical method order."""
       def sort_key(res: EvaluationResult) -> Tuple[int, int, str]:
           t_order = 0 if res.track == "track1" else 1
           method_idx = CANONICAL_METHODS.index(res.method) if res.method in CANONICAL_METHODS else 99
           return (t_order, method_idx, res.model)

       sorted_results = sorted(self.results, key=sort_key)
       return ComparisonTable(sorted_results, title=self.title)
   ```
   - In `sort_key`, `method_idx` precedes `res.model`. In tables containing multiple models, rows alternate models method-by-method:
     1. Track 1, Baseline, DeBERTa-v3
     2. Track 1, Baseline, Mock
     3. Track 1, BSA, DeBERTa-v3
     4. Track 1, BSA, Mock
   - The reader cannot see the coherent progression of methods for a single model; instead, rows sharing the same method name visually collide in Markdown, ASCII, and stacked LaTeX tables.

5. **Missing Group Separators and Inconsistent Column Ordering in Stacked Tables (Lines 896–916)**:
   ```python
   \begin{tabular}{lll ccc}
   \toprule
   \textbf{Track} & \textbf{Method} & \textbf{Model} & \textbf{Weighted F$_1$} & \textbf{SoftCons} & \textbf{HardCons} \\
   \midrule
   ...
   curr_track = None
   for r in table.results:
       if curr_track is not None and r.track != curr_track:
           lines.append("\midrule")
       curr_track = r.track
       ...
       lines.append(f"{t_disp} & {m_disp} & {r.model} & {f1_str} & {sc_str} & {hc_str} \\\\")
   ```
   - Stacked LaTeX only inserts `\midrule` when `r.track != curr_track`. Multiple models within the same track have no dividing line between them.
   - The column header in stacked LaTeX is `Track & Method & Model`, whereas Markdown and ASCII use `Track & Model & Method`. This violates hierarchical consistency (Language/Track $\to$ Backbone Model $\to$ Mitigation Method).

6. **Missing LaTeX Character Escaping (Lines 849, 881, 915)**:
   - Neither `r.model` nor `m_disp` is escaped for LaTeX special characters (`_`, `&`, `%`, `#`, `$`). If a model name is `roberta_large` or `bert_base_uncased`, or if a custom method name contains underscores, LaTeX compilation will throw syntax errors (`! Missing $ inserted.`).

7. **Missing Bounds Validation in `EvaluationResult` (Lines 136–141)**:
   ```python
   # Invariant check: HardCons <= SoftCons (with float epsilon)
   if self.hard_cons > self.soft_cons + 1e-6:
       logger.warning(...)
   ```
   - `EvaluationResult.__post_init__` checks `hard_cons <= soft_cons`, but does not validate whether metric values fall within the mathematically required $[0.0, 1.0]$ probability/F1 bounds.

---

## 2. Logic Chain

1. **Premise 1 (Data Integrity & Truthful Presentation)**:
   - Tables must faithfully represent all benchmark results without silently discarding data.
   - *Direct Evidence*: Observation 1.1 item 1 demonstrates that `t1_dict = {r.method: r ...}` discards $(N - 1)$ models when $N > 1$ models exist for a single track.
   - *Deduction*: Side-by-side mode (`can_side_by_side`) can only be valid when each track has at most 1 distinct model (`len({r.model for r in table.results if r.track == t}) <= 1`). When any track contains multiple models, the exporter must automatically fall back to the stacked table layout where every model is displayed.

2. **Premise 2 (Dynamic Labeling vs Hardcoding)**:
   - Academic artifacts must reflect empirical execution, not hardcoded assumptions.
   - *Direct Evidence*: Observation 1.1 item 2 shows static strings `Track 1: English (DeBERTa-v3)` and `Track 2: Spanish (BETO)`.
   - *Deduction*: Side-by-side column headers must dynamically extract model names from `table.results` (`t1_model_name = next(iter(t1_models))` and `t2_model_name = next(iter(t2_models))`).

3. **Premise 3 (Visual Disambiguation & Ergonomics)**:
   - Comparing mitigation methods requires viewing the progression from Baseline to CRL+SD per model architecture.
   - *Direct Evidence*: Observation 1.1 item 4 shows that sorting by `(t_order, method_idx, res.model)` interleaves rows across models.
   - *Deduction*: Sorting by `(t_order, res.model.lower(), method_idx)` groups all rows for a given `(track, model)` contiguously. In stacked LaTeX and ASCII tables, tracking `group_key = (r.track, r.model)` and inserting `\midrule` (or ASCII divider) creates clean, partitioned model blocks.

4. **Premise 4 (ACL SemEval & Booktabs Compliance)**:
   - Tables included in `report/paper.tex` must adhere to standard ACL SemEval guidelines and `booktabs` typography:
     - No vertical rules (`|`).
     - Heavy `\toprule` and `\bottomrule`, light `\midrule`, clipped `\cmidrule(lr){...}`.
     - Robust LaTeX special character escaping (`_` $\to$ `\_`).
     - Standard math mode for metrics (`Weighted F$_1$`, `SoftCons`, `HardCons`).
     - Proper float environments (`table*` for two-column side-by-side; `table` for single-column stacked).

---

## 3. Caveats

1. **Side-by-Side Column Budget**:
   - The side-by-side table layout spans 8 columns (`Method`, `Strategy`, and 3 metrics $\times$ 2 tracks). It is strictly intended for 2-column wide floats (`\begin{table*}[t]`). It cannot fit within a single column without clipping.
2. **Delta Columns in Side-by-Side Tables**:
   - Including $\Delta F_1$, $\Delta \text{SoftCons}$, and $\Delta \text{HardCons}$ in side-by-side mode would expand the table to 14 columns, exceeding standard ACL page width. Deltas are supported in Markdown, ASCII, and optionally in stacked LaTeX tables.
3. **No Direct Code Changes**:
   - As an explorer agent in read-only mode, no source code in `src/evaluation/tables.py` or `tests/test_evaluation.py` has been modified directly. All changes are provided as exact, machine-verifiable blueprints below.

---

## 4. Conclusion & Architectural Blueprint

### Blueprint Component 1: `src/evaluation/tables.py` Modifications

#### 1.1 Helper: LaTeX Character Escaping
Add helper function before formatters:
```python
def _latex_escape(text: str) -> str:
    """Escapes special LaTeX characters for text mode rendering."""
    replacements = [
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    ]
    escaped = str(text)
    # Perform safe non-nested replacement
    for old, new in replacements:
        if old == "\\":
            continue
        escaped = escaped.replace(old, new)
    return escaped
```

#### 1.2 Metric Bounds Validation in `EvaluationResult.__post_init__`
Replace lines 136–141 in `src/evaluation/tables.py`:
```python
        # Invariant check: HardCons <= SoftCons (with float epsilon)
        if self.hard_cons > self.soft_cons + 1e-6:
            logger.warning(
                "HardCons (%.4f) exceeds SoftCons (%.4f) in result for %s/%s. Invariant violated.",
                self.hard_cons, self.soft_cons, self.track, self.method
            )

        # Value range checks: metrics must be in [0.0, 1.0] (with float epsilon)
        for metric_name, val in [
            ("weighted_f1", self.weighted_f1),
            ("soft_cons", self.soft_cons),
            ("hard_cons", self.hard_cons),
        ]:
            if not (-1e-6 <= val <= 1.0 + 1e-6):
                logger.warning(
                    "%s (%.4f) out of standard [0.0, 1.0] range in result for %s/%s.",
                    metric_name, val, self.track, self.method
                )
```

#### 1.3 Contiguous Multi-Model Sorting in `ComparisonTable.sort()`
Replace lines 213–222 in `src/evaluation/tables.py`:
```python
    def sort(self, by: str = "canonical") -> ComparisonTable:
        """Sort results canonically by track, model, then canonical method order."""
        def sort_key(res: EvaluationResult) -> Tuple[Any, ...]:
            t_order = 0 if res.track == "track1" else 1
            method_idx = CANONICAL_METHODS.index(res.method) if res.method in CANONICAL_METHODS else 99
            if by == "method":
                return (t_order, method_idx, res.model.lower())
            # Default canonical: group by track, then by model, then by method progression
            return (t_order, res.model.lower(), method_idx)

        sorted_results = sorted(self.results, key=sort_key)
        return ComparisonTable(sorted_results, title=self.title)
```

#### 1.4 Group Dividers in ASCII Exporter
Replace lines 757–763 in `src/evaluation/tables.py`:
```python
    current_group = None
    for r in table.results:
        group_key = (r.track, r.model)
        if current_group is not None and group_key != current_group:
            lines.append(div)
        current_group = group_key
```

#### 1.5 Multi-Model Safe LaTeX Exporter (`generate_latex_table`)
Replace lines 813–925 in `src/evaluation/tables.py`:
```python
def generate_latex_table(
    table: ComparisonTable,
    caption: str = "Experimental evaluation of reversal mitigation methods on Track 1 (English) and Track 2 (Spanish) dev sets. Best scores per track are bolded.",
    label: str = "tab:main_results",
    multi_track: bool = True,
    bold_best: bool = True,
    bold_scope: str = "model",
    table_env: Optional[str] = None,
) -> str:
    """Formats comparison table into publication-ready LaTeX tabular code for ACL paper."""
    if not table.results:
        return "% No evaluation results to render in LaTeX table."

    # Identify tracks present
    tracks_present = []
    for t in ["track1", "track2"]:
        if any(r.track == t for r in table.results):
            tracks_present.append(t)

    # Check distinct models per track
    t1_models = sorted(list({r.model for r in table.results if r.track == "track1"}))
    t2_models = sorted(list({r.model for r in table.results if r.track == "track2"}))
    has_single_model_per_track = len(t1_models) <= 1 and len(t2_models) <= 1

    # Compute best values for bolding:
    # If bold_scope == "model", compute best per (track, model) group.
    # If bold_scope == "track", compute best per track.
    best_f1: Dict[Tuple[str, str], float] = {}
    best_sc: Dict[Tuple[str, str], float] = {}
    best_hc: Dict[Tuple[str, str], float] = {}

    for r in table.results:
        key = (r.track, r.model if bold_scope == "model" else "")
        best_f1[key] = max(best_f1.get(key, 0.0), r.weighted_f1)
        best_sc[key] = max(best_sc.get(key, 0.0), r.soft_cons)
        best_hc[key] = max(best_hc.get(key, 0.0), r.hard_cons)

    def _latex_cell(val: float, best_val: float, is_bold_allowed: bool) -> str:
        s = f"{val:.4f}"
        if is_bold_allowed and abs(val - best_val) < 1e-4:
            return f"\\textbf{{{s}}}"
        return s

    # Side-by-side mode is strictly valid only when:
    # 1. multi_track is requested
    # 2. exactly 2 tracks are present
    # 3. each track has at most 1 distinct model (to avoid multi-model collision)
    can_side_by_side = multi_track and len(tracks_present) == 2 and has_single_model_per_track

    lines: List[str] = []

    if can_side_by_side:
        # Dynamic model names for headers
        t1_name = _latex_escape(t1_models[0]) if t1_models else DEFAULT_MODELS.get("track1", "DeBERTa-v3")
        t2_name = _latex_escape(t2_models[0]) if t2_models else DEFAULT_MODELS.get("track2", "BETO")

        env = table_env or "table*"
        lines.extend([
            f"\\begin{{{env}}}[t]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{ll ccc ccc}",
            "\\toprule",
            f" & & \\multicolumn{{3}}{{c}}{{\\textbf{{Track 1: English ({t1_name})}}}} & \\multicolumn{{3}}{{c}}{{\\textbf{{Track 2: Spanish ({t2_name})}}}} \\\\",
            "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}",
            "\\textbf{Method} & \\textbf{Strategy} & \\textbf{Weighted F$_1$} & \\textbf{SoftCons} & \\textbf{HardCons} & \\textbf{Weighted F$_1$} & \\textbf{SoftCons} & \\textbf{HardCons} \\\\",
            "\\midrule",
        ])

        t1_dict = {r.method: r for r in table.results if r.track == "track1"}
        t2_dict = {r.method: r for r in table.results if r.track == "track2"}

        # Combine canonical methods and any extra methods present
        methods_to_render = list(CANONICAL_METHODS)
        for r in table.results:
            if r.method not in methods_to_render:
                methods_to_render.append(r.method)

        for m in methods_to_render:
            r1 = t1_dict.get(m)
            r2 = t2_dict.get(m)
            if r1 is None and r2 is None:
                continue

            m_disp = _latex_escape(METHOD_DISPLAY_NAMES.get(m, m.upper()))
            m_desc = _latex_escape(METHOD_DESCRIPTIONS.get(m, ""))

            if r1 is not None:
                b_key1 = (r1.track, r1.model if bold_scope == "model" else "")
                f1_1 = _latex_cell(r1.weighted_f1, best_f1.get(b_key1, 0.0), bold_best)
                sc_1 = _latex_cell(r1.soft_cons, best_sc.get(b_key1, 0.0), bold_best)
                hc_1 = _latex_cell(r1.hard_cons, best_hc.get(b_key1, 0.0), bold_best)
            else:
                f1_1, sc_1, hc_1 = "-", "-", "-"

            if r2 is not None:
                b_key2 = (r2.track, r2.model if bold_scope == "model" else "")
                f1_2 = _latex_cell(r2.weighted_f1, best_f1.get(b_key2, 0.0), bold_best)
                sc_2 = _latex_cell(r2.soft_cons, best_sc.get(b_key2, 0.0), bold_best)
                hc_2 = _latex_cell(r2.hard_cons, best_hc.get(b_key2, 0.0), bold_best)
            else:
                f1_2, sc_2, hc_2 = "-", "-", "-"

            lines.append(f"{m_disp} & {m_desc} & {f1_1} & {sc_1} & {hc_1} & {f1_2} & {sc_2} & {hc_2} \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\end{{{env}}}",
        ])
    else:
        # Standard stacked single-table layout (Used when multi_track=False or when multi-model sweep is detected)
        env = table_env or "table"
        lines.extend([
            f"\\begin{{{env}}}[ht]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{lll ccc}",
            "\\toprule",
            "\\textbf{Track} & \\textbf{Model} & \\textbf{Method} & \\textbf{Weighted F$_1$} & \\textbf{SoftCons} & \\textbf{HardCons} \\\\",
            "\\midrule",
        ])

        curr_group: Optional[Tuple[str, str]] = None
        for r in table.results:
            group_key = (r.track, r.model)
            if curr_group is not None and group_key != curr_group:
                lines.append("\\midrule")
            curr_group = group_key

            m_disp = _latex_escape(METHOD_DISPLAY_NAMES.get(r.method, r.method.upper()))
            t_disp = "Track 1 (EN)" if r.track == "track1" else "Track 2 (ES)"
            model_disp = _latex_escape(r.model)

            b_key = (r.track, r.model if bold_scope == "model" else "")
            f1_str = _latex_cell(r.weighted_f1, best_f1.get(b_key, 0.0), bold_best)
            sc_str = _latex_cell(r.soft_cons, best_sc.get(b_key, 0.0), bold_best)
            hc_str = _latex_cell(r.hard_cons, best_hc.get(b_key, 0.0), bold_best)

            lines.append(f"{t_disp} & {model_disp} & {m_disp} & {f1_str} & {sc_str} & {hc_str} \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\end{{{env}}}",
        ])

    return "\n".join(lines)
```

---

## 5. Verification Method

### 5.1 Unit Tests to Add to `tests/test_evaluation.py`

Implement the following test cases in `TestComparisonTables` inside `tests/test_evaluation.py`:

```python
    def test_multi_model_side_by_side_fallback(self) -> None:
        """Verify that when a track contains multiple models, to_latex falls back to stacked layout."""
        # Track 1 has DeBERTa-v3 AND Mock
        r1 = EvaluationResult("track1", "baseline", "DeBERTa-v3", "dev", 0.75, 0.60, 0.55)
        r2 = EvaluationResult("track1", "baseline", "Mock", "dev", 0.31, 0.66, 0.26)
        r3 = EvaluationResult("track2", "baseline", "BETO", "dev", 0.71, 0.58, 0.50)

        tbl = ComparisonTable([r1, r2, r3])
        tex = tbl.to_latex(multi_track=True)

        # Fallback to stacked table prevents silent dropping of Mock
        self.assertIn("\\begin{table}[ht]", tex)
        self.assertIn("DeBERTa-v3", tex)
        self.assertIn("Mock", tex)
        self.assertIn("BETO", tex)
        # All 3 models present
        self.assertEqual(tex.count("Baseline"), 3)

    def test_dynamic_model_headers_in_side_by_side_latex(self) -> None:
        """Verify dynamic model names appear in side-by-side LaTeX headers instead of hardcoded defaults."""
        r1 = EvaluationResult("track1", "baseline", "RoBERTa-large", "dev", 0.78, 0.62, 0.56)
        r2 = EvaluationResult("track2", "baseline", "RoBERTa-bne", "dev", 0.74, 0.60, 0.52)

        tbl = ComparisonTable([r1, r2])
        tex = tbl.to_latex(multi_track=True)

        self.assertIn("Track 1: English (RoBERTa-large)", tex)
        self.assertIn("Track 2: Spanish (RoBERTa-bne)", tex)
        self.assertNotIn("DeBERTa-v3", tex)
        self.assertNotIn("(BETO)", tex)

    def test_multi_model_sorting_groups_by_model(self) -> None:
        """Verify canonical sorting groups contiguous methods per model architecture."""
        r_deb_base = EvaluationResult("track1", "baseline", "DeBERTa-v3", "dev", 0.75, 0.60, 0.55)
        r_deb_sd = EvaluationResult("track1", "sd", "DeBERTa-v3", "dev", 0.82, 1.00, 0.78)
        r_mock_base = EvaluationResult("track1", "baseline", "Mock", "dev", 0.31, 0.66, 0.26)
        r_mock_sd = EvaluationResult("track1", "sd", "Mock", "dev", 0.40, 1.00, 0.35)

        tbl = ComparisonTable([r_deb_sd, r_mock_base, r_mock_sd, r_deb_base])
        sorted_tbl = tbl.sort()

        pairs = [(r.model, r.method) for r in sorted_tbl.results]
        expected = [
            ("DeBERTa-v3", "baseline"),
            ("DeBERTa-v3", "sd"),
            ("Mock", "baseline"),
            ("Mock", "sd"),
        ]
        self.assertEqual(pairs, expected)

    def test_latex_special_character_escaping(self) -> None:
        """Verify model and method names with underscores or special characters do not break LaTeX."""
        r = EvaluationResult("track1", "baseline", "custom_model_v1", "dev", 0.75, 0.60, 0.55)
        tbl = ComparisonTable([r])
        tex = tbl.to_latex(multi_track=False)

        self.assertIn("custom\\_model\\_v1", tex)
        self.assertNotIn("custom_model_v1", tex.replace("custom\\_model\\_v1", ""))
```

### 5.2 Verification Commands
```bash
# 1. Run all unit and integration tests
python3 -m unittest discover -s tests -p "test_*.py" -v

# 2. Run CLI fast evaluation test run with table generation
python3 experiments/evaluate.py --test_run --generate_tables

# 3. Check generated LaTeX artifact
cat results/evaluation/comparison_table.tex
```

### 5.3 Invalidation Conditions
- Any test failure in `tests/test_evaluation.py`.
- Silent dropping of any `EvaluationResult` when multiple models share track/method names.
- Hardcoded `DeBERTa-v3` or `BETO` strings appearing in LaTeX headers when evaluated models differ.
- Unescaped LaTeX characters (`_`, `%`, `&`) in generated `.tex` files.
- Visual interleaving of methods across models in `ComparisonTable.sort()`.
