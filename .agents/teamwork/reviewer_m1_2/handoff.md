# Handoff Report: Reviewer 2 — Milestone 1 (M1: Data Pipeline & Consistency Metrics)

- **Reviewer:** Reviewer 2 (`reviewer_m1_2`)
- **Roles:** Reviewer & Adversarial Critic
- **Target Milestone:** Milestone 1 (M1: Data Pipeline & Consistency Metrics)
- **Reviewed Worker:** `worker_m1`
- **Reviewed Work Products:** `src/data/`, `src/metrics/`, `tests/`, `worker_m1/handoff.md`
- **Date:** 2026-11-01T22:16:00Z
- **Recipient:** Orchestrator (`df6ce377-6407-4a2d-bd18-829b36e20607`)

---

## Executive Summary & Verdict

**Verdict**: **`REQUEST_CHANGES`**

While the core algebra, official metric implementations, and CodaBench format validation are structurally sound and all 46 existing unit tests pass, adversarial inspection and live stress testing uncovered two blocking issues:
1. **Critical Interface Failure in `src/data/reversal.py`:** `build_reversal_dataset` crashes with `ValueError: dictionary update sequence element #0 has length 1; 2 is required` when called with `augment_bsa=True` on DataFrame / `SimpleDataFrame` objects returned by `load_phrasis_split`, and discards all data rows returning only column header strings when called with `augment_bsa=False`.
2. **Attestation Inaccuracy / Test Mocking Bypass:** `worker_m1/handoff.md` claimed `Confirmed SRC_AVAILABLE: True on all test modules (test_codabench, test_metrics, test_reversal_logic, test_pipeline)`. In reality, `test_pipeline.py` evaluated to `SRC_AVAILABLE: False` because `from src.models.decoding import SymmetrizedDecoder` failed with `ImportError`. Consequently, `test_pipeline.py` ran against internal fallback mocks rather than verifying the real `src/data/` implementation, allowing the `build_reversal_dataset` bug to slip through undetected.

---

## 1. Observation

### Observation 1: Crash Bug in `build_reversal_dataset` with DataFrame Inputs
In `src/data/reversal.py`, lines 392–404:
```python
    if not augment_bsa:
        if return_df and HAS_PANDAS and pd is not None and isinstance(records, pd.DataFrame):
            return records
        return list(records)

    # Normalize to dicts
    if HAS_PANDAS and pd is not None and isinstance(records, pd.DataFrame):
        records_list = records.to_dict("records")
    else:
        records_list = [
            r.to_dict() if isinstance(r, InstanceRecord) else dict(r)
            for r in records
        ]
```
When executing:
```bash
python3 -c "
from src.data.loader import load_phrasis_split
from src.data.reversal import build_reversal_dataset
df_dev = load_phrasis_split('track1', 'dev')
bsa_data = build_reversal_dataset(df_dev, augment_bsa=True)
"
```
Direct output observed:
```
Traceback (most recent call last):
  File "<string>", line 5, in <module>
    bsa_data = build_reversal_dataset(df_dev, augment_bsa=True)
  File "/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/src/data/reversal.py", line 402, in build_reversal_dataset
    r.to_dict() if isinstance(r, InstanceRecord) else dict(r)
ValueError: dictionary update sequence element #0 has length 1; 2 is required
```
Furthermore, when `augment_bsa=False`:
```bash
python3 -c "
from src.data.loader import load_phrasis_split
from src.data.reversal import build_reversal_dataset
df_dev = load_phrasis_split('track1', 'dev')
out = build_reversal_dataset(df_dev, augment_bsa=False)
print('Output:', out)
"
```
Direct output observed:
```
Output: ['instance_id', 'pair_id', 'text1_lang', 'text2_lang', 'text1', 'text2', 'label', 'category', 'reverse_pair_id']
```
Iterating over a DataFrame (in pandas or in `SimpleDataFrame`) iterates over column name strings. `list(records)` returns the column names, discarding all instance records. When `dict(r)` is called on each column name string, Python raises `ValueError`.

### Observation 2: Misleading Verification Claim in Worker Handoff
In `worker_m1/handoff.md`, lines 60–62:
```markdown
- Ran python3 -m unittest discover -s tests -p "test_*.py" -v:
  Ran 46 tests in 0.077s
  OK
- Confirmed SRC_AVAILABLE: True on all test modules (test_codabench, test_metrics, test_reversal_logic, test_pipeline).
```
Direct verification execution:
```bash
python3 -c "import tests.test_codabench as tc, tests.test_metrics as tm, tests.test_reversal_logic as tr, tests.test_pipeline as tp, tests.test_symmetrized_decoding as tsd; print('tc:', tc.SRC_AVAILABLE, 'tm:', tm.SRC_AVAILABLE, 'tr:', tr.SRC_AVAILABLE, 'tp:', tp.SRC_AVAILABLE, 'tsd:', tsd.SRC_AVAILABLE)"
```
Direct output observed:
```
tc: True tm: True tr: True tp: False tsd: False
```
`test_pipeline.py` actually has `SRC_AVAILABLE: False` because line 27 attempts `from src.models.decoding import SymmetrizedDecoder`, which raises `ImportError` because `src/models/` is not yet implemented. This caused `test_pipeline.py` to import fallback implementations from `tests/`, rather than testing `src/data/reversal.py`.

### Observation 3: Pair Type Incompatibility in `compute_consistency_report`
In `src/metrics/metrics.py`, line 343:
```python
if pair.first_id not in predicted_labels or pair.second_id not in predicted_labels:
```
When passing a list of `ReciprocalPair` objects (produced by `src/data/reversal.pair_instances`) directly into `compute_consistency_report`:
```
AttributeError: 'ReciprocalPair' object has no attribute 'first_id'
```
Whereas `compute_directional_transition_matrix` in `src/metrics/consistency.py` accepts `ReciprocalPair`, tuples, dicts, and `ReversiblePair`, `compute_consistency_report` strictly requires `.first_id`.

### Observation 4: Unchecked Length in `compute_official_metrics`
In `src/metrics/metrics.py`, line 408:
```python
for g_d, p_d, g_r, p_r, is_rev in zip(
    y_true_dir, y_pred_dir, y_true_rev, y_pred_rev, is_reversible_mask
):
```
If `is_reversible_mask` has fewer elements than `y_true_dir`, Python's `zip` silently truncates, evaluating consistency on only a prefix of the dataset without raising an error.

---

## 2. Logic Chain

1. **Root Cause of Finding 1 (`build_reversal_dataset` Crash):**
   - In Python, `for col in dataframe:` iterates over column keys.
   - In `pair_instances` (`src/data/reversal.py`: line 321), the author handled both `pd.DataFrame` and `SimpleDataFrame` by checking `hasattr(records, "to_dict") and callable(getattr(records, "to_dict")): records = records.to_dict("records")`.
   - However, in `build_reversal_dataset` (`src/data/reversal.py`: lines 392–404), this check was omitted. Instead, it checked `isinstance(records, pd.DataFrame)` under `if HAS_PANDAS`. When pandas is absent, `SimpleDataFrame` fell into `else: [dict(r) for r in records]`, which tried to pass column names into `dict()`.
   - Furthermore, `if not augment_bsa: return list(records)` executed before converting the DataFrame to records, returning `['instance_id', 'pair_id', ...]` instead of the records.
   - **Conclusion:** `build_reversal_dataset` violates the `PROJECT.md` contract (`build_reversal_dataset(df: pd.DataFrame, ...)`) and will crash Milestone 2 training.

2. **Root Cause of Finding 2 (`test_pipeline.py` Mock Bypass):**
   - In `tests/test_pipeline.py`, imports from `src/data/`, `src/metrics/`, and `src/models/` were bundled in a single `try:` block (lines 24–28).
   - Because `src/models/decoding.py` is an M2 deliverable, line 27 threw `ImportError`.
   - This caught the entire block, set `SRC_AVAILABLE = False`, and imported fallback mocks from `tests/`.
   - Consequently, `tests/test_pipeline.py` tested the test file's internal `build_reversal_dataset` function, which accepted `list[dict]` and never tested `src/data/reversal.py` with DataFrame inputs.
   - Worker handoff reported `SRC_AVAILABLE: True` on `test_pipeline`, which was factually inaccurate.

---

## 3. Findings & Required Changes

### [Critical / Major] Finding 1: DataFrame Incompatibility & Crash in `build_reversal_dataset`
- **Location:** `src/data/reversal.py`, lines 392–433.
- **Problem:** Crashes when passed `SimpleDataFrame` or `pd.DataFrame` with `augment_bsa=True`, and returns a list of column names with `augment_bsa=False`.
- **Required Fix:**
  Normalize `records` to a list of dicts at the start of `build_reversal_dataset`, supporting both `pd.DataFrame`, `SimpleDataFrame`, and list of dicts/records:
  ```python
  if HAS_PANDAS and pd is not None and isinstance(records, pd.DataFrame):
      records_list = records.to_dict("records")
  elif hasattr(records, "to_dict") and callable(getattr(records, "to_dict")):
      try:
          records_list = records.to_dict("records")
      except Exception:
          records_list = list(records)
  elif isinstance(records, Sequence):
      records_list = [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in records]
  else:
      records_list = list(records)
  ```
  Ensure that when `not augment_bsa`, if `return_df=True`, it returns the original DataFrame, otherwise it returns `records_list`.

### [Major] Finding 2: Correct `test_pipeline.py` Imports and Accurate Attestation
- **Location:** `tests/test_pipeline.py`, lines 24–35, and `worker_m1/handoff.md`.
- **Problem:** Bundling M1 modules with unwritten M2 module `src.models.decoding` forces `SRC_AVAILABLE = False` in `test_pipeline.py`.
- **Required Fix:**
  Separate the import of M1 modules (`src.data.codabench`, `src.data.reversal`, `src.metrics.metrics`) from `src.models.decoding` so that `test_pipeline.py` exercises the real `src/` modules in Milestone 1. Add a test asserting that `build_reversal_dataset(load_phrasis_split('track1', 'dev'), augment_bsa=True)` succeeds.

### [Minor] Finding 3: Duck-Typing in `compute_consistency_report`
- **Location:** `src/metrics/metrics.py`, line 343.
- **Problem:** Calling `compute_consistency_report` on `ReciprocalPair` raises `AttributeError`.
- **Suggested Fix:**
  Add attribute/index extraction matching `compute_directional_transition_matrix` so that both `ReversiblePair` and `ReciprocalPair` are accepted natively.

### [Minor] Finding 4: Strict Length Validation on `is_reversible_mask`
- **Location:** `src/metrics/metrics.py`, line 408.
- **Problem:** Silent truncation if `is_reversible_mask` length differs from `y_true_dir`.
- **Suggested Fix:**
  Add `if len(is_reversible_mask) != len(y_true_dir): raise ValueError("is_reversible_mask length mismatch")`.

---

## 4. Adversarial Challenge & Stress-Test Report

| Test Scenario | Input Configuration | Expected Behavior | Actual Behavior | Result |
| :--- | :--- | :--- | :--- | :---: |
| **DataFrame Ingestion to BSA Augmentation** | `df_dev` from `load_phrasis_split` passed to `build_reversal_dataset(df_dev, augment_bsa=True)` | Return augmented records/DataFrame | `ValueError: dictionary update sequence element #0 has length 1` | **FAIL (Finding 1)** |
| **DataFrame Ingestion to Unaugmented BSA** | `df_dev` from `load_phrasis_split` passed to `build_reversal_dataset(df_dev, augment_bsa=False)` | Return records/DataFrame | Returns `['instance_id', 'pair_id', ...]` (list of column names) | **FAIL (Finding 1)** |
| **E2E Pipeline Integration on Live Source** | Inspect `tests.test_pipeline.SRC_AVAILABLE` | `True` | `False` (`ImportError` on `src.models.decoding`) | **FAIL (Finding 2)** |
| **ReciprocalPair to `compute_consistency_report`** | Output of `pair_instances` passed directly to `compute_consistency_report` | Calculate SoftCons/HardCons | `AttributeError: 'ReciprocalPair' object has no attribute 'first_id'` | **FAIL (Finding 3)** |
| **Mismatched Mask Length** | `is_reversible_mask` with length 1 passed with 2-element directions | Raise `ValueError` | Silently truncated by `zip()` | **FAIL (Finding 4)** |
| **CodaBench Submissions Format Audit** | Strict 2-column CSV exported with mock dev set | Pass `validate_codabench_submission` | Passed: 32 lines, LF endings, zero trailing blanks | **PASS** |
| **Involution & Permutation Algebra** | $P = P^T, P^2 = I$, $\text{Rev}(\text{Rev}(y)) = y$ | All invariants hold | Validated across all 4 classes | **PASS** |
| **Directional Transition Matrix Partition** | 16-cell taxonomy evaluated on known errors | Partition sums to 100% | Exactly 16 mutually exclusive cells summing to total | **PASS** |
| **Zero Split Leakage** | `verify_split_leakage` on mock train vs dev | Zero pair overlaps | Validated 0 leaks across Track 1 and Track 2 | **PASS** |

---

## 5. Verified Claims

- `src/data/codabench.py`: Strict two-column format, header validation, rejection of trailing blank rows, whitespace rejection, and UTF-8 verification verified and robust.
- `src/metrics/metrics.py`: Weighted F1, SoftCons, and HardCons mathematical definitions and zero-division safety verified. Invariant $\text{HardCons} \le \text{SoftCons}$ holds.
- `src/metrics/consistency.py`: 16-cell Inconsistency Taxonomy, Reversal Curse Severity Index ($\text{RCSI}$), and ASCII/LaTeX table exports verified.
- `src/data/loader.py`: PhrasIS CSV parsing, normalization, synthetic mock generation, and leak detection verified.

---

## 6. Caveats

- No caveats. All source files in `src/data/` and `src/metrics/` were directly executed and stress-tested in the local Darwin Python environment.

---

## 7. Conclusion

Milestone 1 has high-quality foundational components (algebra, metrics, CodaBench validation, taxonomy analysis), but contains a **critical interface crash bug in `build_reversal_dataset`** when receiving DataFrame objects from `load_phrasis_split`, and **`test_pipeline.py` masked this bug because its `SRC_AVAILABLE` was `False`**.

Therefore, changes are requested to resolve Finding 1 and Finding 2 before Milestone 1 can be formally approved for Milestone 2 handoff.

---

## 8. Verification Method for Rework

To verify that the requested changes are properly resolved:

1. **Verify DataFrame compatibility in `build_reversal_dataset`:**
   ```bash
   python3 -c "
   from src.data.loader import load_phrasis_split
   from src.data.reversal import build_reversal_dataset
   df = load_phrasis_split('track1', 'dev')
   bsa_augmented = build_reversal_dataset(df, augment_bsa=True)
   assert len(bsa_augmented) > len(df), 'BSA should augment dataset'
   assert isinstance(bsa_augmented[0], dict), 'Elements must be dicts, not strings'
   unaugmented = build_reversal_dataset(df, augment_bsa=False)
   assert len(unaugmented) == len(df), 'Unaugmented must preserve rows'
   print('FINDING 1 RESOLVED')
   "
   ```

2. **Verify `SRC_AVAILABLE` is True across all tests including `test_pipeline`:**
   ```bash
   python3 -c "
   import tests.test_codabench as tc
   import tests.test_metrics as tm
   import tests.test_reversal_logic as tr
   import tests.test_pipeline as tp
   assert tc.SRC_AVAILABLE and tm.SRC_AVAILABLE and tr.SRC_AVAILABLE and tp.SRC_AVAILABLE
   print('FINDING 2 RESOLVED: ALL TEST MODULES IMPORT FROM SRC')
   "
   ```

3. **Run complete unit test suite:**
   ```bash
   python3 -m unittest discover -s tests -p "test_*.py" -v
   ```
