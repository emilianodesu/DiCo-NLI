# Adversarial Review & Empirical Challenge Report: Milestone 1 (M1)

- **Reviewer:** Challenger 1 (`challenger_m1_1`)
- **Target Deliverable:** Milestone 1 (Data Pipeline & Consistency Metrics)
- **Reviewed Modules:** `src/data/loader.py`, `src/data/reversal.py`, `src/data/codabench.py`, `src/metrics/metrics.py`, `tests/`
- **Date:** 2026-11-01T22:17:00Z
- **Verdict:** **REQUEST_CHANGES**

---

## 1. Observation

### Observation 1: Full Test Suite Baseline
Executed standard test suite:
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```
Output:
```
Ran 46 tests in 0.062s
OK
```
All existing tests passed. However, the existing tests evaluated only artificial dictionaries or mocks and completely missed inter-module pipeline integration and edge case propagation.

---

### Observation 2: Pipeline Integration Crash in `build_reversal_dataset` with `SimpleDataFrame` / `DataFrame`
- **Location:** `src/data/reversal.py`, lines 392–405
- **Command:**
```bash
python3 -c "
from src.data.loader import load_phrasis_split
from src.data.reversal import build_reversal_dataset
df = load_phrasis_split('track1', 'train')
build_reversal_dataset(df, augment_bsa=True)
"
```
- **Verbatim Error:**
```
Traceback (most recent call last):
  File "<string>", line 5, in <module>
    build_reversal_dataset(df, augment_bsa=True)
  File "/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/src/data/reversal.py", line 402, in build_reversal_dataset
    r.to_dict() if isinstance(r, InstanceRecord) else dict(r)
                                                      ~~~~^^^
ValueError: dictionary update sequence element #0 has length 1; 2 is required
```
- **When `augment_bsa=False`:**
```bash
python3 -c "
from src.data.loader import load_phrasis_split
from src.data.reversal import build_reversal_dataset
df = load_phrasis_split('track1', 'train')
res = build_reversal_dataset(df, augment_bsa=False)
print('res len:', len(res), 'res:', res)
"
```
Output:
```
res len: 9 res: ['instance_id', 'pair_id', 'text1_lang', 'text2_lang', 'text1', 'text2', 'label', 'category', 'reverse_pair_id']
```
When `build_reversal_dataset` is called on the output of `load_phrasis_split`, it returns a list of 9 column name strings instead of the dataset records, discarding the entire dataset.

---

### Observation 3: Shadowed / Duplicate `InstanceRecord` Class Definition
- **Location:** `src/data/loader.py:102` vs `src/data/reversal.py:205`
- **Command:**
```bash
python3 -c "
from src.data.loader import InstanceRecord
from src.data.reversal import build_reversal_dataset
rec = InstanceRecord('id1', 'p1', 'en', 'en', 'hello', 'world', 'FORWARD_ENTAILMENT', 'verbal')
build_reversal_dataset([rec], augment_bsa=True)
"
```
- **Verbatim Error:**
```
Traceback (most recent call last):
  File "<string>", line 5, in <module>
    build_reversal_dataset([rec], augment_bsa=True)
  File "/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/src/data/reversal.py", line 402, in build_reversal_dataset
    r.to_dict() if isinstance(r, InstanceRecord) else dict(r)
                                                      ~~~~^^^
  File "/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/src/data/loader.py", line 119, in __getitem__
    return getattr(self, key)
TypeError: attribute name must be string, not 'int'
```
Because `InstanceRecord` is defined independently in both `src/data/loader.py` and `src/data/reversal.py`, `isinstance(r, InstanceRecord)` inside `src/data/reversal.py` evaluates to `False` for instances produced by `src/data/loader.py`. It falls through to `dict(r)`, which attempts integer-indexed attribute lookup, raising `TypeError`.

---

### Observation 4: `ReciprocalPair.is_reversible` Defaults to `True` for `NEGATIVE_OTHER`, Causing Metric Assertion Crash
- **Location:** `src/data/reversal.py`, lines 285, 345, 363–367; `src/metrics/metrics.py`, line 377
- **Command:**
```bash
python3 -c "
from src.data.reversal import pair_instances
records = [
    {'instance_id': 'neg_01__original', 'pair_id': 'neg_01', 'reverse_pair_id': 'neg_01__flipped', 'label': 'NEGATIVE_OTHER'},
    {'instance_id': 'neg_01__flipped', 'pair_id': 'neg_01', 'reverse_pair_id': 'neg_01__original', 'label': 'NEGATIVE_OTHER'},
]
pairs = pair_instances(records)
print('is_reversible:', pairs[0].is_reversible)
"
```
Output:
```
is_reversible: True
```
When this pair is evaluated and the model correctly predicts `NEGATIVE_OTHER` in both directions:
```bash
python3 -c "
from src.metrics.metrics import compute_official_metrics
m = compute_official_metrics(['NEGATIVE_OTHER'], ['NEGATIVE_OTHER'], ['NEGATIVE_OTHER'], ['NEGATIVE_OTHER'], [True])
"
```
- **Verbatim Error:**
```
Traceback (most recent call last):
  File "<string>", line 3, in <module>
    m = compute_official_metrics(['NEGATIVE_OTHER'], ['NEGATIVE_OTHER'], ['NEGATIVE_OTHER'], ['NEGATIVE_OTHER'], [True])
  File "/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/src/metrics/metrics.py", line 419, in compute_official_metrics
    cons_report = compute_consistency_report(pairs, pred_map)
  File "/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/src/metrics/metrics.py", line 377, in compute_consistency_report
    assert hard_cons <= soft_cons + 1e-12, (
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Mathematical invariant violated: HardCons (1.0) > SoftCons (0.0)
```
Because `pair_instances` defaults `is_reversible` to `True` without checking `is_reversible_label(label)`, `NEGATIVE_OTHER` pairs are passed into the consistency calculation. Since `soft_ok` requires `pred in REVERSIBLE_LABELS`, `soft_ok` evaluates to `False` while `hard_ok` evaluates to `True`, breaking the mathematical invariant `hard_cons <= soft_cons` and aborting execution.

---

### Observation 5: Dead Parameter `strict_reversal` in `pair_instances`
- **Location:** `src/data/reversal.py`, lines 304–370
- **Command:**
```bash
python3 -c "
from src.data.reversal import pair_instances
records = [
    {'instance_id': 'id1', 'pair_id': 'p1', 'reverse_pair_id': 'id2', 'label': 'FORWARD_ENTAILMENT'},
    {'instance_id': 'id2', 'pair_id': 'p1', 'reverse_pair_id': 'id1', 'label': 'FORWARD_ENTAILMENT'},
]
pairs = pair_instances(records, strict_reversal=True)
print('Pairs count:', len(pairs))
"
```
Output:
```
Pairs count: 1
```
The parameter `strict_reversal: bool = False` is documented in the docstring as *"raises ValueError if paired labels violate Rev_BSA"*, but the argument is completely ignored in the body of `pair_instances`. Contradictory pairs (`FORWARD` paired with `FORWARD`) are accepted silently.

---

### Observation 6: Fragile Multi-Strategy Fallback in `pair_instances` Drops Records
- **Location:** `src/data/reversal.py`, line 348
- **Command:**
```bash
python3 -c "
from src.data.reversal import pair_instances
records = [
    {'instance_id': 'id1', 'pair_id': 'p1', 'reverse_pair_id': 'id2', 'label': 'EQUIVALENCE'},
    {'instance_id': 'id2', 'pair_id': 'p1', 'reverse_pair_id': 'id1', 'label': 'EQUIVALENCE'},
    {'instance_id': 'id3__original', 'pair_id': 'p2', 'reverse_pair_id': None, 'label': 'EQUIVALENCE'},
    {'instance_id': 'id3__flipped', 'pair_id': 'p2', 'reverse_pair_id': None, 'label': 'EQUIVALENCE'},
]
pairs = pair_instances(records)
print('Found pairs:', len(pairs))
"
```
Output:
```
Found pairs: 1
```
Because line 348 specifies `if not pairs:`, if any single record in the dataset is paired via `reverse_pair_id`, the fallback to `pair_id` is never executed. Valid pairs that only specify `pair_id` are silently dropped.

---

### Observation 7: Naive Adverbial Suffix Extraction in `extract_syntactic_category`
- **Location:** `src/data/loader.py`, line 433
- **Command:**
```bash
python3 -c "
from src.data.loader import extract_syntactic_category
print('family:', extract_syntactic_category('a royal family', 'a noble house', 'en'))
print('butterfly:', extract_syntactic_category('a monarch butterfly', 'an insect', 'en'))
"
```
Output:
```
family: adverbial
butterfly: adverbial
```
Naive checking `any(w.endswith("ly") for w in all_words)` causes common English nouns ("family", "supply", "butterfly", "assembly", "ally") to be misclassified as `adverbial` instead of `nominal`.

---

## 2. Logic Chain

1. **Interface Contract Violation (Observation 2):**
   `PROJECT.md` establishes the interface contract: `load_phrasis_split(...) -> DataFrame`, followed by `build_reversal_dataset(df: pd.DataFrame, augment_bsa: bool = False) -> Dataset`. Because `build_reversal_dataset` does not convert `SimpleDataFrame` (or `pd.DataFrame`) into a record list before iteration, calling `build_reversal_dataset` on loaded data either discards all data rows (returning column strings) or crashes with `ValueError`. Downstream modeling in Milestone 2 cannot function with this defect.

2. **Cross-Module Type Drift (Observation 3):**
   `InstanceRecord` was duplicated in both `src/data/loader.py` and `src/data/reversal.py`. Type identity checks (`isinstance(r, InstanceRecord)`) fail when records flow from the loader module to the reversal module.

3. **Inconsistency in Mathematical Invariant Engine (Observation 4):**
   `PROJECT.md` and SemEval-2027 Task 2 state that consistency metrics (`SoftCons` and `HardCons`) are defined strictly over the closed reversible set $\mathcal{R} = \{\text{EQUIVALENCE}, \text{FORWARD\_ENTAILMENT}, \text{BACKWARD\_ENTAILMENT}\}$. `NEGATIVE\_OTHER` is non-reversible. By allowing `ReciprocalPair.is_reversible` to default to `True` for `NEGATIVE_OTHER` pairs, `compute_official_metrics` attempts to compute consistency on `NEGATIVE_OTHER`. When the model correctly predicts `NEGATIVE_OTHER`, `HardCons = 1.0` and `SoftCons = 0.0`, violating the mathematical invariant assert and crashing the evaluation pipeline.

4. **Dead Validation Contract (Observation 5):**
   `strict_reversal` is declared in the signature and docstring of `pair_instances`, but never evaluated, silently allowing inconsistent label pairings.

5. **Partial Data Loss in Mixed Partitions (Observation 6):**
   `if not pairs:` creates mutual exclusivity between `reverse_pair_id` matching and `pair_id` matching. Mixed datasets lose all unlinked pairs.

---

## 3. Caveats

- **No Caveats.** All observations were confirmed by direct execution in Darwin sandbox with Python 3.14.3. No external factors or speculative hypotheses were used.

---

## 4. Conclusion

**Verdict: REQUEST_CHANGES**

Milestone 1 cannot be approved in its current state. The following fixes are required:

1. **Fix `build_reversal_dataset`**:
   - Support `SimpleDataFrame` and `pd.DataFrame` seamlessly:
     ```python
     if hasattr(records, "to_dict") and callable(getattr(records, "to_dict")):
         records_list = records.to_dict("records")
     elif isinstance(records, Sequence):
         records_list = [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in records]
     ```
   - Ensure that if `augment_bsa=False`, it returns `records_list` (or DataFrame if `return_df=True`), NOT `list(records)`.
2. **Consolidate `InstanceRecord`**:
   - Define `InstanceRecord` in one place (e.g. `src/data/loader.py`) and import it in `src/data/reversal.py` (or vice versa), and check `hasattr(r, "to_dict")` instead of strict `isinstance`.
3. **Correct `is_reversible` assignment in `pair_instances`**:
   - Compute `is_reversible`:
     ```python
     is_rev = is_reversible_label(lbl1) and is_reversible_label(lbl2)
     ```
     and pass `is_reversible=is_rev` when constructing `ReciprocalPair`.
4. **Implement `strict_reversal` in `pair_instances`**:
   - If `strict_reversal=True`, assert `reverse_label_bsa(lbl1) == lbl2`, raising `ValueError` on contradiction.
5. **Decouple pairing passes in `pair_instances`**:
   - Do not use `if not pairs:` for `pair_id` matching. Instead, track `seen_keys` across both passes so that records with `reverse_pair_id` are paired first, and remaining unpaired records are paired by `pair_id`.
6. **Refine syntactic category rules**:
   - Exclude common nouns ending in `-ly` (such as "family", "butterfly", "assembly", "supply") from the adverbial rule in `extract_syntactic_category`.

---

## 5. Verification Method

To verify resolution of these findings:

1. **Verify DataFrame input to `build_reversal_dataset`:**
   ```bash
   python3 -c "
   from src.data.loader import load_phrasis_split
   from src.data.reversal import build_reversal_dataset
   df = load_phrasis_split('track1', 'train')
   aug = build_reversal_dataset(df, augment_bsa=True)
   orig = build_reversal_dataset(df, augment_bsa=False)
   assert len(orig) == len(df), f'orig len {len(orig)} != df len {len(df)}'
   assert len(aug) > len(df)
   assert isinstance(orig[0], dict)
   print('DataFrame integration test: PASSED')
   "
   ```

2. **Verify `InstanceRecord` list input to `build_reversal_dataset`:**
   ```bash
   python3 -c "
   from src.data.loader import read_phrasis_csv
   from src.data.reversal import build_reversal_dataset
   records = read_phrasis_csv('data/raw/dico_nli_train_track1_participant_labeled.csv')
   aug = build_reversal_dataset(records, augment_bsa=True)
   print('InstanceRecord integration test: PASSED, len =', len(aug))
   "
   ```

3. **Verify `is_reversible` flag and `compute_official_metrics` with `NEGATIVE_OTHER`:**
   ```bash
   python3 -c "
   from src.data.reversal import pair_instances
   from src.metrics.metrics import compute_official_metrics
   records = [
       {'instance_id': 'n1', 'pair_id': 'p1', 'reverse_pair_id': 'n2', 'label': 'NEGATIVE_OTHER'},
       {'instance_id': 'n2', 'pair_id': 'p1', 'reverse_pair_id': 'n1', 'label': 'NEGATIVE_OTHER'},
   ]
   pairs = pair_instances(records)
   assert pairs[0].is_reversible is False, 'NEGATIVE_OTHER pair must have is_reversible=False'
   m = compute_official_metrics(['NEGATIVE_OTHER'], ['NEGATIVE_OTHER'], ['NEGATIVE_OTHER'], ['NEGATIVE_OTHER'], [pairs[0].is_reversible])
   assert m['weighted_f1'] == 1.0
   assert m['soft_cons'] == 1.0
   print('NEGATIVE_OTHER invariant test: PASSED')
   "
   ```

4. **Verify `strict_reversal` rejection:**
   ```bash
   python3 -c "
   from src.data.reversal import pair_instances
   records = [
       {'instance_id': 'id1', 'pair_id': 'p1', 'reverse_pair_id': 'id2', 'label': 'FORWARD_ENTAILMENT'},
       {'instance_id': 'id2', 'pair_id': 'p1', 'reverse_pair_id': 'id1', 'label': 'FORWARD_ENTAILMENT'},
   ]
   try:
       pair_instances(records, strict_reversal=True)
       raise AssertionError('Failed: strict_reversal=True did not raise ValueError')
   except ValueError:
       print('strict_reversal rejection test: PASSED')
   "
   ```

5. **Run entire test suite:**
   ```bash
   python3 -m unittest discover -s tests -p "test_*.py" -v
   ```
