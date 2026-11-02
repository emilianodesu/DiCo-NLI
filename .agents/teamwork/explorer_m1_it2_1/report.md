# Technical Analysis & Blueprint: Data Contracts & BSA Ingestion Fixes (Milestone 1, Iteration 2)

**Author:** Explorer 1 (`explorer_m1_it2_1`)  
**Target Worker:** `worker_m1`  
**Milestone:** Milestone 1 (Data Pipeline & Consistency Metrics) — Iteration 2  
**Date:** 2026-11-01T22:20:00Z  

---

## Executive Summary

During Gate Iteration 1 for Milestone 1, Reviewers 1 & 2 and Challenger 1 issued `REQUEST_CHANGES` due to critical defects in dataset ingestion, cross-module type identity, and consistency metric invariants:
1. **`build_reversal_dataset` Interface Failure:** Passing DataFrame or `SimpleDataFrame` inputs (as returned by `load_phrasis_split`) returned only column header strings when `augment_bsa=False` (discarding all data) and crashed with `ValueError` when `augment_bsa=True`.
2. **Duplicated `InstanceRecord`:** `InstanceRecord` was independently defined in `src/data/loader.py` and `src/data/reversal.py`, causing `isinstance` checks to fail across modules and raising `TypeError: attribute name must be string, not 'int'` when converting records.
3. **`ReciprocalPair.is_reversible` Invariant Failure:** `NEGATIVE_OTHER` pairs defaulted to `is_reversible=True`, violating the closed reversible algebra $\mathcal{R}$, which caused `compute_official_metrics` to violate the mathematical assertion $\text{HardCons} \le \text{SoftCons}$.
4. **`test_pipeline.py` Import Masking:** Cross-milestone import of unimplemented `src.models.decoding` forced `SRC_AVAILABLE = False`, running tests against local test mocks rather than testing `src/data/` live.

This document provides a comprehensive root-cause analysis, unified data contract design, and complete file-by-file patch blueprints for `worker_m1`.

---

## 1. Deep Root-Cause Analysis: `build_reversal_dataset`

### 1.1 Ingestion Failure Mechanisms on DataFrame / SimpleDataFrame
`PROJECT.md` line 61 specifies the interface contract:
```python
build_reversal_dataset(df: pd.DataFrame, augment_bsa: bool = False) -> Dataset
```
And `src/data/loader.py:load_phrasis_split` returns either `pandas.DataFrame` (when pandas is installed) or `SimpleDataFrame` (offline fallback).

In `src/data/reversal.py` lines 392–405, the original code had:
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

#### Defect A: Data Loss when `augment_bsa=False`
In Python, iterating over a DataFrame (`for item in df:`) iterates over the **column names** (`__iter__` yields `'instance_id'`, `'pair_id'`, etc.).
- When `augment_bsa=False` and `return_df=False` (default), `return list(records)` executed `list(df)`.
- For `SimpleDataFrame` (or `pd.DataFrame` with `return_df=False`), this evaluated to:
  `['instance_id', 'pair_id', 'text1_lang', 'text2_lang', 'text1', 'text2', 'label', 'category', 'reverse_pair_id']`
- **Result:** Exactly 9 column header strings were returned. All dataset rows were completely dropped!

#### Defect B: Crash when `augment_bsa=True`
When `records` was a `SimpleDataFrame` (because pandas was not present or not detected):
- `isinstance(records, pd.DataFrame)` was `False`.
- Execution entered the `else` branch:
  ```python
  records_list = [
      r.to_dict() if isinstance(r, InstanceRecord) else dict(r)
      for r in records
  ]
  ```
- Because `records` was iterated directly, `r` was the string column name `'instance_id'`.
- `'instance_id'` is not an `InstanceRecord`, so Python executed `dict('instance_id')`.
- Python's `dict()` constructor expects an iterable of key-value pairs (length 2). The character `'i'` has length 1.
- **Verbatim Error:** `ValueError: dictionary update sequence element #0 has length 1; 2 is required`.

### 1.2 Polymorphic Row Extraction Solution
To cleanly handle `pandas.DataFrame`, `SimpleDataFrame`, `list[dict]`, `list[InstanceRecord]`, and arbitrary sequences, row extraction must occur at the entry of the function before any branching:

```python
def extract_record_dicts(records: Any) -> Tuple[List[Dict[str, Any]], bool, Any]:
    """
    Polymorphic row extractor supporting:
    1. pandas.DataFrame (.to_dict('records'))
    2. SimpleDataFrame (.to_dict('records'))
    3. Sequence[InstanceRecord] (.to_dict() per item)
    4. Sequence[dict] (dict(r) per item)
    5. Arbitrary mappings or objects with to_dict()
    
    Returns:
        (records_list, is_dataframe, df_class)
    """
    if hasattr(records, "to_dict") and callable(getattr(records, "to_dict")):
        try:
            records_list = records.to_dict("records")
            return records_list, True, type(records)
        except (TypeError, ValueError):
            pass

    records_list = [
        r.to_dict() if hasattr(r, "to_dict") and callable(getattr(r, "to_dict"))
        else (dict(r) if hasattr(r, "keys") or isinstance(r, dict) else r.__dict__)
        for r in records
    ]
    return records_list, False, None
```

#### Behavior with `augment_bsa`:
- When `not augment_bsa`:
  - If `return_df=True`: if input was already a DataFrame, return `records` directly. Otherwise construct `df_class(records_list)` or `pd.DataFrame(records_list)` / `SimpleDataFrame(records_list)`.
  - If `return_df=False` (default): return `list(records_list)` (guaranteed list of row dicts).
- When `augment_bsa=True`:
  - Generates inverted pairs `(B, A)` with inverted labels $\text{Rev}_{\text{BSA}}(y)$.
  - If `return_df=True`: return `df_class(augmented)` or `pd.DataFrame(augmented)` / `SimpleDataFrame(augmented)`.
  - If `return_df=False`: return `augmented` (list of row dicts).

---

## 2. Unifying `InstanceRecord` (Single Source of Truth)

### 2.1 The Cross-Module Divergence Problem
- In `src/data/loader.py:102`, `InstanceRecord` was defined as:
  ```python
  @dataclass
  class InstanceRecord:
      instance_id: str
      pair_id: str
      text1_lang: str
      text2_lang: str
      text1: str
      text2: str
      label: Optional[str] = None
      category: Optional[str] = None
      reverse_pair_id: Optional[str] = None
  ```
- In `src/data/reversal.py:205`, `InstanceRecord` was redefined independently with different default arguments and a `.transposed()` method:
  ```python
  @dataclass
  class InstanceRecord:
      instance_id: str
      pair_id: str
      text1_lang: str = "en"
      text2_lang: str = "en"
      text1: str = ""
      text2: str = ""
      label: Optional[str] = None
      category: Optional[str] = "nominal"
      reverse_pair_id: Optional[str] = None
  ```

Because Python uses nominal typing for classes, `loader.InstanceRecord is not reversal.InstanceRecord`.
When `read_phrasis_csv(...)` returned `loader.InstanceRecord` instances, `isinstance(r, InstanceRecord)` inside `src/data/reversal.py` evaluated to `False`.
This caused code to fall back to `dict(r)`.
Because `loader.InstanceRecord` implemented `__getitem__` without `keys()` or mapping iteration, Python attempted integer index access `r[0]`, executing `getattr(self, 0)`.
**Verbatim Error:** `TypeError: attribute name must be string, not 'int'`.

### 2.2 Canonical Architecture for `InstanceRecord`
`src/data/loader.py` is the data ingestion foundation. Therefore:
1. **Single Source of Truth:** Define `InstanceRecord` strictly in `src/data/loader.py`.
2. **Imported in `src/data/reversal.py`:** `from .loader import InstanceRecord, SimpleDataFrame`.
3. **Re-exported in `src/data/__init__.py`:** Single unified export.
4. **Complete Mapping Protocol & Transposed Method:**
   To make `InstanceRecord` bulletproof under any Python dict or subscript operation, implement:
   - `keys(self) -> List[str]`
   - `__iter__(self) -> Iterator[str]` (iterating yields keys, so `dict(r)` and `for k in r:` work as a standard mapping)
   - `__contains__(self, key: str) -> bool`
   - `to_dict(self) -> Dict[str, Any]`
   - `transposed(self, new_instance_id: Optional[str] = None, new_label: Optional[str] = None) -> InstanceRecord`

```python
@dataclass
class InstanceRecord:
    """
    Strongly typed representation of a single DiCo-NLI evaluation or training instance.
    Supports both attribute and full dict-like subscript/mapping access.
    """
    instance_id: str
    pair_id: str
    text1_lang: str = "en"
    text2_lang: str = "en"
    text1: str = ""
    text2: str = ""
    label: Optional[str] = None
    category: Optional[str] = "nominal"
    reverse_pair_id: Optional[str] = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self) -> List[str]:
        return [
            "instance_id", "pair_id", "text1_lang", "text2_lang",
            "text1", "text2", "label", "category", "reverse_pair_id",
        ]

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __contains__(self, key: str) -> bool:
        return key in self.keys()

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a standard dictionary."""
        return {
            "instance_id": self.instance_id,
            "pair_id": self.pair_id,
            "text1_lang": self.text1_lang,
            "text2_lang": self.text2_lang,
            "text1": self.text1,
            "text2": self.text2,
            "label": self.label,
            "category": self.category,
            "reverse_pair_id": self.reverse_pair_id,
        }

    def transposed(
        self,
        new_instance_id: Optional[str] = None,
        new_label: Optional[str] = None,
    ) -> InstanceRecord:
        """Synthesizes the transposed reciprocal instance (B, A) with inverted label."""
        if new_label is not None:
            inv_label = new_label
        elif self.label == "FORWARD_ENTAILMENT":
            inv_label = "BACKWARD_ENTAILMENT"
        elif self.label == "BACKWARD_ENTAILMENT":
            inv_label = "FORWARD_ENTAILMENT"
        else:
            inv_label = self.label

        synth_id = new_instance_id or f"{self.instance_id}__bsa_rev"
        return InstanceRecord(
            instance_id=synth_id,
            pair_id=self.pair_id,
            text1_lang=self.text2_lang,
            text2_lang=self.text1_lang,
            text1=self.text2,
            text2=self.text1,
            label=inv_label,
            category=self.category,
            reverse_pair_id=self.instance_id,
        )
```

---

## 3. Ancillary Data Contract & Invariant Issues

### 3.1 `ReciprocalPair.is_reversible` Handling for `NEGATIVE_OTHER`
- **Issue:** In `src/data/reversal.py`, `ReciprocalPair` defaulted `is_reversible=True`.
- When `pair_instances` paired two `NEGATIVE_OTHER` items, `is_reversible` remained `True`.
- In `compute_consistency_report`, only reversible labels qualify for `soft_ok = (pred in REVERSIBLE_LABELS and pred_rev == Rev(pred))`.
- For `NEGATIVE_OTHER`, `soft_ok` evaluates to `0`, while if the model predicted correctly, `hard_ok` evaluates to `1`.
- This broke the invariant $\text{HardCons} \le \text{SoftCons}$ and triggered `AssertionError`.
- **Fix:** In `pair_instances`, calculate:
  ```python
  lbl1 = r1.get("label") if hasattr(r1, "get") else getattr(r1, "label", None)
  lbl2 = r2.get("label") if hasattr(r2, "get") else getattr(r2, "label", None)
  is_rev = is_reversible_label(lbl1) and is_reversible_label(lbl2) if (lbl1 and lbl2) else False
  ```
  Pass `is_reversible=is_rev` into `ReciprocalPair`.

### 3.2 Implement `strict_reversal` in `pair_instances`
- **Issue:** `strict_reversal: bool = False` was declared in `pair_instances` docstring but never evaluated in the body.
- **Fix:** When `strict_reversal=True` and both instances have labels, assert `lbl2 == reverse_label_bsa(lbl1)` and raise `ValueError` on contradiction.

### 3.3 Decouple Pairing Passes in `pair_instances`
- **Issue:** `if not pairs:` skipped `pair_id` matching if any single record was paired via `reverse_pair_id`, silently dropping valid pairs in mixed datasets.
- **Fix:** Pass 1 pairs records via `reverse_pair_id` and records paired IDs. Pass 2 groups remaining unpaired records by `pair_id`.

### 3.4 Duck-Typing Pair Attributes in `ReciprocalPair`
- **Issue:** `compute_consistency_report` accesses `pair.first_id`, `pair.second_id`, `pair.first_label`, `pair.second_label`. `ReciprocalPair` lacked these attributes.
- **Fix:** Add properties `.first_id`, `.second_id`, `.first_label`, `.second_label` to `ReciprocalPair`.

### 3.5 Defensive Zero-Slice Handling in `compute_official_metrics`
- **Issue:** When an evaluation slice has no reversible pairs (`len(pairs) == 0`), `compute_consistency_report` raised `ValueError: Cannot compute consistency metrics without reversible pairs.`
- **Fix:** If `len(pairs) == 0`, set `soft_cons = 1.0` and `hard_cons = 1.0` (vacuously consistent).

### 3.6 Suffix False Positives on Nominals ending in `-ly`
- **Issue:** `extract_syntactic_category` classified words like "family", "butterfly", "supply", "assembly", "ally" as `adverbial` because of naive `w.endswith("ly")`.
- **Fix:** Add exclusion set `_NON_ADVERB_LY_WORDS` and check `w.endswith("ly") and w not in _NON_ADVERB_LY_WORDS`.

### 3.7 Decouple `tests/test_pipeline.py` Imports
- **Issue:** Line 27 imported `src.models.decoding.SymmetrizedDecoder`, setting `SRC_AVAILABLE = False` because `src/models/` is Milestone 2.
- **Fix:** Separate the import of Milestone 1 modules (`src.data.*`, `src.metrics.*`) from `src.models.decoding` so `SRC_AVAILABLE` is `True`.

---

## 4. Exact Code Patch Blueprints for Worker

### Blueprint 1: `src/data/loader.py`

#### Edit 1.1: Replace `InstanceRecord` with Unified Implementation
**Target File:** `src/data/loader.py`  
**Lines:** ~102–141  
**Replace With:**
```python
@dataclass
class InstanceRecord:
    """
    Strongly typed representation of a single DiCo-NLI evaluation or training instance.
    Supports both attribute and full dict-like subscript/mapping access.
    """
    instance_id: str
    pair_id: str
    text1_lang: str = "en"
    text2_lang: str = "en"
    text1: str = ""
    text2: str = ""
    label: Optional[str] = None
    category: Optional[str] = "nominal"
    reverse_pair_id: Optional[str] = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self) -> List[str]:
        return [
            "instance_id",
            "pair_id",
            "text1_lang",
            "text2_lang",
            "text1",
            "text2",
            "label",
            "category",
            "reverse_pair_id",
        ]

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __contains__(self, key: str) -> bool:
        return key in self.keys()

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a standard dictionary."""
        return {
            "instance_id": self.instance_id,
            "pair_id": self.pair_id,
            "text1_lang": self.text1_lang,
            "text2_lang": self.text2_lang,
            "text1": self.text1,
            "text2": self.text2,
            "label": self.label,
            "category": self.category,
            "reverse_pair_id": self.reverse_pair_id,
        }

    def transposed(
        self,
        new_instance_id: Optional[str] = None,
        new_label: Optional[str] = None,
    ) -> InstanceRecord:
        """Synthesizes the transposed reciprocal instance (B, A) with inverted label."""
        if new_label is not None:
            inv_label = new_label
        elif self.label == "FORWARD_ENTAILMENT":
            inv_label = "BACKWARD_ENTAILMENT"
        elif self.label == "BACKWARD_ENTAILMENT":
            inv_label = "FORWARD_ENTAILMENT"
        else:
            inv_label = self.label

        synth_id = new_instance_id or f"{self.instance_id}__bsa_rev"
        return InstanceRecord(
            instance_id=synth_id,
            pair_id=self.pair_id,
            text1_lang=self.text2_lang,
            text2_lang=self.text1_lang,
            text1=self.text2,
            text2=self.text1,
            label=inv_label,
            category=self.category,
            reverse_pair_id=self.instance_id,
        )
```

#### Edit 1.2: Add `_NON_ADVERB_LY_WORDS` and Fix Adverbial Check
**Target File:** `src/data/loader.py`  
**Before line 425:**
```python
_NON_ADVERB_LY_WORDS: FrozenSet[str] = frozenset({
    "family", "supply", "butterfly", "assembly", "ally", "monopoly", "anomaly",
    "belly", "jelly", "rally", "italy", "early", "friendly", "lovely", "holy",
    "daily", "weekly", "monthly", "yearly", "ugly", "lonely", "silly", "lively",
    "costly", "orderly", "timely", "fly", "apply", "imply", "rely",
})
```
**At line 433:**
```python
    else:
        if (
            any(w.endswith("ly") and w not in _NON_ADVERB_LY_WORDS for w in all_words)
            or first_word_1 in adv_lexicon
            or first_word_2 in adv_lexicon
        ):
            return SyntacticCategory.ADVERBIAL.value
        if first_word_1 == "as" or first_word_2 == "as":
            return SyntacticCategory.ADVERBIAL.value
```

---

### Blueprint 2: `src/data/reversal.py`

#### Edit 2.1: Import Unified `InstanceRecord` and `SimpleDataFrame`
**Target File:** `src/data/reversal.py`  
**At line 26 (Imports):**
```python
from .loader import InstanceRecord, SimpleDataFrame
```
**Delete lines ~204–265** (the duplicate `class InstanceRecord:` definition).

#### Edit 2.2: Add Duck-Typing Properties to `ReciprocalPair`
**Target File:** `src/data/reversal.py`  
**Inside `class ReciprocalPair(tuple):` (~line 270):**
```python
    @property
    def first_id(self) -> str:
        inst = self.direct_instance
        return inst.get("instance_id") if hasattr(inst, "get") else getattr(inst, "instance_id", str(inst))

    @property
    def second_id(self) -> str:
        inst = self.inverse_instance
        return inst.get("instance_id") if hasattr(inst, "get") else getattr(inst, "instance_id", str(inst))

    @property
    def first_label(self) -> Optional[str]:
        inst = self.direct_instance
        return inst.get("label") if hasattr(inst, "get") else getattr(inst, "label", None)

    @property
    def second_label(self) -> Optional[str]:
        inst = self.inverse_instance
        return inst.get("label") if hasattr(inst, "get") else getattr(inst, "label", None)
```

#### Edit 2.3: Fix `pair_instances` (Strict Reversal, `NEGATIVE_OTHER` Reversibility, Decoupled Passes)
**Target File:** `src/data/reversal.py`  
**Replace `pair_instances` (~lines 304–370):**
```python
def pair_instances(
    records: Union[Sequence[Dict[str, Any]], Sequence[InstanceRecord], Any],
    strict_reversal: bool = False,
) -> List[ReciprocalPair]:
    """
    Pairs instances (A, B) and (B, A) by matching reverse_pair_id or pair_id.

    Args:
        records: Collection of records (dicts, InstanceRecord, or DataFrame).
        strict_reversal: If True, raises ValueError if paired labels violate Rev_BSA.

    Returns:
        List of ReciprocalPair objects (unpackable as (fwd, rev)).
    """
    if hasattr(records, "to_dict") and callable(getattr(records, "to_dict")):
        try:
            records = records.to_dict("records")
        except (TypeError, ValueError):
            pass

    records_norm = [
        r.to_dict() if hasattr(r, "to_dict") and callable(getattr(r, "to_dict")) else r
        for r in records
    ]

    by_id: Dict[str, Any] = {}
    for r in records_norm:
        inst_id = r["instance_id"] if hasattr(r, "__getitem__") else getattr(r, "instance_id")
        by_id[inst_id] = r

    seen_keys: Set[frozenset] = set()
    paired_ids: Set[str] = set()
    pairs: List[ReciprocalPair] = []

    # Pass 1: Match by explicit reverse_pair_id pointer
    for r in records_norm:
        inst_id = r["instance_id"] if hasattr(r, "__getitem__") else getattr(r, "instance_id")
        rev_id = r.get("reverse_pair_id") if hasattr(r, "get") else getattr(r, "reverse_pair_id", None)

        if rev_id and rev_id in by_id:
            key = canonical_pair_key(inst_id, rev_id)
            if key not in seen_keys:
                seen_keys.add(key)
                counterpart = by_id[rev_id]
                paired_ids.add(inst_id)
                paired_ids.add(rev_id)

                lbl1 = r.get("label") if hasattr(r, "get") else getattr(r, "label", None)
                lbl2 = counterpart.get("label") if hasattr(counterpart, "get") else getattr(counterpart, "label", None)
                is_rev = bool(is_reversible_label(lbl1) and is_reversible_label(lbl2)) if (lbl1 and lbl2) else False

                if strict_reversal and lbl1 is not None and lbl2 is not None:
                    expected_lbl2 = reverse_label_bsa(lbl1)
                    if lbl2 != expected_lbl2:
                        raise ValueError(
                            f"Strict reversal violation for pair ({inst_id}, {rev_id}): "
                            f"label1={lbl1}, label2={lbl2}, expected={expected_lbl2}"
                        )

                p_id = r.get("pair_id") if hasattr(r, "get") else getattr(r, "pair_id", None)
                pairs.append(ReciprocalPair(r, counterpart, pair_id=p_id, is_reversible=is_rev))

    # Pass 2: Match remaining unpaired instances by pair_id
    grouped_by_pair: Dict[str, List[Any]] = {}
    for r in records_norm:
        inst_id = r["instance_id"] if hasattr(r, "__getitem__") else getattr(r, "instance_id")
        if inst_id in paired_ids:
            continue
        p_id = r.get("pair_id") if hasattr(r, "get") else getattr(r, "pair_id", "")
        if p_id:
            grouped_by_pair.setdefault(p_id, []).append(r)

    for p_id, group in grouped_by_pair.items():
        if len(group) == 2:
            r1, r2 = group[0], group[1]
            id1 = r1["instance_id"] if hasattr(r1, "__getitem__") else getattr(r1, "instance_id")
            id2 = r2["instance_id"] if hasattr(r2, "__getitem__") else getattr(r2, "instance_id")
            key = canonical_pair_key(id1, id2)
            if key not in seen_keys:
                seen_keys.add(key)
                paired_ids.add(id1)
                paired_ids.add(id2)

                lbl1 = r1.get("label") if hasattr(r1, "get") else getattr(r1, "label", None)
                lbl2 = r2.get("label") if hasattr(r2, "get") else getattr(r2, "label", None)
                is_rev = bool(is_reversible_label(lbl1) and is_reversible_label(lbl2)) if (lbl1 and lbl2) else False

                if strict_reversal and lbl1 is not None and lbl2 is not None:
                    expected_lbl2 = reverse_label_bsa(lbl1)
                    if lbl2 != expected_lbl2:
                        raise ValueError(
                            f"Strict reversal violation for pair ({id1}, {id2}): "
                            f"label1={lbl1}, label2={lbl2}, expected={expected_lbl2}"
                        )

                if "__flipped" in id1 and "__original" in id2:
                    pairs.append(ReciprocalPair(r2, r1, pair_id=p_id, is_reversible=is_rev))
                else:
                    pairs.append(ReciprocalPair(r1, r2, pair_id=p_id, is_reversible=is_rev))

    return pairs
```

#### Edit 2.4: Fix `build_reversal_dataset`
**Target File:** `src/data/reversal.py`  
**Replace `build_reversal_dataset` (~lines 376–435):**
```python
def build_reversal_dataset(
    records: Union[Sequence[Dict[str, Any]], Sequence[InstanceRecord], Any],
    augment_bsa: bool = False,
    return_df: bool = False,
) -> Union[List[Dict[str, Any]], Any]:
    """
    Augments dataset with transposed pairs and inverted labels if augment_bsa=True.

    Args:
        records: Input dataset records (list of dicts, InstanceRecords, or DataFrame).
        augment_bsa: If True, generates inverted pairs. If False, returns original records.
        return_df: If True, returns a pd.DataFrame (or SimpleDataFrame).

    Returns:
        Augmented dataset as list of dicts (or DataFrame).
    """
    input_is_df = False
    if hasattr(records, "to_dict") and callable(getattr(records, "to_dict")):
        try:
            records_list = records.to_dict("records")
            input_is_df = True
        except (TypeError, ValueError):
            records_list = None
    else:
        records_list = None

    if records_list is None:
        records_list = [
            r.to_dict() if hasattr(r, "to_dict") and callable(getattr(r, "to_dict")) else dict(r)
            for r in records
        ]

    if not augment_bsa:
        if return_df:
            if input_is_df:
                return records
            if HAS_PANDAS and pd is not None:
                return pd.DataFrame(records_list)
            return SimpleDataFrame(records_list)
        return list(records_list)

    augmented: List[Dict[str, Any]] = list(records_list)
    existing_ids = {r["instance_id"] for r in records_list}

    for r in records_list:
        synth_id = f"{r['instance_id']}__bsa_rev"
        if synth_id in existing_ids:
            continue

        raw_label = r.get("label")
        inv_label = reverse_label(raw_label, allow_negative_other=True) if raw_label else None

        inverted = {
            "instance_id": synth_id,
            "pair_id": r.get("pair_id", "synth"),
            "text1_lang": r.get("text2_lang", r.get("text1_lang", "en")),
            "text2_lang": r.get("text1_lang", r.get("text2_lang", "en")),
            "text1": r["text2"],
            "text2": r["text1"],
            "label": inv_label,
            "category": r.get("category", "nominal"),
            "reverse_pair_id": r["instance_id"],
        }
        augmented.append(inverted)

    if return_df:
        if HAS_PANDAS and pd is not None:
            return pd.DataFrame(augmented)
        return SimpleDataFrame(augmented)

    return augmented
```

---

### Blueprint 3: `src/data/__init__.py`

#### Edit 3.1: Clean Unified Export of `InstanceRecord`
**Target File:** `src/data/__init__.py`  
**Lines 6–27:**
```python
from .loader import (
    ALL_CATEGORIES,
    ALL_LABELS as LOADER_ALL_LABELS,
    DatasetNotFoundError,
    DiCoDataError,
    DiCoLabel,
    InstanceRecord,
    REVERSIBLE_LABELS as LOADER_REVERSIBLE_LABELS,
    SchemaValidationError,
    SimpleDataFrame,
    SimpleSeries,
    SyntacticCategory,
    extract_syntactic_category,
    find_phrasis_file,
    generate_mock_phrasis_data,
    load_phrasis_split,
    normalize_split,
    normalize_track,
    read_phrasis_csv,
    save_mock_dataset,
    verify_split_leakage,
)
```
*(Remove `InstanceRecord as LoaderInstanceRecord` — import `InstanceRecord` directly).*

---

### Blueprint 4: `src/metrics/metrics.py`

#### Edit 4.1: Defensive Handling in `compute_official_metrics`
**Target File:** `src/metrics/metrics.py`  
**At line ~408:**
```python
    if not (
        len(y_true_dir) == len(y_pred_dir) == len(y_true_rev) == len(y_pred_rev) == len(is_reversible_mask)
    ):
        raise ValueError("Length mismatch among input sequences to compute_official_metrics")

    all_gold = list(y_true_dir) + list(y_true_rev)
    all_pred = list(y_pred_dir) + list(y_pred_rev)
    weighted_f1 = compute_weighted_f1(all_gold, all_pred)

    pairs: List[ReversiblePair] = []
    pred_map: Dict[str, str] = {}
    pair_idx = 0
    for g_d, p_d, g_r, p_r, is_rev in zip(
        y_true_dir, y_pred_dir, y_true_rev, y_pred_rev, is_reversible_mask
    ):
        if is_rev:
            id_d = f"inst_{pair_idx}_dir"
            id_r = f"inst_{pair_idx}_rev"
            pairs.append(ReversiblePair(id_d, id_r, g_d, g_r))
            pred_map[id_d] = p_d
            pred_map[id_r] = p_r
            pair_idx += 1

    if pairs:
        cons_report = compute_consistency_report(pairs, pred_map)
        soft_cons = cons_report.soft_cons
        hard_cons = cons_report.hard_cons
    else:
        soft_cons = 1.0  # Vacuously consistent when no reversible pairs exist
        hard_cons = 1.0

    return {
        "weighted_f1": weighted_f1,
        "soft_cons": soft_cons,
        "hard_cons": hard_cons,
    }
```

---

### Blueprint 5: `tests/test_pipeline.py`

#### Edit 5.1: Decouple M1 Imports for Live Source Verification
**Target File:** `tests/test_pipeline.py`  
**Lines 23–35:**
```python
# Import data and metrics directly from live source
try:
    from src.data.codabench import export_codabench_submission, validate_codabench_submission
    from src.data.reversal import build_reversal_dataset, pair_instances, reverse_label
    from src.metrics.metrics import compute_official_metrics
    SRC_AVAILABLE = True
except ImportError:
    SRC_AVAILABLE = False
    from tests.test_codabench import export_codabench_submission, validate_codabench_submission
    from tests.test_metrics import compute_official_metrics
    from tests.test_reversal_logic import build_reversal_dataset, pair_instances, reverse_label

# M2 deliverable: fallback to local test oracle until M2 implementation
try:
    from src.models.decoding import SymmetrizedDecoder
except ImportError:
    from tests.test_symmetrized_decoding import SymmetrizedDecoder
```

---

## 5. Verification Commands for Worker & Reviewers

1. **Verify DataFrame & `InstanceRecord` Ingestion in `build_reversal_dataset`:**
   ```bash
   python3 -c "
   from src.data.loader import load_phrasis_split, read_phrasis_csv
   from src.data.reversal import build_reversal_dataset
   df = load_phrasis_split('track1', 'train')
   orig = build_reversal_dataset(df, augment_bsa=False)
   assert len(orig) == len(df), f'Expected {len(df)}, got {len(orig)}'
   assert isinstance(orig[0], dict), 'Expected dict row'
   aug = build_reversal_dataset(df, augment_bsa=True)
   assert len(aug) > len(df), 'Augmented must have more rows'
   assert isinstance(aug[0], dict)
   print('DataFrame build_reversal_dataset test: PASSED')
   "
   ```

2. **Verify `InstanceRecord` Unification and dict Protocol:**
   ```bash
   python3 -c "
   from src.data.loader import InstanceRecord as LRecord
   from src.data.reversal import InstanceRecord as RRecord
   assert LRecord is RRecord, 'InstanceRecord must be unified single source of truth'
   rec = LRecord('id1', 'p1', text1='hello', text2='world', label='FORWARD_ENTAILMENT')
   assert dict(rec)['text1'] == 'hello'
   assert rec.to_dict()['text2'] == 'world'
   trans = rec.transposed()
   assert trans.label == 'BACKWARD_ENTAILMENT'
   print('InstanceRecord unification test: PASSED')
   "
   ```

3. **Verify `ReciprocalPair.is_reversible` on `NEGATIVE_OTHER` & Metric Safety:**
   ```bash
   python3 -c "
   from src.data.reversal import pair_instances
   from src.metrics.metrics import compute_official_metrics
   records = [
       {'instance_id': 'n1', 'pair_id': 'p1', 'reverse_pair_id': 'n2', 'label': 'NEGATIVE_OTHER'},
       {'instance_id': 'n2', 'pair_id': 'p1', 'reverse_pair_id': 'n1', 'label': 'NEGATIVE_OTHER'},
   ]
   pairs = pair_instances(records)
   assert pairs[0].is_reversible is False, 'NEGATIVE_OTHER must have is_reversible=False'
   m = compute_official_metrics(['NEGATIVE_OTHER'], ['NEGATIVE_OTHER'], ['NEGATIVE_OTHER'], ['NEGATIVE_OTHER'], [pairs[0].is_reversible])
   assert m['weighted_f1'] == 1.0
   assert m['soft_cons'] == 1.0
   assert m['hard_cons'] == 1.0
   print('NEGATIVE_OTHER invariant test: PASSED')
   "
   ```

4. **Verify `strict_reversal` Rejection & Mixed Dataset Pairing:**
   ```bash
   python3 -c "
   from src.data.reversal import pair_instances
   # 1. Contradiction rejection
   bad = [
       {'instance_id': '1', 'pair_id': 'p1', 'reverse_pair_id': '2', 'label': 'FORWARD_ENTAILMENT'},
       {'instance_id': '2', 'pair_id': 'p1', 'reverse_pair_id': '1', 'label': 'FORWARD_ENTAILMENT'},
   ]
   try:
       pair_instances(bad, strict_reversal=True)
       raise AssertionError('Should have raised ValueError')
   except ValueError:
       pass
   # 2. Mixed pairing
   mixed = [
       {'instance_id': 'id1', 'pair_id': 'p1', 'reverse_pair_id': 'id2', 'label': 'EQUIVALENCE'},
       {'instance_id': 'id2', 'pair_id': 'p1', 'reverse_pair_id': 'id1', 'label': 'EQUIVALENCE'},
       {'instance_id': 'id3__original', 'pair_id': 'p2', 'reverse_pair_id': None, 'label': 'EQUIVALENCE'},
       {'instance_id': 'id3__flipped', 'pair_id': 'p2', 'reverse_pair_id': None, 'label': 'EQUIVALENCE'},
   ]
   p = pair_instances(mixed)
   assert len(p) == 2
   print('Strict reversal and mixed pairing tests: PASSED')
   "
   ```

5. **Verify `extract_syntactic_category` Nominal Fix on `-ly` Words:**
   ```bash
   python3 -c "
   from src.data.loader import extract_syntactic_category
   assert extract_syntactic_category('a royal family', 'a noble house', 'en') == 'nominal'
   assert extract_syntactic_category('a monarch butterfly', 'an insect', 'en') == 'nominal'
   assert extract_syntactic_category('very quickly', 'extremely fast', 'en') == 'adverbial'
   print('Syntactic category test: PASSED')
   "
   ```

6. **Verify `SRC_AVAILABLE: True` across ALL Tests:**
   ```bash
   python3 -c "
   import tests.test_codabench as tc
   import tests.test_metrics as tm
   import tests.test_reversal_logic as tr
   import tests.test_pipeline as tp
   assert tc.SRC_AVAILABLE and tm.SRC_AVAILABLE and tr.SRC_AVAILABLE and tp.SRC_AVAILABLE, 'All tests must import from src'
   print('ALL TEST MODULES IMPORT FROM SRC: TRUE')
   "
   ```

7. **Run Automated Test Suite:**
   ```bash
   python3 -m unittest discover -s tests -p "test_*.py" -v
   ```
