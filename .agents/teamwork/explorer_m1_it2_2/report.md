# Technical Investigation Report: Reversal Pairing & Reciprocal Invariants

- **Author:** Explorer 2 (`explorer_m1_it2_2`)
- **Archetype / Roles:** Explorer, Investigator, Blueprint Designer
- **Milestone:** Milestone 1 Iteration 2 (M1: Data Pipeline & Consistency Metrics)
- **Target File:** `src/data/reversal.py`
- **Reviewed Sources:** `PROJECT.md`, `src/data/reversal.py`, `src/metrics/metrics.py`, `tests/test_reversal_logic.py`, `tests/test_metrics.py`, Reviewer 1/2 handoffs, Challenger 1 handoff
- **Date:** 2026-11-01T22:24:00Z

---

## 1. Executive Summary

During Milestone 1 Iteration 1 review, Challenger 1 and Reviewers 1 & 2 identified critical defects in `src/data/reversal.py` concerning reciprocal pairing, label consistency validation, and mathematical metric bounds:
1. **Mathematical Invariant Breakdown ($HardCons > SoftCons$):** `ReciprocalPair.is_reversible` was hardcoded or defaulted to `True` for all pairs, including pairs with ground truth `NEGATIVE_OTHER`. When evaluated on `NEGATIVE_OTHER` pairs with correct predictions, `SoftCons` evaluated to `0.0` (because `NEGATIVE_OTHER` is excluded from the closed reversible set $\mathcal{R}$), while `HardCons` evaluated to `1.0`, triggering an unhandled `AssertionError` in `compute_consistency_report`.
2. **Dead Parameter (`strict_reversal`):** `pair_instances(records, strict_reversal=False)` documented `strict_reversal: If True, raises ValueError if paired labels violate Rev_BSA`, but completely ignored the parameter in its body. Contradictory pairs (e.g. `FORWARD_ENTAILMENT` paired with `FORWARD_ENTAILMENT`) were accepted silently.
3. **Data Loss in Mixed Datasets:** `pair_instances` gated its secondary pairing pass on `if not pairs:`. If even a single pair had `reverse_pair_id`, the fallback to `pair_id` was skipped entirely, silently dropping all instances that relied on `pair_id` grouping.
4. **Interface Duck-Typing Incompatibility:** Passing `ReciprocalPair` objects directly into `compute_consistency_report` crashed with `AttributeError: 'ReciprocalPair' object has no attribute 'first_id'`.

This report provides the mathematical analysis, complete empirical reproduction, and an exact drop-in code patch blueprint for Worker M1.

---

## 2. Technical Investigation & Root Cause Analysis

### 2.1 Defect 1: `ReciprocalPair.is_reversible` and Invariant Violation

#### Observation & Reproduction
In `src/data/reversal.py:285, 345, 363-367`:
```python
class ReciprocalPair(tuple):
    def __new__(
        cls,
        direct_instance: Any,
        inverse_instance: Any,
        pair_id: Optional[str] = None,
        is_synthetic: bool = False,
        is_reversible: bool = True,   # <-- Default True unconditionally
    ):
        ...
        instance.is_reversible = is_reversible
        return instance
```
In `pair_instances`:
```python
pairs.append(ReciprocalPair(r, counterpart))  # is_reversible not passed -> True
```
When executing:
```python
from src.data.reversal import pair_instances
from src.metrics.metrics import compute_official_metrics

records = [
    {'instance_id': 'neg_1', 'pair_id': 'p1', 'reverse_pair_id': 'neg_2', 'label': 'NEGATIVE_OTHER'},
    {'instance_id': 'neg_2', 'pair_id': 'p1', 'reverse_pair_id': 'neg_1', 'label': 'NEGATIVE_OTHER'},
]
pairs = pair_instances(records)
compute_official_metrics(
    ['NEGATIVE_OTHER'], ['NEGATIVE_OTHER'],
    ['NEGATIVE_OTHER'], ['NEGATIVE_OTHER'],
    [pairs[0].is_reversible]  # True
)
```
**Verbatim Traceback:**
```
Traceback (most recent call last):
  File "<string>", line 9, in <module>
    compute_official_metrics(...)
  File "/src/metrics/metrics.py", line 419, in compute_official_metrics
    cons_report = compute_consistency_report(pairs, pred_map)
  File "/src/metrics/metrics.py", line 377, in compute_consistency_report
    assert hard_cons <= soft_cons + 1e-12, (
AssertionError: Mathematical invariant violated: HardCons (1.0) > SoftCons (0.0)
```

#### Mathematical Logic Chain
1. In SemEval-2027 Task 2 (DiCo-NLI), the label inventory is $\mathcal{L} = \{\text{EQUIVALENCE}, \text{FORWARD\_ENTAILMENT}, \text{BACKWARD\_ENTAILMENT}, \text{NEGATIVE\_OTHER}\}$.
2. The closed reversible subset is:
   $$\mathcal{R} = \{\text{EQUIVALENCE}, \text{FORWARD\_ENTAILMENT}, \text{BACKWARD\_ENTAILMENT}\}$$
3. By definition, $\text{NEGATIVE\_OTHER} \notin \mathcal{R}$.
4. The definition of Soft Consistency ($SoftCons$) over predicted labels $(\hat{y}_1, \hat{y}_2)$ is:
   $$\text{soft\_ok} = \mathbb{I}(\hat{y}_1 \in \mathcal{R} \land \hat{y}_2 = \text{Rev}(\hat{y}_1))$$
   Note that if $\hat{y}_1 = \text{NEGATIVE\_OTHER}$, $\hat{y}_1 \notin \mathcal{R}$, so $\text{soft\_ok} = 0$.
5. The definition of Hard Consistency ($HardCons$) against ground truth $(y_1, y_2)$ is:
   $$\text{hard\_ok} = \mathbb{I}(\hat{y}_1 = y_1 \land \hat{y}_2 = y_2)$$
6. **The Fundamental Invariant Theorem:**
   If $(y_1, y_2)$ is a valid reversible pair with $y_1 \in \mathcal{R}$ and $y_2 = \text{Rev}(y_1)$, then whenever $\text{hard\_ok} = 1$, we have $\hat{y}_1 = y_1 \in \mathcal{R}$ and $\hat{y}_2 = y_2 = \text{Rev}(y_1) = \text{Rev}(\hat{y}_1)$, which implies $\text{soft\_ok} = 1$. Therefore:
   $$\forall \text{ predictions } (\hat{y}_1, \hat{y}_2), \quad HardCons \le SoftCons$$
7. **The Breakdown Condition:**
   If a pair with ground truth $y_1 = y_2 = \text{NEGATIVE\_OTHER}$ is evaluated under consistency (i.e. `is_reversible = True`), and the model predicts $\hat{y}_1 = \hat{y}_2 = \text{NEGATIVE\_OTHER}$:
   - $\text{hard\_ok} = 1$ (perfect joint prediction matching ground truth).
   - $\text{soft\_ok} = 0$ (because $\text{NEGATIVE\_OTHER} \notin \mathcal{R}$).
   - Therefore, $HardCons = 1.0 > SoftCons = 0.0$, violating the fundamental theorem.
8. **Conclusion:** Any reciprocal pair whose ground truth is `NEGATIVE_OTHER` (or non-reversible) **MUST** have `is_reversible = False`.

---

### 2.2 Defect 2: Dead Parameter `strict_reversal` in `pair_instances`

#### Observation & Reproduction
In `src/data/reversal.py:304-370`:
```python
def pair_instances(
    records: Union[Sequence[Dict[str, Any]], Sequence[InstanceRecord], Any],
    strict_reversal: bool = False,
) -> List[ReciprocalPair]:
```
Searching for occurrences of `strict_reversal` in `src/data/reversal.py` reveals that it appears only in line 306 (parameter declaration) and line 313 (docstring). It is never referenced anywhere in the function body.

Executing:
```python
from src.data.reversal import pair_instances
records = [
    {'instance_id': 'id1', 'pair_id': 'p1', 'reverse_pair_id': 'id2', 'label': 'FORWARD_ENTAILMENT'},
    {'instance_id': 'id2', 'pair_id': 'p1', 'reverse_pair_id': 'id1', 'label': 'FORWARD_ENTAILMENT'},
]
pairs = pair_instances(records, strict_reversal=True)
print("Returned pairs:", len(pairs))
```
**Output observed:**
```
Returned pairs: 1
```
The function accepted a pair where $(A, B)$ is `FORWARD_ENTAILMENT` and $(B, A)$ is also `FORWARD_ENTAILMENT`, which violates the involution algebra $\text{Rev}_{\text{BSA}}(\text{FORWARD\_ENTAILMENT}) = \text{BACKWARD\_ENTAILMENT}$.

#### Remediation Logic
When `strict_reversal=True`:
1. If both instances in a pair have labels $y_1$ and $y_2$:
   - Compute $y_2^* = \text{reverse\_label\_bsa}(y_1)$.
   - If $y_2 \ne y_2^*$, immediately raise `ValueError` describing the contradiction and identifying instance IDs.
2. If one instance is labeled and the counterpart is unlabeled, raise `ValueError`.
3. If either label is invalid (not in `ALL_LABELS`), raise `ValueError`.

When `strict_reversal=False`:
1. Do not raise `ValueError`.
2. Log a warning describing the contradictory labels.
3. Automatically set `is_reversible = False` on the resulting `ReciprocalPair`, because self-contradictory ground truth cannot satisfy reversible consistency.

---

### 2.3 Defect 3: Fragile `if not pairs:` Multi-Pass Gating

#### Observation & Reproduction
In `src/data/reversal.py:339-354`:
```python
    for r in records:
        ...
        if rev_id and rev_id in by_id:
            ...
            pairs.append(ReciprocalPair(r, counterpart))

    # 2. Fallback: match by pair_id if reverse_pair_id was absent
    if not pairs:
        grouped_by_pair: Dict[str, List[Any]] = {}
        for r in records:
            p_id = r.get("pair_id")
            grouped_by_pair.setdefault(p_id, []).append(r)
        ...
```
Executing:
```python
from src.data.reversal import pair_instances
records = [
    {'instance_id': 'id1', 'pair_id': 'p1', 'reverse_pair_id': 'id2', 'label': 'EQUIVALENCE'},
    {'instance_id': 'id2', 'pair_id': 'p1', 'reverse_pair_id': 'id1', 'label': 'EQUIVALENCE'},
    {'instance_id': 'id3__original', 'pair_id': 'p2', 'reverse_pair_id': None, 'label': 'EQUIVALENCE'},
    {'instance_id': 'id3__flipped', 'pair_id': 'p2', 'reverse_pair_id': None, 'label': 'EQUIVALENCE'},
]
pairs = pair_instances(records)
print("Found pairs:", len(pairs))
```
**Output observed:**
```
Found pairs: 1
```
Because `pairs` was non-empty after Pass 1, `if not pairs:` evaluated to `False`. The second pair (`id3__original`, `id3__flipped`) was never grouped or paired, resulting in a 50% data loss.

#### Remediation Logic
Eliminate `if not pairs:`. Replace with a decoupled multi-pass pipeline tracking `seen_keys` and `paired_inst_ids`:
1. **Pass 1 (Explicit Pointer):** Match records via `reverse_pair_id`. Add matched instances to `paired_inst_ids`.
2. **Pass 2 (Pair ID Fallback):** For all remaining instances not in `paired_inst_ids`, group by `pair_id`.
   - If a group has 2 instances, pair them.
   - If a group has >2 instances, pair by matching transposed texts `text1 == other.text2 and text2 == other.text1`.
3. **Pass 3 (Transposed Text Fallback):** For any remaining unpaired instances across the dataset, pair by exact text transposition.

---

### 2.4 Defect 4: Duck-Typing Incompatibility with `compute_consistency_report`

#### Observation & Reproduction
In `src/metrics/metrics.py:343-356`:
```python
for pair in pairs:
    pred_1 = predicted_labels[pair.first_id]
    pred_2 = predicted_labels[pair.second_id]
    hard_ok = (pred_1 == pair.first_label and pred_2 == pair.second_label)
```
When `pairs` is a list of `ReciprocalPair`:
```python
compute_consistency_report(pairs, preds)
```
**Verbatim Traceback:**
```
AttributeError: 'ReciprocalPair' object has no attribute 'first_id'
```

#### Remediation Logic
Add four properties to `ReciprocalPair`:
- `first_id`: returns `instance_id` of `direct_instance`.
- `second_id`: returns `instance_id` of `inverse_instance`.
- `first_label`: returns `label` of `direct_instance`.
- `second_label`: returns `label` of `inverse_instance`.

This allows `ReciprocalPair` to act as a 100% drop-in replacement for `ReversiblePair` anywhere in `src/metrics/`.

---

## 3. Exact Code Patch Blueprint for Worker

The following blueprint contains the exact code to replace in `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/src/data/reversal.py`.

### 3.1 Helper Functions (Insert before `ReciprocalPair`, around line 265)

```python
def _extract_label(item: Any) -> Optional[str]:
    """Extract string label from dictionary, InstanceRecord, or Enum safely."""
    if item is None:
        return None
    lbl = None
    if isinstance(item, dict) or hasattr(item, "__getitem__"):
        try:
            lbl = item.get("label") if hasattr(item, "get") else item["label"]
        except (KeyError, TypeError, IndexError):
            pass
    if lbl is None:
        lbl = getattr(item, "label", None)
    if isinstance(lbl, Enum):
        lbl = lbl.value
    if lbl is not None:
        lbl = str(lbl).strip()
    return lbl


def _get_field(record: Any, field: str, default: Any = None) -> Any:
    """Safely retrieves a field from dict, dataclass, or object."""
    if isinstance(record, dict) or hasattr(record, "__getitem__"):
        try:
            val = record.get(field, default) if hasattr(record, "get") else record[field]
            if val is not None:
                return val
        except (KeyError, TypeError, IndexError):
            pass
    val = getattr(record, field, default)
    return val if val is not None else default


def _canonical_orientation(r1: Any, r2: Any, id1: str, id2: str) -> Tuple[Any, Any]:
    """Ensures direct (forward) instance is first and inverse (reverse) is second."""
    if "__flipped" in id1 and "__original" in id2:
        return (r2, r1)
    if "__original" in id1 and "__flipped" in id2:
        return (r1, r2)
    if "__bsa_rev" in id1 and "__bsa_rev" not in id2:
        return (r2, r1)
    if "__bsa_rev" in id2 and "__bsa_rev" not in id1:
        return (r1, r2)
    rev_tokens = ("_rev", "_inverse", "_inv", "_flipped")
    if any(tok in id1.lower() for tok in rev_tokens) and not any(tok in id2.lower() for tok in rev_tokens):
        return (r2, r1)
    return (r1, r2)
```

### 3.2 Replacement for `ReciprocalPair` (Lines 267–298)

```python
class ReciprocalPair(tuple):
    """
    Represents a bound reciprocal pair: direct (A, B) and inverse (B, A).
    Inherits from tuple so that `for fwd, rev in pairs:` unpacking works natively,
    while also providing named attribute and duck-typing property access.
    """
    direct_instance: Any
    inverse_instance: Any
    pair_id: str
    is_synthetic: bool
    is_reversible: bool

    def __new__(
        cls,
        direct_instance: Any,
        inverse_instance: Any,
        pair_id: Optional[str] = None,
        is_synthetic: bool = False,
        is_reversible: Optional[bool] = None,
    ):
        instance = super().__new__(cls, (direct_instance, inverse_instance))
        instance.direct_instance = direct_instance
        instance.inverse_instance = inverse_instance

        p_id = (
            pair_id
            or _get_field(direct_instance, "pair_id", "")
            or _get_field(inverse_instance, "pair_id", "")
        )
        instance.pair_id = str(p_id) if p_id else ""
        instance.is_synthetic = bool(is_synthetic)

        # Invariant enforcement: NEGATIVE_OTHER is strictly NON-REVERSIBLE.
        # Consistency metrics (SoftCons, HardCons) are defined strictly over REVERSIBLE_LABELS.
        # If is_reversible were True for NEGATIVE_OTHER, HardCons could be 1.0 while
        # SoftCons is 0.0 (since SoftCons requires reversible predictions), violating HardCons <= SoftCons.
        lbl_dir = _extract_label(direct_instance)
        lbl_inv = _extract_label(inverse_instance)

        is_neg = (
            lbl_dir == RelationLabel.NEGATIVE_OTHER.value
            or lbl_inv == RelationLabel.NEGATIVE_OTHER.value
        )

        if is_neg:
            instance.is_reversible = False
        elif lbl_dir is not None and lbl_inv is not None:
            both_reversible = is_reversible_label(lbl_dir) and is_reversible_label(lbl_inv)
            # If labels are contradictory (e.g. FWD paired with FWD), cannot be consistently reversible
            is_consistent = (
                (reverse_label_bsa(lbl_dir) == lbl_inv)
                if (lbl_dir in ALL_LABELS and lbl_inv in ALL_LABELS)
                else False
            )
            computed = both_reversible and is_consistent
            if is_reversible is not None:
                instance.is_reversible = computed and bool(is_reversible)
            else:
                instance.is_reversible = computed
        elif is_reversible is not None:
            instance.is_reversible = bool(is_reversible)
        else:
            instance.is_reversible = True

        return instance

    @property
    def first_id(self) -> str:
        return str(_get_field(self.direct_instance, "instance_id", ""))

    @property
    def second_id(self) -> str:
        return str(_get_field(self.inverse_instance, "instance_id", ""))

    @property
    def first_label(self) -> Optional[str]:
        return _extract_label(self.direct_instance)

    @property
    def second_label(self) -> Optional[str]:
        return _extract_label(self.inverse_instance)

    def to_tuple(self) -> Tuple[Any, Any]:
        return (self.direct_instance, self.inverse_instance)
```

### 3.3 Replacement for `pair_instances` (Lines 304–370)

```python
def pair_instances(
    records: Union[Sequence[Dict[str, Any]], Sequence[InstanceRecord], Any],
    strict_reversal: bool = False,
) -> List[ReciprocalPair]:
    """
    Pairs instances (A, B) and (B, A) using a robust multi-pass pairing engine:
    1. Primary pass: Explicit reverse_pair_id matching.
    2. Fallback pass 1: Matching remaining unpaired instances by pair_id.
    3. Fallback pass 2: Matching remaining unpaired instances by transposed text pairs.

    Args:
        records: Collection of records (dicts, InstanceRecords, or DataFrame).
        strict_reversal: If True, raises ValueError if paired labels violate Rev_BSA.

    Returns:
        List of ReciprocalPair objects (unpackable as (fwd, rev)).
    """
    # Normalize records to list of dicts/records
    if HAS_PANDAS and pd is not None and isinstance(records, pd.DataFrame):
        records_list = records.to_dict("records")
    elif hasattr(records, "to_dict") and callable(getattr(records, "to_dict")):
        try:
            records_list = records.to_dict("records")
        except TypeError:
            records_list = list(records)
    elif isinstance(records, (list, tuple)):
        records_list = list(records)
    else:
        records_list = list(records)

    by_id: Dict[str, Any] = {}
    for r in records_list:
        inst_id = _get_field(r, "instance_id")
        if inst_id is not None:
            by_id[str(inst_id)] = r

    seen_keys: Set[frozenset] = set()
    paired_inst_ids: Set[str] = set()
    pairs: List[ReciprocalPair] = []

    def _try_add_pair(r1: Any, r2: Any, pair_id: Optional[str] = None) -> bool:
        id1 = str(_get_field(r1, "instance_id", ""))
        id2 = str(_get_field(r2, "instance_id", ""))
        if not id1 or not id2 or id1 == id2:
            return False

        key = canonical_pair_key(id1, id2)
        if key in seen_keys or id1 in paired_inst_ids or id2 in paired_inst_ids:
            return False

        lbl1 = _extract_label(r1)
        lbl2 = _extract_label(r2)

        # Active validation for strict_reversal
        if strict_reversal:
            if lbl1 is not None and lbl2 is not None:
                expected_lbl2 = reverse_label_bsa(lbl1)
                if lbl2 != expected_lbl2:
                    raise ValueError(
                        f"Contradictory reciprocal pair labels for ('{id1}', '{id2}'): "
                        f"'{lbl1}' paired with '{lbl2}' violates Rev_BSA (expected '{expected_lbl2}')."
                    )
            elif (lbl1 is None) != (lbl2 is None):
                raise ValueError(
                    f"Contradictory labeling in pair ('{id1}', '{id2}'): "
                    f"one instance has label '{lbl1 or lbl2}' while counterpart is unlabeled."
                )
        elif lbl1 is not None and lbl2 is not None:
            if lbl1 in ALL_LABELS and lbl2 in ALL_LABELS:
                if reverse_label_bsa(lbl1) != lbl2:
                    logger.warning(
                        "Reciprocal pair (%s, %s) has contradictory labels: %s vs %s",
                        id1, id2, lbl1, lbl2
                    )

        direct, inverse = _canonical_orientation(r1, r2, id1, id2)
        p_id = pair_id or _get_field(direct, "pair_id", "") or _get_field(inverse, "pair_id", "") or ""

        rec_pair = ReciprocalPair(direct, inverse, pair_id=p_id)
        pairs.append(rec_pair)
        seen_keys.add(key)
        paired_inst_ids.add(id1)
        paired_inst_ids.add(id2)
        return True

    # Pass 1: Explicit reverse_pair_id pointer
    for r in records_list:
        inst_id = str(_get_field(r, "instance_id", ""))
        if not inst_id or inst_id in paired_inst_ids:
            continue
        rev_id = _get_field(r, "reverse_pair_id")
        if rev_id and str(rev_id) in by_id:
            rev_id_str = str(rev_id)
            if rev_id_str not in paired_inst_ids:
                counterpart = by_id[rev_id_str]
                _try_add_pair(r, counterpart)

    # Pass 2: Fallback by pair_id for remaining unpaired records
    grouped_by_pair: Dict[str, List[Any]] = {}
    for r in records_list:
        inst_id = str(_get_field(r, "instance_id", ""))
        if not inst_id or inst_id in paired_inst_ids:
            continue
        p_id = _get_field(r, "pair_id")
        if p_id:
            grouped_by_pair.setdefault(str(p_id), []).append(r)

    for p_id, group in grouped_by_pair.items():
        if len(group) == 2:
            _try_add_pair(group[0], group[1], pair_id=p_id)
        elif len(group) > 2:
            # Multi-instance group: match by transposed texts
            unpaired = [r for r in group if str(_get_field(r, "instance_id", "")) not in paired_inst_ids]
            for i in range(len(unpaired)):
                r1 = unpaired[i]
                id1 = str(_get_field(r1, "instance_id", ""))
                if id1 in paired_inst_ids:
                    continue
                t1_1 = str(_get_field(r1, "text1", "")).strip()
                t1_2 = str(_get_field(r1, "text2", "")).strip()
                for j in range(i + 1, len(unpaired)):
                    r2 = unpaired[j]
                    id2 = str(_get_field(r2, "instance_id", ""))
                    if id2 in paired_inst_ids:
                        continue
                    t2_1 = str(_get_field(r2, "text1", "")).strip()
                    t2_2 = str(_get_field(r2, "text2", "")).strip()
                    if t1_1 == t2_2 and t1_2 == t2_1 and t1_1:
                        _try_add_pair(r1, r2, pair_id=p_id)
                        break

    # Pass 3: Fallback by transposed texts across any remaining unpaired records
    remaining = [r for r in records_list if str(_get_field(r, "instance_id", "")) not in paired_inst_ids]
    if len(remaining) >= 2:
        text_map: Dict[Tuple[str, str], List[Any]] = {}
        for r in remaining:
            t1 = str(_get_field(r, "text1", "")).strip()
            t2 = str(_get_field(r, "text2", "")).strip()
            if t1 and t2:
                text_map.setdefault((t1, t2), []).append(r)

        for (t1, t2), r_list in list(text_map.items()):
            inv_key = (t2, t1)
            if inv_key in text_map:
                inv_list = text_map[inv_key]
                while r_list and inv_list:
                    r1 = r_list.pop(0)
                    if r1 in inv_list:
                        inv_list.remove(r1)
                    if not inv_list:
                        break
                    r2 = inv_list.pop(0)
                    _try_add_pair(r1, r2)

    return pairs
```

---

## 4. Empirical Verification & Evidence

All proposed implementations were compiled into a standalone prototype and tested directly in the project environment (`.agents/teamwork/explorer_m1_it2_2/test_prototype.py` and `run_tests.py`):

1. **Test 1: `NEGATIVE_OTHER` is non-reversible:**
   - Pair with `label='NEGATIVE_OTHER'` generated `is_reversible=False`.
   - Evaluated in `compute_official_metrics` with correct predictions.
   - Result: `soft_cons=1.0`, `hard_cons=1.0`, `weighted_f1=1.0`, $HardCons \le SoftCons$ strictly preserved.
2. **Test 2: `strict_reversal` rejection:**
   - Contradictory pair (`FORWARD_ENTAILMENT` paired with `FORWARD_ENTAILMENT`) raised `ValueError: Contradictory reciprocal pair labels for ('id1', 'id2'): 'FORWARD_ENTAILMENT' paired with 'FORWARD_ENTAILMENT' violates Rev_BSA (expected 'BACKWARD_ENTAILMENT')`.
   - With `strict_reversal=False`, accepted the pair and marked `is_reversible=False`.
3. **Test 3: Mixed dataset pairing:**
   - Dataset with 2 records specifying `reverse_pair_id` and 2 records specifying only `pair_id` produced exactly 2 pairs (`p1` and `p2`), resolving the data loss bug.
4. **Test 4: Transposed text pairing:**
   - Records with no `reverse_pair_id` and no `pair_id` were successfully paired via text transposition `(text1, text2) == (other.text2, other.text1)`.
5. **Test 5: Duck-typing in `compute_consistency_report`:**
   - Output of `pair_instances` was passed directly into `compute_consistency_report(pairs, preds)`, returning `soft_cons=1.0, hard_cons=1.0` with zero `AttributeError`.
6. **Live Data Integration:**
   - Verified on live PhrasIS Track 1 train dataset (`load_phrasis_split('track1', 'train')`): found all 27 canonical pairs without errors.
   - Verified on BSA-augmented PhrasIS dataset: found all 59 reciprocal pairs; all 5 `NEGATIVE_OTHER` pairs had `is_reversible=False`.

---

## 5. Instructions for Worker M1

1. Apply the blueprint in Section 3 to `src/data/reversal.py`.
2. Add the unit tests from Section 4 to `tests/test_reversal_logic.py`.
3. Run the full test suite:
   ```bash
   python3 -m unittest discover -s tests -p "test_*.py" -v
   ```
4. Verify that `tests.test_reversal_logic` and `tests.test_metrics` pass with 100% success rate.
