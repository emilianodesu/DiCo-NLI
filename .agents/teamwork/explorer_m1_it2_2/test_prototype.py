"""
Prototype test script for Reversal Pairing & Reciprocal Invariants.
Validates the proposed changes to ReciprocalPair and pair_instances.
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class RelationLabel(str, Enum):
    BACKWARD_ENTAILMENT = "BACKWARD_ENTAILMENT"
    EQUIVALENCE = "EQUIVALENCE"
    FORWARD_ENTAILMENT = "FORWARD_ENTAILMENT"
    NEGATIVE_OTHER = "NEGATIVE_OTHER"

REVERSIBLE_LABELS: Set[str] = {
    RelationLabel.EQUIVALENCE.value,
    RelationLabel.FORWARD_ENTAILMENT.value,
    RelationLabel.BACKWARD_ENTAILMENT.value,
}
ALL_LABELS: Set[str] = REVERSIBLE_LABELS | {RelationLabel.NEGATIVE_OTHER.value}

def canonical_pair_key(id1: str, id2: str) -> frozenset:
    return frozenset({id1, id2})

def is_reversible_label(label: Any) -> bool:
    if label is None:
        return False
    val = label.value if isinstance(label, RelationLabel) else str(label).strip()
    return val in REVERSIBLE_LABELS

_REVERSE_LABEL_MAP_EVAL: Dict[str, str] = {
    RelationLabel.EQUIVALENCE.value: RelationLabel.EQUIVALENCE.value,
    RelationLabel.FORWARD_ENTAILMENT.value: RelationLabel.BACKWARD_ENTAILMENT.value,
    RelationLabel.BACKWARD_ENTAILMENT.value: RelationLabel.FORWARD_ENTAILMENT.value,
}

_REVERSE_LABEL_MAP_BSA: Dict[str, str] = {
    **_REVERSE_LABEL_MAP_EVAL,
    RelationLabel.NEGATIVE_OTHER.value: RelationLabel.NEGATIVE_OTHER.value,
}

def reverse_label_bsa(label: Union[str, RelationLabel]) -> str:
    if label is None:
        raise TypeError("Label cannot be None.")
    val = label.value if isinstance(label, RelationLabel) else str(label).strip()
    if val not in ALL_LABELS:
        raise ValueError(f"Invalid label '{val}'. Must be one of {sorted(ALL_LABELS)}")
    return _REVERSE_LABEL_MAP_BSA[val]

def _extract_label(item: Any) -> Optional[str]:
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
    if isinstance(record, dict) or hasattr(record, "__getitem__"):
        try:
            val = record.get(field, default) if hasattr(record, "get") else record[field]
            if val is not None:
                return val
        except (KeyError, TypeError, IndexError):
            pass
    val = getattr(record, field, default)
    return val if val is not None else default


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

        p_id = pair_id or _get_field(direct_instance, "pair_id", "") or _get_field(inverse_instance, "pair_id", "")
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
            is_consistent = (reverse_label_bsa(lbl_dir) == lbl_inv) if (lbl_dir in ALL_LABELS and lbl_inv in ALL_LABELS) else False
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


def pair_instances(
    records: Union[Sequence[Dict[str, Any]], Any],
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
    if hasattr(records, "to_dict") and callable(getattr(records, "to_dict")):
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
