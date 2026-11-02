"""
src/data/reversal.py
Deterministic Reversal Algebra, Reciprocal Pairing & Symmetrical Augmentation (BSA).
Conforms to SemEval-2027 Task 2 (DiCo-NLI) specification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)

from .loader import InstanceRecord, SimpleDataFrame

# Optional third-party integrations with graceful fallbacks
try:
    import pandas as pd  # type: ignore
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None  # type: ignore

try:
    import torch  # type: ignore
    from torch.utils.data import Dataset  # type: ignore
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class Dataset:  # type: ignore
        """Fallback lightweight Dataset base class when PyTorch is not installed."""
        def __len__(self) -> int:
            raise NotImplementedError
        def __getitem__(self, idx: int) -> Any:
            raise NotImplementedError


# ============================================================================
# 1. Official Label Enumerations & Algebra Constants
# ============================================================================

class RelationLabel(str, Enum):
    """Canonical 4-class label inventory for SemEval-2027 Task 2."""
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

# Class index encoding matching model heads and metric arrays:
# 0: BACKWARD_ENTAILMENT, 1: EQUIVALENCE, 2: FORWARD_ENTAILMENT, 3: NEGATIVE_OTHER
LABEL_TO_ID: Dict[str, int] = {
    RelationLabel.BACKWARD_ENTAILMENT.value: 0,
    RelationLabel.EQUIVALENCE.value: 1,
    RelationLabel.FORWARD_ENTAILMENT.value: 2,
    RelationLabel.NEGATIVE_OTHER.value: 3,
}

ID_TO_LABEL: Dict[int, str] = {idx: label for label, idx in LABEL_TO_ID.items()}
NUM_CLASSES: int = len(LABEL_TO_ID)

# Permutation Involution Matrix P such that P * [BWD, EQUIV, FWD, NEG]^T = [FWD, EQUIV, BWD, NEG]^T
INVOLUTION_MATRIX: List[List[int]] = [
    [0, 0, 1, 0],  # BWD -> FWD
    [0, 1, 0, 0],  # EQUIV -> EQUIV
    [1, 0, 0, 0],  # FWD -> BWD
    [0, 0, 0, 1],  # NEG -> NEG
]

_REVERSE_LABEL_MAP_EVAL: Dict[str, str] = {
    RelationLabel.EQUIVALENCE.value: RelationLabel.EQUIVALENCE.value,
    RelationLabel.FORWARD_ENTAILMENT.value: RelationLabel.BACKWARD_ENTAILMENT.value,
    RelationLabel.BACKWARD_ENTAILMENT.value: RelationLabel.FORWARD_ENTAILMENT.value,
}

_REVERSE_LABEL_MAP_BSA: Dict[str, str] = {
    **_REVERSE_LABEL_MAP_EVAL,
    RelationLabel.NEGATIVE_OTHER.value: RelationLabel.NEGATIVE_OTHER.value,
}


# ============================================================================
# 2. Canonical Functions & Reversal Algebra Operators
# ============================================================================

def canonical_pair_key(id1: str, id2: str) -> frozenset:
    """Return an unordered identifier set for a reciprocal pair."""
    return frozenset({id1, id2})


def reverse_label(
    label: Union[str, RelationLabel],
    allow_negative_other: bool = True,
    allow_negative: Optional[bool] = None,
    strict_eval: bool = False,
) -> str:
    """
    Applies the deterministic reversal operator to an NLI relation label.

    Args:
        label: Input relation label string or RelationLabel enum.
        allow_negative_other: If True, maps NEGATIVE_OTHER -> NEGATIVE_OTHER (BSA semantics).
                              If False, raises ValueError (official evaluation semantics).
        allow_negative: Alias for allow_negative_other if provided.
        strict_eval: If True, forces strict evaluation mode (raises ValueError on NEGATIVE_OTHER).

    Returns:
        The inverted label string.

    Raises:
        ValueError: If label is invalid or if NEGATIVE_OTHER is supplied under strict evaluation.
        TypeError: If label is None or not a string/enum.
    """
    if label is None:
        raise TypeError("Label cannot be None.")

    val = label.value if isinstance(label, RelationLabel) else str(label).strip()

    if not val or val not in ALL_LABELS:
        raise ValueError(f"Invalid label '{val}'. Must be one of {sorted(ALL_LABELS)}")

    # Resolve allow_negative_other flag
    effective_allow_neg = allow_negative_other
    if allow_negative is not None:
        effective_allow_neg = allow_negative
    if strict_eval:
        effective_allow_neg = False

    if val == RelationLabel.NEGATIVE_OTHER.value:
        if effective_allow_neg:
            return RelationLabel.NEGATIVE_OTHER.value
        raise ValueError("NEGATIVE_OTHER is non-reversible in strict evaluation mode")

    return _REVERSE_LABEL_MAP_EVAL[val]


def reverse_label_strict(label: Union[str, RelationLabel]) -> str:
    """Strict evaluation reversal: raises ValueError on NEGATIVE_OTHER."""
    return reverse_label(label, allow_negative_other=False, strict_eval=True)


def reverse_label_bsa(label: Union[str, RelationLabel]) -> str:
    """Data augmentation reversal: maps NEGATIVE_OTHER -> NEGATIVE_OTHER."""
    return reverse_label(label, allow_negative_other=True, strict_eval=False)


def is_reversible_label(label: Union[str, RelationLabel]) -> bool:
    """Returns True if the label belongs to the reversible subset R."""
    if label is None:
        return False
    val = label.value if isinstance(label, RelationLabel) else str(label).strip()
    return val in REVERSIBLE_LABELS


def verify_algebra_invariants() -> bool:
    """
    Formally validates the mathematical involution properties of the reversal algebra.
    Returns True if all invariants hold, raises AssertionError otherwise.
    """
    # 1. Evaluation involution: Rev(Rev(y)) == y for all y in R
    for lbl in REVERSIBLE_LABELS:
        assert reverse_label_strict(reverse_label_strict(lbl)) == lbl, f"Involution failed for {lbl}"

    # 2. BSA involution: Rev_BSA(Rev_BSA(y)) == y for all y in L
    for lbl in ALL_LABELS:
        assert reverse_label_bsa(reverse_label_bsa(lbl)) == lbl, f"BSA involution failed for {lbl}"

    # 3. Involution Matrix P properties
    p = INVOLUTION_MATRIX
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            assert p[i][j] == p[j][i], "Matrix P is not symmetric"
            dot = sum(p[i][k] * p[k][j] for k in range(NUM_CLASSES))
            expected = 1 if i == j else 0
            assert dot == expected, f"Matrix P^2 != I at ({i}, {j})"

    return True


# ============================================================================
# 3. Data Structures & Reciprocal Pairing Helpers
# ============================================================================

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


class ReciprocalPair(tuple):
    """
    Represents a bound reciprocal pair: direct (A, B) and inverse (B, A).
    Inherits from tuple so that `for fwd, rev in pairs:` unpacking works natively,
    while also providing named attribute access and duck-typing properties.
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


# ============================================================================
# 4. Reciprocal Pairing Engine
# ============================================================================

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
    if HAS_PANDAS and pd is not None and isinstance(records, pd.DataFrame):
        records_list = records.to_dict("records")
    elif hasattr(records, "to_dict") and callable(getattr(records, "to_dict")):
        try:
            records_list = records.to_dict("records")
        except (TypeError, ValueError):
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


# ============================================================================
# 5. Bi-directional Symmetrical Augmentation (BSA) Engine
# ============================================================================

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
        return_df: If True and pandas/SimpleDataFrame is available, returns a DataFrame.

    Returns:
        Augmented dataset as list of dicts (or DataFrame).
    """
    input_is_df = False
    records_list = None

    if HAS_PANDAS and pd is not None and isinstance(records, pd.DataFrame):
        records_list = records.to_dict("records")
        input_is_df = True
    elif hasattr(records, "to_dict") and callable(getattr(records, "to_dict")):
        try:
            records_list = records.to_dict("records")
            input_is_df = True
        except (TypeError, ValueError):
            records_list = None

    if records_list is None:
        records_list = [
            r.to_dict() if hasattr(r, "to_dict") and callable(getattr(r, "to_dict"))
            else (dict(r) if hasattr(r, "keys") or isinstance(r, dict) else r.__dict__)
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


# Alias for compatibility with survey/specification terminology
build_bsa_dataset = build_reversal_dataset


# ============================================================================
# 6. PyTorch Dataset & Collation for Siamese/CRL Models
# ============================================================================

class PairedReversalDataset(Dataset):
    """
    PyTorch Dataset providing aligned direct (A, B) and inverse (B, A) pairs
    for Consistency Regularization Loss (CRL) and Siamese training.
    """

    def __init__(
        self,
        pairs: Sequence[ReciprocalPair],
        tokenizer: Optional[Any] = None,
        max_length: int = 128,
    ) -> None:
        self.pairs = list(pairs)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair = self.pairs[idx]
        d = pair.direct_instance
        inv = pair.inverse_instance

        d_label = d.get("label") if hasattr(d, "get") else getattr(d, "label", None)
        inv_label = inv.get("label") if hasattr(inv, "get") else getattr(inv, "label", None)

        d_label_id = LABEL_TO_ID.get(d_label, -100) if d_label else -100
        inv_label_id = LABEL_TO_ID.get(inv_label, -100) if inv_label else -100

        d_text1 = d.get("text1") if hasattr(d, "get") else getattr(d, "text1", "")
        d_text2 = d.get("text2") if hasattr(d, "get") else getattr(d, "text2", "")
        inv_text1 = inv.get("text1") if hasattr(inv, "get") else getattr(inv, "text1", "")
        inv_text2 = inv.get("text2") if hasattr(inv, "get") else getattr(inv, "text2", "")

        d_id = d.get("instance_id") if hasattr(d, "get") else getattr(d, "instance_id", "")
        inv_id = inv.get("instance_id") if hasattr(inv, "get") else getattr(inv, "instance_id", "")

        item: Dict[str, Any] = {
            "pair_id": pair.pair_id,
            "direct_instance_id": d_id,
            "direct_text1": d_text1,
            "direct_text2": d_text2,
            "direct_label": d_label,
            "direct_label_id": d_label_id,
            "inverse_instance_id": inv_id,
            "inverse_text1": inv_text1,
            "inverse_text2": inv_text2,
            "inverse_label": inv_label,
            "inverse_label_id": inv_label_id,
            "is_reversible": pair.is_reversible,
            "is_synthetic": pair.is_synthetic,
        }

        if self.tokenizer is not None:
            enc_dir = self.tokenizer(
                d_text1,
                d_text2,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt" if HAS_TORCH else None,
            )
            enc_inv = self.tokenizer(
                inv_text1,
                inv_text2,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt" if HAS_TORCH else None,
            )

            if HAS_TORCH:
                for k, v in enc_dir.items():
                    item[f"dir_{k}"] = v.squeeze(0)
                for k, v in enc_inv.items():
                    item[f"inv_{k}"] = v.squeeze(0)
            else:
                for k, v in enc_dir.items():
                    item[f"dir_{k}"] = v
                for k, v in enc_inv.items():
                    item[f"inv_{k}"] = v

        return item
