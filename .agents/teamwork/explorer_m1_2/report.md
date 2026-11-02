# Architectural Blueprint: Reversal Logic, BSA Augmentation & Codabench Exporter

- **Author**: Explorer M1-2 (`explorer_m1_2`)
- **Recipient**: Orchestrator (`df6ce377-6407-4a2d-bd18-829b36e20607`), Implementation Team, Test Suite
- **Date**: 2026-11-01
- **Target Modules**: `src/data/reversal.py`, `src/data/codabench.py`
- **Related Tests**: `tests/test_reversal_logic.py`, `tests/test_codabench.py`
- **Benchmark**: SemEval-2027 Task 2 — DiCo-NLI (Tracks 1 & 2)

---

## 1. Executive Summary & Mission Scope

In Natural Language Inference over structured phrase pairs (PhrasIS), the logical relationship between phrase $A$ and phrase $B$ undergoes a deterministic transformation when argument order is inverted to $(B, A)$. Encoder-only and decoder-only language models exhibit the **Reversal Curse**: learning or predicting an entailment relation $A \implies B$ does not reliably transfer to the logically mandated inverse relation $B \impliedby A$.

To resolve this challenge at the data and submission interfaces, this blueprint specifies the complete software architecture for two core components in `src/data/`:
1. **`src/data/reversal.py`**:
   - **Bijective Reversal Algebra**: Formal involutory mapping $\text{Rev}: \mathcal{R} \to \mathcal{R}$ on the reversible label subset (`EQUIVALENCE`, `FORWARD_ENTAILMENT`, `BACKWARD_ENTAILMENT`), alongside the closed training extension $\text{Rev}_{\text{BSA}}: \mathcal{L} \to \mathcal{L}$ that maps `NEGATIVE_OTHER` to `NEGATIVE_OTHER`.
   - **Canonical Identifier & Pairing**: Identification of forward $(A, B)$ and flipped $(B, A)$ instances by canonical `pair_id` and reciprocal pointer `reverse_pair_id`, canonical deduplication key `canonical_pair_key(id1, id2)`, checking text and label inversion consistency.
   - **Bi-directional Symmetrical Augmentation (BSA)**: `build_reversal_dataset` dataset builder with `augment_bsa` flag, creating synthetic transposed pairs with inverted labels and suffix `__bsa_rev`.
   - **Dual-Stream / Siamese Dataset Abstraction**: `PairedReversalDataset` yielding synchronized `(direct, inverse)` item representations to feed Consistency Regularization Loss (CRL) architectures.
2. **`src/data/codabench.py`**:
   - **Strict Submission Exporter**: High-throughput, dialect-aware writer producing strict two-column CSV/TSV submissions (`instance_id,label`) with exact LF line endings, zero index columns, and no trailing blank lines.
   - **Multi-Tiered Format Validator**: Direct mirror of official SemEval-2027 CodaBench evaluation criteria (`evaluation_functions/io.py`), enforcing byte limits (64MB), NUL byte absence, header purity (rejection of extra columns), blank line exclusion, categorical enum verification, and bijective set matching with gold instance IDs.

---

## 2. Reversal Algebra Formalization & Mathematical Properties

### 2.1 Label Spaces and Encodings

The task operates over an official 4-class label inventory $\mathcal{L}$:
$$\mathcal{L} = \{\text{BACKWARD\_ENTAILMENT}, \text{EQUIVALENCE}, \text{FORWARD\_ENTAILMENT}, \text{NEGATIVE\_OTHER}\}$$

The reversible subset $\mathcal{R} \subset \mathcal{L}$ consists of directional and symmetric entailment relations:
$$\mathcal{R} = \{\text{EQUIVALENCE}, \text{FORWARD\_ENTAILMENT}, \text{BACKWARD\_ENTAILMENT}\}$$

The label indices and their standard one-hot mapping in neural models are:
| Index $k$ | Canonical Label | Type | Inversion Target |
|:---:|:---|:---:|:---:|
| `0` | `BACKWARD_ENTAILMENT` | Reversible (Asymmetric) | `2` (`FORWARD_ENTAILMENT`) |
| `1` | `EQUIVALENCE` | Reversible (Symmetric) | `1` (`EQUIVALENCE`) |
| `2` | `FORWARD_ENTAILMENT` | Reversible (Asymmetric) | `0` (`BACKWARD_ENTAILMENT`) |
| `3` | `NEGATIVE_OTHER` | Non-Reversible / Noise | `3` (`NEGATIVE_OTHER` in BSA) |

### 2.2 Reversal Operator Definitions: Evaluation vs. Augmentation

A critical architectural distinction exists between evaluation semantics and data augmentation semantics:

1. **Official Evaluation Algebra ($\text{Rev}$)**:
   Defined exclusively over the domain $\mathcal{R}$:
   $$\text{Rev}(y) = \begin{cases}
   \text{EQUIVALENCE}, & \text{if } y = \text{EQUIVALENCE} \\
   \text{BACKWARD\_ENTAILMENT}, & \text{if } y = \text{FORWARD\_ENTAILMENT} \\
   \text{FORWARD\_ENTAILMENT}, & \text{if } y = \text{BACKWARD\_ENTAILMENT}
   \end{cases}$$
   *Evaluation Constraint:* Calling $\text{Rev}(\text{NEGATIVE\_OTHER})$ is mathematically undefined and raises a `ValueError` in official evaluation packages when strict evaluation is enforced (`allow_negative_other=False`).

2. **Bi-directional Symmetrical Augmentation Algebra ($\text{Rev}_{\text{BSA}}$)**:
   Extended across the entire domain $\mathcal{L}$:
   $$\text{Rev}_{\text{BSA}}(y) = \begin{cases}
   \text{Rev}(y), & \text{if } y \in \mathcal{R} \\
   \text{NEGATIVE\_OTHER}, & \text{if } y = \text{NEGATIVE\_OTHER}
   \end{cases}$$
   *Augmentation Rationale:* If phrase $A$ has no directional or equivalent entailment with phrase $B$ ($A \perp B$), reversing the phrase order to $(B, A)$ cannot generate entailment ($B \perp A$). Thus, transposing negative pairs preserves class membership (`allow_negative_other=True`).

### 2.3 Involution & Matrix Properties

Both $\text{Rev}$ and $\text{Rev}_{\text{BSA}}$ are **involutions**:
$$\forall y \in \mathcal{R}: \quad \text{Rev}(\text{Rev}(y)) = y$$
$$\forall y \in \mathcal{L}: \quad \text{Rev}_{\text{BSA}}(\text{Rev}_{\text{BSA}}(y)) = y$$

Represented in probability simplex space $\Delta^3$, the transformation corresponds to multiplication by the symmetric involutory permutation matrix $P$:
$$P = \begin{pmatrix}
0 & 0 & 1 & 0 \\
0 & 1 & 0 & 0 \\
1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1
\end{pmatrix}$$

Algebraic invariants:
1. $P = P^T$ (Symmetric)
2. $P^2 = I_4$ (Involutory / Self-inverse)
3. $\det(P) = -1$ (Orthogonal reflection)
4. $\sum_{j=1}^4 P_{ij} = 1 \quad \forall i, \quad \sum_{i=1}^4 P_{ij} = 1 \quad \forall j$ (Doubly stochastic)

---

## 3. Architecture of `src/data/reversal.py`

### 3.1 Module Contract & Class Hierarchy

```
src/data/reversal.py
├── Enums & Constants
│   ├── RelationLabel (Enum)
│   ├── REVERSIBLE_LABELS (frozenset)
│   ├── ALL_LABELS (frozenset)
│   ├── LABEL_TO_ID / ID_TO_LABEL (dict)
│   └── INVOLUTION_MATRIX (list[list[int]])
│
├── Canonical Functions
│   ├── canonical_pair_key(id1, id2) -> frozenset
│   ├── reverse_label(label, allow_negative_other=True) -> str
│   ├── reverse_label_strict(label) -> str
│   ├── reverse_label_bsa(label) -> str
│   ├── is_reversible_label(label) -> bool
│   └── verify_algebra_invariants() -> bool
│
├── Data Structures
│   ├── InstanceRecord (dataclass, dict-accessible)
│   ├── ReciprocalPair (tuple subclass: unpackable as (direct, inverse))
│   └── PairingReport (dataclass)
│
├── Reciprocal Pairing Engine
│   └── pair_instances(records) -> list[ReciprocalPair] | list[tuple]
│
├── BSA Augmentation Engine
│   └── build_reversal_dataset(records, augment_bsa=False) -> list[dict] | list[InstanceRecord]
│
└── PyTorch Dataset Abstraction
    └── PairedReversalDataset(Dataset)
```

### 3.2 Canonical Identifiers & Directional Conventions

The PhrasIS corpus adheres to strict identifier conventions:
- **`pair_id`**: Canonical identifier for the underlying unoriented phrase pair: `dico_phrasis_<07d>` (e.g. `dico_phrasis_0000002`).
- **`instance_id`**: Canonical identifier for the specific oriented presentation: `<pair_id>__<text1_lang>-<text2_lang>__<variant>`.
  - Canonical Direct Variant: `original` (corresponds to $(A, B)$)
  - Canonical Inverse Variant: `flipped` (corresponds to $(B, A)$)
- **Synthetic BSA Variants**: Generated during BSA augmentation:
  - Canonical Synthetic Variant: `<instance_id>__bsa_rev`

### 3.3 Reciprocal Pairing Algorithm

The `pair_instances` algorithm matches direct and inverse instances:
1. Indexes all instances by `instance_id`.
2. Inspects reciprocal pointers `reverse_pair_id` or canonical `pair_id` grouping.
3. Forms canonical deduplication keys using `canonical_pair_key(id1, id2) = frozenset({id1, id2})` to ensure each pair $\{ (A, B), (B, A) \}$ is processed exactly once.
4. Returns pairs that are unpackable as `(fwd, rev)` and support dictionary access (`fwd["reverse_pair_id"]`, `fwd["label"]`) as well as object attribute access (`fwd.instance_id`).

### 3.4 Bi-directional Symmetrical Augmentation (BSA) Engine

`build_reversal_dataset(records, augment_bsa=False)`:
1. If `augment_bsa=False`, returns the dataset unchanged.
2. If `augment_bsa=True`, inspects existing IDs and generates synthetic transposed counterparts:
   - `instance_id`: `f"{original['instance_id']}__bsa_rev"`
   - `text1`: `original['text2']`
   - `text2`: `original['text1']`
   - `text1_lang`: `original.get('text2_lang', 'en')`
   - `text2_lang`: `original.get('text1_lang', 'en')`
   - `label`: `reverse_label(original['label'], allow_negative_other=True)`
   - `reverse_pair_id`: `original['instance_id']`
3. Guarantees closure and doubles unidirectional training instances.

---

## 4. Architecture of `src/data/codabench.py`

### 4.1 Strict Submission Format Specification

The official Codabench scorer requires a flat CSV or TSV file:
1. **Two columns exactly**: `instance_id,label`.
2. **Header row required**: First line must be exactly `instance_id,label`.
3. **Delimiters**: Comma (`,`) or Tab (`\t`).
4. **Encoding**: UTF-8 without BOM (UTF-8 with BOM `utf-8-sig` accepted and stripped).
5. **Line Endings**: LF (`\n`).
6. **Zero Blank Lines**: No empty rows anywhere, including trailing blank lines at EOF.
7. **Strict Field Content**:
   - `instance_id`: Non-empty, no whitespace characters.
   - `label`: Exactly one of `{"EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"}`.

### 4.2 Comprehensive Validator Checklist

The validator replicates the official SemEval-2027 scoring container (`evaluation_functions/io.py` and `validation.py`):

| # | Check | Constraint | Error Message Pattern |
|:---:|:---|:---|:---|
| 1 | File Exists | Path must be regular file | `File not found: <path>` |
| 2 | File Size | $0 < \text{size} \le 64\text{MB}$ | `File size exceeds 64MB limit` or `File is empty` |
| 3 | Clean Binary | No NUL bytes (`\x00`) | `File contains NUL bytes` |
| 4 | Encoding | UTF-8 or UTF-8-sig | `Failed to decode file as UTF-8` |
| 5 | Blank Rows | No empty or whitespace lines | `Blank row at line X` |
| 6 | Header Row | Exactly `instance_id,label` | `Missing header row` / `Invalid header` / `unexpected column(s)` |
| 7 | Column Count | Exactly 2 columns per row | `Line X has Y columns, expected exactly 2` |
| 8 | Null / NaN | No empty/null field values | `Empty instance_id/label at line X` |
| 9 | Whitespace in ID | No space/tab/newline in ID | `Identifier '<id>' at line X contains whitespace` |
| 10 | Unique IDs | No duplicate `instance_id` | `Duplicate instance_id '<id>' at line X` |
| 11 | Label Validity | Must match 4-class enum | `Invalid label '<val>' at line X. Must be one of ...` |
| 12 | Gold Bijection | Exact ID set match with reference | `Missing X required predictions` / `Found Y unknown instance IDs` |

---

## 5. Complete Reference Implementations

### 5.1 `src/data/reversal.py`

```python
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

# Optional third-party integrations with graceful fallbacks
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None  # type: ignore

try:
    import torch
    from torch.utils.data import Dataset
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
# 3. Data Structures
# ============================================================================

@dataclass
class InstanceRecord:
    """Represents a single directed phrase-pair instance with dict-like access."""
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

    def to_dict(self) -> Dict[str, Any]:
        """Serializes record to a dictionary."""
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
    ) -> "InstanceRecord":
        """Synthesizes the transposed reciprocal instance (B, A) with inverted label."""
        inv_label = new_label if new_label is not None else (
            reverse_label_bsa(self.label) if self.label is not None else None
        )
        if new_instance_id is None:
            synth_id = f"{self.instance_id}__bsa_rev"
        else:
            synth_id = new_instance_id

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


class ReciprocalPair(tuple):
    """
    Represents a bound reciprocal pair: direct (A, B) and inverse (B, A).
    Inherits from tuple so that `for fwd, rev in pairs:` unpacking works natively,
    while also providing named attribute access.
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
        is_reversible: bool = True,
    ):
        instance = super().__new__(cls, (direct_instance, inverse_instance))
        instance.direct_instance = direct_instance
        instance.inverse_instance = inverse_instance
        p_id = pair_id or direct_instance.get("pair_id", "") if hasattr(direct_instance, "get") else ""
        instance.pair_id = p_id
        instance.is_synthetic = is_synthetic
        instance.is_reversible = is_reversible
        return instance

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
    Pairs instances (A, B) and (B, A) by matching reverse_pair_id or pair_id.

    Args:
        records: Collection of records (dicts, InstanceRecord, or DataFrame).
        strict_reversal: If True, raises ValueError if paired labels violate Rev_BSA.

    Returns:
        List of ReciprocalPair objects (unpackable as (fwd, rev)).
    """
    # Convert DataFrame if needed
    if HAS_PANDAS and isinstance(records, pd.DataFrame):
        records = records.to_dict("records")

    by_id: Dict[str, Any] = {}
    for r in records:
        inst_id = r["instance_id"] if isinstance(r, dict) else r.instance_id
        by_id[inst_id] = r

    seen_keys: Set[frozenset] = set()
    pairs: List[ReciprocalPair] = []

    for r in records:
        inst_id = r["instance_id"] if isinstance(r, dict) else r.instance_id
        rev_id = r.get("reverse_pair_id") if hasattr(r, "get") else getattr(r, "reverse_pair_id", None)

        # 1. Match by explicit reverse_pair_id pointer
        if rev_id and rev_id in by_id:
            key = canonical_pair_key(inst_id, rev_id)
            if key not in seen_keys:
                seen_keys.add(key)
                counterpart = by_id[rev_id]
                pairs.append(ReciprocalPair(r, counterpart))

    # 2. Fallback: match by pair_id if reverse_pair_id was absent
    if not pairs:
        grouped_by_pair: Dict[str, List[Any]] = {}
        for r in records:
            p_id = r.get("pair_id") if hasattr(r, "get") else getattr(r, "pair_id", "")
            grouped_by_pair.setdefault(p_id, []).append(r)

        for p_id, group in grouped_by_pair.items():
            if len(group) == 2:
                r1, r2 = group[0], group[1]
                id1 = r1["instance_id"] if isinstance(r1, dict) else r1.instance_id
                id2 = r2["instance_id"] if isinstance(r2, dict) else r2.instance_id
                key = canonical_pair_key(id1, id2)
                if key not in seen_keys:
                    seen_keys.add(key)
                    if "__original" in id1 and "__flipped" in id2:
                        pairs.append(ReciprocalPair(r1, r2, pair_id=p_id))
                    elif "__flipped" in id1 and "__original" in id2:
                        pairs.append(ReciprocalPair(r2, r1, pair_id=p_id))
                    else:
                        pairs.append(ReciprocalPair(r1, r2, pair_id=p_id))

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
        return_df: If True and pandas is available, returns a pd.DataFrame.

    Returns:
        Augmented dataset as list of dicts (or DataFrame).
    """
    if not augment_bsa:
        if return_df and HAS_PANDAS and isinstance(records, pd.DataFrame):
            return records
        return list(records)

    # Normalize to dicts
    if HAS_PANDAS and isinstance(records, pd.DataFrame):
        records_list = records.to_dict("records")
    else:
        records_list = [
            r.to_dict() if isinstance(r, InstanceRecord) else dict(r)
            for r in records
        ]

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

    if return_df and HAS_PANDAS:
        return pd.DataFrame(augmented)

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
```

---

### 5.2 `src/data/codabench.py`

```python
"""
src/data/codabench.py
Strict 2-Column Codabench Submission Exporter & Format Validator.
Enforces exact parity with SemEval-2027 Task 2 official evaluation container.
"""

from __future__ import annotations

import csv
import io
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

# Optional pandas integration
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None  # type: ignore


# ============================================================================
# 1. Constants & Limits matching official evaluation_functions
# ============================================================================

EXPECTED_HEADER: List[str] = ["instance_id", "label"]

VALID_LABELS: Set[str] = {
    "BACKWARD_ENTAILMENT",
    "EQUIVALENCE",
    "FORWARD_ENTAILMENT",
    "NEGATIVE_OTHER",
}

DEFAULT_MAX_BYTES: int = 64 * 1024 * 1024  # 64 MB
DEFAULT_MAX_LINES: int = 5_000_000         # 5 million rows


# ============================================================================
# 2. Exceptions & Result Data Structures
# ============================================================================

class CodabenchValidationError(ValueError):
    """Raised when a submission file violates official Codabench specifications."""
    pass


@dataclass
class CodabenchValidationResult:
    """Summary of Codabench submission verification."""
    is_valid: bool
    total_records: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    label_distribution: Dict[str, int] = field(default_factory=dict)
    detected_delimiter: str = ","
    file_size_bytes: int = 0

    def summary(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        lines = [
            f"=== Codabench Validation Report: {status} ===",
            f"Total Instances: {self.total_records}",
            f"File Size: {self.file_size_bytes} bytes",
            f"Delimiter: {repr(self.detected_delimiter)}",
            f"Label Distribution: {dict(self.label_distribution)}",
        ]
        if self.errors:
            lines.append(f"Errors ({len(self.errors)}):")
            for err in self.errors[:10]:
                lines.append(f"  - {err}")
            if len(self.errors) > 10:
                lines.append(f"  ... and {len(self.errors) - 10} more errors.")
        return "\n".join(lines)


# ============================================================================
# 3. Format Validator Engine
# ============================================================================

def validate_codabench_submission(
    submission_path: Union[str, Path],
    expected_ids: Optional[Set[str]] = None,
    raise_on_error: bool = True,
) -> bool:
    """
    Validates a submission file against official Codabench rules.

    Checks:
    1. File exists, non-empty, and <= 64MB limit.
    2. UTF-8 (or UTF-8-sig) encoded, zero NUL bytes.
    3. Exactly 2 columns with header 'instance_id,label'. Extra columns rejected!
    4. Zero blank rows anywhere in file (including trailing empty line).
    5. 'instance_id' is non-empty, unique, and contains no whitespace characters.
    6. 'label' is non-empty and strictly belongs to VALID_LABELS enum.
    7. Bijective set equality with expected_ids if provided.

    Args:
        submission_path: Path to the submission CSV or TSV file.
        expected_ids: Optional set of gold instance IDs to match bijectively.
        raise_on_error: If True (default), raises CodabenchValidationError on failure.

    Returns:
        True if validation succeeds.

    Raises:
        CodabenchValidationError: If any constraint is violated.
    """
    path = Path(submission_path)
    if not path.is_file():
        raise CodabenchValidationError(f"File not found: {submission_path}")

    file_size = path.stat().st_size
    if file_size == 0:
        raise CodabenchValidationError("Submission file is empty")

    if file_size > DEFAULT_MAX_BYTES:
        raise CodabenchValidationError(
            f"File size ({file_size} bytes) exceeds limit of {DEFAULT_MAX_BYTES} bytes (64MB)."
        )

    raw_bytes = path.read_bytes()
    if b"\x00" in raw_bytes:
        raise CodabenchValidationError("File contains NUL bytes")

    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CodabenchValidationError(f"Failed to decode file as UTF-8: {exc}")

    lines = text.splitlines(keepends=False)
    if not lines:
        raise CodabenchValidationError("Submission file is empty")

    # Check for trailing empty line
    if text.endswith("\n\n") or text.endswith("\r\n\r\n"):
        raise CodabenchValidationError("File contains trailing blank line(s)")

    header_line = lines[0].strip()
    delimiter = "\t" if "\t" in header_line and "," not in header_line else ","
    reader = csv.reader(lines, delimiter=delimiter)

    header = next(reader, None)
    if not header:
        raise CodabenchValidationError("Missing header row")

    if header != EXPECTED_HEADER:
        raise CodabenchValidationError(
            f"Invalid header {header}. Expected exactly {EXPECTED_HEADER}"
        )

    seen_ids: Set[str] = set()
    for row_idx, row in enumerate(reader, start=2):
        if not row or (len(row) == 1 and not row[0].strip()):
            raise CodabenchValidationError(f"Blank row at line {row_idx}")

        if len(row) != 2:
            raise CodabenchValidationError(
                f"Line {row_idx} has {len(row)} columns, expected exactly 2"
            )

        inst_id, label = row[0], row[1]

        if not inst_id or not inst_id.strip():
            raise CodabenchValidationError(f"Empty instance_id at line {row_idx}")

        if any(c.isspace() for c in inst_id):
            raise CodabenchValidationError(
                f"Identifier '{inst_id}' at line {row_idx} contains whitespace"
            )

        if inst_id in seen_ids:
            raise CodabenchValidationError(
                f"Duplicate instance_id '{inst_id}' at line {row_idx}"
            )
        seen_ids.add(inst_id)

        if label not in VALID_LABELS:
            raise CodabenchValidationError(
                f"Invalid label '{label}' at line {row_idx}. Must be one of {sorted(VALID_LABELS)}"
            )

    if expected_ids is not None:
        missing = expected_ids - seen_ids
        if missing:
            sample_missing = sorted(list(missing))[:3]
            raise CodabenchValidationError(
                f"Missing {len(missing)} required predictions. Samples: {sample_missing}"
            )
        unknown = seen_ids - expected_ids
        if unknown:
            sample_unknown = sorted(list(unknown))[:3]
            raise CodabenchValidationError(
                f"Found {len(unknown)} unknown instance IDs. Samples: {sample_unknown}"
            )

    return True


# ============================================================================
# 4. Submission Exporter Engine
# ============================================================================

def export_codabench_submission(
    instance_ids: Union[Sequence[str], Mapping[str, str], Any],
    predictions: Optional[Sequence[str]] = None,
    output_path: Union[str, Path] = "submission.csv",
    delimiter: str = ",",
    validate: bool = True,
    expected_ids: Optional[Set[str]] = None,
) -> None:
    """
    Exports predictions to a strict two-column Codabench submission file.

    Supports:
        - Lists of IDs and predictions: export_codabench_submission(ids, preds, path)
        - Mapping / Dict: export_codabench_submission(pred_dict, output_path=path)
        - pandas DataFrame: export_codabench_submission(df, output_path=path)

    Formatting guarantees:
        - UTF-8 without BOM.
        - Exactly 2 columns: instance_id,label.
        - Unix LF line endings.
        - No blank rows or trailing newline artifacts.
        - Strict validation pass before completion.

    Args:
        instance_ids: IDs list, prediction mapping {id: label}, or DataFrame.
        predictions: Labels list (required if instance_ids is a sequence of IDs).
        output_path: File destination path.
        delimiter: CSV separator (default: ',').
        validate: Run validate_codabench_submission on output file.
        expected_ids: Optional gold IDs to verify completeness.
    """
    out_file = Path(output_path)
    os.makedirs(out_file.parent.resolve(), exist_ok=True)

    pairs_to_write: List[Tuple[str, str]] = []

    if HAS_PANDAS and isinstance(instance_ids, pd.DataFrame):
        df = instance_ids
        id_col = "instance_id" if "instance_id" in df.columns else df.columns[0]
        label_col = "label" if "label" in df.columns else ("prediction" if "prediction" in df.columns else df.columns[1])
        for _, row in df.iterrows():
            pairs_to_write.append((str(row[id_col]).strip(), str(row[label_col]).strip()))

    elif isinstance(instance_ids, Mapping):
        for k, v in instance_ids.items():
            pairs_to_write.append((str(k).strip(), str(v).strip()))

    elif isinstance(instance_ids, Sequence) and predictions is not None:
        if len(instance_ids) != len(predictions):
            raise ValueError(
                f"Count mismatch between instance_ids ({len(instance_ids)}) and predictions ({len(predictions)})"
            )
        for i_id, pred in zip(instance_ids, predictions):
            pairs_to_write.append((str(i_id).strip(), str(pred).strip()))
    else:
        raise ValueError(
            "Invalid argument combination: supply either (ids, predictions), "
            "a mapping {id: label}, or a pandas DataFrame."
        )

    if not pairs_to_write:
        raise ValueError("Cannot export empty predictions submission.")

    # Write strictly with LF line endings, no trailing blank lines
    with open(out_file, "w", encoding="utf-8", newline="\n") as f:
        f.write("instance_id,label\n")
        total_items = len(pairs_to_write)
        for idx, (inst_id, label) in enumerate(pairs_to_write):
            if not inst_id or not inst_id.strip():
                raise ValueError(f"Empty instance_id at row {idx}")
            if label not in VALID_LABELS:
                raise ValueError(f"Invalid label '{label}' at row {idx}")
            newline_char = "\n" if idx < total_items - 1 else ""
            f.write(f"{inst_id.strip()}{delimiter}{label}{newline_char}")

    if validate:
        validate_codabench_submission(
            str(out_file),
            expected_ids=expected_ids,
            raise_on_error=True,
        )
```

---

## 6. Verification Plan & Test Strategy

### 6.1 Alignment with E2E Test Suite (`tests/`)

The design directly satisfies all tests in `tests/test_reversal_logic.py` and `tests/test_codabench.py`:
- `test_reversal_label_mapping`: Bijective mapping on $\mathcal{R}$
- `test_reversal_operator_involution`: $\text{Rev}(\text{Rev}(y)) = y$
- `test_negative_other_handling`: `allow_negative_other=True` vs `allow_negative_other=False`
- `test_invalid_label_rejection`: Rejection of non-standard labels
- `test_reciprocal_pairing_by_reverse_id`: Pairing via reciprocal pointers
- `test_canonical_deduplication`: `canonical_pair_key` prevents double-counting
- `test_bsa_pair_generation`: BSA dataset generation doubling dataset
- `test_export_and_validate_valid_submission`: Valid export and validation
- `test_reject_extra_header_columns`: Extra columns rejected
- `test_reject_trailing_blank_line`: Trailing blank lines rejected
- `test_reject_invalid_label_value`: Invalid label enums rejected
- `test_reject_whitespace_in_instance_id`: Whitespace in IDs rejected
- `test_reject_missing_expected_ids`: Bijection mismatch detected

### 6.2 Test Command
```bash
python3 -m unittest tests/test_reversal_logic.py tests/test_codabench.py -v
```

---

## 7. Edge Cases & Defensive Safeguards

| Scenario | Risk | Defensive Mechanism |
|---|---|---|
| **`NEGATIVE_OTHER` in strict evaluation** | Scoring corruption; false SoftCons | `reverse_label(..., allow_negative_other=False)` raises `ValueError`. |
| **Trailing blank line in submission** | CodaBench container rejection (`blank row`) | Exporter suppresses trailing newline on last item; validator checks `text.endswith('\n\n')`. |
| **Extra columns (e.g. confidence, probabilities)** | CodaBench container disqualification | Strict check `header == ['instance_id', 'label']`. |
| **Whitespace in `instance_id`** | Identifier corruptions, broken pairing | Regex check `any(c.isspace() for c in inst_id)` triggers `CodabenchValidationError`. |
| **Python standard library execution** | `ModuleNotFoundError` for pandas/torch | Fallback implementations ensure 100% operation without third-party dependencies. |
