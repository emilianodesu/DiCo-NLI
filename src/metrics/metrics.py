"""
src/metrics/metrics.py

Official evaluation metrics implementation for SemEval-2027 Task 2 (DiCo-NLI).
Guarantees exact parity with official evaluation functions and CodaBench scoring:
  - Multiclass Weighted F1 across 4 labels with zero-division handling.
  - SoftCons over unique reversible source pairs.
  - HardCons over unique reversible source pairs.
  - Enforces mathematical invariant: 0.0 <= HardCons <= SoftCons <= 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

try:
    import numpy as np  # type: ignore
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False


# -----------------------------------------------------------------------------
# 1. Label Inventory and Inversion Constants
# -----------------------------------------------------------------------------

class TaskLabel(str, Enum):
    """Canonical 4-class labels for SemEval-2027 Task 2."""
    BACKWARD_ENTAILMENT = "BACKWARD_ENTAILMENT"
    EQUIVALENCE = "EQUIVALENCE"
    FORWARD_ENTAILMENT = "FORWARD_ENTAILMENT"
    NEGATIVE_OTHER = "NEGATIVE_OTHER"


LABEL_TO_IDX: Dict[str, int] = {
    TaskLabel.BACKWARD_ENTAILMENT.value: 0,
    TaskLabel.EQUIVALENCE.value: 1,
    TaskLabel.FORWARD_ENTAILMENT.value: 2,
    TaskLabel.NEGATIVE_OTHER.value: 3,
}

IDX_TO_LABEL: Dict[int, str] = {idx: label for label, idx in LABEL_TO_IDX.items()}

ALL_LABELS_TUPLE: Tuple[str, ...] = (
    TaskLabel.BACKWARD_ENTAILMENT.value,
    TaskLabel.EQUIVALENCE.value,
    TaskLabel.FORWARD_ENTAILMENT.value,
    TaskLabel.NEGATIVE_OTHER.value,
)

ALL_LABELS: Set[str] = set(ALL_LABELS_TUPLE)

REVERSIBLE_LABELS_TUPLE: Tuple[str, ...] = (
    TaskLabel.BACKWARD_ENTAILMENT.value,
    TaskLabel.EQUIVALENCE.value,
    TaskLabel.FORWARD_ENTAILMENT.value,
)

REVERSIBLE_LABELS: Set[str] = set(REVERSIBLE_LABELS_TUPLE)

REVERSE_LABEL_MAP: Dict[str, str] = {
    TaskLabel.BACKWARD_ENTAILMENT.value: TaskLabel.FORWARD_ENTAILMENT.value,
    TaskLabel.EQUIVALENCE.value: TaskLabel.EQUIVALENCE.value,
    TaskLabel.FORWARD_ENTAILMENT.value: TaskLabel.BACKWARD_ENTAILMENT.value,
}

if HAS_NUMPY and np is not None:
    INVOLUTION_PERMUTATION_MATRIX = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)
else:
    INVOLUTION_PERMUTATION_MATRIX = [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def is_reversible_label(label: str) -> bool:
    """Return True if label belongs to the closed reversible set."""
    return label in REVERSE_LABEL_MAP


def reverse_label(label: str, allow_negative_other: bool = False) -> str:
    """
    Return deterministic reverse of a reversible label.
    Raises ValueError if label is not in REVERSIBLE_LABELS and allow_negative_other is False.
    """
    if label == TaskLabel.NEGATIVE_OTHER.value and allow_negative_other:
        return TaskLabel.NEGATIVE_OTHER.value
    if label not in REVERSE_LABEL_MAP:
        raise ValueError(f"Label {label!r} has no deterministic reverse.")
    return REVERSE_LABEL_MAP[label]


# -----------------------------------------------------------------------------
# 2. Typed Report Structures
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ReversiblePair:
    """Unordered reciprocal pair linking two evaluation instances."""
    first_id: str
    second_id: str
    first_label: str
    second_label: str


# Alias for compatibility with blueprint naming
ReversiblePairSpec = ReversiblePair


@dataclass(frozen=True)
class PerLabelMetrics:
    """Precision, recall, F1, support and confusion statistics for a single label."""
    label: str
    support: int
    predicted: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> Dict[str, Union[str, int, float]]:
        return {
            "label": self.label,
            "support": self.support,
            "predicted": self.predicted,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass(frozen=True)
class ClassificationReport:
    """Item-level multiclass classification metrics across all instances."""
    weighted_f1: float
    macro_f1: float
    accuracy: float
    total_instances: int
    per_label: Dict[str, PerLabelMetrics]
    confusion_matrix: Dict[str, Dict[str, int]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "weighted_f1": self.weighted_f1,
            "macro_f1": self.macro_f1,
            "accuracy": self.accuracy,
            "total_instances": self.total_instances,
            "per_label": {k: v.to_dict() for k, v in self.per_label.items()},
            "confusion_matrix": self.confusion_matrix,
        }


@dataclass(frozen=True)
class ConsistencyReport:
    """Pair-level directional consistency metrics over reversible pairs."""
    soft_cons: float
    hard_cons: float
    reversible_pairs: int
    soft_consistent_pairs: int
    hard_correct_pairs: int
    pair_errors: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "soft_cons": self.soft_cons,
            "hard_cons": self.hard_cons,
            "reversible_pairs": self.reversible_pairs,
            "soft_consistent_pairs": self.soft_consistent_pairs,
            "hard_correct_pairs": self.hard_correct_pairs,
            "pair_errors": self.pair_errors,
        }


@dataclass(frozen=True)
class EvaluationReport:
    """Master official evaluation report matching CodaBench output."""
    weighted_f1: float
    soft_cons: float
    hard_cons: float
    classification: ClassificationReport
    consistency: ConsistencyReport

    def to_dict(self) -> Dict[str, object]:
        return {
            "weighted_f1": self.weighted_f1,
            "soft_cons": self.soft_cons,
            "hard_cons": self.hard_cons,
            "classification": self.classification.to_dict(),
            "consistency": self.consistency.to_dict(),
        }

    def format_scores_text(self) -> str:
        """Format 12-decimal precision text identical to CodaBench scores.txt."""
        return (
            f"weighted_f1: {self.weighted_f1:.12f}\n"
            f"soft_cons: {self.soft_cons:.12f}\n"
            f"hard_cons: {self.hard_cons:.12f}\n"
        )


# -----------------------------------------------------------------------------
# 3. Official Computation Engine
# -----------------------------------------------------------------------------

def safe_divide(numerator: float, denominator: float) -> float:
    """Return 0.0 on zero division, maintaining exact float precision."""
    if denominator == 0.0 or denominator == 0:
        return 0.0
    return numerator / denominator


def compute_weighted_f1(gold: Sequence[str], pred: Sequence[str]) -> float:
    """
    Compute multiclass F1 weighted by support across all 4 classes.
    """
    if len(gold) != len(pred):
        raise ValueError(f"Length mismatch: {len(gold)} != {len(pred)}")
    if len(gold) == 0:
        raise ValueError("Cannot compute Weighted F1 on empty sequences")

    classes = sorted(ALL_LABELS)
    total = len(gold)

    f1_sum = 0.0
    for c in classes:
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        support = tp + fn

        if support == 0:
            continue

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        f1_sum += (support / total) * f1

    return f1_sum


def compute_classification_report(
    gold_by_id: Mapping[str, str],
    predicted_labels: Mapping[str, str],
    labels: Tuple[str, ...] = ALL_LABELS_TUPLE,
) -> ClassificationReport:
    """
    Compute item-level multiclass Weighted F1, Macro F1, and Accuracy.
    """
    total = len(gold_by_id)
    if total == 0:
        raise ValueError("Cannot compute classification metrics for zero instances.")

    confusion: Dict[str, Dict[str, int]] = {g: {p: 0 for p in labels} for g in labels}
    correct = 0

    for instance_id, gold_label in gold_by_id.items():
        if instance_id not in predicted_labels:
            raise KeyError(f"Missing prediction for instance_id: '{instance_id}'")
        pred_label = predicted_labels[instance_id]
        if gold_label not in confusion:
            confusion[gold_label] = {p: 0 for p in labels}
        confusion[gold_label][pred_label] = confusion[gold_label].get(pred_label, 0) + 1
        if gold_label == pred_label:
            correct += 1

    per_label: Dict[str, PerLabelMetrics] = {}
    weighted_f1 = 0.0
    macro_f1_values: List[float] = []

    for label in labels:
        tp = confusion[label].get(label, 0)
        fp = sum(confusion[other].get(label, 0) for other in labels if other != label)
        fn = sum(confusion[label].get(other, 0) for other in labels if other != label)
        support = sum(confusion[label].values())
        predicted = sum(confusion[other].get(label, 0) for other in labels)

        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)

        per_label[label] = PerLabelMetrics(
            label=label,
            support=support,
            predicted=predicted,
            true_positive=tp,
            false_positive=fp,
            false_negative=fn,
            precision=precision,
            recall=recall,
            f1=f1,
        )
        weighted_f1 += (support / total) * f1
        if support > 0:
            macro_f1_values.append(f1)

    macro_f1 = sum(macro_f1_values) / len(macro_f1_values) if macro_f1_values else 0.0
    accuracy = correct / total

    return ClassificationReport(
        weighted_f1=weighted_f1,
        macro_f1=macro_f1,
        accuracy=accuracy,
        total_instances=total,
        per_label=per_label,
        confusion_matrix=confusion,
    )


def compute_consistency_report(
    pairs: Sequence[Any],
    predicted_labels: Mapping[str, str],
    max_pair_errors: int = 50,
) -> ConsistencyReport:
    """
    Compute official SoftCons and HardCons over reversible source pairs.
    Accepts ReversiblePair, ReciprocalPair, 2-tuples of records/dicts, or pair dicts.
    """
    total = len(pairs)
    if total == 0:
        raise ValueError("Cannot compute consistency metrics without reversible pairs.")

    soft_count = 0
    hard_count = 0
    pair_errors: List[Dict[str, str]] = []

    for pair in pairs:
        # Polymorphic ID and ground-truth label extraction
        if hasattr(pair, "first_id") and hasattr(pair, "second_id"):
            id1 = pair.first_id
            id2 = pair.second_id
            gold1 = getattr(pair, "first_label", None)
            gold2 = getattr(pair, "second_label", None)
        elif isinstance(pair, (tuple, list)) and len(pair) == 2:
            p1, p2 = pair[0], pair[1]
            id1 = p1.get("instance_id") if hasattr(p1, "get") else getattr(p1, "instance_id", str(p1))
            id2 = p2.get("instance_id") if hasattr(p2, "get") else getattr(p2, "instance_id", str(p2))
            gold1 = p1.get("label") if hasattr(p1, "get") else getattr(p1, "label", None)
            gold2 = p2.get("label") if hasattr(p2, "get") else getattr(p2, "label", None)
        elif isinstance(pair, dict):
            id1 = pair["first_id"]
            id2 = pair["second_id"]
            gold1 = pair.get("first_label")
            gold2 = pair.get("second_label")
        else:
            raise TypeError(f"Unsupported pair type for consistency evaluation: {type(pair)}")

        if id1 not in predicted_labels or id2 not in predicted_labels:
            raise KeyError(f"Missing predictions for pair ({id1}, {id2})")

        pred_1 = predicted_labels[id1]
        pred_2 = predicted_labels[id2]

        soft_ok = (
            pred_1 in REVERSIBLE_LABELS
            and pred_2 == REVERSE_LABEL_MAP.get(pred_1)
        )
        hard_ok = (
            pred_1 == gold1
            and pred_2 == gold2
        ) if (gold1 is not None and gold2 is not None) else False

        if soft_ok:
            soft_count += 1
        if hard_ok:
            hard_count += 1
        elif len(pair_errors) < max_pair_errors:
            pair_errors.append({
                "first_id": id1,
                "second_id": id2,
                "first_gold": str(gold1),
                "second_gold": str(gold2),
                "first_prediction": pred_1,
                "second_prediction": pred_2,
                "soft_consistent": str(soft_ok).lower(),
                "hard_correct": str(hard_ok).lower(),
            })

    soft_cons = soft_count / total
    hard_cons = hard_count / total

    assert hard_cons <= soft_cons + 1e-12, (
        f"Mathematical invariant violated: HardCons ({hard_cons}) > SoftCons ({soft_cons})"
    )

    return ConsistencyReport(
        soft_cons=soft_cons,
        hard_cons=hard_cons,
        reversible_pairs=total,
        soft_consistent_pairs=soft_count,
        hard_correct_pairs=hard_count,
        pair_errors=pair_errors,
    )


def compute_official_metrics(
    y_true_dir: Sequence[str],
    y_pred_dir: Sequence[str],
    y_true_rev: Sequence[str],
    y_pred_rev: Sequence[str],
    is_reversible_mask: Sequence[bool],
) -> Dict[str, float]:
    """
    Compute the official metric trio: Weighted F1, SoftCons, HardCons.

    Defensive behaviors:
    - Enforces identical lengths across all 5 input sequences to prevent silent zip truncation.
    - If is_reversible_mask contains zero True values (no reversible pairs in slice),
      returns 0.0 for soft_cons and hard_cons instead of raising ValueError.
    """
    n_dir = len(y_true_dir)
    if not (
        len(y_pred_dir) == n_dir
        and len(y_true_rev) == n_dir
        and len(y_pred_rev) == n_dir
        and len(is_reversible_mask) == n_dir
    ):
        raise ValueError(
            f"Input length mismatch in compute_official_metrics: "
            f"y_true_dir={len(y_true_dir)}, y_pred_dir={len(y_pred_dir)}, "
            f"y_true_rev={len(y_true_rev)}, y_pred_rev={len(y_pred_rev)}, "
            f"is_reversible_mask={len(is_reversible_mask)}"
        )

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

    if not pairs:
        # Zero reversible pairs in this evaluation slice: return 0.0 without exception
        return {
            "weighted_f1": weighted_f1,
            "soft_cons": 0.0,
            "hard_cons": 0.0,
        }

    cons_report = compute_consistency_report(pairs, pred_map)
    return {
        "weighted_f1": weighted_f1,
        "soft_cons": cons_report.soft_cons,
        "hard_cons": cons_report.hard_cons,
    }


def compute_official_evaluation(
    gold_by_id: Mapping[str, str],
    predicted_labels: Mapping[str, str],
    reversible_pairs: Sequence[ReversiblePair],
    max_pair_errors: int = 50,
) -> EvaluationReport:
    """
    Compute complete official evaluation profile (Weighted F1, SoftCons, HardCons).
    Matches SemEval-2027 CodaBench evaluation exactly.
    """
    cls_report = compute_classification_report(gold_by_id, predicted_labels)
    cons_report = compute_consistency_report(
        reversible_pairs, predicted_labels, max_pair_errors=max_pair_errors
    )

    return EvaluationReport(
        weighted_f1=cls_report.weighted_f1,
        soft_cons=cons_report.soft_cons,
        hard_cons=cons_report.hard_cons,
        classification=cls_report,
        consistency=cons_report,
    )


# -----------------------------------------------------------------------------
# 4. High-Throughput Vectorized API
# -----------------------------------------------------------------------------

def compute_metrics_from_arrays(
    y_true: Any,
    y_pred: Any,
    pair_indices: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Vectorized computation for GPU/CPU PyTorch training loops.
    """
    if HAS_NUMPY and np is not None and isinstance(y_true, np.ndarray):
        n_classes = len(ALL_LABELS_TUPLE)
        total = len(y_true)
        if total == 0:
            raise ValueError("Cannot evaluate empty arrays.")

        conf = np.bincount(
            y_true * n_classes + y_pred, minlength=n_classes * n_classes
        ).reshape(n_classes, n_classes)

        tp = np.diag(conf)
        fp = np.sum(conf, axis=0) - tp
        fn = np.sum(conf, axis=1) - tp
        support = np.sum(conf, axis=1)

        precision = np.divide(tp, tp + fp, out=np.zeros(n_classes), where=(tp + fp) != 0)
        recall = np.divide(tp, tp + fn, out=np.zeros(n_classes), where=(tp + fn) != 0)
        denom = precision + recall
        f1 = np.divide(2 * precision * recall, denom, out=np.zeros(n_classes), where=denom != 0)

        weighted_f1 = float(np.sum((support / total) * f1))
        valid_support = support > 0
        macro_f1 = float(np.mean(f1[valid_support])) if np.any(valid_support) else 0.0
        accuracy = float(np.sum(tp) / total)

        result: Dict[str, float] = {
            "weighted_f1": weighted_f1,
            "macro_f1": macro_f1,
            "accuracy": accuracy,
        }

        if pair_indices is not None and len(pair_indices) > 0:
            idx_1 = pair_indices[:, 0]
            idx_2 = pair_indices[:, 1]

            p1 = y_pred[idx_1]
            p2 = y_pred[idx_2]
            t1 = y_true[idx_1]
            t2 = y_true[idx_2]

            # Invert p1 in index space: 0 <-> 2, 1 -> 1, 3 -> -1
            rev_p1 = np.where(p1 == 0, 2, np.where(p1 == 2, 0, np.where(p1 == 1, 1, -1)))

            soft_ok = (p1 != 3) & (p2 == rev_p1)
            hard_ok = (p1 == t1) & (p2 == t2)

            result["soft_cons"] = float(np.mean(soft_ok))
            result["hard_cons"] = float(np.mean(hard_ok))

        return result
    else:
        # Pure-Python fallback
        y_true_list = [IDX_TO_LABEL[int(x)] if isinstance(x, (int, float)) else str(x) for x in y_true]
        y_pred_list = [IDX_TO_LABEL[int(x)] if isinstance(x, (int, float)) else str(x) for x in y_pred]
        weighted_f1 = compute_weighted_f1(y_true_list, y_pred_list)
        total = len(y_true_list)
        correct = sum(1 for g, p in zip(y_true_list, y_pred_list) if g == p)
        res = {
            "weighted_f1": weighted_f1,
            "accuracy": correct / total if total > 0 else 0.0,
        }

        if pair_indices is not None and len(pair_indices) > 0:
            soft_count = 0
            hard_count = 0
            p_total = len(pair_indices)
            for idx_pair in pair_indices:
                i1, i2 = idx_pair[0], idx_pair[1]
                p1, p2 = y_pred_list[i1], y_pred_list[i2]
                t1, t2 = y_true_list[i1], y_true_list[i2]

                soft_ok = p1 in REVERSIBLE_LABELS and p2 == REVERSE_LABEL_MAP.get(p1)
                hard_ok = p1 == t1 and p2 == t2
                if soft_ok:
                    soft_count += 1
                if hard_ok:
                    hard_count += 1

            res["soft_cons"] = soft_count / p_total if p_total > 0 else 0.0
            res["hard_cons"] = hard_count / p_total if p_total > 0 else 0.0

        return res
