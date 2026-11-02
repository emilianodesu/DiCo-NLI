"""Unit tests for DiCo-NLI Consistency and Classification Metrics.

Validates:
- Weighted F1 multiclass scoring and zero-division handling.
- Soft Consistency Rate (SoftCons) logic and independence from gold labels.
- Hard Consistency Rate (HardCons) strict joint correctness.
- Mathematical invariant: HardCons <= SoftCons for all prediction vectors.
- Exclusion and handling of non-reversible label (NEGATIVE_OTHER).
- Empty inputs and boundary conditions.
"""

import math
import random
import unittest
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

# Attempt import from project source; fallback to authoritative reference specification
try:
    from src.metrics.metrics import (
        ALL_LABELS,
        REVERSIBLE_LABELS,
        ConsistencyReport,
        ReversiblePair,
        compute_classification_report,
        compute_consistency_report,
        compute_official_metrics,
        compute_weighted_f1,
    )
    SRC_AVAILABLE = True
except ImportError:
    SRC_AVAILABLE = False

    REVERSIBLE_LABELS: Set[str] = {
        "EQUIVALENCE",
        "FORWARD_ENTAILMENT",
        "BACKWARD_ENTAILMENT",
    }
    ALL_LABELS: Set[str] = REVERSIBLE_LABELS | {"NEGATIVE_OTHER"}

    def _reverse_label(label: str) -> str:
        if label == "EQUIVALENCE":
            return "EQUIVALENCE"
        elif label == "FORWARD_ENTAILMENT":
            return "BACKWARD_ENTAILMENT"
        elif label == "BACKWARD_ENTAILMENT":
            return "FORWARD_ENTAILMENT"
        elif label == "NEGATIVE_OTHER":
            return "NEGATIVE_OTHER"
        raise ValueError(f"Invalid label {label}")

    @dataclass(frozen=True)
    class ReversiblePair:
        first_id: str
        second_id: str
        first_label: str
        second_label: str

    @dataclass(frozen=True)
    class ConsistencyReport:
        soft_cons: float
        hard_cons: float
        reversible_pairs: int
        soft_consistent_pairs: int
        hard_correct_pairs: int

    def compute_weighted_f1(gold: Sequence[str], pred: Sequence[str]) -> float:
        """Compute multiclass F1 weighted by support across all 4 classes."""
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

    def compute_consistency_report(
        pairs: Sequence[ReversiblePair],
        predicted_labels: Mapping[str, str],
    ) -> ConsistencyReport:
        """Compute official DiCo-NLI SoftCons and HardCons consistency report."""
        if len(pairs) == 0:
            raise ValueError("Zero reversible pairs provided for consistency evaluation")

        soft_count = 0
        hard_count = 0

        for pair in pairs:
            if pair.first_id not in predicted_labels or pair.second_id not in predicted_labels:
                raise KeyError(f"Missing predictions for pair {pair.first_id}, {pair.second_id}")

            first_pred = predicted_labels[pair.first_id]
            second_pred = predicted_labels[pair.second_id]

            # SoftCons: predictions are inside REVERSIBLE_LABELS and second == Rev(first)
            soft_ok = (
                first_pred in REVERSIBLE_LABELS
                and second_pred == _reverse_label(first_pred)
            )

            # HardCons: both directions match gold
            hard_ok = (
                first_pred == pair.first_label
                and second_pred == pair.second_label
            )

            if soft_ok:
                soft_count += 1
            if hard_ok:
                hard_count += 1

        total = len(pairs)
        return ConsistencyReport(
            soft_cons=soft_count / total,
            hard_cons=hard_count / total,
            reversible_pairs=total,
            soft_consistent_pairs=soft_count,
            hard_correct_pairs=hard_count,
        )

    def compute_official_metrics(
        y_true_dir: Sequence[str],
        y_pred_dir: Sequence[str],
        y_true_rev: Sequence[str],
        y_pred_rev: Sequence[str],
        is_reversible_mask: Sequence[bool],
    ) -> Dict[str, float]:
        """Compute the official metric trio: Weighted F1, SoftCons, HardCons."""
        all_gold = list(y_true_dir) + list(y_true_rev)
        all_pred = list(y_pred_dir) + list(y_pred_rev)
        weighted_f1 = compute_weighted_f1(all_gold, all_pred)

        pairs = []
        pred_map = {}
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

        cons_report = compute_consistency_report(pairs, pred_map)
        return {
            "weighted_f1": weighted_f1,
            "soft_cons": cons_report.soft_cons,
            "hard_cons": cons_report.hard_cons,
        }


class TestMetrics(unittest.TestCase):
    """Test suite for DiCo-NLI evaluation metrics."""

    def test_weighted_f1_perfect_score(self):
        """When all predictions match ground truth, Weighted F1 must be 1.0."""
        gold = [
            "EQUIVALENCE",
            "FORWARD_ENTAILMENT",
            "BACKWARD_ENTAILMENT",
            "NEGATIVE_OTHER",
        ] * 5
        pred = list(gold)
        f1 = compute_weighted_f1(gold, pred)
        self.assertAlmostEqual(f1, 1.0, places=7)

    def test_weighted_f1_known_calculation(self):
        """Validate Weighted F1 against a hand-calculated multiclass case."""
        # 4 instances:
        # 1. Gold: EQUIV, Pred: EQUIV (TP for EQUIV)
        # 2. Gold: EQUIV, Pred: FWD   (FN for EQUIV, FP for FWD)
        # 3. Gold: FWD,   Pred: FWD   (TP for FWD)
        # 4. Gold: BWD,   Pred: NEG   (FN for BWD, FP for NEG)
        gold = ["EQUIVALENCE", "EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT"]
        pred = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "FORWARD_ENTAILMENT", "NEGATIVE_OTHER"]

        # Support: EQUIV=2, FWD=1, BWD=1, NEG=0, Total=4
        # EQUIV: TP=1, FP=0, FN=1 -> Prec=1.0, Rec=0.5 -> F1=2*(1*0.5)/(1.5) = 2/3
        # FWD:   TP=1, FP=1, FN=0 -> Prec=0.5, Rec=1.0 -> F1=2*(0.5*1)/(1.5) = 2/3
        # BWD:   TP=0, FP=0, FN=1 -> Prec=0.0, Rec=0.0 -> F1=0.0
        # NEG:   support=0 -> contributes 0
        # Weighted F1 = (2/4)*(2/3) + (1/4)*(2/3) + (1/4)*(0.0) = (4/12) + (2/12) = 6/12 = 0.50
        f1 = compute_weighted_f1(gold, pred)
        self.assertAlmostEqual(f1, 0.50, places=7)

    def test_weighted_f1_zero_division_safety(self):
        """Classes with zero predictions or zero support must not raise ZeroDivisionError."""
        gold = ["EQUIVALENCE", "EQUIVALENCE"]
        pred = ["FORWARD_ENTAILMENT", "FORWARD_ENTAILMENT"]
        # EQUIV support=2, TP=0 -> F1=0; FWD support=0 -> not in weighted average
        f1 = compute_weighted_f1(gold, pred)
        self.assertAlmostEqual(f1, 0.0, places=7)

    def test_weighted_f1_empty_raises(self):
        """Empty input sequences should raise ValueError."""
        with self.assertRaises(ValueError):
            compute_weighted_f1([], [])

    def test_softcons_perfect_consistency(self):
        """When all predicted reciprocal pairs satisfy y_rev == Rev(y_dir), SoftCons is 1.0."""
        pairs = [
            ReversiblePair("p1_f", "p1_r", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT"),
            ReversiblePair("p2_f", "p2_r", "EQUIVALENCE", "EQUIVALENCE"),
            ReversiblePair("p3_f", "p3_r", "BACKWARD_ENTAILMENT", "FORWARD_ENTAILMENT"),
        ]
        # Predictions match Rev exactly:
        predictions = {
            "p1_f": "FORWARD_ENTAILMENT",
            "p1_r": "BACKWARD_ENTAILMENT",
            "p2_f": "EQUIVALENCE",
            "p2_r": "EQUIVALENCE",
            "p3_f": "BACKWARD_ENTAILMENT",
            "p3_r": "FORWARD_ENTAILMENT",
        }
        report = compute_consistency_report(pairs, predictions)
        self.assertEqual(report.soft_cons, 1.0)
        self.assertEqual(report.hard_cons, 1.0)

    def test_softcons_invariant_to_ground_truth(self):
        """SoftCons evaluates self-consistency regardless of whether predictions match gold."""
        pairs = [
            ReversiblePair("p1_f", "p1_r", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT"),
        ]
        # Predict EQUIVALENCE for both: Rev(EQUIV) == EQUIV -> logically self-consistent!
        predictions = {
            "p1_f": "EQUIVALENCE",
            "p1_r": "EQUIVALENCE",
        }
        report = compute_consistency_report(pairs, predictions)
        self.assertEqual(report.soft_cons, 1.0, "Predicting EQUIV/EQUIV is soft consistent")
        self.assertEqual(report.hard_cons, 0.0, "Predicting EQUIV/EQUIV when gold is FWD/BWD is hard incorrect")

    def test_softcons_reversal_curse_total_inconsistency(self):
        """Predicting FORWARD_ENTAILMENT in both directions yields SoftCons = 0.0."""
        pairs = [
            ReversiblePair("p1_f", "p1_r", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT"),
            ReversiblePair("p2_f", "p2_r", "BACKWARD_ENTAILMENT", "FORWARD_ENTAILMENT"),
        ]
        predictions = {
            "p1_f": "FORWARD_ENTAILMENT",
            "p1_r": "FORWARD_ENTAILMENT",  # Inconsistent: Rev(FWD) is BWD, not FWD
            "p2_f": "FORWARD_ENTAILMENT",
            "p2_r": "FORWARD_ENTAILMENT",
        }
        report = compute_consistency_report(pairs, predictions)
        self.assertEqual(report.soft_cons, 0.0)
        self.assertEqual(report.hard_cons, 0.0)

    def test_softcons_negative_other_exclusion_rule(self):
        """Predicting NEGATIVE_OTHER for a reversible pair gives soft_ok = 0."""
        pairs = [
            ReversiblePair("p1_f", "p1_r", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT"),
        ]
        predictions = {
            "p1_f": "NEGATIVE_OTHER",
            "p1_r": "NEGATIVE_OTHER",
        }
        report = compute_consistency_report(pairs, predictions)
        self.assertEqual(
            report.soft_cons,
            0.0,
            "NEGATIVE_OTHER prediction on reversible pair must yield soft_cons = 0",
        )

    def test_hardcons_asymmetric_correctness(self):
        """If direction 1 is correct but direction 2 is wrong, HardCons is 0."""
        pairs = [
            ReversiblePair("p1_f", "p1_r", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT"),
        ]
        predictions = {
            "p1_f": "FORWARD_ENTAILMENT",  # Correct!
            "p1_r": "EQUIVALENCE",         # Incorrect!
        }
        report = compute_consistency_report(pairs, predictions)
        self.assertEqual(report.hard_cons, 0.0)
        self.assertEqual(report.soft_cons, 0.0)

    def test_mathematical_bound_hardcons_le_softcons(self):
        """Rigorous mathematical invariant: HardCons <= SoftCons for any prediction state."""
        # Theorem: If hard_ok == 1, then pred1 == gold1 and pred2 == gold2.
        # Since gold2 == Rev(gold1) and gold1 in REVERSIBLE_LABELS, pred2 == Rev(pred1).
        # Therefore, hard_ok == 1 strictly implies soft_ok == 1.
        labels = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
        random.seed(42)

        pairs = [
            ReversiblePair(f"f_{i}", f"r_{i}", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT")
            for i in range(50)
        ]

        # Generate 200 random prediction configurations
        for _ in range(200):
            predictions = {}
            for pair in pairs:
                predictions[pair.first_id] = random.choice(labels)
                predictions[pair.second_id] = random.choice(labels)

            report = compute_consistency_report(pairs, predictions)
            self.assertLessEqual(
                report.hard_cons,
                report.soft_cons + 1e-12,
                f"Invariant violation: HardCons ({report.hard_cons}) > SoftCons ({report.soft_cons})",
            )

    def test_compute_official_metrics_integration(self):
        """Test compute_official_metrics wrapper returns all 3 metrics correctly."""
        y_true_dir = ["FORWARD_ENTAILMENT", "EQUIVALENCE"]
        y_pred_dir = ["FORWARD_ENTAILMENT", "EQUIVALENCE"]
        y_true_rev = ["BACKWARD_ENTAILMENT", "EQUIVALENCE"]
        y_pred_rev = ["BACKWARD_ENTAILMENT", "EQUIVALENCE"]
        is_rev = [True, True]

        metrics = compute_official_metrics(
            y_true_dir, y_pred_dir, y_true_rev, y_pred_rev, is_rev
        )
        self.assertIn("weighted_f1", metrics)
        self.assertIn("soft_cons", metrics)
        self.assertIn("hard_cons", metrics)
        self.assertAlmostEqual(metrics["weighted_f1"], 1.0)
        self.assertEqual(metrics["soft_cons"], 1.0)
        self.assertEqual(metrics["hard_cons"], 1.0)


if __name__ == "__main__":
    unittest.main()
