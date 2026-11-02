"""
Unit tests for src/metrics/consistency.py (Directional Transition Matrix & Inconsistency Taxonomy).
"""

import unittest
from src.metrics.consistency import (
    DirectionalTransitionReport,
    InconsistencyTaxonomyBreakdown,
    compute_consistency_by_category,
    compute_directional_transition_matrix,
)
from src.metrics.metrics import ReversiblePair


class TestConsistencyAnalysis(unittest.TestCase):
    """Test suite for Directional Transition Matrix and Inconsistency Taxonomy."""

    def test_transition_matrix_known_counts(self):
        """Verify transition matrix and taxonomy on a known synthetic configuration."""
        pairs = [
            # 1. Consistent FWD -> BWD
            ReversiblePair("p1_d", "p1_r", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT"),
            # 2. Consistent BWD -> FWD
            ReversiblePair("p2_d", "p2_r", "BACKWARD_ENTAILMENT", "FORWARD_ENTAILMENT"),
            # 3. Consistent EQUIV -> EQUIV
            ReversiblePair("p3_d", "p3_r", "EQUIVALENCE", "EQUIVALENCE"),
            # 4. Entailment Loop (FWD -> FWD)
            ReversiblePair("p4_d", "p4_r", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT"),
            # 5. Equivalence Asymmetry (EQUIV -> FWD)
            ReversiblePair("p5_d", "p5_r", "EQUIVALENCE", "EQUIVALENCE"),
            # 6. Noise Dropout (FWD -> NEGATIVE_OTHER)
            ReversiblePair("p6_d", "p6_r", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT"),
        ]

        predictions = {
            "p1_d": "FORWARD_ENTAILMENT",
            "p1_r": "BACKWARD_ENTAILMENT",
            "p2_d": "BACKWARD_ENTAILMENT",
            "p2_r": "FORWARD_ENTAILMENT",
            "p3_d": "EQUIVALENCE",
            "p3_r": "EQUIVALENCE",
            "p4_d": "FORWARD_ENTAILMENT",
            "p4_r": "FORWARD_ENTAILMENT",
            "p5_d": "EQUIVALENCE",
            "p5_r": "FORWARD_ENTAILMENT",
            "p6_d": "FORWARD_ENTAILMENT",
            "p6_r": "NEGATIVE_OTHER",
        }

        report = compute_directional_transition_matrix(pairs, predictions)

        # Total pairs
        self.assertEqual(report.taxonomy.total_pairs, 6)

        # Consistent transitions
        self.assertEqual(report.taxonomy.consistent_count, 3)
        self.assertEqual(report.taxonomy.fwd_to_bwd, 1)
        self.assertEqual(report.taxonomy.bwd_to_fwd, 1)
        self.assertEqual(report.taxonomy.equiv_to_equiv, 1)
        self.assertAlmostEqual(report.taxonomy.consistent_rate, 3 / 6)

        # Entailment loop
        self.assertEqual(report.taxonomy.entailment_loop_count, 1)
        self.assertEqual(report.taxonomy.fwd_fwd_bias, 1)
        self.assertAlmostEqual(report.taxonomy.entailment_loop_rate, 1 / 6)

        # Equivalence asymmetry
        self.assertEqual(report.taxonomy.equiv_asymmetry_count, 1)
        self.assertEqual(report.taxonomy.equiv_to_dir, 1)

        # Noise collapse
        self.assertEqual(report.taxonomy.noise_collapse_count, 1)
        self.assertEqual(report.taxonomy.dir_to_noise, 1)

        # Invariant: sum of all 4 categories must equal total pairs
        tax = report.taxonomy
        sum_counts = (
            tax.consistent_count
            + tax.entailment_loop_count
            + tax.equiv_asymmetry_count
            + tax.noise_collapse_count
        )
        self.assertEqual(sum_counts, tax.total_pairs)

        sum_rates = (
            tax.consistent_rate
            + tax.entailment_loop_rate
            + tax.equiv_asymmetry_rate
            + tax.noise_collapse_rate
        )
        self.assertAlmostEqual(sum_rates, 1.0, places=7)

    def test_ascii_and_latex_formatting(self):
        """Verify ASCII and LaTeX table formatters produce valid strings."""
        pairs = [
            ReversiblePair("p1_d", "p1_r", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT"),
            ReversiblePair("p2_d", "p2_r", "EQUIVALENCE", "EQUIVALENCE"),
        ]
        predictions = {
            "p1_d": "FORWARD_ENTAILMENT",
            "p1_r": "BACKWARD_ENTAILMENT",
            "p2_d": "EQUIVALENCE",
            "p2_r": "EQUIVALENCE",
        }
        report = compute_directional_transition_matrix(pairs, predictions)

        ascii_table = report.format_ascii_table()
        self.assertIn("Direct \\ Reverse", ascii_table)
        self.assertIn("BWD (0)", ascii_table)
        self.assertIn("Total", ascii_table)

        latex_table = report.format_latex_table(caption="Test Transition")
        self.assertIn("\\begin{table}", latex_table)
        self.assertIn("\\caption{Test Transition}", latex_table)
        self.assertIn("\\end{table}", latex_table)

    def test_compute_consistency_by_category(self):
        """Verify disaggregated metrics partitioned by syntactic category."""
        pairs = [
            ReversiblePair("p1_d", "p1_r", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT"),
            ReversiblePair("p2_d", "p2_r", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT"),
            ReversiblePair("p3_d", "p3_r", "EQUIVALENCE", "EQUIVALENCE"),
        ]
        categories = {
            ("p1_d", "p1_r"): "nominal",
            ("p2_d", "p2_r"): "verbal",
            ("p3_d", "p3_r"): "nominal",
        }
        predictions = {
            "p1_d": "FORWARD_ENTAILMENT",
            "p1_r": "BACKWARD_ENTAILMENT",  # consistent
            "p2_d": "FORWARD_ENTAILMENT",
            "p2_r": "FORWARD_ENTAILMENT",  # loop
            "p3_d": "EQUIVALENCE",
            "p3_r": "EQUIVALENCE",          # consistent
        }

        breakdown = compute_consistency_by_category(pairs, categories, predictions)
        self.assertIn("nominal", breakdown)
        self.assertIn("verbal", breakdown)

        self.assertEqual(breakdown["nominal"]["count"], 2)
        self.assertEqual(breakdown["nominal"]["soft_cons"], 1.0)

        self.assertEqual(breakdown["verbal"]["count"], 1)
        self.assertEqual(breakdown["verbal"]["soft_cons"], 0.0)
        self.assertEqual(breakdown["verbal"]["entailment_loop_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
