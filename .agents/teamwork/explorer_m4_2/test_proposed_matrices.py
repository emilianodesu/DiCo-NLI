"""
Unit tests for .agents/teamwork/explorer_m4_2/proposed_matrices.py.
Verifies all mathematical properties, formatting, data integration, and edge cases.
"""

import json
import math
import os
import tempfile
import unittest

from proposed_matrices import (
    CANONICAL_LABELS,
    IDEAL_INVOLUTION_MATRIX,
    ConfusionMatrixReport,
    DirectionalTransitionReport,
    compute_confusion_matrix,
    compute_directional_transition_matrix,
    evaluate_matrices_from_data,
    extract_reciprocal_pairs_from_dataset,
    get_ideal_involution_matrix,
    load_predictions,
)


class TestProposedMatrices(unittest.TestCase):
    """Test suite for Confusion Matrix and Directional Transition Matrix."""

    def test_ideal_involution_matrix_properties(self):
        """Verify algebraic properties of involution matrix P."""
        # For canonical labels: EQU, FWD, BWD, NEG
        P = get_ideal_involution_matrix(CANONICAL_LABELS)
        expected = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        self.assertEqual(P, expected)

        # Involution property: P * P = Identity
        P2 = [
            [sum(P[i][k] * P[k][j] for k in range(4)) for j in range(4)]
            for i in range(4)
        ]
        identity = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
        self.assertEqual(P2, identity)

        # For BWD-first labels:
        bwd_first = ("BACKWARD_ENTAILMENT", "EQUIVALENCE", "FORWARD_ENTAILMENT", "NEGATIVE_OTHER")
        P_bwd = get_ideal_involution_matrix(bwd_first)
        expected_bwd = [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        self.assertEqual(P_bwd, expected_bwd)

    def test_confusion_matrix_synthetic(self):
        """Verify confusion matrix counts, normalized matrices, and per-class metrics."""
        # Ground truth: 10 EQU, 10 FWD, 10 BWD, 10 NEG
        # Preds:
        # EQU (10): 8 EQU, 1 FWD, 1 NEG
        # FWD (10): 7 FWD, 2 BWD, 1 NEG
        # BWD (10): 9 BWD, 1 FWD
        # NEG (10): 10 NEG
        gold = (
            ["EQUIVALENCE"] * 10
            + ["FORWARD_ENTAILMENT"] * 10
            + ["BACKWARD_ENTAILMENT"] * 10
            + ["NEGATIVE_OTHER"] * 10
        )
        pred = (
            ["EQUIVALENCE"] * 8 + ["FORWARD_ENTAILMENT"] * 1 + ["NEGATIVE_OTHER"] * 1
            + ["FORWARD_ENTAILMENT"] * 7 + ["BACKWARD_ENTAILMENT"] * 2 + ["NEGATIVE_OTHER"] * 1
            + ["BACKWARD_ENTAILMENT"] * 9 + ["FORWARD_ENTAILMENT"] * 1
            + ["NEGATIVE_OTHER"] * 10
        )

        rep = compute_confusion_matrix(gold, pred)

        # Total instances
        self.assertEqual(rep.total_instances, 40)

        # Raw matrix: Row 0 (EQU)
        self.assertEqual(rep.raw_matrix[0], [8, 1, 0, 1])
        # Row 1 (FWD)
        self.assertEqual(rep.raw_matrix[1], [0, 7, 2, 1])
        # Row 2 (BWD)
        self.assertEqual(rep.raw_matrix[2], [0, 1, 9, 0])
        # Row 3 (NEG)
        self.assertEqual(rep.raw_matrix[3], [0, 0, 0, 10])

        # Accuracy: (8 + 7 + 9 + 10) / 40 = 34 / 40 = 0.85
        self.assertAlmostEqual(rep.accuracy, 0.85)

        # Per-class checks for EQU:
        # TP = 8, FP = 0, FN = 2, TN = 30
        m_equ = rep.per_class["EQUIVALENCE"]
        self.assertEqual(m_equ.tp, 8)
        self.assertEqual(m_equ.fp, 0)
        self.assertEqual(m_equ.fn, 2)
        self.assertEqual(m_equ.support, 10)
        self.assertEqual(m_equ.predicted_count, 8)
        self.assertAlmostEqual(m_equ.precision, 1.0)
        self.assertAlmostEqual(m_equ.recall, 0.8)
        self.assertAlmostEqual(m_equ.f1, 2 * 1.0 * 0.8 / 1.8)

        # Total normalized matrix sum = 1.0
        tot_sum = sum(sum(row) for row in rep.total_normalized)
        self.assertAlmostEqual(tot_sum, 1.0)

        # Row normalized rows sum = 1.0
        for row in rep.row_normalized:
            self.assertAlmostEqual(sum(row), 1.0)

        # ASCII and LaTeX formatting
        ascii_tbl = rep.format_ascii_table()
        self.assertIn("Multiclass Confusion Matrix", ascii_tbl)
        self.assertIn("EQU", ascii_tbl)
        self.assertIn("Summary: Accuracy: 0.8500", ascii_tbl)

        latex_tbl = rep.format_latex_table()
        self.assertIn("\\begin{table}", latex_tbl)
        self.assertIn("\\textbf{EQU}", latex_tbl)
        self.assertIn("\\end{table}", latex_tbl)

        # JSON serialization
        rep_json = rep.to_json()
        parsed = json.loads(rep_json)
        self.assertEqual(parsed["total_instances"], 40)
        self.assertEqual(parsed["accuracy"], 0.85)

    def test_confusion_matrix_zero_division(self):
        """Verify zero support and zero predictions handle division gracefully."""
        # Only EQUIVALENCE in gold and predictions
        gold = ["EQUIVALENCE"] * 5
        pred = ["EQUIVALENCE"] * 5

        rep = compute_confusion_matrix(gold, pred)
        self.assertEqual(rep.accuracy, 1.0)

        # FWD has 0 support and 0 predictions
        m_fwd = rep.per_class["FORWARD_ENTAILMENT"]
        self.assertEqual(m_fwd.support, 0)
        self.assertEqual(m_fwd.predicted_count, 0)
        self.assertEqual(m_fwd.precision, 0.0)
        self.assertEqual(m_fwd.recall, 0.0)
        self.assertEqual(m_fwd.f1, 0.0)

    def test_confusion_matrix_validation_errors(self):
        """Verify error checking on inputs."""
        with self.assertRaises(ValueError):
            compute_confusion_matrix([], [])

        with self.assertRaises(ValueError):
            compute_confusion_matrix(["EQUIVALENCE"], ["EQUIVALENCE", "FORWARD_ENTAILMENT"])

        with self.assertRaises(ValueError):
            compute_confusion_matrix(["INVALID_LABEL"], ["EQUIVALENCE"])

    def test_transition_matrix_ideal(self):
        """Verify ideal involution transition yields 0 violation and distance 0."""
        # 10 pairs adhering strictly to involution P:
        # 3 EQU -> EQU (0, 0)
        # 3 FWD -> BWD (1, 2)
        # 2 BWD -> FWD (2, 1)
        # 2 NEG -> NEG (3, 3)
        pairs = [
            ("p1_d", "p1_r"), ("p2_d", "p2_r"), ("p3_d", "p3_r"),
            ("p4_d", "p4_r"), ("p5_d", "p5_r"), ("p6_d", "p6_r"),
            ("p7_d", "p7_r"), ("p8_d", "p8_r"),
            ("p9_d", "p9_r"), ("p10_d", "p10_r"),
        ]
        predictions = {
            "p1_d": "EQUIVALENCE", "p1_r": "EQUIVALENCE",
            "p2_d": "EQUIVALENCE", "p2_r": "EQUIVALENCE",
            "p3_d": "EQUIVALENCE", "p3_r": "EQUIVALENCE",
            "p4_d": "FORWARD_ENTAILMENT", "p4_r": "BACKWARD_ENTAILMENT",
            "p5_d": "FORWARD_ENTAILMENT", "p5_r": "BACKWARD_ENTAILMENT",
            "p6_d": "FORWARD_ENTAILMENT", "p6_r": "BACKWARD_ENTAILMENT",
            "p7_d": "BACKWARD_ENTAILMENT", "p7_r": "FORWARD_ENTAILMENT",
            "p8_d": "BACKWARD_ENTAILMENT", "p8_r": "FORWARD_ENTAILMENT",
            "p9_d": "NEGATIVE_OTHER", "p9_r": "NEGATIVE_OTHER",
            "p10_d": "NEGATIVE_OTHER", "p10_r": "NEGATIVE_OTHER",
        }

        rep = compute_directional_transition_matrix(pairs, predictions)

        # Inconsistency metrics
        m = rep.metrics
        self.assertEqual(m.total_pairs, 10)
        self.assertEqual(m.consistent_pairs, 10)
        self.assertAlmostEqual(m.consistent_rate, 1.0)
        self.assertEqual(m.violation_pairs, 0)
        self.assertAlmostEqual(m.violation_rate, 0.0)
        self.assertAlmostEqual(m.frobenius_distance, 0.0)
        self.assertAlmostEqual(m.normalized_frobenius, 0.0)
        self.assertAlmostEqual(m.mean_tvd, 0.0)
        self.assertAlmostEqual(m.mean_transition_entropy, 0.0)
        self.assertAlmostEqual(m.reversal_curse_index, 0.0)

        # Exporter outputs
        ascii_tbl = rep.format_ascii_table()
        self.assertIn("Directional Transition Matrix", ascii_tbl)
        self.assertIn("Consistent Pairs (SoftCons):      10 (100.0000%)", ascii_tbl)

        latex_tbl = rep.format_latex_table()
        self.assertIn("\\begin{table}", latex_tbl)
        self.assertIn("\\textbf{3}", latex_tbl)
        self.assertIn("SoftCons", latex_tbl)
        self.assertIn("100.0\\%", latex_tbl)

    def test_transition_matrix_reversal_curse_loop(self):
        """Verify reversal curse entrapment (FWD->FWD and BWD->BWD)."""
        pairs = [
            ("p1_d", "p1_r"), ("p2_d", "p2_r"),
            ("p3_d", "p3_r"), ("p4_d", "p4_r"),
        ]
        predictions = {
            "p1_d": "FORWARD_ENTAILMENT", "p1_r": "FORWARD_ENTAILMENT", # trap
            "p2_d": "FORWARD_ENTAILMENT", "p2_r": "FORWARD_ENTAILMENT", # trap
            "p3_d": "BACKWARD_ENTAILMENT", "p3_r": "BACKWARD_ENTAILMENT", # trap
            "p4_d": "BACKWARD_ENTAILMENT", "p4_r": "BACKWARD_ENTAILMENT", # trap
        }

        rep = compute_directional_transition_matrix(pairs, predictions)
        m = rep.metrics
        self.assertEqual(m.fwd_fwd_trap, 2)
        self.assertEqual(m.bwd_bwd_trap, 2)
        self.assertAlmostEqual(m.reversal_curse_index, 1.0)
        self.assertEqual(m.violation_pairs, 4)
        self.assertAlmostEqual(m.violation_rate, 1.0)
        self.assertGreater(m.frobenius_distance, 0.0)

    def test_data_integration_codabench_csv(self):
        """Verify load_predictions reads 2-column Codabench CSV files properly."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
            tmp.write("instance_id,label\n")
            tmp.write("inst_001,FORWARD_ENTAILMENT\n")
            tmp.write("inst_002,BACKWARD_ENTAILMENT\n")
            tmp.write("inst_003,EQUIVALENCE\n")
            tmp.write("inst_004,NEGATIVE_OTHER")
            tmp_path = tmp.name

        try:
            loaded = load_predictions(tmp_path)
            self.assertEqual(len(loaded), 4)
            self.assertEqual(loaded["inst_001"], "FORWARD_ENTAILMENT")
            self.assertEqual(loaded["inst_002"], "BACKWARD_ENTAILMENT")
            self.assertEqual(loaded["inst_003"], "EQUIVALENCE")
            self.assertEqual(loaded["inst_004"], "NEGATIVE_OTHER")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_evaluate_matrices_from_data_pipeline(self):
        """Verify unified evaluation from dataset records and prediction file."""
        dataset = [
            {"instance_id": "i1_dir", "pair_id": "pair_01", "label": "FORWARD_ENTAILMENT"},
            {"instance_id": "i1_rev", "pair_id": "pair_01", "label": "BACKWARD_ENTAILMENT"},
            {"instance_id": "i2_dir", "pair_id": "pair_02", "label": "EQUIVALENCE"},
            {"instance_id": "i2_rev", "pair_id": "pair_02", "label": "EQUIVALENCE"},
        ]
        predictions = {
            "i1_dir": "FORWARD_ENTAILMENT",
            "i1_rev": "BACKWARD_ENTAILMENT",
            "i2_dir": "EQUIVALENCE",
            "i2_rev": "EQUIVALENCE",
        }

        conf_rep, trans_rep = evaluate_matrices_from_data(dataset, predictions)
        self.assertEqual(conf_rep.total_instances, 4)
        self.assertEqual(conf_rep.accuracy, 1.0)
        self.assertEqual(trans_rep.total_pairs, 2)
        self.assertAlmostEqual(trans_rep.metrics.consistent_rate, 1.0)

    def test_save_json_roundtrip(self):
        """Verify save_json and disk persistence for both reports."""
        gold = ["EQUIVALENCE", "FORWARD_ENTAILMENT"]
        pred = ["EQUIVALENCE", "BACKWARD_ENTAILMENT"]
        conf_rep = compute_confusion_matrix(gold, pred)

        pairs = [("id1", "id2")]
        preds = {"id1": "EQUIVALENCE", "id2": "EQUIVALENCE"}
        trans_rep = compute_directional_transition_matrix(pairs, preds)

        with tempfile.TemporaryDirectory() as tmp_dir:
            conf_path = os.path.join(tmp_dir, "conf.json")
            trans_path = os.path.join(tmp_dir, "trans.json")

            conf_rep.save_json(conf_path)
            trans_rep.save_json(trans_path)

            self.assertTrue(os.path.isfile(conf_path))
            self.assertTrue(os.path.isfile(trans_path))

            with open(conf_path, "r", encoding="utf-8") as f:
                c_data = json.load(f)
                self.assertEqual(c_data["total_instances"], 2)

            with open(trans_path, "r", encoding="utf-8") as f:
                t_data = json.load(f)
                self.assertEqual(t_data["total_pairs"], 1)

    def test_simple_dataframe_integration(self):
        """Verify compatibility with SimpleDataFrame from src.data.loader."""
        import sys
        sys.path.insert(0, "/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject")
        from src.data.loader import SimpleDataFrame

        records = [
            {"instance_id": "i1_dir", "pair_id": "p1", "reverse_pair_id": "i1_rev", "label": "FORWARD_ENTAILMENT"},
            {"instance_id": "i1_rev", "pair_id": "p1", "reverse_pair_id": "i1_dir", "label": "BACKWARD_ENTAILMENT"},
        ]
        df = SimpleDataFrame(records)

        # Test pair extraction
        pairs = extract_reciprocal_pairs_from_dataset(df)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0], ("i1_dir", "i1_rev"))

        # Test prediction loading
        pred_records = [
            {"instance_id": "i1_dir", "label": "FORWARD_ENTAILMENT"},
            {"instance_id": "i1_rev", "label": "BACKWARD_ENTAILMENT"},
        ]
        pred_df = SimpleDataFrame(pred_records)
        pred_map = load_predictions(pred_df)
        self.assertEqual(pred_map["i1_dir"], "FORWARD_ENTAILMENT")
        self.assertEqual(pred_map["i1_rev"], "BACKWARD_ENTAILMENT")

    def test_taxonomy_breakdown_cells(self):
        """Verify detailed cell counts for equivalence asymmetry and noise collapse."""
        pairs = [
            ("p1_d", "p1_r"), # EQU -> FWD (Equiv Asymmetry)
            ("p2_d", "p2_r"), # FWD -> NEG (Noise Collapse)
        ]
        preds = {
            "p1_d": "EQUIVALENCE", "p1_r": "FORWARD_ENTAILMENT",
            "p2_d": "FORWARD_ENTAILMENT", "p2_r": "NEGATIVE_OTHER",
        }
        rep = compute_directional_transition_matrix(pairs, preds)
        m = rep.metrics
        self.assertEqual(m.total_pairs, 2)
        self.assertEqual(m.consistent_pairs, 0)
        self.assertEqual(m.violation_pairs, 2)
        self.assertGreaterEqual(m.equiv_asymmetry_count, 1)
        self.assertGreaterEqual(m.noise_collapse_count, 1)


if __name__ == "__main__":
    unittest.main()

