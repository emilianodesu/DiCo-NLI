"""
tests/test_evaluation.py

Comprehensive Unit and Integration Test Suite for SemEval-2027 Task 2 (DiCo-NLI)
Milestone 4: Evaluation Tables, Metric Aggregation, Directional Matrices,
Syntactic Category Error Analysis, and Standalone CLI Engine.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List

def _find_project_root() -> Path:
    curr = Path(__file__).resolve().parent
    for _ in range(6):
        if (curr / "src").is_dir() or (curr / "PROJECT.md").is_file():
            return curr
        curr = curr.parent
    return Path.cwd()

PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Evaluation module imports
from src.evaluation.tables import (
    CANONICAL_METHODS,
    ComparisonTable,
    EvaluationResult,
    aggregate_results_from_dir,
    build_comparison_table,
    evaluate_prediction_file,
    evaluate_prediction_mapping,
    format_comparison_table,
    generate_ascii_table,
    generate_latex_table,
    generate_markdown_table,
    load_predictions_from_csv,
    parse_run_metadata,
)
from src.evaluation.matrices import (
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
from src.evaluation.analysis import (
    CategoryMetrics,
    CrossMethodCategoryReport,
    DisaggregatedAnalysisReport,
    ErrorCase,
    ErrorType,
    analyze_syntactic_categories,
    compare_methods_syntactic,
    compute_category_metrics,
    diagnose_error,
    export_analysis_artifacts,
    extract_qualitative_error_cases,
    format_latex_tables,
    format_markdown_report,
)
from experiments.evaluate import (
    parse_args,
    run_dynamic_model_evaluation,
    run_evaluation,
    validate_cli_args,
)
from src.data.loader import (
    InstanceRecord,
    SimpleDataFrame,
    SyntacticCategory,
    generate_mock_phrasis_data,
    load_phrasis_split,
)
from src.data.reversal import pair_instances
from src.data.codabench import export_codabench_submission


# =============================================================================
# 1. Master Comparison Tables & Aggregation Tests
# =============================================================================

class TestComparisonTables(unittest.TestCase):
    """Test suite covering EvaluationResult and ComparisonTable functionality."""

    def setUp(self) -> None:
        self.sample_result_baseline = EvaluationResult(
            track="track1",
            method="baseline",
            model="DeBERTa-v3",
            split="dev",
            weighted_f1=0.7500,
            soft_cons=0.6000,
            hard_cons=0.5500,
            num_instances=100,
            num_pairs=40,
        )
        self.sample_result_sd = EvaluationResult(
            track="track1",
            method="sd",
            model="DeBERTa-v3",
            split="dev",
            weighted_f1=0.8200,
            soft_cons=1.0000,
            hard_cons=0.7800,
            num_instances=100,
            num_pairs=40,
        )
        self.sample_result_t2 = EvaluationResult(
            track="track2",
            method="baseline",
            model="BETO",
            split="dev",
            weighted_f1=0.7100,
            soft_cons=0.5800,
            hard_cons=0.5000,
            num_instances=100,
            num_pairs=40,
        )

    def test_evaluation_result_dataclass(self) -> None:
        """Verify initialization, validation, and serialization of EvaluationResult."""
        res = self.sample_result_baseline
        self.assertEqual(res.track, "track1")
        self.assertEqual(res.method, "baseline")
        self.assertEqual(res.model, "DeBERTa-v3")
        self.assertEqual(res.method_display, "Baseline")
        self.assertEqual(res.track_display, "Track 1 (English)")

        # Invariant check: HardCons <= SoftCons
        self.assertLessEqual(res.hard_cons, res.soft_cons)

        # Dictionary conversion
        d = res.to_dict()
        self.assertIn("weighted_f1", d)
        self.assertIn("soft_cons", d)
        self.assertIn("hard_cons", d)
        self.assertEqual(d["weighted_f1"], 0.75)

    def test_comparison_table_filter(self) -> None:
        """Verify filtering comparison tables by track, method, and model."""
        tbl = ComparisonTable([self.sample_result_baseline, self.sample_result_sd, self.sample_result_t2])
        self.assertEqual(len(tbl.results), 3)

        t1_only = tbl.filter(track="track1")
        self.assertEqual(len(t1_only.results), 2)

        sd_only = tbl.filter(method="sd")
        self.assertEqual(len(sd_only.results), 1)
        self.assertEqual(sd_only.results[0].method, "sd")

    def test_comparison_table_sort_canonical(self) -> None:
        """Verify canonical sorting: Track 1 before Track 2, baseline -> bsa -> crl -> sd -> crl_sd."""
        r_crl = EvaluationResult("track1", "crl", "DeBERTa-v3", "dev", 0.79, 0.85, 0.70)
        r_bsa = EvaluationResult("track1", "bsa", "DeBERTa-v3", "dev", 0.77, 0.75, 0.65)

        tbl = ComparisonTable([self.sample_result_sd, self.sample_result_t2, r_crl, self.sample_result_baseline, r_bsa])
        sorted_tbl = tbl.sort()

        methods_order = [r.method for r in sorted_tbl.results if r.track == "track1"]
        self.assertEqual(methods_order, ["baseline", "bsa", "crl", "sd"])
        self.assertEqual(sorted_tbl.results[-1].track, "track2")

    def test_comparison_table_deltas(self) -> None:
        """Verify computation of delta improvements relative to baseline."""
        tbl = ComparisonTable([self.sample_result_baseline, self.sample_result_sd])
        with_deltas = tbl.compute_deltas(baseline_method="baseline")

        sd_res = [r for r in with_deltas.results if r.method == "sd"][0]
        self.assertIsNotNone(sd_res.delta_f1)
        self.assertAlmostEqual(sd_res.delta_f1 or 0.0, 0.8200 - 0.7500, places=4)
        self.assertAlmostEqual(sd_res.delta_soft_cons or 0.0, 1.0000 - 0.6000, places=4)
        self.assertAlmostEqual(sd_res.delta_hard_cons or 0.0, 0.7800 - 0.5500, places=4)

    def test_comparison_table_markdown_exporter(self) -> None:
        """Verify GitHub-Flavored Markdown table formatting."""
        tbl = ComparisonTable([self.sample_result_baseline, self.sample_result_sd])
        md = tbl.to_markdown(include_deltas=True)
        self.assertIn("| Track | Model | Method | Weighted F1 | SoftCons | HardCons |", md)
        self.assertIn("Track 1 (English)", md)
        self.assertIn("Baseline", md)
        self.assertIn("SD", md)
        self.assertIn("1.0000", md)

    def test_comparison_table_ascii_exporter(self) -> None:
        """Verify plain-text aligned ASCII table formatting."""
        tbl = ComparisonTable([self.sample_result_baseline, self.sample_result_sd])
        ascii_tbl = tbl.to_ascii(include_deltas=True)
        self.assertIn("+=", ascii_tbl)
        self.assertIn("Baseline", ascii_tbl)
        self.assertIn("SD", ascii_tbl)
        self.assertIn("SoftCons", ascii_tbl)

    def test_comparison_table_latex_exporter(self) -> None:
        """Verify publication-ready ACL LaTeX table formatting."""
        tbl = ComparisonTable([self.sample_result_baseline, self.sample_result_sd, self.sample_result_t2])
        tex = tbl.to_latex(multi_track=True, bold_best=True)
        self.assertIn("\\begin{table*}[t]", tex)
        self.assertIn("\\toprule", tex)
        self.assertIn("\\bottomrule", tex)
        self.assertIn("\\multicolumn{3}{c}{\\textbf{Track 1: English (DeBERTa-v3)}}", tex)
        self.assertIn("\\textbf{1.0000}", tex)  # SoftCons for SD should be bolded as best
        self.assertIn("\\end{table*}", tex)

    def test_comparison_table_json_exporter(self) -> None:
        """Verify structured JSON serialization of comparison table."""
        tbl = ComparisonTable([self.sample_result_baseline, self.sample_result_sd])
        json_str = tbl.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["total_results"], 2)
        self.assertIn("results", parsed)
        self.assertEqual(parsed["results"][0]["track"], "track1")

    def test_parse_run_metadata(self) -> None:
        """Verify filename parsing into track, method, and model components."""
        p1 = parse_run_metadata("t1_baseline_deberta_submission.csv")
        self.assertEqual(p1["track"], "track1")
        self.assertEqual(p1["method"], "baseline")
        self.assertEqual(p1["model"], "DeBERTa-v3")

        p2 = parse_run_metadata("t2_sd_beto_submission.csv")
        self.assertEqual(p2["track"], "track2")
        self.assertEqual(p2["method"], "sd")
        self.assertEqual(p2["model"], "BETO")

        p3 = parse_run_metadata("test_track1_crl_sd_mock_20261102_023522_submission.csv")
        self.assertEqual(p3["track"], "track1")
        self.assertEqual(p3["method"], "crl_sd")
        self.assertEqual(p3["model"], "Mock")

    def test_hardcons_le_softcons_invariant(self) -> None:
        """Verify that EvaluationResult respects the mathematical invariant HardCons <= SoftCons."""
        # Valid: HardCons <= SoftCons
        res_valid = EvaluationResult("track1", "test", "model", "dev", 0.8, 0.7, 0.7)
        self.assertEqual(res_valid.hard_cons, 0.7)

        # Warning logged if HardCons > SoftCons
        with self.assertLogs("src.evaluation.tables", level="WARNING"):
            _ = EvaluationResult("track1", "test", "model", "dev", 0.8, 0.5, 0.9)

    def test_sd_100_percent_softcons_in_table(self) -> None:
        """Verify that Symmetrized Decoding runs always record SoftCons == 1.0 (100%)."""
        res_sd = EvaluationResult("track1", "sd", "DeBERTa-v3", "dev", 0.85, 1.0, 0.82)
        self.assertAlmostEqual(res_sd.soft_cons, 1.0000)

        tbl = ComparisonTable([res_sd])
        md = tbl.to_markdown()
        self.assertIn("1.0000", md)

    def test_single_prediction_file_evaluation(self) -> None:
        """Verify evaluate_prediction_mapping against gold records."""
        gold_records = [
            {"instance_id": "i1_dir", "pair_id": "p1", "reverse_pair_id": "i1_rev", "label": "FORWARD_ENTAILMENT"},
            {"instance_id": "i1_rev", "pair_id": "p1", "reverse_pair_id": "i1_dir", "label": "BACKWARD_ENTAILMENT"},
            {"instance_id": "i2_dir", "pair_id": "p2", "reverse_pair_id": "i2_rev", "label": "EQUIVALENCE"},
            {"instance_id": "i2_rev", "pair_id": "p2", "reverse_pair_id": "i2_dir", "label": "EQUIVALENCE"},
        ]
        perfect_preds = {
            "i1_dir": "FORWARD_ENTAILMENT",
            "i1_rev": "BACKWARD_ENTAILMENT",
            "i2_dir": "EQUIVALENCE",
            "i2_rev": "EQUIVALENCE",
        }
        res = evaluate_prediction_mapping(gold_records, perfect_preds, track="track1", method="test")
        self.assertAlmostEqual(res.weighted_f1, 1.0)
        self.assertAlmostEqual(res.soft_cons, 1.0)
        self.assertAlmostEqual(res.hard_cons, 1.0)

    def test_multi_model_side_by_side_fallback(self) -> None:
        """Verify that when a track contains multiple models, to_latex falls back to stacked layout."""
        # Track 1 has DeBERTa-v3 AND Mock
        r1 = EvaluationResult("track1", "baseline", "DeBERTa-v3", "dev", 0.75, 0.60, 0.55)
        r2 = EvaluationResult("track1", "baseline", "Mock", "dev", 0.31, 0.66, 0.26)
        r3 = EvaluationResult("track2", "baseline", "BETO", "dev", 0.71, 0.58, 0.50)

        tbl = ComparisonTable([r1, r2, r3])
        tex = tbl.to_latex(multi_track=True)

        # Fallback to stacked table prevents silent dropping of Mock
        self.assertIn("\\begin{table}[ht]", tex)
        self.assertIn("DeBERTa-v3", tex)
        self.assertIn("Mock", tex)
        self.assertIn("BETO", tex)
        # All 3 models present
        self.assertEqual(tex.count("Baseline"), 3)

    def test_dynamic_model_headers_in_side_by_side_latex(self) -> None:
        """Verify dynamic model names appear in side-by-side LaTeX headers instead of hardcoded defaults."""
        r1 = EvaluationResult("track1", "baseline", "RoBERTa-large", "dev", 0.78, 0.62, 0.56)
        r2 = EvaluationResult("track2", "baseline", "RoBERTa-bne", "dev", 0.74, 0.60, 0.52)

        tbl = ComparisonTable([r1, r2])
        tex = tbl.to_latex(multi_track=True)

        self.assertIn("Track 1: English (RoBERTa-large)", tex)
        self.assertIn("Track 2: Spanish (RoBERTa-bne)", tex)
        self.assertNotIn("DeBERTa-v3", tex)
        self.assertNotIn("(BETO)", tex)

    def test_multi_model_sorting_groups_by_model(self) -> None:
        """Verify canonical sorting groups contiguous methods per model architecture."""
        r_deb_base = EvaluationResult("track1", "baseline", "DeBERTa-v3", "dev", 0.75, 0.60, 0.55)
        r_deb_sd = EvaluationResult("track1", "sd", "DeBERTa-v3", "dev", 0.82, 1.00, 0.78)
        r_mock_base = EvaluationResult("track1", "baseline", "Mock", "dev", 0.31, 0.66, 0.26)
        r_mock_sd = EvaluationResult("track1", "sd", "Mock", "dev", 0.40, 1.00, 0.35)

        tbl = ComparisonTable([r_deb_sd, r_mock_base, r_mock_sd, r_deb_base])
        sorted_tbl = tbl.sort()

        pairs = [(r.model, r.method) for r in sorted_tbl.results]
        expected = [
            ("DeBERTa-v3", "baseline"),
            ("DeBERTa-v3", "sd"),
            ("Mock", "baseline"),
            ("Mock", "sd"),
        ]
        self.assertEqual(pairs, expected)

    def test_latex_special_character_escaping(self) -> None:
        """Verify model and method names with underscores or special characters do not break LaTeX."""
        r = EvaluationResult("track1", "baseline", "custom_model_v1", "dev", 0.75, 0.60, 0.55)
        tbl = ComparisonTable([r])
        tex = tbl.to_latex(multi_track=False)

        self.assertIn("custom\\_model\\_v1", tex)
        self.assertNotIn("custom_model_v1", tex.replace("custom\\_model\\_v1", ""))


# =============================================================================
# 2. Confusion & Directional Transition Matrices Tests
# =============================================================================

class TestDirectionalAndConfusionMatrices(unittest.TestCase):
    """Test suite for Confusion Matrix and Directional Transition Matrix."""

    def test_ideal_involution_matrix_properties(self) -> None:
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

    def test_confusion_matrix_synthetic(self) -> None:
        """Verify confusion matrix counts, normalized matrices, and per-class metrics."""
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
        self.assertEqual(rep.total_instances, 40)
        self.assertEqual(rep.raw_matrix[0], [8, 1, 0, 1])
        self.assertEqual(rep.raw_matrix[1], [0, 7, 2, 1])
        self.assertEqual(rep.raw_matrix[2], [0, 1, 9, 0])
        self.assertEqual(rep.raw_matrix[3], [0, 0, 0, 10])
        self.assertAlmostEqual(rep.accuracy, 0.85)

        m_equ = rep.per_class["EQUIVALENCE"]
        self.assertEqual(m_equ.tp, 8)
        self.assertEqual(m_equ.fp, 0)
        self.assertEqual(m_equ.fn, 2)
        self.assertEqual(m_equ.support, 10)
        self.assertAlmostEqual(m_equ.precision, 1.0)
        self.assertAlmostEqual(m_equ.recall, 0.8)

        # ASCII and LaTeX formatting
        ascii_tbl = rep.format_ascii_table()
        self.assertIn("Multiclass Confusion Matrix", ascii_tbl)
        self.assertIn("EQU", ascii_tbl)
        self.assertIn("Summary: Accuracy: 0.8500", ascii_tbl)

        latex_tbl = rep.format_latex_table()
        self.assertIn("\\begin{table}", latex_tbl)
        self.assertIn("\\textbf{EQU}", latex_tbl)
        self.assertIn("\\end{table}", latex_tbl)

    def test_confusion_matrix_zero_division(self) -> None:
        """Verify zero support and zero predictions handle division gracefully."""
        gold = ["EQUIVALENCE"] * 5
        pred = ["EQUIVALENCE"] * 5
        rep = compute_confusion_matrix(gold, pred)
        self.assertEqual(rep.accuracy, 1.0)

        m_fwd = rep.per_class["FORWARD_ENTAILMENT"]
        self.assertEqual(m_fwd.support, 0)
        self.assertEqual(m_fwd.predicted_count, 0)
        self.assertEqual(m_fwd.precision, 0.0)
        self.assertEqual(m_fwd.recall, 0.0)
        self.assertEqual(m_fwd.f1, 0.0)

    def test_confusion_matrix_validation_errors(self) -> None:
        """Verify error checking on inputs."""
        with self.assertRaises(ValueError):
            compute_confusion_matrix([], [])

        with self.assertRaises(ValueError):
            compute_confusion_matrix(["EQUIVALENCE"], ["EQUIVALENCE", "FORWARD_ENTAILMENT"])

        with self.assertRaises(ValueError):
            compute_confusion_matrix(["INVALID_LABEL"], ["EQUIVALENCE"])

    def test_transition_matrix_ideal(self) -> None:
        """Verify ideal involution transition yields 0 violation and distance 0."""
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

        ascii_tbl = rep.format_ascii_table()
        self.assertIn("Directional Transition Matrix", ascii_tbl)
        self.assertIn("Consistent Pairs (SoftCons):      10 (100.0000%)", ascii_tbl)

        latex_tbl = rep.format_latex_table()
        self.assertIn("\\begin{table}", latex_tbl)
        self.assertIn("\\textbf{3}", latex_tbl)
        self.assertIn("SoftCons", latex_tbl)
        self.assertIn("100.0\\%", latex_tbl)

    def test_transition_matrix_reversal_curse_loop(self) -> None:
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

    def test_data_integration_codabench_csv(self) -> None:
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

    def test_evaluate_matrices_from_data_pipeline(self) -> None:
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

    def test_save_json_roundtrip(self) -> None:
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

    def test_simple_dataframe_integration(self) -> None:
        """Verify compatibility with SimpleDataFrame from src.data.loader."""
        records = [
            {"instance_id": "i1_dir", "pair_id": "p1", "reverse_pair_id": "i1_rev", "label": "FORWARD_ENTAILMENT"},
            {"instance_id": "i1_rev", "pair_id": "p1", "reverse_pair_id": "i1_dir", "label": "BACKWARD_ENTAILMENT"},
        ]
        df = SimpleDataFrame(records)

        pairs = extract_reciprocal_pairs_from_dataset(df)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0], ("i1_dir", "i1_rev"))

        pred_records = [
            {"instance_id": "i1_dir", "label": "FORWARD_ENTAILMENT"},
            {"instance_id": "i1_rev", "label": "BACKWARD_ENTAILMENT"},
        ]
        pred_df = SimpleDataFrame(pred_records)
        pred_map = load_predictions(pred_df)
        self.assertEqual(pred_map["i1_dir"], "FORWARD_ENTAILMENT")
        self.assertEqual(pred_map["i1_rev"], "BACKWARD_ENTAILMENT")

    def test_taxonomy_breakdown_cells(self) -> None:
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


# =============================================================================
# 3. Syntactic Category Error Analysis Tests
# =============================================================================

class TestSyntacticErrorAnalysis(unittest.TestCase):
    """Test suite covering disaggregated syntactic performance and error analysis."""

    def setUp(self) -> None:
        self.dev_df = load_phrasis_split("track1", "dev")
        self.dev_records = self.dev_df.to_dict("records")
        self.pairs = pair_instances(self.dev_records)

    def test_syntactic_category_disaggregation_counts(self) -> None:
        """Verify grouping of instances and pairs across nominal, verbal, prepositional, etc."""
        gold_preds = {r["instance_id"]: r["label"] for r in self.dev_records}
        report = analyze_syntactic_categories(
            dataset=self.dev_records,
            predictions=gold_preds,
            method="oracle",
            track="track1",
        )

        self.assertEqual(report.method, "oracle")
        self.assertEqual(report.track, "track1")
        self.assertEqual(report.total_instances, len(self.dev_records))
        self.assertEqual(report.total_pairs, len(self.pairs))

        # Check that core categories exist
        for cat in ["nominal", "verbal", "prepositional", "adverbial", "quantificational"]:
            self.assertIn(cat, report.categories)
            met = report.categories[cat]
            self.assertGreater(met.num_instances, 0)
            self.assertGreater(met.num_pairs, 0)
            self.assertAlmostEqual(met.weighted_f1, 1.0)
            self.assertAlmostEqual(met.soft_cons, 1.0)
            self.assertAlmostEqual(met.hard_cons, 1.0)
            self.assertEqual(met.entailment_loop_count, 0)
            self.assertEqual(met.equiv_asymmetry_count, 0)
            self.assertAlmostEqual(met.reversal_curse_index, 0.0)

    def test_reversal_curse_loop_verbal_and_prepositional(self) -> None:
        """Verify detection of Entailment Loop (Reversal Curse) in verbal and prepositional chunks."""
        baseline_preds = {r["instance_id"]: "FORWARD_ENTAILMENT" for r in self.dev_records}

        report = analyze_syntactic_categories(
            dataset=self.dev_records,
            predictions=baseline_preds,
            method="baseline",
            track="track1",
        )

        prep_met = report.categories["prepositional"]
        verb_met = report.categories["verbal"]

        self.assertGreater(prep_met.entailment_loop_count, 0)
        self.assertGreater(prep_met.reversal_curse_index, 0.0)
        self.assertAlmostEqual(prep_met.soft_cons, 0.0)

        self.assertGreater(verb_met.entailment_loop_count, 0)
        self.assertGreater(verb_met.reversal_curse_index, 0.0)
        self.assertAlmostEqual(verb_met.soft_cons, 0.0)

        # Error cases extracted
        self.assertGreater(len(report.error_cases), 0)
        loop_cases = [ec for ec in report.error_cases if ec.error_type == ErrorType.REVERSAL_CURSE_LOOP]
        self.assertGreater(len(loop_cases), 0)
        self.assertTrue(any(ec.category in ["verbal", "prepositional"] for ec in loop_cases))

    def test_equivalence_asymmetry_detection(self) -> None:
        """Verify detection of Equivalence Asymmetry in symmetric chunks."""
        asym_preds = {}
        for r in self.dev_records:
            if "flipped" in r["instance_id"]:
                asym_preds[r["instance_id"]] = "FORWARD_ENTAILMENT"
            else:
                asym_preds[r["instance_id"]] = r["label"]

        report = analyze_syntactic_categories(
            dataset=self.dev_records,
            predictions=asym_preds,
            method="asym_test",
            track="track1",
        )

        adv_met = report.categories["adverbial"]
        self.assertGreater(adv_met.equiv_asymmetry_count, 0)
        self.assertGreater(adv_met.equiv_asymmetry_rate, 0.0)

    def test_noise_collapse_detection(self) -> None:
        """Verify detection of Noise Collapse (spurious NEGATIVE_OTHER prediction)."""
        noise_preds = {r["instance_id"]: "NEGATIVE_OTHER" for r in self.dev_records}
        report = analyze_syntactic_categories(
            dataset=self.dev_records,
            predictions=noise_preds,
            method="noise_test",
            track="track1",
        )

        for cat, met in report.categories.items():
            if met.num_reversible_pairs > 0:
                self.assertGreater(met.noise_collapse_count, 0)
                self.assertAlmostEqual(met.noise_collapse_rate, 1.0)

    def test_cross_method_comparison_deltas(self) -> None:
        """Verify comparison across Baseline, BSA, CRL, and SD."""
        # 1. Baseline: all FWD
        baseline_preds = {r["instance_id"]: "FORWARD_ENTAILMENT" for r in self.dev_records}

        # 2. BSA: 50% consistent
        bsa_preds = {r["instance_id"]: r["label"] for r in self.dev_records}
        for idx, r in enumerate(self.dev_records):
            if idx % 2 == 1:
                bsa_preds[r["instance_id"]] = "FORWARD_ENTAILMENT"

        # 3. Symmetrized Decoding: 100% consistent (matches gold truth or consistent inverted)
        sd_preds = {r["instance_id"]: r["label"] for r in self.dev_records}

        method_predictions = {
            "baseline": baseline_preds,
            "bsa": bsa_preds,
            "sd": sd_preds,
        }

        cross_rep = compare_methods_syntactic(
            dataset=self.dev_records,
            method_predictions=method_predictions,
            baseline_method="baseline",
            track="track1",
        )

        self.assertEqual(cross_rep.methods, ["baseline", "bsa", "sd"])
        self.assertIn("prepositional", cross_rep.categories)

        # SD SoftCons must be 1.0 across all categories
        for cat in cross_rep.categories:
            self.assertAlmostEqual(cross_rep.metrics["sd"][cat].soft_cons, 1.0)
            delta_soft = cross_rep.deltas_vs_baseline["sd"][cat]["delta_soft_cons"]
            self.assertGreaterEqual(delta_soft, 0.0)

        # Check most improved category
        self.assertIsNotNone(cross_rep.most_improved_category_sd)
        self.assertIn(cross_rep.most_vulnerable_category_baseline, cross_rep.categories)

    def test_missing_category_column_fallback(self) -> None:
        """Verify that when 'category' column is stripped, it dynamically infers categories."""
        stripped_records = []
        for r in self.dev_records:
            copy_r = dict(r)
            if "category" in copy_r:
                del copy_r["category"]
            stripped_records.append(copy_r)

        report = analyze_syntactic_categories(
            dataset=stripped_records,
            predictions={r["instance_id"]: r["label"] for r in stripped_records},
            method="test_infer",
            track="track1",
        )

        for cat in ["nominal", "verbal", "prepositional", "adverbial"]:
            self.assertIn(cat, report.categories)
            self.assertGreater(report.categories[cat].num_instances, 0)

    def test_empty_dataset_boundary(self) -> None:
        """Verify safe handling of empty dataset without crashing."""
        report = analyze_syntactic_categories([], {})
        self.assertEqual(report.total_instances, 0)
        self.assertEqual(report.total_pairs, 0)
        self.assertEqual(report.overall_weighted_f1, 0.0)
        self.assertEqual(report.overall_soft_cons, 0.0)
        self.assertEqual(report.overall_hard_cons, 0.0)

    def test_markdown_and_latex_exporters(self) -> None:
        """Verify Markdown and LaTeX exporters generate clean, valid syntax."""
        preds = {r["instance_id"]: r["label"] for r in self.dev_records}
        report = analyze_syntactic_categories(self.dev_records, preds, method="baseline")

        # Markdown
        md_text = format_markdown_report(report)
        self.assertIn("# Syntactic Category Error Analysis", md_text)
        self.assertIn("| Category | Instances | Pairs |", md_text)
        self.assertIn("Prepositional", md_text)

        # LaTeX
        tex_dict = format_latex_tables(report)
        self.assertIn("category_performance_table", tex_dict)
        self.assertIn("error_cases_table", tex_dict)
        self.assertIn("\\begin{table*}", tex_dict["category_performance_table"])
        self.assertIn("\\toprule", tex_dict["category_performance_table"])
        self.assertIn("\\bottomrule", tex_dict["category_performance_table"])

    def test_export_analysis_artifacts_to_disk(self) -> None:
        """Verify end-to-end writing of .md, .json, and .tex artifacts to a directory."""
        gold_preds = {r["instance_id"]: r["label"] for r in self.dev_records}
        report = analyze_syntactic_categories(self.dev_records, gold_preds, method="baseline", track="track1")

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = export_analysis_artifacts(report, output_dir=tmp_dir)

            self.assertIn("markdown", artifacts)
            self.assertIn("json", artifacts)
            self.assertIn("latex", artifacts)

            for art_path in artifacts.values():
                self.assertTrue(art_path.is_file())
                self.assertGreater(art_path.stat().st_size, 0)


# =============================================================================
# 4. Standalone CLI Engine Integration Tests
# =============================================================================

class TestEvaluateCLI(unittest.TestCase):
    """Integration test suite for experiments/evaluate.py CLI entrypoint."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="test_eval_cli_")

    def tearDown(self) -> None:
        if os.path.isdir(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_cli_argument_parsing_defaults(self) -> None:
        """Verify default CLI argument values."""
        args = parse_args([])
        self.assertEqual(args.track, "all")
        self.assertEqual(args.split, "dev")
        self.assertEqual(args.method, "all")
        self.assertEqual(args.device, "auto")
        self.assertTrue(args.generate_tables)
        self.assertFalse(args.test_run)

    def test_cli_argument_parsing_custom_flags(self) -> None:
        """Verify CLI custom argument overrides."""
        args = parse_args([
            "--track", "1",
            "--method", "sd",
            "--split", "test",
            "--output_dir", self.temp_dir,
            "--test_run",
            "--generate_matrices",
            "--generate_analysis",
        ])
        self.assertEqual(args.track, "1")
        self.assertEqual(args.method, "sd")
        self.assertEqual(args.split, "test")
        self.assertEqual(args.output_dir, self.temp_dir)
        self.assertTrue(args.test_run)
        self.assertTrue(args.generate_matrices)
        self.assertTrue(args.generate_analysis)

    def test_cli_test_run_execution_speed_and_artifacts(self) -> None:
        """Verifies that evaluate CLI completes in <2.0s under --test_run and creates all artifacts."""
        script_path = PROJECT_ROOT / "experiments" / "evaluate.py"
        out_dir = Path(self.temp_dir) / "cli_run"

        t0 = time.time()
        cmd = [
            sys.executable,
            str(script_path),
            "--test_run",
            "--output_dir", str(out_dir),
            "--generate_tables",
            "--generate_matrices",
            "--generate_analysis",
        ]

        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        elapsed = time.time() - t0

        self.assertEqual(
            result.returncode,
            0,
            f"CLI failed with exit code {result.returncode}:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )

        # Strict timing assertion: execution under --test_run must be strictly under 2.0 seconds
        self.assertLess(
            elapsed,
            2.0,
            f"CLI execution under --test_run was too slow ({elapsed:.3f}s > 2.0s budget).",
        )

        # Verify generated table files
        self.assertTrue((out_dir / "comparison_table.md").is_file())
        self.assertTrue((out_dir / "comparison_table.tex").is_file())
        self.assertTrue((out_dir / "comparison_table.txt").is_file())
        self.assertTrue((out_dir / "comparison_table.json").is_file())
        self.assertTrue((out_dir / "evaluation_summary.json").is_file())

        # Verify LaTeX table contents
        with open(out_dir / "comparison_table.tex", "r", encoding="utf-8") as f:
            tex_content = f.read()
        self.assertIn("\\begin{table*}[t]", tex_content)
        self.assertIn("SD", tex_content)
        self.assertIn("\\textbf{1.0000}", tex_content)

        # Verify matrix artifacts
        self.assertTrue((out_dir / "transition_matrix_track1_sd.txt").is_file())
        self.assertTrue((out_dir / "transition_matrix_track1_sd.tex").is_file())
        self.assertTrue((out_dir / "transition_matrix_track1_sd.json").is_file())
        self.assertTrue((out_dir / "confusion_matrix_track1_sd.txt").is_file())
        self.assertTrue((out_dir / "confusion_matrix_track1_sd.tex").is_file())
        self.assertTrue((out_dir / "confusion_matrix_track1_sd.json").is_file())

        # Verify syntactic analysis output
        self.assertTrue((out_dir / "syntactic_error_analysis.json").is_file())
        self.assertTrue((out_dir / "track1_sd_syntactic_analysis.md").is_file())
        self.assertTrue((out_dir / "track1_sd_syntactic_analysis.json").is_file())
        self.assertTrue((out_dir / "track1_sd_syntactic_tables.tex").is_file())

    def test_cli_single_prediction_file_evaluation(self) -> None:
        """Verifies CLI evaluation of an existing prediction CSV."""
        sample_csv = Path(self.temp_dir) / "t1_mock_sample.csv"
        gold_df = load_phrasis_split(track="track1", split="dev")
        records = gold_df.to_dict("records")
        all_ids = [r["instance_id"] for r in records]
        all_preds = ["EQUIVALENCE"] * len(all_ids)
        export_codabench_submission(all_ids, all_preds, output_path=str(sample_csv))

        script_path = PROJECT_ROOT / "experiments" / "evaluate.py"
        out_dir = Path(self.temp_dir) / "single_eval"
        cmd = [
            sys.executable,
            str(script_path),
            "--prediction_file", str(sample_csv),
            "--track", "1",
            "--split", "dev",
            "--output_dir", str(out_dir),
        ]

        res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=10)
        self.assertEqual(res.returncode, 0, f"Error:\n{res.stderr}")
        self.assertTrue((out_dir / "comparison_table.md").is_file())

    def test_cli_nonexistent_checkpoint_error(self) -> None:
        """Verifies CLI exits with non-zero code (exit 1) and FileNotFoundError when checkpoint is missing."""
        script_path = PROJECT_ROOT / "experiments" / "evaluate.py"
        missing_ckpt = Path(self.temp_dir) / "nonexistent_model.pt"

        cmd = [
            sys.executable,
            str(script_path),
            "--checkpoint_path", str(missing_ckpt),
            "--output_dir", str(Path(self.temp_dir) / "out_missing_ckpt"),
            "--test_run",
        ]

        res = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertNotEqual(
            res.returncode,
            0,
            f"Expected CLI failure on missing checkpoint, but got returncode 0.\nSTDOUT:\n{res.stdout}",
        )
        self.assertEqual(res.returncode, 1, f"Expected returncode 1, got {res.returncode}")
        combined_err = res.stderr + res.stdout
        self.assertIn("FileNotFoundError", combined_err)
        self.assertIn(str(missing_ckpt), combined_err)

    def test_cli_nonexistent_prediction_file_error(self) -> None:
        """Verifies CLI exits with non-zero code (exit 1) and FileNotFoundError when prediction file is missing."""
        script_path = PROJECT_ROOT / "experiments" / "evaluate.py"
        missing_pred = Path(self.temp_dir) / "nonexistent_predictions.csv"

        cmd = [
            sys.executable,
            str(script_path),
            "--prediction_file", str(missing_pred),
            "--output_dir", str(Path(self.temp_dir) / "out_missing_pred"),
            "--test_run",
        ]

        res = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertNotEqual(
            res.returncode,
            0,
            f"Expected CLI failure on missing prediction file, but got returncode 0.\nSTDOUT:\n{res.stdout}",
        )
        self.assertEqual(res.returncode, 1, f"Expected returncode 1, got {res.returncode}")
        combined_err = res.stderr + res.stdout
        self.assertIn("FileNotFoundError", combined_err)
        self.assertIn(str(missing_pred), combined_err)

    def test_cli_nonexistent_results_dir_error(self) -> None:
        """Verifies CLI evaluation errors and exits with code 1 when given a missing results directory."""
        script_path = PROJECT_ROOT / "experiments" / "evaluate.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--results_dir", str(Path(self.temp_dir) / "custom_missing_results"),
        ]
        res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=10)
        self.assertEqual(res.returncode, 1, f"Expected returncode 1, got {res.returncode}.\nStderr: {res.stderr}")
        combined_err = res.stderr + res.stdout
        self.assertIn("FileNotFoundError", combined_err)
        self.assertIn("Results directory not found", combined_err)

    def test_cli_valid_checkpoint_path_execution(self) -> None:
        """Verifies CLI succeeds (exit code 0) and generates outputs when a valid checkpoint exists."""
        valid_ckpt = Path(self.temp_dir) / "valid_model.pt"
        valid_ckpt.write_text("mock checkpoint content for evaluation")

        out_dir = Path(self.temp_dir) / "valid_ckpt_eval"
        script_path = PROJECT_ROOT / "experiments" / "evaluate.py"

        cmd = [
            sys.executable,
            str(script_path),
            "--checkpoint_path", str(valid_ckpt),
            "--output_dir", str(out_dir),
            "--track", "1",
            "--method", "baseline",
            "--test_run",
        ]

        res = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertEqual(
            res.returncode,
            0,
            f"CLI failed for valid checkpoint with returncode {res.returncode}:\n{res.stderr}",
        )
        self.assertTrue((out_dir / "comparison_table.md").is_file())
        self.assertTrue((out_dir / "evaluation_summary.json").is_file())

        with open(out_dir / "evaluation_summary.json", "r", encoding="utf-8") as f:
            summary = json.load(f)
        self.assertEqual(summary["status"], "success")
        self.assertEqual(summary["total_evaluated"], 1)

    def test_run_evaluation_path_validation_programmatic(self) -> None:
        """Verifies run_evaluation raises FileNotFoundError programmatically for missing files."""
        missing_ckpt = Path(self.temp_dir) / "missing_checkpoint_prog.pt"
        args_ckpt = parse_args([
            "--checkpoint_path", str(missing_ckpt),
            "--output_dir", str(Path(self.temp_dir) / "out_prog_ckpt"),
            "--test_run",
        ])
        with self.assertRaises(FileNotFoundError) as ctx_ckpt:
            run_evaluation(args_ckpt)
        self.assertIn("Checkpoint file not found", str(ctx_ckpt.exception))

        missing_pred = Path(self.temp_dir) / "missing_pred_prog.csv"
        args_pred = parse_args([
            "--prediction_file", str(missing_pred),
            "--output_dir", str(Path(self.temp_dir) / "out_prog_pred"),
            "--test_run",
        ])
        with self.assertRaises(FileNotFoundError) as ctx_pred:
            run_evaluation(args_pred)
        self.assertIn("Prediction file not found", str(ctx_pred.exception))

    def test_validate_cli_args_direct_exceptions(self) -> None:
        """Direct unit testing of validate_cli_args exception raising."""
        # Missing checkpoint
        args_ckpt = parse_args(["--checkpoint_path", "/nonexistent/path/ckpt.pt"])
        with self.assertRaises(FileNotFoundError) as ctx_ckpt:
            validate_cli_args(args_ckpt)
        self.assertIn("Checkpoint file not found", str(ctx_ckpt.exception))

        # Missing prediction file
        args_pred = parse_args(["--prediction_file", "/nonexistent/path/pred.csv"])
        with self.assertRaises(FileNotFoundError) as ctx_pred:
            validate_cli_args(args_pred)
        self.assertIn("Prediction file not found", str(ctx_pred.exception))

        # Missing results dir
        args_res = parse_args(["--results_dir", "/nonexistent/path/results"])
        with self.assertRaises(FileNotFoundError) as ctx_res:
            validate_cli_args(args_res)
        self.assertIn("Results directory not found", str(ctx_res.exception))


if __name__ == "__main__":
    unittest.main()
