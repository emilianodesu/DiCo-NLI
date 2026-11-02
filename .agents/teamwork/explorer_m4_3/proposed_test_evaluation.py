"""
Unit tests for src/evaluation/analysis.py (Syntactic Category Error Analysis & Qualitative Diagnosis).
"""

import json
import tempfile
import unittest
from pathlib import Path

# Import from our proposed module
import sys
sys.path.insert(0, "/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m4_3")
from proposed_analysis import (
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

from src.data.loader import (
    InstanceRecord,
    SimpleDataFrame,
    SyntacticCategory,
    generate_mock_phrasis_data,
    load_phrasis_split,
)
from src.data.reversal import pair_instances


class TestSyntacticErrorAnalysis(unittest.TestCase):
    """Test suite covering disaggregated syntactic performance and error analysis."""

    def setUp(self):
        self.dev_df = load_phrasis_split("track1", "dev")
        self.dev_records = self.dev_df.to_dict("records")
        self.pairs = pair_instances(self.dev_records)

    def test_syntactic_category_disaggregation_counts(self):
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
            # Under oracle, Weighted F1, SoftCons, and HardCons must all be 1.0
            self.assertAlmostEqual(met.weighted_f1, 1.0)
            self.assertAlmostEqual(met.soft_cons, 1.0)
            self.assertAlmostEqual(met.hard_cons, 1.0)
            self.assertEqual(met.entailment_loop_count, 0)
            self.assertEqual(met.equiv_asymmetry_count, 0)
            self.assertAlmostEqual(met.reversal_curse_index, 0.0)

    def test_reversal_curse_loop_verbal_and_prepositional(self):
        """Verify detection of Entailment Loop (Reversal Curse) in verbal and prepositional chunks."""
        # Baseline simulation: predicts FORWARD_ENTAILMENT for everything
        baseline_preds = {r["instance_id"]: "FORWARD_ENTAILMENT" for r in self.dev_records}

        report = analyze_syntactic_categories(
            dataset=self.dev_records,
            predictions=baseline_preds,
            method="baseline",
            track="track1",
        )

        # Baseline predicting all FWD creates entailment loop on every asymmetric pair!
        prep_met = report.categories["prepositional"]
        verb_met = report.categories["verbal"]

        self.assertGreater(prep_met.entailment_loop_count, 0)
        self.assertGreater(prep_met.reversal_curse_index, 0.0)
        self.assertAlmostEqual(prep_met.soft_cons, 0.0)

        self.assertGreater(verb_met.entailment_loop_count, 0)
        self.assertGreater(verb_met.reversal_curse_index, 0.0)
        self.assertAlmostEqual(verb_met.soft_cons, 0.0)

        # Error cases extracted
        errors = report.error_cases
        self.assertGreater(len(errors), 0)
        loop_errors = [e for e in errors if e.error_type == ErrorType.REVERSAL_CURSE_LOOP]
        self.assertGreater(len(loop_errors), 0)

        first_loop = loop_errors[0]
        self.assertIn("Reversal Curse", first_loop.diagnosis)
        self.assertFalse(first_loop.is_soft_consistent)
        self.assertFalse(first_loop.is_hard_correct)

    def test_equivalence_asymmetry_detection(self):
        """Verify detection of Equivalence Asymmetry in symmetric chunks."""
        preds = {r["instance_id"]: r["label"] for r in self.dev_records}

        # Corrupt one adverbial equivalence pair
        adv_pairs = [
            p for p in self.pairs
            if p.direct_instance.get("category") == "adverbial"
            and p.direct_instance.get("label") == "EQUIVALENCE"
        ]
        self.assertGreater(len(adv_pairs), 0)
        target_pair = adv_pairs[0]
        id_dir = target_pair.direct_instance["instance_id"]
        id_rev = target_pair.inverse_instance["instance_id"]

        # Predict FWD one way and EQUIVALENCE the other way
        preds[id_dir] = "FORWARD_ENTAILMENT"
        preds[id_rev] = "EQUIVALENCE"

        report = analyze_syntactic_categories(
            dataset=self.dev_records,
            predictions=preds,
            method="asym_test",
        )

        adv_met = report.categories["adverbial"]
        self.assertGreater(adv_met.equiv_asymmetry_count, 0)
        self.assertGreater(adv_met.equiv_asymmetry_rate, 0.0)

        # Check qualitative error
        error_cases = extract_qualitative_error_cases(
            self.dev_records,
            preds,
            target_categories=["adverbial"],
        )
        asym_cases = [e for e in error_cases if e.error_type == ErrorType.EQUIVALENCE_ASYMMETRY]
        self.assertGreater(len(asym_cases), 0)
        self.assertIn("Equivalence Asymmetry", asym_cases[0].diagnosis)

    def test_noise_collapse_detection(self):
        """Verify detection of Noise Collapse (spurious NEGATIVE_OTHER prediction)."""
        preds = {r["instance_id"]: r["label"] for r in self.dev_records}

        # Corrupt one prepositional pair
        prep_pairs = [
            p for p in self.pairs
            if p.direct_instance.get("category") == "prepositional"
        ]
        self.assertGreater(len(prep_pairs), 0)
        p = prep_pairs[0]
        preds[p.direct_instance["instance_id"]] = "NEGATIVE_OTHER"

        report = analyze_syntactic_categories(
            dataset=self.dev_records,
            predictions=preds,
            method="noise_test",
        )

        prep_met = report.categories["prepositional"]
        self.assertGreater(prep_met.noise_collapse_count, 0)
        self.assertGreater(prep_met.noise_collapse_rate, 0.0)

    def test_cross_method_comparison_deltas(self):
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

    def test_markdown_and_latex_exporters(self):
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

    def test_export_analysis_artifacts_to_disk(self):
        """Verify end-to-end writing of .md, .json, and .tex artifacts to a directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            preds = {r["instance_id"]: r["label"] for r in self.dev_records}
            report = analyze_syntactic_categories(self.dev_records, preds, method="baseline")
            created = export_analysis_artifacts(report, output_dir=tmp_dir)

            self.assertIn("markdown", created)
            self.assertIn("json", created)
            self.assertIn("latex", created)

            self.assertTrue(Path(created["markdown"]).is_file())
            self.assertTrue(Path(created["json"]).is_file())
            self.assertTrue(Path(created["latex"]).is_file())

            # Verify JSON loadable
            with open(created["json"], "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("report", data)
            self.assertEqual(data["report"]["method"], "baseline")

    def test_missing_category_column_fallback(self):
        """Verify that when 'category' column is stripped, it dynamically infers categories."""
        stripped_records = []
        for r in self.dev_records:
            r_copy = dict(r)
            del r_copy["category"]
            stripped_records.append(r_copy)

        gold_preds = {r["instance_id"]: r["label"] for r in stripped_records}
        report = analyze_syntactic_categories(stripped_records, gold_preds)

        # Dynamic extraction should find the standard syntactic categories
        self.assertIn("prepositional", report.categories)
        self.assertIn("verbal", report.categories)
        self.assertIn("nominal", report.categories)
        self.assertGreater(report.categories["prepositional"].num_instances, 0)

    def test_empty_dataset_boundary(self):
        """Verify safe handling of empty dataset without crashing."""
        report = analyze_syntactic_categories([], {})
        self.assertEqual(report.total_instances, 0)
        self.assertEqual(report.total_pairs, 0)
        self.assertEqual(report.overall_weighted_f1, 0.0)
        self.assertEqual(report.overall_soft_cons, 0.0)
        self.assertEqual(report.overall_hard_cons, 0.0)
