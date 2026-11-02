"""
tests/test_evaluation.py

Comprehensive Unit and Integration Test Suite for SemEval-2027 Task 2 (DiCo-NLI)
Milestone 4: Evaluation Tables, Metric Aggregation, and Standalone CLI Engine.
"""

from __future__ import annotations

import csv
import json
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

# Tables and evaluation module imports with sandbox fallback
try:
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
    TABLES_AVAILABLE = True
except ImportError:
    TABLES_AVAILABLE = False
    import importlib.util
    spec_path = PROJECT_ROOT / ".agents" / "teamwork" / "explorer_m4_1" / "proposed_tables.py"
    if spec_path.is_file():
        spec = importlib.util.spec_from_file_location("tables_fallback", str(spec_path))
        tables_mod = importlib.util.module_from_spec(spec)
        sys.modules["tables_fallback"] = tables_mod
        spec.loader.exec_module(tables_mod)
        CANONICAL_METHODS = tables_mod.CANONICAL_METHODS
        ComparisonTable = tables_mod.ComparisonTable
        EvaluationResult = tables_mod.EvaluationResult
        aggregate_results_from_dir = tables_mod.aggregate_results_from_dir
        build_comparison_table = tables_mod.build_comparison_table
        evaluate_prediction_file = tables_mod.evaluate_prediction_file
        evaluate_prediction_mapping = tables_mod.evaluate_prediction_mapping
        format_comparison_table = tables_mod.format_comparison_table
        generate_ascii_table = tables_mod.generate_ascii_table
        generate_latex_table = tables_mod.generate_latex_table
        generate_markdown_table = tables_mod.generate_markdown_table
        load_predictions_from_csv = tables_mod.load_predictions_from_csv
        parse_run_metadata = tables_mod.parse_run_metadata
        TABLES_AVAILABLE = True

# CLI module import with sandbox fallback
try:
    from experiments.evaluate import (
        parse_args,
        run_dynamic_model_evaluation,
        run_evaluation,
    )
    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False
    import importlib.util
    spec_path = PROJECT_ROOT / ".agents" / "teamwork" / "explorer_m4_1" / "proposed_evaluate.py"
    if spec_path.is_file():
        spec = importlib.util.spec_from_file_location("evaluate_fallback", str(spec_path))
        cli_mod = importlib.util.module_from_spec(spec)
        sys.modules["evaluate_fallback"] = cli_mod
        spec.loader.exec_module(cli_mod)
        parse_args = cli_mod.parse_args
        run_dynamic_model_evaluation = cli_mod.run_dynamic_model_evaluation
        run_evaluation = cli_mod.run_evaluation
        CLI_AVAILABLE = True

from src.data.loader import load_phrasis_split
from src.data.codabench import export_codabench_submission


class TestEvaluationTables(unittest.TestCase):
    """Unit tests for EvaluationResult, ComparisonTable, and table exporters."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="test_tables_")
        self.sample_results = [
            EvaluationResult(
                track="track1",
                method="baseline",
                model="DeBERTa-v3",
                weighted_f1=0.8120,
                soft_cons=0.6540,
                hard_cons=0.5820,
            ),
            EvaluationResult(
                track="track1",
                method="bsa",
                model="DeBERTa-v3",
                weighted_f1=0.8250,
                soft_cons=0.7810,
                hard_cons=0.6720,
            ),
            EvaluationResult(
                track="track1",
                method="crl",
                model="DeBERTa-v3",
                weighted_f1=0.8310,
                soft_cons=0.8420,
                hard_cons=0.7100,
            ),
            EvaluationResult(
                track="track1",
                method="sd",
                model="DeBERTa-v3",
                weighted_f1=0.8350,
                soft_cons=1.0000,
                hard_cons=0.7310,
            ),
            EvaluationResult(
                track="track1",
                method="crl_sd",
                model="DeBERTa-v3",
                weighted_f1=0.8480,
                soft_cons=1.0000,
                hard_cons=0.7620,
            ),
            EvaluationResult(
                track="track2",
                method="baseline",
                model="BETO",
                weighted_f1=0.7950,
                soft_cons=0.6320,
                hard_cons=0.5610,
            ),
            EvaluationResult(
                track="track2",
                method="bsa",
                model="BETO",
                weighted_f1=0.8090,
                soft_cons=0.7550,
                hard_cons=0.6500,
            ),
            EvaluationResult(
                track="track2",
                method="crl",
                model="BETO",
                weighted_f1=0.8140,
                soft_cons=0.8200,
                hard_cons=0.6910,
            ),
            EvaluationResult(
                track="track2",
                method="sd",
                model="BETO",
                weighted_f1=0.8200,
                soft_cons=1.0000,
                hard_cons=0.7150,
            ),
            EvaluationResult(
                track="track2",
                method="crl_sd",
                model="BETO",
                weighted_f1=0.8310,
                soft_cons=1.0000,
                hard_cons=0.7420,
            ),
        ]

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_evaluation_result_dataclass(self) -> None:
        """Verify EvaluationResult field parsing, track normalization, and to_dict."""
        res = EvaluationResult(
            track="1",
            method="SD",
            model="DeBERTa-v3",
            weighted_f1=0.85,
            soft_cons=1.0,
            hard_cons=0.82,
        )
        self.assertEqual(res.track, "track1")
        self.assertEqual(res.method, "sd")
        self.assertEqual(res.method_display, "SD")
        self.assertEqual(res.track_display, "Track 1 (English)")
        d = res.to_dict()
        self.assertIn("weighted_f1", d)
        self.assertEqual(d["weighted_f1"], 0.85)
        self.assertEqual(d["soft_cons"], 1.0)
        self.assertEqual(d["hard_cons"], 0.82)

    def test_comparison_table_filtering_and_sorting(self) -> None:
        """Verify ComparisonTable filtering by track/method and canonical sorting."""
        table = ComparisonTable(self.sample_results)
        self.assertEqual(len(table.results), 10)

        t1_table = table.filter(track="track1")
        self.assertEqual(len(t1_table.results), 5)
        for r in t1_table.results:
            self.assertEqual(r.track, "track1")

        sd_table = table.filter(method="sd")
        self.assertEqual(len(sd_table.results), 2)
        for r in sd_table.results:
            self.assertEqual(r.method, "sd")

        sorted_table = table.sort()
        methods_order = [r.method for r in sorted_table.results[:5]]
        self.assertEqual(methods_order, ["baseline", "bsa", "crl", "sd", "crl_sd"])

    def test_compute_deltas_relative_to_baseline(self) -> None:
        """Verify relative delta calculations (metric - baseline)."""
        table = ComparisonTable(self.sample_results).compute_deltas(baseline_method="baseline")
        t1_res = {r.method: r for r in table.results if r.track == "track1"}

        base = t1_res["baseline"]
        self.assertAlmostEqual(base.delta_f1 or 0.0, 0.0, places=4)
        self.assertAlmostEqual(base.delta_soft_cons or 0.0, 0.0, places=4)
        self.assertAlmostEqual(base.delta_hard_cons or 0.0, 0.0, places=4)

        sd = t1_res["sd"]
        self.assertAlmostEqual(sd.delta_f1 or 0.0, 0.8350 - 0.8120, places=4)
        self.assertAlmostEqual(sd.delta_soft_cons or 0.0, 1.0000 - 0.6540, places=4)
        self.assertAlmostEqual(sd.delta_hard_cons or 0.0, 0.7310 - 0.5820, places=4)

    def test_markdown_formatting(self) -> None:
        """Verify GitHub-Flavored Markdown table generation."""
        table = ComparisonTable(self.sample_results).compute_deltas()
        md = table.to_markdown(include_deltas=True)
        self.assertIn("| Track | Model | Method | Weighted F1 | SoftCons | HardCons |", md)
        self.assertIn("Track 1 (English)", md)
        self.assertIn("DeBERTa-v3", md)
        self.assertIn("Baseline", md)
        self.assertIn("0.8120", md)
        self.assertIn("1.0000", md)

    def test_ascii_formatting(self) -> None:
        """Verify aligned ASCII text table generation."""
        table = ComparisonTable(self.sample_results)
        txt = table.to_ascii(include_deltas=False)
        self.assertIn("Weighted F1", txt)
        self.assertIn("SoftCons", txt)
        self.assertIn("HardCons", txt)
        self.assertIn("+", txt)
        self.assertIn("|", txt)

    def test_latex_side_by_side_table(self) -> None:
        """Verify publication-ready LaTeX table for ACL paper."""
        table = ComparisonTable(self.sample_results)
        tex = table.to_latex(multi_track=True, bold_best=True)
        self.assertIn("\\begin{table*}[t]", tex)
        self.assertIn("\\toprule", tex)
        self.assertIn("\\multicolumn{3}{c}{\\textbf{Track 1: English (DeBERTa-v3)}}", tex)
        self.assertIn("\\multicolumn{3}{c}{\\textbf{Track 2: Spanish (BETO)}}", tex)
        self.assertIn("\\cmidrule(lr){3-5}", tex)
        self.assertIn("\\cmidrule(lr){6-8}", tex)
        self.assertIn("\\textbf{1.0000}", tex)  # Best SoftCons bolded
        self.assertIn("\\bottomrule", tex)
        self.assertIn("\\label{tab:main_results}", tex)
        self.assertIn("\\end{table*}", tex)

    def test_latex_single_track_table(self) -> None:
        """Verify single track LaTeX table generation."""
        t1_results = [r for r in self.sample_results if r.track == "track1"]
        table = ComparisonTable(t1_results)
        tex = table.to_latex(multi_track=False)
        self.assertIn("\\begin{table}[ht]", tex)
        self.assertIn("\\end{table}", tex)

    def test_json_and_file_save(self) -> None:
        """Verify JSON export and disk saving across multiple formats."""
        table = ComparisonTable(self.sample_results)
        json_str = table.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["total_results"], 10)

        # Save to disk
        md_file = Path(self.temp_dir) / "test.md"
        tex_file = Path(self.temp_dir) / "test.tex"
        json_file = Path(self.temp_dir) / "test.json"
        txt_file = Path(self.temp_dir) / "test.txt"

        table.save(md_file, fmt="md")
        table.save(tex_file, fmt="tex")
        table.save(json_file, fmt="json")
        table.save(txt_file, fmt="txt")

        self.assertTrue(md_file.is_file() and md_file.stat().st_size > 0)
        self.assertTrue(tex_file.is_file() and tex_file.stat().st_size > 0)
        self.assertTrue(json_file.is_file() and json_file.stat().st_size > 0)
        self.assertTrue(txt_file.is_file() and txt_file.stat().st_size > 0)

    def test_parse_run_metadata(self) -> None:
        """Verify filename parsing into track, method, and model."""
        meta1 = parse_run_metadata("t1_baseline_deberta_submission.csv")
        self.assertEqual(meta1["track"], "track1")
        self.assertEqual(meta1["method"], "baseline")
        self.assertEqual(meta1["model"], "DeBERTa-v3")

        meta2 = parse_run_metadata("t2_crl_sd_beto_submission.csv")
        self.assertEqual(meta2["track"], "track2")
        self.assertEqual(meta2["method"], "crl_sd")
        self.assertEqual(meta2["model"], "BETO")

        meta3 = parse_run_metadata("test_track1_sd_mock_20261102_023522_submission.csv")
        self.assertEqual(meta3["track"], "track1")
        self.assertEqual(meta3["method"], "sd")
        self.assertEqual(meta3["model"], "Mock")

    def test_load_predictions_from_csv(self) -> None:
        """Verify Codabench submission CSV parsing."""
        csv_path = Path(self.temp_dir) / "sample_pred.csv"
        rows = [
            ("inst_1", "FORWARD_ENTAILMENT"),
            ("inst_2", "BACKWARD_ENTAILMENT"),
            ("inst_3", "EQUIVALENCE"),
        ]
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("instance_id,label\n")
            for iid, lbl in rows:
                f.write(f"{iid},{lbl}\n")

        preds = load_predictions_from_csv(csv_path)
        self.assertEqual(len(preds), 3)
        self.assertEqual(preds["inst_1"], "FORWARD_ENTAILMENT")
        self.assertEqual(preds["inst_2"], "BACKWARD_ENTAILMENT")
        self.assertEqual(preds["inst_3"], "EQUIVALENCE")

    def test_mathematical_invariants_and_sd_guarantee(self) -> None:
        """Formally verifies that HardCons <= SoftCons for all rows and SD yields 1.0."""
        for r in self.sample_results:
            self.assertLessEqual(
                r.hard_cons,
                r.soft_cons + 1e-6,
                f"Invariant HardCons <= SoftCons violated for {r.track}/{r.method}",
            )
            if r.method in ["sd", "crl_sd"]:
                self.assertAlmostEqual(
                    r.soft_cons,
                    1.0,
                    places=5,
                    msg=f"Symmetrized Decoding failed 100% SoftCons for {r.track}/{r.method}",
                )


class TestEvaluationCLI(unittest.TestCase):
    """Unit and functional tests for experiments/evaluate.py CLI engine."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="test_cli_")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cli_argument_parsing_defaults(self) -> None:
        """Verify CLI default argument parsing."""
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
        script_path = PROJECT_ROOT / ".agents" / "teamwork" / "explorer_m4_1" / "proposed_evaluate.py"
        if not script_path.is_file():
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

        # Verify syntactic analysis output
        self.assertTrue((out_dir / "syntactic_error_analysis.json").is_file())

    def test_cli_single_prediction_file_evaluation(self) -> None:
        """Verifies CLI evaluation of an existing prediction CSV."""
        sample_csv = Path(self.temp_dir) / "t1_mock_sample.csv"
        # Generate valid mock submission
        gold_df = load_phrasis_split(track="track1", split="dev")
        records = gold_df.to_dict("records")
        all_ids = [r["instance_id"] for r in records]
        all_preds = ["EQUIVALENCE"] * len(all_ids)
        export_codabench_submission(all_ids, all_preds, output_path=str(sample_csv))

        script_path = PROJECT_ROOT / ".agents" / "teamwork" / "explorer_m4_1" / "proposed_evaluate.py"
        if not script_path.is_file():
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


if __name__ == "__main__":
    unittest.main()
