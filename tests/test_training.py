"""
tests/test_training.py

Comprehensive unit and integration test suite for SemEval-2027 Task 2 (DiCo-NLI)
Milestone 3: CLI Training Pipeline, Dual Engine Trainer, Utils, and GPU Script.
"""

from __future__ import annotations

import ast
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_phrasis_split
from src.data.reversal import pair_instances, ReciprocalPair
from src.data.codabench import validate_codabench_submission
from src.models.factory import build_model, build_tokenizer, build_loss
from src.training.utils import (
    JsonlLogger,
    can_use_amp,
    get_autocast_context,
    get_autocast_dtype,
    get_device,
    get_device_info,
    set_seed,
)
from src.training.trainer import (
    CrossEncoderDataset,
    Trainer,
    TrainerConfig,
    TrainingArguments,
    _create_grad_scaler,
    _DummyGradScaler,
)


class TestTrainingUtils(unittest.TestCase):
    """Unit tests for src/training/utils.py."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="test_utils_")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_set_seed_reproducibility(self) -> None:
        set_seed(42, deterministic=True)
        self.assertEqual(os.environ.get("PYTHONHASHSEED"), "42")

    def test_get_device(self) -> None:
        cpu_dev = get_device("cpu")
        self.assertIn(str(cpu_dev).lower(), ["cpu", "device(type='cpu')"])

        auto_dev = get_device("auto")
        self.assertTrue(auto_dev is not None)

        # Unknown fallback
        fallback = get_device("quantum_accelerator_9000")
        self.assertTrue(fallback is not None)

    def test_get_device_info(self) -> None:
        info = get_device_info("cpu")
        self.assertIsInstance(info, dict)
        self.assertIn("device", info)
        self.assertIn("device_type", info)
        self.assertIn("has_torch", info)
        self.assertEqual(info["device_type"], "cpu")

    def test_can_use_amp(self) -> None:
        # On CPU, can_use_amp should return False
        amp_cpu = can_use_amp("cpu")
        self.assertFalse(amp_cpu)

    def test_get_autocast_dtype(self) -> None:
        dt_fp16 = get_autocast_dtype(fp16=True, bf16=False)
        dt_bf16 = get_autocast_dtype(fp16=False, bf16=True)
        dt_none = get_autocast_dtype(fp16=False, bf16=False)
        # Should return a valid type or None when no torch
        self.assertTrue(dt_fp16 is not None or dt_none is None)

    def test_get_autocast_context(self) -> None:
        # Context manager should yield cleanly without raising
        with get_autocast_context(device="cpu", enabled=False):
            val = 42
        self.assertEqual(val, 42)

    def test_jsonl_logger(self) -> None:
        log_file = Path(self.temp_dir) / "metrics.jsonl"
        logger = JsonlLogger(log_file)

        logger.log({"epoch": 1, "loss": 0.5})
        logger.log({"epoch": 2, "loss": 0.3, "f1": 0.85})

        self.assertTrue(log_file.is_file())
        records = logger.read_all()
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["epoch"], 1)
        self.assertEqual(records[0]["loss"], 0.5)
        self.assertIn("timestamp", records[0])
        self.assertEqual(records[1]["epoch"], 2)
        self.assertEqual(records[1]["f1"], 0.85)

    def test_jsonl_logger_non_serializable_objects(self) -> None:
        log_file = Path(self.temp_dir) / "metrics_non_serializable.jsonl"
        logger = JsonlLogger(log_file)
        # Logging non-serializable objects (set, Path) must not raise TypeError
        logger.log({"set_val": {"a", "b"}, "path_val": Path("/tmp/model.pt")})
        records = logger.read_all()
        self.assertEqual(len(records), 1)
        self.assertIn("set_val", records[0])
        self.assertIn("path_val", records[0])


class TestTrainer(unittest.TestCase):
    """Unit tests for src/training/trainer.py."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="test_trainer_")
        self.train_df = load_phrasis_split("track1", "train")
        self.dev_df = load_phrasis_split("track1", "dev")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_and_arguments_alias(self) -> None:
        cfg = TrainerConfig(run_name="test_cfg", epochs=2)
        self.assertEqual(cfg.epochs, 2)
        self.assertIs(TrainerConfig, TrainingArguments)

        cfg_file = Path(self.temp_dir) / "config.json"
        cfg.save(cfg_file)
        self.assertTrue(cfg_file.is_file())

        loaded = TrainerConfig.load(cfg_file)
        self.assertEqual(loaded.run_name, "test_cfg")
        self.assertEqual(loaded.epochs, 2)

    def test_cross_encoder_dataset(self) -> None:
        records = [
            {"instance_id": "1", "text1": "Premise A", "text2": "Hypothesis A", "label": "FORWARD_ENTAILMENT"},
            {"instance_id": "2", "text1": "Premise B", "text2": "Hypothesis B", "label": "EQUIVALENCE"},
        ]
        ds = CrossEncoderDataset(records=records)
        self.assertEqual(len(ds), 2)
        item0 = ds[0]
        self.assertEqual(item0["instance_id"], "1")
        self.assertEqual(item0["text1"], "Premise A")
        self.assertEqual(item0["label_id"], 2)

    def test_trainer_mock_training_and_checkpoints(self) -> None:
        model = build_model("mock", method="baseline", is_mock=True)
        config = TrainerConfig(
            output_dir=os.path.join(self.temp_dir, "checkpoints"),
            log_dir=os.path.join(self.temp_dir, "logs"),
            results_dir=os.path.join(self.temp_dir, "results"),
            run_name="unit_test_run",
            epochs=2,
            batch_size=8,
            is_mock=True,
        )
        trainer = Trainer(
            model=model,
            config=config,
            train_dataset=self.train_df,
            eval_dataset=self.dev_df,
        )

        summary = trainer.train()
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["mode"], "mock_cpu")
        self.assertEqual(len(summary["history"]), 2)

        # Verify artifacts
        ckpt_dir = Path(config.output_dir) / config.run_name
        self.assertTrue((ckpt_dir / "best_model.pt").is_file())
        self.assertTrue((ckpt_dir / "checkpoint_last.pt").is_file())
        self.assertTrue((ckpt_dir / "config.json").is_file())
        self.assertTrue((Path(config.log_dir) / f"{config.run_name}.jsonl").is_file())
        self.assertTrue((Path(config.log_dir) / f"{config.run_name}_summary.json").is_file())

    def test_trainer_evaluation_metrics(self) -> None:
        model = build_model("mock", method="baseline", is_mock=True)
        config = TrainerConfig(
            output_dir=os.path.join(self.temp_dir, "checkpoints"),
            log_dir=os.path.join(self.temp_dir, "logs"),
            run_name="eval_test_run",
            is_mock=True,
        )
        trainer = Trainer(model=model, config=config, eval_dataset=self.dev_df)
        metrics = trainer.evaluate()

        self.assertIn("weighted_f1", metrics)
        self.assertIn("soft_cons", metrics)
        self.assertIn("hard_cons", metrics)
        self.assertGreaterEqual(metrics["weighted_f1"], 0.0)
        self.assertLessEqual(metrics["weighted_f1"], 1.0)

    def test_trainer_sd_100_percent_softcons(self) -> None:
        # Verify that Symmetrized Decoding produces exactly 100% SoftCons
        model_sd = build_model("mock", method="sd", mask_negative_other=True, is_mock=True)
        config = TrainerConfig(
            output_dir=os.path.join(self.temp_dir, "checkpoints"),
            log_dir=os.path.join(self.temp_dir, "logs"),
            run_name="sd_test_run",
            method="sd",
            mask_negative_other=True,
            is_mock=True,
        )
        trainer = Trainer(model=model_sd, config=config, eval_dataset=self.dev_df)
        metrics = trainer.evaluate()
        self.assertAlmostEqual(metrics["soft_cons"], 1.0, places=6)

    def test_trainer_predict_and_export_codabench(self) -> None:
        model = build_model("mock", method="baseline", is_mock=True)
        config = TrainerConfig(
            output_dir=os.path.join(self.temp_dir, "checkpoints"),
            log_dir=os.path.join(self.temp_dir, "logs"),
            results_dir=os.path.join(self.temp_dir, "results"),
            run_name="export_test_run",
            is_mock=True,
        )
        trainer = Trainer(model=model, config=config, eval_dataset=self.dev_df)

        sub_file = Path(config.results_dir) / "submission.csv"
        trainer.predict_and_export(self.dev_df, output_path=sub_file)

        self.assertTrue(sub_file.is_file())
        dev_records = self.dev_df.to_dict("records") if hasattr(self.dev_df, "to_dict") else list(self.dev_df)
        expected_ids = {str(r["instance_id"]) for r in dev_records}
        self.assertTrue(validate_codabench_submission(str(sub_file), expected_ids=expected_ids))

    def test_trainer_prepare_dataset_pre_paired_reciprocal_pairs(self) -> None:
        """Verify _prepare_dataset and evaluate handle pre-paired ReciprocalPair lists."""
        train_pairs = pair_instances(self.train_df)
        self.assertGreater(len(train_pairs), 0)
        trainer = Trainer(
            model=build_model("mock", method="crl", is_mock=True),
            config=TrainerConfig(method="crl", is_mock=True),
            train_dataset=train_pairs,
        )
        ds = trainer._prepare_dataset(trainer.train_dataset_raw, is_paired=True)
        self.assertEqual(len(ds), len(train_pairs))

        dev_pairs = pair_instances(self.dev_df)
        metrics = trainer.evaluate(eval_dataset=dev_pairs)
        self.assertGreater(metrics["weighted_f1"], 0.0)

    def test_cross_encoder_dataset_nan_and_none_text_handling(self) -> None:
        """Verify CrossEncoderDataset cleanly converts None and NaN texts to empty strings."""
        tok = build_tokenizer("mock")
        records = [
            {"instance_id": 1, "text1": None, "text2": None, "label": "EQUIVALENCE"},
            {"instance_id": 2, "text1": float("nan"), "text2": float("nan"), "label": "FORWARD_ENTAILMENT"},
            {"instance_id": float("nan"), "text1": "premise", "text2": "hypothesis", "label": "EQUIVALENCE"},
        ]
        ds = CrossEncoderDataset(records=records, tokenizer=tok)
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds[0]["text1"], "")
        self.assertEqual(ds[0]["text2"], "")
        self.assertEqual(ds[1]["text1"], "")
        self.assertEqual(ds[1]["text2"], "")
        self.assertEqual(ds[2]["instance_id"], "2")  # NaN instance_id falls back to index str

    def test_grad_scaler_compatibility(self) -> None:
        """Verify _create_grad_scaler returns safe scaler across PyTorch versions."""
        scaler_cpu = _create_grad_scaler("cpu", enabled=False)
        self.assertFalse(scaler_cpu.is_enabled())
        self.assertEqual(scaler_cpu.scale(42.0), 42.0)
        scaler_cuda = _create_grad_scaler("cuda:0", enabled=True)
        self.assertIsNotNone(scaler_cuda)

    def test_single_evaluate_method_ast(self) -> None:
        """Verify exactly one evaluate method exists on Trainer class."""
        trainer_path = PROJECT_ROOT / "src" / "training" / "trainer.py"
        with open(trainer_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        trainer_cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Trainer"][0]
        eval_methods = [n for n in trainer_cls.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate"]
        self.assertEqual(len(eval_methods), 1)


class TestCLIIntegration(unittest.TestCase):
    """Integration tests for experiments/train.py CLI."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="test_cli_")
        self.python_bin = sys.executable

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cli_help(self) -> None:
        res = subprocess.run(
            [self.python_bin, "experiments/train.py", "--help"],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("SemEval-2027 Task 2 (DiCo-NLI)", res.stdout)
        self.assertIn("--track", res.stdout)
        self.assertIn("--method", res.stdout)
        self.assertIn("--test_run", res.stdout)

    def test_cli_test_run_track1_all_methods(self) -> None:
        for method in ["baseline", "bsa", "crl", "sd", "crl_sd"]:
            out_pred = Path(self.temp_dir) / f"pred_{method}.csv"
            res = subprocess.run(
                [
                    self.python_bin,
                    "experiments/train.py",
                    "--track", "1",
                    "--method", method,
                    "--test_run",
                    "--output_dir", os.path.join(self.temp_dir, f"ckpt_{method}"),
                    "--log_dir", os.path.join(self.temp_dir, f"logs_{method}"),
                    "--predictions_file", str(out_pred),
                ],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(res.returncode, 0, f"Method {method} failed: {res.stderr}")
            self.assertTrue(out_pred.is_file(), f"Missing prediction file for {method}")
            self.assertTrue(validate_codabench_submission(str(out_pred)))

    def test_cli_test_run_track2(self) -> None:
        out_pred = Path(self.temp_dir) / "pred_track2_crl.csv"
        res = subprocess.run(
            [
                self.python_bin,
                "experiments/train.py",
                "--track", "2",
                "--method", "crl",
                "--test_run",
                "--output_dir", os.path.join(self.temp_dir, "ckpt_t2"),
                "--log_dir", os.path.join(self.temp_dir, "logs_t2"),
                "--predictions_file", str(out_pred),
            ],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(res.returncode, 0, f"Track 2 failed: {res.stderr}")
        self.assertTrue(out_pred.is_file())
        self.assertTrue(validate_codabench_submission(str(out_pred)))

    def test_cli_eval_only_mode(self) -> None:
        out_pred = Path(self.temp_dir) / "pred_eval_only.csv"
        res = subprocess.run(
            [
                self.python_bin,
                "experiments/train.py",
                "--track", "1",
                "--method", "sd",
                "--eval_only",
                "--test_run",
                "--predictions_file", str(out_pred),
            ],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(res.returncode, 0, f"eval_only failed: {res.stderr}")
        self.assertTrue(out_pred.is_file())

    def test_cli_nonexistent_checkpoint_error(self) -> None:
        res = subprocess.run(
            [
                self.python_bin,
                "experiments/train.py",
                "--eval_only",
                "--checkpoint_path", "/nonexistent/path/model.pt",
                "--test_run",
            ],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("FileNotFoundError", res.stderr + res.stdout)

    def test_cli_conflicting_precision_flags(self) -> None:
        res = subprocess.run(
            [
                self.python_bin,
                "experiments/train.py",
                "--fp16",
                "--bf16",
                "--test_run",
            ],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(res.returncode, 2)
        self.assertIn("Cannot specify both --fp16 and --bf16", res.stderr)


class TestGPUScript(unittest.TestCase):
    """Integration tests for scripts/run_gpu_training.sh."""

    def test_bash_syntax(self) -> None:
        res = subprocess.run(
            ["bash", "-n", "scripts/run_gpu_training.sh"],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(res.returncode, 0, f"Syntax error in script: {res.stderr}")

    def test_script_help(self) -> None:
        res = subprocess.run(
            ["bash", "scripts/run_gpu_training.sh", "--help"],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("Master training orchestrator", res.stdout)
        self.assertIn("--quick-test", res.stdout)
        self.assertIn("--dry-run", res.stdout)

    def test_script_dry_run(self) -> None:
        res = subprocess.run(
            ["bash", "scripts/run_gpu_training.sh", "--dry-run"],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("Dry-Run: Generated Invocations", res.stdout)
        self.assertIn("t1_baseline_deberta", res.stdout)
        self.assertIn("t2_sd_beto", res.stdout)

    def test_script_quick_test_single(self) -> None:
        res = subprocess.run(
            ["bash", "scripts/run_gpu_training.sh", "--quick-test", "--track", "1", "--method", "sd"],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(res.returncode, 0, f"Quick test failed: {res.stderr}")
        self.assertIn("Codabench submission format verification: PASSED", res.stdout)
        self.assertIn("All benchmark runs completed successfully", res.stdout)

    def test_gpu_script_continue_on_error_fails(self) -> None:
        res = subprocess.run(
            [
                "bash",
                "scripts/run_gpu_training.sh",
                "--python", "/usr/bin/false",
                "--continue-on-error",
                "--track", "1",
            ],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("failed", (res.stderr + res.stdout).lower())

    def test_gpu_script_crl_sd_filter(self) -> None:
        res = subprocess.run(
            [
                "bash",
                "scripts/run_gpu_training.sh",
                "--method", "crl_sd",
                "--dry-run",
            ],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("t1_crl_sd_deberta", res.stdout)
        self.assertIn("t2_crl_sd_beto", res.stdout)


if __name__ == "__main__":
    unittest.main()
