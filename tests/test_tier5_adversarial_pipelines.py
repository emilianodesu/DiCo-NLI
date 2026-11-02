"""
tests/test_tier5_adversarial_pipelines.py

Tier 5 Adversarial Hardening Suite: Data Pipelines, Fault Injection, Pairing Anomalies & CLI Invariants.
Adversarial verification for SemEval-2027 Task 2 (DiCo-NLI):
1. Data Pipeline Fault Injection:
   - Malformed CSVs: missing headers, extra columns, corrupted lines, unclosed quotes, NaN/empty text, non-ASCII/emojis, whitespace-only texts.
   - Syntactic category extraction on adversarial & degenerate inputs.
2. Deterministic Pairing Edge Cases:
   - Odd instance counts, orphans, duplicate instance IDs, self-referencing pairs, contradictory reciprocal labels, unknown labels.
   - Formal enforcement of is_reversible=False invariant on NEGATIVE_OTHER pairs.
   - Reciprocal pairing behavior on partially labeled and unlabeled pairs.
3. CLI Script Error Handling & Invariants:
   - Invalid methods/strategies (--method invalid_xyz, --strategy invalid_xyz).
   - Incompatible mixed precision flags (--fp16 with --bf16, --precision fp32 with --fp16).
   - Missing input datasets, nonexistent checkpoints, and nonexistent prediction files.
   - Bash orchestrator scripts/run_gpu_training.sh fault tolerance and exit codes.
4. Codabench Fault Injection & Export Integrity:
   - Extra columns, trailing blank lines, NUL bytes, duplicate/whitespace IDs, invalid labels, bijection mismatches.
5. Pure-Python SimpleDataFrame Fallback Robustness:
   - Boundary condition indexing, empty frame slicing, column dimension mismatches, series operations.
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import (
    ALL_CATEGORIES,
    ALL_LABELS,
    DatasetNotFoundError,
    DiCoLabel,
    InstanceRecord,
    SchemaValidationError,
    SimpleDataFrame,
    SimpleSeries,
    SyntacticCategory,
    extract_syntactic_category,
    load_phrasis_split,
    read_phrasis_csv,
)
from src.data.reversal import (
    ALL_LABELS as REV_ALL_LABELS,
    REVERSIBLE_LABELS,
    RelationLabel,
    ReciprocalPair,
    build_reversal_dataset,
    pair_instances,
    reverse_label,
    reverse_label_bsa,
    reverse_label_strict,
)
from src.data.codabench import (
    CodabenchValidationError,
    export_codabench_submission,
    validate_codabench_submission,
)
import experiments.train as train_mod
import experiments.evaluate as eval_mod


class TestAdversarialCSVParsing(unittest.TestCase):
    """Adversarial stress-testing of CSV parsing and schema validation."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="adv_csv_")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_csv(self, filename: str, content: str, encoding: str = "utf-8") -> str:
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, "w", encoding=encoding, newline="\n") as f:
            f.write(content)
        return filepath

    def _create_binary_csv(self, filename: str, raw_bytes: bytes) -> str:
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, "wb") as f:
            f.write(raw_bytes)
        return filepath

    def test_missing_header_empty_file(self) -> None:
        """Adversarial test: 0-byte CSV file must raise SchemaValidationError."""
        csv_path = self._create_csv("empty.csv", "")
        with self.assertRaises(SchemaValidationError) as cm:
            read_phrasis_csv(csv_path)
        self.assertIn("empty or has no header", str(cm.exception).lower())

    def test_missing_required_columns(self) -> None:
        """Adversarial test: CSV missing one or more required columns."""
        required = ["instance_id", "pair_id", "text1_lang", "text2_lang", "text1", "text2"]
        for col_to_drop in required:
            cols = [c for c in required if c != col_to_drop] + ["label"]
            content = f"{','.join(cols)}\nval1,val2,en,en,hello,world,EQUIVALENCE\n"
            csv_path = self._create_csv(f"missing_{col_to_drop}.csv", content)
            with self.assertRaises(SchemaValidationError, msg=f"Dropping {col_to_drop} should raise") as cm:
                read_phrasis_csv(csv_path)
            self.assertIn("missing required columns", str(cm.exception).lower())

    def test_extra_columns_tolerance(self) -> None:
        """Adversarial test: CSV with extra columns must be gracefully tolerated."""
        content = (
            "instance_id,pair_id,text1_lang,text2_lang,text1,text2,label,extra_meta,annotator_id,score\n"
            "inst_001,pair_001,en,en,golden retriever,dog,FORWARD_ENTAILMENT,some_meta,user_42,0.99\n"
            "inst_002,pair_001,en,en,dog,golden retriever,BACKWARD_ENTAILMENT,other_meta,user_99,0.88\n"
        )
        csv_path = self._create_csv("extra_cols.csv", content)
        recs = read_phrasis_csv(csv_path)
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[0].instance_id, "inst_001")
        self.assertEqual(recs[0].label, "FORWARD_ENTAILMENT")
        self.assertEqual(recs[1].instance_id, "inst_002")
        self.assertEqual(recs[1].label, "BACKWARD_ENTAILMENT")

    def test_empty_string_and_whitespace_text_rejected(self) -> None:
        """Adversarial test: Empty or whitespace-only texts must raise SchemaValidationError."""
        bad_texts = [
            ("inst_1", "pair_1", "", "valid text"),
            ("inst_2", "pair_1", "   ", "valid text"),
            ("inst_3", "pair_2", "valid text", ""),
            ("inst_4", "pair_2", "valid text", " \t \n "),
        ]
        for inst_id, pair_id, t1, t2 in bad_texts:
            content = (
                "instance_id,pair_id,text1_lang,text2_lang,text1,text2,label\n"
                f"{inst_id},{pair_id},en,en,\"{t1}\",\"{t2}\",EQUIVALENCE\n"
            )
            csv_path = self._create_csv(f"bad_text_{inst_id}.csv", content)
            with self.assertRaises(SchemaValidationError, msg=f"Text '{t1}' vs '{t2}' should fail") as cm:
                read_phrasis_csv(csv_path)
            self.assertIn("empty text1 or text2", str(cm.exception).lower())

    def test_unclosed_quotes_handling(self) -> None:
        """Adversarial test: Unclosed quotes or broken quoting in CSV must raise SchemaValidationError."""
        content = (
            "instance_id,pair_id,text1_lang,text2_lang,text1,text2,label\n"
            "inst_01,p_01,en,en,\"unclosed quote string,phrase b,EQUIVALENCE\n"
        )
        csv_path = self._create_csv("unclosed_quote.csv", content)
        with self.assertRaises(SchemaValidationError) as cm:
            read_phrasis_csv(csv_path)
        self.assertIn("empty text1 or text2", str(cm.exception).lower())

    def test_whitespace_in_instance_and_pair_ids(self) -> None:
        """Adversarial test: Whitespace in instance_id or pair_id must raise SchemaValidationError."""
        content_inst = (
            "instance_id,pair_id,text1_lang,text2_lang,text1,text2,label\n"
            "inst 001,pair_001,en,en,phrase a,phrase b,EQUIVALENCE\n"
        )
        csv_path_inst = self._create_csv("ws_inst.csv", content_inst)
        with self.assertRaises(SchemaValidationError) as cm:
            read_phrasis_csv(csv_path_inst)
        self.assertIn("whitespace in instance_id", str(cm.exception).lower())

        content_pair = (
            "instance_id,pair_id,text1_lang,text2_lang,text1,text2,label\n"
            "inst_001,pair 001,en,en,phrase a,phrase b,EQUIVALENCE\n"
        )
        csv_path_pair = self._create_csv("ws_pair.csv", content_pair)
        with self.assertRaises(SchemaValidationError) as cm:
            read_phrasis_csv(csv_path_pair)
        self.assertIn("whitespace in pair_id", str(cm.exception).lower())

    def test_duplicate_instance_ids_rejected(self) -> None:
        """Adversarial test: Duplicate instance_ids within the same file must be rejected."""
        content = (
            "instance_id,pair_id,text1_lang,text2_lang,text1,text2,label\n"
            "dup_id,pair_1,en,en,hello,world,EQUIVALENCE\n"
            "dup_id,pair_2,en,en,goodbye,world,NEGATIVE_OTHER\n"
        )
        csv_path = self._create_csv("dup_ids.csv", content)
        with self.assertRaises(SchemaValidationError) as cm:
            read_phrasis_csv(csv_path)
        self.assertIn("duplicate instance_id", str(cm.exception).lower())

    def test_invalid_label_values(self) -> None:
        """Adversarial test: Non-standard, corrupt, or case-mismatched labels must raise SchemaValidationError."""
        invalid_labels = [
            "equivalence",       # lowercase
            "ENTAILMENT",        # standard MNLI label, not DiCo
            "CONTRADICTION",     # standard MNLI label, not DiCo
            "NEUTRAL",           # standard MNLI label, not DiCo
            "BWD_ENTAILMENT",    # abbreviated
            "POSITIVE",          # non-existent
            "RANDOM_XYZ",        # junk
        ]
        for bad_lbl in invalid_labels:
            content = (
                "instance_id,pair_id,text1_lang,text2_lang,text1,text2,label\n"
                f"inst_01,pair_01,en,en,phrase one,phrase two,{bad_lbl}\n"
            )
            csv_path = self._create_csv(f"bad_label_{bad_lbl}.csv", content)
            with self.assertRaises(SchemaValidationError, msg=f"Label '{bad_lbl}' should fail"):
                read_phrasis_csv(csv_path)

    def test_unlabeled_test_split_handling(self) -> None:
        """Adversarial test: Missing label column permitted only when allow_missing_label=True."""
        content_unlabeled = (
            "instance_id,pair_id,text1_lang,text2_lang,text1,text2\n"
            "test_01,pair_01,en,en,phrase one,phrase two\n"
        )
        csv_path = self._create_csv("unlabeled_split.csv", content_unlabeled)

        # In strict labeled mode, missing label column must raise SchemaValidationError
        with self.assertRaises(SchemaValidationError) as cm:
            read_phrasis_csv(csv_path, allow_missing_label=False)
        self.assertIn("label", str(cm.exception).lower())

        # In test mode, missing label column must be parsed with record.label = None
        recs = read_phrasis_csv(csv_path, allow_missing_label=True)
        self.assertEqual(len(recs), 1)
        self.assertIsNone(recs[0].label)

    def test_non_ascii_and_emojis_fidelity(self) -> None:
        """Adversarial test: Multi-lingual characters, accents, emojis, and symbols preserved without corruption."""
        content = (
            "instance_id,pair_id,text1_lang,text2_lang,text1,text2,label\n"
            "es_01,p_01,es,es,El niño pequeño 👦,El infante 👶,FORWARD_ENTAILMENT\n"
            "en_02,p_02,en,en,🚀 rocket speed,fast movement,EQUIVALENCE\n"
            "intl_03,p_03,es,es,¿Hablas español con café y azúcar?,lengua española,FORWARD_ENTAILMENT\n"
            "intl_04,p_04,en,en,Москва и Токио 🌏,capital cities,EQUIVALENCE\n"
        )
        csv_path = self._create_csv("unicode_emojis.csv", content)
        recs = read_phrasis_csv(csv_path)
        self.assertEqual(len(recs), 4)
        self.assertEqual(recs[0].text1, "El niño pequeño 👦")
        self.assertEqual(recs[1].text1, "🚀 rocket speed")
        self.assertIn("¿Hablas", recs[2].text1)
        self.assertIn("Москва", recs[3].text1)

    def test_corrupted_encoding_rejection(self) -> None:
        """Adversarial test: Non-UTF8 corrupted bytes trigger UnicodeDecodeError."""
        bad_bytes = b"instance_id,pair_id,text1_lang,text2_lang,text1,text2,label\ninst_1,p_1,en,en,corrupt \xff\xfe bytes,test,EQUIVALENCE\n"
        csv_path = self._create_binary_csv("corrupt_enc.csv", bad_bytes)
        with self.assertRaises(UnicodeDecodeError):
            read_phrasis_csv(csv_path)

    def test_syntactic_category_adversarial_inputs(self) -> None:
        """Adversarial test: Syntactic category classifier behavior on degenerate and edge-case phrases."""
        # 1. Pure emojis and pure symbols default safely to NOMINAL
        cat_emojis = extract_syntactic_category("🎉🚀🔥", "🧠💡", "en")
        self.assertEqual(cat_emojis, SyntacticCategory.NOMINAL.value)

        cat_punct = extract_syntactic_category("...", "???", "es")
        self.assertEqual(cat_punct, SyntacticCategory.NOMINAL.value)

        # 2. Numerals in Spanish and English -> QUANTIFICATIONAL
        cat_num_en = extract_syntactic_category("42 answers", "some answer", "en")
        self.assertEqual(cat_num_en, SyntacticCategory.QUANTIFICATIONAL.value)

        cat_num_es = extract_syntactic_category("siete vidas", "varias vidas", "es")
        self.assertEqual(cat_num_es, SyntacticCategory.QUANTIFICATIONAL.value)

        # 3. Gerunds and infinitives in Spanish -> VERBAL
        cat_verb_es = extract_syntactic_category("corriendo velozmente", "caminar", "es")
        self.assertEqual(cat_verb_es, SyntacticCategory.VERBAL.value)

        # 4. Spanish prepositional phrase -> PREPOSITIONAL
        cat_prep_es = extract_syntactic_category("en el tren", "en el vagón", "es")
        self.assertEqual(cat_prep_es, SyntacticCategory.PREPOSITIONAL.value)


class TestAdversarialPairingEdgeCases(unittest.TestCase):
    """Adversarial stress-testing of reciprocal pairing engine (pair_instances)."""

    def test_odd_instance_count_orphan_handling(self) -> None:
        """Adversarial test: Odd instance counts (e.g. 3 or 5 records) must pair valid counterparts and leave orphans."""
        records = [
            InstanceRecord("id_1", "p_1", "en", "en", "phrase a", "phrase b", "EQUIVALENCE", reverse_pair_id="id_2"),
            InstanceRecord("id_2", "p_1", "en", "en", "phrase b", "phrase a", "EQUIVALENCE", reverse_pair_id="id_1"),
            InstanceRecord("id_orphan", "p_orphan", "en", "en", "lonely phrase", "unpaired", "NEGATIVE_OTHER"),
        ]
        pairs = pair_instances(records)
        self.assertEqual(len(pairs), 1)
        self.assertEqual({pairs[0].first_id, pairs[0].second_id}, {"id_1", "id_2"})

    def test_duplicate_instance_ids_handling(self) -> None:
        """Adversarial test: Duplicate instance_ids in input list must not double-pair or crash."""
        records = [
            InstanceRecord("id_1", "p_1", "en", "en", "jump", "leap", "EQUIVALENCE", reverse_pair_id="id_2"),
            InstanceRecord("id_2", "p_1", "en", "en", "leap", "jump", "EQUIVALENCE", reverse_pair_id="id_1"),
            # Injected exact duplicate of id_1
            InstanceRecord("id_1", "p_1", "en", "en", "jump", "leap", "EQUIVALENCE", reverse_pair_id="id_2"),
        ]
        pairs = pair_instances(records)
        self.assertEqual(len(pairs), 1)
        self.assertEqual({pairs[0].first_id, pairs[0].second_id}, {"id_1", "id_2"})

    def test_self_referential_instance_pairing(self) -> None:
        """Adversarial test: Instance pointing to itself as reverse_pair_id cannot form a self-pair."""
        records = [
            InstanceRecord("id_self", "p_self", "en", "en", "text", "text", "EQUIVALENCE", reverse_pair_id="id_self")
        ]
        pairs = pair_instances(records)
        self.assertEqual(len(pairs), 0)

    def test_conflicting_reciprocal_labels_strict_vs_lenient(self) -> None:
        """Adversarial test: Reciprocal pair with contradictory labels (e.g. FWD paired with FWD)."""
        # FORWARD_ENTAILMENT requires BACKWARD_ENTAILMENT on reciprocal
        r_fwd1 = InstanceRecord("fwd1", "p_clash", "en", "en", "cat", "animal", "FORWARD_ENTAILMENT", reverse_pair_id="fwd2")
        r_fwd2 = InstanceRecord("fwd2", "p_clash", "en", "en", "animal", "cat", "FORWARD_ENTAILMENT", reverse_pair_id="fwd1")

        # In strict_reversal mode, this must raise ValueError
        with self.assertRaises(ValueError) as cm:
            pair_instances([r_fwd1, r_fwd2], strict_reversal=True)
        self.assertIn("contradictory reciprocal pair labels", str(cm.exception).lower())

        # In lenient mode, pair is formed but is_reversible must be False
        pairs_lenient = pair_instances([r_fwd1, r_fwd2], strict_reversal=False)
        self.assertEqual(len(pairs_lenient), 1)
        self.assertFalse(pairs_lenient[0].is_reversible)

    def test_unknown_or_corrupted_labels_pairing(self) -> None:
        """Adversarial test: Unknown label on one instance."""
        r_good = InstanceRecord("r1", "p_unk", "en", "en", "A", "B", "EQUIVALENCE", reverse_pair_id="r2")
        r_bad = InstanceRecord("r2", "p_unk", "en", "en", "B", "A", "CORRUPTED_LABEL_FOO", reverse_pair_id="r1")

        # Strict mode must reject
        with self.assertRaises(ValueError):
            pair_instances([r_good, r_bad], strict_reversal=True)

        # Lenient mode pairs with is_reversible=False
        pairs = pair_instances([r_good, r_bad], strict_reversal=False)
        self.assertEqual(len(pairs), 1)
        self.assertFalse(pairs[0].is_reversible)

    def test_asymmetric_unlabeled_reciprocal_instance(self) -> None:
        """Adversarial test: One instance has a label, counterpart is unlabeled (None)."""
        r_labeled = InstanceRecord("r1", "p_asym", "en", "en", "A", "B", "EQUIVALENCE", reverse_pair_id="r2")
        r_unlabeled = InstanceRecord("r2", "p_asym", "en", "en", "B", "A", None, reverse_pair_id="r1")

        # In strict mode, mismatched labeling between reciprocal counterparts raises ValueError
        with self.assertRaises(ValueError) as cm:
            pair_instances([r_labeled, r_unlabeled], strict_reversal=True)
        self.assertIn("one instance has label", str(cm.exception).lower())

        # In lenient mode (test/inference evaluation), pair forms and defaults to reversible for test prediction
        pairs = pair_instances([r_labeled, r_unlabeled], strict_reversal=False)
        self.assertEqual(len(pairs), 1)
        self.assertTrue(pairs[0].is_reversible)

    def test_negative_other_pair_is_strictly_non_reversible(self) -> None:
        """
        Adversarial test: Ground truth NEGATIVE_OTHER pairs must have is_reversible=False.
        This is an essential mathematical invariant to preserve HardCons <= SoftCons <= 1.0.
        """
        r_neg1 = InstanceRecord("neg1", "p_neg", "en", "en", "airplane", "banana", "NEGATIVE_OTHER", reverse_pair_id="neg2")
        r_neg2 = InstanceRecord("neg2", "p_neg", "en", "en", "banana", "airplane", "NEGATIVE_OTHER", reverse_pair_id="neg1")

        # Direct ReciprocalPair instantiation with is_reversible=True override must be neutralized
        pair_forced = ReciprocalPair(r_neg1, r_neg2, is_reversible=True)
        self.assertFalse(pair_forced.is_reversible)

        # Pairing engine output must be non-reversible
        pairs = pair_instances([r_neg1, r_neg2])
        self.assertEqual(len(pairs), 1)
        self.assertFalse(pairs[0].is_reversible)

    def test_multi_instance_group_transposition(self) -> None:
        """Adversarial test: 3 instances sharing the same pair_id correctly match transposed texts."""
        r_dir = InstanceRecord("inst_dir", "p_shared", "en", "en", "fast run", "running", "FORWARD_ENTAILMENT")
        r_inv = InstanceRecord("inst_inv", "p_shared", "en", "en", "running", "fast run", "BACKWARD_ENTAILMENT")
        r_third = InstanceRecord("inst_third", "p_shared", "en", "en", "jogging", "walking", "EQUIVALENCE")

        pairs = pair_instances([r_dir, r_inv, r_third])
        self.assertEqual(len(pairs), 1)
        self.assertEqual({pairs[0].first_id, pairs[0].second_id}, {"inst_dir", "inst_inv"})

    def test_empty_and_singleton_pairing(self) -> None:
        """Adversarial test: Empty list or single instance returns empty pair list without error."""
        self.assertEqual(pair_instances([]), [])
        r_single = InstanceRecord("only_one", "p_1", "en", "en", "text a", "text b", "EQUIVALENCE")
        self.assertEqual(pair_instances([r_single]), [])

    def test_bsa_augmentation_stress_and_preservation(self) -> None:
        """Adversarial test: BSA augmentation correctly inverts labels and preserves Unicode/emojis."""
        records = [
            InstanceRecord("en_01", "p_01", "en", "en", "golden retriever 🐕", "dog", "FORWARD_ENTAILMENT"),
            InstanceRecord("es_02", "p_02", "es", "es", "coche veloz 🏎️", "vehículo rápido", "EQUIVALENCE"),
            InstanceRecord("neg_03", "p_03", "en", "en", "table", "cloud", "NEGATIVE_OTHER"),
        ]
        aug = build_reversal_dataset(records, augment_bsa=True)
        self.assertEqual(len(aug), 6)  # 3 original + 3 synthetic inverses

        by_id = {r["instance_id"]: r for r in aug}
        self.assertEqual(by_id["en_01__bsa_rev"]["label"], "BACKWARD_ENTAILMENT")
        self.assertEqual(by_id["en_01__bsa_rev"]["text1"], "dog")
        self.assertEqual(by_id["en_01__bsa_rev"]["text2"], "golden retriever 🐕")

        self.assertEqual(by_id["es_02__bsa_rev"]["label"], "EQUIVALENCE")
        self.assertEqual(by_id["neg_03__bsa_rev"]["label"], "NEGATIVE_OTHER")


class TestAdversarialCLIErrorHandling(unittest.TestCase):
    """Adversarial stress-testing of CLI scripts, argument parsing, and execution invariants."""

    def test_train_cli_invalid_method_rejected(self) -> None:
        """Adversarial test: Passing invalid --method to experiments/train.py exits with status 2."""
        with self.assertRaises(SystemExit) as cm:
            train_mod.parse_args(["--method", "invalid_reversal_xyz"])
        self.assertEqual(cm.exception.code, 2)

    def test_train_cli_unknown_strategy_rejected(self) -> None:
        """Adversarial test: Passing unrecognized --strategy flag to train.py exits with status 2."""
        with self.assertRaises(SystemExit) as cm:
            train_mod.parse_args(["--strategy", "baseline"])
        self.assertEqual(cm.exception.code, 2)

    def test_train_cli_conflicting_precision_flags(self) -> None:
        """Adversarial test: Specifying mutually incompatible precision flags exits with status 2."""
        conflicting_combos = [
            ["--fp16", "--bf16"],
            ["--precision", "fp16", "--bf16"],
            ["--precision", "bf16", "--fp16"],
            ["--precision", "fp32", "--fp16"],
            ["--precision", "fp32", "--bf16"],
        ]
        for combo in conflicting_combos:
            with self.assertRaises(SystemExit, msg=f"Combo {combo} should exit with status 2") as cm:
                train_mod.parse_args(combo)
            self.assertEqual(cm.exception.code, 2)

    def test_train_cli_nonexistent_checkpoint_fails(self) -> None:
        """Adversarial test: train.py raises FileNotFoundError when --checkpoint_path does not exist."""
        args = train_mod.parse_args(["--checkpoint_path", "/nonexistent/checkpoint_foo.pt", "--test_run"])
        with self.assertRaises(FileNotFoundError) as cm:
            train_mod.run_pipeline(args)
        self.assertIn("checkpoint file not found", str(cm.exception).lower())

    def test_evaluate_cli_nonexistent_checkpoint_fails(self) -> None:
        """Adversarial test: evaluate.py raises FileNotFoundError when --checkpoint_path does not exist."""
        args = eval_mod.parse_args(["--checkpoint_path", "/nonexistent/eval_checkpoint.pt"])
        with self.assertRaises(FileNotFoundError) as cm:
            eval_mod.run_evaluation(args)
        self.assertIn("checkpoint file not found", str(cm.exception).lower())

    def test_evaluate_cli_nonexistent_prediction_file_fails(self) -> None:
        """Adversarial test: evaluate.py raises FileNotFoundError when --prediction_file does not exist."""
        args = eval_mod.parse_args(["--prediction_file", "/nonexistent/custom_preds.csv"])
        with self.assertRaises(FileNotFoundError) as cm:
            eval_mod.run_evaluation(args)
        self.assertIn("prediction file not found", str(cm.exception).lower())

    def test_loader_missing_dataset_directory_error(self) -> None:
        """Adversarial test: load_phrasis_split with non-existent data_dir and allow_mock_fallback=False raises DatasetNotFoundError."""
        with self.assertRaises(DatasetNotFoundError) as cm:
            load_phrasis_split(
                track="track1",
                split="train",
                data_dir="/nonexistent/benchmark/dataset/dir",
                allow_mock_fallback=False,
            )
        self.assertIn("not found in", str(cm.exception).lower())

    def test_gpu_script_unknown_options_rejected(self) -> None:
        """Adversarial test: scripts/run_gpu_training.sh exits with status 1 on unknown option."""
        cmd = ["bash", "scripts/run_gpu_training.sh", "--nonexistent-adversarial-flag"]
        p = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        self.assertEqual(p.returncode, 1)
        self.assertIn("Unknown option", p.stderr)

    def test_gpu_script_invalid_track_rejected(self) -> None:
        """Adversarial test: scripts/run_gpu_training.sh exits with status 1 on invalid --track."""
        cmd = ["bash", "scripts/run_gpu_training.sh", "--track", "invalid_track_99"]
        p = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        self.assertEqual(p.returncode, 1)
        self.assertIn("Invalid --track", p.stderr)

    def test_gpu_script_invalid_method_rejected(self) -> None:
        """Adversarial test: scripts/run_gpu_training.sh exits with status 1 on invalid --method."""
        cmd = ["bash", "scripts/run_gpu_training.sh", "--method", "invalid_strategy_xyz"]
        p = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        self.assertEqual(p.returncode, 1)
        self.assertIn("Invalid --method", p.stderr)


class TestAdversarialCodabenchFaultInjection(unittest.TestCase):
    """Adversarial stress-testing of Codabench submission validation and export."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="adv_coda_")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write_file(self, filename: str, content: str) -> str:
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        return filepath

    def _write_bytes(self, filename: str, raw_bytes: bytes) -> str:
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, "wb") as f:
            f.write(raw_bytes)
        return filepath

    def test_codabench_extra_columns_rejected(self) -> None:
        """Adversarial test: Submissions with extra columns (e.g. 3 columns) must be rejected."""
        content = "instance_id,label,confidence\ninst_01,EQUIVALENCE,0.99\n"
        path = self._write_file("three_cols.csv", content)
        with self.assertRaises(CodabenchValidationError) as cm:
            validate_codabench_submission(path)
        self.assertIn("invalid header", str(cm.exception).lower())

    def test_codabench_inverted_header_rejected(self) -> None:
        """Adversarial test: Submissions with swapped header order (label,instance_id) must be rejected."""
        content = "label,instance_id\nEQUIVALENCE,inst_01\n"
        path = self._write_file("swapped_hdr.csv", content)
        with self.assertRaises(CodabenchValidationError) as cm:
            validate_codabench_submission(path)
        self.assertIn("invalid header", str(cm.exception).lower())

    def test_codabench_trailing_blank_line_rejected(self) -> None:
        """Adversarial test: Submissions with trailing empty lines at EOF must be rejected."""
        content = "instance_id,label\ninst_01,EQUIVALENCE\n\n"
        path = self._write_file("trailing_blank.csv", content)
        with self.assertRaises(CodabenchValidationError) as cm:
            validate_codabench_submission(path)
        self.assertIn("trailing blank line", str(cm.exception).lower())

    def test_codabench_whitespace_and_empty_id_rejected(self) -> None:
        """Adversarial test: Empty instance_ids or IDs containing spaces must be rejected."""
        content_empty = "instance_id,label\n,EQUIVALENCE\n"
        path_empty = self._write_file("empty_id.csv", content_empty)
        with self.assertRaises(CodabenchValidationError) as cm:
            validate_codabench_submission(path_empty)
        self.assertIn("empty instance_id", str(cm.exception).lower())

        content_ws = "instance_id,label\ninst 01,EQUIVALENCE\n"
        path_ws = self._write_file("ws_id.csv", content_ws)
        with self.assertRaises(CodabenchValidationError) as cm:
            validate_codabench_submission(path_ws)
        self.assertIn("contains whitespace", str(cm.exception).lower())

    def test_codabench_duplicate_ids_rejected(self) -> None:
        """Adversarial test: Duplicate instance IDs in submission must be rejected."""
        content = "instance_id,label\ninst_01,EQUIVALENCE\ninst_01,BACKWARD_ENTAILMENT\n"
        path = self._write_file("dup_id.csv", content)
        with self.assertRaises(CodabenchValidationError) as cm:
            validate_codabench_submission(path)
        self.assertIn("duplicate instance_id", str(cm.exception).lower())

    def test_codabench_nul_bytes_injection_rejected(self) -> None:
        """Adversarial test: NUL byte injection in Codabench submission must be rejected."""
        raw = b"instance_id,label\ninst_01\x00,EQUIVALENCE\n"
        path = self._write_bytes("nul_byte.csv", raw)
        with self.assertRaises(CodabenchValidationError) as cm:
            validate_codabench_submission(path)
        self.assertIn("nul bytes", str(cm.exception).lower())

    def test_codabench_unknown_label_values_rejected(self) -> None:
        """Adversarial test: Invalid label strings in Codabench submission must be rejected."""
        content = "instance_id,label\ninst_01,INVALID_CLASS_XYZ\n"
        path = self._write_file("bad_lbl.csv", content)
        with self.assertRaises(CodabenchValidationError) as cm:
            validate_codabench_submission(path)
        self.assertIn("invalid label", str(cm.exception).lower())

    def test_codabench_bijection_mismatch(self) -> None:
        """Adversarial test: Submissions missing expected gold IDs or having extraneous IDs."""
        content = "instance_id,label\ninst_01,EQUIVALENCE\n"
        path = self._write_file("subset.csv", content)

        # Expected 2 instances, submission has 1
        with self.assertRaises(CodabenchValidationError) as cm:
            validate_codabench_submission(path, expected_ids={"inst_01", "inst_02"})
        self.assertIn("missing 1 required predictions", str(cm.exception).lower())

        # Expected 1 instance, submission has 1 different
        with self.assertRaises(CodabenchValidationError) as cm:
            validate_codabench_submission(path, expected_ids={"inst_other"})
        self.assertIn("missing", str(cm.exception).lower())

    def test_export_codabench_validation_invariants(self) -> None:
        """Adversarial test: export_codabench_submission validates length and parameters."""
        out_path = os.path.join(self.temp_dir, "export_test.csv")

        # Length mismatch
        with self.assertRaises(ValueError):
            export_codabench_submission(["id_1", "id_2"], ["EQUIVALENCE"], out_path)

        # Empty or whitespace ID
        with self.assertRaises(ValueError):
            export_codabench_submission(["   "], ["EQUIVALENCE"], out_path)

        # Invalid label
        with self.assertRaises(ValueError):
            export_codabench_submission(["id_1"], ["INVALID_LABEL"], out_path)

        # Clean export produces perfectly validated file
        export_codabench_submission(
            ["id_001", "id_002", "id_003"],
            ["EQUIVALENCE", "FORWARD_ENTAILMENT", "NEGATIVE_OTHER"],
            out_path,
        )
        self.assertTrue(validate_codabench_submission(out_path, expected_ids={"id_001", "id_002", "id_003"}))


class TestAdversarialSimpleDataFrameFallback(unittest.TestCase):
    """Adversarial stress-testing of pure-Python SimpleDataFrame and SimpleSeries fallback."""

    def test_empty_dataframe_properties(self) -> None:
        """Adversarial test: Properties of empty SimpleDataFrame."""
        df_empty = SimpleDataFrame()
        self.assertEqual(len(df_empty), 0)
        self.assertEqual(df_empty.shape, (0, 0))
        self.assertEqual(df_empty.columns, [])
        self.assertEqual(df_empty.to_dict("records"), [])

    def test_key_error_on_missing_column(self) -> None:
        """Adversarial test: Accessing nonexistent column raises KeyError."""
        df = SimpleDataFrame([{"col_a": 1, "col_b": 2}])
        with self.assertRaises(KeyError):
            _ = df["col_nonexistent"]

    def test_assignment_length_mismatch_error(self) -> None:
        """Adversarial test: Assigning column with mismatched row count raises ValueError."""
        df = SimpleDataFrame([{"a": 1}, {"a": 2}])
        with self.assertRaises(ValueError) as cm:
            df["b"] = [10]  # Only 1 value for 2 rows
        self.assertIn("does not match dataframe rows", str(cm.exception).lower())

    def test_simpleseries_operations_and_edge_cases(self) -> None:
        """Adversarial test: SimpleSeries slicing, value_counts, unique, isin, eq."""
        s = SimpleSeries([DiCoLabel.EQUIVALENCE.value, DiCoLabel.FORWARD_ENTAILMENT.value, DiCoLabel.EQUIVALENCE.value])
        self.assertEqual(len(s), 3)
        self.assertEqual(s.unique(), [DiCoLabel.EQUIVALENCE.value, DiCoLabel.FORWARD_ENTAILMENT.value])
        self.assertEqual(s.value_counts()[DiCoLabel.EQUIVALENCE.value], 2)
        self.assertEqual(s.isin([DiCoLabel.EQUIVALENCE.value]), [True, False, True])
        self.assertEqual(s == DiCoLabel.FORWARD_ENTAILMENT.value, [False, True, False])

        # Slice
        sub_s = s[0:2]
        self.assertEqual(len(sub_s), 2)
        self.assertEqual(sub_s.tolist(), [DiCoLabel.EQUIVALENCE.value, DiCoLabel.FORWARD_ENTAILMENT.value])


if __name__ == "__main__":
    unittest.main()
