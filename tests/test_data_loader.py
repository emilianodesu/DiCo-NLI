"""
Unit tests for src/data/loader.py (Data Ingestion, Schema Parsing, Category Extraction, and SimpleDataFrame).
"""

import io
import os
import tempfile
import unittest
from pathlib import Path
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
    find_phrasis_file,
    generate_mock_phrasis_data,
    load_phrasis_split,
    normalize_split,
    normalize_track,
    read_phrasis_csv,
    save_mock_dataset,
    verify_split_leakage,
)


class TestDataLoader(unittest.TestCase):
    """Test suite for PhrasIS data loader and utilities."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_normalization(self):
        """Test track and split string normalization."""
        self.assertEqual(normalize_track(1), "track1")
        self.assertEqual(normalize_track("1"), "track1")
        self.assertEqual(normalize_track("en"), "track1")
        self.assertEqual(normalize_track("track1"), "track1")
        self.assertEqual(normalize_track(2), "track2")
        self.assertEqual(normalize_track("es"), "track2")
        self.assertEqual(normalize_track("track2"), "track2")

        with self.assertRaises(ValueError):
            normalize_track("track3")

        self.assertEqual(normalize_split("train"), "train")
        self.assertEqual(normalize_split("dev"), "dev")
        self.assertEqual(normalize_split("val"), "dev")
        self.assertEqual(normalize_split("test"), "test")

        with self.assertRaises(ValueError):
            normalize_split("unknown_split")

    def test_syntactic_category_extraction_english(self):
        """Verify deterministic syntactic category rule engine on English."""
        # Quantificational
        self.assertEqual(
            extract_syntactic_category("nine children", "several children", "en"),
            "quantificational",
        )
        self.assertEqual(
            extract_syntactic_category("3 dogs", "three animals", "en"),
            "quantificational",
        )
        self.assertEqual(
            extract_syntactic_category("all voters", "every citizen", "en"),
            "quantificational",
        )

        # Verbal
        self.assertEqual(
            extract_syntactic_category("is sprinting", "is running", "en"),
            "verbal",
        )
        self.assertEqual(
            extract_syntactic_category("to destroy", "to demolish", "en"),
            "verbal",
        )
        self.assertEqual(
            extract_syntactic_category("in destroying", "destroying", "en"),
            "verbal",
        )

        # Prepositional
        self.assertEqual(
            extract_syntactic_category("in the center of Madrid", "in Spain", "en"),
            "prepositional",
        )
        self.assertEqual(
            extract_syntactic_category("under the table", "beneath the wood", "en"),
            "prepositional",
        )

        # Adverbial
        self.assertEqual(
            extract_syntactic_category("extremely rapidly", "very fast", "en"),
            "adverbial",
        )
        self.assertEqual(
            extract_syntactic_category("carefully", "with great care", "en"),
            "adverbial",
        )

        # Nominal (default)
        self.assertEqual(
            extract_syntactic_category("Pope Francis", "the pope", "en"),
            "nominal",
        )
        self.assertEqual(
            extract_syntactic_category("a red sports car", "a vehicle", "en"),
            "nominal",
        )

    def test_syntactic_category_extraction_spanish(self):
        """Verify deterministic syntactic category rule engine on Spanish."""
        # Quantificational
        self.assertEqual(
            extract_syntactic_category("nueve niños", "varios niños", "es"),
            "quantificational",
        )
        self.assertEqual(
            extract_syntactic_category("todos los perros", "cada perro", "es"),
            "quantificational",
        )

        # Verbal
        self.assertEqual(
            extract_syntactic_category("está corriendo", "se está desplazando", "es"),
            "verbal",
        )
        self.assertEqual(
            extract_syntactic_category("al destruir el muro", "demoliendo", "es"),
            "verbal",
        )

        # Prepositional
        self.assertEqual(
            extract_syntactic_category("en el centro de Madrid", "en España", "es"),
            "prepositional",
        )
        self.assertEqual(
            extract_syntactic_category("con una raqueta", "con equipo", "es"),
            "prepositional",
        )

        # Adverbial
        self.assertEqual(
            extract_syntactic_category("extremadamente rápido", "muy velozmente", "es"),
            "adverbial",
        )

        # Nominal
        self.assertEqual(
            extract_syntactic_category("el papa Francisco", "el sumo pontífice", "es"),
            "nominal",
        )

    def test_mock_data_generation(self):
        """Verify synthetic mock generator yields balanced reciprocal pairs."""
        records = generate_mock_phrasis_data(num_pairs=10, track="track1", split="dev")
        self.assertGreater(len(records), 0)

        # Check instance_id format
        for r in records:
            self.assertTrue(r.instance_id.startswith("dico_phrasis_"))
            self.assertIn("__en-en__", r.instance_id)
            self.assertTrue(r.category in ALL_CATEGORIES)

        # Check reciprocal pairing
        fwd_count = sum(1 for r in records if r.label == "FORWARD_ENTAILMENT")
        bwd_count = sum(1 for r in records if r.label == "BACKWARD_ENTAILMENT")
        self.assertEqual(fwd_count, bwd_count, "Forward and backward count must match")

    def test_save_and_load_mock_dataset(self):
        """Verify disk saving and loading of mock dataset."""
        saved_paths = save_mock_dataset(
            output_dir=self.temp_path / "mock_data",
            num_train_pairs=8,
            num_dev_pairs=4,
        )
        self.assertEqual(len(saved_paths), 8)

        # Load split from the saved directory
        df_train = load_phrasis_split(
            track="track1",
            split="train",
            data_dir=self.temp_path / "mock_data",
            allow_mock_fallback=False,
        )
        self.assertGreater(len(df_train), 0)
        self.assertIn("instance_id", df_train.columns)
        self.assertIn("category", df_train.columns)

    def test_split_leakage_verifier(self):
        """Verify leak detection between train and eval splits."""
        train_records = generate_mock_phrasis_data(num_pairs=5, track="track1", split="train")
        dev_records = generate_mock_phrasis_data(num_pairs=5, track="track1", split="dev")

        # Independent splits: leak-free
        is_clean, leaks = verify_split_leakage(train_records, dev_records)
        self.assertTrue(is_clean)
        self.assertEqual(len(leaks), 0)

        # Injected leak
        leaked_records = dev_records + [train_records[0]]
        is_clean_leaked, detected_leaks = verify_split_leakage(train_records, leaked_records)
        self.assertFalse(is_clean_leaked)
        self.assertEqual(len(detected_leaks), 1)

    def test_simple_dataframe_operations(self):
        """Verify SimpleDataFrame fallback operations."""
        records = [
            {"instance_id": "id1", "label": "EQUIVALENCE", "category": "nominal"},
            {"instance_id": "id2", "label": "FORWARD_ENTAILMENT", "category": "verbal"},
            {"instance_id": "id3", "label": "EQUIVALENCE", "category": "adverbial"},
        ]
        df = SimpleDataFrame.from_records(records)
        self.assertEqual(len(df), 3)
        self.assertEqual(df.shape, (3, 3))
        self.assertEqual(df.columns, ["instance_id", "label", "category"])

        # Column series access
        labels = df["label"]
        self.assertEqual(labels.tolist(), ["EQUIVALENCE", "FORWARD_ENTAILMENT", "EQUIVALENCE"])
        counts = labels.value_counts()
        self.assertEqual(counts["EQUIVALENCE"], 2)
        self.assertEqual(counts["FORWARD_ENTAILMENT"], 1)

        # Boolean mask filtering
        mask = [lbl == "EQUIVALENCE" for lbl in df["label"]]
        filtered = df[mask]
        self.assertEqual(len(filtered), 2)

        # Apply method
        lens = df.apply(lambda row: len(row["instance_id"]), axis=1)
        self.assertEqual(lens.tolist(), [3, 3, 3])

        # CSV export
        csv_str = df.to_csv()
        self.assertIn("instance_id,label,category", csv_str)
        buf = io.StringIO()
        df.to_csv(buf)
        self.assertIn("instance_id,label,category", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
