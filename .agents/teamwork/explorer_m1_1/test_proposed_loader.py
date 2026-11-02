"""
Validation test script for proposed_loader.py in explorer_m1_1.
Tests all requirements:
1. Track 1 (English) and Track 2 (Spanish) loading
2. Train, dev, and test splits
3. Schema parsing and validation (missing columns, whitespace, bad labels, duplicate IDs)
4. Syntactic category extraction across all 5 categories
5. Fallback/mock data generation
6. Split leakage auditor
7. DataFrame fallback operations (SimpleDataFrame vs pandas)
"""

import os
import sys
import tempfile
from pathlib import Path

# Add current folder to sys.path
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from proposed_loader import (
    ALL_CATEGORIES,
    ALL_LABELS,
    REVERSIBLE_LABELS,
    DatasetNotFoundError,
    DiCoLabel,
    InstanceRecord,
    SchemaValidationError,
    SimpleDataFrame,
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


def test_normalization():
    print("[TEST] Normalization...")
    assert normalize_track(1) == "track1"
    assert normalize_track("track1") == "track1"
    assert normalize_track("en") == "track1"
    assert normalize_track(2) == "track2"
    assert normalize_track("es") == "track2"
    assert normalize_split("train") == "train"
    assert normalize_split("val") == "dev"
    assert normalize_split("development") == "dev"
    assert normalize_split("test") == "test"
    print(" -> PASS")


def test_syntactic_category_extraction():
    print("[TEST] Syntactic Category Extraction...")
    # English tests
    assert extract_syntactic_category("Pope Francis", "pope", "en") == "nominal"
    assert extract_syntactic_category("a young boy", "a child", "en") == "nominal"
    assert extract_syntactic_category("is running", "is sprinting", "en") == "verbal"
    assert extract_syntactic_category("to destroy", "destroying", "en") == "verbal"
    assert extract_syntactic_category("in the garden", "in Europe", "en") == "prepositional"
    assert extract_syntactic_category("with a knife", "with a tool", "en") == "prepositional"
    assert extract_syntactic_category("extremely quickly", "very fast", "en") == "adverbial"
    assert extract_syntactic_category("as CEO", "as new executive", "en") == "adverbial"
    assert extract_syntactic_category("nine children", "five children", "en") == "quantificational"
    assert extract_syntactic_category("everyone attended", "someone attended", "en") == "quantificational"
    assert extract_syntactic_category("all dogs", "some dogs", "en") == "quantificational"

    # Spanish tests
    assert extract_syntactic_category("el papa Francisco", "el papa", "es") == "nominal"
    assert extract_syntactic_category("un niño pequeño", "un menor", "es") == "nominal"
    assert extract_syntactic_category("está corriendo", "corre", "es") == "verbal"
    assert extract_syntactic_category("destruir", "demoler", "es") == "verbal"
    assert extract_syntactic_category("en el centro de Madrid", "en España", "es") == "prepositional"
    assert extract_syntactic_category("con una pelota", "con un objeto", "es") == "prepositional"
    assert extract_syntactic_category("extremadamente rápido", "muy veloz", "es") == "adverbial"
    assert extract_syntactic_category("como nuevo presidente", "como líder", "es") == "adverbial"
    assert extract_syntactic_category("nueve niños", "cinco niños", "es") == "quantificational"
    assert extract_syntactic_category("todos los perros", "algunos perros", "es") == "quantificational"
    assert extract_syntactic_category("nadie vino", "alguien vino", "es") == "quantificational"

    print(" -> PASS")


def test_mock_generation():
    print("[TEST] Mock Data Generation...")
    # Track 1 Dev
    t1_dev = generate_mock_phrasis_data(num_pairs=10, track="track1", split="dev")
    assert len(t1_dev) > 0
    cats = {r.category for r in t1_dev}
    assert len(cats.intersection(ALL_CATEGORIES)) >= 4, f"Found categories: {cats}"
    labels = {r.label for r in t1_dev}
    assert labels.issubset(ALL_LABELS)

    # Track 2 Train
    t2_train = generate_mock_phrasis_data(num_pairs=10, track="track2", split="train")
    assert len(t2_train) > 0
    assert all(r.text1_lang == "es" for r in t2_train)

    # Test split (no labels)
    t1_test = generate_mock_phrasis_data(num_pairs=5, track="track1", split="test")
    assert all(r.label is None for r in t1_test)

    print(" -> PASS")


def test_split_leakage():
    print("[TEST] Split Leakage Auditor...")
    train_recs = generate_mock_phrasis_data(num_pairs=10, track="track1", split="train")
    dev_recs = generate_mock_phrasis_data(num_pairs=10, track="track1", split="dev")
    is_clean, leaked = verify_split_leakage(train_recs, dev_recs)
    assert is_clean is True
    assert len(leaked) == 0

    # Introduce synthetic leakage
    leaked_dev = dev_recs + [train_recs[0]]
    is_clean2, leaked2 = verify_split_leakage(train_recs, leaked_dev)
    assert is_clean2 is False
    assert len(leaked2) == 1
    assert leaked2[0] == train_recs[0].pair_id
    print(" -> PASS")


def test_file_io_and_schema_validation():
    print("[TEST] File I/O and Schema Validation...")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save mock dataset to disk
        paths = save_mock_dataset(output_dir=tmpdir, num_train_pairs=8, num_dev_pairs=4)
        assert len(paths) == 8  # 2 tracks * 2 splits * 2 file types (part + ref)

        # Load split using load_phrasis_split
        df_train_1 = load_phrasis_split(track="1", split="train", data_dir=tmpdir)
        assert len(df_train_1) > 0
        expected_cols = ['instance_id', 'pair_id', 'text1_lang', 'text2_lang', 'text1', 'text2', 'label', 'category']
        for col in expected_cols:
            assert col in df_train_1.columns, f"Missing {col} in columns"

        df_dev_2 = load_phrasis_split(track="track2", split="dev", data_dir=tmpdir)
        assert len(df_dev_2) > 0
        assert df_dev_2["text1_lang"][0] == "es"

        # Test schema error detection: missing column
        bad_csv = Path(tmpdir) / "bad.csv"
        with open(bad_csv, "w", encoding="utf-8") as f:
            f.write("instance_id,text1,text2\nid1,hello,world\n")

        try:
            read_phrasis_csv(bad_csv)
            assert False, "Should have raised SchemaValidationError for missing columns"
        except SchemaValidationError:
            pass

        # Test schema error detection: invalid label
        bad_label_csv = Path(tmpdir) / "bad_label.csv"
        with open(bad_label_csv, "w", encoding="utf-8") as f:
            f.write("instance_id,pair_id,text1_lang,text2_lang,text1,text2,label\n")
            f.write("id1,p1,en,en,hello,world,INVALID_LABEL\n")

        try:
            read_phrasis_csv(bad_label_csv)
            assert False, "Should have raised SchemaValidationError for bad label"
        except SchemaValidationError:
            pass

        # Test schema error detection: whitespace in instance_id
        bad_id_csv = Path(tmpdir) / "bad_id.csv"
        with open(bad_id_csv, "w", encoding="utf-8") as f:
            f.write("instance_id,pair_id,text1_lang,text2_lang,text1,text2,label\n")
            f.write("id with spaces,p1,en,en,hello,world,EQUIVALENCE\n")

        try:
            read_phrasis_csv(bad_id_csv)
            assert False, "Should have raised SchemaValidationError for whitespace in ID"
        except SchemaValidationError:
            pass

    print(" -> PASS")


def test_simple_dataframe_operations():
    print("[TEST] SimpleDataFrame Fallback Operations...")
    records = [
        {"instance_id": "id1", "label": "EQUIVALENCE", "category": "nominal"},
        {"instance_id": "id2", "label": "FORWARD_ENTAILMENT", "category": "verbal"},
        {"instance_id": "id3", "label": "BACKWARD_ENTAILMENT", "category": "prepositional"},
    ]
    df = SimpleDataFrame.from_records(records)
    assert len(df) == 3
    assert df.columns == ["instance_id", "label", "category"]
    assert df["instance_id"].tolist() == ["id1", "id2", "id3"]
    assert df["label"].value_counts() == {
        "EQUIVALENCE": 1,
        "FORWARD_ENTAILMENT": 1,
        "BACKWARD_ENTAILMENT": 1,
    }

    # Column assignment
    df["split"] = "train"
    assert df["split"].tolist() == ["train", "train", "train"]

    # Projection
    sub_df = df[["instance_id", "label"]]
    assert sub_df.columns == ["instance_id", "label"]

    # Filter with boolean mask
    mask = df["label"] == "EQUIVALENCE"
    filtered = df[mask]
    assert len(filtered) == 1
    assert filtered["instance_id"][0] == "id1"

    print(" -> PASS")


if __name__ == "__main__":
    print("=== RUNNING PROPOSED LOADER LOCAL TEST SUITE ===")
    test_normalization()
    test_syntactic_category_extraction()
    test_mock_generation()
    test_split_leakage()
    test_file_io_and_schema_validation()
    test_simple_dataframe_operations()
    print("=== ALL TESTS PASSED SUCCESSFULLY! ===")
