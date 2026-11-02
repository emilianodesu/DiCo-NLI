"""Unit tests for DiCo-NLI Reversal Logic, Label Algebra, and Pairing.

Validates:
- Deterministic label inversion mapping (EQUIVALENCE, FORWARD_ENTAILMENT, BACKWARD_ENTAILMENT).
- Involution property: Rev(Rev(y)) = y for all labels.
- Reciprocal index linking and pair identification.
- Bidirectional Symmetrical Augmentation (BSA) pair generation.
- Canonical deduplication and edge-case handling.
"""

import unittest
from typing import Any, Dict, List, Optional, Set, Tuple

# Attempt import from project source; fallback to authoritative reference specification
try:
    from src.data.reversal import (
        ALL_LABELS,
        REVERSIBLE_LABELS,
        build_reversal_dataset,
        canonical_pair_key,
        pair_instances,
        reverse_label,
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

    def reverse_label(label: str, allow_negative_other: bool = True) -> str:
        """Invert an NLI label under argument reversal.
        
        Args:
            label: One of ALL_LABELS.
            allow_negative_other: If True, maps NEGATIVE_OTHER -> NEGATIVE_OTHER (BSA mode).
                                  If False, raises ValueError (strict eval mode).
        """
        if label not in ALL_LABELS:
            raise ValueError(f"Invalid label '{label}'. Must be one of {sorted(ALL_LABELS)}")
        if label == "EQUIVALENCE":
            return "EQUIVALENCE"
        elif label == "FORWARD_ENTAILMENT":
            return "BACKWARD_ENTAILMENT"
        elif label == "BACKWARD_ENTAILMENT":
            return "FORWARD_ENTAILMENT"
        elif label == "NEGATIVE_OTHER":
            if allow_negative_other:
                return "NEGATIVE_OTHER"
            raise ValueError("NEGATIVE_OTHER is non-reversible in strict evaluation mode")
        raise ValueError(f"Unhandled label '{label}'")

    def canonical_pair_key(id1: str, id2: str) -> frozenset:
        """Return an unordered identifier set for a reciprocal pair."""
        return frozenset({id1, id2})

    def pair_instances(records: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Pair instances (A, B) and (B, A) by matching pair_id or reverse_pair_id."""
        by_id = {r["instance_id"]: r for r in records}
        seen_keys = set()
        paired = []

        for r in records:
            rev_id = r.get("reverse_pair_id")
            if not rev_id or rev_id not in by_id:
                continue
            key = canonical_pair_key(r["instance_id"], rev_id)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            paired.append((r, by_id[rev_id]))
        return paired

    def build_reversal_dataset(
        records: List[Dict[str, Any]], augment_bsa: bool = False
    ) -> List[Dict[str, Any]]:
        """Augment dataset with transposed pairs and inverted labels if augment_bsa=True."""
        if not augment_bsa:
            return list(records)

        augmented = list(records)
        existing_ids = {r["instance_id"] for r in records}

        for r in records:
            synth_id = f"{r['instance_id']}__bsa_rev"
            if synth_id in existing_ids:
                continue
            inverted = {
                "instance_id": synth_id,
                "pair_id": r.get("pair_id", "synth"),
                "text1_lang": r.get("text2_lang", r.get("text1_lang", "en")),
                "text2_lang": r.get("text1_lang", r.get("text2_lang", "en")),
                "text1": r["text2"],
                "text2": r["text1"],
                "label": reverse_label(r["label"], allow_negative_other=True),
                "category": r.get("category", "nominal"),
                "reverse_pair_id": r["instance_id"],
            }
            augmented.append(inverted)
        return augmented


class TestReversalLogic(unittest.TestCase):
    """Test suite for label algebra and reversal pairing contracts."""

    def test_reversal_label_mapping(self):
        """Verify the deterministic label inversion mapping for all reversible classes."""
        self.assertEqual(reverse_label("EQUIVALENCE"), "EQUIVALENCE")
        self.assertEqual(reverse_label("FORWARD_ENTAILMENT"), "BACKWARD_ENTAILMENT")
        self.assertEqual(reverse_label("BACKWARD_ENTAILMENT"), "FORWARD_ENTAILMENT")

    def test_reversal_operator_involution(self):
        """Verify the involution property: Rev(Rev(y)) == y for all reversible classes."""
        for label in ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT"]:
            with self.subTest(label=label):
                first_inversion = reverse_label(label)
                second_inversion = reverse_label(first_inversion)
                self.assertEqual(
                    second_inversion,
                    label,
                    f"Rev(Rev({label})) != {label}",
                )

    def test_negative_other_handling(self):
        """Verify NEGATIVE_OTHER behavior under BSA mode vs strict eval mode."""
        # BSA mode: NEGATIVE_OTHER inverts to NEGATIVE_OTHER
        self.assertEqual(
            reverse_label("NEGATIVE_OTHER", allow_negative_other=True),
            "NEGATIVE_OTHER",
        )
        # Strict eval mode: calling Rev on NEGATIVE_OTHER must raise ValueError
        with self.assertRaises(ValueError):
            reverse_label("NEGATIVE_OTHER", allow_negative_other=False)

    def test_invalid_label_rejection(self):
        """Verify rejection of non-standard or corrupt labels."""
        invalid_labels = [
            "CONTRADICTION",
            "NEUTRAL",
            "forward_entailment",
            "EQUIV",
            "",
            "   ",
            None,
        ]
        for bad_label in invalid_labels:
            with self.subTest(bad_label=bad_label):
                with self.assertRaises((ValueError, KeyError, TypeError)):
                    reverse_label(bad_label)

    def test_reciprocal_pairing_by_reverse_id(self):
        """Verify pairing of forward and backward instances via reverse_pair_id."""
        records = [
            {
                "instance_id": "pair_01__original",
                "pair_id": "pair_01",
                "text1": "a red car",
                "text2": "a vehicle",
                "label": "FORWARD_ENTAILMENT",
                "reverse_pair_id": "pair_01__flipped",
            },
            {
                "instance_id": "pair_01__flipped",
                "pair_id": "pair_01",
                "text1": "a vehicle",
                "text2": "a red car",
                "label": "BACKWARD_ENTAILMENT",
                "reverse_pair_id": "pair_01__original",
            },
            {
                "instance_id": "pair_02__original",
                "pair_id": "pair_02",
                "text1": "a huge dog",
                "text2": "a big canine",
                "label": "EQUIVALENCE",
                "reverse_pair_id": "pair_02__flipped",
            },
            {
                "instance_id": "pair_02__flipped",
                "pair_id": "pair_02",
                "text1": "a big canine",
                "text2": "a huge dog",
                "label": "EQUIVALENCE",
                "reverse_pair_id": "pair_02__original",
            },
        ]
        pairs = pair_instances(records)
        self.assertEqual(len(pairs), 2, "Should produce exactly 2 canonical pairs")

        for fwd, rev in pairs:
            self.assertEqual(fwd["reverse_pair_id"], rev["instance_id"])
            self.assertEqual(rev["reverse_pair_id"], fwd["instance_id"])
            self.assertEqual(rev["label"], reverse_label(fwd["label"]))

    def test_canonical_deduplication(self):
        """Verify that canonical pair keys prevent double-counting of reciprocal pairs."""
        key1 = canonical_pair_key("id_A", "id_B")
        key2 = canonical_pair_key("id_B", "id_A")
        self.assertEqual(key1, key2)
        self.assertEqual(len({key1, key2}), 1)

    def test_bsa_pair_generation(self):
        """Verify that BSA (Symmetrical Augmentation) creates inverted pairs with transposed texts and labels."""
        raw_dataset = [
            {
                "instance_id": "dico_001__en-en__original",
                "pair_id": "dico_001",
                "text1": "Pope Francis",
                "text2": "the pope",
                "label": "FORWARD_ENTAILMENT",
                "category": "nominal",
            },
            {
                "instance_id": "dico_002__en-en__original",
                "pair_id": "dico_002",
                "text1": "is running quickly",
                "text2": "is moving fast",
                "label": "EQUIVALENCE",
                "category": "verbal",
            },
            {
                "instance_id": "dico_003__en-en__original",
                "pair_id": "dico_003",
                "text1": "the president of France",
                "text2": "a green bicycle",
                "label": "NEGATIVE_OTHER",
                "category": "nominal",
            },
        ]

        augmented = build_reversal_dataset(raw_dataset, augment_bsa=True)
        self.assertEqual(len(augmented), 6, "BSA must double the dataset size for unaugmented items")

        aug_map = {r["instance_id"]: r for r in augmented}

        # Check dico_001 flipped
        bsa_001 = aug_map["dico_001__en-en__original__bsa_rev"]
        self.assertEqual(bsa_001["text1"], "the pope")
        self.assertEqual(bsa_001["text2"], "Pope Francis")
        self.assertEqual(bsa_001["label"], "BACKWARD_ENTAILMENT")

        # Check dico_002 flipped
        bsa_002 = aug_map["dico_002__en-en__original__bsa_rev"]
        self.assertEqual(bsa_002["text1"], "is moving fast")
        self.assertEqual(bsa_002["text2"], "is running quickly")
        self.assertEqual(bsa_002["label"], "EQUIVALENCE")

        # Check dico_003 flipped
        bsa_003 = aug_map["dico_003__en-en__original__bsa_rev"]
        self.assertEqual(bsa_003["text1"], "a green bicycle")
        self.assertEqual(bsa_003["text2"], "the president of France")
        self.assertEqual(bsa_003["label"], "NEGATIVE_OTHER")

    def test_bsa_disabled_returns_original(self):
        """When augment_bsa=False, dataset records remain unchanged."""
        raw = [{"instance_id": "1", "text1": "a", "text2": "b", "label": "EQUIVALENCE"}]
        result = build_reversal_dataset(raw, augment_bsa=False)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["instance_id"], "1")

    def test_strict_reversal_validation(self):
        """Verify strict_reversal rejects contradictions and accepts valid reciprocal pairs."""
        # Valid pair
        valid = [
            {"instance_id": "1", "pair_id": "p1", "reverse_pair_id": "2", "label": "FORWARD_ENTAILMENT"},
            {"instance_id": "2", "pair_id": "p1", "reverse_pair_id": "1", "label": "BACKWARD_ENTAILMENT"},
        ]
        pairs = pair_instances(valid, strict_reversal=True)
        self.assertEqual(len(pairs), 1)
        self.assertTrue(pairs[0].is_reversible)

        # Contradictory pair under strict_reversal=True
        bad = [
            {"instance_id": "1", "pair_id": "p1", "reverse_pair_id": "2", "label": "FORWARD_ENTAILMENT"},
            {"instance_id": "2", "pair_id": "p1", "reverse_pair_id": "1", "label": "FORWARD_ENTAILMENT"},
        ]
        with self.assertRaises(ValueError):
            pair_instances(bad, strict_reversal=True)

        # Contradictory pair under strict_reversal=False (sets is_reversible=False)
        pairs_lenient = pair_instances(bad, strict_reversal=False)
        self.assertEqual(len(pairs_lenient), 1)
        self.assertFalse(pairs_lenient[0].is_reversible)

    def test_negative_other_pair_is_non_reversible(self):
        """Pairs with NEGATIVE_OTHER ground truth must have is_reversible=False."""
        records = [
            {"instance_id": "neg1", "pair_id": "p1", "reverse_pair_id": "neg2", "label": "NEGATIVE_OTHER"},
            {"instance_id": "neg2", "pair_id": "p1", "reverse_pair_id": "neg1", "label": "NEGATIVE_OTHER"},
        ]
        pairs = pair_instances(records)
        self.assertEqual(len(pairs), 1)
        self.assertFalse(pairs[0].is_reversible, "NEGATIVE_OTHER must be marked is_reversible=False")

    def test_mixed_dataset_pairing(self):
        """Verify decoupled pairing when dataset has both reverse_pair_id and pair_id records."""
        mixed = [
            {"instance_id": "id1", "pair_id": "p1", "reverse_pair_id": "id2", "label": "EQUIVALENCE"},
            {"instance_id": "id2", "pair_id": "p1", "reverse_pair_id": "id1", "label": "EQUIVALENCE"},
            {"instance_id": "id3__original", "pair_id": "p2", "reverse_pair_id": None, "label": "EQUIVALENCE"},
            {"instance_id": "id3__flipped", "pair_id": "p2", "reverse_pair_id": None, "label": "EQUIVALENCE"},
        ]
        pairs = pair_instances(mixed)
        self.assertEqual(len(pairs), 2, "Must find both pairs regardless of pairing pass")

    def test_transposed_text_pairing(self):
        """Verify fallback pairing by exact text transposition when IDs are uninformative."""
        unlabelled_pairs = [
            {"instance_id": "r1", "text1": "alpha", "text2": "beta"},
            {"instance_id": "r2", "text1": "beta", "text2": "alpha"},
        ]
        pairs = pair_instances(unlabelled_pairs)
        self.assertEqual(len(pairs), 1)

    def test_build_reversal_dataset_dataframe_ingestion(self):
        """Verify build_reversal_dataset correctly extracts rows from DataFrames and SimpleDataFrames."""
        from src.data.loader import SimpleDataFrame, load_phrasis_split
        df = load_phrasis_split("track1", "train")
        rows_unaug = build_reversal_dataset(df, augment_bsa=False)
        self.assertEqual(len(rows_unaug), len(df))
        self.assertIsInstance(rows_unaug[0], dict)
        self.assertIn("instance_id", rows_unaug[0])

        rows_aug = build_reversal_dataset(df, augment_bsa=True)
        self.assertEqual(len(rows_aug), len(df) * 2)

    def test_instance_record_mapping_and_transposition(self):
        """Verify InstanceRecord supports dict mapping protocol and transposition."""
        from src.data.loader import InstanceRecord
        rec = InstanceRecord(
            instance_id="inst_1",
            pair_id="pair_1",
            text1="sky is blue",
            text2="blue sky",
            label="FORWARD_ENTAILMENT",
        )
        self.assertEqual(rec["instance_id"], "inst_1")
        self.assertEqual(dict(rec)["text1"], "sky is blue")
        self.assertIn("text2", rec)
        trans = rec.transposed()
        self.assertEqual(trans.instance_id, "inst_1__bsa_rev")
        self.assertEqual(trans.text1, "blue sky")
        self.assertEqual(trans.text2, "sky is blue")
        self.assertEqual(trans.label, "BACKWARD_ENTAILMENT")
        self.assertEqual(trans.reverse_pair_id, "inst_1")

    def test_reciprocal_pair_duck_typing(self):
        """Verify ReciprocalPair duck-typing properties match compute_consistency_report."""
        from src.data.reversal import ReciprocalPair
        p = ReciprocalPair(
            {"instance_id": "fwd_1", "label": "FORWARD_ENTAILMENT"},
            {"instance_id": "rev_1", "label": "BACKWARD_ENTAILMENT"},
            pair_id="p1",
        )
        self.assertEqual(p.first_id, "fwd_1")
        self.assertEqual(p.second_id, "rev_1")
        self.assertEqual(p.first_label, "FORWARD_ENTAILMENT")
        self.assertEqual(p.second_label, "BACKWARD_ENTAILMENT")


if __name__ == "__main__":
    unittest.main()
