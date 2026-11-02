"""
Comprehensive test suite for prototype reversal pairing and reciprocal invariants.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject")

from test_prototype import ReciprocalPair, pair_instances, canonical_pair_key, reverse_label_bsa
from src.metrics.metrics import compute_official_metrics, compute_consistency_report

class TestPrototypeReversal(unittest.TestCase):

    def test_negative_other_is_reversible_is_false(self):
        """1. ReciprocalPair.is_reversible must be False when ground truth is NEGATIVE_OTHER."""
        records = [
            {'instance_id': 'n1', 'pair_id': 'p1', 'reverse_pair_id': 'n2', 'label': 'NEGATIVE_OTHER'},
            {'instance_id': 'n2', 'pair_id': 'p1', 'reverse_pair_id': 'n1', 'label': 'NEGATIVE_OTHER'},
        ]
        pairs = pair_instances(records)
        self.assertEqual(len(pairs), 1)
        self.assertFalse(pairs[0].is_reversible, "NEGATIVE_OTHER pair must have is_reversible=False")

        # In current metrics.py, compute_official_metrics raises ValueError if pairs list is empty
        # Explorer 3 is adding defensive handling for empty reversible list.
        # But if there is at least one reversible pair plus NEGATIVE_OTHER:
        mixed_records = [
            {'instance_id': 'f1', 'pair_id': 'p1', 'reverse_pair_id': 'b1', 'label': 'FORWARD_ENTAILMENT'},
            {'instance_id': 'b1', 'pair_id': 'p1', 'reverse_pair_id': 'f1', 'label': 'BACKWARD_ENTAILMENT'},
            {'instance_id': 'n1', 'pair_id': 'p2', 'reverse_pair_id': 'n2', 'label': 'NEGATIVE_OTHER'},
            {'instance_id': 'n2', 'pair_id': 'p2', 'reverse_pair_id': 'n1', 'label': 'NEGATIVE_OTHER'},
        ]
        m_pairs = pair_instances(mixed_records)
        self.assertEqual(len(m_pairs), 2)
        # Pair 1 is reversible, Pair 2 is NOT
        self.assertTrue(m_pairs[0].is_reversible)
        self.assertFalse(m_pairs[1].is_reversible)

        # When evaluated with all-correct predictions:
        # If is_reversible was True for NEGATIVE_OTHER, HardCons > SoftCons crash would happen.
        # With is_reversible=False for NEGATIVE_OTHER, only Pair 1 enters consistency, giving 1.0!
        y_true_dir = ['FORWARD_ENTAILMENT', 'NEGATIVE_OTHER']
        y_pred_dir = ['FORWARD_ENTAILMENT', 'NEGATIVE_OTHER']
        y_true_rev = ['BACKWARD_ENTAILMENT', 'NEGATIVE_OTHER']
        y_pred_rev = ['BACKWARD_ENTAILMENT', 'NEGATIVE_OTHER']
        is_rev_mask = [m_pairs[0].is_reversible, m_pairs[1].is_reversible]

        m = compute_official_metrics(y_true_dir, y_pred_dir, y_true_rev, y_pred_rev, is_rev_mask)
        self.assertAlmostEqual(m['weighted_f1'], 1.0)
        self.assertEqual(m['soft_cons'], 1.0)
        self.assertEqual(m['hard_cons'], 1.0)
        self.assertLessEqual(m['hard_cons'], m['soft_cons'] + 1e-12)
        print("Test 1 (NEGATIVE_OTHER is_reversible=False): PASS")

    def test_strict_reversal_rejects_contradictions(self):
        """2. strict_reversal=True raises ValueError on contradictory pairs."""
        records = [
            {'instance_id': 'id1', 'pair_id': 'p1', 'reverse_pair_id': 'id2', 'label': 'FORWARD_ENTAILMENT'},
            {'instance_id': 'id2', 'pair_id': 'p1', 'reverse_pair_id': 'id1', 'label': 'FORWARD_ENTAILMENT'},
        ]
        with self.assertRaises(ValueError) as ctx:
            pair_instances(records, strict_reversal=True)
        self.assertIn("Contradictory reciprocal pair labels", str(ctx.exception))

        # But with strict_reversal=False, it should succeed and set is_reversible=False
        pairs = pair_instances(records, strict_reversal=False)
        self.assertEqual(len(pairs), 1)
        self.assertFalse(pairs[0].is_reversible, "Contradictory pair should have is_reversible=False")
        print("Test 2 (strict_reversal rejection): PASS")

    def test_mixed_pairing_fallback(self):
        """3. Robust fallback pairing when reverse_pair_id is missing in some rows."""
        records = [
            {'instance_id': 'id1', 'pair_id': 'p1', 'reverse_pair_id': 'id2', 'label': 'EQUIVALENCE'},
            {'instance_id': 'id2', 'pair_id': 'p1', 'reverse_pair_id': 'id1', 'label': 'EQUIVALENCE'},
            {'instance_id': 'id3__original', 'pair_id': 'p2', 'reverse_pair_id': None, 'label': 'EQUIVALENCE'},
            {'instance_id': 'id3__flipped', 'pair_id': 'p2', 'reverse_pair_id': None, 'label': 'EQUIVALENCE'},
        ]
        pairs = pair_instances(records)
        self.assertEqual(len(pairs), 2, "Must find BOTH pairs in mixed dataset")
        pair_ids = {p.pair_id for p in pairs}
        self.assertEqual(pair_ids, {'p1', 'p2'})
        print("Test 3 (Mixed fallback pairing): PASS")

    def test_text_transposition_fallback(self):
        """Verify fallback pass 3 when both reverse_pair_id and pair_id are missing."""
        records = [
            {'instance_id': 'raw_a', 'pair_id': None, 'reverse_pair_id': None, 'text1': 'dog', 'text2': 'canine', 'label': 'FORWARD_ENTAILMENT'},
            {'instance_id': 'raw_b', 'pair_id': None, 'reverse_pair_id': None, 'text1': 'canine', 'text2': 'dog', 'label': 'BACKWARD_ENTAILMENT'},
        ]
        pairs = pair_instances(records)
        self.assertEqual(len(pairs), 1)
        self.assertTrue(pairs[0].is_reversible)
        print("Test 4 (Text transposition fallback): PASS")

    def test_reciprocal_pair_duck_typing_with_compute_consistency_report(self):
        """Verify ReciprocalPair can be passed directly to compute_consistency_report."""
        records = [
            {'instance_id': 'id1', 'pair_id': 'p1', 'reverse_pair_id': 'id2', 'label': 'FORWARD_ENTAILMENT'},
            {'instance_id': 'id2', 'pair_id': 'p1', 'reverse_pair_id': 'id1', 'label': 'BACKWARD_ENTAILMENT'},
        ]
        pairs = pair_instances(records)
        preds = {'id1': 'FORWARD_ENTAILMENT', 'id2': 'BACKWARD_ENTAILMENT'}
        report = compute_consistency_report(pairs, preds)
        self.assertEqual(report.soft_cons, 1.0)
        self.assertEqual(report.hard_cons, 1.0)
        print("Test 5 (ReciprocalPair duck-typing in compute_consistency_report): PASS")

if __name__ == '__main__':
    unittest.main()
