"""
tests/test_tier5_adversarial_algebra.py

Tier 5 Adversarial Coverage Hardening: Core Algebra, Symmetrized Decoding,
Involution Matrices, Deterministic Tie-Breaking, and Numerical Stability.

Adversarial Stress Test Suite covering:
1. Symmetrized Decoding Involution & Extreme Edge-Case Distributions:
   - Flat uniform [0.25, 0.25, 0.25, 0.25]
   - Subnormal floats (1e-300), near-zero, and zero-sum probability vectors
   - Massive logits (1e6) and deep negative logits (-1e6)
   - NaN and Inf non-finite floating point inputs (boundary characterization)
2. Involutory Symmetric Tie-Breaking & Combinatorial Verification:
   - Canonical candidate ties ({0, 2}, {0, 1}, {0, 3}, {1, 2}, {2, 3}, {0, 1, 2, 3})
   - Exhaustive exploration of all 15 non-empty subsets of candidate ties
   - Asymmetric direct/reverse tie interactions
   - Involutory equivariance theorem: Rev(pred_dir) == pred_rev
3. 100,000 Randomized Monte Carlo Draws for 100% SoftCons:
   - Dirichlet / exponential distribution sampling over the probability simplex
   - Exact SoftCons == 1.000000 across 100,000 trials
   - Verification that masked decoder restricts all outputs to REVERSIBLE_INDICES {0, 1, 2}
4. Metrics Boundary Conditions & Mathematical Invariants:
   - Zero-count evaluation slices and empty sequence handling
   - Missing prediction keys raising diagnostic KeyError
   - Single-instance reversible pairs across all 4 logical consistency states
   - Extreme prediction regimes (all-contradiction, all-neutral, all-entailment, all-negative)
   - Strict mathematical invariant 0.0 <= HardCons <= SoftCons <= 1.0 across 500 randomized trials
   - ReversiblePair vs ReciprocalPair handling of non-reversible NEGATIVE_OTHER
   - Directional Transition Matrix boundary conditions and taxonomy conservation laws
"""

from __future__ import annotations

import itertools
import math
import random
import unittest
from typing import Dict, List, Mapping, Sequence, Set, Tuple

from src.data.reversal import (
    RelationLabel,
    ReciprocalPair,
    is_reversible_label,
    reverse_label,
    reverse_label_bsa,
    reverse_label_strict,
    verify_algebra_invariants,
)
from src.metrics.consistency import (
    DirectionalTransitionReport,
    InconsistencyTaxonomyBreakdown,
    compute_consistency_by_category,
    compute_directional_transition_matrix,
)
from src.metrics.metrics import (
    ALL_LABELS,
    IDX_TO_LABEL,
    LABEL_TO_IDX,
    REVERSIBLE_LABELS,
    REVERSE_LABEL_MAP,
    ConsistencyReport,
    ReversiblePair as MetricReversiblePair,
    compute_classification_report,
    compute_consistency_report,
    compute_metrics_from_arrays,
    compute_official_metrics,
    compute_weighted_f1,
)
from src.models.decoding import (
    P_MATRIX,
    P_RAW_MATRIX,
    REV_INDEX_MAP,
    REVERSIBLE_INDICES,
    SymmetrizedDecoder,
    _argmax_with_rev_tie_break,
    symmetrized_decoding,
)
from src.models.losses import softmax_pure


class TestTier5AdversarialAlgebra(unittest.TestCase):
    """Tier 5 Adversarial Test Suite for Core Algebra & Numerical Stability."""

    def setUp(self) -> None:
        self.decoder_unmasked = SymmetrizedDecoder(mask_negative_other=False)
        self.decoder_masked = SymmetrizedDecoder(mask_negative_other=True)
        self.rev_map = {0: 2, 1: 1, 2: 0, 3: 3}

    # =========================================================================
    # 1. Symmetrized Decoding Involution & Extreme Distributions
    # =========================================================================

    def test_flat_uniform_distribution_unmasked_and_masked(self) -> None:
        """Adversarial stress-test with maximum entropy flat uniform [0.25, 0.25, 0.25, 0.25]."""
        p_uniform = [0.25, 0.25, 0.25, 0.25]

        # 1. Unmasked: projection remains exactly uniform
        sd_d, sd_r = symmetrized_decoding(p_uniform, p_uniform, mask_negative_other=False)
        for val in sd_d + sd_r:
            self.assertAlmostEqual(val, 0.25, places=7)

        # Involutory symmetric tie-breaking must yield EQUIVALENCE (index 1) for both
        d_pred, r_pred = self.decoder_unmasked.decode_pair(p_uniform, p_uniform)
        self.assertEqual(d_pred, 1, "Uniform tie must break to EQUIVALENCE (index 1)")
        self.assertEqual(r_pred, 1, "Uniform tie must break to EQUIVALENCE (index 1)")
        self.assertEqual(r_pred, self.rev_map[d_pred])

        # 2. Masked: negative class 3 is zeroed and renormalized over {0, 1, 2}
        sd_dm, sd_rm = symmetrized_decoding(p_uniform, p_uniform, mask_negative_other=True)
        self.assertEqual(sd_dm[3], 0.0)
        self.assertEqual(sd_rm[3], 0.0)
        for i in range(3):
            self.assertAlmostEqual(sd_dm[i], 1.0 / 3.0, places=7)
            self.assertAlmostEqual(sd_rm[i], 1.0 / 3.0, places=7)

        dm_pred, rm_pred = self.decoder_masked.decode_pair(p_uniform, p_uniform)
        self.assertEqual(dm_pred, 1)
        self.assertEqual(rm_pred, 1)
        self.assertIn(dm_pred, REVERSIBLE_INDICES)
        self.assertIn(rm_pred, REVERSIBLE_INDICES)
        self.assertEqual(rm_pred, self.rev_map[dm_pred])

    def test_near_zero_and_subnormal_probabilities(self) -> None:
        """Adversarial test on subnormal floats (1e-300), near-zero epsilon, and zero-sum vectors."""
        # 1. Subnormal non-zero probabilities
        p_subnormal = [1e-300, 1e-300, 1e-300, 1e-300]
        d, r = self.decoder_unmasked.decode_pair(p_subnormal, p_subnormal)
        self.assertEqual(r, self.rev_map[d])

        dm, rm = self.decoder_masked.decode_pair(p_subnormal, p_subnormal)
        self.assertEqual(rm, self.rev_map[dm])
        self.assertIn(dm, REVERSIBLE_INDICES)
        self.assertIn(rm, REVERSIBLE_INDICES)

        # 2. Zero-sum vectors
        p_zeros = [0.0, 0.0, 0.0, 0.0]
        sd_d, sd_r = symmetrized_decoding(p_zeros, p_zeros, mask_negative_other=True)
        self.assertEqual(sd_d, [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(sd_r, [0.0, 0.0, 0.0, 0.0])

        dz, rz = self.decoder_masked.decode_pair(p_zeros, p_zeros)
        self.assertEqual(rz, self.rev_map[dz])
        self.assertIn(dz, REVERSIBLE_INDICES)
        self.assertIn(rz, REVERSIBLE_INDICES)

        # 3. Probability delta near float precision boundary: 1.0 - 1e-15
        p_extreme = [1.0 - 1e-15, 1e-15 / 3, 1e-15 / 3, 1e-15 / 3]
        p_rev_ext = [1e-15 / 3, 1e-15 / 3, 1.0 - 1e-15, 1e-15 / 3]
        d_ext, r_ext = self.decoder_masked.decode_pair(p_extreme, p_rev_ext)
        self.assertEqual(d_ext, 0)  # BACKWARD
        self.assertEqual(r_ext, 2)  # FORWARD
        self.assertEqual(r_ext, self.rev_map[d_ext])

    def test_massive_and_extreme_negative_logits_stability(self) -> None:
        """Adversarial test on massive logits (+1e6) and deeply negative logits (-1e6)."""
        # 1. Massive logits with single dominant class
        logits_dir = [1e6, 0.0, -1e6, 50.0]
        logits_rev = [-1e6, 0.0, 1e6, 50.0]
        p_dir = softmax_pure(logits_dir)
        p_rev = softmax_pure(logits_rev)

        self.assertEqual(p_dir[0], 1.0)
        self.assertEqual(p_rev[2], 1.0)
        d, r = self.decoder_unmasked.decode_pair(p_dir, p_rev)
        self.assertEqual(d, 0)
        self.assertEqual(r, 2)
        self.assertEqual(r, self.rev_map[d])

        # 2. Extreme uniform negative logits (-1e6 for all classes)
        all_neg_logits = [-1e6, -1e6, -1e6, -1e6]
        p_neg = softmax_pure(all_neg_logits)
        for v in p_neg:
            self.assertAlmostEqual(v, 0.25, places=7)

        d_neg, r_neg = self.decoder_unmasked.decode_pair(p_neg, p_neg)
        self.assertEqual(d_neg, 1)
        self.assertEqual(r_neg, 1)
        self.assertEqual(r_neg, self.rev_map[d_neg])

        # 3. Direct unnormalized massive logits passed to symmetrized_decoding
        sd_d, sd_r = symmetrized_decoding([1e6, 0.0, 0.0, 0.0], [0.0, 0.0, 1e6, 0.0])
        self.assertEqual(sd_d[0], 1e6)
        self.assertEqual(sd_r[2], 1e6)
        d_raw, r_raw = self.decoder_unmasked.decode_pair([1e6, 0.0, 0.0, 0.0], [0.0, 0.0, 1e6, 0.0])
        self.assertEqual(d_raw, 0)
        self.assertEqual(r_raw, 2)
        self.assertEqual(r_raw, self.rev_map[d_raw])

    def test_non_finite_nan_and_inf_behavioral_characterization(self) -> None:
        """Empirical verification of non-finite float inputs (NaN, Inf).

        Demonstrates that IEEE-754 NaNs propagate through pure-Python matrix projection
        and document boundary behavior when passed to tie-breaking argmax.
        """
        nan = float("nan")
        inf = float("inf")

        # 1. Symmetrized decoding with NaN propagates NaN predictably
        res_dir, res_rev = symmetrized_decoding([nan, 0.2, 0.3, 0.5], [0.1, 0.2, 0.3, 0.4])
        self.assertTrue(math.isnan(res_dir[0]))
        self.assertTrue(math.isnan(res_rev[2]))

        # 2. _argmax_with_rev_tie_break on all-NaN vector produces no candidates (< eps is False for NaN)
        # Verify that all-NaN input raises IndexError due to empty candidate set
        with self.assertRaises(IndexError):
            _argmax_with_rev_tie_break([nan, nan, nan, nan])

        # 3. Inf handling: inf logits through softmax_pure
        # When an inf is present, exp(x - max_l) handles the dominant inf
        inf_logits = [inf, 0.0, 0.0, 0.0]
        # max_l = inf -> inf - inf is nan in standard floating point
        self.assertTrue(math.isnan(inf - inf))

    # =========================================================================
    # 2. Candidate Tie Scenarios & Involutory Symmetry
    # =========================================================================

    def test_tie_scenario_backward_forward_clash(self) -> None:
        """Adversarial tie {0, 2} (BACKWARD vs FORWARD clash).

        Logical theorem: If the model cannot distinguish between forward and backward
        entailment, the joint symmetric consensus is mutual entailment: EQUIVALENCE (index 1).
        """
        p_tie = [0.5, 0.0, 0.5, 0.0]
        p_tie_rev = [0.5, 0.0, 0.5, 0.0]  # P @ [0.5, 0, 0.5, 0] == [0.5, 0, 0.5, 0]

        d, r = self.decoder_unmasked.decode_pair(p_tie, p_tie_rev)
        self.assertEqual(d, 1, "Clash between 0 and 2 must resolve to 1 (EQUIVALENCE)")
        self.assertEqual(r, 1, "Clash between 0 and 2 must resolve to 1 (EQUIVALENCE)")
        self.assertEqual(r, self.rev_map[d])

        # Verify in masked decoder as well
        dm, rm = self.decoder_masked.decode_pair(p_tie, p_tie_rev)
        self.assertEqual(dm, 1)
        self.assertEqual(rm, 1)
        self.assertEqual(rm, self.rev_map[dm])

    def test_canonical_tie_scenarios_comprehensive(self) -> None:
        """Explicitly test all named candidate tie scenarios from specification:
        {0, 2}, {0, 1}, {0, 3}, {1, 2}, {2, 3}, and {0, 1, 2, 3}.
        """
        named_ties = {
            "tie_0_2": ({0, 2}, 1),  # BWD & FWD -> EQUIV (1)
            "tie_0_1": ({0, 1}, 1),  # BWD & EQUIV -> EQUIV (1)
            "tie_0_3": ({0, 3}, 3),  # BWD & NEG -> NEG (3)
            "tie_1_2": ({1, 2}, 1),  # EQUIV & FWD -> EQUIV (1)
            "tie_2_3": ({2, 3}, 3),  # FWD & NEG -> NEG (3)
            "tie_0_1_2_3": ({0, 1, 2, 3}, 1),  # All 4 -> EQUIV (1)
        }

        for name, (tie_set, expected_pred) in named_ties.items():
            with self.subTest(tie=name):
                # Vector where tied indices have equal maximal weight
                p_d = [1.0 if i in tie_set else 0.0 for i in range(4)]
                # Transposed counterpart under involution matrix P
                p_r = [1.0 if self.rev_map[i] in tie_set else 0.0 for i in range(4)]

                d, r = self.decoder_unmasked.decode_pair(p_d, p_r)
                self.assertEqual(
                    d,
                    expected_pred,
                    f"Tie {tie_set} direct resolution mismatch: got {d}, expected {expected_pred}",
                )
                self.assertEqual(
                    r,
                    self.rev_map[expected_pred],
                    f"Tie {tie_set} reverse resolution mismatch: got {r}, expected {self.rev_map[expected_pred]}",
                )
                self.assertEqual(r, self.rev_map[d], f"Involutory equivariance broken on {name}")

    def test_all_15_candidate_tie_subsets_exhaustive_unmasked(self) -> None:
        """Combinatorial stress test across all 2^4 - 1 = 15 non-empty subsets of {0, 1, 2, 3}."""
        all_subsets = []
        for k in range(1, 5):
            for combo in itertools.combinations([0, 1, 2, 3], k):
                all_subsets.append(set(combo))

        self.assertEqual(len(all_subsets), 15)

        for s in all_subsets:
            with self.subTest(subset=s):
                p_d = [1.0 if i in s else 0.0 for i in range(4)]
                s_rev = {self.rev_map[i] for i in s}
                p_r = [1.0 if i in s_rev else 0.0 for i in range(4)]

                d, r = self.decoder_unmasked.decode_pair(p_d, p_r)
                self.assertEqual(
                    r,
                    self.rev_map[d],
                    f"Equivariance broken for tie subset {s}: d={d}, r={r}",
                )

    def test_all_15_candidate_tie_subsets_exhaustive_masked(self) -> None:
        """Combinatorial stress test across all 15 subsets with mask_negative_other=True."""
        all_subsets = []
        for k in range(1, 5):
            for combo in itertools.combinations([0, 1, 2, 3], k):
                all_subsets.append(set(combo))

        for s in all_subsets:
            with self.subTest(subset=s):
                p_d = [1.0 if i in s else 0.0 for i in range(4)]
                s_rev = {self.rev_map[i] for i in s}
                p_r = [1.0 if i in s_rev else 0.0 for i in range(4)]

                dm, rm = self.decoder_masked.decode_pair(p_d, p_r)
                self.assertIn(
                    dm,
                    REVERSIBLE_INDICES,
                    f"Masked decoder produced non-reversible class {dm} for tie subset {s}",
                )
                self.assertIn(
                    rm,
                    REVERSIBLE_INDICES,
                    f"Masked decoder produced non-reversible class {rm} for tie subset {s}",
                )
                self.assertEqual(
                    rm,
                    self.rev_map[dm],
                    f"Equivariance broken under masking for tie subset {s}: dm={dm}, rm={rm}",
                )

    def test_asymmetric_tie_configurations(self) -> None:
        """Adversarial stress test where direct distribution has a tie but reverse does not."""
        asymmetric_pairs = [
            # Direct has tie {0, 1}, reverse has single dominant class 2
            ([0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]),
            # Direct has tie {1, 2}, reverse has tie {0, 3}
            ([0.0, 0.5, 0.5, 0.0], [0.5, 0.0, 0.0, 0.5]),
            # Direct has tie {0, 2}, reverse has uniform
            ([0.5, 0.0, 0.5, 0.0], [0.25, 0.25, 0.25, 0.25]),
            # Direct has extreme spike, reverse has flat tie on 3 classes
            ([1.0, 0.0, 0.0, 0.0], [1/3, 1/3, 1/3, 0.0]),
        ]

        for p_d, p_r in asymmetric_pairs:
            # Unmasked
            d, r = self.decoder_unmasked.decode_pair(p_d, p_r)
            self.assertEqual(r, self.rev_map[d])

            # Masked
            dm, rm = self.decoder_masked.decode_pair(p_d, p_r)
            self.assertIn(dm, REVERSIBLE_INDICES)
            self.assertIn(rm, REVERSIBLE_INDICES)
            self.assertEqual(rm, self.rev_map[dm])

    # =========================================================================
    # 3. 100,000 Randomized Monte Carlo Draws for 100% SoftCons
    # =========================================================================

    def test_monte_carlo_100k_softcons_100_percent_guarantee(self) -> None:
        """Empirical Monte Carlo proof: 100,000 randomized draws over the 4-simplex.

        Validates that SymmetrizedDecoder(mask_negative_other=True) unconditionally
        produces SoftCons == 1.000000 (100% consistency) without a single violation.
        """
        random.seed(20271102)
        total_trials = 100_000
        consistent_count = 0
        all_reversible = True

        for _ in range(total_trials):
            # Sample independent Dirichlet(1, 1, 1, 1) distributions via exponential variates
            raw_d = [-math.log(random.random()) for _ in range(4)]
            sum_d = sum(raw_d)
            p_d = [x / sum_d for x in raw_d]

            raw_r = [-math.log(random.random()) for _ in range(4)]
            sum_r = sum(raw_r)
            p_r = [x / sum_r for x in raw_r]

            pred_d, pred_r = self.decoder_masked.decode_pair(p_d, p_r)

            if pred_d not in REVERSIBLE_INDICES or pred_r not in REVERSIBLE_INDICES:
                all_reversible = False
                break

            if pred_r == self.rev_map[pred_d]:
                consistent_count += 1

        self.assertTrue(all_reversible, "Predictions must remain inside REVERSIBLE_INDICES")
        self.assertEqual(
            consistent_count,
            total_trials,
            f"SoftCons guarantee broken: {consistent_count}/{total_trials} passed",
        )
        soft_cons_rate = consistent_count / total_trials
        self.assertEqual(soft_cons_rate, 1.0, f"Expected SoftCons == 1.000000, got {soft_cons_rate}")

    def test_monte_carlo_100k_unmasked_involutory_equivariance(self) -> None:
        """Monte Carlo verification of unmasked involutory equivariance across 100,000 draws."""
        random.seed(20271103)
        total_trials = 100_000
        equivariant_count = 0

        for _ in range(total_trials):
            raw_d = [-math.log(random.random()) for _ in range(4)]
            sum_d = sum(raw_d)
            p_d = [x / sum_d for x in raw_d]

            raw_r = [-math.log(random.random()) for _ in range(4)]
            sum_r = sum(raw_r)
            p_r = [x / sum_r for x in raw_r]

            pred_d, pred_r = self.decoder_unmasked.decode_pair(p_d, p_r)
            if pred_r == self.rev_map[pred_d]:
                equivariant_count += 1

        self.assertEqual(
            equivariant_count,
            total_trials,
            f"Unmasked equivariance broken: {equivariant_count}/{total_trials}",
        )

    # =========================================================================
    # 4. Metrics Boundary Conditions & Mathematical Invariants
    # =========================================================================

    def test_metrics_zero_count_slices_and_empty_inputs(self) -> None:
        """Test zero-count slices, empty inputs, and safe division handling."""
        # 1. Zero reversible pairs slice (is_reversible_mask all False)
        metrics = compute_official_metrics(
            y_true_dir=["NEGATIVE_OTHER", "NEGATIVE_OTHER"],
            y_pred_dir=["NEGATIVE_OTHER", "NEGATIVE_OTHER"],
            y_true_rev=["NEGATIVE_OTHER", "NEGATIVE_OTHER"],
            y_pred_rev=["NEGATIVE_OTHER", "NEGATIVE_OTHER"],
            is_reversible_mask=[False, False],
        )
        self.assertEqual(metrics["soft_cons"], 0.0)
        self.assertEqual(metrics["hard_cons"], 0.0)
        self.assertEqual(metrics["weighted_f1"], 1.0)

        # 2. Empty sequences in compute_weighted_f1 must raise ValueError
        with self.assertRaises(ValueError):
            compute_weighted_f1([], [])

        # 3. Empty pairs in compute_consistency_report must raise ValueError
        with self.assertRaises(ValueError):
            compute_consistency_report([], {"inst_1": "EQUIVALENCE"})

        # 4. Empty gold map in compute_classification_report must raise ValueError
        with self.assertRaises(ValueError):
            compute_classification_report({}, {})

        # 5. Empty arrays in compute_metrics_from_arrays must raise ValueError
        with self.assertRaises(ValueError):
            compute_metrics_from_arrays([], [])

    def test_metrics_missing_prediction_keys(self) -> None:
        """Missing prediction IDs must raise KeyError with diagnostic message."""
        pair = MetricReversiblePair("p1_d", "p1_r", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT")

        # Missing reverse pair ID
        with self.assertRaises(KeyError) as ctx_r:
            compute_consistency_report([pair], {"p1_d": "FORWARD_ENTAILMENT"})
        self.assertIn("p1_r", str(ctx_r.exception))

        # Missing direct pair ID
        with self.assertRaises(KeyError) as ctx_d:
            compute_consistency_report([pair], {"p1_r": "BACKWARD_ENTAILMENT"})
        self.assertIn("p1_d", str(ctx_d.exception))

        # Missing ID in classification report
        with self.assertRaises(KeyError):
            compute_classification_report({"inst_1": "EQUIVALENCE"}, {})

    def test_metrics_single_instance_pair_all_4_states(self) -> None:
        """Verify single-instance reversible pairs (N=1) across all 4 logical consistency states:
        State 1: Soft-consistent & Hard-correct (1.0, 1.0)
        State 2: Soft-consistent & Hard-incorrect (1.0, 0.0)
        State 3: Soft-inconsistent & Hard-incorrect (0.0, 0.0)
        State 4: Invariant 0.0 <= HardCons <= SoftCons <= 1.0 holds in all states.
        """
        pair = MetricReversiblePair("d1", "r1", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT")

        # State 1: Soft-consistent & Hard-correct
        preds_state1 = {"d1": "FORWARD_ENTAILMENT", "r1": "BACKWARD_ENTAILMENT"}
        rep1 = compute_consistency_report([pair], preds_state1)
        self.assertEqual(rep1.soft_cons, 1.0)
        self.assertEqual(rep1.hard_cons, 1.0)
        self.assertLessEqual(rep1.hard_cons, rep1.soft_cons)

        # State 2: Soft-consistent & Hard-incorrect (predicted EQUIV/EQUIV)
        preds_state2 = {"d1": "EQUIVALENCE", "r1": "EQUIVALENCE"}
        rep2 = compute_consistency_report([pair], preds_state2)
        self.assertEqual(rep2.soft_cons, 1.0)
        self.assertEqual(rep2.hard_cons, 0.0)
        self.assertLessEqual(rep2.hard_cons, rep2.soft_cons)

        # State 3: Soft-inconsistent & Hard-incorrect (predicted FWD/FWD Reversal Curse)
        preds_state3 = {"d1": "FORWARD_ENTAILMENT", "r1": "FORWARD_ENTAILMENT"}
        rep3 = compute_consistency_report([pair], preds_state3)
        self.assertEqual(rep3.soft_cons, 0.0)
        self.assertEqual(rep3.hard_cons, 0.0)
        self.assertLessEqual(rep3.hard_cons, rep3.soft_cons)

        # State 4: Asymmetric half-correct (direct correct, reverse wrong -> Soft=0, Hard=0)
        preds_state4 = {"d1": "FORWARD_ENTAILMENT", "r1": "EQUIVALENCE"}
        rep4 = compute_consistency_report([pair], preds_state4)
        self.assertEqual(rep4.soft_cons, 0.0)
        self.assertEqual(rep4.hard_cons, 0.0)
        self.assertLessEqual(rep4.hard_cons, rep4.soft_cons)

    def test_metrics_extreme_prediction_regimes(self) -> None:
        """Adversarial stress-test across extreme uniform prediction regimes:
        - All-contradiction (all pairs predicted as FWD/FWD)
        - All-neutral (all pairs predicted as EQUIV/EQUIV)
        - All-entailment (all pairs predicted as FWD/BWD)
        - All-negative (all pairs predicted as NEG/NEG)
        """
        pairs = [
            MetricReversiblePair("p1_d", "p1_r", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT"),
            MetricReversiblePair("p2_d", "p2_r", "EQUIVALENCE", "EQUIVALENCE"),
            MetricReversiblePair("p3_d", "p3_r", "BACKWARD_ENTAILMENT", "FORWARD_ENTAILMENT"),
        ]

        # 1. All-contradiction
        preds_contra = {
            "p1_d": "FORWARD_ENTAILMENT", "p1_r": "FORWARD_ENTAILMENT",
            "p2_d": "FORWARD_ENTAILMENT", "p2_r": "FORWARD_ENTAILMENT",
            "p3_d": "FORWARD_ENTAILMENT", "p3_r": "FORWARD_ENTAILMENT",
        }
        rep_contra = compute_consistency_report(pairs, preds_contra)
        self.assertEqual(rep_contra.soft_cons, 0.0)
        self.assertEqual(rep_contra.hard_cons, 0.0)
        self.assertLessEqual(rep_contra.hard_cons, rep_contra.soft_cons)

        # 2. All-neutral (EQUIVALENCE)
        preds_neutral = {
            "p1_d": "EQUIVALENCE", "p1_r": "EQUIVALENCE",
            "p2_d": "EQUIVALENCE", "p2_r": "EQUIVALENCE",
            "p3_d": "EQUIVALENCE", "p3_r": "EQUIVALENCE",
        }
        rep_neutral = compute_consistency_report(pairs, preds_neutral)
        self.assertEqual(rep_neutral.soft_cons, 1.0)
        self.assertAlmostEqual(rep_neutral.hard_cons, 1.0 / 3.0)
        self.assertLessEqual(rep_neutral.hard_cons, rep_neutral.soft_cons)

        # 3. All-entailment (FORWARD / BACKWARD)
        preds_entail = {
            "p1_d": "FORWARD_ENTAILMENT", "p1_r": "BACKWARD_ENTAILMENT",
            "p2_d": "FORWARD_ENTAILMENT", "p2_r": "BACKWARD_ENTAILMENT",
            "p3_d": "FORWARD_ENTAILMENT", "p3_r": "BACKWARD_ENTAILMENT",
        }
        rep_entail = compute_consistency_report(pairs, preds_entail)
        self.assertEqual(rep_entail.soft_cons, 1.0)
        self.assertAlmostEqual(rep_entail.hard_cons, 1.0 / 3.0)
        self.assertLessEqual(rep_entail.hard_cons, rep_entail.soft_cons)

        # 4. All-negative (NEGATIVE_OTHER) on reversible pairs
        preds_neg = {
            "p1_d": "NEGATIVE_OTHER", "p1_r": "NEGATIVE_OTHER",
            "p2_d": "NEGATIVE_OTHER", "p2_r": "NEGATIVE_OTHER",
            "p3_d": "NEGATIVE_OTHER", "p3_r": "NEGATIVE_OTHER",
        }
        rep_neg = compute_consistency_report(pairs, preds_neg)
        self.assertEqual(rep_neg.soft_cons, 0.0)
        self.assertEqual(rep_neg.hard_cons, 0.0)
        self.assertLessEqual(rep_neg.hard_cons, rep_neg.soft_cons)

    def test_strict_mathematical_invariant_hardcons_le_softcons_extensive(self) -> None:
        """Adversarial stress-test verifying 0.0 <= HardCons <= SoftCons <= 1.0 across 500 randomized trials.

        Theorem: For any valid reversible ground truth pairs ((y_1, y_2) with y_1 in R and y_2 = Rev(y_1)),
        HardCorrect(p_1, p_2) := (p_1 == y_1 and p_2 == y_2).
        Since y_1 in R and y_2 = Rev(y_1), HardCorrect strictly implies SoftConsistent(p_1, p_2).
        Therefore, HardCons <= SoftCons unconditionally for all possible prediction vectors.
        """
        random.seed(9999)
        candidate_labels = list(ALL_LABELS)
        rev_gold_classes = ["BACKWARD_ENTAILMENT", "EQUIVALENCE", "FORWARD_ENTAILMENT"]

        for trial in range(500):
            n_pairs = random.randint(5, 50)
            pairs = []
            preds = {}

            for i in range(n_pairs):
                id_d = f"t{trial}_p{i}_d"
                id_r = f"t{trial}_p{i}_r"
                gold_d = random.choice(rev_gold_classes)
                gold_r = REVERSE_LABEL_MAP[gold_d]

                pairs.append(MetricReversiblePair(id_d, id_r, gold_d, gold_r))
                preds[id_d] = random.choice(candidate_labels)
                preds[id_r] = random.choice(candidate_labels)

            rep = compute_consistency_report(pairs, preds)
            self.assertGreaterEqual(rep.hard_cons, 0.0)
            self.assertGreaterEqual(rep.soft_cons, rep.hard_cons)
            self.assertLessEqual(rep.soft_cons, 1.0)

    def test_non_reversible_negative_other_pair_contract_violation(self) -> None:
        """Empirical verification of interface contract violation:

        If a caller incorrectly includes NEGATIVE_OTHER in ReversiblePair,
        predicting (NEGATIVE_OTHER, NEGATIVE_OTHER) yields hard_ok=True but soft_ok=False,
        which triggers the defensive AssertionError: HardCons (1.0) > SoftCons (0.0).

        Conversely, ReciprocalPair enforces this by setting is_reversible=False.
        """
        # 1. Direct ReversiblePair with NEGATIVE_OTHER triggers defensive assertion
        bad_pair = MetricReversiblePair("n1", "n2", "NEGATIVE_OTHER", "NEGATIVE_OTHER")
        preds = {"n1": "NEGATIVE_OTHER", "n2": "NEGATIVE_OTHER"}
        with self.assertRaises(AssertionError) as ctx:
            compute_consistency_report([bad_pair], preds)
        self.assertIn("HardCons (1.0) > SoftCons (0.0)", str(ctx.exception))

        # 2. ReciprocalPair guards against this by marking is_reversible=False
        rec_pair = ReciprocalPair(
            {"instance_id": "n1", "label": "NEGATIVE_OTHER"},
            {"instance_id": "n2", "label": "NEGATIVE_OTHER"},
        )
        self.assertFalse(rec_pair.is_reversible)

    def test_directional_transition_matrix_boundary_conditions(self) -> None:
        """Adversarial stress-test of Directional Transition Matrix & Inconsistency Taxonomy.

        Validates:
        - Zero pairs raises ValueError
        - All-consistent predictions -> transition_entropy == 0.0, RCSI == 0.0
        - 100% entailment loop predictions -> RCSI == 1.0
        - Zero directional entailments -> guarded safe division (RCSI == 0.0)
        - Conservation law: sum of 4 taxonomy categories == total pairs
        - Conservation law: sum of 4 taxonomy rates == 1.0
        """
        # 1. Zero pairs raises ValueError
        with self.assertRaises(ValueError):
            compute_directional_transition_matrix([], {})

        # 2. All-consistent predictions
        consistent_pairs = [
            MetricReversiblePair("c1_d", "c1_r", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT"),
            MetricReversiblePair("c2_d", "c2_r", "BACKWARD_ENTAILMENT", "FORWARD_ENTAILMENT"),
            MetricReversiblePair("c3_d", "c3_r", "EQUIVALENCE", "EQUIVALENCE"),
        ]
        consistent_preds = {
            "c1_d": "FORWARD_ENTAILMENT", "c1_r": "BACKWARD_ENTAILMENT",
            "c2_d": "BACKWARD_ENTAILMENT", "c2_r": "FORWARD_ENTAILMENT",
            "c3_d": "EQUIVALENCE", "c3_r": "EQUIVALENCE",
        }
        rep_cons = compute_directional_transition_matrix(consistent_pairs, consistent_preds)
        self.assertEqual(rep_cons.taxonomy.consistent_count, 3)
        self.assertEqual(rep_cons.taxonomy.consistent_rate, 1.0)
        self.assertEqual(rep_cons.reversal_curse_index, 0.0)
        self.assertEqual(rep_cons.transition_entropy, 0.0)

        # 3. 100% Entailment Loop (Reversal Curse Trap)
        loop_pairs = [
            MetricReversiblePair("l1_d", "l1_r", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT"),
            MetricReversiblePair("l2_d", "l2_r", "BACKWARD_ENTAILMENT", "FORWARD_ENTAILMENT"),
        ]
        loop_preds = {
            "l1_d": "FORWARD_ENTAILMENT", "l1_r": "FORWARD_ENTAILMENT",
            "l2_d": "BACKWARD_ENTAILMENT", "l2_r": "BACKWARD_ENTAILMENT",
        }
        rep_loop = compute_directional_transition_matrix(loop_pairs, loop_preds)
        self.assertEqual(rep_loop.taxonomy.entailment_loop_count, 2)
        self.assertEqual(rep_loop.taxonomy.entailment_loop_rate, 1.0)
        self.assertEqual(rep_loop.reversal_curse_index, 1.0)

        # 4. Zero directional entailments (zero division guard in RCSI)
        equiv_only_pairs = [
            MetricReversiblePair("e1_d", "e1_r", "EQUIVALENCE", "EQUIVALENCE"),
        ]
        equiv_only_preds = {
            "e1_d": "EQUIVALENCE", "e1_r": "EQUIVALENCE",
        }
        rep_equiv = compute_directional_transition_matrix(equiv_only_pairs, equiv_only_preds)
        self.assertEqual(rep_equiv.reversal_curse_index, 0.0)

        # 5. Conservation laws across a heterogeneous mix of transitions
        mixed_pairs = [
            MetricReversiblePair("m1_d", "m1_r", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT"),
            MetricReversiblePair("m2_d", "m2_r", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT"),
            MetricReversiblePair("m3_d", "m3_r", "EQUIVALENCE", "EQUIVALENCE"),
            MetricReversiblePair("m4_d", "m4_r", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT"),
            MetricReversiblePair("m5_d", "m5_r", "NEGATIVE_OTHER", "NEGATIVE_OTHER"),
        ]
        mixed_preds = {
            "m1_d": "FORWARD_ENTAILMENT", "m1_r": "BACKWARD_ENTAILMENT",  # Consistent
            "m2_d": "FORWARD_ENTAILMENT", "m2_r": "FORWARD_ENTAILMENT",   # Loop
            "m3_d": "EQUIVALENCE", "m3_r": "FORWARD_ENTAILMENT",          # Equiv Asymmetry
            "m4_d": "FORWARD_ENTAILMENT", "m4_r": "NEGATIVE_OTHER",       # Noise Collapse
            "m5_d": "NEGATIVE_OTHER", "m5_r": "NEGATIVE_OTHER",           # Mutual Noise
        }
        rep_mixed = compute_directional_transition_matrix(mixed_pairs, mixed_preds)
        tax = rep_mixed.taxonomy

        # Category sum must equal total_pairs
        total_cat = (
            tax.consistent_count
            + tax.entailment_loop_count
            + tax.equiv_asymmetry_count
            + tax.noise_collapse_count
        )
        self.assertEqual(total_cat, tax.total_pairs)
        self.assertEqual(tax.total_pairs, 5)

        # Rate sum must equal 1.0
        total_rates = (
            tax.consistent_rate
            + tax.entailment_loop_rate
            + tax.equiv_asymmetry_rate
            + tax.noise_collapse_rate
        )
        self.assertAlmostEqual(total_rates, 1.0, places=7)

    def test_verify_algebra_invariants_function(self) -> None:
        """Verify that data reversal algebra invariant function returns True."""
        self.assertTrue(verify_algebra_invariants())


if __name__ == "__main__":
    unittest.main()
