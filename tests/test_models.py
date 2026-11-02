"""
tests/test_models.py

Comprehensive unit tests for Milestone 2 (M2: Models & Reversal Strategies) for DiCo-NLI.
Tests:
1. Baseline Cross-Encoder & Mock Engine (CrossEncoderNLI, MockCrossEncoder, MockTokenizer, CrossEncoderOutput).
2. Symmetrical Augmentation (BSA) model & loss instantiation.
3. Consistency Regularization Loss & Siamese Architecture (SiameseCRLModel, TranspositionOperator,
   Symmetric KL, Jensen-Shannon, dynamic warmup scheduling).
4. Symmetrized Decoding & Model Wrapper (SymmetrizedDecoder, SymmetrizedModelWrapper, 100% SoftCons).
5. Model Factory & Loss Factory (build_model, build_loss, build_tokenizer, device auto-resolution).
"""

import math
import unittest
from typing import Any, Dict, List

from src.models import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    P_MATRIX,
    P_RAW_MATRIX,
    REV_INDEX_MAP,
    REVERSIBLE_INDICES,
    ConsistencyRegularizationLoss,
    CrossEncoderNLI,
    CrossEncoderOutput,
    JensenShannonDivergenceLoss,
    LabelIndex,
    MockCrossEncoder,
    MockModelOutput,
    MockSequenceEncoder,
    MockTokenizer,
    MockTransformerModel,
    SiameseCRLModel,
    SiameseCRLOutput,
    SymmetrizedDecoder,
    SymmetrizedModelWrapper,
    SymmetricKLDivergenceLoss,
    TranspositionOperator,
    build_loss,
    build_model,
    build_tokenizer,
    canonicalize_model_name,
    cross_entropy_pure,
    detect_backbone_family,
    jensen_shannon_pure,
    kl_divergence_pure,
    matmul_mat_mat,
    matmul_vec_mat,
    resolve_device,
    resolve_torch_dtype,
    softmax_pure,
    symmetric_kl_pure,
    symmetrized_decoding,
)


class TestCrossEncoderBaseline(unittest.TestCase):
    """Unit tests for standard cross-encoder backbone and mock engine."""

    def test_label_constants_and_enum(self):
        """Verify canonical 4-class label index definitions."""
        self.assertEqual(LabelIndex.BACKWARD_ENTAILMENT, 0)
        self.assertEqual(LabelIndex.EQUIVALENCE, 1)
        self.assertEqual(LabelIndex.FORWARD_ENTAILMENT, 2)
        self.assertEqual(LabelIndex.NEGATIVE_OTHER, 3)

        self.assertEqual(LABEL_TO_ID["BACKWARD_ENTAILMENT"], 0)
        self.assertEqual(LABEL_TO_ID["EQUIVALENCE"], 1)
        self.assertEqual(LABEL_TO_ID["FORWARD_ENTAILMENT"], 2)
        self.assertEqual(LABEL_TO_ID["NEGATIVE_OTHER"], 3)

        for k, v in LABEL_TO_ID.items():
            self.assertEqual(ID_TO_LABEL[v], k)

    def test_backbone_canonicalization_and_detection(self):
        """Verify backbone family resolution and model alias normalization."""
        self.assertEqual(
            canonicalize_model_name("deberta-v3-base"),
            "microsoft/deberta-v3-base",
        )
        self.assertEqual(
            canonicalize_model_name("beto"),
            "dccuchile/bert-base-spanish-wwm-cased",
        )
        self.assertEqual(
            canonicalize_model_name("roberta-bne"),
            "PlanTL-GOB-ES/roberta-base-bne",
        )

        self.assertEqual(detect_backbone_family("microsoft/deberta-v3-base"), "deberta")
        self.assertEqual(detect_backbone_family("dccuchile/bert-base-spanish-wwm-cased"), "bert")
        self.assertEqual(detect_backbone_family("PlanTL-GOB-ES/roberta-base-bne"), "roberta")
        self.assertEqual(detect_backbone_family("custom-model"), "generic")

    def test_mock_tokenizer_delimiter_formatting(self):
        """Verify delimiter formatting: BERT [CLS]/[SEP] vs RoBERTa <s>/</s></s>."""
        bert_tok = MockTokenizer(model_name="dccuchile/bert-base-spanish-wwm-cased")
        self.assertEqual(bert_tok.cls_token, "[CLS]")
        self.assertEqual(bert_tok.sep_token, "[SEP]")

        out_bert = bert_tok("Premise text", "Hypothesis text", return_tensors=None)
        self.assertIn("input_ids", out_bert)
        self.assertIn("attention_mask", out_bert)
        self.assertIn("token_type_ids", out_bert)
        # BERT has segment type ids: 0 for premise, 1 for hypothesis
        self.assertIn(1, out_bert["token_type_ids"][0])

        roberta_tok = MockTokenizer(model_name="PlanTL-GOB-ES/roberta-base-bne")
        self.assertEqual(roberta_tok.cls_token, "<s>")
        self.assertEqual(roberta_tok.sep_token, "</s>")

        out_roberta = roberta_tok("Premise text", "Hypothesis text", return_tensors=None)
        self.assertIn("input_ids", out_roberta)
        # RoBERTa does not use token_type_ids
        self.assertNotIn("token_type_ids", out_roberta)

    def test_mock_cross_encoder_heuristics_and_forward(self):
        """Verify calibrated heuristics and CrossEncoderOutput container."""
        model = MockCrossEncoder(model_name="mock-cross-encoder")

        # Equal text -> EQUIVALENCE (index 1)
        pred_eq = model.predict("the car is blue", "the car is blue")
        self.assertEqual(pred_eq, "EQUIVALENCE")

        # More specific premise -> entails shorter hypothesis -> FORWARD_ENTAILMENT (index 2)
        pred_fwd = model.predict("a red sports car speeding down the avenue", "a vehicle")
        self.assertEqual(pred_fwd, "FORWARD_ENTAILMENT")

        # Shorter premise -> entailed by longer hypothesis -> BACKWARD_ENTAILMENT (index 0)
        pred_bwd = model.predict("a vehicle", "a red sports car speeding down the avenue")
        self.assertEqual(pred_bwd, "BACKWARD_ENTAILMENT")

        # No lexical overlap -> NEGATIVE_OTHER (index 3)
        pred_neg = model.predict("a golden retriever dog", "a microwave oven cooking")
        self.assertEqual(pred_neg, "NEGATIVE_OTHER")

        # Forward pass returning CrossEncoderOutput
        tok = MockTokenizer()
        batch = tok(["a vehicle"], ["a sports car"])
        output = model.forward(input_ids=batch["input_ids"], labels=[0])

        self.assertIsInstance(output, CrossEncoderOutput)
        self.assertIsNotNone(output.logits)
        self.assertIsNotNone(output.probabilities)
        self.assertIsNotNone(output.predictions)
        self.assertIsNotNone(output.loss)

        # Container access: attribute, dict, and tuple unpacking
        self.assertIsNotNone(output["loss"])
        loss, logits, probs, preds = output
        self.assertEqual(loss, output.loss)
        self.assertEqual(logits, output.logits)

    def test_cross_encoder_nli_wrapper(self):
        """Verify CrossEncoderNLI wrapper in mock mode."""
        model = CrossEncoderNLI(model_name_or_path="microsoft/deberta-v3-base", is_mock=True)
        self.assertTrue(model.is_mock)
        pred = model.predict("the red sports car", "the vehicle")
        self.assertIn(pred, ["FORWARD_ENTAILMENT", "EQUIVALENCE", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"])

    def test_cross_encoder_nli_eval_and_train_cpu(self):
        """Verify .eval() and .train() work on CrossEncoderNLI and MockCrossEncoder without PyTorch."""
        mock_enc = MockCrossEncoder()
        self.assertIs(mock_enc.eval(), mock_enc)
        self.assertFalse(mock_enc.training)
        self.assertIs(mock_enc.train(True), mock_enc)
        self.assertTrue(mock_enc.training)

        nli = CrossEncoderNLI("mock", is_mock=True)
        self.assertIs(nli.eval(), nli)
        self.assertFalse(nli.training)
        self.assertFalse(nli.mock_engine.training)
        self.assertIs(nli.train(True), nli)
        self.assertTrue(nli.training)
        self.assertTrue(nli.mock_engine.training)

        m_base = build_model("mock", method="baseline")
        self.assertIs(m_base.eval(), m_base)
        self.assertFalse(m_base.training)


class TestLossesAndCRL(unittest.TestCase):
    """Unit tests for losses, involution permutation algebra, and Siamese CRL."""

    def test_involution_matrix_properties(self):
        """Verify involution matrix P satisfies P == P^T and P^2 == I."""
        I4 = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
        P_sq = matmul_mat_mat(P_RAW_MATRIX, P_RAW_MATRIX)

        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(P_sq[i][j], I4[i][j], places=7)
                self.assertAlmostEqual(P_RAW_MATRIX[i][j], P_RAW_MATRIX[j][i], places=7)

    def test_transposition_operator(self):
        """Verify TranspositionOperator swaps index 0 and index 2."""
        op = TranspositionOperator()
        # Vector: [p_bwd, p_eq, p_fwd, p_neg]
        vec = [0.1, 0.2, 0.6, 0.1]
        out = op(vec)
        # Expected: [p_fwd, p_eq, p_bwd, p_neg]
        self.assertAlmostEqual(out[0], 0.6, places=6)
        self.assertAlmostEqual(out[1], 0.2, places=6)
        self.assertAlmostEqual(out[2], 0.1, places=6)
        self.assertAlmostEqual(out[3], 0.1, places=6)

    def test_symmetric_kl_divergence(self):
        """Verify Symmetric KL Divergence is non-negative, symmetric, and zero for identical inputs."""
        p = [0.7, 0.1, 0.15, 0.05]
        q = [0.4, 0.3, 0.2, 0.1]

        # Zero for identical distributions
        self.assertAlmostEqual(symmetric_kl_pure(p, p), 0.0, places=6)

        # Symmetry: D(P, Q) == D(Q, P)
        d_pq = symmetric_kl_pure(p, q)
        d_qp = symmetric_kl_pure(q, p)
        self.assertAlmostEqual(d_pq, d_qp, places=6)
        self.assertGreater(d_pq, 0.0)

        # Module test
        loss_mod = SymmetricKLDivergenceLoss()
        mod_loss = loss_mod(p, q)
        self.assertAlmostEqual(mod_loss, d_pq, places=5)

    def test_jensen_shannon_divergence(self):
        """Verify Jensen-Shannon Divergence is symmetric, bounded in [0, ln 2], and zero for identical inputs."""
        p = [0.7, 0.1, 0.15, 0.05]
        q = [0.4, 0.3, 0.2, 0.1]

        # Zero for identical
        self.assertAlmostEqual(jensen_shannon_pure(p, p), 0.0, places=6)

        # Symmetry: D(P, Q) == D(Q, P)
        js_pq = jensen_shannon_pure(p, q)
        js_qp = jensen_shannon_pure(q, p)
        self.assertAlmostEqual(js_pq, js_qp, places=6)

        # Bounded by ln(2) approx 0.693147
        self.assertGreater(js_pq, 0.0)
        self.assertLessEqual(js_pq, math.log(2.0) + 1e-6)

        # Module test
        js_mod = JensenShannonDivergenceLoss()
        mod_loss = js_mod(p, q)
        self.assertAlmostEqual(mod_loss, js_pq, places=5)

    def test_dynamic_warmup_scheduling(self):
        """Verify linear warmup of consistency loss lambda."""
        crl_loss = ConsistencyRegularizationLoss(lambda_cons=1.5, warmup_epochs=3)
        # epoch 0: 0.0
        self.assertAlmostEqual(crl_loss.get_effective_lambda(0.0), 0.0, places=6)
        # epoch 1.5: 1.5 * (1.5 / 3.0) = 0.75
        self.assertAlmostEqual(crl_loss.get_effective_lambda(1.5), 0.75, places=6)
        # epoch 3: 1.5
        self.assertAlmostEqual(crl_loss.get_effective_lambda(3.0), 1.5, places=6)
        # epoch 5 (post-warmup capped at 1.0): 1.5
        self.assertAlmostEqual(crl_loss.get_effective_lambda(5.0), 1.5, places=6)

    def test_crl_loss_computation_and_diagnostics(self):
        """Verify CRL loss computation returns total loss and diagnostic dictionary."""
        crl_loss = ConsistencyRegularizationLoss(divergence="sym_kl", lambda_cons=1.0, warmup_epochs=1)

        logits_dir = [0.1, 0.2, 2.5, 0.1]  # predicts FWD (2)
        logits_rev = [2.5, 0.2, 0.1, 0.1]  # predicts BWD (0)
        label_dir = 2
        label_rev = 0

        loss, metrics = crl_loss(
            logits_dir=[logits_dir],
            logits_rev=[logits_rev],
            labels_dir=[label_dir],
            labels_rev=[label_rev],
            epoch_or_progress=1.0,
        )

        self.assertGreater(loss, 0.0)
        self.assertIn("loss_total", metrics)
        self.assertIn("loss_task", metrics)
        self.assertIn("loss_ce_dir", metrics)
        self.assertIn("loss_ce_rev", metrics)
        self.assertIn("loss_cons", metrics)
        self.assertIn("effective_lambda", metrics)
        self.assertEqual(metrics["divergence"], "sym_kl")

    def test_siamese_crl_model_forward(self):
        """Verify SiameseCRLModel processing direct and reverse pairs."""
        mock_enc = MockCrossEncoder()
        siamese = SiameseCRLModel(encoder=mock_enc, divergence="sym_kl", lambda_cons=1.0)

        tok = MockTokenizer()
        b_dir = tok(["Pope Francis"], ["the pope"])
        b_rev = tok(["the pope"], ["Pope Francis"])

        output = siamese.forward(
            input_ids_dir=b_dir["input_ids"],
            input_ids_rev=b_rev["input_ids"],
            labels_dir=[2],
            labels_rev=[0],
            epoch=1.0,
            apply_symmetrized_decoding=True,
        )

        self.assertIsInstance(output, SiameseCRLOutput)
        self.assertIsNotNone(output.loss)
        self.assertIsNotNone(output.logits_dir)
        self.assertIsNotNone(output.logits_rev)
        self.assertIsNotNone(output.preds_dir)
        self.assertIsNotNone(output.preds_rev)
        self.assertIsNotNone(output.metrics)

        # SoftCons check on output predictions
        pred_d = output.preds_dir[0] if isinstance(output.preds_dir, list) else int(output.preds_dir[0])
        pred_r = output.preds_rev[0] if isinstance(output.preds_rev, list) else int(output.preds_rev[0])
        self.assertEqual(pred_r, REV_INDEX_MAP[pred_d])

    def test_cross_entropy_pure_ignore_index_single(self):
        """Verify cross_entropy_pure ignores target=-100 and out-of-bounds targets."""
        logits = [0.1, 0.2, 2.5, 0.1]
        loss_valid = cross_entropy_pure(logits, 2)
        self.assertGreater(loss_valid, 0.0)

        # PyTorch ignore_index (-100) returns 0.0
        loss_ignored = cross_entropy_pure(logits, -100)
        self.assertEqual(loss_ignored, 0.0)

        # Negative out-of-bounds target returns 0.0
        self.assertEqual(cross_entropy_pure(logits, -1), 0.0)
        # Out-of-bounds target beyond class count returns 0.0
        self.assertEqual(cross_entropy_pure(logits, 10), 0.0)

    def test_cross_entropy_pure_batch_divided_by_non_ignored(self):
        """Verify batch cross_entropy_pure divides loss ONLY by non-ignored count."""
        l1 = [0.1, 0.2, 2.5, 0.1]  # target 2
        l2 = [0.1, 2.5, 0.2, 0.1]  # target 1
        l3 = [2.5, 0.1, 0.1, 0.1]  # target 0

        loss1 = cross_entropy_pure(l1, 2)
        loss3 = cross_entropy_pure(l3, 0)

        # Batch with 1 masked target (-100) out of 3 samples
        batch_logits = [l1, l2, l3]
        batch_targets = [2, -100, 0]

        batch_loss = cross_entropy_pure(batch_logits, batch_targets)
        # Should divide sum(loss1, loss3) by 2 (non-ignored count), NOT by 3!
        expected_loss = 0.5 * (loss1 + loss3)
        self.assertAlmostEqual(batch_loss, expected_loss, places=6)

        # When all entries in batch are ignored, return 0.0
        all_ignored_loss = cross_entropy_pure(batch_logits, [-100, -100, -100])
        self.assertEqual(all_ignored_loss, 0.0)

    def test_crl_loss_with_ignore_index_handling(self):
        """Verify ConsistencyRegularizationLoss handles target=-100 without KeyError or IndexError."""
        crl_loss = ConsistencyRegularizationLoss(divergence="sym_kl", lambda_cons=1.0)

        ld = [[0.1, 0.2, 2.5, 0.1], [0.1, 2.5, 0.2, 0.1]]
        lr = [[2.5, 0.2, 0.1, 0.1], [0.1, 2.5, 0.2, 0.1]]
        yd = [2, -100]  # Second pair is masked

        loss, metrics = crl_loss(
            logits_dir=ld,
            logits_rev=lr,
            labels_dir=yd,
            labels_rev=None,  # Auto-derives yr=[0, -100] safely
            epoch_or_progress=1.0,
        )

        self.assertGreater(loss, 0.0)
        expected_ce_d = cross_entropy_pure(ld[0], 2)
        self.assertAlmostEqual(metrics["loss_ce_dir"], expected_ce_d, places=5)

    def test_siamese_crl_pure_ablation(self):
        """Verify that SiameseCRLModel with apply_symmetrized_decoding=False disables SD."""
        mock_enc = MockCrossEncoder()
        siamese = SiameseCRLModel(
            encoder=mock_enc,
            divergence="sym_kl",
            apply_symmetrized_decoding=False,
        )
        siamese.eval()

        tok = MockTokenizer()
        b_dir = tok(["Pope Francis"], ["the pope"])
        b_rev = tok(["the pope"], ["Pope Francis"])

        # Forward in eval mode without SD
        output = siamese.forward(
            input_ids_dir=b_dir["input_ids"],
            input_ids_rev=b_rev["input_ids"],
            apply_symmetrized_decoding=False,
        )
        self.assertIsNone(output.sym_probs_dir, "Pure CRL must not compute sym_probs_dir")
        self.assertIsNone(output.sym_probs_rev, "Pure CRL must not compute sym_probs_rev")
        self.assertIsNotNone(output.probs_dir)
        self.assertIsNotNone(output.preds_dir)

    def test_siamese_crl_predict_pair_and_proba(self):
        """Verify predict_pair and predict_proba on SiameseCRLModel."""
        mock_enc = MockCrossEncoder()
        siamese = SiameseCRLModel(encoder=mock_enc)

        # 1. predict_proba
        probs = siamese.predict_proba("the red car", "a vehicle")
        self.assertEqual(len(probs), 4)
        self.assertAlmostEqual(sum(probs), 1.0, places=4)

        # 2. predict_pair with pure CRL
        p_d_raw, p_r_raw = siamese.predict_pair("the red car", "a vehicle", apply_symmetrized_decoding=False)
        self.assertIsInstance(p_d_raw, int)
        self.assertIsInstance(p_r_raw, int)

        # 3. predict_pair with SD enabled
        p_d_sd, p_r_sd = siamese.predict_pair("the red car", "a vehicle", apply_symmetrized_decoding=True)
        self.assertEqual(p_r_sd, REV_INDEX_MAP[p_d_sd])


class TestSymmetrizedDecoding(unittest.TestCase):
    """Unit tests for Symmetrized Decoding post-hoc projection and wrapper."""

    def test_symmetrized_decoding_100_percent_softcons(self):
        """Verify SD produces SoftCons = 100% on arbitrary asymmetrical pairs."""
        decoder = SymmetrizedDecoder(mask_negative_other=True)

        pairs = [
            ([0.05, 0.05, 0.85, 0.05], [0.05, 0.05, 0.85, 0.05]),  # Reversal curse: both predict FWD
            ([0.80, 0.10, 0.05, 0.05], [0.10, 0.10, 0.10, 0.70]),  # Confident vs uncertain
            ([0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]),  # Uniform tie
            ([0.10, 0.80, 0.05, 0.05], [0.10, 0.80, 0.05, 0.05]),  # EQUIVALENCE
        ]

        for p_dir, p_rev in pairs:
            pred_dir, pred_rev = decoder.decode_pair(p_dir, p_rev)
            self.assertEqual(
                pred_rev,
                REV_INDEX_MAP[pred_dir],
                f"SoftCons failed: pred_dir={pred_dir}, pred_rev={pred_rev}",
            )

    def test_symmetrized_model_wrapper(self):
        """Verify SymmetrizedModelWrapper wraps base model and predicts consistently."""
        base_model = MockTransformerModel()
        wrapper = SymmetrizedModelWrapper(base_model=base_model)

        pred_dir, pred_rev = wrapper.predict_pair("Pope Francis", "the pope")
        self.assertEqual(pred_rev, REV_INDEX_MAP[pred_dir])

    def test_symmetrized_model_wrapper_predict_pair_delegation(self):
        """Verify SymmetrizedModelWrapper delegates predict_pair for both sd and crl_sd methods."""
        # 1. method='sd'
        model_sd = build_model("mock", method="sd", mask_negative_other=True)
        p_dir, p_rev = model_sd.predict_pair("the red car", "a vehicle")
        self.assertIsInstance(p_dir, int)
        self.assertIsInstance(p_rev, int)
        self.assertEqual(p_rev, REV_INDEX_MAP[p_dir])

        # 2. method='crl_sd'
        model_crl_sd = build_model("mock", method="crl_sd", mask_negative_other=True)
        p_d, p_r = model_crl_sd.predict_pair("the red car", "a vehicle")
        self.assertIsInstance(p_d, int)
        self.assertIsInstance(p_r, int)
        self.assertEqual(p_r, REV_INDEX_MAP[p_d])

        # 3. Keyword argument aliases
        p_d_named, p_r_named = model_crl_sd.predict_pair(text_a="the red car", text_b="a vehicle")
        self.assertEqual((p_d, p_r), (p_d_named, p_r_named))

        p_d_legacy, p_r_legacy = model_crl_sd.predict_pair(text1="the red car", text2="a vehicle")
        self.assertEqual((p_d, p_r), (p_d_legacy, p_r_legacy))

        # 4. Predict proba & discrete predict
        probs = model_crl_sd.predict_proba("the red car", "a vehicle")
        self.assertEqual(len(probs), 4)
        pred_label = model_crl_sd.predict("the red car", "a vehicle")
        self.assertIn(pred_label, ID_TO_LABEL.values())

    def test_symmetrized_decoder_mask_negative_other_configurability(self):
        """Verify mask_negative_other is configurable via __init__, attribute, and per-call override."""
        p_dir = [0.1, 0.1, 0.2, 0.6]  # Highest is NEGATIVE_OTHER (3)
        p_rev = [0.2, 0.1, 0.1, 0.6]  # Highest is NEGATIVE_OTHER (3)

        # 1. Unmasked decoder predicts index 3
        decoder_unmasked = SymmetrizedDecoder(mask_negative_other=False)
        self.assertFalse(decoder_unmasked.mask_negative_other)
        d_pred, r_pred = decoder_unmasked.decode_pair(p_dir, p_rev)
        self.assertEqual(d_pred, 3)
        self.assertEqual(r_pred, 3)

        # 2. Masked decoder zeroes index 3 and predicts among {0, 1, 2}
        decoder_masked = SymmetrizedDecoder(mask_negative_other=True)
        self.assertTrue(decoder_masked.mask_negative_other)
        d_pred_m, r_pred_m = decoder_masked.decode_pair(p_dir, p_rev)
        self.assertEqual(d_pred_m, 2)  # FORWARD_ENTAILMENT
        self.assertEqual(r_pred_m, 0)  # BACKWARD_ENTAILMENT

        # 3. Dynamic attribute mutation
        decoder_unmasked.mask_negative_other = True
        d_pred_dyn, r_pred_dyn = decoder_unmasked.decode_pair(p_dir, p_rev)
        self.assertEqual(d_pred_dyn, 2)
        self.assertEqual(r_pred_dyn, 0)

        # 4. Method-level override: call with mask_negative_other=False on masked decoder
        d_override, r_override = decoder_masked.decode_pair(p_dir, p_rev, mask_negative_other=False)
        self.assertEqual(d_override, 3)
        self.assertEqual(r_override, 3)

        # 5. Batch method-level override
        batch_d, batch_r = decoder_unmasked.decode_pairs(
            [p_dir, p_dir],
            [p_rev, p_rev],
            mask_negative_other=False,
        )
        self.assertEqual(batch_d, [3, 3])
        self.assertEqual(batch_r, [3, 3])

    def test_symmetrized_decoder_semeval_softcons_guarantee_with_masking(self):
        """Verify that mask_negative_other=True guarantees 100% SoftCons under SemEval metrics."""
        from src.metrics.metrics import compute_official_metrics

        p_dir = [0.15, 0.10, 0.25, 0.50]
        p_rev = [0.25, 0.10, 0.15, 0.50]

        decoder = SymmetrizedDecoder(mask_negative_other=True)
        pred_d, pred_r = decoder.decode_pair(p_dir, p_rev)
        self.assertEqual(pred_d, 2)  # FORWARD_ENTAILMENT
        self.assertEqual(pred_r, 0)  # BACKWARD_ENTAILMENT

        metrics = compute_official_metrics(
            y_true_dir=["FORWARD_ENTAILMENT"],
            y_pred_dir=[ID_TO_LABEL[pred_d]],
            y_true_rev=["BACKWARD_ENTAILMENT"],
            y_pred_rev=[ID_TO_LABEL[pred_r]],
            is_reversible_mask=[True],
        )
        self.assertEqual(metrics["soft_cons"], 1.0)
        self.assertEqual(metrics["hard_cons"], 1.0)


class TestModelFactory(unittest.TestCase):
    """Unit tests for model and loss factory."""

    def test_build_model_methods(self):
        """Verify build_model builds models for baseline, bsa, crl, sd, crl_sd."""
        m_baseline = build_model(model_name_or_path="mock", method="baseline")
        self.assertIsNotNone(m_baseline)

        m_bsa = build_model(model_name_or_path="mock", method="bsa")
        self.assertIsNotNone(m_bsa)

        m_crl = build_model(model_name_or_path="mock", method="crl")
        self.assertIsInstance(m_crl, SiameseCRLModel)

        m_sd = build_model(model_name_or_path="mock", method="sd")
        self.assertIsInstance(m_sd, SymmetrizedModelWrapper)

        m_crl_sd = build_model(model_name_or_path="mock", method="crl_sd")
        self.assertIsInstance(m_crl_sd, SymmetrizedModelWrapper)

    def test_build_loss_methods(self):
        """Verify build_loss returns appropriate loss functions."""
        loss_ce = build_loss(method="baseline")
        self.assertTrue(callable(loss_ce))

        loss_crl = build_loss(method="crl", divergence="sym_kl", lambda_cons=1.2)
        self.assertIsInstance(loss_crl, ConsistencyRegularizationLoss)

        with self.assertRaises(ValueError):
            build_loss(method="unknown_method")

    def test_build_tokenizer(self):
        """Verify build_tokenizer returns MockTokenizer for mock identifier."""
        tok = build_tokenizer("mock")
        self.assertIsInstance(tok, MockTokenizer)

    def test_device_and_dtype_resolution(self):
        """Verify auto-resolution of compute device and dtype."""
        dev = resolve_device()
        self.assertIn(dev, ["cuda", "mps", "cpu"])
        self.assertEqual(resolve_device("cpu"), "cpu")

    def test_crl_sd_factory_predict_pair(self):
        """Verify build_model(method='crl_sd') can invoke predict_pair without error."""
        model = build_model("mock", method="crl_sd")
        p_dir, p_rev = model.predict_pair("Pope Francis", "the pope")
        self.assertEqual(p_rev, REV_INDEX_MAP[p_dir])

    def test_symmetrized_model_wrapper_parameter_and_state_delegation(self):
        """Verify SymmetrizedModelWrapper delegates PyTorch module methods (parameters, state_dict, to)."""
        class MockTrainableModule:
            def __init__(self):
                self.device = "cpu"
                self.training = True
                self.zeroed = False
            def parameters(self, recurse: bool = True):
                return ["param_w", "param_b"]
            def named_parameters(self, prefix: str = "", recurse: bool = True):
                return [("weight", "param_w"), ("bias", "param_b")]
            def state_dict(self, *args, **kwargs):
                return {"weight": [1.0, 2.0], "bias": [0.0]}
            def load_state_dict(self, state_dict, strict: bool = True):
                return "loaded_successfully"
            def to(self, *args, **kwargs):
                self.device = args[0] if args else kwargs.get("device", "cpu")
                return self
            def eval(self):
                self.training = False
                return self
            def train(self, mode: bool = True):
                self.training = mode
                return self
            def zero_grad(self):
                self.zeroed = True

        base = MockTrainableModule()
        wrapped = SymmetrizedModelWrapper(base)

        # 1. Parameter delegation
        self.assertTrue(hasattr(wrapped, "parameters"))
        self.assertEqual(list(wrapped.parameters()), ["param_w", "param_b"])
        self.assertEqual(list(wrapped.parameters(recurse=False)), ["param_w", "param_b"])

        # 2. Named parameter delegation
        self.assertTrue(hasattr(wrapped, "named_parameters"))
        self.assertEqual(list(wrapped.named_parameters()), [("weight", "param_w"), ("bias", "param_b")])

        # 3. State dict delegation
        self.assertTrue(hasattr(wrapped, "state_dict"))
        self.assertEqual(wrapped.state_dict(), {"weight": [1.0, 2.0], "bias": [0.0]})

        # 4. Load state dict delegation
        self.assertTrue(hasattr(wrapped, "load_state_dict"))
        self.assertEqual(wrapped.load_state_dict({"weight": [1.0]}), "loaded_successfully")

        # 5. Device transfer delegation
        self.assertIs(wrapped.to("cuda:0"), wrapped)
        self.assertEqual(base.device, "cuda:0")

        # 6. Mode switching
        self.assertIs(wrapped.eval(), wrapped)
        self.assertFalse(base.training)
        self.assertIs(wrapped.train(True), wrapped)
        self.assertTrue(base.training)

        # 7. Reviewer test contract (zero-argument methods)
        class ReviewerMock:
            def parameters(self): return [1]
            def named_parameters(self): return [('w', 1)]
            def state_dict(self): return {'w': 1}

        r_wrapped = SymmetrizedModelWrapper(ReviewerMock())
        self.assertTrue(hasattr(r_wrapped, 'parameters') and len(list(r_wrapped.parameters())) == 1)
        self.assertTrue(hasattr(r_wrapped, 'named_parameters') and len(list(r_wrapped.named_parameters())) == 1)
        self.assertTrue(hasattr(r_wrapped, 'state_dict') and r_wrapped.state_dict() == {'w': 1})


if __name__ == "__main__":
    unittest.main()

