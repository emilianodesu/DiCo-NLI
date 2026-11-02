"""
Unit tests for proposed_cross_encoder.py in explorer_m2_1.

Verifies all functional contracts, backbone mappings, mock architectures,
tokenization delimiter conventions, and output containers.
"""

import math
import os
import shutil
import tempfile
import unittest

from proposed_cross_encoder import (
    CrossEncoderNLI,
    CrossEncoderOutput,
    ID_TO_LABEL,
    LABEL_TO_ID,
    LabelIndex,
    MockCrossEncoder,
    MockTokenizer,
    NUM_CLASSES,
    SUPPORTED_BACKBONES,
    canonicalize_model_name,
    detect_backbone_family,
)


class TestCrossEncoderBlueprint(unittest.TestCase):
    """Test suite verifying the CrossEncoder implementation blueprint."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_label_encoding_and_constants(self):
        """Verify 4-class label indices and bidirectional mappings."""
        self.assertEqual(NUM_CLASSES, 4)
        self.assertEqual(LABEL_TO_ID["BACKWARD_ENTAILMENT"], 0)
        self.assertEqual(LABEL_TO_ID["EQUIVALENCE"], 1)
        self.assertEqual(LABEL_TO_ID["FORWARD_ENTAILMENT"], 2)
        self.assertEqual(LABEL_TO_ID["NEGATIVE_OTHER"], 3)

        self.assertEqual(ID_TO_LABEL[0], "BACKWARD_ENTAILMENT")
        self.assertEqual(ID_TO_LABEL[1], "EQUIVALENCE")
        self.assertEqual(ID_TO_LABEL[2], "FORWARD_ENTAILMENT")
        self.assertEqual(ID_TO_LABEL[3], "NEGATIVE_OTHER")

        self.assertEqual(LabelIndex.BACKWARD_ENTAILMENT, 0)
        self.assertEqual(LabelIndex.EQUIVALENCE, 1)
        self.assertEqual(LabelIndex.FORWARD_ENTAILMENT, 2)
        self.assertEqual(LabelIndex.NEGATIVE_OTHER, 3)

    def test_backbone_canonicalization_and_detection(self):
        """Verify resolution of aliases and backbone family detection."""
        self.assertEqual(
            canonicalize_model_name("deberta-v3-base"),
            "microsoft/deberta-v3-base",
        )
        self.assertEqual(
            canonicalize_model_name("deberta-v3-small"),
            "microsoft/deberta-v3-small",
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

    def test_mock_tokenizer_delimiter_formatting(self):
        """Verify tokenizer delimiter formatting across DeBERTa, BETO, and RoBERTa."""
        # 1. BERT / DeBERTa format: [CLS] text1 [SEP] text2 [SEP]
        tok_bert = MockTokenizer(model_name="beto")
        encoded_bert = tok_bert("el papa", "Francisco", padding=False)
        self.assertIn("input_ids", encoded_bert)
        self.assertIn("token_type_ids", encoded_bert)
        self.assertIn("attention_mask", encoded_bert)

        ids_bert = encoded_bert["input_ids"][0]
        types_bert = encoded_bert["token_type_ids"][0]
        # Must start with [CLS] (id 1) and end with [SEP] (id 2)
        self.assertEqual(ids_bert[0], 1)
        self.assertEqual(ids_bert[-1], 2)
        self.assertIn(2, ids_bert[1:-1])  # middle [SEP]
        # token_type_ids must distinguish segment 0 and segment 1
        self.assertIn(0, types_bert)
        self.assertIn(1, types_bert)

        # 2. RoBERTa format: <s> text1 </s></s> text2 </s> (no token_type_ids)
        tok_roberta = MockTokenizer(model_name="roberta-bne")
        encoded_roberta = tok_roberta("el papa", "Francisco", padding=False)
        self.assertIn("input_ids", encoded_roberta)
        self.assertIn("attention_mask", encoded_roberta)
        self.assertNotIn("token_type_ids", encoded_roberta)

        ids_roberta = encoded_roberta["input_ids"][0]
        # Must start with <s> (id 0) and end with </s> (id 2)
        self.assertEqual(ids_roberta[0], 0)
        self.assertEqual(ids_roberta[-1], 2)
        # Check double </s></s> delimiter in the middle
        sep_indices = [i for i, x in enumerate(ids_roberta) if x == 2]
        self.assertGreaterEqual(len(sep_indices), 3)

    def test_mock_cross_encoder_heuristics(self):
        """Verify deterministic logit calibration and directional heuristics."""
        encoder = MockCrossEncoder()

        # 1. Equivalence: near-identical strings with similar length
        logits_eq = encoder.predict_logits("extremely quickly", "very quickly")
        self.assertEqual(logits_eq.index(max(logits_eq)), 1)
        self.assertEqual(encoder.predict("extremely quickly", "very quickly"), "EQUIVALENCE")

        # 2. Forward Entailment: specific long premise -> short general hypothesis
        logits_fwd = encoder.predict_logits("a red high performance sports car", "a vehicle")
        self.assertEqual(logits_fwd.index(max(logits_fwd)), 2)
        self.assertEqual(encoder.predict("a red high performance sports car", "a vehicle"), "FORWARD_ENTAILMENT")

        # 3. Backward Entailment: short premise <- long specific hypothesis
        logits_bwd = encoder.predict_logits("a vehicle", "a red high performance sports car")
        self.assertEqual(logits_bwd.index(max(logits_bwd)), 0)
        self.assertEqual(encoder.predict("a vehicle", "a red high performance sports car"), "BACKWARD_ENTAILMENT")

        # 4. Negative Other: zero lexical overlap, divergent concepts
        logits_neg = encoder.predict_logits("golden retriever puppy", "microwave oven appliance")
        self.assertEqual(logits_neg.index(max(logits_neg)), 3)
        self.assertEqual(encoder.predict("golden retriever puppy", "microwave oven appliance"), "NEGATIVE_OTHER")

    def test_mock_softmax_properties(self):
        """Verify softmax output forms a valid probability distribution."""
        encoder = MockCrossEncoder()
        probs = encoder.predict_proba("sample premise", "sample hypothesis")
        self.assertEqual(len(probs), 4)
        self.assertAlmostEqual(sum(probs), 1.0, places=6)
        for p in probs:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_forward_pass_with_loss(self):
        """Verify forward pass output structure and cross entropy loss computation."""
        encoder = MockCrossEncoder()
        tokenizer = MockTokenizer()

        batch_inputs = tokenizer(
            ["the pope", "a sports car"],
            ["Pope Francis", "a vehicle"],
            padding=True,
        )

        labels = [0, 2]  # BACKWARD, FORWARD
        output = encoder.forward(
            input_ids=batch_inputs["input_ids"],
            attention_mask=batch_inputs["attention_mask"],
            labels=labels,
        )

        self.assertIsInstance(output, CrossEncoderOutput)
        self.assertIsNotNone(output.loss)
        self.assertGreater(output.loss, 0.0)
        self.assertEqual(len(output.logits), 2)
        self.assertEqual(len(output.logits[0]), 4)
        self.assertEqual(len(output.probabilities), 2)
        self.assertEqual(len(output.predictions), 2)

    def test_cross_encoder_output_container_interfaces(self):
        """Verify CrossEncoderOutput duck-typing, dictionary access, and tuple unpacking."""
        out = CrossEncoderOutput(
            loss=0.45,
            logits=[[0.1, 0.2, 0.3, 0.4]],
            probabilities=[[0.2, 0.2, 0.4, 0.2]],
            predictions=[2],
        )

        # Attribute access
        self.assertEqual(out.loss, 0.45)
        self.assertEqual(out.predictions, [2])

        # Dictionary access
        self.assertEqual(out["loss"], 0.45)
        self.assertEqual(out["predictions"], [2])

        # Tuple indexing
        self.assertEqual(out[0], 0.45)
        self.assertEqual(out[3], [2])

        # Unpacking
        loss, logits, probs, preds = out
        self.assertEqual(loss, 0.45)
        self.assertEqual(preds, [2])

        # to_dict
        d = out.to_dict()
        self.assertIn("logits", d)
        self.assertIn("probabilities", d)

    def test_cross_encoder_nli_mock_wrapper(self):
        """Verify CrossEncoderNLI wrapper instantiates in mock mode with complete API parity."""
        model = CrossEncoderNLI("deberta-v3-base", is_mock=True)
        self.assertTrue(model.is_mock)
        self.assertEqual(model.family, "deberta")
        self.assertTrue(model.supports_token_type_ids)

        tok = model.get_tokenizer()
        self.assertIsInstance(tok, MockTokenizer)

        pred_label = model.predict("el papa", "el sumo pontífice")
        self.assertIn(pred_label, ID_TO_LABEL.values())

        # Test save and load
        save_path = os.path.join(self.temp_dir, "test_mock_save")
        model.save_pretrained(save_path)
        self.assertTrue(os.path.isfile(os.path.join(save_path, "config.json")))

        loaded_model = CrossEncoderNLI.from_pretrained(save_path)
        self.assertTrue(loaded_model.is_mock)

    def test_roberta_token_type_id_suppression(self):
        """Verify RoBERTa backbone disables token_type_ids."""
        model = CrossEncoderNLI("roberta-bne", is_mock=True)
        self.assertFalse(model.supports_token_type_ids)
        tok = model.get_tokenizer()
        enc = tok("primera frase", "segunda frase")
        self.assertNotIn("token_type_ids", enc)


if __name__ == "__main__":
    unittest.main()
