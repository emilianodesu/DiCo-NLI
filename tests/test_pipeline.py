"""Fast CPU End-to-End Pipeline Smoke Test (<60s).

Validates the complete execution flow on CPU without external network access or downloads:
1. Synthetic dataset generation covering English & Spanish across 4 syntactic categories
   (Nominal, Verbal, Prepositional, Adverbial).
2. Deterministic reversal pairing and BSA augmentation.
3. Fast CPU training step (forward + loss + backward simulation).
4. Equivariant inference pass with Symmetrized Decoding.
5. Export to strict Codabench two-column CSV format.
6. Automated format audit via Codabench validator.
7. Official metric calculation (Weighted F1, SoftCons, HardCons).
8. Strict timing budget assertion (<60 seconds).
"""

import os
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Milestone 1: Live source code imports (Data pipeline and consistency metrics)
try:
    from src.data.codabench import export_codabench_submission, validate_codabench_submission
    from src.data.reversal import build_reversal_dataset, pair_instances, reverse_label
    from src.metrics.metrics import compute_official_metrics
    SRC_AVAILABLE = True
except ImportError:
    SRC_AVAILABLE = False
    from tests.test_codabench import export_codabench_submission, validate_codabench_submission
    from tests.test_metrics import compute_official_metrics
    from tests.test_reversal_logic import build_reversal_dataset, pair_instances, reverse_label

# Milestone 2: Neural modeling & decoding (Decoupled: uses live src.models if available; fallback to test oracle)
try:
    from src.models.decoding import SymmetrizedDecoder
    M2_MODELS_AVAILABLE = True
except ImportError:
    M2_MODELS_AVAILABLE = False
    from tests.test_symmetrized_decoding import SymmetrizedDecoder


def generate_synthetic_phrasis_benchmark() -> List[Dict[str, Any]]:
    """Generate 32 synthetic phrase pairs modeled on PhrasIS (EN & ES, 4 categories)."""
    samples = [
        # Track 1 (English) - Nominal (NP)
        ("syn_en_01", "en", "en", "Pope Francis", "the pope", "FORWARD_ENTAILMENT", "nominal"),
        ("syn_en_02", "en", "en", "a red sports car", "a vehicle", "FORWARD_ENTAILMENT", "nominal"),
        ("syn_en_03", "en", "en", "the CEO of Apple", "the Apple chief executive", "EQUIVALENCE", "nominal"),
        ("syn_en_04", "en", "en", "a golden retriever", "a microwave oven", "NEGATIVE_OTHER", "nominal"),
        # Track 1 (English) - Verbal (VP)
        ("syn_en_05", "en", "en", "is sprinting rapidly", "is running", "FORWARD_ENTAILMENT", "verbal"),
        ("syn_en_06", "en", "en", "is playing guitar", "is making music", "FORWARD_ENTAILMENT", "verbal"),
        ("syn_en_07", "en", "en", "shouting loudly", "yelling loudly", "EQUIVALENCE", "verbal"),
        ("syn_en_08", "en", "en", "cooking dinner", "sleeping in bed", "NEGATIVE_OTHER", "verbal"),
        # Track 1 (English) - Prepositional (PP)
        ("syn_en_09", "en", "en", "in downtown Madrid", "in Spain", "FORWARD_ENTAILMENT", "prepositional"),
        ("syn_en_10", "en", "en", "across the wide river", "across the water", "FORWARD_ENTAILMENT", "prepositional"),
        ("syn_en_11", "en", "en", "under the wooden table", "beneath the wooden table", "EQUIVALENCE", "prepositional"),
        ("syn_en_12", "en", "en", "inside the refrigerator", "on top of the roof", "NEGATIVE_OTHER", "prepositional"),
        # Track 1 (English) - Adverbial (ADVP)
        ("syn_en_13", "en", "en", "extremely quickly", "very fast", "EQUIVALENCE", "adverbial"),
        ("syn_en_14", "en", "en", "completely destroyed", "damaged", "FORWARD_ENTAILMENT", "adverbial"),
        ("syn_en_15", "en", "en", "very carefully", "with great caution", "EQUIVALENCE", "adverbial"),
        ("syn_en_16", "en", "en", "almost always", "never at all", "NEGATIVE_OTHER", "adverbial"),
        # Track 2 (Spanish) - Nominal (NP)
        ("syn_es_01", "es", "es", "el papa Francisco", "el papa", "FORWARD_ENTAILMENT", "nominal"),
        ("syn_es_02", "es", "es", "un automóvil rojo", "un vehículo", "FORWARD_ENTAILMENT", "nominal"),
        ("syn_es_03", "es", "es", "el director ejecutivo", "el director general", "EQUIVALENCE", "nominal"),
        ("syn_es_04", "es", "es", "un perro labrador", "un horno de microondas", "NEGATIVE_OTHER", "nominal"),
        # Track 2 (Spanish) - Verbal (VP)
        ("syn_es_05", "es", "es", "está corriendo rápidamente", "se está moviendo", "FORWARD_ENTAILMENT", "verbal"),
        ("syn_es_06", "es", "es", "toca la guitarra", "hace música", "FORWARD_ENTAILMENT", "verbal"),
        ("syn_es_07", "es", "es", "gritando fuertemente", "chillando con fuerza", "EQUIVALENCE", "verbal"),
        ("syn_es_08", "es", "es", "cocinando la cena", "durmiendo plácidamente", "NEGATIVE_OTHER", "verbal"),
        # Track 2 (Spanish) - Prepositional (PP)
        ("syn_es_09", "es", "es", "en el centro de Madrid", "en España", "FORWARD_ENTAILMENT", "prepositional"),
        ("syn_es_10", "es", "es", "a través del río caudaloso", "por el agua", "FORWARD_ENTAILMENT", "prepositional"),
        ("syn_es_11", "es", "es", "debajo de la mesa de madera", "bajo la mesa de madera", "EQUIVALENCE", "prepositional"),
        ("syn_es_12", "es", "es", "dentro de la cocina", "en la cima de la montaña", "NEGATIVE_OTHER", "prepositional"),
        # Track 2 (Spanish) - Adverbial (ADVP)
        ("syn_es_13", "es", "es", "con extrema rapidez", "muy velozmente", "EQUIVALENCE", "adverbial"),
        ("syn_es_14", "es", "es", "totalmente destruido", "dañado", "FORWARD_ENTAILMENT", "adverbial"),
        ("syn_es_15", "es", "es", "con mucho cuidado", "con suma cautela", "EQUIVALENCE", "adverbial"),
        ("syn_es_16", "es", "es", "casi siempre", "en ningún momento", "NEGATIVE_OTHER", "adverbial"),
    ]

    records = []
    for pair_id, l1, l2, t1, t2, label, cat in samples:
        dir_id = f"{pair_id}__{l1}-{l2}__original"
        rev_id = f"{pair_id}__{l1}-{l2}__flipped" if label != "NEGATIVE_OTHER" else ""

        records.append({
            "instance_id": dir_id,
            "pair_id": pair_id,
            "text1_lang": l1,
            "text2_lang": l2,
            "text1": t1,
            "text2": t2,
            "label": label,
            "category": cat,
            "reverse_pair_id": rev_id,
        })

        if label != "NEGATIVE_OTHER":
            records.append({
                "instance_id": rev_id,
                "pair_id": pair_id,
                "text1_lang": l2,
                "text2_lang": l1,
                "text1": t2,
                "text2": t1,
                "label": reverse_label(label),
                "category": cat,
                "reverse_pair_id": dir_id,
            })

    return records


class FastCPUMockModel:
    """Deterministic, lightweight CPU model producing calibrated logits in <5ms."""

    def __init__(self, seed: int = 42) -> None:
        self.weights = [0.1, 0.2, 0.3, 0.4]

    def train_step(self, records: List[Dict[str, Any]]) -> float:
        """Simulate 1 epoch of backpropagation and loss reduction."""
        loss = 0.0
        for r in records:
            # Deterministic pseudo loss based on token counts
            loss += 0.01 * (len(r["text1"]) + len(r["text2"]))
        return loss / len(records)

    def predict_logits(self, text1: str, text2: str) -> List[float]:
        """Generate deterministic 4-class logits [BWD, EQUIV, FWD, NEG]."""
        # Character-level heuristic for fast deterministic logits
        l1, l2 = len(text1), len(text2)
        diff = l1 - l2
        if diff > 3:
            # Longer text1 -> more specific -> entails shorter text2 (FWD = index 2)
            logits = [0.1, 0.2, 2.5, 0.2]
        elif diff < -3:
            # Shorter text1 -> entailed by text2 (BWD = index 0)
            logits = [2.5, 0.2, 0.1, 0.2]
        else:
            # Similar length -> EQUIV = index 1
            logits = [0.2, 2.2, 0.2, 0.3]
        return logits

    @staticmethod
    def softmax(logits: Sequence[float]) -> List[float]:
        max_l = max(logits)
        exp_vals = [pow(2.718281828459045, x - max_l) for x in logits]
        s = sum(exp_vals)
        return [v / s for v in exp_vals]


class TestPipeline(unittest.TestCase):
    """End-to-End smoke test executing on CPU in under 60 seconds."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_m1_source_available(self):
        """Verify that M1 source modules are imported directly from src/ without fallback."""
        self.assertTrue(
            SRC_AVAILABLE,
            "M1 source modules (src.data, src.metrics) must be imported live from src/."
        )

    def test_m2_models_available(self):
        """Verify that M2 modeling modules (src.models) are imported directly from src/ without fallback."""
        self.assertTrue(
            M2_MODELS_AVAILABLE,
            "M2 modeling modules (src.models.decoding) must be imported live from src/."
        )

    def test_fast_cpu_smoke_pipeline(self):
        """Execute complete end-to-end training, inference, export, validation, and scoring in <60s."""
        start_time = time.time()

        # Step 1: Generate synthetic corpus
        corpus = generate_synthetic_phrasis_benchmark()
        self.assertGreaterEqual(len(corpus), 48, "Synthetic benchmark should contain at least 48 instances")

        # Step 2: Build reversible pairing and BSA dataset
        paired = pair_instances(corpus)
        self.assertGreater(len(paired), 0, "Must identify canonical reciprocal pairs")
        bsa_corpus = build_reversal_dataset(corpus, augment_bsa=False)

        # Step 2b: Verify DataFrame ingestion via load_phrasis_split (Finding F-02 regression guard)
        from src.data.loader import load_phrasis_split
        df_dev = load_phrasis_split("track1", "dev")
        df_unaugmented = build_reversal_dataset(df_dev, augment_bsa=False)
        self.assertEqual(len(df_unaugmented), len(df_dev), "Unaugmented DataFrame must preserve row count")
        df_augmented = build_reversal_dataset(df_dev, augment_bsa=True)
        self.assertGreater(len(df_augmented), len(df_dev), "BSA augmented DataFrame must add synthetic pairs")

        # Step 3: Fast CPU Training Simulation
        model = FastCPUMockModel(seed=42)
        train_loss = model.train_step(bsa_corpus)
        self.assertIsInstance(train_loss, float)
        self.assertGreater(train_loss, 0.0)

        # Step 4: Symmetrized Decoding Inference Pass
        decoder = SymmetrizedDecoder(mask_negative_other=True)
        id_to_pred = {}
        index_to_label = {0: "BACKWARD_ENTAILMENT", 1: "EQUIVALENCE", 2: "FORWARD_ENTAILMENT", 3: "NEGATIVE_OTHER"}

        y_true_dir = []
        y_pred_dir = []
        y_true_rev = []
        y_pred_rev = []
        is_rev_mask = []

        for fwd, rev in paired:
            p_dir = model.softmax(model.predict_logits(fwd["text1"], fwd["text2"]))
            p_rev = model.softmax(model.predict_logits(rev["text1"], rev["text2"]))

            idx_dir, idx_rev = decoder.decode_pair(p_dir, p_rev)
            label_dir = index_to_label[idx_dir]
            label_rev = index_to_label[idx_rev]

            id_to_pred[fwd["instance_id"]] = label_dir
            id_to_pred[rev["instance_id"]] = label_rev

            y_true_dir.append(fwd["label"])
            y_pred_dir.append(label_dir)
            y_true_rev.append(rev["label"])
            y_pred_rev.append(label_rev)
            is_rev_mask.append(True)

        # Add unpaired negative instances
        for r in corpus:
            if r["instance_id"] not in id_to_pred:
                logits = model.predict_logits(r["text1"], r["text2"])
                pred_idx = logits.index(max(logits))
                id_to_pred[r["instance_id"]] = index_to_label[pred_idx]

        # Step 5: Export Predictions to Codabench CSV
        all_ids = [r["instance_id"] for r in corpus]
        all_preds = [id_to_pred[iid] for iid in all_ids]
        sub_csv_path = str(self.temp_path / "smoke_submission.csv")

        export_codabench_submission(all_ids, all_preds, sub_csv_path)
        self.assertTrue(os.path.isfile(sub_csv_path))

        # Step 6: Validate Exported Submission File with Codabench Auditor
        is_valid = validate_codabench_submission(sub_csv_path, expected_ids=set(all_ids))
        self.assertTrue(is_valid, "Exported Codabench submission file must pass official validation")

        # Step 7: Compute Official Evaluation Metrics
        metrics = compute_official_metrics(
            y_true_dir=y_true_dir,
            y_pred_dir=y_pred_dir,
            y_true_rev=y_true_rev,
            y_pred_rev=y_pred_rev,
            is_reversible_mask=is_rev_mask,
        )

        self.assertIn("weighted_f1", metrics)
        self.assertIn("soft_cons", metrics)
        self.assertIn("hard_cons", metrics)

        # Verify mathematical invariant under Symmetrized Decoding
        self.assertEqual(
            metrics["soft_cons"],
            1.0,
            "Symmetrized Decoding must guarantee SoftCons = 1.0 on all reversible pairs",
        )
        self.assertLessEqual(metrics["hard_cons"], metrics["soft_cons"])
        self.assertGreater(metrics["weighted_f1"], 0.0)

        # Step 8: Execution Time Budget Verification (<60 seconds)
        elapsed = time.time() - start_time
        self.assertLess(
            elapsed,
            60.0,
            f"Pipeline smoke test exceeded 60-second budget: {elapsed:.2f}s",
        )


if __name__ == "__main__":
    unittest.main()
