"""
src/models/cross_encoder.py

Baseline Cross-Encoder Architecture for Directional & Consistent Natural Language Inference (DiCo-NLI).
Supports:
- Track 1 (English): microsoft/deberta-v3-base
- Track 2 (Spanish): dccuchile/bert-base-spanish-wwm-cased (BETO), PlanTL-GOB-ES/roberta-base-bne
- Deterministic CPU Mock Engine (MockCrossEncoder & MockTokenizer) for fast unit testing and offline execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
import re
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    F = None
    HAS_TORCH = False

try:
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    AutoConfig = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    HAS_TRANSFORMERS = False

from src.data.reversal import (
    ID_TO_LABEL,
    LABEL_TO_ID,
)

BACKWARD_ENTAILMENT = "BACKWARD_ENTAILMENT"
EQUIVALENCE = "EQUIVALENCE"
FORWARD_ENTAILMENT = "FORWARD_ENTAILMENT"
NEGATIVE_OTHER = "NEGATIVE_OTHER"
VALID_LABELS = {BACKWARD_ENTAILMENT, EQUIVALENCE, FORWARD_ENTAILMENT, NEGATIVE_OTHER}
REV_MAP = {
    BACKWARD_ENTAILMENT: FORWARD_ENTAILMENT,
    EQUIVALENCE: EQUIVALENCE,
    FORWARD_ENTAILMENT: BACKWARD_ENTAILMENT,
    NEGATIVE_OTHER: NEGATIVE_OTHER,
}


# ==============================================================================
# 1. Canonical 4-Way Label Index Specification
# ==============================================================================

class LabelIndex(IntEnum):
    """
    Canonical 4-class integer label mapping for DiCo-NLI:
      0 -> BACKWARD_ENTAILMENT
      1 -> EQUIVALENCE
      2 -> FORWARD_ENTAILMENT
      3 -> NEGATIVE_OTHER
    """
    BACKWARD_ENTAILMENT = 0
    EQUIVALENCE = 1
    FORWARD_ENTAILMENT = 2
    NEGATIVE_OTHER = 3


# ==============================================================================
# 2. Backbone Metadata & Delimiter Configurations
# ==============================================================================

SUPPORTED_BACKBONES: Dict[str, str] = {
    # Track 1: English
    "deberta-v3-base": "microsoft/deberta-v3-base",
    "microsoft/deberta-v3-base": "microsoft/deberta-v3-base",
    "deberta-v3-large": "microsoft/deberta-v3-large",
    "microsoft/deberta-v3-large": "microsoft/deberta-v3-large",
    # Track 2: Spanish
    "beto": "dccuchile/bert-base-spanish-wwm-cased",
    "bert-base-spanish-wwm-cased": "dccuchile/bert-base-spanish-wwm-cased",
    "dccuchile/bert-base-spanish-wwm-cased": "dccuchile/bert-base-spanish-wwm-cased",
    "roberta-bne": "PlanTL-GOB-ES/roberta-base-bne",
    "roberta-base-bne": "PlanTL-GOB-ES/roberta-base-bne",
    "PlanTL-GOB-ES/roberta-base-bne": "PlanTL-GOB-ES/roberta-base-bne",
    # Testing / Mock aliases
    "mock": "mock-cross-encoder",
    "mock-cross-encoder": "mock-cross-encoder",
}


def canonicalize_model_name(model_name_or_path: str) -> str:
    """Normalizes shorthand model aliases to their canonical Hugging Face hub identifiers."""
    clean = model_name_or_path.strip()
    return SUPPORTED_BACKBONES.get(clean, clean)


def detect_backbone_family(model_name_or_path: str) -> str:
    """
    Identifies the architectural family ('deberta', 'roberta', 'bert', or 'generic')
    to configure special delimiters and token_type_ids handling.
    """
    lower_name = canonicalize_model_name(model_name_or_path).lower()
    if "deberta" in lower_name:
        return "deberta"
    if "roberta" in lower_name or "bne" in lower_name:
        return "roberta"
    if "bert" in lower_name or "beto" in lower_name:
        return "bert"
    return "generic"


# ==============================================================================
# 3. Unified Cross-Encoder Output Container
# ==============================================================================

@dataclass
class CrossEncoderOutput:
    """
    Unified output container for CrossEncoderNLI and MockCrossEncoder.
    Compatible with Hugging Face SequenceClassifierOutput and standard tuple/dict access.
    """
    loss: Optional[Any] = None
    logits: Any = None
    probabilities: Optional[Any] = None
    predictions: Optional[Any] = None
    hidden_states: Optional[Any] = None
    attentions: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loss": self.loss,
            "logits": self.logits,
            "probabilities": self.probabilities,
            "predictions": self.predictions,
            "hidden_states": self.hidden_states,
            "attentions": self.attentions,
        }

    def __getitem__(self, key: Union[str, int]) -> Any:
        if isinstance(key, int):
            ordered = (self.loss, self.logits, self.probabilities, self.predictions)
            return ordered[key]
        return getattr(self, key)

    def __iter__(self) -> Iterator[Any]:
        yield self.loss
        yield self.logits
        yield self.probabilities
        yield self.predictions


# ==============================================================================
# 4. Pure-Python / Torch Math Helpers
# ==============================================================================

def _softmax_1d(logits: Sequence[float]) -> List[float]:
    if not logits:
        return []
    max_val = max(logits)
    exps = [math.exp(x - max_val) for x in logits]
    total = sum(exps)
    if total <= 0.0:
        return [1.0 / len(logits)] * len(logits)
    return [e / total for e in exps]


def _softmax_2d(batch_logits: Sequence[Sequence[float]]) -> List[List[float]]:
    return [_softmax_1d(row) for row in batch_logits]


def _cross_entropy_loss_2d(
    batch_logits: Sequence[Sequence[float]],
    labels: Sequence[int],
) -> float:
    if not batch_logits or not labels:
        return 0.0
    total_loss = 0.0
    valid_count = 0
    for row, target in zip(batch_logits, labels):
        idx = int(target)
        if idx == -100 or idx < 0 or idx >= len(row):
            continue
        probs = _softmax_1d(row)
        p_target = max(probs[idx], 1e-12)
        total_loss -= math.log(p_target)
        valid_count += 1
    if valid_count == 0:
        return 0.0
    return total_loss / valid_count


# ==============================================================================
# 5. Deterministic Mock Tokenizer & Cross-Encoder (Offline / CPU Testing)
# ==============================================================================

class MockTokenizer:
    """
    Lightweight, deterministic tokenizer that emulates Hugging Face AutoTokenizer
    for DeBERTa-v3, BETO (BERT), and RoBERTa-BNE without requiring network or model weights.

    Delimiter conventions:
    - BERT / DeBERTa: [CLS] text1 [SEP] text2 [SEP]  (with token_type_ids 0 and 1)
    - RoBERTa:        <s> text1 </s> </s> text2 </s> (no token_type_ids)
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        max_length: int = 128,
    ) -> None:
        self.name_or_path = canonicalize_model_name(model_name)
        self.family = detect_backbone_family(self.name_or_path)
        self.max_length = max_length

        if self.family == "roberta":
            self.cls_token = "<s>"
            self.sep_token = "</s>"
            self.pad_token = "<pad>"
            self.cls_token_id = 0
            self.pad_token_id = 1
            self.sep_token_id = 2
        else:
            self.cls_token = "[CLS]"
            self.sep_token = "[SEP]"
            self.pad_token = "[PAD]"
            self.cls_token_id = 101
            self.sep_token_id = 102
            self.pad_token_id = 0

    def _encode_words(self, text: str) -> List[int]:
        words = re.findall(r"\w+", (text or "").lower())
        return [(sum(ord(c) * (i + 1) for i, c in enumerate(w)) % 29000) + 1000 for w in words]

    def format_pair_string(self, text1: str, text2: str) -> str:
        """Returns the exact canonical delimiter string for inspection and testing."""
        if self.family == "roberta":
            return f"{self.cls_token} {text1} {self.sep_token} {self.sep_token} {text2} {self.sep_token}"
        return f"{self.cls_token} {text1} {self.sep_token} {text2} {self.sep_token}"

    def __call__(
        self,
        text: Union[str, List[str]],
        text_pair: Optional[Union[str, List[str]]] = None,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        max_len = max_length or self.max_length
        is_single = isinstance(text, str)
        texts_a = [text] if is_single else list(text)
        if text_pair is None:
            texts_b = [""] * len(texts_a)
        elif isinstance(text_pair, str):
            texts_b = [text_pair]
        else:
            texts_b = list(text_pair)

        batch_input_ids: List[List[int]] = []
        batch_attention_mask: List[List[int]] = []
        batch_token_type_ids: List[List[int]] = []

        for t1, t2 in zip(texts_a, texts_b):
            ids1 = self._encode_words(t1)
            ids2 = self._encode_words(t2)

            if self.family == "roberta":
                # <s> text1 </s> </s> text2 </s>
                budget = max(0, max_len - 4) if truncation else len(ids1) + len(ids2)
                if truncation and len(ids1) + len(ids2) > budget:
                    half = budget // 2
                    ids1 = ids1[:half]
                    ids2 = ids2[: budget - len(ids1)]
                seq = (
                    [self.cls_token_id]
                    + ids1
                    + [self.sep_token_id, self.sep_token_id]
                    + ids2
                    + [self.sep_token_id]
                )
                type_ids = [0] * len(seq)
            else:
                # [CLS] text1 [SEP] text2 [SEP]
                budget = max(0, max_len - 3) if truncation else len(ids1) + len(ids2)
                if truncation and len(ids1) + len(ids2) > budget:
                    half = budget // 2
                    ids1 = ids1[:half]
                    ids2 = ids2[: budget - len(ids1)]
                seg0 = [self.cls_token_id] + ids1 + [self.sep_token_id]
                seg1 = ids2 + [self.sep_token_id]
                seq = seg0 + seg1
                type_ids = [0] * len(seg0) + [1] * len(seg1)

            mask = [1] * len(seq)
            batch_input_ids.append(seq)
            batch_attention_mask.append(mask)
            batch_token_type_ids.append(type_ids)

        # Pad to uniform length in batch (or max_len if padding='max_length')
        if padding:
            target_len = max_len if padding == "max_length" else max(len(s) for s in batch_input_ids)
            for i in range(len(batch_input_ids)):
                pad_len = max(0, target_len - len(batch_input_ids[i]))
                batch_input_ids[i] = batch_input_ids[i] + [self.pad_token_id] * pad_len
                batch_attention_mask[i] = batch_attention_mask[i] + [0] * pad_len
                batch_token_type_ids[i] = batch_token_type_ids[i] + [0] * pad_len

        result: Dict[str, Any] = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
        }
        if self.family != "roberta":
            result["token_type_ids"] = batch_token_type_ids

        if return_tensors == "pt" and HAS_TORCH:
            result = {k: torch.tensor(v, dtype=torch.long) for k, v in result.items()}

        return result


class MockCrossEncoder:
    """
    Deterministic Mock Cross-Encoder for fast CPU unit tests, baseline dry-runs,
    and pipeline verification. Supports both token-ID tensor/list inputs (`forward`)
    and raw text pairs (`predict_logits`, `predict_proba`, `predict`).
    """

    def __init__(
        self,
        model_name: str = "mock-cross-encoder",
        num_labels: int = 4,
        seed: int = 42,
    ) -> None:
        self.model_name = canonicalize_model_name(model_name)
        self.family = detect_backbone_family(self.model_name)
        self.num_labels = num_labels
        self.seed = seed
        self.is_mock = True
        self.training = False
        self.id2label = dict(ID_TO_LABEL)
        self.label2id = dict(LABEL_TO_ID)
        # Learnable-like state updated if .train_step() is called
        self.pair_memory: Dict[Tuple[str, str], str] = {}
        self.step_count: int = 0

    def train(self, mode: bool = True) -> "MockCrossEncoder":
        """Sets the mock engine into training or evaluation mode (mirrors nn.Module.train)."""
        self.training = bool(mode)
        return self

    def eval(self) -> "MockCrossEncoder":
        """Sets the mock engine into evaluation mode (mirrors nn.Module.eval)."""
        return self.train(False)

    def to(self, *args: Any, **kwargs: Any) -> "MockCrossEncoder":
        """No-op device transfer method for API parity with nn.Module.to()."""
        return self

    @staticmethod
    def _normalize(text: str) -> List[str]:
        return re.findall(r"\w+", (text or "").lower())

    def score_text_pair(self, text1: str, text2: str) -> List[float]:
        """
        Computes deterministic 4-class logits [BWD, EQ, FWD, NEG] for a (text1, text2) pair
        using calibrated lexical & structural overlap heuristics.
        """
        key = ((text1 or "").strip(), (text2 or "").strip())
        if key in self.pair_memory:
            target_label = self.pair_memory[key]
            target_idx = LABEL_TO_ID[target_label]
            logits = [-1.5, -1.5, -1.5, -1.5]
            logits[target_idx] = 3.5
            return logits

        tokens1 = self._normalize(text1)
        tokens2 = self._normalize(text2)
        if not tokens1 and not tokens2:
            return [-1.0, 2.5, -1.0, -1.0]
        if not tokens1 or not tokens2:
            return [-1.0, -1.0, -1.0, 2.5]

        set1, set2 = set(tokens1), set(tokens2)
        if tokens1 == tokens2 or set1 == set2:
            # High equivalence score
            return [-1.2, 3.2, -1.2, -1.5]

        inter = len(set1 & set2)
        union = len(set1 | set2)
        jaccard = inter / max(1, union)

        # Negation detection
        neg_words = {"no", "not", "never", "nunca", "ningun", "ninguno", "sin", "false", "falso"}
        has_neg1 = bool(set1 & neg_words)
        has_neg2 = bool(set2 & neg_words)

        if has_neg1 != has_neg2 or jaccard == 0.0:
            return [-1.2, -1.5, -1.2, 2.8]

        # Directional specificity: more specific text (longer/richer) entails more general text
        len1, len2 = len(tokens1), len(tokens2)
        cov1_in_2 = inter / max(1, len(set2))  # how much of text2 is covered by text1
        cov2_in_1 = inter / max(1, len(set1))  # how much of text1 is covered by text2

        if len1 > len2 and cov1_in_2 >= 0.5:
            # text1 is more specific -> text1 entails text2 (FORWARD_ENTAILMENT = 2)
            return [-0.8, 0.2, 2.6, -1.2]
        if len2 > len1 and cov2_in_1 >= 0.5:
            # text2 is more specific -> text2 entails text1 (BACKWARD_ENTAILMENT = 0)
            return [2.6, 0.2, -0.8, -1.2]

        if jaccard >= 0.65:
            return [-0.3, 2.2, -0.3, -1.0]

        if cov1_in_2 < 0.5 and cov2_in_1 < 0.5:
            return [-1.2, -1.5, -1.2, 2.8]

        # Deterministic tie-breaker from character Lengths
        if len(key[0]) > len(key[1]):
            return [-0.5, 0.1, 1.8, 0.4]
        elif len(key[1]) > len(key[0]):
            return [1.8, 0.1, -0.5, 0.4]
        return [-0.4, 0.8, -0.4, 1.2]

    def _score_token_ids_row(self, ids: Sequence[int]) -> List[float]:
        """Produces deterministic 4-class logits from a sequence of token IDs."""
        # Locate separator token (102 for BERT/DeBERTa, 2 for RoBERTa)
        sep_id = 2 if (ids and ids[0] == 0) else 102
        pad_id = 1 if sep_id == 2 else 0
        non_pad = [x for x in ids if x != pad_id]

        if sep_id in non_pad:
            first_sep = non_pad.index(sep_id)
            seg1 = [x for x in non_pad[1:first_sep] if x not in (0, 1, 2, 101, 102)]
            seg2 = [x for x in non_pad[first_sep + 1 :] if x not in (0, 1, 2, 101, 102)]
        else:
            mid = len(non_pad) // 2
            seg1 = non_pad[:mid]
            seg2 = non_pad[mid:]

        if not seg1 and not seg2:
            return [-1.0, 2.5, -1.0, -1.0]
        if seg1 == seg2:
            return [-1.2, 3.2, -1.2, -1.5]

        set1, set2 = set(seg1), set(seg2)
        inter = len(set1 & set2)
        if inter == 0 and len(set1) > 0 and len(set2) > 0:
            # Also provide slight smooth dependence on token values for test sensitivity
            s1 = (sum(seg1) % 100) / 100.0
            s2 = (sum(seg2) % 100) / 100.0
            if len(seg1) > len(seg2):
                return [-0.6 + 0.1 * s2, 0.0, 1.8 + 0.1 * s1, 0.9]
            elif len(seg2) > len(seg1):
                return [1.8 + 0.1 * s2, 0.0, -0.6 + 0.1 * s1, 0.9]
            return [-0.8, -0.8, -0.8, 2.4]

        if len(seg1) > len(seg2):
            return [-0.8, 0.2, 2.5, -1.0]
        if len(seg2) > len(seg1):
            return [2.5, 0.2, -0.8, -1.0]
        return [-0.3, 2.1, -0.3, -0.9]

    def forward(
        self,
        input_ids: Any = None,
        attention_mask: Any = None,
        token_type_ids: Any = None,
        labels: Any = None,
        text1: Optional[Union[str, Sequence[str]]] = None,
        text2: Optional[Union[str, Sequence[str]]] = None,
        **kwargs: Any,
    ) -> CrossEncoderOutput:
        """
        Executes a forward pass over either tokenized `input_ids` or raw `(text1, text2)` pairs.
        Returns a `CrossEncoderOutput` containing `loss`, `logits`, `probabilities`, and `predictions`.
        """
        use_torch_out = HAS_TORCH and (
            (input_ids is not None and isinstance(input_ids, torch.Tensor))
            or (labels is not None and isinstance(labels, torch.Tensor))
        )
        device = input_ids.device if (use_torch_out and isinstance(input_ids, torch.Tensor)) else None

        batch_logits: List[List[float]] = []

        if text1 is not None and text2 is not None:
            if isinstance(text1, str) and isinstance(text2, str):
                batch_logits.append(self.score_text_pair(text1, text2))
            else:
                for t1, t2 in zip(text1, text2):
                    batch_logits.append(self.score_text_pair(str(t1), str(t2)))
        elif input_ids is not None:
            if HAS_TORCH and isinstance(input_ids, torch.Tensor):
                ids_list = input_ids.detach().cpu().tolist()
            else:
                ids_list = list(input_ids)
            if ids_list and isinstance(ids_list[0], int):
                ids_list = [ids_list]
            for row in ids_list:
                batch_logits.append(self._score_token_ids_row(row))
        else:
            batch_logits = [[0.0, 0.0, 0.0, 0.0]]

        probs_list = _softmax_2d(batch_logits)
        preds_list = [ int(max(range(len(r)), key=lambda i: r[i])) for r in probs_list ]

        loss_val: Optional[Any] = None
        if labels is not None:
            if HAS_TORCH and isinstance(labels, torch.Tensor):
                labels_list = labels.detach().cpu().view(-1).tolist()
            elif isinstance(labels, int):
                labels_list = [labels]
            else:
                labels_list = [int(x) for x in labels]
            raw_loss = _cross_entropy_loss_2d(batch_logits, labels_list)
            loss_val = torch.tensor(raw_loss, dtype=torch.float32, device=device) if use_torch_out else raw_loss

        if use_torch_out:
            logits_out: Any = torch.tensor(batch_logits, dtype=torch.float32, device=device)
            probs_out: Any = torch.tensor(probs_list, dtype=torch.float32, device=device)
            preds_out: Any = torch.tensor(preds_list, dtype=torch.long, device=device)
        else:
            logits_out = batch_logits
            probs_out = probs_list
            preds_out = preds_list

        return CrossEncoderOutput(
            loss=loss_val,
            logits=logits_out,
            probabilities=probs_out,
            predictions=preds_out,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> CrossEncoderOutput:
        return self.forward(*args, **kwargs)

    def predict_logits(self, text1: str, text2: str) -> List[float]:
        return self.score_text_pair(text1, text2)

    def predict_proba(self, text1: str, text2: str) -> List[float]:
        return _softmax_1d(self.score_text_pair(text1, text2))

    def predict(self, text1: str, text2: str) -> str:
        probs = self.predict_proba(text1, text2)
        best_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        return ID_TO_LABEL[best_idx]

    def train_step(self, records: Sequence[Any]) -> float:
        """Simulates a single optimization step on a batch of dict or ReciprocalPair records and returns mock loss."""
        if not records:
            return 0.0
        batch_logits: List[List[float]] = []
        batch_labels: List[int] = []
        flat_records: List[Any] = []
        for r in records:
            if hasattr(r, "direct_instance") and hasattr(r, "inverse_instance"):
                flat_records.append(r.direct_instance)
                flat_records.append(r.inverse_instance)
            else:
                flat_records.append(r)

        for r in flat_records:
            if isinstance(r, dict):
                t1 = str(r.get("text1", "")).strip()
                t2 = str(r.get("text2", "")).strip()
                lbl = r.get("label")
            else:
                t1 = str(getattr(r, "text1", "")).strip()
                t2 = str(getattr(r, "text2", "")).strip()
                lbl = getattr(r, "label", None)
            if lbl in LABEL_TO_ID:
                batch_logits.append(self.score_text_pair(t1, t2))
                batch_labels.append(LABEL_TO_ID[lbl])
                self.pair_memory[(t1, t2)] = lbl
        self.step_count += 1
        return _cross_entropy_loss_2d(batch_logits, batch_labels)


# ==============================================================================
# 6. Main CrossEncoderNLI Wrapper (PyTorch + HuggingFace or Mock Fallback)
# ==============================================================================

_BaseModule = nn.Module if HAS_TORCH else object


class CrossEncoderNLI(_BaseModule):  # type: ignore[misc]
    """
    Standard Cross-Encoder architecture for 4-way NLI classification.

    Concatenates `(text1, text2)` with backbone-specific delimiters:
      - DeBERTa-v3 / BETO: `[CLS] text1 [SEP] text2 [SEP]`
      - RoBERTa-BNE:       `<s> text1 </s> </s> text2 </s>`
    Pools the contextual representation of the initial classification token
    (`h_{[CLS]}` or `h_{<s>}`) and projects it to `R^4`:
      - 0: BACKWARD_ENTAILMENT
      - 1: EQUIVALENCE
      - 2: FORWARD_ENTAILMENT
      - 3: NEGATIVE_OTHER

    Trained via standard Cross-Entropy loss:
      L_{CE} = - sum_{c=0}^3 y_c log p_theta(c | x_1, x_2)
    """

    def __init__(
        self,
        model_name_or_path: str = "microsoft/deberta-v3-base",
        num_labels: int = 4,
        dropout: Optional[float] = None,
        is_mock: bool = False,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        if HAS_TORCH:
            super().__init__()

        self.model_name_or_path = canonicalize_model_name(model_name_or_path)
        self.backbone_family = detect_backbone_family(self.model_name_or_path)
        self.num_labels = num_labels
        self.seed = seed
        self.id2label = dict(ID_TO_LABEL)
        self.label2id = dict(LABEL_TO_ID)
        self.training = True
        self._tokenizer: Optional[Any] = None

        # Determine whether to run in mock mode
        self.is_mock = (
            is_mock
            or self.model_name_or_path == "mock-cross-encoder"
            or "mock" in self.model_name_or_path.lower()
            or not HAS_TORCH
            or not HAS_TRANSFORMERS
        )

        if self.is_mock:
            self.mock_engine = MockCrossEncoder(
                model_name=self.model_name_or_path,
                num_labels=self.num_labels,
                seed=self.seed,
            )
            self.model = None
            self.config = None
            self.loss_fn = None
        else:
            self.mock_engine = None
            config_kwargs: Dict[str, Any] = {
                "num_labels": self.num_labels,
                "id2label": self.id2label,
                "label2id": self.label2id,
            }
            if dropout is not None:
                config_kwargs["hidden_dropout_prob"] = dropout
                config_kwargs["classifier_dropout"] = dropout

            self.config = AutoConfig.from_pretrained(
                self.model_name_or_path,
                **config_kwargs,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name_or_path,
                config=self.config,
            )
            if HAS_TORCH and hasattr(self.model, "float"):
                self.model = self.model.float()
            if HAS_TORCH and torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if vram_gb < 6.0:
                    for name, param in self.model.named_parameters():
                        if "word_embeddings" in name:
                            param.requires_grad = False
            self.loss_fn = nn.CrossEntropyLoss()

    def train(self, mode: bool = True) -> "CrossEncoderNLI":
        """Sets the module in training mode (works with or without PyTorch installed)."""
        if HAS_TORCH and hasattr(super(), "train"):
            super().train(mode)
        self.training = bool(mode)
        if getattr(self, "mock_engine", None) is not None:
            self.mock_engine.train(mode)
        return self

    def eval(self) -> "CrossEncoderNLI":
        """Sets the module in evaluation mode (works with or without PyTorch installed)."""
        return self.train(False)

    def to(self, *args: Any, **kwargs: Any) -> "CrossEncoderNLI":
        """Moves and/or casts parameters and buffers (works with or without PyTorch installed)."""
        if HAS_TORCH and hasattr(super(), "to"):
            super().to(*args, **kwargs)
        return self

    def get_tokenizer(self, max_length: int = 128) -> Any:
        """Returns a cached matching Hugging Face AutoTokenizer or MockTokenizer."""
        if self.is_mock:
            if self._tokenizer is None:
                self._tokenizer = MockTokenizer(model_name=self.model_name_or_path, max_length=max_length)
            return self._tokenizer
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        return self._tokenizer

    def forward(
        self,
        input_ids: Any = None,
        attention_mask: Any = None,
        token_type_ids: Any = None,
        labels: Any = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        **kwargs: Any,
    ) -> CrossEncoderOutput:
        """
        Forward pass for single-pair cross-encoder classification.

        Parameters
        ----------
        input_ids : Tensor or List[List[int]] of shape (batch_size, seq_len)
        attention_mask : Tensor or List[List[int]] of shape (batch_size, seq_len)
        token_type_ids : Optional Tensor or List[List[int]] of shape (batch_size, seq_len)
        labels : Optional Tensor or List[int] of shape (batch_size,) with values in {0, 1, 2, 3}

        Returns
        -------
        CrossEncoderOutput
            Contains `loss`, `logits` (B, 4), `probabilities` (B, 4), and `predictions` (B,).
        """
        if self.is_mock or self.model is None:
            return self.mock_engine.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
                **kwargs,
            )

        # Dynamically trim trailing batch padding to minimize O(L^2) VRAM consumption
        if (
            HAS_TORCH
            and isinstance(input_ids, torch.Tensor)
            and isinstance(attention_mask, torch.Tensor)
            and input_ids.dim() == 2
            and attention_mask.dim() == 2
        ):
            max_active = int(attention_mask.sum(dim=1).max().item())
            max_active = max(4, min(int(input_ids.shape[1]), max_active))
            input_ids = input_ids[:, :max_active]
            attention_mask = attention_mask[:, :max_active]
            if isinstance(token_type_ids, torch.Tensor) and token_type_ids.dim() == 2:
                token_type_ids = token_type_ids[:, :max_active]

        # Prepare inputs for Hugging Face AutoModelForSequenceClassification
        forward_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        # RoBERTa does not take token_type_ids; BERT and DeBERTa support them
        if token_type_ids is not None and self.backbone_family != "roberta":
            forward_kwargs["token_type_ids"] = token_type_ids
        if output_hidden_states is not None:
            forward_kwargs["output_hidden_states"] = output_hidden_states
        if output_attentions is not None:
            forward_kwargs["output_attentions"] = output_attentions

        hf_out = self.model(**forward_kwargs)
        logits = hf_out.logits  # shape: (batch_size, 4)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        return CrossEncoderOutput(
            loss=loss,
            logits=logits,
            probabilities=probs,
            predictions=preds,
            hidden_states=getattr(hf_out, "hidden_states", None),
            attentions=getattr(hf_out, "attentions", None),
        )

    def __call__(self, *args: Any, **kwargs: Any) -> CrossEncoderOutput:
        if HAS_TORCH and not self.is_mock:
            return super().__call__(*args, **kwargs)
        return self.forward(*args, **kwargs)

    def predict_logits(self, text1: str, text2: str) -> List[float]:
        """Computes raw 4-class logits [BWD, EQ, FWD, NEG] for a single (text1, text2) pair."""
        if self.is_mock or self.model is None:
            return self.mock_engine.predict_logits(text1, text2)

        tokenizer = self.get_tokenizer()
        device = next(self.model.parameters()).device
        encoded = tokenizer(
            text1,
            text2,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        ).to(device)
        with torch.no_grad():
            out = self.forward(**encoded)
            logits_tensor = out.logits[0].detach().cpu().tolist()
            return [float(x) for x in logits_tensor]

    def predict_proba(self, text1: str, text2: str) -> List[float]:
        """Computes normalized 4-class probabilities [p_BWD, p_EQ, p_FWD, p_NEG] for a single pair."""
        logits = self.predict_logits(text1, text2)
        return _softmax_1d(logits)

    def predict(self, text1: str, text2: str) -> str:
        """Predicts the canonical string label for a single (text1, text2) pair."""
        probs = self.predict_proba(text1, text2)
        best_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        return ID_TO_LABEL[best_idx]

    def train_step(self, records: Sequence[Dict[str, Any]]) -> float:
        """Convenience method for mock/dry-run training step compatibility."""
        if self.is_mock and self.mock_engine is not None:
            return self.mock_engine.train_step(records)
        return 0.0
