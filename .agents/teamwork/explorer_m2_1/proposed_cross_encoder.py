"""
src/models/cross_encoder.py

Standard Cross-Encoder NLI Architecture for SemEval-2027 Task 2 (DiCo-NLI).
Provides:
  1. Full PyTorch / Hugging Face Transformers cross-encoder fine-tuning backbone.
  2. Native support for DeBERTa-v3 (with proper cls_dropout handling), BETO, and RoBERTa-bne.
  3. 4-class classification head:
       0: BACKWARD_ENTAILMENT
       1: EQUIVALENCE
       2: FORWARD_ENTAILMENT
       3: NEGATIVE_OTHER
  4. Unified CrossEncoderOutput dataclass supporting logits, probabilities, predictions, and loss.
  5. Pure-Python fallback mock architecture (MockCrossEncoder & MockTokenizer)
     for fast CPU execution (<60s) without torch/transformers dependencies.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

# Conditional PyTorch and Transformers imports with graceful pure-Python fallback
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
    import transformers
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        PreTrainedModel,
        PreTrainedTokenizerBase,
    )
    from transformers.modeling_outputs import SequenceClassifierOutput
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = None
    AutoConfig = None
    AutoModel = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    PreTrainedModel = None
    PreTrainedTokenizerBase = None
    SequenceClassifierOutput = None
    HAS_TRANSFORMERS = False


# ==============================================================================
# 1. Label Inventory & Supported Backbones
# ==============================================================================

class LabelIndex(int, Enum):
    """Canonical 4-class label index encoding for SemEval-2027 Task 2."""
    BACKWARD_ENTAILMENT = 0
    EQUIVALENCE = 1
    FORWARD_ENTAILMENT = 2
    NEGATIVE_OTHER = 3


LABEL_TO_ID: Dict[str, int] = {
    "BACKWARD_ENTAILMENT": 0,
    "EQUIVALENCE": 1,
    "FORWARD_ENTAILMENT": 2,
    "NEGATIVE_OTHER": 3,
}

ID_TO_LABEL: Dict[int, str] = {
    0: "BACKWARD_ENTAILMENT",
    1: "EQUIVALENCE",
    2: "FORWARD_ENTAILMENT",
    3: "NEGATIVE_OTHER",
}

NUM_CLASSES: int = 4

# Target Foundation Model Backbones
SUPPORTED_BACKBONES: Dict[str, str] = {
    "deberta-v3-base": "microsoft/deberta-v3-base",
    "deberta-v3-small": "microsoft/deberta-v3-small",
    "beto": "dccuchile/bert-base-spanish-wwm-cased",
    "roberta-bne": "PlanTL-GOB-ES/roberta-base-bne",
}


def canonicalize_model_name(name_or_path: str) -> str:
    """Resolve short aliases to canonical Hugging Face repository identifiers."""
    cleaned = name_or_path.strip().lower()
    if cleaned in SUPPORTED_BACKBONES:
        return SUPPORTED_BACKBONES[cleaned]
    return name_or_path.strip()


def detect_backbone_family(model_name_or_path: str) -> str:
    """Identify the underlying transformer architecture family."""
    name = model_name_or_path.lower()
    if "deberta" in name:
        return "deberta"
    elif "roberta" in name:
        return "roberta"
    elif "bert" in name:
        return "bert"
    return "generic"


# ==============================================================================
# 2. Standardized Output Container
# ==============================================================================

@dataclass
class CrossEncoderOutput:
    """Container for CrossEncoder forward pass outputs.
    
    Provides dictionary-style, attribute-style, and tuple unpacking interfaces
    for full compatibility with Hugging Face Trainer and custom loops.
    """
    loss: Optional[Any] = None
    logits: Any = None
    probabilities: Optional[Any] = None
    predictions: Optional[Any] = None
    hidden_states: Optional[Any] = None
    attentions: Optional[Any] = None

    def __getitem__(self, item: Union[int, str]) -> Any:
        if isinstance(item, str):
            return getattr(self, item)
        fields = [self.loss, self.logits, self.probabilities, self.predictions]
        return fields[item]

    def __iter__(self):
        for val in [self.loss, self.logits, self.probabilities, self.predictions]:
            yield val

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loss": self.loss,
            "logits": self.logits,
            "probabilities": self.probabilities,
            "predictions": self.predictions,
        }


# ==============================================================================
# 3. Pure-Python Fallback Mock Transformer Architecture
# ==============================================================================

class MockTokenizer:
    """Lightweight deterministic tokenizer running without transformers library."""

    def __init__(
        self,
        model_name: str = "mock-encoder",
        model_max_length: int = 128,
    ) -> None:
        self.model_name = model_name
        self.family = detect_backbone_family(model_name)
        self.model_max_length = model_max_length

        if self.family == "roberta":
            self.cls_token = "<s>"
            self.sep_token = "</s>"
            self.pad_token = "<pad>"
            self.cls_token_id = 0
            self.pad_token_id = 1
            self.sep_token_id = 2
            self.unk_token_id = 3
        else:
            self.cls_token = "[CLS]"
            self.sep_token = "[SEP]"
            self.pad_token = "[PAD]"
            self.cls_token_id = 1
            self.pad_token_id = 0
            self.sep_token_id = 2
            self.unk_token_id = 3

    def _tokenize_text(self, text: str) -> List[int]:
        tokens = re.findall(r"\w+|[^\w\s]", text.lower())
        token_ids = []
        for t in tokens:
            token_ids.append((abs(hash(t)) % 1000) + 10)
        return token_ids

    def __call__(
        self,
        text_a: Union[str, Sequence[str]],
        text_b: Optional[Union[str, Sequence[str]]] = None,
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, Any]:
        max_len = max_length or self.model_max_length

        single_input = isinstance(text_a, str)
        list_a = [text_a] if single_input else list(text_a)
        list_b = None
        if text_b is not None:
            list_b = [text_b] if isinstance(text_b, str) else list(text_b)
            if len(list_a) != len(list_b):
                raise ValueError("text_a and text_b lists must have equal length")

        batch_input_ids = []
        batch_attention_mask = []
        batch_token_type_ids = []

        for i, a in enumerate(list_a):
            b = list_b[i] if list_b is not None else None
            tokens_a = self._tokenize_text(a)

            if b is not None:
                tokens_b = self._tokenize_text(b)
                if self.family == "roberta":
                    # RoBERTa pair delimiter format: <s> A </s></s> B </s>
                    input_ids = [self.cls_token_id] + tokens_a + [self.sep_token_id, self.sep_token_id] + tokens_b + [self.sep_token_id]
                    token_types = [0] * len(input_ids)
                else:
                    # BERT / DeBERTa delimiter format: [CLS] A [SEP] B [SEP]
                    input_ids = [self.cls_token_id] + tokens_a + [self.sep_token_id] + tokens_b + [self.sep_token_id]
                    len_first = len(tokens_a) + 2
                    token_types = [0] * len_first + [1] * (len(tokens_b) + 1)
            else:
                input_ids = [self.cls_token_id] + tokens_a + [self.sep_token_id]
                token_types = [0] * len(input_ids)

            if truncation and len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                token_types = token_types[:max_len]

            batch_input_ids.append(input_ids)
            batch_token_type_ids.append(token_types)

        if not padding:
            out = {
                "input_ids": batch_input_ids,
                "attention_mask": [[1] * len(ids) for ids in batch_input_ids],
            }
            if self.family != "roberta":
                out["token_type_ids"] = batch_token_type_ids
            if return_tensors == "pt" and HAS_TORCH:
                return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}
            return out

        actual_max = max(len(ids) for ids in batch_input_ids)
        target_len = min(actual_max, max_len)

        padded_input_ids = []
        padded_attention_mask = []
        padded_token_type_ids = []

        for ids, types in zip(batch_input_ids, batch_token_type_ids):
            pad_needed = target_len - len(ids)
            if pad_needed > 0:
                p_ids = ids + [self.pad_token_id] * pad_needed
                p_mask = [1] * len(ids) + [0] * pad_needed
                p_types = types + [0] * pad_needed
            else:
                p_ids = ids[:target_len]
                p_mask = [1] * target_len
                p_types = types[:target_len]

            padded_input_ids.append(p_ids)
            padded_attention_mask.append(p_mask)
            padded_token_type_ids.append(p_types)

        out = {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
        }
        if self.family != "roberta":
            out["token_type_ids"] = padded_token_type_ids

        if return_tensors == "pt" and HAS_TORCH:
            return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}
        return out


class MockCrossEncoder:
    """Deterministic, lightweight CPU model producing calibrated 4-class logits."""

    def __init__(
        self,
        model_name: str = "mock-cross-encoder",
        num_labels: int = 4,
        seed: int = 42,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.family = detect_backbone_family(model_name)
        self.num_labels = num_labels
        self.seed = seed
        self.training = True

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def eval(self):
        self.training = False
        return self

    def to(self, device: Any):
        return self

    def parameters(self):
        return iter([])

    @staticmethod
    def _compute_softmax(logits: Sequence[float]) -> List[float]:
        max_l = max(logits)
        exp_vals = [math.exp(x - max_l) for x in logits]
        s = sum(exp_vals)
        return [v / s if s > 0 else 0.25 for v in exp_vals]

    def predict_logits(self, text1: str, text2: str) -> List[float]:
        """Generate deterministic 4-class logits [BWD, EQUIV, FWD, NEG]."""
        words1 = set(re.findall(r"\w+", text1.lower()))
        words2 = set(re.findall(r"\w+", text2.lower()))
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        jaccard = len(intersection) / max(1, len(union))

        len1, len2 = len(text1), len(text2)
        diff = len1 - len2

        # 1. Negative / Non-entailing: zero lexical overlap and divergent semantics
        if len(intersection) == 0 and abs(diff) < 25:
            # High probability on NEGATIVE_OTHER (index 3)
            return [0.2, 0.3, 0.2, 3.2]

        # 2. High overlap, identical words, or close length with common token -> EQUIVALENCE (index 1)
        if words1 == words2 or jaccard >= 0.33 or abs(diff) <= 3:
            return [0.2, 3.0, 0.3, 0.1]

        # 3. Longer text1 -> more specific premise entails shorter hypothesis -> FORWARD (index 2)
        if diff > 3:
            return [0.2, 0.4, 3.1, 0.1]

        # 4. Shorter text1 -> entailed by longer text2 -> BACKWARD (index 0)
        return [3.1, 0.4, 0.2, 0.1]

    def predict_proba(self, text1: str, text2: str) -> List[float]:
        return self._compute_softmax(self.predict_logits(text1, text2))

    def predict(self, text1: str, text2: str) -> str:
        probs = self.predict_proba(text1, text2)
        best_idx = probs.index(max(probs))
        return ID_TO_LABEL[best_idx]

    def train_step(self, records: List[Dict[str, Any]]) -> float:
        """Simulate training step loss computation."""
        if not records:
            return 0.0
        total_loss = 0.0
        for r in records:
            t1 = r.get("text1", "")
            t2 = r.get("text2", "")
            logits = self.predict_logits(t1, t2)
            probs = self._compute_softmax(logits)
            label_str = r.get("label", "EQUIVALENCE")
            target_idx = LABEL_TO_ID.get(label_str, 1)
            target_p = max(probs[target_idx], 1e-12)
            total_loss += -math.log(target_p)
        return total_loss / len(records)

    def forward(
        self,
        input_ids: Any,
        attention_mask: Optional[Any] = None,
        token_type_ids: Optional[Any] = None,
        labels: Optional[Any] = None,
        **kwargs,
    ) -> CrossEncoderOutput:
        """Forward pass for mock architecture returning CrossEncoderOutput."""
        # Convert tensor to nested list if PyTorch tensor is passed
        if HAS_TORCH and isinstance(input_ids, torch.Tensor):
            batch_ids = input_ids.detach().cpu().tolist()
        else:
            batch_ids = input_ids if isinstance(input_ids, list) else [input_ids]

        batch_size = len(batch_ids)
        batch_logits = []
        batch_probs = []
        batch_preds = []

        for seq in batch_ids:
            # Deterministic pseudo logits from sequence token patterns
            s_len = len(seq)
            token_sum = sum(int(x) for x in seq if int(x) != 0)
            mod_val = (token_sum + s_len) % 4

            if mod_val == 0:
                logits = [2.8, 0.5, 0.2, 0.3]
            elif mod_val == 1:
                logits = [0.4, 2.9, 0.4, 0.2]
            elif mod_val == 2:
                logits = [0.2, 0.5, 2.8, 0.3]
            else:
                logits = [0.3, 0.3, 0.3, 2.7]

            probs = self._compute_softmax(logits)
            pred = probs.index(max(probs))

            batch_logits.append(logits)
            batch_probs.append(probs)
            batch_preds.append(pred)

        # Compute cross entropy loss if labels are provided
        loss_val = None
        if labels is not None:
            if HAS_TORCH and isinstance(labels, torch.Tensor):
                targets = labels.detach().cpu().tolist()
            else:
                targets = labels if isinstance(labels, list) else [labels]

            total_loss = 0.0
            for p, target in zip(batch_probs, targets):
                p_c = max(p[int(target)], 1e-12)
                total_loss += -math.log(p_c)
            loss_val = total_loss / max(1, len(targets))

        # Wrap in PyTorch tensors if torch is available
        if HAS_TORCH:
            out_logits = torch.tensor(batch_logits, dtype=torch.float32)
            out_probs = torch.tensor(batch_probs, dtype=torch.float32)
            out_preds = torch.tensor(batch_preds, dtype=torch.long)
            out_loss = torch.tensor(loss_val, dtype=torch.float32) if loss_val is not None else None
        else:
            out_logits = batch_logits
            out_probs = batch_probs
            out_preds = batch_preds
            out_loss = loss_val

        return CrossEncoderOutput(
            loss=out_loss,
            logits=out_logits,
            probabilities=out_probs,
            predictions=out_preds,
        )


# ==============================================================================
# 4. PyTorch / Transformers Cross-Encoder Architecture
# ==============================================================================

# Base class dispatching to nn.Module if PyTorch is available, else object
if HAS_TORCH:
    _ModuleBase = torch.nn.Module
else:
    class _ModuleBase:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)


class CrossEncoderNLI(_ModuleBase):
    """Production Cross-Encoder for Natural Language Inference.
    
    Supports:
      - DeBERTa-v3 (`microsoft/deberta-v3-base`, `small`) with native cls_dropout handling.
      - BETO (`dccuchile/bert-base-spanish-wwm-cased`) with BERT pooling head.
      - RoBERTa-bne (`PlanTL-GOB-ES/roberta-base-bne`) with RoBERTa classification head.
      - Automatic fallback to MockCrossEncoder when running on CPU without PyTorch.
    """

    def __init__(
        self,
        model_name_or_path: str = "microsoft/deberta-v3-base",
        num_labels: int = NUM_CLASSES,
        cls_dropout: Optional[float] = None,
        classifier_dropout: Optional[float] = None,
        hidden_dropout_prob: Optional[float] = None,
        is_mock: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_name_or_path = canonicalize_model_name(model_name_or_path)
        self.family = detect_backbone_family(self.model_name_or_path)
        self.num_labels = num_labels
        self.is_mock = is_mock or (not HAS_TORCH) or (not HAS_TRANSFORMERS)

        if self.is_mock:
            self.mock_engine = MockCrossEncoder(
                model_name=self.model_name_or_path,
                num_labels=self.num_labels,
                **kwargs,
            )
            self.supports_token_type_ids = (self.family != "roberta")
            return

        # ----------------------------------------------------------------------
        # Full PyTorch & Hugging Face Initialization
        # ----------------------------------------------------------------------
        self.config = AutoConfig.from_pretrained(
            self.model_name_or_path,
            num_labels=self.num_labels,
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
            **kwargs,
        )

        # Architectural special handling: cls_dropout vs classifier_dropout
        if self.family == "deberta":
            # DeBERTa-v3 uses `cls_dropout` rather than `classifier_dropout`
            effective_drop = cls_dropout if cls_dropout is not None else classifier_dropout
            if effective_drop is not None:
                self.config.cls_dropout = effective_drop
            elif not hasattr(self.config, "cls_dropout") or self.config.cls_dropout is None:
                self.config.cls_dropout = getattr(self.config, "hidden_dropout_prob", 0.1)
        else:
            if classifier_dropout is not None:
                self.config.classifier_dropout = classifier_dropout

        if hidden_dropout_prob is not None:
            self.config.hidden_dropout_prob = hidden_dropout_prob

        # Load Hugging Face Sequence Classification Model
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path,
            config=self.config,
        )

        # Flag determining whether backbone accepts token_type_ids
        self.supports_token_type_ids = (self.family != "roberta")
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: Any,
        attention_mask: Optional[Any] = None,
        token_type_ids: Optional[Any] = None,
        labels: Optional[Any] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> CrossEncoderOutput:
        """Execute forward pass.
        
        Args:
            input_ids: Input token sequences of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            token_type_ids: Segment token type IDs (omitted for RoBERTa)
            labels: Ground truth class indices of shape (batch_size,)
            output_hidden_states: Whether to return all hidden layers
            output_attentions: Whether to return attention weights
            return_dict: Whether to wrap output in CrossEncoderOutput
            
        Returns:
            CrossEncoderOutput containing logits, probabilities, predictions, and loss.
        """
        if self.is_mock:
            return self.mock_engine.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
                **kwargs,
            )

        # Suppress token_type_ids for RoBERTa architectures to prevent runtime warnings/errors
        if not self.supports_token_type_ids:
            token_type_ids = None

        hf_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )

        logits = hf_outputs.logits
        probabilities = F.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)
        loss = hf_outputs.loss

        return CrossEncoderOutput(
            loss=loss,
            logits=logits,
            probabilities=probabilities,
            predictions=predictions,
            hidden_states=hf_outputs.hidden_states,
            attentions=hf_outputs.attentions,
        )

    def predict_logits(self, text1: str, text2: str) -> Any:
        """Generate logits for an individual text pair."""
        if self.is_mock:
            return self.mock_engine.predict_logits(text1, text2)
        tokenizer = self.get_tokenizer()
        inputs = tokenizer(text1, text2, return_tensors="pt")
        inputs = {k: v.to(self.backbone.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self.forward(**inputs)
        return output.logits.squeeze(0).cpu().tolist()

    def predict_proba(self, text1: str, text2: str) -> Any:
        """Generate probabilities for an individual text pair."""
        if self.is_mock:
            return self.mock_engine.predict_proba(text1, text2)
        logits = self.predict_logits(text1, text2)
        probs = F.softmax(torch.tensor(logits), dim=-1)
        return probs.tolist()

    def predict(self, text1: str, text2: str) -> str:
        """Predict discrete label string for an individual text pair."""
        if self.is_mock:
            return self.mock_engine.predict(text1, text2)
        probs = self.predict_proba(text1, text2)
        idx = probs.index(max(probs))
        return ID_TO_LABEL[idx]

    def get_tokenizer(self) -> Any:
        """Retrieve compatible tokenizer for the configured backbone."""
        if self.is_mock or not HAS_TRANSFORMERS:
            return MockTokenizer(self.model_name_or_path)
        return AutoTokenizer.from_pretrained(self.model_name_or_path)

    def save_pretrained(self, save_directory: str) -> None:
        """Save model checkpoint to directory."""
        os.makedirs(save_directory, exist_ok=True)
        if self.is_mock:
            config_path = os.path.join(save_directory, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(f'{{"model_name": "{self.model_name_or_path}", "is_mock": true, "num_labels": {self.num_labels}}}\n')
        else:
            self.backbone.save_pretrained(save_directory)
            self.config.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, save_directory: str, **kwargs) -> CrossEncoderNLI:
        """Load model checkpoint from directory."""
        config_path = os.path.join(save_directory, "config.json")
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                content = f.read()
            if '"is_mock": true' in content:
                return cls(model_name_or_path="mock-cross-encoder", is_mock=True, **kwargs)
        return cls(model_name_or_path=save_directory, **kwargs)
