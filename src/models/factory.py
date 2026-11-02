"""
src/models/factory.py

Unified Model, Loss, and Tokenizer Factory for DiCo-NLI.
Supports:
- Methods: 'baseline', 'bsa', 'crl', 'sd', 'crl_sd'
- Backbones: 'microsoft/deberta-v3-base', 'dccuchile/bert-base-spanish-wwm-cased',
             'PlanTL-GOB-ES/roberta-base-bne', and 'mock' (CPU fast execution).
- Device & dtype auto-resolution for seamless remote GPU (CUDA/MPS) and local CPU execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    AutoTokenizer = None
    HAS_TRANSFORMERS = False

from src.data.reversal import (
    ID_TO_LABEL,
    LABEL_TO_ID,
)
from src.models.cross_encoder import (
    BACKWARD_ENTAILMENT,
    EQUIVALENCE,
    FORWARD_ENTAILMENT,
    NEGATIVE_OTHER,
    REV_MAP,
    CrossEncoderNLI,
    MockCrossEncoder,
    MockTokenizer,
    canonicalize_model_name,
)
from src.models.decoding import (
    P_RAW_MATRIX,
    REV_INDEX_MAP,
    SymmetrizedDecoder,
)
from src.models.losses import (
    ConsistencyRegularizationLoss,
    JensenShannonDivergenceLoss,
    SymmetricKLDivergenceLoss,
    cross_entropy_pure,
    softmax_pure,
)
from src.models.siamese_crl import SiameseCRLModel


SUPPORTED_METHODS = {"baseline", "bsa", "crl", "sd", "crl_sd"}
MOCK_IDENTIFIERS = {"mock", "dummy", "test", "cpu_mock", "mock-cross-encoder"}


# ==============================================================================
# 1. Hardware & Precision Resolution
# ==============================================================================

def resolve_device(device: Optional[str] = None) -> str:
    """
    Automatically resolves the optimal compute device ('cuda', 'mps', or 'cpu').
    Respects explicit override if provided and available.
    """
    if device is not None and str(device).strip().lower() not in {"auto", ""}:
        req = str(device).strip().lower()
        if req.startswith("cuda") and HAS_TORCH and torch.cuda.is_available():
            return req
        if req == "mps" and HAS_TORCH and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if req == "cpu":
            return "cpu"
        return "cpu"

    if HAS_TORCH and torch is not None:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def resolve_torch_dtype(precision: Optional[str] = None, device: str = "cpu") -> Any:
    """
    Resolves the PyTorch dtype for mixed-precision execution ('fp16', 'bf16', 'fp32').
    Returns None if PyTorch is not installed.
    """
    if not HAS_TORCH or torch is None:
        return None
    if precision is None or str(precision).lower() in {"fp32", "float32", "none", ""}:
        return torch.float32
    prec = str(precision).lower()
    if prec in {"fp16", "float16"} and device.startswith("cuda"):
        return torch.float16
    if prec in {"bf16", "bfloat16"} and (device.startswith("cuda") or device == "cpu"):
        return torch.bfloat16
    return torch.float32


# ==============================================================================
# 2. Deterministic Mock Model (Fast CPU Testing)
# ==============================================================================

@dataclass
class MockModelOutput:
    """Container mimicking HuggingFace SequenceClassifierOutput."""
    loss: Optional[Any] = None
    logits: Any = None
    probs: Optional[Any] = None


class MockTransformerModel:
    """
    Lightweight, deterministic 4-class NLI mock model for fast CPU testing (< 1s).
    Produces reproducible 4-way logits and probabilities based on input token IDs or text pairs.
    """

    def __init__(
        self,
        model_name_or_path: str = "mock",
        num_labels: int = 4,
        method: str = "baseline",
        device: str = "cpu",
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels
        self.method = method
        self.device = device
        self.is_mock = True
        self.training = False

    def to(self, device: Any, *args: Any, **kwargs: Any) -> "MockTransformerModel":
        self.device = str(device)
        return self

    def eval(self) -> "MockTransformerModel":
        self.training = False
        return self

    def train(self, mode: bool = True) -> "MockTransformerModel":
        self.training = mode
        return self

    def _hash_to_logits(self, key: str) -> List[float]:
        digest = hashlib.md5(key.encode("utf-8")).digest()
        raw = [((digest[i] / 255.0) * 4.0 - 2.0) for i in range(self.num_labels)]
        return raw

    def predict_logits(self, text_a: str, text_b: str) -> List[float]:
        """Returns deterministic 4-class logits for a text pair (text_a, text_b)."""
        norm_a = (text_a or "").strip().lower()
        norm_b = (text_b or "").strip().lower()
        if norm_a == norm_b and norm_a != "":
            return [-1.0, 3.0, -1.0, -1.0]  # High score for EQUIVALENCE (index 1)
        return self._hash_to_logits(f"{norm_a}|||{norm_b}")

    def predict_proba(self, text_a: str, text_b: str) -> List[float]:
        """Returns deterministic 4-class probability distribution for (text_a, text_b)."""
        logits = self.predict_logits(text_a, text_b)
        return softmax_pure(logits)

    def predict(self, text_a: str, text_b: str) -> str:
        """Predicts canonical string label for (text_a, text_b)."""
        probs = self.predict_proba(text_a, text_b)
        idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        return ID_TO_LABEL[idx]

    def forward(
        self,
        input_ids: Any = None,
        attention_mask: Any = None,
        token_type_ids: Any = None,
        labels: Any = None,
        text_a: Optional[Union[str, Sequence[str]]] = None,
        text_b: Optional[Union[str, Sequence[str]]] = None,
        **kwargs: Any,
    ) -> MockModelOutput:
        if text_a is not None and text_b is not None:
            if isinstance(text_a, str) and isinstance(text_b, str):
                logits = self.predict_logits(text_a, text_b)
                probs = softmax_pure(logits)
                loss = cross_entropy_pure(logits, int(labels)) if labels is not None else None
                return MockModelOutput(loss=loss, logits=logits, probs=probs)
            else:
                logits_batch = [self.predict_logits(a, b) for a, b in zip(text_a, text_b)]
                probs_batch = [softmax_pure(l) for l in logits_batch]
                loss = cross_entropy_pure(logits_batch, labels) if labels is not None else None
                return MockModelOutput(loss=loss, logits=logits_batch, probs=probs_batch)

        if HAS_TORCH and torch is not None and isinstance(input_ids, torch.Tensor):
            bsz = input_ids.shape[0]
            logits_t = torch.zeros((bsz, self.num_labels), dtype=torch.float32, device=input_ids.device)
            for b in range(bsz):
                row_sum = int(input_ids[b].sum().item())
                for c in range(self.num_labels):
                    logits_t[b, c] = float(((row_sum + c * 17) % 19) - 9) * 0.25
            probs_t = torch.softmax(logits_t, dim=-1)
            loss_t = None
            if labels is not None:
                loss_t = nn.CrossEntropyLoss()(logits_t, labels.long())
            return MockModelOutput(loss=loss_t, logits=logits_t, probs=probs_t)

        if isinstance(input_ids, list):
            if len(input_ids) > 0 and isinstance(input_ids[0], list):
                logits_b = []
                for row in input_ids:
                    s = sum(int(x) for x in row)
                    logits_b.append([float(((s + c * 17) % 19) - 9) * 0.25 for c in range(self.num_labels)])
                probs_b = [softmax_pure(l) for l in logits_b]
                loss_b = cross_entropy_pure(logits_b, labels) if labels is not None else None
                return MockModelOutput(loss=loss_b, logits=logits_b, probs=probs_b)
            else:
                s = sum(int(x) for x in input_ids)
                logits_single = [float(((s + c * 17) % 19) - 9) * 0.25 for c in range(self.num_labels)]
                probs_single = softmax_pure(logits_single)
                loss_single = cross_entropy_pure(logits_single, int(labels)) if labels is not None else None
                return MockModelOutput(loss=loss_single, logits=logits_single, probs=probs_single)

        default_logits = [0.0, 0.0, 0.0, 0.0]
        return MockModelOutput(loss=None, logits=default_logits, probs=[0.25, 0.25, 0.25, 0.25])

    def __call__(self, *args: Any, **kwargs: Any) -> MockModelOutput:
        return self.forward(*args, **kwargs)


# ==============================================================================
# 3. Symmetrized Decoding Wrapper
# ==============================================================================

class SymmetrizedModelWrapper:
    """
    Wraps any base NLI model (`CrossEncoderNLI`, `MockTransformerModel`, `SiameseCRLModel`,
    or HuggingFace model) with a `SymmetrizedDecoder` to guarantee 100% SoftCons during pair inference,
    while delegating all PyTorch `nn.Module` lifecycle methods (`parameters`, `named_parameters`,
    `state_dict`, `load_state_dict`, `to`, `train`, `eval`, `zero_grad`) to `self.base_model`.
    """

    def __init__(
        self,
        base_model: Any,
        mask_negative_other: bool = True,
    ) -> None:
        self.base_model = base_model
        self.is_mock = getattr(base_model, "is_mock", False)
        self.decoder = SymmetrizedDecoder(mask_negative_other=mask_negative_other)
        self.training = getattr(base_model, "training", False)

    def parameters(self, recurse: bool = True) -> Any:
        """Delegates parameter iteration to the underlying base_model for PyTorch optimizers."""
        if hasattr(self.base_model, "parameters"):
            try:
                sig = inspect.signature(self.base_model.parameters)
                if "recurse" in sig.parameters or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                ):
                    return self.base_model.parameters(recurse=recurse)
            except (ValueError, TypeError):
                pass
            return self.base_model.parameters()
        return iter([])

    def named_parameters(self, prefix: str = "", recurse: bool = True) -> Any:
        """Delegates named parameter iteration to the underlying base_model."""
        if hasattr(self.base_model, "named_parameters"):
            try:
                sig = inspect.signature(self.base_model.named_parameters)
                kwargs: Dict[str, Any] = {}
                has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                if "prefix" in sig.parameters or has_varkw:
                    kwargs["prefix"] = prefix
                if "recurse" in sig.parameters or has_varkw:
                    kwargs["recurse"] = recurse
                return self.base_model.named_parameters(**kwargs)
            except (ValueError, TypeError):
                pass
            return self.base_model.named_parameters()
        return iter([])

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Delegates state_dict extraction to the underlying base_model for checkpoint saving."""
        if hasattr(self.base_model, "state_dict"):
            return self.base_model.state_dict(*args, **kwargs)
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> Any:
        """Delegates state_dict loading to the underlying base_model for checkpoint restoration."""
        if hasattr(self.base_model, "load_state_dict"):
            try:
                sig = inspect.signature(self.base_model.load_state_dict)
                if "strict" in sig.parameters or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                ):
                    return self.base_model.load_state_dict(state_dict, strict=strict)
            except (ValueError, TypeError):
                pass
            return self.base_model.load_state_dict(state_dict)
        return None

    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        """Delegates gradient zeroing to the underlying base_model."""
        if hasattr(self.base_model, "zero_grad"):
            self.base_model.zero_grad(*args, **kwargs)

    def to(self, *args: Any, **kwargs: Any) -> "SymmetrizedModelWrapper":
        if hasattr(self.base_model, "to"):
            self.base_model.to(*args, **kwargs)
        return self

    def eval(self) -> "SymmetrizedModelWrapper":
        self.training = False
        if hasattr(self.base_model, "eval"):
            self.base_model.eval()
        return self

    def train(self, mode: bool = True) -> "SymmetrizedModelWrapper":
        self.training = bool(mode)
        if hasattr(self.base_model, "train"):
            self.base_model.train(mode)
        return self

    def predict_proba(self, text_a: str, text_b: str) -> List[float]:
        """Computes normalized 4-class probability vector for (text_a, text_b)."""
        if hasattr(self.base_model, "predict_proba"):
            return list(self.base_model.predict_proba(text_a, text_b))
        if hasattr(self.base_model, "predict_logits"):
            return softmax_pure(self.base_model.predict_logits(text_a, text_b))
        if hasattr(self.base_model, "encoder") and hasattr(self.base_model.encoder, "predict_proba"):
            return list(self.base_model.encoder.predict_proba(text_a, text_b))
        return [0.25, 0.25, 0.25, 0.25]

    def predict(self, text_a: str, text_b: str) -> str:
        """Predicts canonical string label for a single (text_a, text_b) pair."""
        if hasattr(self.base_model, "predict"):
            return str(self.base_model.predict(text_a, text_b))
        probs = self.predict_proba(text_a, text_b)
        best_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        return ID_TO_LABEL[best_idx]

    def predict_pair(
        self,
        text_a: Optional[str] = None,
        text_b: Optional[str] = None,
        mask_negative_other: Optional[bool] = None,
        **kwargs: Any,
    ) -> Tuple[int, int]:
        """
        Predicts symmetrized class indices `(pred_dir, pred_rev)` for a pair `(text_a, text_b)`
        and its reversal `(text_b, text_a)`. Accepts positional or keyword arguments
        (`text_a`/`text_b` or `text1`/`text2`).
        """
        t1 = text_a if text_a is not None else kwargs.get("text1", "")
        t2 = text_b if text_b is not None else kwargs.get("text2", "")
        p_dir = self.predict_proba(t1, t2)
        p_rev = self.predict_proba(t2, t1)
        return self.decoder.decode_pair(
            p_dir,
            p_rev,
            mask_negative_other=mask_negative_other,
        )

    def decode_pair(
        self,
        p_dir: Any,
        p_rev: Any,
        mask_negative_other: Optional[bool] = None,
    ) -> Tuple[int, int]:
        return self.decoder.decode_pair(
            p_dir,
            p_rev,
            mask_negative_other=mask_negative_other,
        )

    def decode_pairs(
        self,
        probs_dir_list: Sequence[Sequence[float]],
        probs_rev_list: Sequence[Sequence[float]],
        mask_negative_other: Optional[bool] = None,
    ) -> Tuple[List[int], List[int]]:
        return self.decoder.decode_pairs(
            probs_dir_list,
            probs_rev_list,
            mask_negative_other=mask_negative_other,
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if hasattr(self.base_model, "forward"):
            return self.base_model.forward(*args, **kwargs)
        return self.base_model(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)


# ==============================================================================
# 4. Unified Factory Functions: build_tokenizer, build_loss, build_model
# ==============================================================================

def build_tokenizer(
    model_name_or_path: str = "mock",
    use_fast: bool = True,
    max_length: int = 128,
    **kwargs: Any,
) -> Any:
    """
    Instantiates a HuggingFace AutoTokenizer for real backbones or a fast MockTokenizer
    when `model_name_or_path` is 'mock' (or when `transformers` is not installed).
    """
    clean_name = str(model_name_or_path).strip().lower()
    if clean_name in MOCK_IDENTIFIERS or not HAS_TRANSFORMERS or AutoTokenizer is None:
        return MockTokenizer(model_name=model_name_or_path, max_length=max_length)

    canonical_name = canonicalize_model_name(model_name_or_path)
    return AutoTokenizer.from_pretrained(canonical_name, use_fast=use_fast, **kwargs)


def build_loss(
    method: str = "baseline",
    divergence: str = "sym_kl",
    lambda_cons: float = 1.0,
    warmup_epochs: Union[int, float] = 1,
    temperature: float = 1.0,
) -> Callable[..., Any]:
    """
    Factory function returning the loss module/callable corresponding to `method`:
    - 'baseline', 'bsa', 'sd': Standard 4-way Cross-Entropy loss (`nn.CrossEntropyLoss` or `cross_entropy_pure`).
    - 'crl', 'crl_sd': `ConsistencyRegularizationLoss` with configurable divergence and warmup.
    """
    clean_method = str(method).strip().lower()
    if clean_method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method '{method}'. Expected one of {sorted(SUPPORTED_METHODS)}.")

    if clean_method in {"crl", "crl_sd"}:
        return ConsistencyRegularizationLoss(
            divergence=divergence,
            lambda_cons=lambda_cons,
            warmup_epochs=warmup_epochs,
            temperature=temperature,
        )

    if HAS_TORCH and nn is not None:
        ce_fn = nn.CrossEntropyLoss()

        def _ce_loss_wrapper(logits: Any, labels: Any) -> Any:
            if isinstance(logits, torch.Tensor) and isinstance(labels, torch.Tensor):
                return ce_fn(logits, labels.long())
            return cross_entropy_pure(logits, labels)

        return _ce_loss_wrapper

    return cross_entropy_pure


def build_model(
    model_name_or_path: str = "mock",
    method: str = "baseline",
    num_labels: int = 4,
    divergence: str = "sym_kl",
    lambda_cons: float = 1.0,
    warmup_epochs: Union[int, float] = 1,
    mask_negative_other: bool = False,
    device: Optional[str] = None,
    precision: Optional[str] = None,
    is_mock: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Unified Model Factory for DiCo-NLI.

    Supported methods:
    - 'baseline': Standard single-direction cross-encoder (`CrossEncoderNLI`).
    - 'bsa': Standard cross-encoder trained on Bi-directional Symmetrically Augmented data (`CrossEncoderNLI`).
    - 'crl': Siamese dual-pass architecture (`SiameseCRLModel`) with `apply_symmetrized_decoding=False`.
    - 'sd': Cross-encoder wrapped with post-hoc `SymmetrizedModelWrapper`.
    - 'crl_sd': `SiameseCRLModel` (`apply_symmetrized_decoding=True`) wrapped with `SymmetrizedModelWrapper`.
    """
    clean_method = str(method).strip().lower()
    if clean_method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method '{method}'. Expected one of {sorted(SUPPORTED_METHODS)}.")

    target_device = resolve_device(device)
    clean_name = str(model_name_or_path).strip().lower()
    use_mock = is_mock or (clean_name in MOCK_IDENTIFIERS) or (not HAS_TORCH) or (not HAS_TRANSFORMERS)

    base_encoder = CrossEncoderNLI(
        model_name_or_path=model_name_or_path,
        num_labels=num_labels,
        is_mock=use_mock,
        **kwargs,
    )

    if not use_mock and HAS_TORCH and hasattr(base_encoder, "to"):
        base_encoder.to(target_device)

    if clean_method in {"baseline", "bsa"}:
        return base_encoder

    if clean_method == "crl":
        siamese = SiameseCRLModel(
            encoder=base_encoder,
            divergence=divergence,
            lambda_cons=lambda_cons,
            warmup_epochs=warmup_epochs,
            apply_symmetrized_decoding=False,
            mask_negative_other=mask_negative_other,
        )
        return siamese.to(target_device) if hasattr(siamese, "to") else siamese

    if clean_method == "sd":
        return SymmetrizedModelWrapper(
            base_model=base_encoder,
            mask_negative_other=mask_negative_other,
        )

    if clean_method == "crl_sd":
        siamese = SiameseCRLModel(
            encoder=base_encoder,
            divergence=divergence,
            lambda_cons=lambda_cons,
            warmup_epochs=warmup_epochs,
            apply_symmetrized_decoding=True,
            mask_negative_other=mask_negative_other,
        )
        if hasattr(siamese, "to"):
            siamese.to(target_device)
        return SymmetrizedModelWrapper(
            base_model=siamese,
            mask_negative_other=mask_negative_other,
        )

    return base_encoder
