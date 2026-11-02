"""
src/models/siamese_crl.py

Siamese Dual-Pass Architecture with Consistency Regularization Loss (CRL)
and Optional Post-Hoc Symmetrized Decoding (SD) for DiCo-NLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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

from src.data.reversal import ID_TO_LABEL
from src.models.decoding import SymmetrizedDecoder, _argmax_with_rev_tie_break
from src.models.losses import (
    ConsistencyRegularizationLoss,
    TranspositionOperator,
    softmax_pure,
)


@dataclass
class SiameseCRLOutput:
    """
    Structured output container for SiameseCRLModel forward and inference passes.
    Compatible with both attribute access (`out.loss`, `out.logits_dir`) and dictionary conversion.
    """
    loss: Optional[Any] = None
    logits_dir: Optional[Any] = None
    logits_rev: Optional[Any] = None
    probs_dir: Optional[Any] = None
    probs_rev: Optional[Any] = None
    sym_probs_dir: Optional[Any] = None
    sym_probs_rev: Optional[Any] = None
    preds_dir: Optional[Any] = None
    preds_rev: Optional[Any] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loss": self.loss,
            "logits_dir": self.logits_dir,
            "logits_rev": self.logits_rev,
            "probs_dir": self.probs_dir,
            "probs_rev": self.probs_rev,
            "sym_probs_dir": self.sym_probs_dir,
            "sym_probs_rev": self.sym_probs_rev,
            "preds_dir": self.preds_dir,
            "preds_rev": self.preds_rev,
            "metrics": self.metrics,
        }


class MockSequenceEncoder:
    """
    Deterministic Mock Sequence Classification Backbone for fast CPU testing
    when HuggingFace / PyTorch model weights are not loaded.
    """

    def __init__(self, num_labels: int = 4, seed: int = 42) -> None:
        self.num_labels = num_labels
        self.seed = seed
        self.is_mock = True
        self.training = False

    def train(self, mode: bool = True) -> "MockSequenceEncoder":
        self.training = bool(mode)
        return self

    def eval(self) -> "MockSequenceEncoder":
        return self.train(False)

    def to(self, *args: Any, **kwargs: Any) -> "MockSequenceEncoder":
        return self

    def __call__(
        self,
        input_ids: Any = None,
        attention_mask: Any = None,
        token_type_ids: Any = None,
        **kwargs: Any,
    ) -> Any:
        if HAS_TORCH and torch is not None and isinstance(input_ids, torch.Tensor):
            batch_size = input_ids.shape[0]
            logits = torch.zeros((batch_size, self.num_labels), dtype=torch.float32, device=input_ids.device)
            for b in range(batch_size):
                row = input_ids[b].tolist()
                for idx, val in enumerate(row):
                    logits[b, (idx + int(val)) % self.num_labels] += float((val % 7) + 1) * 0.1
            return type("MockBackboneOutput", (), {"logits": logits})()

        if isinstance(input_ids, list):
            if len(input_ids) > 0 and isinstance(input_ids[0], list):
                batch_logits = []
                for row in input_ids:
                    l_vec = [0.0] * self.num_labels
                    for idx, val in enumerate(row):
                        l_vec[(idx + int(val)) % self.num_labels] += float((val % 7) + 1) * 0.1
                    batch_logits.append(l_vec)
                return type("MockBackboneOutput", (), {"logits": batch_logits})()
            else:
                l_vec = [0.0] * self.num_labels
                for idx, val in enumerate(input_ids):
                    l_vec[(idx + int(val)) % self.num_labels] += float((val % 7) + 1) * 0.1
                return type("MockBackboneOutput", (), {"logits": l_vec})()

        return type("MockBackboneOutput", (), {"logits": [[0.25, 0.25, 0.25, 0.25]]})()


_BaseModule = nn.Module if HAS_TORCH else object


class SiameseCRLModel(_BaseModule):  # type: ignore[misc]
    """
    Siamese Dual-Pass Cross-Encoder for Directional & Consistent NLI.

    Architecture:
    - Shared-weight encoder `f_theta` (e.g., DeBERTa-v3-base, BETO, RoBERTa-BNE, or CrossEncoderNLI).
    - Processes `(x_1, x_2)` and `(x_2, x_1)` in two forward passes through the exact same weights.
    - Computes `ConsistencyRegularizationLoss` during training.
    - Optionally applies `SymmetrizedDecoder` during inference/evaluation when
      `apply_symmetrized_decoding=True` (for `crl_sd`), or raw directional predictions
      when `apply_symmetrized_decoding=False` (for pure `crl` ablation).
    """

    def __init__(
        self,
        encoder: Optional[Any] = None,
        divergence: str = "sym_kl",
        lambda_cons: float = 1.0,
        warmup_epochs: Union[int, float] = 1,
        temperature: float = 1.0,
        apply_symmetrized_decoding: bool = False,
        mask_negative_other: bool = False,
    ) -> None:
        if HAS_TORCH:
            super().__init__()
        self.encoder = encoder if encoder is not None else MockSequenceEncoder()
        self.is_mock = getattr(self.encoder, "is_mock", False)
        self.loss_fn = ConsistencyRegularizationLoss(
            divergence=divergence,
            lambda_cons=lambda_cons,
            warmup_epochs=warmup_epochs,
            temperature=temperature,
        )
        self.transposition = TranspositionOperator()
        self.apply_symmetrized_decoding = apply_symmetrized_decoding
        self.mask_negative_other = mask_negative_other
        self.decoder = SymmetrizedDecoder(mask_negative_other=mask_negative_other)
        self.training_mode = True
        self.training = True

    def train(self, mode: bool = True) -> "SiameseCRLModel":
        if HAS_TORCH and hasattr(super(), "train"):
            super().train(mode)
        self.training_mode = bool(mode)
        self.training = bool(mode)
        if hasattr(self.encoder, "train"):
            self.encoder.train(mode)
        return self

    def eval(self) -> "SiameseCRLModel":
        return self.train(False)

    def to(self, *args: Any, **kwargs: Any) -> "SiameseCRLModel":
        if HAS_TORCH and hasattr(super(), "to"):
            super().to(*args, **kwargs)
        if hasattr(self.encoder, "to"):
            self.encoder.to(*args, **kwargs)
        return self

    def _encode_single_direction(
        self,
        input_ids: Any,
        attention_mask: Any = None,
        token_type_ids: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Runs a single direction through the shared backbone encoder and extracts logits."""
        call_kwargs: Dict[str, Any] = {"input_ids": input_ids}
        if attention_mask is not None:
            call_kwargs["attention_mask"] = attention_mask
        if token_type_ids is not None:
            call_kwargs["token_type_ids"] = token_type_ids
        call_kwargs.update(kwargs)

        out = self.encoder(**call_kwargs)
        if hasattr(out, "logits"):
            return out.logits
        if isinstance(out, dict) and "logits" in out:
            return out["logits"]
        return out

    def forward(
        self,
        input_ids_dir: Any = None,
        attention_mask_dir: Any = None,
        token_type_ids_dir: Any = None,
        input_ids_rev: Any = None,
        attention_mask_rev: Any = None,
        token_type_ids_rev: Any = None,
        labels_dir: Any = None,
        labels_rev: Any = None,
        is_reversible: Any = None,
        epoch: Optional[float] = None,
        progress: Optional[float] = None,
        apply_symmetrized_decoding: Optional[bool] = None,
        mask_negative_other: Optional[bool] = None,
        batch: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> SiameseCRLOutput:
        """
        Dual-pass Siamese forward execution.
        Accepts either explicit directional arguments OR a unified dictionary `batch`
        produced by `PairedReversalDataset`.
        """
        if batch is not None:
            input_ids_dir = batch.get("input_ids_dir", batch.get("dir_input_ids", input_ids_dir))
            attention_mask_dir = batch.get("attention_mask_dir", batch.get("dir_attention_mask", attention_mask_dir))
            token_type_ids_dir = batch.get("token_type_ids_dir", batch.get("dir_token_type_ids", token_type_ids_dir))
            input_ids_rev = batch.get("input_ids_rev", batch.get("inv_input_ids", input_ids_rev))
            attention_mask_rev = batch.get("attention_mask_rev", batch.get("inv_attention_mask", attention_mask_rev))
            token_type_ids_rev = batch.get("token_type_ids_rev", batch.get("inv_token_type_ids", token_type_ids_rev))
            labels_dir = batch.get("labels_dir", batch.get("label_dir_id", batch.get("direct_label_id", labels_dir)))
            labels_rev = batch.get("labels_rev", batch.get("label_rev_id", batch.get("inverse_label_id", labels_rev)))
            is_reversible = batch.get("is_reversible", is_reversible)

        logits_dir = self._encode_single_direction(
            input_ids=input_ids_dir,
            attention_mask=attention_mask_dir,
            token_type_ids=token_type_ids_dir,
            **kwargs,
        )

        logits_rev = None
        if input_ids_rev is not None:
            logits_rev = self._encode_single_direction(
                input_ids=input_ids_rev,
                attention_mask=attention_mask_rev,
                token_type_ids=token_type_ids_rev,
                **kwargs,
            )

        epoch_or_prog = epoch if epoch is not None else progress

        loss = None
        metrics: Dict[str, Any] = {}
        if logits_rev is not None and (labels_dir is not None or self.training_mode):
            loss, metrics = self.loss_fn(
                logits_dir=logits_dir,
                logits_rev=logits_rev,
                labels_dir=labels_dir,
                labels_rev=labels_rev,
                epoch_or_progress=epoch_or_prog,
                is_reversible=is_reversible,
            )

        use_sd = (
            apply_symmetrized_decoding
            if apply_symmetrized_decoding is not None
            else self.apply_symmetrized_decoding
        )
        effective_mask = (
            self.mask_negative_other
            if mask_negative_other is None
            else mask_negative_other
        )

        use_torch = HAS_TORCH and torch is not None and isinstance(logits_dir, torch.Tensor)

        if use_torch:
            probs_dir = torch.softmax(logits_dir, dim=-1)
            probs_rev = torch.softmax(logits_rev, dim=-1) if logits_rev is not None else None
            if logits_rev is not None and use_sd:
                preds_dir, preds_rev, sym_probs_dir, sym_probs_rev = self.decoder.decode_batch(
                    probs_dir,
                    probs_rev,
                    return_distributions=True,
                    mask_negative_other=effective_mask,
                )
            else:
                sym_probs_dir = None
                sym_probs_rev = None
                preds_dir = torch.argmax(probs_dir, dim=-1)
                preds_rev = torch.argmax(probs_rev, dim=-1) if probs_rev is not None else None
        else:
            is_batch = isinstance(logits_dir[0], (list, tuple))
            if is_batch:
                probs_dir = [softmax_pure(row) for row in logits_dir]
                probs_rev = [softmax_pure(row) for row in logits_rev] if logits_rev is not None else None
                if probs_rev is not None and use_sd:
                    preds_dir, preds_rev, sym_probs_dir, sym_probs_rev = self.decoder.decode_batch(
                        probs_dir,
                        probs_rev,
                        return_distributions=True,
                        mask_negative_other=effective_mask,
                    )
                else:
                    sym_probs_dir = None
                    sym_probs_rev = None
                    preds_dir = [_argmax_with_rev_tie_break(row) for row in probs_dir]
                    preds_rev = (
                        [_argmax_with_rev_tie_break(row) for row in probs_rev]
                        if probs_rev is not None
                        else None
                    )
            else:
                probs_dir = softmax_pure(logits_dir)
                probs_rev = softmax_pure(logits_rev) if logits_rev is not None else None
                if probs_rev is not None and use_sd:
                    p_d, p_r, s_d, s_r = self.decoder.decode_pair(
                        probs_dir,
                        probs_rev,
                        return_distributions=True,
                        mask_negative_other=effective_mask,
                    )
                    preds_dir, preds_rev, sym_probs_dir, sym_probs_rev = p_d, p_r, s_d, s_r
                else:
                    sym_probs_dir = None
                    sym_probs_rev = None
                    preds_dir = _argmax_with_rev_tie_break(probs_dir)
                    preds_rev = _argmax_with_rev_tie_break(probs_rev) if probs_rev is not None else None

        return SiameseCRLOutput(
            loss=loss,
            logits_dir=logits_dir,
            logits_rev=logits_rev,
            probs_dir=probs_dir,
            probs_rev=probs_rev,
            sym_probs_dir=sym_probs_dir,
            sym_probs_rev=sym_probs_rev,
            preds_dir=preds_dir,
            preds_rev=preds_rev,
            metrics=metrics,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> SiameseCRLOutput:
        if HAS_TORCH and not self.is_mock and hasattr(super(), "__call__"):
            return super().__call__(*args, **kwargs)
        return self.forward(*args, **kwargs)

    def predict_logits(self, text1: str, text2: str) -> List[float]:
        """Computes raw 4-class logits for a single (text1, text2) pair via the shared encoder."""
        if hasattr(self.encoder, "predict_logits"):
            return list(self.encoder.predict_logits(text1, text2))
        if hasattr(self.encoder, "score_text_pair"):
            return list(self.encoder.score_text_pair(text1, text2))
        if hasattr(self.encoder, "predict_proba"):
            return [float(x) for x in self.encoder.predict_proba(text1, text2)]
        return [0.25, 0.25, 0.25, 0.25]

    def predict_proba(self, text1: str, text2: str) -> List[float]:
        """Computes normalized 4-class probability distribution for a single (text1, text2) pair."""
        if hasattr(self.encoder, "predict_proba"):
            return list(self.encoder.predict_proba(text1, text2))
        logits = self.predict_logits(text1, text2)
        return softmax_pure(logits)

    def predict(self, text1: str, text2: str) -> str:
        """Predicts the canonical string label for a single (text1, text2) pair."""
        probs = self.predict_proba(text1, text2)
        best_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        return ID_TO_LABEL[best_idx]

    def predict_pair(
        self,
        text1: str,
        text2: str,
        apply_symmetrized_decoding: Optional[bool] = None,
        mask_negative_other: Optional[bool] = None,
    ) -> Tuple[int, int]:
        """
        Predicts integer class indices `(pred_dir, pred_rev)` for a pair `(text1, text2)` and `(text2, text1)`.
        When `apply_symmetrized_decoding` is False (default for pure `crl`), returns independent argmax predictions.
        When `apply_symmetrized_decoding` is True (for `crl_sd`), returns symmetrized predictions.
        """
        use_sd = (
            self.apply_symmetrized_decoding
            if apply_symmetrized_decoding is None
            else apply_symmetrized_decoding
        )
        effective_mask = (
            self.mask_negative_other
            if mask_negative_other is None
            else mask_negative_other
        )
        p_dir = self.predict_proba(text1, text2)
        p_rev = self.predict_proba(text2, text1)

        if use_sd:
            return self.decoder.decode_pair(
                p_dir,
                p_rev,
                mask_negative_other=effective_mask,
            )

        pred_dir = int(max(range(len(p_dir)), key=lambda i: p_dir[i]))
        pred_rev = int(max(range(len(p_rev)), key=lambda i: p_rev[i]))
        return pred_dir, pred_rev

    def decode_pair(
        self,
        p_dir: Any,
        p_rev: Any,
        mask_negative_other: Optional[bool] = None,
    ) -> Tuple[int, int]:
        """Convenience method delegating single-pair decoding to the internal SymmetrizedDecoder."""
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
        """Convenience method delegating batch pair decoding to the internal SymmetrizedDecoder."""
        return self.decoder.decode_pairs(
            probs_dir_list,
            probs_rev_list,
            mask_negative_other=mask_negative_other,
        )
