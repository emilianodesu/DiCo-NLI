"""
src/models/decoding.py

Post-hoc Symmetrized Decoding (SD) Engine for SemEval-2027 Task 2 (DiCo-NLI).
Guarantees 100% SoftCons (SoftCons == 1.000000) on all reversible phrase pairs.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


# =============================================================================
# 1. Constants & Permutation Matrix
# =============================================================================

# Label Index Encoding:
# 0: BACKWARD_ENTAILMENT
# 1: EQUIVALENCE
# 2: FORWARD_ENTAILMENT
# 3: NEGATIVE_OTHER

REV_INDEX_MAP: Dict[int, int] = {0: 2, 1: 1, 2: 0, 3: 3}
REVERSIBLE_INDICES: Set[int] = {0, 1, 2}

P_RAW_MATRIX: List[List[float]] = [
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


class InvolutoryPermutationMatrix:
    """Wrapper exposing indexing [i][j], .numpy(), and torch/numpy math operations."""

    def __init__(self, raw: List[List[float]]) -> None:
        self._raw = raw
        self._np = np.array(raw, dtype=np.float64) if HAS_NUMPY and np is not None else None
        self._torch = torch.tensor(raw, dtype=torch.float32) if HAS_TORCH and torch is not None else None

    def __getitem__(self, idx: int) -> Sequence[float]:
        return self._raw[idx]

    def __len__(self) -> int:
        return 4

    def __iter__(self):
        return iter(self._raw)

    def numpy(self) -> Any:
        if self._np is not None:
            return self._np
        if HAS_NUMPY and np is not None:
            self._np = np.array(self._raw, dtype=np.float64)
            return self._np
        return self._raw

    def as_torch(self, device: Optional[Any] = None, dtype: Optional[Any] = None) -> Any:
        if not HAS_TORCH or torch is None:
            raise RuntimeError("PyTorch is not available.")
        t = torch.tensor(self._raw, dtype=dtype or torch.float32)
        if device is not None:
            t = t.to(device)
        return t


P_MATRIX = InvolutoryPermutationMatrix(P_RAW_MATRIX)


# =============================================================================
# 2. Pure-Python Fallback Math & Symmetric Tie-Breaking
# =============================================================================

def _matmul_1d_4x4(vec: Sequence[float], mat: Sequence[Sequence[float]]) -> List[float]:
    """Multiply 1x4 row vector by 4x4 matrix: out = vec @ mat."""
    out = [0.0] * 4
    for j in range(4):
        out[j] = sum(vec[i] * mat[i][j] for i in range(4))
    return out


def _argmax_with_rev_tie_break(p: Sequence[float], eps: float = 1e-9) -> int:
    """Deterministic argmax with symmetric tie-breaking compatible with Rev involution.

    Tie-breaking semantics:
    - If index 0 (BACKWARD) and index 2 (FORWARD) tie for maximum, the joint consensus
      is symmetric: EQUIVALENCE (index 1).
    - If index 1 ties with any other candidate, select 1 (self-inverse under Rev).
    - If index 3 ties with non-reversible classes, select 3 (self-inverse under Rev).
    - Unconditionally preserves: argmax(p @ P) == Rev(argmax(p)).
    """
    max_val = max(p)
    candidates = [i for i, v in enumerate(p) if abs(v - max_val) < eps]
    if len(candidates) == 1:
        return candidates[0]
    if 0 in candidates and 2 in candidates:
        return 1
    if 1 in candidates:
        return 1
    if 3 in candidates:
        return 3
    return candidates[0]


# =============================================================================
# 3. Core Symmetrized Decoding Functions
# =============================================================================

def symmetrized_decoding(
    probs_dir: Any,
    probs_rev: Any,
    mask_negative_other: bool = False,
) -> Tuple[Any, Any]:
    """Projects directional probability distributions into symmetrized space.

    Computes:
        p_sym_dir = 0.5 * (p_dir + p_rev @ P)
        p_sym_rev = 0.5 * (p_rev + p_dir @ P)

    Args:
        probs_dir: 1D sequence (4,) or 2D array/tensor (N, 4) for direct pairs.
        probs_rev: 1D sequence (4,) or 2D array/tensor (N, 4) for reversed pairs.
        mask_negative_other: If True, zeroes index 3 (NEGATIVE_OTHER) and renormalizes
            over reversible indices {0, 1, 2}.

    Returns:
        Tuple of (p_sym_dir, p_sym_rev) with same type and shape as inputs.
    """
    # 1. PyTorch Tensor Support
    if HAS_TORCH and torch is not None and isinstance(probs_dir, torch.Tensor):
        is_1d = probs_dir.dim() == 1
        p_dir = probs_dir.unsqueeze(0) if is_1d else probs_dir
        p_rev = probs_rev.unsqueeze(0) if is_1d else probs_rev

        P = P_MATRIX.as_torch(device=p_dir.device, dtype=p_dir.dtype)
        p_sym_dir = 0.5 * (p_dir + torch.matmul(p_rev, P))
        p_sym_rev = 0.5 * (p_rev + torch.matmul(p_dir, P))

        if mask_negative_other:
            p_sym_dir = p_sym_dir.clone()
            p_sym_rev = p_sym_rev.clone()
            p_sym_dir[:, 3] = 0.0
            p_sym_rev[:, 3] = 0.0
            sum_d = p_sym_dir.sum(dim=-1, keepdim=True)
            sum_r = p_sym_rev.sum(dim=-1, keepdim=True)
            p_sym_dir = torch.where(sum_d > 0, p_sym_dir / sum_d.clamp(min=1e-300), p_sym_dir)
            p_sym_rev = torch.where(sum_r > 0, p_sym_rev / sum_r.clamp(min=1e-300), p_sym_rev)

        if is_1d:
            return p_sym_dir.squeeze(0), p_sym_rev.squeeze(0)
        return p_sym_dir, p_sym_rev

    # 2. NumPy Array Support
    if HAS_NUMPY and np is not None and isinstance(probs_dir, np.ndarray):
        is_1d = probs_dir.ndim == 1
        p_dir = np.expand_dims(probs_dir, axis=0) if is_1d else probs_dir
        p_rev = np.expand_dims(probs_rev, axis=0) if is_1d else probs_rev

        P = P_MATRIX.numpy()
        p_sym_dir = 0.5 * (p_dir + np.matmul(p_rev, P))
        p_sym_rev = 0.5 * (p_rev + np.matmul(p_dir, P))

        if mask_negative_other:
            p_sym_dir = p_sym_dir.copy()
            p_sym_rev = p_sym_rev.copy()
            p_sym_dir[:, 3] = 0.0
            p_sym_rev[:, 3] = 0.0
            sum_d = np.sum(p_sym_dir, axis=-1, keepdims=True)
            sum_r = np.sum(p_sym_rev, axis=-1, keepdims=True)
            sum_d[sum_d == 0] = 1.0
            sum_r[sum_r == 0] = 1.0
            p_sym_dir /= sum_d
            p_sym_rev /= sum_r

        if is_1d:
            return p_sym_dir[0], p_sym_rev[0]
        return p_sym_dir, p_sym_rev

    # 3. Pure-Python Sequence Support
    if isinstance(probs_dir[0], (int, float)):
        p_rev_P = _matmul_1d_4x4(probs_rev, P_RAW_MATRIX)
        p_dir_P = _matmul_1d_4x4(probs_dir, P_RAW_MATRIX)
        p_sym_dir = [0.5 * (probs_dir[i] + p_rev_P[i]) for i in range(4)]
        p_sym_rev = [0.5 * (probs_rev[i] + p_dir_P[i]) for i in range(4)]

        if mask_negative_other:
            p_sym_dir[3] = 0.0
            p_sym_rev[3] = 0.0
            sd = sum(p_sym_dir)
            sr = sum(p_sym_rev)
            if sd > 0:
                p_sym_dir = [v / sd for v in p_sym_dir]
            if sr > 0:
                p_sym_rev = [v / sr for v in p_sym_rev]
        return p_sym_dir, p_sym_rev
    else:
        res_d = []
        res_r = []
        for pd_row, pr_row in zip(probs_dir, probs_rev):
            sd_row, sr_row = symmetrized_decoding(pd_row, pr_row, mask_negative_other=mask_negative_other)
            res_d.append(sd_row)
            res_r.append(sr_row)
        return res_d, res_r


# =============================================================================
# 4. SymmetrizedDecoder Class
# =============================================================================

class SymmetrizedDecoder:
    """Post-hoc probability projection engine strictly guaranteeing SoftCons = 100%."""

    def __init__(self, mask_negative_other: bool = False, tie_break: str = "symmetric") -> None:
        self.mask_negative_other = mask_negative_other
        self.tie_break = tie_break

    def project_probabilities(
        self,
        probs_dir: Any,
        probs_rev: Any,
        mask_negative_other: Optional[bool] = None,
    ) -> Tuple[Any, Any]:
        """Project raw direct and reverse probabilities into symmetrized space."""
        mask = self.mask_negative_other if mask_negative_other is None else mask_negative_other
        return symmetrized_decoding(probs_dir, probs_rev, mask_negative_other=mask)

    def decode_pair(
        self,
        p_dir: Sequence[float],
        p_rev: Sequence[float],
        mask_negative_other: Optional[bool] = None,
        return_distributions: bool = False,
    ) -> Any:
        """Decode single pair into discrete class indices."""
        p_s_dir, p_s_rev = self.project_probabilities(p_dir, p_rev, mask_negative_other=mask_negative_other)
        if hasattr(p_s_dir, "tolist"):
            p_s_dir = p_s_dir.tolist()
        if hasattr(p_s_rev, "tolist"):
            p_s_rev = p_s_rev.tolist()
        pred_dir = _argmax_with_rev_tie_break(p_s_dir)
        pred_rev = _argmax_with_rev_tie_break(p_s_rev)
        if return_distributions:
            return pred_dir, pred_rev, p_s_dir, p_s_rev
        return pred_dir, pred_rev

    def decode_pairs(
        self,
        probs_dir: Any,
        probs_rev: Any,
        mask_negative_other: Optional[bool] = None,
        return_distributions: bool = False,
    ) -> Any:
        """Decode batch of direct and reverse probability arrays into discrete indices."""
        p_s_dir, p_s_rev = self.project_probabilities(probs_dir, probs_rev, mask_negative_other=mask_negative_other)

        if HAS_NUMPY and np is not None and isinstance(p_s_dir, np.ndarray):
            preds_dir = np.array([_argmax_with_rev_tie_break(row) for row in p_s_dir], dtype=np.int64)
            preds_rev = np.array([_argmax_with_rev_tie_break(row) for row in p_s_rev], dtype=np.int64)
        elif HAS_TORCH and torch is not None and isinstance(p_s_dir, torch.Tensor):
            preds_dir = torch.tensor(
                [_argmax_with_rev_tie_break(row.tolist()) for row in p_s_dir],
                device=p_s_dir.device,
                dtype=torch.long,
            )
            preds_rev = torch.tensor(
                [_argmax_with_rev_tie_break(row.tolist()) for row in p_s_rev],
                device=p_s_rev.device,
                dtype=torch.long,
            )
        else:
            preds_dir = [_argmax_with_rev_tie_break(row) for row in p_s_dir]
            preds_rev = [_argmax_with_rev_tie_break(row) for row in p_s_rev]

        if return_distributions:
            return preds_dir, preds_rev, p_s_dir, p_s_rev
        return preds_dir, preds_rev

    def decode_batch(
        self,
        probs_dir: Any,
        probs_rev: Any,
        mask_negative_other: Optional[bool] = None,
        return_distributions: bool = False,
    ) -> Any:
        """Alias for decode_pairs."""
        return self.decode_pairs(
            probs_dir,
            probs_rev,
            mask_negative_other=mask_negative_other,
            return_distributions=return_distributions,
        )

    def decode(
        self,
        probs_dir: Any,
        probs_rev: Any,
        mask_negative_other: Optional[bool] = None,
    ) -> Tuple[Any, Any]:
        """Smart dispatcher: auto-detects 1D vs 2D input with optional masking override."""
        if HAS_NUMPY and np is not None and isinstance(probs_dir, np.ndarray):
            if probs_dir.ndim == 1:
                return self.decode_pair(probs_dir, probs_rev, mask_negative_other=mask_negative_other)
            return self.decode_pairs(probs_dir, probs_rev, mask_negative_other=mask_negative_other)
        if HAS_TORCH and torch is not None and isinstance(probs_dir, torch.Tensor):
            if probs_dir.dim() == 1:
                return self.decode_pair(probs_dir.tolist(), probs_rev.tolist(), mask_negative_other=mask_negative_other)
            return self.decode_pairs(probs_dir, probs_rev, mask_negative_other=mask_negative_other)
        if not probs_dir:
            return (0, 0)
        if isinstance(probs_dir[0], (int, float)):
            return self.decode_pair(probs_dir, probs_rev, mask_negative_other=mask_negative_other)
        return self.decode_pairs(probs_dir, probs_rev, mask_negative_other=mask_negative_other)
