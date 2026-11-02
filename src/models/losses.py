"""
src/models/losses.py

Loss functions and transposition operators for DiCo-NLI (SemEval-2027 Task 2).

Includes:
- Involution permutation matrix P and TranspositionOperator.
- Standard Categorical Cross-Entropy Loss with label smoothing and ignore_index=-100.
- Symmetric KL Divergence Loss.
- Jensen-Shannon Divergence (JSD) Loss.
- Consistency Regularization Loss (CRL) with dynamic warmup annealing.
- Pure-Python fallback routines for zero-dependency execution.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

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

# ============================================================================
# 1. Permutation Constants & Involution Algebra
# ============================================================================

# Label ordering matching model heads and metrics:
# 0: BACKWARD_ENTAILMENT, 1: EQUIVALENCE, 2: FORWARD_ENTAILMENT, 3: NEGATIVE_OTHER
P_MATRIX_LIST: List[List[float]] = [
    [0.0, 0.0, 1.0, 0.0],  # 0: BWD -> 2: FWD
    [0.0, 1.0, 0.0, 0.0],  # 1: EQUIV -> 1: EQUIV
    [1.0, 0.0, 0.0, 0.0],  # 2: FWD -> 0: BWD
    [0.0, 0.0, 0.0, 1.0],  # 3: NEG -> 3: NEG
]
P_RAW_MATRIX: List[List[float]] = P_MATRIX_LIST

REV_INDEX_MAP: Dict[int, int] = {0: 2, 1: 1, 2: 0, 3: 3}
REVERSIBLE_INDICES: Set[int] = {0, 1, 2}

if HAS_TORCH and torch is not None:
    P_MATRIX = torch.tensor(P_MATRIX_LIST, dtype=torch.float32)
else:
    P_MATRIX = P_MATRIX_LIST  # type: ignore


# ============================================================================
# 2. Pure-Python Numerical Fallback Helpers
# ============================================================================

def matmul_vec_mat(vec: Sequence[float], mat: Sequence[Sequence[float]]) -> List[float]:
    """Row vector times matrix: out = vec @ mat."""
    out = [0.0] * len(mat[0])
    for j in range(len(mat[0])):
        out[j] = sum(vec[i] * mat[i][j] for i in range(len(vec)))
    return out


def matmul_mat_mat(
    m1: Sequence[Sequence[float]],
    m2: Sequence[Sequence[float]],
) -> List[List[float]]:
    """Matrix multiplication: out = m1 @ m2."""
    rows = len(m1)
    cols = len(m2[0])
    inner = len(m2)
    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            out[i][j] = sum(m1[i][k] * m2[k][j] for k in range(inner))
    return out


def softmax_pure(logits: Sequence[float]) -> List[float]:
    """Numerically stable pure-Python softmax."""
    max_l = max(logits)
    exps = [math.exp(x - max_l) for x in logits]
    s = sum(exps)
    return [e / s if s > 0 else 0.25 for e in exps]


def cross_entropy_pure(
    logits: Union[Sequence[float], Sequence[Sequence[float]]],
    target: Union[int, Sequence[int]],
    ignore_index: int = -100,
    reduction: str = "mean",
) -> float:
    """Pure-Python Cross-Entropy Loss with PyTorch ignore_index parity."""
    if not logits:
        return 0.0

    # 1D single sample case
    if isinstance(logits[0], (int, float)):
        if isinstance(target, (list, tuple)):
            if len(target) == 1:
                tgt = int(target[0])
            else:
                raise ValueError("Shape mismatch: 1D logits with multi-element target.")
        else:
            tgt = int(target)

        if tgt == ignore_index or tgt < 0 or tgt >= len(logits):
            return 0.0
        probs = softmax_pure(logits)  # type: ignore[arg-type]
        p = max(probs[tgt], 1e-15)
        return -math.log(p)

    # 2D batch of samples case
    batch_targets: Sequence[int]
    if isinstance(target, (int, float)):
        batch_targets = [int(target)] * len(logits)
    else:
        batch_targets = target

    total_loss = 0.0
    non_ignored_count = 0

    for row_logits, row_tgt in zip(logits, batch_targets):
        if row_tgt == ignore_index or row_tgt < 0 or row_tgt >= len(row_logits):  # type: ignore[arg-type]
            continue
        probs = softmax_pure(row_logits)  # type: ignore[arg-type]
        p = max(probs[row_tgt], 1e-15)
        total_loss += -math.log(p)
        non_ignored_count += 1

    if reduction == "sum":
        return total_loss
    return (total_loss / non_ignored_count) if non_ignored_count > 0 else 0.0


def kl_divergence_pure(p: Sequence[float], q: Sequence[float], eps: float = 1e-12) -> float:
    """Pure-Python KL divergence: D_KL(P || Q) = sum p * log(p / q)."""
    total = 0.0
    for p_c, q_c in zip(p, q):
        p_safe = max(float(p_c), eps)
        q_safe = max(float(q_c), eps)
        total += p_safe * math.log(p_safe / q_safe)
    return total


def symmetric_kl_pure(
    p: Sequence[float],
    q: Sequence[float],
    scale: float = 0.5,
    eps: float = 1e-12,
) -> float:
    """Pure-Python Symmetric KL divergence: scale * (KL(P||Q) + KL(Q||P))."""
    return scale * (kl_divergence_pure(p, q, eps) + kl_divergence_pure(q, p, eps))


def jensen_shannon_pure(
    p: Sequence[float],
    q: Sequence[float],
    eps: float = 1e-12,
) -> float:
    """Pure-Python JSD: 0.5 * KL(P||M) + 0.5 * KL(Q||M), M = 0.5*(P+Q)."""
    m = [0.5 * (float(p_c) + float(q_c)) for p_c, q_c in zip(p, q)]
    return 0.5 * (kl_divergence_pure(p, m, eps) + kl_divergence_pure(q, m, eps))


def consistency_regularization_loss_pure(
    logits_dir: Sequence[float],
    logits_rev: Sequence[float],
    target_dir: int,
    target_rev: Optional[int] = None,
    divergence: str = "sym_kl",
    lambda_cons: float = 1.0,
    warmup_epochs: float = 1.0,
    current_epoch: float = 1.0,
) -> Tuple[float, Dict[str, Any]]:
    """Pure-Python Consistency Regularization Loss for a single pair."""
    if target_rev is None:
        target_rev = REV_INDEX_MAP.get(target_dir, target_dir)

    loss_ce_dir = cross_entropy_pure(logits_dir, target_dir, ignore_index=-100)
    loss_ce_rev = cross_entropy_pure(logits_rev, target_rev, ignore_index=-100)

    valid_task = 0
    task_sum = 0.0
    if target_dir != -100 and 0 <= target_dir < len(logits_dir):
        task_sum += loss_ce_dir
        valid_task += 1
    if target_rev != -100 and 0 <= target_rev < len(logits_rev):
        task_sum += loss_ce_rev
        valid_task += 1
    loss_task = (task_sum / valid_task) if valid_task > 0 else 0.0

    p_dir = softmax_pure(logits_dir)
    p_rev = softmax_pure(logits_rev)
    p_rev_trans = matmul_vec_mat(p_rev, P_MATRIX_LIST)

    div_name = divergence.lower()
    if div_name in ("sym_kl", "symmetric_kl", "kl"):
        loss_cons = symmetric_kl_pure(p_dir, p_rev_trans, scale=0.5)
    elif div_name in ("jsd", "jensen_shannon", "js"):
        loss_cons = jensen_shannon_pure(p_dir, p_rev_trans)
    else:
        raise ValueError(f"Unknown divergence: {divergence}")

    if warmup_epochs > 0:
        effective_lambda = lambda_cons * min(1.0, float(current_epoch) / float(warmup_epochs))
    else:
        effective_lambda = lambda_cons

    total_loss = loss_task + effective_lambda * loss_cons
    metrics: Dict[str, Any] = {
        "loss_total": total_loss,
        "loss_task": loss_task,
        "loss_ce_dir": loss_ce_dir,
        "loss_ce_rev": loss_ce_rev,
        "loss_cons": loss_cons,
        "effective_lambda": effective_lambda,
        "divergence": div_name,
    }
    return total_loss, metrics


# ============================================================================
# 3. Transposition Operator & Divergence Loss Modules
# ============================================================================

if HAS_TORCH and nn is not None:
    _LossModuleBase = nn.Module
else:
    class _LossModuleBase:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.forward(*args, **kwargs)


class TranspositionOperator(_LossModuleBase):
    """
    Permutation operator mapping reverse distribution vectors to the direct reference frame.
    Computes p_trans = p @ P.
    """

    def __init__(self, p_matrix: Optional[Any] = None) -> None:
        super().__init__()
        if HAS_TORCH and torch is not None:
            if p_matrix is None:
                p_tensor = torch.tensor(P_MATRIX_LIST, dtype=torch.float32)
            elif isinstance(p_matrix, torch.Tensor):
                p_tensor = p_matrix.clone().detach().to(dtype=torch.float32)
            else:
                p_tensor = torch.tensor(p_matrix, dtype=torch.float32)
            self.register_buffer("P", p_tensor)
        else:
            self.P = P_MATRIX_LIST

    def forward(self, probs: Any) -> Any:
        if HAS_TORCH and torch is not None and isinstance(probs, torch.Tensor):
            P_dev = self.P.to(device=probs.device, dtype=probs.dtype)
            return torch.matmul(probs, P_dev)
        if not probs:
            return []
        if isinstance(probs[0], (int, float)):
            return matmul_vec_mat(probs, P_MATRIX_LIST)
        return [matmul_vec_mat(row, P_MATRIX_LIST) for row in probs]

    def extra_repr(self) -> str:
        return "shape=(4, 4), involution=True"


class SymmetricKLDivergenceLoss(_LossModuleBase):
    """
    Symmetric Kullback-Leibler Divergence Loss:
    scale * (D_KL(P || Q) + D_KL(Q || P))
    """

    def __init__(
        self,
        scale: float = 0.5,
        eps: float = 1e-8,
        reduction: str = "batchmean",
    ) -> None:
        super().__init__()
        self.scale = scale
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        p: Any,
        q: Any,
        from_logits: bool = False,
    ) -> Any:
        if HAS_TORCH and torch is not None and isinstance(p, torch.Tensor):
            if not isinstance(q, torch.Tensor):
                q = torch.tensor(q, device=p.device, dtype=p.dtype)
            is_1d = p.dim() == 1
            p_in = p.unsqueeze(0) if is_1d else p
            q_in = q.unsqueeze(0) if is_1d else q
            if from_logits:
                log_p = F.log_softmax(p_in, dim=-1)
                log_q = F.log_softmax(q_in, dim=-1)
                p_prob = log_p.exp()
                q_prob = log_q.exp()
            else:
                p_prob = torch.clamp(p_in, min=self.eps)
                q_prob = torch.clamp(q_in, min=self.eps)
                p_prob = p_prob / p_prob.sum(dim=-1, keepdim=True)
                q_prob = q_prob / q_prob.sum(dim=-1, keepdim=True)
                log_p = torch.log(p_prob)
                log_q = torch.log(q_prob)

            kl_pq = F.kl_div(log_q, p_prob, reduction=self.reduction)
            kl_qp = F.kl_div(log_p, q_prob, reduction=self.reduction)
            return self.scale * (kl_pq + kl_qp)

        # Pure-Python sequence fallback
        if isinstance(p[0], (int, float)):
            p_vec = softmax_pure(p) if from_logits else p
            q_vec = softmax_pure(q) if from_logits else q
            return symmetric_kl_pure(p_vec, q_vec, scale=self.scale, eps=self.eps)
        total = 0.0
        n = max(1, len(p))
        for p_row, q_row in zip(p, q):
            p_vec = softmax_pure(p_row) if from_logits else p_row
            q_vec = softmax_pure(q_row) if from_logits else q_row
            total += symmetric_kl_pure(p_vec, q_vec, scale=self.scale, eps=self.eps)
        return total / n


class JensenShannonDivergenceLoss(_LossModuleBase):
    """
    Jensen-Shannon Divergence (JSD):
    0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M) where M = 0.5*(P+Q).
    """

    def __init__(
        self,
        eps: float = 1e-8,
        reduction: str = "batchmean",
    ) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        p: Any,
        q: Any,
        from_logits: bool = False,
    ) -> Any:
        if HAS_TORCH and torch is not None and isinstance(p, torch.Tensor):
            if not isinstance(q, torch.Tensor):
                q = torch.tensor(q, device=p.device, dtype=p.dtype)
            is_1d = p.dim() == 1
            p_in = p.unsqueeze(0) if is_1d else p
            q_in = q.unsqueeze(0) if is_1d else q
            if from_logits:
                p_prob = F.softmax(p_in, dim=-1)
                q_prob = F.softmax(q_in, dim=-1)
            else:
                p_prob = torch.clamp(p_in, min=self.eps)
                q_prob = torch.clamp(q_in, min=self.eps)
                p_prob = p_prob / p_prob.sum(dim=-1, keepdim=True)
                q_prob = q_prob / q_prob.sum(dim=-1, keepdim=True)

            m = 0.5 * (p_prob + q_prob)
            m = torch.clamp(m, min=self.eps)
            log_m = torch.log(m)

            kl_pm = F.kl_div(log_m, p_prob, reduction=self.reduction)
            kl_qm = F.kl_div(log_m, q_prob, reduction=self.reduction)
            return 0.5 * (kl_pm + kl_qm)

        # Pure-Python sequence fallback
        if isinstance(p[0], (int, float)):
            p_vec = softmax_pure(p) if from_logits else p
            q_vec = softmax_pure(q) if from_logits else q
            return jensen_shannon_pure(p_vec, q_vec, eps=self.eps)
        total = 0.0
        n = max(1, len(p))
        for p_row, q_row in zip(p, q):
            p_vec = softmax_pure(p_row) if from_logits else p_row
            q_vec = softmax_pure(q_row) if from_logits else q_row
            total += jensen_shannon_pure(p_vec, q_vec, eps=self.eps)
        return total / n


class ConsistencyRegularizationLoss(_LossModuleBase):
    """
    Combined Supervised Task Loss and Directional Consistency Regularization.
    Loss = task_weight * (CE(z_dir, y_dir) + CE(z_rev, y_rev)) + lambda(t) * Div(p_dir, p_rev @ P)
    """

    def __init__(
        self,
        divergence: str = "sym_kl",
        lambda_cons: float = 1.0,
        warmup_epochs: int = 1,
        total_epochs: int = 5,
        task_weight: float = 0.5,
        temperature: float = 1.0,
        eps: float = 1e-8,
        label_smoothing: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.lambda_cons = lambda_cons
        self.warmup_epochs = max(1, warmup_epochs)
        self.total_epochs = max(self.warmup_epochs, total_epochs)
        self.task_weight = task_weight
        self.temperature = temperature
        self.eps = eps
        self.label_smoothing = label_smoothing
        self.transposition = TranspositionOperator()

        if HAS_TORCH and nn is not None:
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing)
        else:
            self.ce_loss = None

        self.divergence_name = divergence.lower()
        if self.divergence_name in ("sym_kl", "symmetric_kl", "kl"):
            self.divergence_name = "sym_kl"
            self.div_fn: Any = SymmetricKLDivergenceLoss(scale=0.5, eps=eps)
        elif self.divergence_name in ("jsd", "jensen_shannon", "js"):
            self.divergence_name = "jsd"
            self.div_fn = JensenShannonDivergenceLoss(eps=eps)
        else:
            raise ValueError(f"Unknown divergence: {divergence}. Use 'sym_kl' or 'jsd'.")

    def get_effective_lambda(self, progress: Optional[float] = 1.0) -> float:
        """Compute dynamic lambda(t) via linear warmup."""
        if progress is None:
            return self.lambda_cons
        if self.warmup_epochs <= 0:
            return self.lambda_cons
        factor = min(1.0, max(0.0, float(progress) / float(self.warmup_epochs)))
        return self.lambda_cons * factor

    def forward(
        self,
        logits_dir: Any,
        logits_rev: Any,
        labels_dir: Optional[Any] = None,
        labels_rev: Optional[Any] = None,
        epoch_or_progress: Optional[float] = 1.0,
        is_reversible: Optional[Any] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Compute total CRL loss and diagnostic dictionary."""
        if HAS_TORCH and torch is not None and isinstance(logits_dir, torch.Tensor):
            if not isinstance(logits_rev, torch.Tensor):
                logits_rev = torch.tensor(logits_rev, device=logits_dir.device, dtype=logits_dir.dtype)

            if labels_dir is not None and not isinstance(labels_dir, torch.Tensor):
                labels_dir = torch.tensor(labels_dir, device=logits_dir.device, dtype=torch.long)
            if labels_rev is not None and not isinstance(labels_rev, torch.Tensor):
                labels_rev = torch.tensor(labels_rev, device=logits_dir.device, dtype=torch.long)

            if labels_dir is not None and labels_rev is None:
                rev_tensor = torch.tensor([2, 1, 0, 3], device=labels_dir.device, dtype=labels_dir.dtype)
                mask_valid = (labels_dir >= 0) & (labels_dir <= 3)
                labels_rev = labels_dir.clone()
                labels_rev[mask_valid] = rev_tensor[labels_dir[mask_valid]]

            if labels_dir is not None and self.ce_loss is not None:
                valid_d = (labels_dir >= 0) & (labels_dir < logits_dir.shape[-1])
                valid_r = (labels_rev >= 0) & (labels_rev < logits_rev.shape[-1]) if labels_rev is not None else torch.zeros_like(valid_d)

                loss_ce_dir = self.ce_loss(logits_dir[valid_d], labels_dir[valid_d]) if valid_d.any() else torch.tensor(0.0, device=logits_dir.device, dtype=logits_dir.dtype)
                loss_ce_rev = self.ce_loss(logits_rev[valid_r], labels_rev[valid_r]) if (labels_rev is not None and valid_r.any()) else torch.tensor(0.0, device=logits_dir.device, dtype=logits_dir.dtype)
                loss_task = self.task_weight * (loss_ce_dir + loss_ce_rev)
            else:
                loss_ce_dir = torch.tensor(0.0, device=logits_dir.device, dtype=logits_dir.dtype)
                loss_ce_rev = torch.tensor(0.0, device=logits_dir.device, dtype=logits_dir.dtype)
                loss_task = torch.tensor(0.0, device=logits_dir.device, dtype=logits_dir.dtype)

            p_dir = F.softmax(logits_dir, dim=-1)
            p_rev = F.softmax(logits_rev, dim=-1)
            p_rev_trans = self.transposition(p_rev)
            loss_cons = self.div_fn(p_dir, p_rev_trans, from_logits=False)

            effective_lambda = self.get_effective_lambda(epoch_or_progress)
            total_loss = loss_task + effective_lambda * loss_cons

            metrics: Dict[str, Any] = {
                "loss_total": float(total_loss.detach().item()),
                "loss_task": float(loss_task.detach().item()),
                "loss_ce_dir": float(loss_ce_dir.detach().item()),
                "loss_ce_rev": float(loss_ce_rev.detach().item()),
                "loss_cons": float(loss_cons.detach().item()),
                "effective_lambda": float(effective_lambda),
                "divergence": self.divergence_name,
            }
            return total_loss, metrics

        # Pure-Python sequence fallback
        ld = [logits_dir] if (logits_dir and isinstance(logits_dir[0], (int, float))) else list(logits_dir)
        lr = [logits_rev] if (logits_rev and isinstance(logits_rev[0], (int, float))) else list(logits_rev)
        b = max(1, len(ld))

        if labels_dir is None:
            yd = [-100] * b
        elif isinstance(labels_dir, (int, float)):
            yd = [int(labels_dir)] * b
        else:
            yd = [int(y) for y in labels_dir]

        if labels_rev is None:
            yr = [REV_INDEX_MAP.get(y, y) for y in yd]
        elif isinstance(labels_rev, (int, float)):
            yr = [int(labels_rev)] * b
        else:
            yr = [int(y) for y in labels_rev]

        total_ce_dir = 0.0
        total_ce_rev = 0.0
        valid_d = 0
        valid_r = 0
        total_cons = 0.0

        for i in range(b):
            if yd[i] != -100 and 0 <= yd[i] < len(ld[i]):
                total_ce_dir += cross_entropy_pure(ld[i], yd[i], ignore_index=-100)
                valid_d += 1
            if yr[i] != -100 and 0 <= yr[i] < len(lr[i]):
                total_ce_rev += cross_entropy_pure(lr[i], yr[i], ignore_index=-100)
                valid_r += 1

            pd = softmax_pure(ld[i])
            pr = softmax_pure(lr[i])
            pr_trans = self.transposition(pr)
            total_cons += float(self.div_fn(pd, pr_trans))

        loss_ce_dir_val = (total_ce_dir / valid_d) if valid_d > 0 else 0.0
        loss_ce_rev_val = (total_ce_rev / valid_r) if valid_r > 0 else 0.0
        loss_task_val = self.task_weight * (loss_ce_dir_val + loss_ce_rev_val)
        loss_cons_val = total_cons / b
        effective_lambda = self.get_effective_lambda(epoch_or_progress)
        total_loss_val = loss_task_val + effective_lambda * loss_cons_val

        metrics_pure: Dict[str, Any] = {
            "loss_total": total_loss_val,
            "loss_task": loss_task_val,
            "loss_ce_dir": loss_ce_dir_val,
            "loss_ce_rev": loss_ce_rev_val,
            "loss_cons": loss_cons_val,
            "effective_lambda": float(effective_lambda),
            "divergence": self.divergence_name,
        }
        return total_loss_val, metrics_pure
