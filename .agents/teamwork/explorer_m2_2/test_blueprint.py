"""
Pure-Python + Mathematical Verification script for M2 Explorer 2 blueprint.
Validates exact formulas, involution matrix properties, KL, JSD, CRL loss,
and Symmetrized Decoding on standard Python 3 runtime without external dependencies.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

# 1. Permutation matrix P
# Ordering: 0: BACKWARD, 1: EQUIVALENCE, 2: FORWARD, 3: NEGATIVE_OTHER
P_MATRIX = [
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]

REV_MAP = {0: 2, 1: 1, 2: 0, 3: 3}


def matmul_vec_mat(vec: Sequence[float], mat: Sequence[Sequence[float]]) -> List[float]:
    """Row vector times matrix: out = vec @ mat."""
    out = [0.0] * len(mat[0])
    for j in range(len(mat[0])):
        out[j] = sum(vec[i] * mat[i][j] for i in range(len(vec)))
    return out


def matmul_mat_mat(m1: Sequence[Sequence[float]], m2: Sequence[Sequence[float]]) -> List[List[float]]:
    """Matrix multiplication: out = m1 @ m2."""
    rows = len(m1)
    cols = len(m2[0])
    inner = len(m2)
    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            out[i][j] = sum(m1[i][k] * m2[k][j] for k in range(inner))
    return out


def softmax(logits: Sequence[float]) -> List[float]:
    """Numerically stable softmax."""
    max_l = max(logits)
    exps = [math.exp(x - max_l) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def cross_entropy_loss(logits: Sequence[float], target_class: int) -> float:
    """Standard Categorical Cross-Entropy Loss."""
    probs = softmax(logits)
    p_c = max(probs[target_class], 1e-15)
    return -math.log(p_c)


def kl_divergence(p: Sequence[float], q: Sequence[float], eps: float = 1e-12) -> float:
    """
    Kullback-Leibler Divergence D_KL(P || Q) = sum_c p_c * log(p_c / q_c).
    Conforms to mathematical definition.
    """
    total = 0.0
    for p_c, q_c in zip(p, q):
        p_safe = max(p_c, eps)
        q_safe = max(q_c, eps)
        total += p_safe * math.log(p_safe / q_safe)
    return total


def symmetric_kl_divergence(p: Sequence[float], q: Sequence[float], scale: float = 0.5, eps: float = 1e-12) -> float:
    """
    Symmetric KL Divergence: scale * (D_KL(P || Q) + D_KL(Q || P)).
    """
    kl_pq = kl_divergence(p, q, eps)
    kl_qp = kl_divergence(q, p, eps)
    return scale * (kl_pq + kl_qp)


def jensen_shannon_divergence(p: Sequence[float], q: Sequence[float], eps: float = 1e-12) -> float:
    """
    Jensen-Shannon Divergence: 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M) where M = 0.5*(P+Q).
    """
    m = [0.5 * (p_c + q_c) for p_c, q_c in zip(p, q)]
    kl_pm = kl_divergence(p, m, eps)
    kl_qm = kl_divergence(q, m, eps)
    return 0.5 * (kl_pm + kl_qm)


def consistency_regularization_loss(
    logits_dir: Sequence[float],
    logits_rev: Sequence[float],
    target_dir: int,
    target_rev: Optional[int] = None,
    divergence: str = "sym_kl",
    lambda_cons: float = 1.0,
    warmup_epochs: float = 1.0,
    current_epoch: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    """
    Consistency Regularization Loss for a single pair:
    L = 0.5 * (CE_dir + CE_rev) + lambda(t) * Div(p_dir, p_rev @ P)
    """
    if target_rev is None:
        target_rev = REV_MAP[target_dir]

    # Task CE losses
    loss_ce_dir = cross_entropy_loss(logits_dir, target_dir)
    loss_ce_rev = cross_entropy_loss(logits_rev, target_rev)
    loss_task = 0.5 * (loss_ce_dir + loss_ce_rev)

    # Probabilities
    p_dir = softmax(logits_dir)
    p_rev = softmax(logits_rev)

    # Transposition: p_rev @ P
    p_rev_trans = matmul_vec_mat(p_rev, P_MATRIX)

    # Divergence
    if divergence in ("sym_kl", "kl"):
        loss_cons = symmetric_kl_divergence(p_dir, p_rev_trans, scale=0.5)
    elif divergence in ("jsd", "js"):
        loss_cons = jensen_shannon_divergence(p_dir, p_rev_trans)
    else:
        raise ValueError(f"Unknown divergence: {divergence}")

    # Effective lambda
    if warmup_epochs > 0:
        effective_lambda = lambda_cons * min(1.0, current_epoch / warmup_epochs)
    else:
        effective_lambda = lambda_cons

    total_loss = loss_task + effective_lambda * loss_cons

    metrics = {
        "loss_total": total_loss,
        "loss_task": loss_task,
        "loss_ce_dir": loss_ce_dir,
        "loss_ce_rev": loss_ce_rev,
        "loss_cons": loss_cons,
        "effective_lambda": effective_lambda,
    }
    return total_loss, metrics


def symmetrized_decoding(
    p_dir: Sequence[float],
    p_rev: Sequence[float],
) -> Tuple[List[float], List[float], int, int]:
    """
    Symmetrized Decoding:
    p_sym,dir = 0.5 * (p_dir + p_rev @ P)
    p_sym,rev = 0.5 * (p_rev + p_dir @ P)
    """
    p_rev_P = matmul_vec_mat(p_rev, P_MATRIX)
    p_dir_P = matmul_vec_mat(p_dir, P_MATRIX)

    p_sym_dir = [0.5 * (p_dir[i] + p_rev_P[i]) for i in range(4)]
    p_sym_rev = [0.5 * (p_rev[i] + p_dir_P[i]) for i in range(4)]

    # Argmax with symmetric tie-breaking
    def argmax_tie_break(p: Sequence[float]) -> int:
        max_v = max(p)
        cands = [i for i, v in enumerate(p) if abs(v - max_v) < 1e-9]
        if len(cands) == 1:
            return cands[0]
        if 0 in cands and 2 in cands:
            return 1
        if 1 in cands:
            return 1
        if 3 in cands:
            return 3
        return cands[0]

    pred_dir = argmax_tie_break(p_sym_dir)
    pred_rev = argmax_tie_break(p_sym_rev)
    return p_sym_dir, p_sym_rev, pred_dir, pred_rev


def run_tests():
    print("Executing Pure-Python Blueprint Mathematical Verification...")

    # Test 1: P matrix involution and symmetry
    I4 = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    P_sq = matmul_mat_mat(P_MATRIX, P_MATRIX)
    for i in range(4):
        for j in range(4):
            assert abs(P_sq[i][j] - I4[i][j]) < 1e-9, f"P^2 != I at ({i}, {j})"
            assert abs(P_MATRIX[i][j] - P_MATRIX[j][i]) < 1e-9, f"P != P^T at ({i}, {j})"
    print("Test 1: P is symmetric, orthogonal involution (P^2 = I) [PASSED]")

    # Test 2: Simplex preservation
    p_test = [0.1, 0.3, 0.4, 0.2]
    p_trans = matmul_vec_mat(p_test, P_MATRIX)
    assert abs(sum(p_trans) - 1.0) < 1e-9
    assert p_trans == [0.4, 0.3, 0.1, 0.2]
    print("Test 2: Simplex preservation & exact coordinate permutation [PASSED]")

    # Test 3: KL Divergence properties
    p = [0.7, 0.15, 0.1, 0.05]
    q = [0.5, 0.25, 0.15, 0.1]
    assert abs(kl_divergence(p, p)) < 1e-9
    kl_pq = kl_divergence(p, q)
    kl_qp = kl_divergence(q, p)
    assert kl_pq > 0.0 and kl_qp > 0.0
    sym_kl = symmetric_kl_divergence(p, q, scale=0.5)
    assert abs(sym_kl - 0.5 * (kl_pq + kl_qp)) < 1e-9
    assert abs(symmetric_kl_divergence(p, q) - symmetric_kl_divergence(q, p)) < 1e-9
    print(f"Test 3: Symmetric KL divergence properties (sym_kl={sym_kl:.6f}) [PASSED]")

    # Test 4: Jensen-Shannon Divergence properties
    jsd_val = jensen_shannon_divergence(p, q)
    assert 0.0 <= jsd_val <= math.log(2)
    assert abs(jensen_shannon_divergence(p, p)) < 1e-9
    assert abs(jensen_shannon_divergence(p, q) - jensen_shannon_divergence(q, p)) < 1e-9
    print(f"Test 4: Jensen-Shannon divergence properties (JSD={jsd_val:.6f} <= ln 2) [PASSED]")

    # Test 5: CRL Loss calculation and lambda annealing
    logits_dir = [0.5, 0.1, 2.8, -0.3]  # Strongly predicts FORWARD (index 2)
    logits_rev = [2.6, 0.2, 0.4, -0.1]  # Strongly predicts BACKWARD (index 0)
    target_dir = 2  # FORWARD
    
    # At epoch 0.5 with warmup 1.0: lambda = 0.5
    loss_half, m_half = consistency_regularization_loss(
        logits_dir, logits_rev, target_dir, divergence="sym_kl", warmup_epochs=1.0, current_epoch=0.5
    )
    assert abs(m_half["effective_lambda"] - 0.5) < 1e-9
    
    # At epoch 1.0 with warmup 1.0: lambda = 1.0
    loss_full, m_full = consistency_regularization_loss(
        logits_dir, logits_rev, target_dir, divergence="sym_kl", warmup_epochs=1.0, current_epoch=1.0
    )
    assert abs(m_full["effective_lambda"] - 1.0) < 1e-9
    assert m_full["loss_cons"] < 0.1, f"Consistent logits should have low consistency loss, got {m_full['loss_cons']}"
    print(f"Test 5: CRL Loss calculation & annealing [PASSED]: total={loss_full:.4f}, cons={m_full['loss_cons']:.4f}")

    # Test 6: Inconsistent logits CRL penalty
    logits_rev_inconsistent = [0.2, 0.1, 2.7, -0.2]  # Both predict FORWARD! (Reversal Curse)
    _, m_inconsistent = consistency_regularization_loss(
        logits_dir, logits_rev_inconsistent, target_dir, divergence="sym_kl", warmup_epochs=1.0, current_epoch=1.0
    )
    assert m_inconsistent["loss_cons"] > m_full["loss_cons"] * 10, "Inconsistent pairs must yield large consistency loss!"
    print(f"Test 6: Reversal Curse detection [PASSED]: consistent={m_full['loss_cons']:.4f} vs inconsistent={m_inconsistent['loss_cons']:.4f}")

    # Test 7: Symmetrized Decoding SoftCons = 100% invariant
    # Test on 1000 randomly perturbed logits
    for step in range(1000):
        # generate pseudo-random logits
        l_dir = [((step * 7 + i * 13) % 100) / 20.0 for i in range(4)]
        l_rev = [((step * 11 + i * 17) % 100) / 20.0 for i in range(4)]
        p_d = softmax(l_dir)
        p_r = softmax(l_rev)
        s_d, s_r, pred_d, pred_r = symmetrized_decoding(p_d, p_r)

        # Coordinate equivariance: s_r must equal s_d @ P
        s_d_P = matmul_vec_mat(s_d, P_MATRIX)
        for i in range(4):
            assert abs(s_r[i] - s_d_P[i]) < 1e-9, f"Equivariance broken at step {step}, idx {i}"

        # SoftCons: pred_r must be REV_MAP[pred_d]
        assert pred_r == REV_MAP[pred_d], f"SoftCons broken at step {step}: dir={pred_d}, rev={pred_r}"
    print("Test 7: Symmetrized Decoding SoftCons = 100% verified across 1000 configurations [PASSED]")

    print("\nALL MATHEMATICAL & ALGORITHMIC TESTS PASSED PERFECTLY!")


if __name__ == "__main__":
    run_tests()
