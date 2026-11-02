"""
src/metrics/consistency.py

Directional Transition Matrix and Inconsistency Taxonomy analysis engine.
Deconstructs model predictions across ordered pairs (A, B) vs (B, A) into:
  - 4x4 Directional Transition Matrix (raw counts, conditional probabilities, joint distribution).
  - 16-cell Inconsistency Taxonomy (Consistent, Entailment Loop, Equivalence Asymmetry, Noise).
  - Syntactic category error attribution.
  - LaTeX tabular and ASCII formatted outputs for paper and logs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

try:
    import numpy as np  # type: ignore
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False

from .metrics import (
    ALL_LABELS,
    IDX_TO_LABEL,
    INVOLUTION_PERMUTATION_MATRIX,
    LABEL_TO_IDX,
    REVERSIBLE_LABELS,
    ReversiblePair,
    TaskLabel,
)


# ============================================================================
# 1. Zero-dependency 2D Matrix Fallback
# ============================================================================

class Matrix2D:
    """Lightweight 2D matrix supporting [row, col] slicing when numpy is absent."""
    def __init__(self, data: List[List[Union[int, float]]]):
        self._data = [list(row) for row in data]
        self.shape = (len(self._data), len(self._data[0]) if self._data else 0)

    def __getitem__(self, idx: Any) -> Any:
        if isinstance(idx, tuple):
            r, c = idx
            return self._data[r][c]
        return self._data[idx]

    def __setitem__(self, idx: Any, value: Any) -> None:
        if isinstance(idx, tuple):
            r, c = idx
            self._data[r][c] = value
        else:
            self._data[idx] = value

    def tolist(self) -> List[List[Any]]:
        return [list(r) for r in self._data]

    def __repr__(self) -> str:
        return f"Matrix2D({self._data})"


# ============================================================================
# 2. Data Structures & Taxonomies
# ============================================================================

@dataclass(frozen=True)
class InconsistencyTaxonomyBreakdown:
    """Detailed partition of all 16 transition matrix cells."""
    total_pairs: int
    # 1. Consistent transitions
    consistent_count: int
    consistent_rate: float
    fwd_to_bwd: int
    bwd_to_fwd: int
    equiv_to_equiv: int
    # 2. Entailment Loop / Reversal Curse Trap
    entailment_loop_count: int
    entailment_loop_rate: float
    fwd_fwd_bias: int
    bwd_bwd_bias: int
    # 3. Equivalence Asymmetry
    equiv_asymmetry_count: int
    equiv_asymmetry_rate: float
    equiv_to_dir: int
    dir_to_equiv: int
    equiv_to_noise: int
    noise_to_equiv: int
    # 4. Semantic Dropout / Noise Collapse
    noise_collapse_count: int
    noise_collapse_rate: float
    dir_to_noise: int
    noise_to_dir: int
    mutual_noise: int

    def to_dict(self) -> Dict[str, Union[int, float]]:
        return {
            "total_pairs": self.total_pairs,
            "consistent_count": self.consistent_count,
            "consistent_rate": self.consistent_rate,
            "fwd_to_bwd": self.fwd_to_bwd,
            "bwd_to_fwd": self.bwd_to_fwd,
            "equiv_to_equiv": self.equiv_to_equiv,
            "entailment_loop_count": self.entailment_loop_count,
            "entailment_loop_rate": self.entailment_loop_rate,
            "fwd_fwd_bias": self.fwd_fwd_bias,
            "bwd_bwd_bias": self.bwd_bwd_bias,
            "equiv_asymmetry_count": self.equiv_asymmetry_count,
            "equiv_asymmetry_rate": self.equiv_asymmetry_rate,
            "equiv_to_dir": self.equiv_to_dir,
            "dir_to_equiv": self.dir_to_equiv,
            "equiv_to_noise": self.equiv_to_noise,
            "noise_to_equiv": self.noise_to_equiv,
            "noise_collapse_count": self.noise_collapse_count,
            "noise_collapse_rate": self.noise_collapse_rate,
            "dir_to_noise": self.dir_to_noise,
            "noise_to_dir": self.noise_to_dir,
            "mutual_noise": self.mutual_noise,
        }


@dataclass(frozen=True)
class DirectionalTransitionReport:
    """Complete diagnostic report for directional transitions."""
    raw_matrix: Any             # Shape (4, 4) np.ndarray or Matrix2D
    conditional_matrix: Any     # Shape (4, 4) row-normalized
    joint_matrix: Any           # Shape (4, 4) total-normalized
    taxonomy: InconsistencyTaxonomyBreakdown
    transition_entropy: float   # Mean conditional transition entropy (bits)
    reversal_curse_index: float # Proportion of directional entailments in loop

    def format_ascii_table(self) -> str:
        """Render readable ASCII transition matrix table."""
        header = f"{'Direct \\ Reverse':<18} | {'BWD (0)':<9} | {'EQUIV (1)':<9} | {'FWD (2)':<9} | {'NEG (3)':<9} | {'Total':<6}"
        divider = "-" * len(header)
        lines = [header, divider]
        short_names = ["BWD (0)", "EQUIV (1)", "FWD (2)", "NEG (3)"]

        for j in range(4):
            row_vals = " | ".join(f"{int(self.raw_matrix[j, k]):>9d}" for k in range(4))
            row_sum = sum(int(self.raw_matrix[j, k]) for k in range(4))
            lines.append(f"{short_names[j]:<18} | {row_vals} | {row_sum:>6d}")

        lines.append(divider)
        col_sums = " | ".join(f"{sum(int(self.raw_matrix[r, k]) for r in range(4)):>9d}" for k in range(4))
        tot = sum(int(self.raw_matrix[r, c]) for r in range(4) for c in range(4))
        lines.append(f"{'Total':<18} | {col_sums} | {tot:>6d}")
        return "\n".join(lines)

    def format_latex_table(self, caption: str = "Directional Transition Matrix") -> str:
        """Render LaTeX tabular block suitable for report/paper.tex."""
        s = [
            "\\begin{table}[ht]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{l|cccc|r}",
            "\\toprule",
            "\\textbf{Direct $\\downarrow$ / Reverse $\\rightarrow$} & \\textbf{BWD} & \\textbf{EQU} & \\textbf{FWD} & \\textbf{NEG} & \\textbf{Total} \\\\",
            "\\midrule",
        ]
        labels = ["\\textbf{BWD}", "\\textbf{EQU}", "\\textbf{FWD}", "\\textbf{NEG}"]
        for j in range(4):
            cells = []
            for k in range(4):
                val = int(self.raw_matrix[j, k])
                is_cons = (j == 0 and k == 2) or (j == 1 and k == 1) or (j == 2 and k == 0)
                if is_cons:
                    cells.append(f"\\textbf{{{val}}}")
                else:
                    cells.append(f"{val}")
            row_sum = sum(int(self.raw_matrix[j, c]) for c in range(4))
            s.append(f"{labels[j]} & " + " & ".join(cells) + f" & {row_sum} \\\\")

        s.append("\\midrule")
        col_sums = [str(sum(int(self.raw_matrix[r, k]) for r in range(4))) for k in range(4)]
        tot = sum(int(self.raw_matrix[r, c]) for r in range(4) for c in range(4))
        s.append("\\textbf{Total} & " + " & ".join(col_sums) + f" & \\textbf{{{tot}}} \\\\")
        s.append("\\bottomrule")
        s.append("\\end{tabular}")
        s.append(f"\\caption{{{caption}}}")
        s.append("\\label{tab:transition_matrix}")
        s.append("\\end{table}")
        return "\n".join(s)


# ============================================================================
# 3. Transition Matrix & Taxonomy Computation Engine
# ============================================================================

def compute_directional_transition_matrix(
    pairs: Sequence[Any],
    predicted_labels: Mapping[str, str],
    canonical_orientation: bool = True,
) -> DirectionalTransitionReport:
    """
    Compute 4x4 Directional Transition Matrix and Inconsistency Taxonomy.
    Args:
        pairs: Reversible pairs list (ReversiblePair, ReciprocalPair, or 2-tuples).
        predicted_labels: Map of instance_id -> predicted label.
        canonical_orientation: If True, first_id is Direct and second_id is Reverse.
    """
    T = [[0 for _ in range(4)] for _ in range(4)]

    for pair in pairs:
        if hasattr(pair, "first_id") and hasattr(pair, "second_id"):
            id1, id2 = pair.first_id, pair.second_id
        elif isinstance(pair, (tuple, list)) and len(pair) == 2:
            p1, p2 = pair[0], pair[1]
            id1 = p1.get("instance_id") if hasattr(p1, "get") else getattr(p1, "instance_id", str(p1))
            id2 = p2.get("instance_id") if hasattr(p2, "get") else getattr(p2, "instance_id", str(p2))
        elif isinstance(pair, dict):
            id1 = pair["first_id"]
            id2 = pair["second_id"]
        else:
            raise TypeError(f"Unsupported pair type: {type(pair)}")

        if id1 not in predicted_labels or id2 not in predicted_labels:
            raise KeyError(f"Missing predictions for pair ({id1}, {id2})")

        pred_dir = predicted_labels[id1]
        pred_rev = predicted_labels[id2]

        idx_dir = LABEL_TO_IDX[pred_dir]
        idx_rev = LABEL_TO_IDX[pred_rev]

        T[idx_dir][idx_rev] += 1

    total = sum(sum(row) for row in T)
    if total == 0:
        raise ValueError("Cannot compute transition matrix on 0 pairs.")

    # Conditional and joint matrices
    cond_T = [[0.0 for _ in range(4)] for _ in range(4)]
    joint_T = [[0.0 for _ in range(4)] for _ in range(4)]

    for j in range(4):
        row_sum = sum(T[j])
        for k in range(4):
            cond_T[j][k] = (T[j][k] / row_sum) if row_sum > 0 else 0.0
            joint_T[j][k] = T[j][k] / total

    # 1. Consistent cells
    fwd_to_bwd = int(T[2][0])
    bwd_to_fwd = int(T[0][2])
    equiv_to_equiv = int(T[1][1])
    consistent_count = fwd_to_bwd + bwd_to_fwd + equiv_to_equiv

    # 2. Entailment Loop (Reversal Curse Trap)
    fwd_fwd_bias = int(T[2][2])
    bwd_bwd_bias = int(T[0][0])
    entailment_loop_count = fwd_fwd_bias + bwd_bwd_bias

    # 3. Equivalence Asymmetry
    equiv_to_dir = int(T[1][0] + T[1][2])
    dir_to_equiv = int(T[0][1] + T[2][1])
    equiv_to_noise = int(T[1][3])
    noise_to_equiv = int(T[3][1])
    equiv_asym_count = equiv_to_dir + dir_to_equiv + equiv_to_noise + noise_to_equiv

    # 4. Semantic Dropout / Noise Collapse
    dir_to_noise = int(T[0][3] + T[2][3])
    noise_to_dir = int(T[3][0] + T[3][2])
    mutual_noise = int(T[3][3])
    noise_collapse_count = dir_to_noise + noise_to_dir + mutual_noise

    taxonomy = InconsistencyTaxonomyBreakdown(
        total_pairs=total,
        consistent_count=consistent_count,
        consistent_rate=consistent_count / total,
        fwd_to_bwd=fwd_to_bwd,
        bwd_to_fwd=bwd_to_fwd,
        equiv_to_equiv=equiv_to_equiv,
        entailment_loop_count=entailment_loop_count,
        entailment_loop_rate=entailment_loop_count / total,
        fwd_fwd_bias=fwd_fwd_bias,
        bwd_bwd_bias=bwd_bwd_bias,
        equiv_asymmetry_count=equiv_asym_count,
        equiv_asymmetry_rate=equiv_asym_count / total,
        equiv_to_dir=equiv_to_dir,
        dir_to_equiv=dir_to_equiv,
        equiv_to_noise=equiv_to_noise,
        noise_to_equiv=noise_to_equiv,
        noise_collapse_count=noise_collapse_count,
        noise_collapse_rate=noise_collapse_count / total,
        dir_to_noise=dir_to_noise,
        noise_to_dir=noise_to_dir,
        mutual_noise=mutual_noise,
    )

    # Transition Entropy: mean Shannon entropy over rows (in bits)
    entropy_per_row: List[float] = []
    for j in range(4):
        h = 0.0
        for k in range(4):
            prob = cond_T[j][k]
            if prob > 0:
                h -= prob * math.log2(prob)
        entropy_per_row.append(h)
    transition_entropy = sum(entropy_per_row) / len(entropy_per_row)

    # Reversal Curse Index
    dir_entailment_total = sum(T[2]) + sum(T[0])
    rcsi = (
        float((fwd_fwd_bias + bwd_bwd_bias) / dir_entailment_total)
        if dir_entailment_total > 0
        else 0.0
    )

    if HAS_NUMPY and np is not None:
        raw_mat = np.array(T, dtype=np.int64)
        cond_mat = np.array(cond_T, dtype=np.float64)
        joint_mat = np.array(joint_T, dtype=np.float64)
    else:
        raw_mat = Matrix2D(T)
        cond_mat = Matrix2D(cond_T)
        joint_mat = Matrix2D(joint_T)

    return DirectionalTransitionReport(
        raw_matrix=raw_mat,
        conditional_matrix=cond_mat,
        joint_matrix=joint_mat,
        taxonomy=taxonomy,
        transition_entropy=transition_entropy,
        reversal_curse_index=rcsi,
    )


# ============================================================================
# 4. Syntactic Breakdown Analysis
# ============================================================================

def compute_consistency_by_category(
    pairs: Sequence[Any],
    pair_categories: Mapping[Union[Tuple[str, str], str], str],
    predicted_labels: Mapping[str, str],
) -> Dict[str, Dict[str, Union[int, float]]]:
    """
    Compute SoftCons, HardCons, and RCSI partitioned by syntactic category.
    """
    grouped_pairs: Dict[str, List[Any]] = {}

    for pair in pairs:
        if hasattr(pair, "first_id") and hasattr(pair, "second_id"):
            id1, id2 = pair.first_id, pair.second_id
        elif isinstance(pair, (tuple, list)) and len(pair) == 2:
            p1, p2 = pair[0], pair[1]
            id1 = p1.get("instance_id") if hasattr(p1, "get") else getattr(p1, "instance_id", str(p1))
            id2 = p2.get("instance_id") if hasattr(p2, "get") else getattr(p2, "instance_id", str(p2))
        elif isinstance(pair, dict):
            id1 = pair["first_id"]
            id2 = pair["second_id"]
        else:
            continue

        cat = (
            pair_categories.get((id1, id2))
            or pair_categories.get(id1)
            or pair_categories.get(id2)
            or "unknown"
        )
        grouped_pairs.setdefault(cat, []).append(pair)

    results: Dict[str, Dict[str, Union[int, float]]] = {}
    for cat, cat_pairs in grouped_pairs.items():
        rep = compute_directional_transition_matrix(cat_pairs, predicted_labels)
        results[cat] = {
            "count": len(cat_pairs),
            "soft_cons": rep.taxonomy.consistent_rate,
            "entailment_loop_rate": rep.taxonomy.entailment_loop_rate,
            "rcsi": rep.reversal_curse_index,
            "transition_entropy": rep.transition_entropy,
        }

    return results
