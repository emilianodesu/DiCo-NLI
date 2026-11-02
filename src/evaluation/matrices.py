r"""
src/evaluation/matrices.py

Multiclass Confusion Matrix and Directional Transition Matrix Analysis Engine
for SemEval-2027 Task 2 (DiCo-NLI).

Provides:
  1. 4x4 Multiclass Confusion Matrix:
     - Canonical class order: EQUIVALENCE, FORWARD_ENTAILMENT, BACKWARD_ENTAILMENT, NEGATIVE_OTHER.
     - Raw counts, row-normalized (recall), column-normalized (precision), total-normalized percentages.
     - Per-class precision, recall, F1, specificity, and support.
     - Aggregate accuracy, macro F1, and support-weighted F1 (identical to official DiCo-NLI metric).
  2. 4x4 Directional Transition Matrix:
     - Directional pairing: predicted direct label \hat{y}_{dir} vs predicted reverse label \hat{y}_{rev}.
     - Joint count matrix T_{i, j} = sum I(\hat{y}_{dir}=i, \hat{y}_{rev}=j).
     - Row-conditional transition probability matrix P(\hat{y}_{rev}=j | \hat{y}_{dir}=i).
     - Joint probability matrix P_{joint}(i, j).
     - Ideal deterministic involution permutation matrix P:
         P = [[1, 0, 0, 0],
              [0, 0, 1, 0],
              [0, 1, 0, 0],
              [0, 0, 0, 1]]
     - Inconsistency Diagnostics:
         * Transition Violation Rate (1 - SoftCons)
         * Frobenius Distance to ideal involution P (raw and normalized in [0, 1])
         * Mean row Total Variation Distance (TVD)
         * Mean Shannon Transition Entropy (in bits)
         * Reversal Curse Index (RCSI): proportion of directional entailments in loop
         * 16-cell taxonomy breakdown (Consistent, Entailment Loop, Equiv Asymmetry, Noise)
  3. Exporters:
     - Formatted ASCII tables for logs and CLI.
     - LaTeX tabular blocks ready for direct inclusion in report/paper.tex (ACL SemEval format).
     - Structured Python dict and JSON serialization.
  4. Data Integration:
     - Codabench 2-column CSV files (instance_id,label).
     - Pandas DataFrames and lightweight SimpleDataFrames.
     - In-memory prediction dictionaries and reciprocal pair lists.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

# Optional numpy integration with pure-Python fallback
try:
    import numpy as np  # type: ignore
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False

# Optional pandas integration with pure-Python fallback
try:
    import pandas as pd  # type: ignore
    HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore
    HAS_PANDAS = False


# ============================================================================
# 1. Label Constants & Involution Permutation Matrices
# ============================================================================

CANONICAL_LABELS: Tuple[str, ...] = (
    "EQUIVALENCE",
    "FORWARD_ENTAILMENT",
    "BACKWARD_ENTAILMENT",
    "NEGATIVE_OTHER",
)

LABEL_TO_CANONICAL_IDX: Dict[str, int] = {
    lbl: idx for idx, lbl in enumerate(CANONICAL_LABELS)
}

CANONICAL_IDX_TO_LABEL: Dict[int, str] = {
    idx: lbl for idx, lbl in enumerate(CANONICAL_LABELS)
}

LABEL_SHORT_NAMES: Dict[str, str] = {
    "EQUIVALENCE": "EQU",
    "FORWARD_ENTAILMENT": "FWD",
    "BACKWARD_ENTAILMENT": "BWD",
    "NEGATIVE_OTHER": "NEG",
}

REVERSE_LABEL_MAP: Dict[str, str] = {
    "EQUIVALENCE": "EQUIVALENCE",
    "FORWARD_ENTAILMENT": "BACKWARD_ENTAILMENT",
    "BACKWARD_ENTAILMENT": "FORWARD_ENTAILMENT",
    "NEGATIVE_OTHER": "NEGATIVE_OTHER",
}

# Ideal deterministic involution permutation matrix for CANONICAL_LABELS:
# Row 0 (EQU): [1, 0, 0, 0] -> EQU
# Row 1 (FWD): [0, 0, 1, 0] -> BWD
# Row 2 (BWD): [0, 1, 0, 0] -> FWD
# Row 3 (NEG): [0, 0, 0, 1] -> NEG
IDEAL_INVOLUTION_MATRIX: List[List[float]] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def get_ideal_involution_matrix(
    labels: Sequence[str] = CANONICAL_LABELS,
) -> List[List[float]]:
    """
    Construct the ideal involution permutation matrix P for an arbitrary label ordering.
    Guarantees P_{i, j} = 1.0 iff labels[j] == reverse_label(labels[i]).
    """
    n = len(labels)
    lbl_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    P = [[0.0 for _ in range(n)] for _ in range(n)]
    for i, lbl in enumerate(labels):
        target = REVERSE_LABEL_MAP.get(lbl, lbl)
        if target in lbl_to_idx:
            P[i][lbl_to_idx[target]] = 1.0
    return P


# ============================================================================
# 2. Multiclass Confusion Matrix Data Structures
# ============================================================================

@dataclass(frozen=True)
class PerClassMetrics:
    """Detailed performance metrics for an individual class."""
    label: str
    support: int          # Actual positive instances in gold (TP + FN)
    predicted_count: int  # Instances predicted as this class (TP + FP)
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float      # TP / (TP + FP)
    recall: float         # TP / (TP + FN)
    f1: float             # 2 * P * R / (P + R)
    specificity: float    # TN / (TN + FP)

    def to_dict(self) -> Dict[str, Union[str, int, float]]:
        return asdict(self)


@dataclass(frozen=True)
class ConfusionMatrixReport:
    """
    Complete multiclass confusion matrix evaluation report.
    Matrix conventions:
      - Rows represent Ground Truth (True) classes.
      - Columns represent Model Predicted classes.
    """
    labels: Tuple[str, ...]
    raw_matrix: List[List[int]]           # C[true_idx][pred_idx]
    row_normalized: List[List[float]]     # C[i][j] / sum_k C[i][k] (Recall distribution)
    col_normalized: List[List[float]]     # C[i][j] / sum_k C[k][j] (Precision distribution)
    total_normalized: List[List[float]]   # C[i][j] / N (Joint distribution %)
    per_class: Dict[str, PerClassMetrics]
    accuracy: float
    macro_f1: float
    weighted_f1: float
    total_instances: int

    def numpy_raw(self) -> Any:
        """Return raw matrix as np.ndarray if numpy is installed."""
        if HAS_NUMPY and np is not None:
            return np.array(self.raw_matrix, dtype=np.int64)
        return self.raw_matrix

    def numpy_row_norm(self) -> Any:
        """Return row-normalized matrix as np.ndarray if numpy is installed."""
        if HAS_NUMPY and np is not None:
            return np.array(self.row_normalized, dtype=np.float64)
        return self.row_normalized

    def format_ascii_table(self, show_metrics: bool = True) -> str:
        """Render a readable, formatted ASCII table of the confusion matrix."""
        n = len(self.labels)
        short_names = [LABEL_SHORT_NAMES.get(l, l[:4]) for l in self.labels]
        col_width = max(8, max(len(s) for s in short_names) + 2)

        # Header
        header_cols = " | ".join(f"{s:^{col_width}}" for s in short_names)
        title = f"{'True \\ Pred':<{col_width + 4}} | {header_cols} | {'Total':>7}"
        divider = "=" * len(title)
        sub_divider = "-" * len(title)

        lines = [
            divider,
            f"Multiclass Confusion Matrix (True Rows \\ Pred Cols, N={self.total_instances})",
            divider,
            title,
            sub_divider,
        ]

        for i in range(n):
            row_vals = " | ".join(f"{self.raw_matrix[i][j]:>{col_width}d}" for j in range(n))
            row_sum = sum(self.raw_matrix[i])
            tag = f"{short_names[i]} ({i})"
            lines.append(f"{tag:<{col_width + 4}} | {row_vals} | {row_sum:>7d}")

        lines.append(sub_divider)
        col_sums = " | ".join(
            f"{sum(self.raw_matrix[r][c] for r in range(n)):>{col_width}d}" for c in range(n)
        )
        lines.append(f"{'Total Pred':<{col_width + 4}} | {col_sums} | {self.total_instances:>7d}")
        lines.append(divider)

        if show_metrics:
            lines.append("Per-Class Performance Diagnostics:")
            lines.append("-" * 76)
            lines.append(
                f"{'Class':<22} | {'Support':>7} | {'Predicted':>9} | {'Precision':>9} | {'Recall':>8} | {'F1':>8}"
            )
            lines.append("-" * 76)
            for lbl in self.labels:
                m = self.per_class[lbl]
                lines.append(
                    f"{lbl:<22} | {m.support:>7d} | {m.predicted_count:>9d} | "
                    f"{m.precision:>9.4f} | {m.recall:>8.4f} | {m.f1:>8.4f}"
                )
            lines.append("-" * 76)
            lines.append(
                f"Summary: Accuracy: {self.accuracy:.4f} | Macro F1: {self.macro_f1:.4f} | "
                f"Weighted F1: {self.weighted_f1:.4f}"
            )
            lines.append(divider)

        return "\n".join(lines)

    def format_latex_table(
        self,
        caption: str = "Multiclass Confusion Matrix and Classification Performance",
        label: str = "tab:confusion_matrix",
    ) -> str:
        """Render a publication-ready LaTeX tabular block for ACL SemEval paper."""
        short_names = [LABEL_SHORT_NAMES.get(l, l[:3]) for l in self.labels]
        col_headers = " & ".join(f"\\textbf{{{s}}}" for s in short_names)

        s = [
            "\\begin{table}[ht]",
            "\\centering",
            "\\small",
            f"\\begin{{tabular}}{{l|{'c' * len(self.labels)}|r|ccc}}",
            "\\toprule",
            f"\\textbf{{Gold $\\downarrow$ / Pred $\\rightarrow$}} & {col_headers} & \\textbf{{Total}} & \\textbf{{Prec}} & \\textbf{{Rec}} & \\textbf{{F1}} \\\\",
            "\\midrule",
        ]

        for i, lbl in enumerate(self.labels):
            m = self.per_class[lbl]
            cells = []
            for j in range(len(self.labels)):
                cnt = self.raw_matrix[i][j]
                if i == j:
                    cells.append(f"\\textbf{{{cnt}}}")
                else:
                    cells.append(str(cnt))
            row_sum = sum(self.raw_matrix[i])
            row_str = " & ".join(cells)
            s.append(
                f"\\textbf{{{short_names[i]}}} & {row_str} & {row_sum} & "
                f"{m.precision:.3f} & {m.recall:.3f} & {m.f1:.3f} \\\\"
            )

        s.append("\\midrule")
        col_sums = [
            str(sum(self.raw_matrix[r][c] for r in range(len(self.labels))))
            for c in range(len(self.labels))
        ]
        s.append(
            f"\\textbf{{Total Pred}} & {' & '.join(col_sums)} & \\textbf{{{self.total_instances}}} & "
            f"\\multicolumn{{3}}{{c}}{{\\textbf{{Acc:}} {self.accuracy:.3f} $\\mid$ \\textbf{{WF1:}} {self.weighted_f1:.3f}}} \\\\"
        )
        s.append("\\bottomrule")
        s.append("\\end{tabular}")
        s.append(f"\\caption{{{caption}}}")
        s.append(f"\\label{{{label}}}")
        s.append("\\end{table}")
        return "\n".join(s)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire report into a nested serializable dictionary."""
        return {
            "labels": list(self.labels),
            "total_instances": self.total_instances,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "raw_matrix": self.raw_matrix,
            "row_normalized": self.row_normalized,
            "col_normalized": self.col_normalized,
            "total_normalized": self.total_normalized,
            "per_class": {k: v.to_dict() for k, v in self.per_class.items()},
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, file_path: Union[str, Path]) -> None:
        """Write JSON report to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())


# ============================================================================
# 3. Directional Transition Matrix Data Structures
# ============================================================================

@dataclass(frozen=True)
class TransitionInconsistencyMetrics:
    """
    Comprehensive diagnostics quantifying directional inconsistency.
    Compares observed transitions T against the ideal deterministic involution matrix P.
    """
    total_pairs: int
    consistent_pairs: int
    consistent_rate: float            # SoftCons rate over reciprocal pairs
    violation_pairs: int
    violation_rate: float             # 1.0 - consistent_rate
    frobenius_distance: float         # ||C - P||_F
    normalized_frobenius: float       # ||C - P||_F / sqrt(2 * num_classes) in [0, 1]
    mean_tvd: float                   # Mean row Total Variation Distance
    mean_transition_entropy: float    # Mean conditional Shannon entropy across rows (bits)
    reversal_curse_index: float       # (FWD-FWD + BWD-BWD) / directional entailment pairs

    # Detailed cell breakdown
    equiv_to_equiv: int
    fwd_to_bwd: int
    bwd_to_fwd: int
    neg_to_neg: int
    fwd_fwd_trap: int                 # FWD predicted in both directions
    bwd_bwd_trap: int                 # BWD predicted in both directions
    equiv_asymmetry_count: int        # Transitions between EQU and {FWD, BWD, NEG}
    noise_collapse_count: int         # Transitions between {FWD, BWD} and NEG

    def to_dict(self) -> Dict[str, Union[int, float]]:
        return asdict(self)


@dataclass(frozen=True)
class DirectionalTransitionReport:
    """
    Complete directional transition matrix report for reciprocal pairs (A->B, B->A).
    """
    labels: Tuple[str, ...]
    raw_matrix: List[List[int]]           # T[dir_idx][rev_idx]
    conditional_matrix: List[List[float]] # P(rev | dir)
    joint_matrix: List[List[float]]       # T[i][j] / N_pairs
    ideal_matrix: List[List[float]]       # Involution matrix P
    metrics: TransitionInconsistencyMetrics
    total_pairs: int

    def numpy_raw(self) -> Any:
        """Return raw transition matrix as np.ndarray if numpy is installed."""
        if HAS_NUMPY and np is not None:
            return np.array(self.raw_matrix, dtype=np.int64)
        return self.raw_matrix

    def numpy_conditional(self) -> Any:
        """Return conditional probability matrix as np.ndarray if numpy is installed."""
        if HAS_NUMPY and np is not None:
            return np.array(self.conditional_matrix, dtype=np.float64)
        return self.conditional_matrix

    def format_ascii_table(self, show_conditional: bool = True) -> str:
        """Render a readable ASCII representation of the transition matrix and diagnostics."""
        n = len(self.labels)
        short_names = [LABEL_SHORT_NAMES.get(l, l[:4]) for l in self.labels]
        col_width = max(8, max(len(s) for s in short_names) + 2)

        header_cols = " | ".join(f"{s:^{col_width}}" for s in short_names)
        title = f"{'Direct \\ Reverse':<{col_width + 4}} | {header_cols} | {'Total':>7}"
        divider = "=" * len(title)
        sub_divider = "-" * len(title)

        lines = [
            divider,
            f"Directional Transition Matrix (Direct Rows \\ Reverse Cols, N_pairs={self.total_pairs})",
            divider,
            title,
            sub_divider,
        ]

        for i in range(n):
            cells = []
            for j in range(n):
                cnt = self.raw_matrix[i][j]
                is_ideal = (self.ideal_matrix[i][j] == 1.0)
                is_trap = (i == j and short_names[i] in ("FWD", "BWD"))
                flag = "*" if is_ideal else ("!" if is_trap else " ")
                cells.append(f"{cnt:>{col_width - 1}d}{flag}")
            row_vals = " | ".join(cells)
            row_sum = sum(self.raw_matrix[i])
            tag = f"{short_names[i]} ({i})"
            lines.append(f"{tag:<{col_width + 4}} | {row_vals} | {row_sum:>7d}")

        lines.append(sub_divider)
        col_sums = " | ".join(
            f"{sum(self.raw_matrix[r][c] for r in range(n)):>{col_width}d}" for c in range(n)
        )
        lines.append(f"{'Total Reverse':<{col_width + 4}} | {col_sums} | {self.total_pairs:>7d}")
        lines.append(divider)

        if show_conditional:
            lines.append("Conditional Transition Probabilities P(Reverse | Direct):")
            lines.append(sub_divider)
            for i in range(n):
                prob_parts = []
                for j in range(n):
                    p = self.conditional_matrix[i][j]
                    is_ideal = (self.ideal_matrix[i][j] == 1.0)
                    is_trap = (i == j and short_names[i] in ("FWD", "BWD"))
                    flag = "*" if is_ideal else ("!" if is_trap else " ")
                    prob_parts.append(f"{short_names[j]}: {p:.3f}{flag}")
                tag = f"{short_names[i]} ({i})"
                lines.append(f"{tag:<10} -> " + " | ".join(prob_parts))
            lines.append(sub_divider)

        m = self.metrics
        lines.extend([
            "Legend: [*] Consistent Involution Target  [!] Reversal Curse Entailment Loop",
            sub_divider,
            "Inconsistency Diagnostics:",
            f"  - Total Reciprocal Pairs:          {m.total_pairs}",
            f"  - Consistent Pairs (SoftCons):      {m.consistent_pairs} ({m.consistent_rate:.4%})",
            f"  - Violation Pairs:                 {m.violation_pairs} ({m.violation_rate:.4%})",
            f"  - Reversal Curse Index (RCSI):     {m.reversal_curse_index:.4%}",
            f"  - Frobenius Distance to Ideal P:   {m.frobenius_distance:.4f} (Normalized: {m.normalized_frobenius:.4f})",
            f"  - Mean Total Variation Distance:   {m.mean_tvd:.4f}",
            f"  - Mean Transition Shannon Entropy: {m.mean_transition_entropy:.4f} bits",
            divider,
        ])

        return "\n".join(lines)

    def format_latex_table(
        self,
        caption: str = "Directional Transition Matrix for Reciprocal Pairs ($A \\to B$ vs $B \\to A$)",
        label: str = "tab:transition_matrix",
    ) -> str:
        """Render publication-ready LaTeX tabular block for ACL SemEval paper."""
        n = len(self.labels)
        short_names = [LABEL_SHORT_NAMES.get(l, l[:3]) for l in self.labels]
        col_headers = " & ".join(f"\\textbf{{{s}}}" for s in short_names)

        s = [
            "\\begin{table}[ht]",
            "\\centering",
            "\\small",
            f"\\begin{{tabular}}{{l|{'c' * n}|r|c}}",
            "\\toprule",
            f"\\textbf{{Direct $\\downarrow$ / Reverse $\\rightarrow$}} & {col_headers} & \\textbf{{Total}} & \\textbf{{$P(\\text{{target}} \\mid \\text{{dir}})$}} \\\\",
            "\\midrule",
        ]

        for i in range(n):
            cells = []
            target_prob = 0.0
            for j in range(n):
                cnt = self.raw_matrix[i][j]
                is_ideal = (self.ideal_matrix[i][j] == 1.0)
                is_trap = (i == j and short_names[i] in ("FWD", "BWD"))
                if is_ideal:
                    cells.append(f"\\textbf{{{cnt}}}")
                    target_prob = self.conditional_matrix[i][j]
                elif is_trap:
                    cells.append(f"\\underline{{{cnt}}}")
                else:
                    cells.append(str(cnt))
            row_sum = sum(self.raw_matrix[i])
            s.append(
                f"\\textbf{{{short_names[i]}}} & {' & '.join(cells)} & {row_sum} & {target_prob * 100.0:.1f}\\% \\\\"
            )

        s.append("\\midrule")
        col_sums = [
            str(sum(self.raw_matrix[r][c] for r in range(n))) for c in range(n)
        ]
        m = self.metrics
        s.append(
            f"\\textbf{{Total Reverse}} & {' & '.join(col_sums)} & \\textbf{{{self.total_pairs}}} & "
            f"\\textbf{{SoftCons:}} {m.consistent_rate * 100.0:.1f}\\% \\\\"
        )
        s.append("\\bottomrule")
        s.append("\\end{tabular}")
        note = (
            f"Violation Rate: {m.violation_rate * 100.0:.1f}\\%, "
            f"RCSI: {m.reversal_curse_index * 100.0:.1f}\\%, "
            f"Frobenius: {m.frobenius_distance:.3f}."
        )
        s.append(f"\\caption{{{caption}. Bold values indicate consistent involution targets; underlined values indicate Reversal Curse loops. {note}}}")
        s.append(f"\\label{{{label}}}")
        s.append("\\end{table}")
        return "\n".join(s)

    def to_dict(self) -> Dict[str, Any]:
        """Convert transition report into serializable dictionary."""
        return {
            "labels": list(self.labels),
            "total_pairs": self.total_pairs,
            "raw_matrix": self.raw_matrix,
            "conditional_matrix": self.conditional_matrix,
            "joint_matrix": self.joint_matrix,
            "ideal_matrix": self.ideal_matrix,
            "metrics": self.metrics.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize transition report to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, file_path: Union[str, Path]) -> None:
        """Write JSON report to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())


# ============================================================================
# 4. Computation Engine: Confusion Matrix
# ============================================================================

def compute_confusion_matrix(
    gold: Union[Sequence[str], Mapping[str, str]],
    pred: Union[Sequence[str], Mapping[str, str]],
    labels: Sequence[str] = CANONICAL_LABELS,
) -> ConfusionMatrixReport:
    """
    Compute 4x4 Multiclass Confusion Matrix with normalized percentages and per-class metrics.

    Args:
        gold: True labels sequence or instance_id -> gold_label mapping.
        pred: Predicted labels sequence or instance_id -> pred_label mapping.
        labels: Label inventory and canonical display ordering.

    Returns:
        ConfusionMatrixReport dataclass.
    """
    labels_tuple = tuple(labels)
    n = len(labels_tuple)
    lbl_to_idx = {lbl: i for i, lbl in enumerate(labels_tuple)}

    # Align inputs if passed as mappings
    if isinstance(gold, Mapping) and isinstance(pred, Mapping):
        common_ids = [k for k in gold if k in pred]
        if not common_ids:
            raise ValueError("No common instance IDs found between gold and predictions.")
        gold_seq = [gold[k] for k in common_ids]
        pred_seq = [pred[k] for k in common_ids]
    elif isinstance(gold, Sequence) and isinstance(pred, Sequence):
        if len(gold) != len(pred):
            raise ValueError(f"Length mismatch: gold has {len(gold)}, pred has {len(pred)}.")
        gold_seq = list(gold)
        pred_seq = list(pred)
    else:
        raise TypeError("Both gold and pred must be Sequences or Mappings.")

    total_instances = len(gold_seq)
    if total_instances == 0:
        raise ValueError("Cannot compute confusion matrix on empty sequences.")

    # Validate labels
    for idx, (g, p) in enumerate(zip(gold_seq, pred_seq)):
        if g not in lbl_to_idx:
            raise ValueError(f"Unrecognized gold label '{g}' at index {idx}.")
        if p not in lbl_to_idx:
            raise ValueError(f"Unrecognized pred label '{p}' at index {idx}.")

    # 1. Raw count matrix C[gold][pred]
    raw_matrix = [[0 for _ in range(n)] for _ in range(n)]
    for g, p in zip(gold_seq, pred_seq):
        raw_matrix[lbl_to_idx[g]][lbl_to_idx[p]] += 1

    # 2. Normalized matrices
    row_normalized = [[0.0 for _ in range(n)] for _ in range(n)]
    col_normalized = [[0.0 for _ in range(n)] for _ in range(n)]
    total_normalized = [[0.0 for _ in range(n)] for _ in range(n)]

    col_sums = [sum(raw_matrix[r][c] for r in range(n)) for c in range(n)]

    for i in range(n):
        row_sum = sum(raw_matrix[i])
        for j in range(n):
            cnt = raw_matrix[i][j]
            row_normalized[i][j] = (cnt / row_sum) if row_sum > 0 else 0.0
            col_normalized[i][j] = (cnt / col_sums[j]) if col_sums[j] > 0 else 0.0
            total_normalized[i][j] = cnt / total_instances

    # 3. Per-class metrics
    per_class: Dict[str, PerClassMetrics] = {}
    f1_weighted_sum = 0.0
    f1_macro_sum = 0.0
    correct_count = 0

    for i, lbl in enumerate(labels_tuple):
        tp = raw_matrix[i][i]
        correct_count += tp
        fp = col_sums[i] - tp
        fn = sum(raw_matrix[i]) - tp
        tn = total_instances - (tp + fp + fn)
        support = tp + fn
        predicted_count = tp + fp

        prec = (tp / predicted_count) if predicted_count > 0 else 0.0
        rec = (tp / support) if support > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        spec = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        per_class[lbl] = PerClassMetrics(
            label=lbl,
            support=support,
            predicted_count=predicted_count,
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            precision=prec,
            recall=rec,
            f1=f1,
            specificity=spec,
        )

        f1_macro_sum += f1
        f1_weighted_sum += (support / total_instances) * f1

    accuracy = correct_count / total_instances
    macro_f1 = f1_macro_sum / n
    weighted_f1 = f1_weighted_sum

    return ConfusionMatrixReport(
        labels=labels_tuple,
        raw_matrix=raw_matrix,
        row_normalized=row_normalized,
        col_normalized=col_normalized,
        total_normalized=total_normalized,
        per_class=per_class,
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        total_instances=total_instances,
    )


# ============================================================================
# 5. Computation Engine: Directional Transition Matrix
# ============================================================================

def compute_directional_transition_matrix(
    pairs: Sequence[Any],
    predictions: Union[Mapping[str, str], Sequence[Tuple[str, str]]],
    labels: Sequence[str] = CANONICAL_LABELS,
) -> DirectionalTransitionReport:
    """
    Compute 4x4 Directional Transition Matrix and Inconsistency Diagnostics.

    Args:
        pairs: Reciprocal pairs collection. Each pair can be:
               - Tuple/List of (id_dir, id_rev)
               - Object with .first_id and .second_id (e.g. ReversiblePair)
               - Object with .direct and .inverse (e.g. ReciprocalPair)
               - Dict with keys 'first_id', 'second_id' or 'direct', 'inverse'
        predictions: Mapping {instance_id: predicted_label} or pairs of labels.
        labels: Label inventory and display ordering.

    Returns:
        DirectionalTransitionReport dataclass.
    """
    labels_tuple = tuple(labels)
    n = len(labels_tuple)
    lbl_to_idx = {lbl: i for i, lbl in enumerate(labels_tuple)}
    ideal_P = get_ideal_involution_matrix(labels_tuple)

    # Standardize predictions mapping
    pred_map: Dict[str, str] = {}
    if isinstance(predictions, Mapping):
        pred_map = {str(k): str(v).strip() for k, v in predictions.items()}
    elif isinstance(predictions, Sequence):
        # Could be sequence of (id, label)
        for item in predictions:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                pred_map[str(item[0])] = str(item[1]).strip()

    T = [[0 for _ in range(n)] for _ in range(n)]
    extracted_count = 0

    for pair in pairs:
        # Extract direct and reverse IDs or labels
        id_dir: Optional[str] = None
        id_rev: Optional[str] = None
        label_dir: Optional[str] = None
        label_rev: Optional[str] = None

        if hasattr(pair, "first_id") and hasattr(pair, "second_id"):
            id_dir, id_rev = str(pair.first_id), str(pair.second_id)
        elif hasattr(pair, "direct") and hasattr(pair, "inverse"):
            d, inv = pair.direct, pair.inverse
            id_dir = str(d.get("instance_id") if hasattr(d, "get") else getattr(d, "instance_id", ""))
            id_rev = str(inv.get("instance_id") if hasattr(inv, "get") else getattr(inv, "instance_id", ""))
        elif isinstance(pair, (tuple, list)) and len(pair) == 2:
            p1, p2 = pair[0], pair[1]
            if isinstance(p1, str) and isinstance(p2, str):
                # Could be pair of instance IDs or direct pair of labels
                if p1 in lbl_to_idx and p2 in lbl_to_idx and (p1 not in pred_map and p2 not in pred_map):
                    label_dir, label_rev = p1, p2
                else:
                    id_dir, id_rev = p1, p2
            else:
                id_dir = str(p1.get("instance_id") if hasattr(p1, "get") else getattr(p1, "instance_id", ""))
                id_rev = str(p2.get("instance_id") if hasattr(p2, "get") else getattr(p2, "instance_id", ""))
        elif isinstance(pair, dict):
            id_dir = str(pair.get("first_id") or pair.get("direct", ""))
            id_rev = str(pair.get("second_id") or pair.get("inverse", ""))

        if label_dir is None or label_rev is None:
            if not id_dir or not id_rev:
                continue
            if id_dir not in pred_map or id_rev not in pred_map:
                raise KeyError(f"Missing predictions for reciprocal pair ({id_dir!r}, {id_rev!r}).")
            label_dir = pred_map[id_dir]
            label_rev = pred_map[id_rev]

        if label_dir not in lbl_to_idx:
            raise ValueError(f"Unrecognized direct label '{label_dir}' in pair ({id_dir}, {id_rev}).")
        if label_rev not in lbl_to_idx:
            raise ValueError(f"Unrecognized reverse label '{label_rev}' in pair ({id_dir}, {id_rev}).")

        T[lbl_to_idx[label_dir]][lbl_to_idx[label_rev]] += 1
        extracted_count += 1

    if extracted_count == 0:
        raise ValueError("Cannot compute transition matrix on 0 reciprocal pairs.")

    # Conditional probability P(rev=j | dir=i) and Joint P(i, j)
    cond_T = [[0.0 for _ in range(n)] for _ in range(n)]
    joint_T = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        row_sum = sum(T[i])
        for j in range(n):
            cond_T[i][j] = (T[i][j] / row_sum) if row_sum > 0 else 0.0
            joint_T[i][j] = T[i][j] / extracted_count

    # Inconsistency Metrics Computation
    consistent_pairs = 0
    violation_pairs = 0
    frobenius_sq = 0.0
    tvd_sum = 0.0
    entropy_sum = 0.0
    active_rows = 0

    for i in range(n):
        row_sum = sum(T[i])
        row_tvd = 0.0
        row_entropy = 0.0

        for j in range(n):
            cnt = T[i][j]
            prob = cond_T[i][j]
            is_consistent = (ideal_P[i][j] == 1.0)

            if is_consistent:
                consistent_pairs += cnt
            else:
                violation_pairs += cnt

            diff = prob - ideal_P[i][j]
            frobenius_sq += diff * diff
            row_tvd += abs(diff)

            if prob > 0.0:
                row_entropy -= prob * math.log2(prob)

        if row_sum > 0:
            active_rows += 1
            tvd_sum += 0.5 * row_tvd
            entropy_sum += row_entropy

    frobenius_distance = math.sqrt(frobenius_sq)
    # Maximum possible Frobenius distance between two row-stochastic matrices of size n x n is sqrt(2 * n)
    max_frobenius = math.sqrt(2.0 * n)
    normalized_frobenius = min(1.0, frobenius_distance / max_frobenius)

    mean_tvd = (tvd_sum / active_rows) if active_rows > 0 else 0.0
    mean_entropy = (entropy_sum / active_rows) if active_rows > 0 else 0.0

    # Specific category counters based on canonical or resolved indices
    idx_equ = lbl_to_idx.get("EQUIVALENCE", -1)
    idx_fwd = lbl_to_idx.get("FORWARD_ENTAILMENT", -1)
    idx_bwd = lbl_to_idx.get("BACKWARD_ENTAILMENT", -1)
    idx_neg = lbl_to_idx.get("NEGATIVE_OTHER", -1)

    equiv_to_equiv = T[idx_equ][idx_equ] if idx_equ >= 0 else 0
    fwd_to_bwd = T[idx_fwd][idx_bwd] if (idx_fwd >= 0 and idx_bwd >= 0) else 0
    bwd_to_fwd = T[idx_bwd][idx_fwd] if (idx_fwd >= 0 and idx_bwd >= 0) else 0
    neg_to_neg = T[idx_neg][idx_neg] if idx_neg >= 0 else 0

    fwd_fwd_trap = T[idx_fwd][idx_fwd] if idx_fwd >= 0 else 0
    bwd_bwd_trap = T[idx_bwd][idx_bwd] if idx_bwd >= 0 else 0

    # Directional entailment pairs total
    dir_entailment_total = 0
    if idx_fwd >= 0:
        dir_entailment_total += sum(T[idx_fwd])
    if idx_bwd >= 0:
        dir_entailment_total += sum(T[idx_bwd])

    rcsi = (
        float(fwd_fwd_trap + bwd_bwd_trap) / dir_entailment_total
        if dir_entailment_total > 0
        else 0.0
    )

    # Equivalence asymmetry: transitions involving EQU and {FWD, BWD, NEG}
    equiv_asym_count = 0
    if idx_equ >= 0:
        for c in range(n):
            if c != idx_equ:
                equiv_asym_count += T[idx_equ][c] + T[c][idx_equ]

    # Noise collapse: transitions between directional entailment and NEG
    noise_collapse_count = 0
    if idx_neg >= 0:
        for d in (idx_fwd, idx_bwd):
            if d >= 0:
                noise_collapse_count += T[d][idx_neg] + T[idx_neg][d]

    metrics = TransitionInconsistencyMetrics(
        total_pairs=extracted_count,
        consistent_pairs=consistent_pairs,
        consistent_rate=consistent_pairs / extracted_count,
        violation_pairs=violation_pairs,
        violation_rate=violation_pairs / extracted_count,
        frobenius_distance=frobenius_distance,
        normalized_frobenius=normalized_frobenius,
        mean_tvd=mean_tvd,
        mean_transition_entropy=mean_entropy,
        reversal_curse_index=rcsi,
        equiv_to_equiv=equiv_to_equiv,
        fwd_to_bwd=fwd_to_bwd,
        bwd_to_fwd=bwd_to_fwd,
        neg_to_neg=neg_to_neg,
        fwd_fwd_trap=fwd_fwd_trap,
        bwd_bwd_trap=bwd_bwd_trap,
        equiv_asymmetry_count=equiv_asym_count,
        noise_collapse_count=noise_collapse_count,
    )

    return DirectionalTransitionReport(
        labels=labels_tuple,
        raw_matrix=T,
        conditional_matrix=cond_T,
        joint_matrix=joint_T,
        ideal_matrix=ideal_P,
        metrics=metrics,
        total_pairs=extracted_count,
    )


# ============================================================================
# 6. Data Integration: Codabench CSV & Prediction DataFrames
# ============================================================================

def load_predictions(
    source: Union[str, Path, Mapping[str, str], Any],
    id_col: str = "instance_id",
    pred_col: str = "label",
) -> Dict[str, str]:
    """
    Load predictions from various sources:
      - Path to Codabench CSV file (instance_id,label).
      - Pandas DataFrame or SimpleDataFrame.
      - Dictionary mapping {instance_id: label}.
      - Sequence of (instance_id, label) tuples.

    Returns:
        Dict[str, str] mapping instance_id -> predicted label.
    """
    if isinstance(source, (str, Path)):
        file_path = Path(source)
        if not file_path.is_file():
            raise FileNotFoundError(f"Predictions file not found: {file_path}")

        preds: Dict[str, str] = {}
        with open(file_path, "r", encoding="utf-8-sig") as f:
            sample = f.read(2048)
            f.seek(0)
            delimiter = "\t" if "\t" in sample and "," not in sample else ","
            reader = csv.reader(f, delimiter=delimiter)

            first_row = next(reader, None)
            if not first_row:
                raise ValueError(f"Empty predictions file: {file_path}")

            has_header = (
                first_row[0].strip().lower() in ("instance_id", "id", "guid")
                or first_row[1].strip().lower() in ("label", "prediction", "pred")
            )

            if not has_header:
                if len(first_row) >= 2:
                    preds[first_row[0].strip()] = first_row[1].strip()

            for row_idx, row in enumerate(reader, start=2):
                if not row or (len(row) == 1 and not row[0].strip()):
                    continue
                if len(row) < 2:
                    raise ValueError(f"Malformed row {row_idx} in '{file_path}': expected >=2 columns.")
                i_id, lbl = row[0].strip(), row[1].strip()
                if not i_id:
                    continue
                preds[i_id] = lbl

        return preds

    elif isinstance(source, Mapping):
        return {str(k).strip(): str(v).strip() for k, v in source.items()}

    elif HAS_PANDAS and pd is not None and isinstance(source, pd.DataFrame):
        df = source
        c_id = id_col if id_col in df.columns else df.columns[0]
        c_pred = (
            pred_col
            if pred_col in df.columns
            else ("prediction" if "prediction" in df.columns else df.columns[1])
        )
        return {str(r[c_id]).strip(): str(r[c_pred]).strip() for _, r in df.iterrows()}

    elif hasattr(source, "columns") and hasattr(source, "iterrows"):
        # SimpleDataFrame
        df = source
        c_id = id_col if id_col in df.columns else df.columns[0]
        c_pred = (
            pred_col
            if pred_col in df.columns
            else ("prediction" if "prediction" in df.columns else df.columns[1])
        )
        return {str(r[c_id]).strip(): str(r[c_pred]).strip() for _, r in df.iterrows()}

    elif isinstance(source, Sequence):
        preds = {}
        for item in source:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                preds[str(item[0]).strip()] = str(item[1]).strip()
        return preds

    raise TypeError(f"Unsupported predictions source type: {type(source)}")


def extract_reciprocal_pairs_from_dataset(
    dataset: Union[str, Path, Sequence[Dict[str, Any]], Any],
) -> List[Tuple[str, str]]:
    """
    Extract paired reciprocal instance IDs (id_direct, id_reverse) from a PhrasIS split.
    Leverages pair_id and reverse_pair_id matching.

    Returns:
        List of (id_direct, id_reverse) tuples.
    """
    records: List[Dict[str, Any]] = []

    if isinstance(dataset, (str, Path)):
        p = Path(dataset)
        if not p.is_file():
            raise FileNotFoundError(f"Dataset file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = list(reader)
    elif HAS_PANDAS and pd is not None and isinstance(dataset, pd.DataFrame):
        records = dataset.to_dict("records")
    elif hasattr(dataset, "to_dict") and callable(getattr(dataset, "to_dict")):
        records = dataset.to_dict("records")
    elif isinstance(dataset, Sequence):
        records = [dict(r) if isinstance(r, dict) else r.__dict__ for r in dataset]
    else:
        raise TypeError(f"Unsupported dataset format: {type(dataset)}")

    by_id: Dict[str, Dict[str, Any]] = {}
    for r in records:
        inst_id = str(r.get("instance_id", "")).strip()
        if inst_id:
            by_id[inst_id] = r

    seen_ids: Set[str] = set()
    pairs: List[Tuple[str, str]] = []

    # Pass 1: Explicit reverse_pair_id pointer
    for inst_id, r in by_id.items():
        if inst_id in seen_ids:
            continue
        rev_id = str(r.get("reverse_pair_id") or "").strip()
        if rev_id and rev_id in by_id and rev_id not in seen_ids and rev_id != inst_id:
            pairs.append((inst_id, rev_id))
            seen_ids.add(inst_id)
            seen_ids.add(rev_id)

    # Pass 2: Fallback matching by pair_id
    grouped_by_pair: Dict[str, List[str]] = {}
    for inst_id, r in by_id.items():
        if inst_id in seen_ids:
            continue
        p_id = str(r.get("pair_id") or "").strip()
        if p_id:
            grouped_by_pair.setdefault(p_id, []).append(inst_id)

    for p_id, ids in grouped_by_pair.items():
        if len(ids) == 2:
            id1, id2 = ids[0], ids[1]
            if id1 not in seen_ids and id2 not in seen_ids:
                pairs.append((id1, id2))
                seen_ids.add(id1)
                seen_ids.add(id2)

    return pairs


def evaluate_matrices_from_data(
    dataset: Union[str, Path, Sequence[Dict[str, Any]], Any],
    predictions: Union[str, Path, Mapping[str, str], Any],
    labels: Sequence[str] = CANONICAL_LABELS,
) -> Tuple[ConfusionMatrixReport, DirectionalTransitionReport]:
    """
    Unified end-to-end evaluation pipeline:
      1. Loads gold labels and reciprocal pairs from the dataset.
      2. Loads predicted labels from Codabench CSV, DataFrame, or mapping.
      3. Computes ConfusionMatrixReport.
      4. Computes DirectionalTransitionReport.

    Returns:
        (ConfusionMatrixReport, DirectionalTransitionReport)
    """
    pred_map = load_predictions(predictions)

    # Extract gold labels
    records: List[Dict[str, Any]] = []
    if isinstance(dataset, (str, Path)):
        p = Path(dataset)
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = list(reader)
    elif HAS_PANDAS and pd is not None and isinstance(dataset, pd.DataFrame):
        records = dataset.to_dict("records")
    elif hasattr(dataset, "to_dict") and callable(getattr(dataset, "to_dict")):
        records = dataset.to_dict("records")
    elif isinstance(dataset, Sequence):
        records = [dict(r) if isinstance(r, dict) else r.__dict__ for r in dataset]

    gold_map: Dict[str, str] = {}
    for r in records:
        inst_id = str(r.get("instance_id", "")).strip()
        lbl = str(r.get("label", "")).strip()
        if inst_id and lbl:
            gold_map[inst_id] = lbl

    # 1. Confusion Matrix
    conf_report = compute_confusion_matrix(gold=gold_map, pred=pred_map, labels=labels)

    # 2. Reciprocal Pairs & Transition Matrix
    pairs = extract_reciprocal_pairs_from_dataset(dataset)
    trans_report = compute_directional_transition_matrix(
        pairs=pairs, predictions=pred_map, labels=labels
    )

    return conf_report, trans_report
