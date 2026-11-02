"""
src/metrics package initialization.
SemEval-2027 Task 2 (DiCo-NLI).
"""

from .metrics import (
    ALL_LABELS,
    ALL_LABELS_TUPLE,
    ClassificationReport,
    ConsistencyReport,
    EvaluationReport,
    IDX_TO_LABEL,
    INVOLUTION_PERMUTATION_MATRIX,
    LABEL_TO_IDX,
    PerLabelMetrics,
    REVERSIBLE_LABELS,
    REVERSIBLE_LABELS_TUPLE,
    REVERSE_LABEL_MAP,
    ReversiblePair,
    ReversiblePairSpec,
    TaskLabel,
    compute_classification_report,
    compute_consistency_report,
    compute_metrics_from_arrays,
    compute_official_evaluation,
    compute_official_metrics,
    compute_weighted_f1,
    is_reversible_label,
    reverse_label,
    safe_divide,
)
from .consistency import (
    DirectionalTransitionReport,
    InconsistencyTaxonomyBreakdown,
    Matrix2D,
    compute_consistency_by_category,
    compute_directional_transition_matrix,
)

__all__ = [
    # Metrics
    "TaskLabel",
    "LABEL_TO_IDX",
    "IDX_TO_LABEL",
    "ALL_LABELS",
    "ALL_LABELS_TUPLE",
    "REVERSIBLE_LABELS",
    "REVERSIBLE_LABELS_TUPLE",
    "REVERSE_LABEL_MAP",
    "INVOLUTION_PERMUTATION_MATRIX",
    "ReversiblePair",
    "ReversiblePairSpec",
    "PerLabelMetrics",
    "ClassificationReport",
    "ConsistencyReport",
    "EvaluationReport",
    "safe_divide",
    "compute_weighted_f1",
    "compute_classification_report",
    "compute_consistency_report",
    "compute_official_metrics",
    "compute_official_evaluation",
    "compute_metrics_from_arrays",
    "is_reversible_label",
    "reverse_label",
    # Consistency
    "Matrix2D",
    "InconsistencyTaxonomyBreakdown",
    "DirectionalTransitionReport",
    "compute_directional_transition_matrix",
    "compute_consistency_by_category",
]
