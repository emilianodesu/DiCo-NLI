"""
src/evaluation/__init__.py

Milestone 4: Evaluation and Error Analysis Suite for SemEval-2027 Task 2 (DiCo-NLI).
Provides:
  - Master comparison tables and multi-format exporters (Markdown, LaTeX, ASCII, JSON)
  - 4x4 multiclass confusion and directional transition matrices
  - Syntactic category disaggregated performance and qualitative error diagnosis
"""

from __future__ import annotations

# Tables module
from .tables import (
    CANONICAL_METHODS,
    DEFAULT_MODELS,
    METHOD_DESCRIPTIONS,
    METHOD_DISPLAY_NAMES,
    TRACK_DISPLAY_NAMES,
    ComparisonTable,
    EvaluationResult,
    aggregate_results_from_dir,
    build_comparison_table,
    evaluate_prediction_file,
    evaluate_prediction_mapping,
    format_comparison_table,
    generate_ascii_table,
    generate_latex_table,
    generate_markdown_table,
    load_predictions_from_csv,
    parse_run_metadata,
)

# Matrices module
from .matrices import (
    CANONICAL_IDX_TO_LABEL,
    CANONICAL_LABELS,
    IDEAL_INVOLUTION_MATRIX,
    LABEL_SHORT_NAMES,
    LABEL_TO_CANONICAL_IDX,
    REVERSE_LABEL_MAP,
    ConfusionMatrixReport,
    DirectionalTransitionReport,
    PerClassMetrics,
    TransitionInconsistencyMetrics,
    compute_confusion_matrix,
    compute_directional_transition_matrix,
    evaluate_matrices_from_data,
    extract_reciprocal_pairs_from_dataset,
    get_ideal_involution_matrix,
    load_predictions,
)

# Syntactic analysis module
from .analysis import (
    CategoryMetrics,
    CrossMethodCategoryReport,
    DisaggregatedAnalysisReport,
    ErrorCase,
    ErrorType,
    analyze_syntactic_categories,
    compare_methods_syntactic,
    compute_category_metrics,
    diagnose_error,
    export_analysis_artifacts,
    extract_qualitative_error_cases,
    format_latex_tables,
    format_markdown_report,
)

__all__ = [
    # Tables
    "CANONICAL_METHODS",
    "DEFAULT_MODELS",
    "METHOD_DESCRIPTIONS",
    "METHOD_DISPLAY_NAMES",
    "TRACK_DISPLAY_NAMES",
    "ComparisonTable",
    "EvaluationResult",
    "aggregate_results_from_dir",
    "build_comparison_table",
    "evaluate_prediction_file",
    "evaluate_prediction_mapping",
    "format_comparison_table",
    "generate_ascii_table",
    "generate_latex_table",
    "generate_markdown_table",
    "load_predictions_from_csv",
    "parse_run_metadata",
    # Matrices
    "CANONICAL_IDX_TO_LABEL",
    "CANONICAL_LABELS",
    "IDEAL_INVOLUTION_MATRIX",
    "LABEL_SHORT_NAMES",
    "LABEL_TO_CANONICAL_IDX",
    "REVERSE_LABEL_MAP",
    "ConfusionMatrixReport",
    "DirectionalTransitionReport",
    "PerClassMetrics",
    "TransitionInconsistencyMetrics",
    "compute_confusion_matrix",
    "compute_directional_transition_matrix",
    "evaluate_matrices_from_data",
    "extract_reciprocal_pairs_from_dataset",
    "get_ideal_involution_matrix",
    "load_predictions",
    # Analysis
    "CategoryMetrics",
    "CrossMethodCategoryReport",
    "DisaggregatedAnalysisReport",
    "ErrorCase",
    "ErrorType",
    "analyze_syntactic_categories",
    "compare_methods_syntactic",
    "compute_category_metrics",
    "diagnose_error",
    "export_analysis_artifacts",
    "extract_qualitative_error_cases",
    "format_latex_tables",
    "format_markdown_report",
]
