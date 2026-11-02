"""
src/evaluation/analysis.py

Syntactic Category Error Analysis & Qualitative Diagnosis Engine
for SemEval-2027 Task 2 (DiCo-NLI).

Conforms to SemEval-2027 Task 2 specification and ACL 2027 paper reporting.
Provides:
1. Syntactic Category Disaggregation:
   - Partitions PhrasIS dataset instances and reciprocal pairs by syntactic category:
     'nominal', 'verbal', 'prepositional', 'adverbial', 'quantificational'.
   - Computes official metrics (Weighted F1, SoftCons, HardCons) disaggregated by chunk category.
   - Computes Reversal Curse vulnerability metrics:
     - Entailment Loop Rate / Reversal Curse Severity Index (RCSI)
     - Equivalence Asymmetry Rate
     - Noise Collapse / Semantic Dropout Rate
2. Cross-Method Comparison:
   - Contrasts disaggregated performance across Baseline, BSA, CRL, SD, and CRL+SD.
   - Pinpoints which syntactic categories suffer most from the Reversal Curse.
   - Measures exact recovery deltas under Symmetrized Decoding (SD) and CRL.
3. Qualitative Error Case Extraction:
   - Extracts concrete failure examples per category.
   - Classifies error mechanisms:
     - REVERSAL_CURSE_LOOP (e.g. forward correct, backward failed / forward-forward trap)
     - EQUIVALENCE_ASYMMETRY (broken symmetry in equivalence relations)
     - DIRECTIONAL_FLIP (inverted entailment arrow)
     - NOISE_COLLAPSE (valid semantic relation collapsed to NEGATIVE_OTHER)
     - SEMANTIC_HALLUCINATION (spurious entailment on negative pairs)
   - Generates linguistic failure diagnosis with sentence pair (A, B), direct label, reverse label, and true labels.
4. Multi-Format Exporters:
   - Markdown summary report.
   - Publication-ready LaTeX tables for ACL research paper (`report/paper.tex`).
   - Structured JSON report for automated benchmark ingestion.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

logger = logging.getLogger(__name__)

# Optional pandas and numpy imports with graceful pure-Python fallbacks
try:
    import pandas as pd  # type: ignore
    HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore
    HAS_PANDAS = False

try:
    import numpy as np  # type: ignore
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False

from src.data.loader import (
    ALL_CATEGORIES,
    InstanceRecord,
    SyntacticCategory,
    extract_syntactic_category,
)
from src.data.reversal import (
    ALL_LABELS,
    ID_TO_LABEL,
    LABEL_TO_ID,
    REVERSIBLE_LABELS,
    ReciprocalPair,
    is_reversible_label,
    pair_instances,
    reverse_label,
    reverse_label_bsa,
)
from src.metrics.metrics import (
    ReversiblePair,
    TaskLabel,
    compute_consistency_report,
    compute_official_metrics,
    compute_weighted_f1,
    safe_divide,
)


# =============================================================================
# 1. Error Types and Linguistic Diagnostic Rules
# =============================================================================

class ErrorType(str):
    """Taxonomy of directional consistency and classification error types."""
    REVERSAL_CURSE_LOOP = "REVERSAL_CURSE_LOOP"
    EQUIVALENCE_ASYMMETRY = "EQUIVALENCE_ASYMMETRY"
    DIRECTIONAL_FLIP = "DIRECTIONAL_FLIP"
    NOISE_COLLAPSE = "NOISE_COLLAPSE"
    SEMANTIC_HALLUCINATION = "SEMANTIC_HALLUCINATION"
    GENERAL_MISCLASSIFICATION = "GENERAL_MISCLASSIFICATION"


def diagnose_error(
    category: str,
    text1: str,
    text2: str,
    gold_dir: str,
    gold_rev: str,
    pred_dir: str,
    pred_rev: str,
    is_soft_consistent: bool,
    is_hard_correct: bool,
) -> Tuple[str, str]:
    """
    Deterministically classifies an error mechanism and produces a detailed
    linguistic diagnosis tailored to the syntactic chunk category.
    """
    if is_hard_correct and is_soft_consistent:
        return ("NONE", "Correct and directionally consistent prediction.")

    # 1. Reversal Curse Entailment Loop (both directions predicted as same asymmetric entailment)
    is_gold_asymmetric = (
        gold_dir in ("FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT")
        or gold_rev in ("FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT")
    )
    if is_gold_asymmetric:
        if (
            (pred_dir == "FORWARD_ENTAILMENT" and pred_rev == "FORWARD_ENTAILMENT")
            or (pred_dir == "BACKWARD_ENTAILMENT" and pred_rev == "BACKWARD_ENTAILMENT")
        ):
            if category == SyntacticCategory.PREPOSITIONAL.value:
                diag = (
                    f"Reversal Curse Entailment Loop in Prepositional Chunk: Model correctly captures "
                    f"spatial/containment hierarchy in forward direction ('{text1}' -> '{text2}'), "
                    f"but repeats '{pred_dir}' under reversal ('{text2}' -> '{text1}') due to unpenalized "
                    f"directional bias, failing to invert locative inclusion."
                )
            elif category == SyntacticCategory.VERBAL.value:
                diag = (
                    f"Reversal Curse Entailment Loop in Verbal Chunk: Model detects event subsumption "
                    f"('{text1}' -> '{text2}'), but high lexical overlap causes spurious forward entailment "
                    f"under reversal ('{text2}' -> '{text1}'), failing to recognize that hyponymic verbs do not reverse."
                )
            elif category == SyntacticCategory.QUANTIFICATIONAL.value:
                diag = (
                    f"Reversal Curse Entailment Loop in Quantificational Chunk: Monotonicity entailment "
                    f"from specific to general bound correctly identified, but model fails to reverse "
                    f"the quantifier entailment arrow under argument reversal."
                )
            elif category == SyntacticCategory.NOMINAL.value:
                diag = (
                    f"Reversal Curse Entailment Loop in Nominal Chunk: Hypernymy hierarchy between '{text1}' "
                    f"and '{text2}' recognized unidirectionally, but inverted relation fails to invert to BACKWARD_ENTAILMENT."
                )
            else:
                diag = (
                    f"Reversal Curse Entailment Loop: Directional entailment '{pred_dir}' predicted symmetrically "
                    f"in both directions for asymmetric relation ({gold_dir} / {gold_rev})."
                )
            return (ErrorType.REVERSAL_CURSE_LOOP, diag)

    # 2. Equivalence Asymmetry
    is_gold_equiv = (gold_dir == "EQUIVALENCE" and gold_rev == "EQUIVALENCE")
    if is_gold_equiv:
        if pred_dir != pred_rev or (pred_dir not in ("EQUIVALENCE",) or pred_rev not in ("EQUIVALENCE",)):
            diag = (
                f"Equivalence Asymmetry in {category.capitalize()} Chunk: Ground truth relation is symmetric "
                f"equivalence ('{text1}' <=> '{text2}'), but model predicts asymmetric '{pred_dir}' forward "
                f"and '{pred_rev}' backward, violating bidirectional equivalence invariance."
            )
            return (ErrorType.EQUIVALENCE_ASYMMETRY, diag)

    # 3. Noise Collapse / Semantic Dropout
    if is_reversible_label(gold_dir) and is_reversible_label(gold_rev):
        if pred_dir == "NEGATIVE_OTHER" or pred_rev == "NEGATIVE_OTHER":
            diag = (
                f"Noise Collapse / Semantic Dropout in {category.capitalize()} Chunk: Valid semantic relation "
                f"({gold_dir} / {gold_rev}) between '{text1}' and '{text2}' collapsed into non-reversible "
                f"NEGATIVE_OTHER (direct='{pred_dir}', reverse='{pred_rev}')."
            )
            return (ErrorType.NOISE_COLLAPSE, diag)

    # 4. Semantic Hallucination (Negative pair predicted as entailment)
    if gold_dir == "NEGATIVE_OTHER" or gold_rev == "NEGATIVE_OTHER":
        if pred_dir in REVERSIBLE_LABELS or pred_rev in REVERSIBLE_LABELS:
            diag = (
                f"Semantic Hallucination in {category.capitalize()} Chunk: Non-entailing pair erroneously "
                f"predicted as {pred_dir} / {pred_rev} due to surface lexical or contextual similarity."
            )
            return (ErrorType.SEMANTIC_HALLUCINATION, diag)

    # 5. Directional Flip
    if (
        (gold_dir == "FORWARD_ENTAILMENT" and pred_dir == "BACKWARD_ENTAILMENT")
        or (gold_dir == "BACKWARD_ENTAILMENT" and pred_dir == "FORWARD_ENTAILMENT")
    ):
        diag = (
            f"Directional Entailment Flip in {category.capitalize()} Chunk: Model inverted the directional "
            f"entailment arrow (predicted '{pred_dir}' for gold '{gold_dir}')."
        )
        return (ErrorType.DIRECTIONAL_FLIP, diag)

    # 6. General Misclassification
    diag = (
        f"General Misclassification in {category.capitalize()} Chunk: Expected {gold_dir} / {gold_rev}, "
        f"predicted {pred_dir} / {pred_rev}."
    )
    return (ErrorType.GENERAL_MISCLASSIFICATION, diag)


# =============================================================================
# 2. Data Structures
# =============================================================================

@dataclass(frozen=True)
class ErrorCase:
    """Represents a single diagnosed qualitative failure case on a reciprocal pair."""
    pair_id: str
    category: str
    text1: str
    text2: str
    gold_dir: str
    gold_rev: str
    pred_dir: str
    pred_rev: str
    is_soft_consistent: bool
    is_hard_correct: bool
    error_type: str
    diagnosis: str
    method: str = "baseline"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CategoryMetrics:
    """Disaggregated performance and directional metrics for a single syntactic category."""
    category: str
    num_instances: int
    num_pairs: int
    num_reversible_pairs: int
    weighted_f1: float
    soft_cons: float
    hard_cons: float
    entailment_loop_count: int
    entailment_loop_rate: float
    equiv_asymmetry_count: int
    equiv_asymmetry_rate: float
    noise_collapse_count: int
    noise_collapse_rate: float
    reversal_curse_index: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DisaggregatedAnalysisReport:
    """Master disaggregated report across all syntactic categories for a given method."""
    method: str
    track: str
    overall_weighted_f1: float
    overall_soft_cons: float
    overall_hard_cons: float
    total_instances: int
    total_pairs: int
    categories: Dict[str, CategoryMetrics]
    vulnerable_categories: List[Tuple[str, float]]  # Sorted by vulnerability (RCSI descending)
    error_cases: List[ErrorCase]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "track": self.track,
            "overall_weighted_f1": self.overall_weighted_f1,
            "overall_soft_cons": self.overall_soft_cons,
            "overall_hard_cons": self.overall_hard_cons,
            "total_instances": self.total_instances,
            "total_pairs": self.total_pairs,
            "categories": {k: v.to_dict() for k, v in self.categories.items()},
            "vulnerable_categories": self.vulnerable_categories,
            "error_cases": [e.to_dict() for e in self.error_cases],
        }


@dataclass(frozen=True)
class CrossMethodCategoryReport:
    """Comparative report contrasting performance across multiple methods per category."""
    track: str
    methods: List[str]
    categories: List[str]
    metrics: Dict[str, Dict[str, CategoryMetrics]]  # [method][category] -> CategoryMetrics
    deltas_vs_baseline: Dict[str, Dict[str, Dict[str, float]]]  # [method][category] -> deltas
    most_vulnerable_category_baseline: str
    most_improved_category_sd: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track": self.track,
            "methods": self.methods,
            "categories": self.categories,
            "metrics": {
                m: {c: met.to_dict() for c, met in c_dict.items()}
                for m, c_dict in self.metrics.items()
            },
            "deltas_vs_baseline": self.deltas_vs_baseline,
            "most_vulnerable_category_baseline": self.most_vulnerable_category_baseline,
            "most_improved_category_sd": self.most_improved_category_sd,
        }


# =============================================================================
# 3. Helper Normalization Functions
# =============================================================================

def _normalize_prediction_map(
    predictions: Union[Mapping[str, str], Sequence[Dict[str, Any]], Any],
) -> Dict[str, str]:
    """Converts diverse prediction formats into a canonical dict {instance_id: label}."""
    if isinstance(predictions, dict):
        return {str(k): str(v).strip() for k, v in predictions.items()}
    elif isinstance(predictions, (list, tuple)):
        out: Dict[str, str] = {}
        for item in predictions:
            if isinstance(item, dict):
                iid = str(item.get("instance_id") or "")
                lbl = str(item.get("label") or item.get("prediction") or "")
                if iid:
                    out[iid] = lbl
        return out
    elif HAS_PANDAS and pd is not None and isinstance(predictions, pd.DataFrame):
        out = {}
        for _, row in predictions.iterrows():
            iid = str(row.get("instance_id", ""))
            lbl = str(row.get("label", ""))
            if iid:
                out[iid] = lbl
        return out
    return {}


def _extract_record_category(record: Any) -> str:
    """Extracts or infers syntactic category from a record."""
    cat = None
    if isinstance(record, dict) or hasattr(record, "__getitem__"):
        try:
            cat = record.get("category") if hasattr(record, "get") else record["category"]
        except (KeyError, TypeError, IndexError):
            pass
    if cat is None:
        cat = getattr(record, "category", None)

    if cat and str(cat).strip().lower() in ALL_CATEGORIES:
        return str(cat).strip().lower()

    # Dynamic extraction fallback
    t1 = str(record.get("text1", "") if hasattr(record, "get") else getattr(record, "text1", ""))
    t2 = str(record.get("text2", "") if hasattr(record, "get") else getattr(record, "text2", ""))
    lang = str(record.get("text1_lang", "en") if hasattr(record, "get") else getattr(record, "text1_lang", "en"))
    return extract_syntactic_category(t1, t2, lang=lang)


def _records_from_data(dataset: Any) -> List[Dict[str, Any]]:
    """Normalizes any dataset container (DataFrame, SimpleDataFrame, list) to list of dicts."""
    if HAS_PANDAS and pd is not None and isinstance(dataset, pd.DataFrame):
        return dataset.to_dict("records")
    elif hasattr(dataset, "to_dict") and callable(getattr(dataset, "to_dict")):
        try:
            return dataset.to_dict("records")
        except (TypeError, ValueError):
            return list(dataset)
    elif isinstance(dataset, (list, tuple)):
        return [
            r.to_dict() if hasattr(r, "to_dict") and callable(getattr(r, "to_dict"))
            else (dict(r) if hasattr(r, "keys") else r.__dict__)
            for r in dataset
        ]
    return list(dataset)


# =============================================================================
# 4. Syntactic Category Disaggregation Engine
# =============================================================================

def compute_category_metrics(
    records: Sequence[Any],
    pairs: Sequence[ReciprocalPair],
    predicted_labels: Mapping[str, str],
    category: str,
) -> CategoryMetrics:
    """
    Computes all disaggregated metrics for a specific syntactic category.
    """
    cat_norm = category.strip().lower()

    # 1. Filter item-level records belonging to this category
    cat_records = [r for r in records if _extract_record_category(r) == cat_norm]
    num_instances = len(cat_records)

    if num_instances > 0:
        gold_labels = []
        pred_labels = []
        for r in cat_records:
            iid = str(r.get("instance_id", "") if hasattr(r, "get") else getattr(r, "instance_id", ""))
            gold = str(r.get("label", "") if hasattr(r, "get") else getattr(r, "label", ""))
            pred = predicted_labels.get(iid, "NEGATIVE_OTHER")
            if gold in ALL_LABELS:
                gold_labels.append(gold)
                pred_labels.append(pred)
        weighted_f1 = compute_weighted_f1(gold_labels, pred_labels) if gold_labels else 0.0
    else:
        weighted_f1 = 0.0

    # 2. Filter reciprocal pairs belonging to this category
    cat_pairs = [
        p for p in pairs
        if _extract_record_category(p.direct_instance) == cat_norm
        or _extract_record_category(p.inverse_instance) == cat_norm
    ]
    num_pairs = len(cat_pairs)
    reversible_pairs = [p for p in cat_pairs if p.is_reversible]
    num_rev = len(reversible_pairs)

    # 3. Calculate SoftCons, HardCons, and Inconsistency breakdowns
    soft_count = 0
    hard_count = 0
    entailment_loop_count = 0
    equiv_asymmetry_count = 0
    noise_collapse_count = 0
    dir_entailment_count = 0

    for p in reversible_pairs:
        d = p.direct_instance
        inv = p.inverse_instance
        id_d = str(d.get("instance_id", "") if hasattr(d, "get") else getattr(d, "instance_id", ""))
        id_r = str(inv.get("instance_id", "") if hasattr(inv, "get") else getattr(inv, "instance_id", ""))
        g_d = str(d.get("label", "") if hasattr(d, "get") else getattr(d, "label", ""))
        g_r = str(inv.get("label", "") if hasattr(inv, "get") else getattr(inv, "label", ""))

        p_d = predicted_labels.get(id_d, "NEGATIVE_OTHER")
        p_r = predicted_labels.get(id_r, "NEGATIVE_OTHER")

        is_soft = (
            p_d in REVERSIBLE_LABELS
            and p_r in REVERSIBLE_LABELS
            and p_r == reverse_label_bsa(p_d)
        )
        is_hard = (p_d == g_d and p_r == g_r)

        if is_soft:
            soft_count += 1
        if is_hard:
            hard_count += 1

        # Track Reversal Curse Entailment Loop
        is_asym_gold = (
            g_d in ("FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT")
            or g_r in ("FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT")
        )
        if is_asym_gold:
            dir_entailment_count += 1
            if (p_d == "FORWARD_ENTAILMENT" and p_r == "FORWARD_ENTAILMENT") or (
                p_d == "BACKWARD_ENTAILMENT" and p_r == "BACKWARD_ENTAILMENT"
            ):
                entailment_loop_count += 1

        # Track Equivalence Asymmetry
        if g_d == "EQUIVALENCE" and g_r == "EQUIVALENCE":
            if p_d != p_r or p_d not in ("EQUIVALENCE",) or p_r not in ("EQUIVALENCE",):
                equiv_asymmetry_count += 1

        # Track Noise Collapse
        if p_d == "NEGATIVE_OTHER" or p_r == "NEGATIVE_OTHER":
            noise_collapse_count += 1

    soft_cons = safe_divide(soft_count, num_rev)
    hard_cons = safe_divide(hard_count, num_rev)
    entailment_loop_rate = safe_divide(entailment_loop_count, num_rev)
    equiv_asymmetry_rate = safe_divide(equiv_asymmetry_count, num_rev)
    noise_collapse_rate = safe_divide(noise_collapse_count, num_rev)
    rcsi = safe_divide(entailment_loop_count, dir_entailment_count)

    return CategoryMetrics(
        category=cat_norm,
        num_instances=num_instances,
        num_pairs=num_pairs,
        num_reversible_pairs=num_rev,
        weighted_f1=weighted_f1,
        soft_cons=soft_cons,
        hard_cons=hard_cons,
        entailment_loop_count=entailment_loop_count,
        entailment_loop_rate=entailment_loop_rate,
        equiv_asymmetry_count=equiv_asymmetry_count,
        equiv_asymmetry_rate=equiv_asymmetry_rate,
        noise_collapse_count=noise_collapse_count,
        noise_collapse_rate=noise_collapse_rate,
        reversal_curse_index=rcsi,
    )


def extract_qualitative_error_cases(
    dataset: Any,
    predictions: Union[Mapping[str, str], Any],
    method: str = "baseline",
    max_cases_per_category: int = 5,
    target_categories: Optional[Sequence[str]] = None,
) -> List[ErrorCase]:
    """
    Extracts concrete, diagnosed failure cases grouped by syntactic category.
    """
    records = _records_from_data(dataset)
    pred_map = _normalize_prediction_map(predictions)
    pairs = pair_instances(records)

    cats_filter = (
        {c.strip().lower() for c in target_categories}
        if target_categories is not None
        else None
    )

    cases_by_cat: Dict[str, List[ErrorCase]] = {}

    for p in pairs:
        d = p.direct_instance
        inv = p.inverse_instance
        cat = _extract_record_category(d)

        if cats_filter and cat not in cats_filter:
            continue

        id_d = str(d.get("instance_id", "") if hasattr(d, "get") else getattr(d, "instance_id", ""))
        id_r = str(inv.get("instance_id", "") if hasattr(inv, "get") else getattr(inv, "instance_id", ""))
        t1 = str(d.get("text1", "") if hasattr(d, "get") else getattr(d, "text1", ""))
        t2 = str(d.get("text2", "") if hasattr(d, "get") else getattr(d, "text2", ""))
        g_d = str(d.get("label", "") if hasattr(d, "get") else getattr(d, "label", ""))
        g_r = str(inv.get("label", "") if hasattr(inv, "get") else getattr(inv, "label", ""))

        p_d = pred_map.get(id_d, "NEGATIVE_OTHER")
        p_r = pred_map.get(id_r, "NEGATIVE_OTHER")

        is_soft = (
            p_d in REVERSIBLE_LABELS
            and p_r in REVERSIBLE_LABELS
            and p_r == reverse_label_bsa(p_d)
        )
        is_hard = (p_d == g_d and p_r == g_r)

        # We extract cases that have at least one error (either inconsistent or incorrect)
        if not (is_soft and is_hard):
            err_type, diag = diagnose_error(
                category=cat,
                text1=t1,
                text2=t2,
                gold_dir=g_d,
                gold_rev=g_r,
                pred_dir=p_d,
                pred_rev=p_r,
                is_soft_consistent=is_soft,
                is_hard_correct=is_hard,
            )

            ec = ErrorCase(
                pair_id=p.pair_id,
                category=cat,
                text1=t1,
                text2=t2,
                gold_dir=g_d,
                gold_rev=g_r,
                pred_dir=p_d,
                pred_rev=p_r,
                is_soft_consistent=is_soft,
                is_hard_correct=is_hard,
                error_type=err_type,
                diagnosis=diag,
                method=method,
            )
            cases_by_cat.setdefault(cat, []).append(ec)

    # Flatten respecting max_cases_per_category
    results: List[ErrorCase] = []
    for cat in sorted(cases_by_cat.keys()):
        results.extend(cases_by_cat[cat][:max_cases_per_category])

    return results


def analyze_syntactic_categories(
    dataset: Any,
    predictions: Union[Mapping[str, str], Any],
    method: str = "baseline",
    track: str = "track1",
    max_error_cases_per_category: int = 3,
) -> DisaggregatedAnalysisReport:
    """
    Performs full disaggregated syntactic chunk analysis across all categories.
    """
    records = _records_from_data(dataset)
    pred_map = _normalize_prediction_map(predictions)
    pairs = pair_instances(records)

    # Detect present categories
    present_cats: Set[str] = set()
    for r in records:
        present_cats.add(_extract_record_category(r))

    # Fallback to standard 5 categories if empty
    target_cats = sorted(present_cats) if present_cats else sorted(ALL_CATEGORIES)

    cat_metrics: Dict[str, CategoryMetrics] = {}
    for cat in target_cats:
        met = compute_category_metrics(
            records=records,
            pairs=pairs,
            predicted_labels=pred_map,
            category=cat,
        )
        cat_metrics[cat] = met

    # Overall metrics across entire dataset
    all_golds = [str(r.get("label", "")) for r in records if str(r.get("label", "")) in ALL_LABELS]
    all_preds = [pred_map.get(str(r.get("instance_id", "")), "NEGATIVE_OTHER") for r in records if str(r.get("label", "")) in ALL_LABELS]
    overall_f1 = compute_weighted_f1(all_golds, all_preds) if all_golds else 0.0

    rev_pairs = [p for p in pairs if p.is_reversible]
    tot_rev = len(rev_pairs)
    soft_c = sum(
        1 for p in rev_pairs
        if pred_map.get(str(p.direct_instance.get("instance_id", ""))) in REVERSIBLE_LABELS
        and pred_map.get(str(p.inverse_instance.get("instance_id", "")))
        == reverse_label_bsa(pred_map.get(str(p.direct_instance.get("instance_id", ""))))
    )
    hard_c = sum(
        1 for p in rev_pairs
        if pred_map.get(str(p.direct_instance.get("instance_id", ""))) == p.direct_instance.get("label")
        and pred_map.get(str(p.inverse_instance.get("instance_id", ""))) == p.inverse_instance.get("label")
    )
    overall_soft = safe_divide(soft_c, tot_rev)
    overall_hard = safe_divide(hard_c, tot_rev)

    # Rank categories by Reversal Curse vulnerability (descending RCSI, then ascending SoftCons)
    ranked = sorted(
        cat_metrics.items(),
        key=lambda item: (-item[1].reversal_curse_index, item[1].soft_cons),
    )
    vulnerable_list = [(cat, met.reversal_curse_index) for cat, met in ranked]

    error_cases = extract_qualitative_error_cases(
        dataset=dataset,
        predictions=pred_map,
        method=method,
        max_cases_per_category=max_error_cases_per_category,
    )

    return DisaggregatedAnalysisReport(
        method=method,
        track=track,
        overall_weighted_f1=overall_f1,
        overall_soft_cons=overall_soft,
        overall_hard_cons=overall_hard,
        total_instances=len(records),
        total_pairs=len(pairs),
        categories=cat_metrics,
        vulnerable_categories=vulnerable_list,
        error_cases=error_cases,
    )


def compare_methods_syntactic(
    dataset: Any,
    method_predictions: Mapping[str, Union[Mapping[str, str], Any]],
    baseline_method: str = "baseline",
    track: str = "track1",
) -> CrossMethodCategoryReport:
    """
    Contrasts performance across multiple methods per syntactic category.
    Computes delta gains relative to baseline_method.
    """
    records = _records_from_data(dataset)
    methods = list(method_predictions.keys())

    reports_by_method: Dict[str, DisaggregatedAnalysisReport] = {}
    for m in methods:
        pred_map = _normalize_prediction_map(method_predictions[m])
        reports_by_method[m] = analyze_syntactic_categories(
            dataset=records,
            predictions=pred_map,
            method=m,
            track=track,
        )

    # Identify all categories
    all_cats = sorted(
        next(iter(reports_by_method.values())).categories.keys()
    ) if reports_by_method else sorted(ALL_CATEGORIES)

    metrics_matrix: Dict[str, Dict[str, CategoryMetrics]] = {
        m: rep.categories for m, rep in reports_by_method.items()
    }

    # Compute deltas relative to baseline
    deltas: Dict[str, Dict[str, Dict[str, float]]] = {}
    base_metrics = metrics_matrix.get(baseline_method, {})

    for m in methods:
        deltas[m] = {}
        for c in all_cats:
            cur = metrics_matrix.get(m, {}).get(c)
            base = base_metrics.get(c)
            if cur and base:
                deltas[m][c] = {
                    "delta_weighted_f1": cur.weighted_f1 - base.weighted_f1,
                    "delta_soft_cons": cur.soft_cons - base.soft_cons,
                    "delta_hard_cons": cur.hard_cons - base.hard_cons,
                    "delta_rcsi": cur.reversal_curse_index - base.reversal_curse_index,
                }
            else:
                deltas[m][c] = {
                    "delta_weighted_f1": 0.0,
                    "delta_soft_cons": 0.0,
                    "delta_hard_cons": 0.0,
                    "delta_rcsi": 0.0,
                }

    # Find most vulnerable category under baseline
    base_rep = reports_by_method.get(baseline_method)
    most_vulnerable = base_rep.vulnerable_categories[0][0] if (base_rep and base_rep.vulnerable_categories) else "prepositional"

    # Find category with largest SoftCons gain under SD
    sd_method = next((m for m in methods if m in ("sd", "crl_sd")), methods[-1] if methods else "")
    most_improved = "prepositional"
    best_gain = -1.0
    if sd_method in deltas:
        for c, d in deltas[sd_method].items():
            gain = d.get("delta_soft_cons", 0.0)
            if gain > best_gain:
                best_gain = gain
                most_improved = c

    return CrossMethodCategoryReport(
        track=track,
        methods=methods,
        categories=all_cats,
        metrics=metrics_matrix,
        deltas_vs_baseline=deltas,
        most_vulnerable_category_baseline=most_vulnerable,
        most_improved_category_sd=most_improved,
    )


# =============================================================================
# 5. Multi-Format Exporters: Markdown, LaTeX, JSON
# =============================================================================

def format_markdown_report(
    report: DisaggregatedAnalysisReport,
    cross_report: Optional[CrossMethodCategoryReport] = None,
) -> str:
    """Renders a comprehensive, publication-grade Markdown analysis report."""
    md = [
        f"# Syntactic Category Error Analysis: {report.method.upper()} ({report.track.upper()})",
        "",
        f"- **Method:** `{report.method}`",
        f"- **Benchmark Track:** `{report.track}`",
        f"- **Total Instances:** {report.total_instances}",
        f"- **Total Reciprocal Pairs:** {report.total_pairs}",
        f"- **Overall Weighted F1:** {report.overall_weighted_f1:.4f}",
        f"- **Overall SoftCons:** {report.overall_soft_cons:.4f}",
        f"- **Overall HardCons:** {report.overall_hard_cons:.4f}",
        "",
        "## 1. Disaggregated Performance by Syntactic Chunk Category",
        "",
        "| Category | Instances | Pairs | Weighted F1 | SoftCons | HardCons | Entailment Loops | Equiv Asym | RCSI (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for cat_name, m in sorted(report.categories.items()):
        rcsi_pct = f"{m.reversal_curse_index * 100:.1f}%"
        md.append(
            f"| **{cat_name.capitalize()}** | {m.num_instances} | {m.num_pairs} | "
            f"{m.weighted_f1:.4f} | {m.soft_cons:.4f} | {m.hard_cons:.4f} | "
            f"{m.entailment_loop_count} | {m.equiv_asymmetry_count} | {rcsi_pct} |"
        )

    md.extend([
        "",
        "### Key Linguistic Vulnerability Findings",
        f"- **Most Vulnerable Category to Reversal Curse:** `{report.vulnerable_categories[0][0]}` "
        f"(RCSI: {report.vulnerable_categories[0][1] * 100:.1f}%).",
    ])

    if len(report.vulnerable_categories) > 1:
        md.append(
            f"- **Secondary Vulnerability:** `{report.vulnerable_categories[1][0]}` "
            f"(RCSI: {report.vulnerable_categories[1][1] * 100:.1f}%)."
        )

    # Cross-method section if available
    if cross_report is not None:
        md.extend([
            "",
            "## 2. Cross-Method Contrast Across Syntactic Categories",
            "",
            "Comparative evaluation contrasting SoftCons across Reversal Curse mitigation methods:",
            "",
        ])
        header_methods = " | ".join(f"**{m.upper()} SoftCons**" for m in cross_report.methods)
        dividers = " | ".join(":---:" for _ in cross_report.methods)
        md.append(f"| Category | {header_methods} |")
        md.append(f"| :--- | {dividers} |")

        for cat in cross_report.categories:
            vals = []
            for m in cross_report.methods:
                met = cross_report.metrics.get(m, {}).get(cat)
                val_str = f"{met.soft_cons:.4f}" if met else "N/A"
                vals.append(val_str)
            md.append(f"| **{cat.capitalize()}** | {' | '.join(vals)} |")

        md.extend([
            "",
            f"- **Most improved category under Symmetrized Decoding:** `{cross_report.most_improved_category_sd}`.",
        ])

    # Qualitative Error Cases Section
    md.extend([
        "",
        "## 3. Qualitative Error Case Diagnoses",
        "",
        "Selected representative failure examples demonstrating directional inconsistency mechanisms:",
        "",
    ])

    if not report.error_cases:
        md.append("_No error cases identified: 100% directional consistency and joint correctness achieved!_")
    else:
        for idx, ec in enumerate(report.error_cases, start=1):
            soft_icon = "✅" if ec.is_soft_consistent else "❌"
            hard_icon = "✅" if ec.is_hard_correct else "❌"
            md.extend([
                f"### Case {idx}: [{ec.category.upper()}] Pair `{ec.pair_id}`",
                f"- **Phrase A:** \"{ec.text1}\"",
                f"- **Phrase B:** \"{ec.text2}\"",
                f"- **Ground Truth:** Direct: `{ec.gold_dir}` | Reverse: `{ec.gold_rev}`",
                f"- **Prediction ({ec.method}):** Direct: `{ec.pred_dir}` | Reverse: `{ec.pred_rev}`",
                f"- **Consistency:** SoftCons: {soft_icon} (`{ec.is_soft_consistent}`) | HardCons: {hard_icon} (`{ec.is_hard_correct}`)",
                f"- **Error Taxonomy:** `{ec.error_type}`",
                f"- **Diagnosis:** {ec.diagnosis}",
                "",
            ])

    return "\n".join(md)


def format_latex_tables(
    report: DisaggregatedAnalysisReport,
    cross_report: Optional[CrossMethodCategoryReport] = None,
) -> Dict[str, str]:
    """
    Generates LaTeX table code ready for direct inclusion in the ACL research paper (`report/paper.tex`).
    """
    tables: Dict[str, str] = {}

    # 1. Disaggregated Performance Table
    s1 = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "\\textbf{Syntactic Category} & \\textbf{Instances} & \\textbf{Pairs} & \\textbf{Weighted F1} & \\textbf{SoftCons} & \\textbf{HardCons} & \\textbf{RCSI (\\%)} \\\\",
        "\\midrule",
    ]
    for cat_name, m in sorted(report.categories.items()):
        rcsi_pct = f"{m.reversal_curse_index * 100:.1f}\\%"
        s1.append(
            f"{cat_name.capitalize()} & {m.num_instances} & {m.num_pairs} & "
            f"{m.weighted_f1:.4f} & {m.soft_cons:.4f} & {m.hard_cons:.4f} & {rcsi_pct} \\\\"
        )
    s1.extend([
        "\\midrule",
        f"\\textbf{{Overall}} & \\textbf{{{report.total_instances}}} & \\textbf{{{report.total_pairs}}} & "
        f"\\textbf{{{report.overall_weighted_f1:.4f}}} & \\textbf{{{report.overall_soft_cons:.4f}}} & "
        f"\\textbf{{{report.overall_hard_cons:.4f}}} & -- \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{Disaggregated performance across PhrasIS syntactic chunk categories for {report.method} on {report.track}.}}",
        "\\label{tab:syntactic_breakdown}",
        "\\end{table*}",
    ])
    tables["category_performance_table"] = "\n".join(s1)

    # 2. Cross-Method Table if provided
    if cross_report is not None:
        num_m = len(cross_report.methods)
        col_spec = "l|" + ("cc|" * (num_m - 1)) + "cc"
        s2 = [
            "\\begin{table*}[t]",
            "\\centering",
            "\\small",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule",
        ]

        top_headers = []
        for m in cross_report.methods:
            top_headers.append(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{m.upper()}}}}}")
        s2.append("\\textbf{Category} & " + " & ".join(top_headers) + " \\\\")

        sub_headers = " & ".join(["\\textbf{F1} & \\textbf{SoftCons}"] * num_m)
        s2.append(" & " + sub_headers + " \\\\")
        s2.append("\\midrule")

        for cat in cross_report.categories:
            row_items = [cat.capitalize()]
            for m in cross_report.methods:
                met = cross_report.metrics.get(m, {}).get(cat)
                if met:
                    f1_str = f"{met.weighted_f1:.3f}"
                    soft_str = f"{met.soft_cons:.3f}"
                    if met.soft_cons >= 0.999:
                        soft_str = f"\\textbf{{{soft_str}}}"
                    row_items.extend([f1_str, soft_str])
                else:
                    row_items.extend(["--", "--"])
            s2.append(" & ".join(row_items) + " \\\\")

        s2.extend([
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{Comparative cross-method performance by syntactic category on {cross_report.track}.}}",
            "\\label{tab:cross_method_syntactic}",
            "\\end{table*}",
        ])
        tables["cross_method_table"] = "\n".join(s2)

    # 3. Qualitative Error Cases Table
    s3 = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\footnotesize",
        "\\begin{tabular}{p{2.0cm}p{3.2cm}p{3.2cm}ccp{5.0cm}}",
        "\\toprule",
        "\\textbf{Category} & \\textbf{Phrase A} & \\textbf{Phrase B} & \\textbf{Gold (Dir/Rev)} & \\textbf{Pred (Dir/Rev)} & \\textbf{Diagnosis} \\\\",
        "\\midrule",
    ]

    def _tex_escape(text: str) -> str:
        return text.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")

    for ec in report.error_cases[:5]:
        cat_str = ec.category.capitalize()
        t1_esc = _tex_escape(ec.text1)
        t2_esc = _tex_escape(ec.text2)
        gold_short = f"{ec.gold_dir[:3]}/{ec.gold_rev[:3]}"
        pred_short = f"{ec.pred_dir[:3]}/{ec.pred_rev[:3]}"
        diag_esc = _tex_escape(ec.diagnosis[:120] + ("..." if len(ec.diagnosis) > 120 else ""))
        s3.append(f"{cat_str} & {t1_esc} & {t2_esc} & {gold_short} & {pred_short} & {diag_esc} \\\\")

    s3.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Representative qualitative failure cases and linguistic error diagnoses.}",
        "\\label{tab:qualitative_error_cases}",
        "\\end{table*}",
    ])
    tables["error_cases_table"] = "\n".join(s3)

    return tables


def export_analysis_artifacts(
    report: DisaggregatedAnalysisReport,
    cross_report: Optional[CrossMethodCategoryReport] = None,
    output_dir: Union[str, Path] = "results/analysis",
) -> Dict[str, Path]:
    """
    Exports Markdown, LaTeX, and JSON artifacts to disk.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created: Dict[str, Path] = {}

    prefix = f"{report.track}_{report.method}"

    # 1. Markdown
    md_content = format_markdown_report(report, cross_report=cross_report)
    md_path = out_dir / f"{prefix}_syntactic_analysis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    created["markdown"] = md_path

    # 2. JSON
    json_path = out_dir / f"{prefix}_syntactic_analysis.json"
    full_json = {
        "report": report.to_dict(),
        "cross_report": cross_report.to_dict() if cross_report else None,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_json, f, indent=2)
    created["json"] = json_path

    # 3. LaTeX Tables
    latex_tables = format_latex_tables(report, cross_report=cross_report)
    tex_path = out_dir / f"{prefix}_syntactic_tables.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by src/evaluation/analysis.py for ACL 2027 paper\n\n")
        for tbl_name, tbl_code in latex_tables.items():
            f.write(f"% --- {tbl_name} ---\n{tbl_code}\n\n")
    created["latex"] = tex_path

    logger.info("Exported syntactic error analysis artifacts to: %s", out_dir)
    return created
