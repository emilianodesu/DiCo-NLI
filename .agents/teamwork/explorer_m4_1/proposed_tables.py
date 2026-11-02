"""
src/evaluation/tables.py

Master Comparison Tables and Metric Aggregation Engine for SemEval-2027 Task 2 (DiCo-NLI).
Aggregates evaluation results across models, methods (baseline, bsa, crl, sd, crl_sd),
and tracks (Track 1 English and Track 2 Spanish).

Supports formatting and exporting to:
  - GitHub-Flavored Markdown table
  - Publication-ready LaTeX tabular table for ACL paper (report/paper.tex)
  - Aligned plain-text ASCII table
  - Structured nested JSON and Python dictionaries
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

logger = logging.getLogger(__name__)

# Data & metrics imports with graceful fallback
try:
    from src.data.loader import (
        ALL_LABELS,
        load_phrasis_split,
        normalize_split,
        normalize_track,
    )
    from src.data.reversal import (
        ID_TO_LABEL,
        LABEL_TO_ID,
        ReciprocalPair,
        pair_instances,
    )
    from src.metrics.metrics import (
        EvaluationReport,
        ReversiblePair,
        compute_official_evaluation,
        compute_official_metrics,
        compute_weighted_f1,
    )
    SRC_AVAILABLE = True
except ImportError:
    SRC_AVAILABLE = False
    ALL_LABELS = {"BACKWARD_ENTAILMENT", "EQUIVALENCE", "FORWARD_ENTAILMENT", "NEGATIVE_OTHER"}
    LABEL_TO_ID = {
        "BACKWARD_ENTAILMENT": 0,
        "EQUIVALENCE": 1,
        "FORWARD_ENTAILMENT": 2,
        "NEGATIVE_OTHER": 3,
    }
    ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

# Pandas import with fallback
try:
    import pandas as pd  # type: ignore
    HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore
    HAS_PANDAS = False


# =============================================================================
# 1. Canonical Methods, Models, and Labels
# =============================================================================

CANONICAL_METHODS: Tuple[str, ...] = ("baseline", "bsa", "crl", "sd", "crl_sd")

METHOD_DISPLAY_NAMES: Dict[str, str] = {
    "baseline": "Baseline",
    "bsa": "BSA",
    "crl": "CRL",
    "sd": "SD",
    "crl_sd": "CRL + SD",
}

METHOD_DESCRIPTIONS: Dict[str, str] = {
    "baseline": "Naive Unidirectional Cross-Encoder",
    "bsa": "Bi-directional Symmetrical Augmentation",
    "crl": "Consistency Regularization Loss (Siamese)",
    "sd": "Symmetrized Decoding (Post-hoc)",
    "crl_sd": "Joint Regularization + Symmetrized Decoding",
}

DEFAULT_MODELS: Dict[str, str] = {
    "track1": "DeBERTa-v3",
    "track2": "BETO",
}

TRACK_DISPLAY_NAMES: Dict[str, str] = {
    "track1": "Track 1 (English)",
    "track2": "Track 2 (Spanish)",
}


# =============================================================================
# 2. Evaluation Data Structures
# =============================================================================

@dataclass
class EvaluationResult:
    """Represents the benchmark evaluation metrics for a single model/method configuration."""
    track: str
    method: str
    model: str
    split: str = "dev"
    weighted_f1: float = 0.0
    soft_cons: float = 0.0
    hard_cons: float = 0.0
    delta_f1: Optional[float] = None
    delta_soft_cons: Optional[float] = None
    delta_hard_cons: Optional[float] = None
    num_instances: int = 0
    num_pairs: int = 0
    submission_file: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Enforce canonical naming
        clean_track = str(self.track).lower().strip()
        if clean_track in ["1", "en", "track1", "t1"]:
            self.track = "track1"
        elif clean_track in ["2", "es", "track2", "t2"]:
            self.track = "track2"

        self.method = str(self.method).lower().strip()

        # Invariant check: HardCons <= SoftCons (with float epsilon)
        if self.hard_cons > self.soft_cons + 1e-6:
            logger.warning(
                "HardCons (%.4f) exceeds SoftCons (%.4f) in result for %s/%s. Invariant violated.",
                self.hard_cons, self.soft_cons, self.track, self.method
            )

    @property
    def method_display(self) -> str:
        return METHOD_DISPLAY_NAMES.get(self.method, self.method.upper())

    @property
    def track_display(self) -> str:
        return TRACK_DISPLAY_NAMES.get(self.track, self.track)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to standard dictionary."""
        return {
            "track": self.track,
            "track_display": self.track_display,
            "method": self.method,
            "method_display": self.method_display,
            "model": self.model,
            "split": self.split,
            "weighted_f1": round(self.weighted_f1, 6),
            "soft_cons": round(self.soft_cons, 6),
            "hard_cons": round(self.hard_cons, 6),
            "delta_f1": round(self.delta_f1, 6) if self.delta_f1 is not None else None,
            "delta_soft_cons": round(self.delta_soft_cons, 6) if self.delta_soft_cons is not None else None,
            "delta_hard_cons": round(self.delta_hard_cons, 6) if self.delta_hard_cons is not None else None,
            "num_instances": self.num_instances,
            "num_pairs": self.num_pairs,
            "submission_file": self.submission_file,
            "metadata": self.metadata,
        }


# =============================================================================
# 3. Master Comparison Table Container
# =============================================================================

class ComparisonTable:
    """Container managing, formatting, and exporting multi-model evaluation comparison tables."""

    def __init__(
        self,
        results: Optional[Sequence[EvaluationResult]] = None,
        title: str = "Experimental Evaluation & Reversal Mitigation Benchmark",
    ) -> None:
        self.results: List[EvaluationResult] = list(results) if results is not None else []
        self.title: str = title

    def add_result(self, result: EvaluationResult) -> None:
        """Add an evaluation result to the table."""
        self.results.append(result)

    def filter(
        self,
        track: Optional[str] = None,
        method: Optional[str] = None,
        model: Optional[str] = None,
        split: Optional[str] = None,
    ) -> ComparisonTable:
        """Return a new filtered ComparisonTable."""
        filtered = []
        for r in self.results:
            if track is not None and r.track != track:
                continue
            if method is not None and r.method != method:
                continue
            if model is not None and r.model.lower() != model.lower():
                continue
            if split is not None and r.split != split:
                continue
            filtered.append(r)
        return ComparisonTable(filtered, title=self.title)

    def sort(self, by: str = "canonical") -> ComparisonTable:
        """Sort results canonically by track, then canonical method order."""
        def sort_key(res: EvaluationResult) -> Tuple[int, int, str]:
            t_order = 0 if res.track == "track1" else 1
            method_idx = CANONICAL_METHODS.index(res.method) if res.method in CANONICAL_METHODS else 99
            return (t_order, method_idx, res.model)

        sorted_results = sorted(self.results, key=sort_key)
        return ComparisonTable(sorted_results, title=self.title)

    def compute_deltas(self, baseline_method: str = "baseline") -> ComparisonTable:
        """Computes relative metric deltas (Δ = metric - baseline_metric) per track and model."""
        baselines: Dict[Tuple[str, str], EvaluationResult] = {}
        for r in self.results:
            if r.method == baseline_method:
                baselines[(r.track, r.model)] = r

        new_results = []
        for r in self.results:
            key = (r.track, r.model)
            base = baselines.get(key)
            if base is not None:
                d_f1 = r.weighted_f1 - base.weighted_f1
                d_sc = r.soft_cons - base.soft_cons
                d_hc = r.hard_cons - base.hard_cons
            else:
                d_f1 = 0.0 if r.method == baseline_method else None
                d_sc = 0.0 if r.method == baseline_method else None
                d_hc = 0.0 if r.method == baseline_method else None

            updated = EvaluationResult(
                track=r.track,
                method=r.method,
                model=r.model,
                split=r.split,
                weighted_f1=r.weighted_f1,
                soft_cons=r.soft_cons,
                hard_cons=r.hard_cons,
                delta_f1=d_f1,
                delta_soft_cons=d_sc,
                delta_hard_cons=d_hc,
                num_instances=r.num_instances,
                num_pairs=r.num_pairs,
                submission_file=r.submission_file,
                metadata=r.metadata,
            )
            new_results.append(updated)

        return ComparisonTable(new_results, title=self.title)

    # -------------------------------------------------------------------------
    # Format Exporters
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire comparison table to structured dictionary."""
        return {
            "title": self.title,
            "total_results": len(self.results),
            "tracks": sorted(list({r.track for r in self.results})),
            "methods": sorted(list({r.method for r in self.results})),
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert table to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_dataframe(self) -> Any:
        """Convert table to pandas.DataFrame (or list of dicts if pandas unavailable)."""
        data = [r.to_dict() for r in self.results]
        if HAS_PANDAS and pd is not None:
            return pd.DataFrame(data)
        return data

    def to_markdown(self, include_deltas: bool = True) -> str:
        """Generate GitHub-Flavored Markdown comparison table."""
        return generate_markdown_table(self, include_deltas=include_deltas)

    def to_ascii(self, include_deltas: bool = False) -> str:
        """Generate aligned plain-text ASCII table."""
        return generate_ascii_table(self, include_deltas=include_deltas)

    def to_latex(
        self,
        caption: str = "Evaluation of Reversal Mitigation Strategies on SemEval-2027 Task 2 PhrasIS Benchmark.",
        label: str = "tab:main_results",
        multi_track: bool = True,
        bold_best: bool = True,
    ) -> str:
        """Generate publication-ready LaTeX table for ACL paper (report/paper.tex)."""
        return generate_latex_table(
            self,
            caption=caption,
            label=label,
            multi_track=multi_track,
            bold_best=bold_best,
        )

    def save(
        self,
        output_path: Union[str, Path],
        fmt: Optional[str] = None,
        **kwargs: Any,
    ) -> Path:
        """Save formatted comparison table to disk based on file extension or fmt parameter."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        ext = path.suffix.lower() if fmt is None else f".{fmt.lower().strip('.')}"
        if ext in [".md", ".markdown"]:
            content = self.to_markdown(**kwargs)
        elif ext in [".tex", ".latex"]:
            content = self.to_latex(**kwargs)
        elif ext in [".json"]:
            content = self.to_json(**kwargs)
        elif ext in [".txt", ".ascii"]:
            content = self.to_ascii(**kwargs)
        elif ext in [".csv"]:
            df = self.to_dataframe()
            if hasattr(df, "to_csv"):
                df.to_csv(path, index=False)
                return path
            else:
                content = self.to_ascii(**kwargs)
        else:
            content = self.to_markdown(**kwargs)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path


# =============================================================================
# 4. Aggregation and Evaluation Functions
# =============================================================================

def parse_run_metadata(identifier: str) -> Dict[str, str]:
    """Extracts track, method, and model from a filename or run_name string.

    Examples:
      - 't1_baseline_deberta_submission.csv' -> track1, baseline, deberta
      - 't2_sd_beto_submission.csv' -> track2, sd, beto
      - 'test_track1_crl_sd_mock_20261102_023522_submission.csv' -> track1, crl_sd, mock
    """
    clean = Path(identifier).stem.lower()

    # 1. Track detection
    track = "track1"
    if "track2" in clean or "t2_" in clean or "_t2" in clean or "_es" in clean:
        track = "track2"
    elif "track1" in clean or "t1_" in clean or "_t1" in clean or "_en" in clean:
        track = "track1"

    # 2. Method detection
    method = "baseline"
    if "crl_sd" in clean or "crl-sd" in clean or "crl+sd" in clean:
        method = "crl_sd"
    elif "_sd_" in clean or clean.endswith("_sd") or "_sd" in clean:
        method = "sd"
    elif "_crl_" in clean or clean.endswith("_crl") or "_crl" in clean:
        method = "crl"
    elif "_bsa_" in clean or clean.endswith("_bsa") or "_bsa" in clean:
        method = "bsa"
    elif "baseline" in clean:
        method = "baseline"

    # 3. Model detection
    model = "mock"
    if "deberta" in clean:
        model = "DeBERTa-v3"
    elif "beto" in clean:
        model = "BETO"
    elif "roberta" in clean:
        model = "RoBERTa-bne"
    elif "mock" in clean:
        model = "Mock"
    else:
        model = DEFAULT_MODELS.get(track, "CrossEncoder")

    return {
        "track": track,
        "method": method,
        "model": model,
        "identifier": identifier,
    }


def load_predictions_from_csv(csv_path: Union[str, Path]) -> Dict[str, str]:
    """Reads a two-column Codabench submission CSV (instance_id,label) into a dictionary."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    predictions: Dict[str, str] = {}
    with open(path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty prediction CSV: {path}")

        # Check column positions
        id_idx = 0
        lbl_idx = 1
        if len(header) >= 2 and header[0].strip().lower() != "instance_id" and "label" in [h.strip().lower() for h in header]:
            cols = [h.strip().lower() for h in header]
            id_idx = cols.index("instance_id") if "instance_id" in cols else 0
            lbl_idx = cols.index("label") if "label" in cols else 1

        for row_idx, row in enumerate(reader, start=2):
            if not row or len(row) < 2:
                continue
            iid = row[id_idx].strip()
            lbl = row[lbl_idx].strip()
            if iid:
                predictions[iid] = lbl

    return predictions


def evaluate_prediction_mapping(
    gold_df_or_records: Any,
    pred_mapping: Mapping[str, str],
    track: str = "track1",
    method: str = "baseline",
    model: str = "model",
    split: str = "dev",
    submission_file: Optional[str] = None,
) -> EvaluationResult:
    """Computes official metrics from a prediction mapping against gold dataset."""
    if hasattr(gold_df_or_records, "to_dict") and callable(getattr(gold_df_or_records, "to_dict")):
        records = gold_df_or_records.to_dict("records")
    elif isinstance(gold_df_or_records, (list, tuple)):
        records = list(gold_df_or_records)
    else:
        records = list(gold_df_or_records)

    if not records:
        raise ValueError("Cannot evaluate against empty gold records.")

    # Build gold mapping and reciprocal pairs
    gold_by_id: Dict[str, str] = {}
    for r in records:
        iid = str(r.get("instance_id") if isinstance(r, dict) else getattr(r, "instance_id", ""))
        lbl = str(r.get("label") if isinstance(r, dict) else getattr(r, "label", ""))
        if iid and lbl:
            gold_by_id[iid] = lbl

    pairs = pair_instances(records)
    rev_pairs = [p for p in pairs if getattr(p, "is_reversible", True)]

    # Compute official evaluation report
    report = compute_official_evaluation(
        gold_by_id=gold_by_id,
        predicted_labels=pred_mapping,
        reversible_pairs=rev_pairs,
    )

    return EvaluationResult(
        track=track,
        method=method,
        model=model,
        split=split,
        weighted_f1=report.weighted_f1,
        soft_cons=report.soft_cons,
        hard_cons=report.hard_cons,
        num_instances=len(gold_by_id),
        num_pairs=len(rev_pairs),
        submission_file=submission_file,
    )


def evaluate_prediction_file(
    pred_csv_path: Union[str, Path],
    gold_df_or_records: Any,
    track: Optional[str] = None,
    method: Optional[str] = None,
    model: Optional[str] = None,
    split: str = "dev",
) -> EvaluationResult:
    """Loads prediction CSV and evaluates against gold data."""
    path = Path(pred_csv_path)
    meta = parse_run_metadata(path.name)

    eff_track = track or meta["track"]
    eff_method = method or meta["method"]
    eff_model = model or meta["model"]

    preds = load_predictions_from_csv(path)
    return evaluate_prediction_mapping(
        gold_df_or_records=gold_df_or_records,
        pred_mapping=preds,
        track=eff_track,
        method=eff_method,
        model=eff_model,
        split=split,
        submission_file=str(path),
    )


def aggregate_results_from_dir(
    results_dir: Union[str, Path] = "results",
    logs_dir: Optional[Union[str, Path]] = "logs",
    data_dir: Union[str, Path] = "data",
    split: str = "dev",
    tracks: Sequence[str] = ("track1", "track2"),
) -> ComparisonTable:
    """Scans results_dir for submission CSVs and logs_dir for summary JSONs, aggregating into a ComparisonTable."""
    res_path = Path(results_dir)
    log_path = Path(logs_dir) if logs_dir else None
    table = ComparisonTable()

    seen_runs: Set[Tuple[str, str, str]] = set()

    # Cache gold datasets per track
    gold_cache: Dict[str, Any] = {}
    for trk in tracks:
        t_norm = "track1" if "1" in trk or "en" in trk else "track2"
        try:
            gold_cache[t_norm] = load_phrasis_split(track=t_norm, split=split, data_dir=data_dir)
        except Exception as exc:
            logger.warning(f"Could not load gold dataset for {t_norm}: {exc}")

    # 1. First scan logs_dir for *_summary.json
    if log_path and log_path.is_dir():
        for summary_file in sorted(log_path.glob("*_summary.json")):
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                metrics = data.get("metrics", {})
                if not metrics or "weighted_f1" not in metrics:
                    continue

                t = data.get("track", "track1")
                m = data.get("method", "baseline")
                mod = data.get("model", "mock")
                if mod == "mock":
                    mod = "Mock"
                elif "deberta" in mod.lower():
                    mod = "DeBERTa-v3"
                elif "beto" in mod.lower():
                    mod = "BETO"

                key = (t, m, mod)
                if key in seen_runs:
                    continue

                res = EvaluationResult(
                    track=t,
                    method=m,
                    model=mod,
                    split=split,
                    weighted_f1=float(metrics.get("weighted_f1", 0.0)),
                    soft_cons=float(metrics.get("soft_cons", 0.0)),
                    hard_cons=float(metrics.get("hard_cons", 0.0)),
                    submission_file=data.get("submission_file"),
                    metadata={"source": "log_summary", "file": str(summary_file)},
                )
                table.add_result(res)
                seen_runs.add(key)
            except Exception as exc:
                logger.debug(f"Failed parsing summary JSON {summary_file}: {exc}")

    # 2. Second scan results_dir for submission CSVs
    if res_path.is_dir():
        for csv_file in sorted(res_path.glob("*.csv")):
            if csv_file.name == "summary.csv":
                continue
            meta = parse_run_metadata(csv_file.name)
            trk = meta["track"]
            mth = meta["method"]
            mdl = meta["model"]

            key = (trk, mth, mdl)
            if key in seen_runs:
                continue

            gold = gold_cache.get(trk)
            if gold is None:
                continue

            try:
                eval_res = evaluate_prediction_file(
                    pred_csv_path=csv_file,
                    gold_df_or_records=gold,
                    track=trk,
                    method=mth,
                    model=mdl,
                    split=split,
                )
                eval_res.metadata["source"] = "evaluated_csv"
                table.add_result(eval_res)
                seen_runs.add(key)
            except Exception as exc:
                logger.debug(f"Could not evaluate CSV {csv_file}: {exc}")

    # Sort and compute deltas
    return table.sort().compute_deltas()


def build_comparison_table(
    results: Sequence[Union[EvaluationResult, Dict[str, Any]]],
    baseline_method: str = "baseline",
    title: str = "Experimental Evaluation & Reversal Mitigation Benchmark",
) -> ComparisonTable:
    """Builds and normalizes a ComparisonTable from a sequence of results."""
    typed_results: List[EvaluationResult] = []
    for r in results:
        if isinstance(r, EvaluationResult):
            typed_results.append(r)
        elif isinstance(r, dict):
            typed_results.append(
                EvaluationResult(
                    track=r.get("track", "track1"),
                    method=r.get("method", "baseline"),
                    model=r.get("model", "mock"),
                    split=r.get("split", "dev"),
                    weighted_f1=float(r.get("weighted_f1", 0.0)),
                    soft_cons=float(r.get("soft_cons", 0.0)),
                    hard_cons=float(r.get("hard_cons", 0.0)),
                    num_instances=int(r.get("num_instances", 0)),
                    num_pairs=int(r.get("num_pairs", 0)),
                    submission_file=r.get("submission_file"),
                    metadata=r.get("metadata", {}),
                )
            )

    table = ComparisonTable(typed_results, title=title)
    return table.sort().compute_deltas(baseline_method=baseline_method)


# =============================================================================
# 5. Formatter Implementations
# =============================================================================

def _fmt_float(val: Optional[float], decimals: int = 4, show_sign: bool = False) -> str:
    """Helper to format float strings with fixed precision."""
    if val is None or math.isnan(val):
        return "-"
    if show_sign:
        return f"{val:+.{decimals}f}"
    return f"{val:.{decimals}f}"


def generate_markdown_table(table: ComparisonTable, include_deltas: bool = True) -> str:
    """Formats comparison table into GitHub-Flavored Markdown."""
    lines: List[str] = [f"## {table.title}\n"]

    if not table.results:
        lines.append("*No evaluation results available.*")
        return "\n".join(lines)

    if include_deltas:
        headers = [
            "Track",
            "Model",
            "Method",
            "Weighted F1",
            "SoftCons",
            "HardCons",
            "ΔF1",
            "ΔSoftCons",
            "ΔHardCons",
        ]
        align = [":---", ":---", ":---", "---:", "---:", "---:", "---:", "---:", "---:"]
    else:
        headers = ["Track", "Model", "Method", "Weighted F1", "SoftCons", "HardCons"]
        align = [":---", ":---", ":---", "---:", "---:", "---:"]

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(align) + " |")

    for r in table.results:
        f1_str = _fmt_float(r.weighted_f1, 4)
        sc_str = _fmt_float(r.soft_cons, 4)
        hc_str = _fmt_float(r.hard_cons, 4)

        if include_deltas:
            d_f1 = _fmt_float(r.delta_f1, 4, show_sign=True)
            d_sc = _fmt_float(r.delta_soft_cons, 4, show_sign=True)
            d_hc = _fmt_float(r.delta_hard_cons, 4, show_sign=True)
            row = [
                r.track_display,
                r.model,
                r.method_display,
                f1_str,
                sc_str,
                hc_str,
                d_f1,
                d_sc,
                d_hc,
            ]
        else:
            row = [r.track_display, r.model, r.method_display, f1_str, sc_str, hc_str]

        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def generate_ascii_table(table: ComparisonTable, include_deltas: bool = False) -> str:
    """Formats comparison table into an aligned plain-text ASCII table."""
    if not table.results:
        return f"{table.title}\n(No evaluation results available)"

    if include_deltas:
        cols = [
            ("Track", 16, "<"),
            ("Model", 12, "<"),
            ("Method", 12, "<"),
            ("WF1", 8, ">"),
            ("SoftCons", 10, ">"),
            ("HardCons", 10, ">"),
            ("ΔF1", 9, ">"),
            ("ΔSC", 9, ">"),
            ("ΔHC", 9, ">"),
        ]
    else:
        cols = [
            ("Track", 18, "<"),
            ("Model", 14, "<"),
            ("Method", 14, "<"),
            ("Weighted F1", 13, ">"),
            ("SoftCons", 12, ">"),
            ("HardCons", 12, ">"),
        ]

    def make_divider(char: str = "-") -> str:
        parts = [char * (w + 2) for _, w, _ in cols]
        return "+" + "+".join(parts) + "+"

    header_cells = [f" {name:<{w}} " if al == "<" else f" {name:>{w}} " for name, w, al in cols]
    header_str = "|" + "|".join(header_cells) + "|"

    div = make_divider("-")
    top_div = make_divider("=")

    lines = [
        f"\n{table.title}",
        top_div,
        header_str,
        top_div,
    ]

    current_track = None
    for r in table.results:
        if current_track is not None and r.track != current_track:
            lines.append(div)
        current_track = r.track

        f1_str = _fmt_float(r.weighted_f1, 4)
        sc_str = _fmt_float(r.soft_cons, 4)
        hc_str = _fmt_float(r.hard_cons, 4)

        if include_deltas:
            vals = [
                r.track_display,
                r.model,
                r.method_display,
                f1_str,
                sc_str,
                hc_str,
                _fmt_float(r.delta_f1, 4, show_sign=True),
                _fmt_float(r.delta_soft_cons, 4, show_sign=True),
                _fmt_float(r.delta_hard_cons, 4, show_sign=True),
            ]
        else:
            vals = [
                r.track_display,
                r.model,
                r.method_display,
                f1_str,
                sc_str,
                hc_str,
            ]

        cells = []
        for val, (_, w, al) in zip(vals, cols):
            if al == "<":
                cells.append(f" {val:<{w}} ")
            else:
                cells.append(f" {val:>{w}} ")

        lines.append("|" + "|".join(cells) + "|")

    lines.append(top_div)
    return "\n".join(lines)


def generate_latex_table(
    table: ComparisonTable,
    caption: str = "Experimental evaluation of reversal mitigation methods on Track 1 (English) and Track 2 (Spanish) dev sets. Best scores per track are bolded.",
    label: str = "tab:main_results",
    multi_track: bool = True,
    bold_best: bool = True,
) -> str:
    """Formats comparison table into publication-ready LaTeX tabular code for ACL paper."""
    if not table.results:
        return "% No evaluation results to render in LaTeX table."

    # Identify tracks present
    tracks_present = []
    for t in ["track1", "track2"]:
        if any(r.track == t for r in table.results):
            tracks_present.append(t)

    # Compute best values per track for bolding
    best_f1: Dict[str, float] = {}
    best_sc: Dict[str, float] = {}
    best_hc: Dict[str, float] = {}
    for t in tracks_present:
        t_res = [r for r in table.results if r.track == t]
        if t_res:
            best_f1[t] = max(r.weighted_f1 for r in t_res)
            best_sc[t] = max(r.soft_cons for r in t_res)
            best_hc[t] = max(r.hard_cons for r in t_res)

    def _latex_cell(val: float, best_val: float, is_bold_allowed: bool) -> str:
        s = f"{val:.4f}"
        if is_bold_allowed and abs(val - best_val) < 1e-4:
            return f"\\textbf{{{s}}}"
        return s

    lines: List[str] = []

    # Check if multi-track side-by-side layout can be used
    can_side_by_side = multi_track and len(tracks_present) == 2

    if can_side_by_side:
        # Side-by-side layout across 2 tracks
        lines.extend([
            "\\begin{table*}[t]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{ll ccc ccc}",
            "\\toprule",
            " & & \\multicolumn{3}{c}{\\textbf{Track 1: English (DeBERTa-v3)}} & \\multicolumn{3}{c}{\\textbf{Track 2: Spanish (BETO)}} \\\\",
            "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}",
            "\\textbf{Method} & \\textbf{Strategy} & \\textbf{Weighted F$_1$} & \\textbf{SoftCons} & \\textbf{HardCons} & \\textbf{Weighted F$_1$} & \\textbf{SoftCons} & \\textbf{HardCons} \\\\",
            "\\midrule",
        ])

        t1_dict = {r.method: r for r in table.results if r.track == "track1"}
        t2_dict = {r.method: r for r in table.results if r.track == "track2"}

        for m in CANONICAL_METHODS:
            r1 = t1_dict.get(m)
            r2 = t2_dict.get(m)
            if r1 is None and r2 is None:
                continue

            m_disp = METHOD_DISPLAY_NAMES.get(m, m.upper())
            m_desc = METHOD_DESCRIPTIONS.get(m, "")

            if r1 is not None:
                f1_1 = _latex_cell(r1.weighted_f1, best_f1.get("track1", 0.0), bold_best)
                sc_1 = _latex_cell(r1.soft_cons, best_sc.get("track1", 0.0), bold_best)
                hc_1 = _latex_cell(r1.hard_cons, best_hc.get("track1", 0.0), bold_best)
            else:
                f1_1, sc_1, hc_1 = "-", "-", "-"

            if r2 is not None:
                f1_2 = _latex_cell(r2.weighted_f1, best_f1.get("track2", 0.0), bold_best)
                sc_2 = _latex_cell(r2.soft_cons, best_sc.get("track2", 0.0), bold_best)
                hc_2 = _latex_cell(r2.hard_cons, best_hc.get("track2", 0.0), bold_best)
            else:
                f1_2, sc_2, hc_2 = "-", "-", "-"

            lines.append(f"{m_disp} & {m_desc} & {f1_1} & {sc_1} & {hc_1} & {f1_2} & {sc_2} & {hc_2} \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{table*}",
        ])
    else:
        # Standard stacked single-table layout
        lines.extend([
            "\\begin{table}[ht]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{lll ccc}",
            "\\toprule",
            "\\textbf{Track} & \\textbf{Method} & \\textbf{Model} & \\textbf{Weighted F$_1$} & \\textbf{SoftCons} & \\textbf{HardCons} \\\\",
            "\\midrule",
        ])

        curr_track = None
        for r in table.results:
            if curr_track is not None and r.track != curr_track:
                lines.append("\\midrule")
            curr_track = r.track

            m_disp = METHOD_DISPLAY_NAMES.get(r.method, r.method.upper())
            t_disp = "Track 1 (EN)" if r.track == "track1" else "Track 2 (ES)"

            f1_str = _latex_cell(r.weighted_f1, best_f1.get(r.track, 0.0), bold_best)
            sc_str = _latex_cell(r.soft_cons, best_sc.get(r.track, 0.0), bold_best)
            hc_str = _latex_cell(r.hard_cons, best_hc.get(r.track, 0.0), bold_best)

            lines.append(f"{t_disp} & {m_disp} & {r.model} & {f1_str} & {sc_str} & {hc_str} \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{table}",
        ])

    return "\n".join(lines)


def format_comparison_table(
    table: ComparisonTable,
    format: str = "markdown",
    **kwargs: Any,
) -> str:
    """Master formatting dispatcher for ComparisonTable."""
    clean_fmt = format.lower().strip()
    if clean_fmt in ["markdown", "md"]:
        return table.to_markdown(**kwargs)
    elif clean_fmt in ["latex", "tex"]:
        return table.to_latex(**kwargs)
    elif clean_fmt in ["ascii", "text", "txt"]:
        return table.to_ascii(**kwargs)
    elif clean_fmt in ["json"]:
        return table.to_json(**kwargs)
    else:
        raise ValueError(f"Unsupported format '{format}'. Valid options: markdown, latex, ascii, json.")
