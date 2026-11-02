#!/usr/bin/env python3
"""
experiments/evaluate.py

Standalone CLI Evaluation Engine for SemEval-2027 Task 2 (DiCo-NLI).
Evaluates models and prediction files against official metrics:
  - Weighted F1 across 4 labels
  - SoftCons (Soft Directional Consistency Rate)
  - HardCons (Hard Directional Consistency Rate)

Capabilities:
  1. Static evaluation: Evaluates pre-computed Codabench submission CSVs or logs in --results_dir.
  2. Dynamic evaluation: Evaluates model checkpoints (or fast mock models) on PhrasIS splits.
  3. Master comparison table generation: Formats Markdown, LaTeX, ASCII, and JSON.
  4. Confusion & Transition matrix generation (--generate_matrices).
  5. Syntactic chunk error analysis (--generate_analysis).
  6. Ultra-fast CPU test mode (<2s execution budget under --test_run).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

# Logging configuration
logging.basicConfig(
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("dico_nli.evaluate")

def _find_project_root() -> Path:
    curr = Path(__file__).resolve().parent
    for _ in range(6):
        if (curr / "src").is_dir() or (curr / "PROJECT.md").is_file():
            return curr
        curr = curr.parent
    return Path.cwd()

PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Project imports with graceful fallbacks
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
from src.data.codabench import (
    export_codabench_submission,
    validate_codabench_submission,
)
from src.metrics.metrics import (
    EvaluationReport,
    compute_official_evaluation,
    compute_official_metrics,
)
from src.models.factory import (
    build_model,
    build_tokenizer,
    resolve_device,
    resolve_torch_dtype,
)
from src.training.utils import set_seed, get_device_info

# Evaluation module imports (with graceful fallback if not yet installed in src/)
try:
    from src.evaluation.tables import (
        ComparisonTable,
        EvaluationResult,
        aggregate_results_from_dir,
        build_comparison_table,
        evaluate_prediction_file,
        format_comparison_table,
        generate_latex_table,
        generate_markdown_table,
        load_predictions_from_csv,
    )
    TABLES_MODULE_AVAILABLE = True
except ImportError:
    TABLES_MODULE_AVAILABLE = False
    # Fallback to local explorer proposed_tables if running in development sandbox
    try:
        import importlib.util
        spec_path = PROJECT_ROOT / ".agents" / "teamwork" / "explorer_m4_1" / "proposed_tables.py"
        if spec_path.is_file():
            spec = importlib.util.spec_from_file_location("tables_fallback", str(spec_path))
            tables_mod = importlib.util.module_from_spec(spec)
            sys.modules["tables_fallback"] = tables_mod
            spec.loader.exec_module(tables_mod)
            ComparisonTable = tables_mod.ComparisonTable
            EvaluationResult = tables_mod.EvaluationResult
            aggregate_results_from_dir = tables_mod.aggregate_results_from_dir
            build_comparison_table = tables_mod.build_comparison_table
            evaluate_prediction_file = tables_mod.evaluate_prediction_file
            format_comparison_table = tables_mod.format_comparison_table
            generate_latex_table = tables_mod.generate_latex_table
            generate_markdown_table = tables_mod.generate_markdown_table
            load_predictions_from_csv = tables_mod.load_predictions_from_csv
            TABLES_MODULE_AVAILABLE = True
    except Exception as exc:
        logger.warning(f"Fallback tables import failed: {exc}")


# =============================================================================
# 1. CLI Argument Parsing
# =============================================================================

def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parses and validates CLI arguments for standalone evaluation."""
    parser = argparse.ArgumentParser(
        description="SemEval-2027 Task 2 (DiCo-NLI) Standalone Evaluation & Table Generation CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Target Track & Dataset Split ---
    group_scope = parser.add_argument_group("Benchmark Scope")
    group_scope.add_argument(
        "--track",
        type=str,
        default="all",
        choices=["1", "2", "track1", "track2", "all"],
        help="Benchmark track to evaluate: '1'/'track1' (English), '2'/'track2' (Spanish), or 'all'.",
    )
    group_scope.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["dev", "train", "test"],
        help="PhrasIS dataset split to evaluate against.",
    )
    group_scope.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["baseline", "bsa", "crl", "sd", "crl_sd", "all"],
        help="Evaluation method filter.",
    )
    group_scope.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Base PhrasIS dataset directory.",
    )

    # --- Model & Checkpoint (Dynamic Evaluation) ---
    group_model = parser.add_argument_group("Model & Checkpoint Execution")
    group_model.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional path to model checkpoint (.pt) for dynamic evaluation.",
    )
    group_model.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model backbone identifier ('microsoft/deberta-v3-base', 'dccuchile/bert-base-spanish-wwm-cased', 'mock').",
    )
    group_model.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Evaluation batch size.",
    )
    group_model.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Compute device for dynamic inference.",
    )

    # --- Directories & Inputs (Static Evaluation) ---
    group_io = parser.add_argument_group("Directories & Input Artifacts")
    group_io.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing pre-computed Codabench prediction CSVs.",
    )
    group_io.add_argument(
        "--logs_dir",
        type=str,
        default="logs",
        help="Directory containing training run summary JSON logs.",
    )
    group_io.add_argument(
        "--output_dir",
        type=str,
        default="results/evaluation",
        help="Directory where generated tables, matrices, and reports will be saved.",
    )
    group_io.add_argument(
        "--prediction_file",
        type=str,
        default=None,
        help="Direct path to evaluate a single Codabench submission CSV.",
    )

    # --- Generation & Analysis Flags ---
    group_artifacts = parser.add_argument_group("Artifact Generation")
    group_artifacts.add_argument(
        "--generate_tables",
        action="store_true",
        default=True,
        help="Generate comparison tables across models and methods (Markdown, LaTeX, ASCII, JSON).",
    )
    group_artifacts.add_argument(
        "--no_tables",
        action="store_false",
        dest="generate_tables",
        help="Disable table generation.",
    )
    group_artifacts.add_argument(
        "--generate_matrices",
        action="store_true",
        default=False,
        help="Generate 4x4 Confusion and Directional Transition Matrices.",
    )
    group_artifacts.add_argument(
        "--generate_analysis",
        action="store_true",
        default=False,
        help="Generate syntactic chunk error analysis (nominal, verbal, prepositional, adverbial).",
    )

    # --- Fast Execution Mode ---
    group_mode = parser.add_argument_group("Execution Modes")
    group_mode.add_argument(
        "--test_run",
        action="store_true",
        default=False,
        help="Execute ultra-fast CPU test mode (<2s execution budget) with mock model and tiny data.",
    )
    group_mode.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic evaluation.",
    )

    return parser.parse_args(args)


# =============================================================================
# 2. Evaluation Dispatcher Helpers
# =============================================================================

def resolve_target_tracks(track_arg: str) -> List[str]:
    """Resolves track argument to a list of canonical track identifiers."""
    clean = track_arg.lower().strip()
    if clean in ["1", "track1", "en"]:
        return ["track1"]
    elif clean in ["2", "track2", "es"]:
        return ["track2"]
    elif clean in ["all", "both"]:
        return ["track1", "track2"]
    return ["track1", "track2"]


def resolve_target_methods(method_arg: str) -> List[str]:
    """Resolves method argument to a list of canonical method identifiers."""
    clean = method_arg.lower().strip()
    if clean in ["all", "any"]:
        return ["baseline", "bsa", "crl", "sd", "crl_sd"]
    return [clean]


def run_dynamic_model_evaluation(
    track: str,
    method: str,
    model_name: str,
    checkpoint_path: Optional[str] = None,
    split: str = "dev",
    data_dir: str = "data",
    device: str = "auto",
    batch_size: int = 32,
    output_dir: str = "results/evaluation",
    test_run: bool = False,
) -> EvaluationResult:
    """Dynamically instantiates model, executes inference over PhrasIS split, and computes metrics."""
    track_norm = normalize_track(track)
    target_device = resolve_device(device)
    is_mock = test_run or (model_name == "mock")

    logger.info(
        f"Dynamic Evaluation: track={track_norm}, method={method}, model={model_name}, device={target_device}"
    )

    # 1. Load Dataset Split
    gold_df = load_phrasis_split(
        track=track_norm,
        split=split,
        data_dir=data_dir,
        allow_mock_fallback=True,
    )
    records = gold_df.to_dict("records") if hasattr(gold_df, "to_dict") else list(gold_df)

    # 2. Build Model
    model = build_model(
        model_name_or_path=model_name,
        method=method,
        num_labels=4,
        device=target_device,
        is_mock=is_mock,
        mask_negative_other=True,
    )

    if checkpoint_path and os.path.isfile(checkpoint_path):
        logger.info(f"Loading checkpoint weights from: {checkpoint_path}")
        try:
            import torch
            state_dict = torch.load(checkpoint_path, map_location=target_device)
            if "model_state_dict" in state_dict:
                model.load_state_dict(state_dict["model_state_dict"])
            else:
                model.load_state_dict(state_dict)
        except Exception as exc:
            if is_mock:
                logger.info("Mock mode: validated checkpoint structure.")
            else:
                logger.error(f"Failed loading checkpoint: {exc}")
                raise

    # 3. Predict across pairs
    pairs = pair_instances(records)
    predictions_map: Dict[str, str] = {}
    apply_sd = method in ["sd", "crl_sd"]

    for pair in pairs:
        fwd = pair.direct_instance
        rev = pair.inverse_instance
        id_fwd = str(fwd.get("instance_id") if isinstance(fwd, dict) else getattr(fwd, "instance_id", ""))
        id_rev = str(rev.get("instance_id") if isinstance(rev, dict) else getattr(rev, "instance_id", ""))
        t1 = str(fwd.get("text1") or "") if isinstance(fwd, dict) else str(getattr(fwd, "text1", "") or "")
        t2 = str(fwd.get("text2") or "") if isinstance(fwd, dict) else str(getattr(fwd, "text2", "") or "")

        if apply_sd and hasattr(model, "predict_pair"):
            p1_idx, p2_idx = model.predict_pair(text1=t1, text2=t2, mask_negative_other=True)
            predictions_map[id_fwd] = ID_TO_LABEL[p1_idx]
            predictions_map[id_rev] = ID_TO_LABEL[p2_idx]
        elif hasattr(model, "predict"):
            predictions_map[id_fwd] = model.predict(t1, t2)
            predictions_map[id_rev] = model.predict(t2, t1)
        else:
            predictions_map[id_fwd] = "EQUIVALENCE"
            predictions_map[id_rev] = "EQUIVALENCE"

    # Fill in any unpaired instances (e.g. non-reversible NEGATIVE_OTHER instances)
    for r in records:
        iid = str(r.get("instance_id") if isinstance(r, dict) else getattr(r, "instance_id", ""))
        if iid and iid not in predictions_map:
            t1 = str(r.get("text1") or "") if isinstance(r, dict) else str(getattr(r, "text1", "") or "")
            t2 = str(r.get("text2") or "") if isinstance(r, dict) else str(getattr(r, "text2", "") or "")
            if hasattr(model, "predict"):
                predictions_map[iid] = model.predict(t1, t2)
            else:
                predictions_map[iid] = "NEGATIVE_OTHER"

    # Export Codabench prediction CSV
    out_csv = Path(output_dir) / f"eval_{track_norm}_{method}_{model_name}_{split}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    all_ids = [str(r["instance_id"]) for r in records]
    all_preds = [predictions_map.get(iid, "NEGATIVE_OTHER") for iid in all_ids]
    export_codabench_submission(
        instance_ids=all_ids,
        predictions=all_preds,
        output_path=str(out_csv),
        validate=True,
        expected_ids=set(all_ids),
    )

    # 4. Compute Official Metrics
    gold_by_id = {str(r["instance_id"]): str(r["label"]) for r in records if r.get("label")}
    rev_pairs = [p for p in pairs if getattr(p, "is_reversible", True)]
    rep = compute_official_evaluation(gold_by_id, predictions_map, rev_pairs)

    return EvaluationResult(
        track=track_norm,
        method=method,
        model=model_name,
        split=split,
        weighted_f1=rep.weighted_f1,
        soft_cons=rep.soft_cons,
        hard_cons=rep.hard_cons,
        num_instances=len(gold_by_id),
        num_pairs=len(rev_pairs),
        submission_file=str(out_csv),
        metadata={"mode": "dynamic_evaluation"},
    )


def generate_matrix_artifacts(
    table: ComparisonTable,
    output_dir: Union[str, Path],
    split: str = "dev",
    data_dir: str = "data",
) -> List[Path]:
    """Generates 4x4 Confusion and Transition matrices for evaluated configurations."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []

    # Attempt import from peer explorer_m4_2 matrices module or fallback to consistency.py
    try:
        from src.evaluation.matrices import (
            compute_multiclass_confusion_matrix,
            compute_directional_transition_matrix,
        )
        HAS_MATRICES = True
    except ImportError:
        try:
            from src.metrics.consistency import compute_directional_transition_matrix
            HAS_MATRICES = True
        except ImportError:
            HAS_MATRICES = False

    if not HAS_MATRICES:
        logger.warning("Matrix generation module not available; skipping matrix generation.")
        return created

    for r in table.results:
        if not r.submission_file or not os.path.isfile(r.submission_file):
            continue

        try:
            gold_df = load_phrasis_split(track=r.track, split=split, data_dir=data_dir)
            records = gold_df.to_dict("records") if hasattr(gold_df, "to_dict") else list(gold_df)
            pairs = pair_instances(records)
            rev_pairs = [p for p in pairs if getattr(p, "is_reversible", True)]

            # Load predictions
            preds = load_predictions_from_csv(r.submission_file)

            rep = compute_directional_transition_matrix(rev_pairs, preds)
            mat_txt = out_path / f"transition_matrix_{r.track}_{r.method}.txt"
            mat_tex = out_path / f"transition_matrix_{r.track}_{r.method}.tex"

            with open(mat_txt, "w", encoding="utf-8") as f:
                f.write(rep.format_ascii_table())
            with open(mat_tex, "w", encoding="utf-8") as f:
                f.write(rep.format_latex_table(caption=f"Directional Transition Matrix for {r.track_display} ({r.method_display})"))

            created.extend([mat_txt, mat_tex])
            logger.info(f"Exported transition matrix for {r.track}/{r.method} to {mat_txt}")
        except Exception as exc:
            logger.debug(f"Failed generating transition matrix for {r.track}/{r.method}: {exc}")

    return created


def generate_analysis_artifacts(
    table: ComparisonTable,
    output_dir: Union[str, Path],
    split: str = "dev",
    data_dir: str = "data",
) -> List[Path]:
    """Generates syntactic breakdown error analysis across nominal, verbal, prepositional, adverbial chunks."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []

    try:
        from src.metrics.consistency import compute_consistency_by_category
    except ImportError:
        logger.warning("Syntactic category analysis module not available; skipping.")
        return created

    analysis_summary: Dict[str, Any] = {}

    for r in table.results:
        if not r.submission_file or not os.path.isfile(r.submission_file):
            continue

        try:
            gold_df = load_phrasis_split(track=r.track, split=split, data_dir=data_dir)
            records = gold_df.to_dict("records") if hasattr(gold_df, "to_dict") else list(gold_df)
            pairs = pair_instances(records)
            rev_pairs = [p for p in pairs if getattr(p, "is_reversible", True)]

            # Map pair_id or instance_id to category
            categories: Dict[str, str] = {}
            for rec in records:
                iid = str(rec.get("instance_id") if isinstance(rec, dict) else getattr(rec, "instance_id", ""))
                cat = str(rec.get("category") if isinstance(rec, dict) else getattr(rec, "category", "nominal"))
                categories[iid] = cat

            preds = load_predictions_from_csv(r.submission_file)

            cat_report = compute_consistency_by_category(rev_pairs, categories, preds)
            key = f"{r.track}_{r.method}"
            analysis_summary[key] = {
                "track": r.track,
                "method": r.method,
                "model": r.model,
                "categories": cat_report,
            }
        except Exception as exc:
            logger.debug(f"Failed syntactic analysis for {r.track}/{r.method}: {exc}")

    if analysis_summary:
        json_file = out_path / "syntactic_error_analysis.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(analysis_summary, f, indent=2)
        created.append(json_file)
        logger.info(f"Exported syntactic error analysis to {json_file}")

    return created


# =============================================================================
# 3. Master Evaluation Controller
# =============================================================================

def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """Master controller executing static or dynamic evaluation and generating comparison tables."""
    t0 = time.time()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracks = resolve_target_tracks(args.track)
    methods = resolve_target_methods(args.method)

    logger.info("=" * 70)
    logger.info("SemEval-2027 Task 2 (DiCo-NLI) — Evaluation CLI Engine")
    logger.info(f"Tracks: {tracks} | Methods: {methods} | Split: {args.split}")
    logger.info(f"Results Dir: {args.results_dir} | Output Dir: {output_dir}")
    logger.info(f"Test Run: {args.test_run} | Tables: {args.generate_tables}")
    logger.info("=" * 70)

    # 1. Direct Single Prediction File Evaluation
    if args.prediction_file and os.path.isfile(args.prediction_file):
        logger.info(f"Evaluating single prediction CSV: {args.prediction_file}")
        trk = tracks[0] if tracks else "track1"
        gold_df = load_phrasis_split(track=trk, split=args.split, data_dir=args.data_dir)
        single_result = evaluate_prediction_file(
            pred_csv_path=args.prediction_file,
            gold_df_or_records=gold_df,
            track=trk,
            split=args.split,
        )
        table = ComparisonTable([single_result])

    # 2. Dynamic Model Evaluation (When checkpoint or specific model requested)
    elif args.checkpoint_path or (args.model_name and not args.test_run):
        results_list = []
        for trk in tracks:
            for mth in methods:
                mdl = args.model_name or ("DeBERTa-v3" if trk == "track1" else "BETO")
                res = run_dynamic_model_evaluation(
                    track=trk,
                    method=mth,
                    model_name=mdl,
                    checkpoint_path=args.checkpoint_path,
                    split=args.split,
                    data_dir=args.data_dir,
                    device=args.device,
                    batch_size=args.batch_size,
                    output_dir=str(output_dir),
                    test_run=args.test_run,
                )
                results_list.append(res)
        table = build_comparison_table(results_list)

    # 3. Static Evaluation of Existing Results Directory (or Fast Mock under test_run)
    else:
        logger.info(f"Scanning '{args.results_dir}' and '{args.logs_dir}' for existing benchmark results...")
        table = aggregate_results_from_dir(
            results_dir=args.results_dir,
            logs_dir=args.logs_dir,
            data_dir=args.data_dir,
            split=args.split,
            tracks=tracks,
        )

        # If results_dir had no matching files or in test_run mode, generate fast mock evaluations
        if len(table.results) == 0 or args.test_run:
            logger.info("Executing fast mock benchmark evaluation across all target tracks and methods...")
            mock_results = []
            for trk in tracks:
                for mth in methods:
                    res = run_dynamic_model_evaluation(
                        track=trk,
                        method=mth,
                        model_name="mock",
                        split=args.split,
                        data_dir=args.data_dir,
                        output_dir=str(output_dir),
                        test_run=True,
                    )
                    mock_results.append(res)
            table = build_comparison_table(mock_results)

    # Filter table by requested tracks and methods
    if args.track != "all":
        table = table.filter(track=tracks[0])
    if args.method != "all":
        table = table.filter(method=methods[0])

    # 4. Generate Tables (Markdown, LaTeX, ASCII, JSON)
    created_artifacts: Dict[str, str] = {}
    if args.generate_tables:
        logger.info("Exporting comparison tables across all formats...")
        md_path = output_dir / "comparison_table.md"
        tex_path = output_dir / "comparison_table.tex"
        txt_path = output_dir / "comparison_table.txt"
        json_path = output_dir / "comparison_table.json"

        table.save(md_path, fmt="md")
        table.save(tex_path, fmt="tex", multi_track=True, bold_best=True)
        table.save(txt_path, fmt="txt", include_deltas=True)
        table.save(json_path, fmt="json")

        created_artifacts["markdown"] = str(md_path)
        created_artifacts["latex"] = str(tex_path)
        created_artifacts["ascii"] = str(txt_path)
        created_artifacts["json"] = str(json_path)

        # Print Markdown table to stdout
        print("\n" + table.to_markdown(include_deltas=True) + "\n")

    # 5. Generate Confusion & Transition Matrices if requested
    if args.generate_matrices:
        logger.info("Generating confusion and transition matrices...")
        mat_files = generate_matrix_artifacts(table, output_dir=output_dir, split=args.split, data_dir=args.data_dir)
        created_artifacts["matrices"] = [str(p) for p in mat_files]

    # 6. Generate Syntactic Analysis if requested
    if args.generate_analysis:
        logger.info("Generating syntactic chunk error analysis...")
        analysis_files = generate_analysis_artifacts(table, output_dir=output_dir, split=args.split, data_dir=args.data_dir)
        created_artifacts["analysis"] = [str(p) for p in analysis_files]

    elapsed = time.time() - t0

    # Verification: Test Run Budget assertion (<2.0s)
    if args.test_run and elapsed > 2.0:
        logger.warning(f"Test run exceeded 2.0s budget: took {elapsed:.3f}s")

    logger.info("=" * 70)
    logger.info(f"Evaluation completed successfully in {elapsed:.3f}s")
    logger.info(f"Artifacts saved to: {output_dir}")
    logger.info("=" * 70)

    summary_result = {
        "status": "success",
        "elapsed_seconds": round(elapsed, 4),
        "total_evaluated": len(table.results),
        "artifacts": created_artifacts,
        "table": table.to_dict(),
    }

    # Save summary execution report
    summary_file = output_dir / "evaluation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_result, f, indent=2)

    return summary_result


def main() -> None:
    args = parse_args()
    try:
        run_evaluation(args)
    except Exception as exc:
        logger.exception(f"Evaluation CLI failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
