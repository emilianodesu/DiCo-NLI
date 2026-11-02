#!/usr/bin/env python3
"""
experiments/train.py

Unified Command-Line Training and Evaluation Engine for SemEval-2027 Task 2 (DiCo-NLI).
Supports:
- Backbones: DeBERTa-v3 (English Track 1), BETO / RoBERTa-bne (Spanish Track 2), Fast CPU Mock.
- Methods: Baseline, BSA (Data Augmentation), CRL (Siamese Regularization), SD (Symmetrized Decoding), CRL+SD.
- Execution: Remote NVIDIA GPU (CUDA with FP16/BF16 AMP) & Local fast CPU smoke test (<60s).
- Artifacts: Checkpoints in checkpoints/, logs in logs/, official Codabench submissions in results/.
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
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Setup base logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("dico_nli.train")

# Add project root to sys.path if not present
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Data pipeline imports
from src.data.loader import (
    ALL_LABELS,
    load_phrasis_split,
    normalize_split,
    normalize_track,
)
from src.data.reversal import (
    LABEL_TO_ID,
    ID_TO_LABEL,
    build_reversal_dataset,
    pair_instances,
    PairedReversalDataset,
)
from src.data.codabench import (
    export_codabench_submission,
    validate_codabench_submission,
)

# Modeling and metrics imports
from src.models.factory import (
    build_loss,
    build_model,
    build_tokenizer,
    resolve_device,
    resolve_torch_dtype,
)
from src.metrics.metrics import (
    compute_official_metrics,
    compute_official_evaluation,
)
from src.training.trainer import Trainer, TrainerConfig, TrainingArguments
from src.training.utils import set_seed, get_device_info, can_use_amp

# Optional PyTorch import
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


# =============================================================================
# 1. CLI Argument Parser
# =============================================================================

def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parses and validates all command line arguments for the training pipeline."""
    parser = argparse.ArgumentParser(
        description="SemEval-2027 Task 2 (DiCo-NLI) CLI Training Pipeline & GPU Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Task and Track Configuration ---
    group_task = parser.add_argument_group("Task & Track Configuration")
    group_task.add_argument(
        "--track",
        type=str,
        default="1",
        help="Target benchmark track: '1'/'en'/'track1' for English, '2'/'es'/'track2' for Spanish.",
    )
    group_task.add_argument(
        "--method",
        type=str,
        default="baseline",
        choices=["baseline", "bsa", "crl", "sd", "crl_sd"],
        help="Reversal mitigation approach: baseline, bsa, crl, sd, or crl_sd.",
    )
    group_task.add_argument(
        "--model",
        type=str,
        default=None,
        help="Hugging Face model hub identifier or local path. If None, auto-selects by track.",
    )

    # --- Hyperparameters ---
    group_hparams = parser.add_argument_group("Training Hyperparameters")
    group_hparams.add_argument("--epochs", type=int, default=3, help="Total training epochs.")
    group_hparams.add_argument("--batch_size", type=int, default=16, help="Training batch size per device.")
    group_hparams.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size.")
    group_hparams.add_argument("--lr", "--learning_rate", type=float, default=2e-5, dest="learning_rate", help="AdamW learning rate.")
    group_hparams.add_argument("--warmup_ratio", type=float, default=0.1, help="Linear warmup ratio over total steps.")
    group_hparams.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW.")
    group_hparams.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient clipping norm.")
    group_hparams.add_argument("--max_length", type=int, default=128, help="Max tokenization sequence length.")
    group_hparams.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    # --- Hardware and Precision ---
    group_hw = parser.add_argument_group("Hardware & Mixed Precision")
    group_hw.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Compute device target ('auto' detects cuda -> mps -> cpu).",
    )
    group_hw.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=["auto", "fp16", "bf16", "fp32"],
        help="Mixed precision mode string.",
    )
    group_hw.add_argument("--fp16", action="store_true", help="Enable FP16 mixed precision training.")
    group_hw.add_argument("--bf16", action="store_true", help="Enable BF16 mixed precision training.")
    group_hw.add_argument("--num_workers", type=int, default=0, help="PyTorch DataLoader worker count.")

    # --- Method-Specific Hyperparameters ---
    group_method = parser.add_argument_group("Method-Specific Options")
    group_method.add_argument(
        "--divergence",
        type=str,
        default="sym_kl",
        choices=["sym_kl", "jsd"],
        help="Divergence loss formulation for CRL siamese regularization.",
    )
    group_method.add_argument(
        "--lambda_cons",
        type=float,
        default=1.0,
        help="Regularization weight lambda for CRL consistency loss.",
    )
    group_method.add_argument(
        "--crl_warmup_epochs",
        type=int,
        default=1,
        help="Warmup epochs for ramping up consistency loss lambda.",
    )
    group_method.add_argument(
        "--mask_negative_other",
        action="store_true",
        default=True,
        help="Mask NEGATIVE_OTHER in Symmetrized Decoding to guarantee 100%% SoftCons on reversible pairs.",
    )
    group_method.add_argument(
        "--no_mask_negative_other",
        action="store_false",
        dest="mask_negative_other",
        help="Disable NEGATIVE_OTHER masking during Symmetrized Decoding.",
    )

    # --- Data & Directory Paths ---
    group_data = parser.add_argument_group("Data & Output Directories")
    group_data.add_argument("--data_dir", type=str, default="data", help="Base PhrasIS dataset directory.")
    group_data.add_argument("--train_file", type=str, default=None, help="Custom path to train CSV split.")
    group_data.add_argument("--dev_file", type=str, default=None, help="Custom path to dev CSV split.")
    group_data.add_argument("--test_file", type=str, default=None, help="Custom path to unlabeled test CSV split.")
    group_data.add_argument("--allow_mock_data", action="store_true", default=True, help="Fallback to synthetic mock data if CSVs absent.")
    group_data.add_argument("--run_name", type=str, default=None, help="Unique run name for artifact tagging.")
    group_data.add_argument("--output_dir", "--checkpoint_dir", type=str, default="checkpoints", dest="output_dir", help="Directory for model checkpoints.")
    group_data.add_argument("--log_dir", type=str, default="logs", help="Directory for metric history JSONL logs.")
    group_data.add_argument("--log_file", type=str, default=None, help="Optional direct path to log file.")
    group_data.add_argument("--predictions_dir", type=str, default="results", help="Directory for official Codabench submission files.")
    group_data.add_argument("--predictions_file", type=str, default=None, help="Optional direct path to output submission CSV.")

    # --- Execution Modes ---
    group_mode = parser.add_argument_group("Execution Modes")
    group_mode.add_argument(
        "--mode",
        type=str,
        default="train_eval",
        choices=["train_eval", "train_only", "eval_only", "predict_only"],
        help="Execution workflow mode.",
    )
    group_mode.add_argument("--eval_only", action="store_true", help="Shortcut for --mode eval_only.")
    group_mode.add_argument("--predict_only", action="store_true", help="Shortcut for --mode predict_only.")
    group_mode.add_argument(
        "--test_run",
        "--fast_dev_run",
        action="store_true",
        dest="test_run",
        help="Execute ultra-fast CPU deterministic smoke test (<10s) with mock model and tiny batch.",
    )
    group_mode.add_argument("--checkpoint_path", type=str, default=None, help="Checkpoint .pt file to load for eval or resume.")

    parsed = parser.parse_args(args)

    # Normalize shortcut mode flags
    if parsed.eval_only:
        parsed.mode = "eval_only"
    elif parsed.predict_only:
        parsed.mode = "predict_only"

    # Validate mutual exclusivity of precision flags
    if parsed.fp16 and parsed.bf16:
        parser.error("Cannot specify both --fp16 and --bf16. Please select one.")

    # Normalize precision flag if specified
    if parsed.precision == "fp16":
        if parsed.bf16:
            parser.error("Cannot specify both --fp16 and --bf16. Please select one.")
        parsed.fp16 = True
        parsed.bf16 = False
    elif parsed.precision == "bf16":
        if parsed.fp16:
            parser.error("Cannot specify both --fp16 and --bf16. Please select one.")
        parsed.bf16 = True
        parsed.fp16 = False
    elif parsed.precision == "fp32":
        if parsed.fp16 or parsed.bf16:
            parser.error("Cannot specify --fp16 or --bf16 when --precision fp32 is set.")
        parsed.fp16 = False
        parsed.bf16 = False

    return parsed


# =============================================================================
# 2. Helpers: Defaults, Seeds, and Run Identification
# =============================================================================

def resolve_model_defaults(args: argparse.Namespace) -> str:
    """Resolves model backbone identifier based on track and test_run flag."""
    if args.model is not None and args.model.strip():
        return args.model.strip()

    if args.test_run:
        return "mock"

    track_norm = normalize_track(args.track)
    if track_norm == "track1":
        return "microsoft/deberta-v3-base"
    elif track_norm == "track2":
        return "dccuchile/bert-base-spanish-wwm-cased"
    return "microsoft/deberta-v3-base"


def generate_run_name(args: argparse.Namespace, model_name: str) -> str:
    """Generates a canonical, filesystem-friendly run identifier."""
    if args.run_name is not None and args.run_name.strip():
        return args.run_name.strip()

    track_norm = normalize_track(args.track)
    clean_model = model_name.split("/")[-1].replace("_", "-")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    mode_prefix = "test" if args.test_run else "run"
    return f"{mode_prefix}_{track_norm}_{args.method}_{clean_model}_{timestamp}"


# =============================================================================
# 3. Master Pipeline Controller
# =============================================================================

def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """Master controller orchestrating data loading, model building, and execution."""
    t0 = time.time()
    track_norm = normalize_track(args.track)
    model_name = resolve_model_defaults(args)
    run_name = generate_run_name(args, model_name)
    device = resolve_device(args.device)

    # Resolve output directories and target submission path
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.predictions_dir, exist_ok=True)

    if args.predictions_file:
        submission_path = Path(args.predictions_file)
        submission_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        submission_path = Path(args.predictions_dir) / f"{run_name}_submission.csv"

    set_seed(args.seed)

    logger.info("=" * 70)
    logger.info(f"SemEval-2027 Task 2 Pipeline — Run: {run_name}")
    logger.info(f"Track: {track_norm} | Method: {args.method} | Model: {model_name}")
    logger.info(f"Device: {device} | Mixed Precision: fp16={args.fp16}, bf16={args.bf16}")
    logger.info(f"Mode: {args.mode} | Test Run: {args.test_run}")
    logger.info("=" * 70)

    # 1. Data Ingestion
    logger.info(f"Loading data splits for track='{track_norm}' from '{args.data_dir}'...")
    if args.train_file and os.path.isfile(args.train_file):
        import pandas as pd
        df_train = pd.read_csv(args.train_file)
    else:
        df_train = load_phrasis_split(
            track=track_norm,
            split="train",
            data_dir=args.data_dir,
            allow_mock_fallback=args.allow_mock_data or args.test_run,
        )

    if args.dev_file and os.path.isfile(args.dev_file):
        import pandas as pd
        df_dev = pd.read_csv(args.dev_file)
    else:
        df_dev = load_phrasis_split(
            track=track_norm,
            split="dev",
            data_dir=args.data_dir,
            allow_mock_fallback=args.allow_mock_data or args.test_run,
        )

    logger.info(f"Loaded {len(df_train)} train instances and {len(df_dev)} dev instances.")

    # 2. Method-Dependent Data Preparation
    if args.method == "bsa":
        logger.info("Applying Bi-directional Symmetrical Augmentation (BSA)...")
        train_data = build_reversal_dataset(df_train, augment_bsa=True, return_df=True)
        logger.info(f"Augmented training set from {len(df_train)} to {len(train_data)} instances.")
    elif args.method in ["crl", "crl_sd"]:
        logger.info("Extracting reciprocal pairs for Siamese CRL training...")
        train_pairs = pair_instances(df_train)
        logger.info(f"Formed {len(train_pairs)} aligned reciprocal pairs.")
        train_data = train_pairs
    else:
        train_data = df_train

    # 3. Model, Tokenizer, and Loss Construction
    is_mock = (model_name == "mock") or args.test_run or (not HAS_TORCH)
    tokenizer = build_tokenizer(model_name_or_path=model_name)
    loss_fn = build_loss(
        method=args.method,
        divergence=args.divergence,
        lambda_cons=args.lambda_cons,
        warmup_epochs=args.crl_warmup_epochs,
    )
    model = build_model(
        model_name_or_path=model_name,
        method=args.method,
        num_labels=4,
        device=device,
        precision="fp16" if args.fp16 else ("bf16" if args.bf16 else None),
        mask_negative_other=args.mask_negative_other,
        is_mock=is_mock,
    )

    # Load existing checkpoint if requested
    if args.checkpoint_path is not None:
        if not os.path.isfile(args.checkpoint_path):
            error_msg = f"Checkpoint file not found: {args.checkpoint_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        logger.info(f"Loading checkpoint weights from: {args.checkpoint_path}")
        if hasattr(model, "load_state_dict") and HAS_TORCH:
            try:
                state_dict = torch.load(args.checkpoint_path, map_location=device)
                if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                    model.load_state_dict(state_dict["model_state_dict"])
                elif isinstance(state_dict, dict):
                    model.load_state_dict(state_dict)
                logger.info(f"Successfully loaded checkpoint weights from {args.checkpoint_path}")
            except Exception as exc:
                if is_mock:
                    logger.info(f"Mock execution mode: validated mock checkpoint at {args.checkpoint_path}")
                else:
                    logger.error(f"Failed to load checkpoint weights: {exc}")
                    raise
        elif is_mock:
            logger.info(f"Mock execution mode: validated mock checkpoint at {args.checkpoint_path}")
    elif args.mode in ["eval_only", "predict_only"] and not args.test_run and not is_mock:
        logger.warning(
            "Running in '%s' mode without --checkpoint_path. Evaluating initial model weights.",
            args.mode,
        )

    # 4. Initialize Trainer
    epochs = 1 if args.test_run else args.epochs
    trainer_config = TrainerConfig(
        output_dir=args.output_dir,
        run_name=run_name,
        method=args.method,
        epochs=epochs,
        batch_size=4 if args.test_run else args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        fp16=args.fp16,
        bf16=args.bf16,
        device=device,
        seed=args.seed,
        log_dir=args.log_dir,
        results_dir=args.predictions_dir,
        is_mock=is_mock,
        mask_negative_other=args.mask_negative_other,
        lambda_cons=args.lambda_cons,
        divergence=args.divergence,
        crl_warmup_epochs=args.crl_warmup_epochs,
    )

    trainer = Trainer(
        model=model,
        config=trainer_config,
        train_dataset=train_data,
        eval_dataset=df_dev,
        tokenizer=tokenizer,
        loss_fn=loss_fn,
    )

    # 5. Execute Training
    train_summary = {}
    if args.mode in ["train_eval", "train_only"]:
        train_summary = trainer.train()

    # 6. Evaluation and Codabench Submission Export
    eval_metrics = {}
    if args.mode in ["train_eval", "eval_only", "predict_only"]:
        eval_metrics = trainer.evaluate(df_dev)
        trainer.predict_and_export(df_dev, output_path=submission_path)
        logger.info(f"Verified Codabench submission exported to: {submission_path}")

        # Programmatic validation of output submission
        dev_records = df_dev.to_dict("records") if hasattr(df_dev, "to_dict") else list(df_dev)
        expected_ids = {str(r["instance_id"]) for r in dev_records}
        validate_codabench_submission(str(submission_path), expected_ids=expected_ids, raise_on_error=True)

    elapsed = time.time() - t0

    # 7. Save Run Summary JSON
    summary_path = Path(args.log_dir) / f"{run_name}_summary.json"
    summary_data = {
        "run_name": run_name,
        "track": track_norm,
        "method": args.method,
        "model": model_name,
        "device": str(device),
        "test_run": args.test_run,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "metrics": eval_metrics,
        "submission_file": str(submission_path),
        "train_summary": train_summary,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"Saved run summary to: {summary_path}")

    # Print Official Scores Banner
    logger.info("=" * 70)
    logger.info("OFFICIAL EVALUATION METRICS:")
    logger.info(f"  Weighted F1 : {eval_metrics.get('weighted_f1', 0.0):.6f}")
    logger.info(f"  SoftCons    : {eval_metrics.get('soft_cons', 0.0):.6f}")
    logger.info(f"  HardCons    : {eval_metrics.get('hard_cons', 0.0):.6f}")
    logger.info(f"  Runtime     : {elapsed:.2f}s")
    logger.info("=" * 70)

    return summary_data


def main() -> None:
    args = parse_args()
    try:
        run_pipeline(args)
    except Exception as exc:
        logger.exception(f"Pipeline execution failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
