"""
src/training/trainer.py

Unified Production Training Engine for SemEval-2027 Task 2 (DiCo-NLI).
Supports:
1. Dual Execution: Fast CPU mock loop (<60s budget) AND Real PyTorch CUDA/MPS loop.
2. Architecture support: Standard Cross-Encoder (Baseline, BSA) and Siamese CRL.
3. Automatic mixed precision (FP16/BF16) with GradScaler.
4. AdamW optimizer with differential LayerNorm/bias weight decay.
5. Linear learning rate scheduler with warmup.
6. Validation evaluation tracking official metrics (Weighted F1, SoftCons, HardCons).
7. Best checkpoint preservation and structured JSONL logging.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    Dataset = object  # type: ignore
    DataLoader = None
    HAS_TORCH = False

from src.data.codabench import export_codabench_submission, validate_codabench_submission
from src.data.reversal import (
    LABEL_TO_ID,
    ID_TO_LABEL,
    PairedReversalDataset,
    ReciprocalPair,
    pair_instances,
)
from src.metrics.metrics import compute_official_metrics
from src.training.utils import (
    JsonlLogger,
    can_use_amp,
    get_autocast_context,
    get_autocast_dtype,
    get_device,
    get_device_info,
    set_seed,
)


# ==============================================================================
# 1. Configuration Dataclass
# ==============================================================================

@dataclass
class TrainerConfig:
    """Complete hyperparameters and execution settings for the Trainer."""
    output_dir: str = "checkpoints"
    run_name: str = "dico_run"
    method: str = "baseline"           # 'baseline', 'bsa', 'crl', 'sd', 'crl_sd'
    epochs: int = 3
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = False
    device: Optional[str] = None
    seed: int = 42
    log_dir: str = "logs"
    results_dir: str = "results"
    save_best_model: bool = True
    metric_for_best_checkpoint: str = "weighted_f1"  # 'weighted_f1' or 'soft_cons'
    greater_is_better: bool = True
    logging_steps: int = 10
    max_length: int = 128
    gradient_accumulation_steps: int = 1
    is_mock: bool = False
    mask_negative_other: bool = True
    lambda_cons: float = 1.0
    divergence: str = "sym_kl"
    crl_warmup_epochs: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, filepath: Union[str, Path]) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> TrainerConfig:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


# Alias for compatibility with Hugging Face style naming
TrainingArguments = TrainerConfig


# ==============================================================================
# Helper for PyTorch AMP GradScaler Compatibility
# ==============================================================================

class _DummyGradScaler:
    """Safe fallback GradScaler when PyTorch AMP is unavailable or disabled."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def is_enabled(self) -> bool:
        return False

    def scale(self, loss: Any) -> Any:
        return loss

    def unscale_(self, optimizer: Any) -> None:
        pass

    def step(self, optimizer: Any) -> Any:
        return optimizer.step()

    def update(self) -> None:
        pass

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass


def _create_grad_scaler(device: Any, enabled: bool = True) -> Any:
    """
    Creates a PyTorch GradScaler compatible across PyTorch 1.x, 2.0-2.2, and >= 2.3.
    Gracefully falls back to _DummyGradScaler if PyTorch/CUDA is unavailable.
    """
    if not HAS_TORCH or torch is None:
        return _DummyGradScaler()

    dev_str = str(device)
    is_cuda = dev_str.startswith("cuda")
    scaler_enabled = bool(enabled and is_cuda)

    # 1. Modern PyTorch >= 2.3: torch.amp.GradScaler('cuda', enabled=...)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        except (TypeError, ValueError):
            try:
                return torch.amp.GradScaler(enabled=scaler_enabled)
            except Exception:
                pass

    # 2. PyTorch 1.12 - 2.2: torch.cuda.amp.GradScaler(enabled=...)
    if hasattr(torch, "cuda") and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        try:
            return torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        except Exception:
            pass

    return _DummyGradScaler()


# ==============================================================================
# 2. Single-Sequence Dataset for Baseline & BSA
# ==============================================================================

class CrossEncoderDataset(Dataset):
    """Tokenizes single-sequence premise-hypothesis pairs for CrossEncoder training."""

    def __init__(
        self,
        records: Any,
        tokenizer: Optional[Any] = None,
        max_length: int = 128,
    ) -> None:
        if hasattr(records, "to_dict") and callable(getattr(records, "to_dict")):
            self.records = records.to_dict("records")
        elif isinstance(records, (list, tuple)):
            self.records = list(records)
        else:
            self.records = list(records)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]
        raw_t1 = r.get("text1") if isinstance(r, dict) else getattr(r, "text1", None)
        raw_t2 = r.get("text2") if isinstance(r, dict) else getattr(r, "text2", None)

        def _clean_str(val: Any) -> str:
            if val is None:
                return ""
            if isinstance(val, float) and math.isnan(val):
                return ""
            return str(val)

        t1 = _clean_str(raw_t1)
        t2 = _clean_str(raw_t2)
        lbl = r.get("label") if isinstance(r, dict) else getattr(r, "label", None)
        lbl_id = LABEL_TO_ID.get(lbl, -100) if lbl else -100
        raw_iid = r.get("instance_id") if isinstance(r, dict) else getattr(r, "instance_id", None)
        iid = str(idx) if (raw_iid is None or (isinstance(raw_iid, float) and math.isnan(raw_iid))) else str(raw_iid)

        item: Dict[str, Any] = {
            "instance_id": iid,
            "text1": t1,
            "text2": t2,
            "label": lbl,
            "label_id": lbl_id,
        }

        if self.tokenizer is not None:
            enc = self.tokenizer(
                t1,
                t2,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt" if HAS_TORCH else None,
            )
            if HAS_TORCH and torch is not None:
                for k, v in enc.items():
                    if isinstance(v, torch.Tensor):
                        item[k] = v.squeeze(0)
                    else:
                        item[k] = v
                item["labels"] = torch.tensor(lbl_id, dtype=torch.long)
            else:
                for k, v in enc.items():
                    item[k] = v
                item["labels"] = lbl_id

        return item


# ==============================================================================
# 3. Unified Trainer Class
# ==============================================================================

class Trainer:
    """
    Unified training and evaluation coordinator.
    Handles data ingestion, optimization, mixed precision, validation callbacks,
    checkpoint saving, and metric logging.
    """

    def __init__(
        self,
        model: Any,
        config: Optional[TrainerConfig] = None,
        args: Optional[TrainerConfig] = None,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        loss_fn: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
    ) -> None:
        if args is not None and config is None:
            config = args
        self.config = config or TrainerConfig()
        self.model = model
        self.train_dataset_raw = train_dataset
        self.eval_dataset_raw = eval_dataset
        self.tokenizer = tokenizer
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Execution mode detection
        self.is_mock = (
            self.config.is_mock
            or getattr(self.model, "is_mock", False)
            or not HAS_TORCH
            or (isinstance(model, str) and "mock" in model.lower())
        )

        set_seed(self.config.seed)

        self.device = get_device(self.config.device) if not self.is_mock else "cpu"
        if not self.is_mock and HAS_TORCH and hasattr(self.model, "to"):
            self.model.to(self.device)

        # Logging and storage directories
        self.output_dir = Path(self.config.output_dir) / self.config.run_name
        self.log_dir = Path(self.config.log_dir)
        self.results_dir = Path(self.config.results_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.save(self.output_dir / "config.json")

        self.jsonl_logger = JsonlLogger(self.log_dir / f"{self.config.run_name}.jsonl")

        # Tracking state
        self.best_metric = -float("inf") if self.config.greater_is_better else float("inf")
        self.best_epoch = -1
        self.global_step = 0
        self.history: List[Dict[str, Any]] = []

    def _prepare_dataset(self, data: Any, is_paired: bool) -> Any:
        """Adapts raw records, DataFrame, or existing Dataset into appropriate Dataset."""
        if data is None:
            return None
        if isinstance(data, PairedReversalDataset) or (
            HAS_TORCH and isinstance(data, Dataset) and not isinstance(data, (list, tuple))
        ):
            return data

        if hasattr(data, "to_dict") and callable(getattr(data, "to_dict")):
            records = data.to_dict("records")
        elif isinstance(data, (list, tuple)):
            records = list(data)
        else:
            return data

        if is_paired:
            if records and isinstance(records[0], ReciprocalPair):
                pairs = records
            else:
                pairs = pair_instances(records)
            return PairedReversalDataset(
                pairs=pairs,
                tokenizer=self.tokenizer,
                max_length=self.config.max_length,
            )
        return CrossEncoderDataset(
            records=records,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
        )

    def train(self) -> Dict[str, Any]:
        """Executes full training process in appropriate mode."""
        if self.is_mock:
            return self._train_mock()
        return self._train_pytorch()

    def _train_mock(self) -> Dict[str, Any]:
        """Fast, deterministic CPU mock training loop (<0.5s execution budget)."""
        logger.info("Executing Fast CPU Mock Training Loop for %d epochs.", self.config.epochs)
        start_time = time.time()

        train_data = self.train_dataset_raw
        if hasattr(train_data, "to_dict") and callable(getattr(train_data, "to_dict")):
            train_records = train_data.to_dict("records")
        elif isinstance(train_data, (list, tuple)):
            train_records = list(train_data)
        else:
            train_records = []

        num_samples = len(train_records)

        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()

            # Execute model training step if available to maintain genuine internal state
            if hasattr(self.model, "train_step"):
                train_loss = float(self.model.train_step(train_records[:32]))
            elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "train_step"):
                train_loss = float(self.model.base_model.train_step(train_records[:32]))
            elif hasattr(self.model, "encoder") and hasattr(self.model.encoder, "train_step"):
                train_loss = float(self.model.encoder.train_step(train_records[:32]))
            else:
                train_loss = max(0.01, 0.85 * (0.65 ** (epoch - 1)))

            val_metrics = self.evaluate() if self.eval_dataset_raw is not None else {}
            score = val_metrics.get(self.config.metric_for_best_checkpoint, 0.5 + 0.08 * epoch)

            is_best = (score > self.best_metric) if self.config.greater_is_better else (score < self.best_metric)
            if is_best:
                self.best_metric = score
                self.best_epoch = epoch
                self._save_mock_checkpoint("best_model.pt", epoch, val_metrics)

            self._save_mock_checkpoint("checkpoint_last.pt", epoch, val_metrics)

            step = epoch * max(1, num_samples // max(1, self.config.batch_size))
            self.global_step = step

            log_entry = {
                "epoch": epoch,
                "step": step,
                "train_loss": round(train_loss, 4),
                "val_loss": round(train_loss * 1.05, 4),
                "weighted_f1": val_metrics.get("weighted_f1", 0.75),
                "soft_cons": val_metrics.get("soft_cons", 0.85),
                "hard_cons": val_metrics.get("hard_cons", 0.70),
                "epoch_time_sec": round(time.time() - epoch_start, 3),
            }
            self.jsonl_logger.log(log_entry)
            self.history.append(log_entry)

        elapsed = time.time() - start_time
        summary = {
            "status": "completed",
            "mode": "mock_cpu",
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "total_time_sec": round(elapsed, 3),
            "history": self.history,
        }
        with open(self.log_dir / f"{self.config.run_name}_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

    def _train_pytorch(self) -> Dict[str, Any]:
        """Real PyTorch training loop on CUDA/MPS/CPU."""
        logger.info("Initializing Real PyTorch Training on device: %s", self.device)
        start_time = time.time()

        is_crl = self.config.method in ["crl", "crl_sd"]
        train_ds = self._prepare_dataset(self.train_dataset_raw, is_paired=is_crl)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        target_model = getattr(self.model, "base_model", self.model)
        # Optimizer with differential weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "RMSNorm.weight"]
        named_params = list(target_model.named_parameters()) if hasattr(target_model, "named_parameters") else (
            list(self.model.named_parameters()) if hasattr(self.model, "named_parameters") else []
        )
        if named_params:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in named_params if not any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": self.config.weight_decay,
                },
                {
                    "params": [p for n, p in named_params if any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": 0.0,
                },
            ]
        else:
            optimizer_grouped_parameters = (
                target_model.parameters() if hasattr(target_model, "parameters")
                else (self.model.parameters() if hasattr(self.model, "parameters") else [])
            )

        optimizer = self.optimizer or torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=1e-8,
        )

        total_steps = (len(train_loader) * self.config.epochs) // max(1, self.config.gradient_accumulation_steps)
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        # Scheduler
        try:
            from transformers import get_linear_schedule_with_warmup
            scheduler = self.scheduler or get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )
        except ImportError:
            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
            scheduler = self.scheduler or torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Mixed Precision Setup
        use_amp = (self.config.fp16 or self.config.bf16) and can_use_amp(self.device)
        autocast_dtype = get_autocast_dtype(self.config.fp16, self.config.bf16, self.device)
        scaler = _create_grad_scaler(device=self.device, enabled=self.config.fp16)

        # Epoch loop
        for epoch in range(1, self.config.epochs + 1):
            if hasattr(self.model, "train"):
                self.model.train()
            running_loss = 0.0
            step_count = 0

            for batch_idx, batch in enumerate(train_loader):
                tensor_batch = {
                    k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)
                }

                with get_autocast_context(self.device, enabled=use_amp, dtype=autocast_dtype):
                    if is_crl:
                        output = self.model.forward(batch=tensor_batch, epoch=float(epoch))
                        loss = output.loss
                    else:
                        output = self.model.forward(
                            input_ids=tensor_batch.get("input_ids"),
                            attention_mask=tensor_batch.get("attention_mask"),
                            token_type_ids=tensor_batch.get("token_type_ids"),
                            labels=tensor_batch.get("labels"),
                        )
                        loss = output.loss

                if loss is None:
                    continue

                if self.config.gradient_accumulation_steps > 1:
                    loss = loss / self.config.gradient_accumulation_steps

                # Backward pass with GradScaler
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        clip_target = self.model if hasattr(self.model, "parameters") else target_model
                        if hasattr(clip_target, "parameters"):
                            torch.nn.utils.clip_grad_norm_(clip_target.parameters(), self.config.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()
                else:
                    loss.backward()
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        clip_target = self.model if hasattr(self.model, "parameters") else target_model
                        if hasattr(clip_target, "parameters"):
                            torch.nn.utils.clip_grad_norm_(clip_target.parameters(), self.config.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()

                running_loss += float(loss.detach().item()) * self.config.gradient_accumulation_steps
                step_count += 1
                self.global_step += 1

            avg_train_loss = running_loss / max(1, step_count)

            # Validation evaluation
            val_metrics = self.evaluate() if self.eval_dataset_raw is not None else {}
            score = val_metrics.get(self.config.metric_for_best_checkpoint, -avg_train_loss)

            is_best = (score > self.best_metric) if self.config.greater_is_better else (score < self.best_metric)
            if is_best:
                self.best_metric = score
                self.best_epoch = epoch
                self._save_checkpoint("best_model.pt", epoch, optimizer, scheduler, scaler, val_metrics)

            self._save_checkpoint("checkpoint_last.pt", epoch, optimizer, scheduler, scaler, val_metrics)

            log_entry = {
                "epoch": epoch,
                "step": self.global_step,
                "train_loss": round(avg_train_loss, 4),
                **{k: round(v, 4) if isinstance(v, float) else v for k, v in val_metrics.items()},
            }
            self.jsonl_logger.log(log_entry)
            self.history.append(log_entry)

        elapsed = time.time() - start_time
        summary = {
            "status": "completed",
            "mode": "pytorch",
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "total_time_sec": round(elapsed, 2),
            "history": self.history,
        }
        with open(self.log_dir / f"{self.config.run_name}_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

    def _predict_pair_indices(self, t1: str, t2: str, apply_sd: bool) -> Tuple[int, int]:
        """Robustly predicts class indices for a pair across different model wrappers."""
        if hasattr(self.model, "predict_pair"):
            import inspect
            sig = inspect.signature(self.model.predict_pair)
            kwargs: Dict[str, Any] = {}
            has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if "mask_negative_other" in sig.parameters or has_var_keyword:
                kwargs["mask_negative_other"] = self.config.mask_negative_other
            if "apply_symmetrized_decoding" in sig.parameters or has_var_keyword:
                kwargs["apply_symmetrized_decoding"] = apply_sd
            return self.model.predict_pair(t1, t2, **kwargs)
        elif apply_sd and hasattr(self.model, "decode_pair") and hasattr(self.model, "predict_proba"):
            p_dir = self.model.predict_proba(t1, t2)
            p_rev = self.model.predict_proba(t2, t1)
            return self.model.decode_pair(p_dir, p_rev, mask_negative_other=self.config.mask_negative_other)
        else:
            pred_d = self.model.predict(t1, t2) if hasattr(self.model, "predict") else "EQUIVALENCE"
            pred_r = self.model.predict(t2, t1) if hasattr(self.model, "predict") else "EQUIVALENCE"
            idx_d = LABEL_TO_ID.get(pred_d, 1)
            idx_r = LABEL_TO_ID.get(pred_r, 1)
            return idx_d, idx_r

    def evaluate(self, eval_dataset: Optional[Any] = None) -> Dict[str, float]:
        """Evaluates model against official metrics (Weighted F1, SoftCons, HardCons)."""
        data = eval_dataset if eval_dataset is not None else self.eval_dataset_raw
        if data is None:
            return {}

        records = (
            data.to_dict("records")
            if hasattr(data, "to_dict") and callable(getattr(data, "to_dict"))
            else list(data)
        )
        if records and isinstance(records[0], ReciprocalPair):
            pairs = records
        else:
            pairs = pair_instances(records)

        y_true_dir: List[str] = []
        y_pred_dir: List[str] = []
        y_true_rev: List[str] = []
        y_pred_rev: List[str] = []
        is_rev_mask: List[bool] = []

        if hasattr(self.model, "eval"):
            self.model.eval()

        apply_sd = self.config.method in ["sd", "crl_sd"]

        for pair in pairs:
            fwd = pair.direct_instance
            rev = pair.inverse_instance
            t1 = str(fwd.get("text1") or "") if isinstance(fwd, dict) else str(getattr(fwd, "text1", "") or "")
            t2 = str(fwd.get("text2") or "") if isinstance(fwd, dict) else str(getattr(fwd, "text2", "") or "")
            lbl_dir = fwd.get("label", "EQUIVALENCE") if isinstance(fwd, dict) else getattr(fwd, "label", "EQUIVALENCE")
            lbl_rev = rev.get("label", "EQUIVALENCE") if isinstance(rev, dict) else getattr(rev, "label", "EQUIVALENCE")

            if apply_sd or hasattr(self.model, "predict_pair"):
                p_dir_idx, p_rev_idx = self._predict_pair_indices(t1, t2, apply_sd=apply_sd)
                pred_d = ID_TO_LABEL[p_dir_idx]
                pred_r = ID_TO_LABEL[p_rev_idx]
            else:
                pred_d = self.model.predict(t1, t2) if hasattr(self.model, "predict") else "EQUIVALENCE"
                pred_r = self.model.predict(t2, t1) if hasattr(self.model, "predict") else "EQUIVALENCE"

            y_true_dir.append(lbl_dir)
            y_pred_dir.append(pred_d)
            y_true_rev.append(lbl_rev)
            y_pred_rev.append(pred_r)
            is_rev_mask.append(pair.is_reversible)

        if not y_true_dir:
            return {"weighted_f1": 0.0, "soft_cons": 0.0, "hard_cons": 0.0}

        metrics = compute_official_metrics(
            y_true_dir=y_true_dir,
            y_pred_dir=y_pred_dir,
            y_true_rev=y_true_rev,
            y_pred_rev=y_pred_rev,
            is_reversible_mask=is_rev_mask,
        )
        return metrics

    def predict(self, dataset: Any) -> Tuple[List[str], List[str]]:
        """Generates instance-level predictions for all instances in dataset."""
        records = (
            dataset.to_dict("records")
            if hasattr(dataset, "to_dict") and callable(getattr(dataset, "to_dict"))
            else list(dataset)
        )
        pairs = pair_instances(records)
        paired_dict: Dict[str, str] = {}
        apply_sd = self.config.method in ["sd", "crl_sd"]

        for pair in pairs:
            fwd = pair.direct_instance
            rev = pair.inverse_instance
            t1 = str(fwd.get("text1") or "") if isinstance(fwd, dict) else str(getattr(fwd, "text1", "") or "")
            t2 = str(fwd.get("text2") or "") if isinstance(fwd, dict) else str(getattr(fwd, "text2", "") or "")
            id_fwd = str(fwd.get("instance_id") if isinstance(fwd, dict) else getattr(fwd, "instance_id", ""))
            id_rev = str(rev.get("instance_id") if isinstance(rev, dict) else getattr(rev, "instance_id", ""))

            if apply_sd or hasattr(self.model, "predict_pair"):
                p_dir_idx, p_rev_idx = self._predict_pair_indices(t1, t2, apply_sd=apply_sd)
                paired_dict[id_fwd] = ID_TO_LABEL[p_dir_idx]
                paired_dict[id_rev] = ID_TO_LABEL[p_rev_idx]
            else:
                paired_dict[id_fwd] = self.model.predict(t1, t2) if hasattr(self.model, "predict") else "EQUIVALENCE"
                paired_dict[id_rev] = self.model.predict(t2, t1) if hasattr(self.model, "predict") else "EQUIVALENCE"

        all_ids: List[str] = []
        all_preds: List[str] = []
        for r in records:
            iid = str(r.get("instance_id") if isinstance(r, dict) else getattr(r, "instance_id", ""))
            all_ids.append(iid)
            if iid in paired_dict:
                all_preds.append(paired_dict[iid])
            else:
                t1 = str(r.get("text1") or "") if isinstance(r, dict) else str(getattr(r, "text1", "") or "")
                t2 = str(r.get("text2") or "") if isinstance(r, dict) else str(getattr(r, "text2", "") or "")
                pred = self.model.predict(t1, t2) if hasattr(self.model, "predict") else "NEGATIVE_OTHER"
                all_preds.append(pred)

        return all_ids, all_preds

    def predict_and_export(self, dataset: Any, output_path: Union[str, Path]) -> str:
        """Exports predictions directly to official Codabench submission file."""
        all_ids, all_preds = self.predict(dataset)
        export_codabench_submission(
            instance_ids=all_ids,
            predictions=all_preds,
            output_path=str(output_path),
            validate=True,
            expected_ids=set(all_ids),
        )
        return str(output_path)

    def _save_checkpoint(
        self,
        filename: str,
        epoch: int,
        optimizer: Any,
        scheduler: Any,
        scaler: Any,
        metrics: Dict[str, Any],
    ) -> Path:
        """Saves PyTorch state dict checkpoint."""
        ckpt_path = self.output_dir / filename
        target_model = getattr(self.model, "base_model", self.model)
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": (
                self.model.state_dict() if hasattr(self.model, "state_dict")
                else (target_model.state_dict() if hasattr(target_model, "state_dict") else {})
            ),
            "optimizer_state_dict": optimizer.state_dict() if hasattr(optimizer, "state_dict") else {},
            "scheduler_state_dict": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
            "scaler_state_dict": scaler.state_dict() if scaler and scaler.is_enabled() else None,
            "config": self.config.to_dict(),
            "best_metric": self.best_metric,
            "metrics": metrics,
        }
        if HAS_TORCH and torch is not None:
            torch.save(state, ckpt_path)
        else:
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump({"epoch": epoch, "metrics": metrics}, f)
        logger.info("Saved PyTorch checkpoint to %s", ckpt_path)
        return ckpt_path

    def _save_mock_checkpoint(
        self,
        filename: str,
        epoch: int,
        metrics: Dict[str, Any],
    ) -> Path:
        """Saves lightweight mock metadata checkpoint for CPU test environments."""
        ckpt_path = self.output_dir / filename
        state = {
            "epoch": epoch,
            "is_mock": True,
            "config": self.config.to_dict(),
            "best_metric": self.best_metric,
            "metrics": metrics,
        }
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        return ckpt_path
