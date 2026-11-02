"""
src/training/utils.py

Reproducibility, hardware detection, mixed precision, and structured logging utilities
for SemEval-2027 Task 2 (DiCo-NLI) training pipeline.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Enforces absolute reproducibility across Python, NumPy, and PyTorch (CPU/CUDA/MPS).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if HAS_NUMPY and np is not None:
        np.random.seed(seed)

    if HAS_TORCH and torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.manual_seed(seed)
            except Exception:
                pass

        if deterministic:
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            if hasattr(torch, "use_deterministic_algorithms"):
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                except Exception:
                    pass


def get_device(preferred: Optional[str] = None) -> Any:
    """
    Detects and returns optimal PyTorch compute device or string fallback ('cuda', 'mps', 'cpu').
    """
    if not HAS_TORCH or torch is None:
        return "cpu"

    if preferred is not None and preferred.strip() and preferred.strip().lower() != "auto":
        pref = preferred.strip().lower()
        if pref.startswith("cuda"):
            if torch.cuda.is_available():
                return torch.device(pref)
            logger.warning("CUDA requested ('%s') but not available; falling back to CPU.", pref)
            return torch.device("cpu")
        elif pref == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            logger.warning("MPS requested but not available; falling back to CPU.")
            return torch.device("cpu")
        elif pref == "cpu":
            return torch.device("cpu")
        else:
            logger.warning("Unknown device '%s'; falling back to auto-detection.", pref)

    # Auto-detection hierarchy: CUDA -> MPS -> CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_info(device: Optional[Any] = None) -> Dict[str, Any]:
    """
    Returns detailed hardware diagnostic dictionary including device type, VRAM, and capabilities.
    """
    dev = device if device is not None else get_device()
    dev_str = str(dev)

    info: Dict[str, Any] = {
        "device": dev_str,
        "device_type": "cpu",
        "has_torch": HAS_TORCH,
        "cuda_available": False,
        "mps_available": False,
    }

    if HAS_TORCH and torch is not None:
        info["cuda_available"] = torch.cuda.is_available()
        info["mps_available"] = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )

        if dev_str.startswith("cuda") and torch.cuda.is_available():
            idx = dev.index if hasattr(dev, "index") and dev.index is not None else 0
            info["device_type"] = "cuda"
            info["device_name"] = torch.cuda.get_device_name(idx)
            info["device_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(idx)
            info["total_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
            info["major_capability"] = props.major
            info["minor_capability"] = props.minor
            info["bf16_supported"] = (
                hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
            )
        elif dev_str == "mps":
            info["device_type"] = "mps"
            info["device_name"] = "Apple Silicon GPU (MPS)"
        else:
            info["device_type"] = "cpu"
            info["device_name"] = "Host CPU"
    else:
        info["device_type"] = "cpu"
        info["device_name"] = "Host CPU (Pure Python / NumPy)"

    return info


def can_use_amp(device: Optional[Any] = None) -> bool:
    """
    Checks whether Automatic Mixed Precision is supported on the target device.
    """
    if not HAS_TORCH or torch is None:
        return False
    dev = device if device is not None else get_device()
    dev_type = dev.type if hasattr(dev, "type") else str(dev)
    if dev_type == "cuda" and torch.cuda.is_available():
        return True
    return False


def get_autocast_dtype(
    fp16: bool = False,
    bf16: bool = False,
    device: Optional[Any] = None,
) -> Any:
    """
    Resolves optimal autocast dtype (torch.bfloat16, torch.float16, or None/float32).
    """
    if not HAS_TORCH or torch is None:
        return None
    if bf16:
        return torch.bfloat16
    if fp16:
        return torch.float16
    return torch.float32


@contextlib.contextmanager
def get_autocast_context(
    device: Optional[Any] = None,
    enabled: bool = True,
    dtype: Optional[Any] = None,
) -> Generator[None, None, None]:
    """
    Unified context manager for PyTorch AMP with graceful fallback.
    """
    if not HAS_TORCH or torch is None or not enabled:
        yield
        return

    dev = device if device is not None else get_device()
    dev_type = dev.type if hasattr(dev, "type") else ("cuda" if "cuda" in str(dev) else "cpu")

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        with torch.amp.autocast(device_type=dev_type, dtype=dtype, enabled=enabled):
            yield
    elif hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast") and dev_type == "cuda":
        with torch.cuda.amp.autocast(enabled=enabled, dtype=dtype):
            yield
    else:
        yield


class JsonlLogger:
    """
    Thread-safe append-only structured JSONL metrics logger.
    """

    def __init__(self, filepath: Union[str, Path]) -> None:
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, record: Dict[str, Any]) -> None:
        """Appends a dictionary record as a JSON line and flushes to disk."""
        record_with_time = dict(record)
        if "timestamp" not in record_with_time:
            record_with_time["timestamp"] = time.time()
        serialized = json.dumps(record_with_time, default=str) + "\n"
        with self._lock:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(serialized)
                f.flush()

    def read_all(self) -> List[Dict[str, Any]]:
        """Reads all records from the JSONL log file."""
        if not self.filepath.is_file():
            return []
        records = []
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    records.append(json.loads(line_str))
        return records
