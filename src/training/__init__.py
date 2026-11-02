"""
src/training/__init__.py

SemEval-2027 Task 2 (DiCo-NLI) Training Infrastructure Module.
Exports Trainer, configuration dataclasses, and execution utilities.
"""

from src.training.utils import (
    JsonlLogger,
    can_use_amp,
    get_autocast_context,
    get_autocast_dtype,
    get_device,
    get_device_info,
    set_seed,
)
from src.training.trainer import (
    CrossEncoderDataset,
    Trainer,
    TrainerConfig,
    TrainingArguments,
)

__all__ = [
    "Trainer",
    "TrainerConfig",
    "TrainingArguments",
    "CrossEncoderDataset",
    "set_seed",
    "get_device",
    "get_device_info",
    "can_use_amp",
    "get_autocast_dtype",
    "get_autocast_context",
    "JsonlLogger",
]
