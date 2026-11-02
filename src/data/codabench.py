"""
src/data/codabench.py
Strict 2-Column Codabench Submission Exporter & Format Validator.
Enforces exact parity with SemEval-2027 Task 2 official evaluation container.
"""

from __future__ import annotations

import csv
import io
import os
import re
from dataclasses import dataclass, field
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

# Optional pandas integration
try:
    import pandas as pd  # type: ignore
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None  # type: ignore


# ============================================================================
# 1. Constants & Limits matching official evaluation_functions
# ============================================================================

EXPECTED_HEADER: List[str] = ["instance_id", "label"]

VALID_LABELS: Set[str] = {
    "BACKWARD_ENTAILMENT",
    "EQUIVALENCE",
    "FORWARD_ENTAILMENT",
    "NEGATIVE_OTHER",
}

DEFAULT_MAX_BYTES: int = 64 * 1024 * 1024  # 64 MB
DEFAULT_MAX_LINES: int = 5_000_000         # 5 million rows


# ============================================================================
# 2. Exceptions & Result Data Structures
# ============================================================================

class CodabenchValidationError(ValueError):
    """Raised when a submission file violates official Codabench specifications."""
    pass


@dataclass
class CodabenchValidationResult:
    """Summary of Codabench submission verification."""
    is_valid: bool
    total_records: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    label_distribution: Dict[str, int] = field(default_factory=dict)
    detected_delimiter: str = ","
    file_size_bytes: int = 0

    def summary(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        lines = [
            f"=== Codabench Validation Report: {status} ===",
            f"Total Instances: {self.total_records}",
            f"File Size: {self.file_size_bytes} bytes",
            f"Delimiter: {repr(self.detected_delimiter)}",
            f"Label Distribution: {dict(self.label_distribution)}",
        ]
        if self.errors:
            lines.append(f"Errors ({len(self.errors)}):")
            for err in self.errors[:10]:
                lines.append(f"  - {err}")
            if len(self.errors) > 10:
                lines.append(f"  ... and {len(self.errors) - 10} more errors.")
        return "\n".join(lines)


# ============================================================================
# 3. Format Validator Engine
# ============================================================================

def validate_codabench_submission(
    submission_path: Union[str, Path],
    expected_ids: Optional[Set[str]] = None,
    raise_on_error: bool = True,
) -> bool:
    """
    Validates a submission file against official Codabench rules.

    Checks:
    1. File exists, non-empty, and <= 64MB limit.
    2. UTF-8 (or UTF-8-sig) encoded, zero NUL bytes.
    3. Exactly 2 columns with header 'instance_id,label'. Extra columns rejected!
    4. Zero blank rows anywhere in file (including trailing empty line).
    5. 'instance_id' is non-empty, unique, and contains no whitespace characters.
    6. 'label' is non-empty and strictly belongs to VALID_LABELS enum.
    7. Bijective set equality with expected_ids if provided.

    Args:
        submission_path: Path to the submission CSV or TSV file.
        expected_ids: Optional set of gold instance IDs to match bijectively.
        raise_on_error: If True (default), raises CodabenchValidationError on failure.

    Returns:
        True if validation succeeds.

    Raises:
        CodabenchValidationError: If any constraint is violated.
    """
    path = Path(submission_path)
    if not path.is_file():
        raise CodabenchValidationError(f"File not found: {submission_path}")

    file_size = path.stat().st_size
    if file_size == 0:
        raise CodabenchValidationError("Submission file is empty")

    if file_size > DEFAULT_MAX_BYTES:
        raise CodabenchValidationError(
            f"File size ({file_size} bytes) exceeds limit of {DEFAULT_MAX_BYTES} bytes (64MB)."
        )

    raw_bytes = path.read_bytes()
    if b"\x00" in raw_bytes:
        raise CodabenchValidationError("File contains NUL bytes")

    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CodabenchValidationError(f"Failed to decode file as UTF-8: {exc}")

    lines = text.splitlines(keepends=False)
    if not lines:
        raise CodabenchValidationError("Submission file is empty")

    # Check for trailing empty line (multiple consecutive newlines at EOF or blank end)
    if text.endswith("\n\n") or text.endswith("\r\n\r\n") or text.endswith("\n\r\n"):
        raise CodabenchValidationError("File contains trailing blank line(s)")

    header_line = lines[0].strip()
    delimiter = "\t" if "\t" in header_line and "," not in header_line else ","
    reader = csv.reader(lines, delimiter=delimiter)

    header = next(reader, None)
    if not header:
        raise CodabenchValidationError("Missing header row")

    if header != EXPECTED_HEADER:
        raise CodabenchValidationError(
            f"Invalid header {header}. Expected exactly {EXPECTED_HEADER}"
        )

    seen_ids: Set[str] = set()
    for row_idx, row in enumerate(reader, start=2):
        if not row or (len(row) == 1 and not row[0].strip()):
            raise CodabenchValidationError(f"Blank row at line {row_idx}")

        if len(row) != 2:
            raise CodabenchValidationError(
                f"Line {row_idx} has {len(row)} columns, expected exactly 2"
            )

        inst_id, label = row[0], row[1]

        if not inst_id or not inst_id.strip():
            raise CodabenchValidationError(f"Empty instance_id at line {row_idx}")

        if any(c.isspace() for c in inst_id):
            raise CodabenchValidationError(
                f"Identifier '{inst_id}' at line {row_idx} contains whitespace"
            )

        if inst_id in seen_ids:
            raise CodabenchValidationError(
                f"Duplicate instance_id '{inst_id}' at line {row_idx}"
            )
        seen_ids.add(inst_id)

        if label not in VALID_LABELS:
            raise CodabenchValidationError(
                f"Invalid label '{label}' at line {row_idx}. Must be one of {sorted(VALID_LABELS)}"
            )

    if expected_ids is not None:
        missing = expected_ids - seen_ids
        if missing:
            sample_missing = sorted(list(missing))[:3]
            raise CodabenchValidationError(
                f"Missing {len(missing)} required predictions. Samples: {sample_missing}"
            )
        unknown = seen_ids - expected_ids
        if unknown:
            sample_unknown = sorted(list(unknown))[:3]
            raise CodabenchValidationError(
                f"Found {len(unknown)} unknown instance IDs. Samples: {sample_unknown}"
            )

    return True


# ============================================================================
# 4. Submission Exporter Engine
# ============================================================================

def export_codabench_submission(
    instance_ids: Union[Sequence[str], Mapping[str, str], Any],
    predictions: Optional[Sequence[str]] = None,
    output_path: Union[str, Path] = "submission.csv",
    delimiter: str = ",",
    validate: bool = True,
    expected_ids: Optional[Set[str]] = None,
) -> None:
    """
    Exports predictions to a strict two-column Codabench submission file.

    Supports:
        - Lists of IDs and predictions: export_codabench_submission(ids, preds, path)
        - Mapping / Dict: export_codabench_submission(pred_dict, output_path=path)
        - pandas DataFrame: export_codabench_submission(df, output_path=path)

    Formatting guarantees:
        - UTF-8 without BOM.
        - Exactly 2 columns: instance_id,label.
        - Unix LF line endings.
        - No blank rows or trailing newline artifacts.
        - Strict validation pass before completion.

    Args:
        instance_ids: IDs list, prediction mapping {id: label}, or DataFrame.
        predictions: Labels list (required if instance_ids is a sequence of IDs).
        output_path: File destination path.
        delimiter: CSV separator (default: ',').
        validate: Run validate_codabench_submission on output file.
        expected_ids: Optional gold IDs to verify completeness.
    """
    out_file = Path(output_path)
    os.makedirs(out_file.parent.resolve(), exist_ok=True)

    pairs_to_write: List[Tuple[str, str]] = []

    if HAS_PANDAS and pd is not None and isinstance(instance_ids, pd.DataFrame):
        df = instance_ids
        id_col = "instance_id" if "instance_id" in df.columns else df.columns[0]
        label_col = "label" if "label" in df.columns else ("prediction" if "prediction" in df.columns else df.columns[1])
        for _, row in df.iterrows():
            pairs_to_write.append((str(row[id_col]).strip(), str(row[label_col]).strip()))

    elif hasattr(instance_ids, "columns") and hasattr(instance_ids, "iterrows"):
        # SimpleDataFrame or custom DataFrame-like object
        df = instance_ids
        id_col = "instance_id" if "instance_id" in df.columns else df.columns[0]
        label_col = "label" if "label" in df.columns else ("prediction" if "prediction" in df.columns else df.columns[1])
        for _, row in df.iterrows():
            pairs_to_write.append((str(row[id_col]).strip(), str(row[label_col]).strip()))

    elif isinstance(instance_ids, Mapping):
        for k, v in instance_ids.items():
            pairs_to_write.append((str(k).strip(), str(v).strip()))

    elif isinstance(instance_ids, Sequence) and predictions is not None:
        if len(instance_ids) != len(predictions):
            raise ValueError(
                f"Count mismatch between instance_ids ({len(instance_ids)}) and predictions ({len(predictions)})"
            )
        for i_id, pred in zip(instance_ids, predictions):
            pairs_to_write.append((str(i_id).strip(), str(pred).strip()))
    else:
        raise ValueError(
            "Invalid argument combination: supply either (ids, predictions), "
            "a mapping {id: label}, or a DataFrame."
        )

    if not pairs_to_write:
        raise ValueError("Cannot export empty predictions submission.")

    # Write strictly with LF line endings, exactly matching no trailing blank rows
    with open(out_file, "w", encoding="utf-8", newline="\n") as f:
        f.write("instance_id,label\n")
        total_items = len(pairs_to_write)
        for idx, (inst_id, label) in enumerate(pairs_to_write):
            if not inst_id or not inst_id.strip():
                raise ValueError(f"Empty instance_id at row {idx}")
            if label not in VALID_LABELS:
                raise ValueError(f"Invalid label '{label}' at row {idx}")
            newline_char = "\n" if idx < total_items - 1 else ""
            f.write(f"{inst_id.strip()}{delimiter}{label}{newline_char}")

    if validate:
        validate_codabench_submission(
            str(out_file),
            expected_ids=expected_ids,
            raise_on_error=True,
        )
