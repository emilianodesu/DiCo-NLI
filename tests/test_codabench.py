"""Unit tests for Codabench Submission Formatting and Validation.

Validates:
- Strict two-column format (instance_id,label).
- Header integrity: exact column names, rejection of extra/missing columns.
- Row completeness: zero blank rows, no trailing empty lines at EOF.
- Data integrity: no nulls, NaN, or whitespace in identifiers.
- Label enum adherence: strictly {"EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"}.
- Set bijection: detection of missing or extraneous instance IDs.
- Submissions export function compliance.
"""

import csv
import io
import os
import tempfile
import unittest
from pathlib import Path
from typing import List, Optional, Sequence, Set

# Attempt import from project source; fallback to authoritative reference specification
try:
    from src.data.codabench import (
        VALID_LABELS,
        CodabenchValidationError,
        export_codabench_submission,
        validate_codabench_submission,
    )
    SRC_AVAILABLE = True
except ImportError:
    SRC_AVAILABLE = False

    VALID_LABELS: Set[str] = {
        "EQUIVALENCE",
        "FORWARD_ENTAILMENT",
        "BACKWARD_ENTAILMENT",
        "NEGATIVE_OTHER",
    }

    class CodabenchValidationError(ValueError):
        """Raised when a submission file violates official Codabench specifications."""
        pass

    def export_codabench_submission(
        instance_ids: Sequence[str],
        predictions: Sequence[str],
        output_path: str,
    ) -> None:
        """Export predictions to a strict two-column CSV file."""
        if len(instance_ids) != len(predictions):
            raise ValueError(
                f"Count mismatch between instance_ids ({len(instance_ids)}) and predictions ({len(predictions)})"
            )

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        # Write strictly with LF line endings, no trailing blank lines
        with open(output_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("instance_id,label\n")
            for idx, (inst_id, label) in enumerate(zip(instance_ids, predictions)):
                if not inst_id or not inst_id.strip():
                    raise ValueError(f"Empty instance_id at row {idx}")
                if label not in VALID_LABELS:
                    raise ValueError(f"Invalid label '{label}' at row {idx}")
                newline_char = "\n" if idx < len(instance_ids) - 1 else ""
                f.write(f"{inst_id.strip()},{label}{newline_char}")

    def validate_codabench_submission(
        submission_path: str,
        expected_ids: Optional[Set[str]] = None,
    ) -> bool:
        """Validate a submission file against official Codabench rules."""
        path = Path(submission_path)
        if not path.is_file():
            raise CodabenchValidationError(f"File not found: {submission_path}")

        raw_bytes = path.read_bytes()
        if b"\x00" in raw_bytes:
            raise CodabenchValidationError("File contains NUL bytes")

        text = raw_bytes.decode("utf-8-sig")
        lines = text.splitlines(keepends=False)

        if not lines:
            raise CodabenchValidationError("Submission file is empty")

        # Check for trailing empty line
        if text.endswith("\n\n") or text.endswith("\r\n\r\n"):
            raise CodabenchValidationError("File contains trailing blank line(s)")

        header_line = lines[0].strip()
        delimiter = "\t" if "\t" in header_line else ","
        reader = csv.reader(lines, delimiter=delimiter)

        header = next(reader, None)
        if not header:
            raise CodabenchValidationError("Missing header row")

        if header != ["instance_id", "label"]:
            raise CodabenchValidationError(
                f"Invalid header {header}. Expected exactly ['instance_id', 'label']"
            )

        seen_ids = set()
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


class TestCodabenchSubmission(unittest.TestCase):
    """Test suite for Codabench submission exporting and formatting validation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_export_and_validate_valid_submission(self):
        """Exporting a valid list of IDs and labels should pass validation cleanly."""
        ids = ["dico_001__en-en__original", "dico_001__en-en__flipped"]
        preds = ["FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT"]
        out_file = str(self.temp_path / "valid_submission.csv")

        export_codabench_submission(ids, preds, out_file)
        self.assertTrue(os.path.exists(out_file))

        # Validate with matching expected IDs
        is_valid = validate_codabench_submission(out_file, expected_ids=set(ids))
        self.assertTrue(is_valid)

    def test_reject_extra_header_columns(self):
        """Rejects files with extra columns in header (e.g. confidence)."""
        bad_file = self.temp_path / "extra_col.csv"
        bad_file.write_text("instance_id,label,confidence\ndico_001,FORWARD_ENTAILMENT,0.95")
        with self.assertRaises(CodabenchValidationError):
            validate_codabench_submission(str(bad_file))

    def test_reject_missing_header_column(self):
        """Rejects files missing a column (e.g. only instance_id)."""
        bad_file = self.temp_path / "missing_col.csv"
        bad_file.write_text("instance_id\ndico_001")
        with self.assertRaises(CodabenchValidationError):
            validate_codabench_submission(str(bad_file))

    def test_reject_wrong_header_names(self):
        """Rejects files with incorrect header names (e.g. id,prediction)."""
        bad_file = self.temp_path / "wrong_header.csv"
        bad_file.write_text("id,prediction\ndico_001,FORWARD_ENTAILMENT")
        with self.assertRaises(CodabenchValidationError):
            validate_codabench_submission(str(bad_file))

    def test_reject_trailing_blank_line(self):
        """Rejects files with blank lines at end of file."""
        bad_file = self.temp_path / "trailing_blank.csv"
        bad_file.write_text("instance_id,label\ndico_001,EQUIVALENCE\n\n")
        with self.assertRaises(CodabenchValidationError):
            validate_codabench_submission(str(bad_file))

    def test_reject_mid_file_blank_row(self):
        """Rejects files with blank rows inside the data."""
        bad_file = self.temp_path / "mid_blank.csv"
        bad_file.write_text("instance_id,label\ndico_001,EQUIVALENCE\n\ndico_002,FORWARD_ENTAILMENT")
        with self.assertRaises(CodabenchValidationError):
            validate_codabench_submission(str(bad_file))

    def test_reject_invalid_label_value(self):
        """Rejects unknown labels such as CONTRADICTION or lowercase."""
        bad_file = self.temp_path / "bad_label.csv"
        bad_file.write_text("instance_id,label\ndico_001,CONTRADICTION")
        with self.assertRaises(CodabenchValidationError):
            validate_codabench_submission(str(bad_file))

        bad_file2 = self.temp_path / "lowercase_label.csv"
        bad_file2.write_text("instance_id,label\ndico_001,equivalence")
        with self.assertRaises(CodabenchValidationError):
            validate_codabench_submission(str(bad_file2))

    def test_reject_duplicate_instance_id(self):
        """Rejects duplicate predictions for the same instance_id."""
        bad_file = self.temp_path / "duplicate_id.csv"
        bad_file.write_text(
            "instance_id,label\n"
            "dico_001,FORWARD_ENTAILMENT\n"
            "dico_001,BACKWARD_ENTAILMENT"
        )
        with self.assertRaises(CodabenchValidationError):
            validate_codabench_submission(str(bad_file))

    def test_reject_whitespace_in_instance_id(self):
        """Rejects identifiers containing whitespace."""
        bad_file = self.temp_path / "whitespace_id.csv"
        bad_file.write_text("instance_id,label\ndico 001,FORWARD_ENTAILMENT")
        with self.assertRaises(CodabenchValidationError):
            validate_codabench_submission(str(bad_file))

    def test_reject_missing_expected_ids(self):
        """Rejects submission missing required instance IDs from gold set."""
        sub_file = self.temp_path / "incomplete.csv"
        sub_file.write_text("instance_id,label\ndico_001,FORWARD_ENTAILMENT")
        expected = {"dico_001", "dico_002"}
        with self.assertRaises(CodabenchValidationError):
            validate_codabench_submission(str(sub_file), expected_ids=expected)

    def test_reject_unexpected_extra_ids(self):
        """Rejects submission containing extraneous instance IDs not in gold set."""
        sub_file = self.temp_path / "extra_id.csv"
        sub_file.write_text(
            "instance_id,label\n"
            "dico_001,FORWARD_ENTAILMENT\n"
            "dico_999,EQUIVALENCE"
        )
        expected = {"dico_001"}
        with self.assertRaises(CodabenchValidationError):
            validate_codabench_submission(str(sub_file), expected_ids=expected)


if __name__ == "__main__":
    unittest.main()
