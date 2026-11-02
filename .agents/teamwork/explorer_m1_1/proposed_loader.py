"""
SemEval-2027 Task 2 (DiCo-NLI) - Data Loading & Ingestion Module.

This module provides data ingestion, schema parsing, validation, syntactic chunk
category extraction, split leakage verification, and fallback mock dataset generation
for Track 1 (English) and Track 2 (Spanish) of the PhrasIS benchmark.
"""

from __future__ import annotations

import csv
import io
import os
import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

# Conditional pandas import with graceful pure-Python fallback
try:
    import pandas as pd  # type: ignore
    HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore
    HAS_PANDAS = False


# ==============================================================================
# 1. Enums and Constants
# ==============================================================================

class DiCoLabel(str, Enum):
    """Official 4-class classification labels for DiCo-NLI."""
    EQUIVALENCE = "EQUIVALENCE"
    FORWARD_ENTAILMENT = "FORWARD_ENTAILMENT"
    BACKWARD_ENTAILMENT = "BACKWARD_ENTAILMENT"
    NEGATIVE_OTHER = "NEGATIVE_OTHER"


ALL_LABELS: Set[str] = {
    DiCoLabel.EQUIVALENCE.value,
    DiCoLabel.FORWARD_ENTAILMENT.value,
    DiCoLabel.BACKWARD_ENTAILMENT.value,
    DiCoLabel.NEGATIVE_OTHER.value,
}

REVERSIBLE_LABELS: Set[str] = {
    DiCoLabel.EQUIVALENCE.value,
    DiCoLabel.FORWARD_ENTAILMENT.value,
    DiCoLabel.BACKWARD_ENTAILMENT.value,
}


class SyntacticCategory(str, Enum):
    """Five core syntactic categories evaluated in PhrasIS."""
    NOMINAL = "nominal"
    VERBAL = "verbal"
    PREPOSITIONAL = "prepositional"
    ADVERBIAL = "adverbial"
    QUANTIFICATIONAL = "quantificational"
    UNKNOWN = "unknown"


ALL_CATEGORIES: Set[str] = {
    SyntacticCategory.NOMINAL.value,
    SyntacticCategory.VERBAL.value,
    SyntacticCategory.PREPOSITIONAL.value,
    SyntacticCategory.ADVERBIAL.value,
    SyntacticCategory.QUANTIFICATIONAL.value,
}


# ==============================================================================
# 2. Exceptions
# ==============================================================================

class DiCoDataError(Exception):
    """Base exception for DiCo-NLI data processing errors."""
    pass


class DatasetNotFoundError(DiCoDataError):
    """Raised when an expected dataset file cannot be located."""
    pass


class SchemaValidationError(DiCoDataError):
    """Raised when a CSV file does not adhere to the expected DiCo-NLI schema."""
    pass


class DataLeakageError(DiCoDataError):
    """Raised when source pair IDs or phrase pairs overlap across dataset splits."""
    pass


# ==============================================================================
# 3. Data Structures
# ==============================================================================

@dataclass(frozen=True)
class InstanceRecord:
    """
    Strongly typed representation of a single DiCo-NLI evaluation or training instance.
    """
    instance_id: str
    pair_id: str
    text1_lang: str
    text2_lang: str
    text1: str
    text2: str
    label: Optional[str] = None
    category: Optional[str] = None
    reverse_pair_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a standard dictionary."""
        return {
            "instance_id": self.instance_id,
            "pair_id": self.pair_id,
            "text1_lang": self.text1_lang,
            "text2_lang": self.text2_lang,
            "text1": self.text1,
            "text2": self.text2,
            "label": self.label,
            "category": self.category,
            "reverse_pair_id": self.reverse_pair_id,
        }


# ==============================================================================
# 4. Pure-Python SimpleDataFrame Fallback (Active when pandas is unavailable)
# ==============================================================================

class SimpleSeries:
    """Minimal series implementation for column indexing in SimpleDataFrame."""
    def __init__(self, data: List[Any], name: str = ""):
        self._data = list(data)
        self.name = name

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._data)

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            return SimpleSeries(self._data[index], name=self.name)
        return self._data[index]

    def tolist(self) -> List[Any]:
        return list(self._data)

    def value_counts(self) -> Dict[Any, int]:
        counts: Dict[Any, int] = {}
        for item in self._data:
            counts[item] = counts.get(item, 0) + 1
        return counts

    def unique(self) -> List[Any]:
        seen = set()
        out = []
        for item in self._data:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    def isin(self, values: Any) -> List[bool]:
        val_set = set(values)
        return [item in val_set for item in self._data]

    def __eq__(self, other: Any) -> List[bool]:  # type: ignore[override]
        return [item == other for item in self._data]

    def __repr__(self) -> str:
        return f"SimpleSeries({self._data[:5]}..., len={len(self._data)})"


class SimpleDataFrame:
    """
    Lightweight, drop-in fallback for pandas.DataFrame.
    Used when pandas is not installed (e.g. offline CPU environments).
    """
    def __init__(self, data: Optional[Union[List[Dict[str, Any]], Dict[str, List[Any]]]] = None):
        if data is None:
            self._records: List[Dict[str, Any]] = []
            self._columns: List[str] = []
        elif isinstance(data, list):
            self._records = [dict(r) for r in data]
            self._columns = list(self._records[0].keys()) if self._records else []
        elif isinstance(data, dict):
            # Dict of columns
            self._columns = list(data.keys())
            num_rows = len(next(iter(data.values()))) if data else 0
            self._records = [
                {col: data[col][i] for col in self._columns}
                for i in range(num_rows)
            ]
        else:
            raise TypeError("Unsupported data format for SimpleDataFrame")

    @classmethod
    def from_records(cls, records: List[Dict[str, Any]]) -> SimpleDataFrame:
        return cls(records)

    @property
    def columns(self) -> List[str]:
        return list(self._columns)

    @property
    def shape(self) -> Tuple[int, int]:
        return len(self._records), len(self._columns)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key not in self._columns:
                raise KeyError(f"Column '{key}' not found in DataFrame.")
            return SimpleSeries([r.get(key) for r in self._records], name=key)
        elif isinstance(key, list):
            # Check if boolean mask
            if key and isinstance(key[0], bool):
                filtered = [r for r, mask in zip(self._records, key) if mask]
                return SimpleDataFrame(filtered)
            # Column projection
            sub_records = [{col: r.get(col) for col in key} for r in self._records]
            return SimpleDataFrame(sub_records)
        elif isinstance(key, slice):
            return SimpleDataFrame(self._records[key])
        raise KeyError(f"Invalid indexing key: {key}")

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in self._columns:
            self._columns.append(key)
        if isinstance(value, (list, SimpleSeries)):
            if len(value) != len(self._records):
                raise ValueError("Length of values does not match DataFrame rows")
            for r, v in zip(self._records, value):
                r[key] = v
        else:
            for r in self._records:
                r[key] = value

    def to_dict(self, orient: str = "records") -> Union[List[Dict[str, Any]], Dict[str, List[Any]]]:
        if orient == "records":
            return [dict(r) for r in self._records]
        elif orient == "list":
            return {col: [r.get(col) for r in self._records] for col in self._columns}
        raise NotImplementedError(f"orient='{orient}' not implemented in SimpleDataFrame")

    def copy(self) -> SimpleDataFrame:
        return SimpleDataFrame([dict(r) for r in self._records])

    def iterrows(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        for idx, row in enumerate(self._records):
            yield idx, row

    def apply(self, func: Callable[[Dict[str, Any]], Any], axis: int = 1) -> SimpleSeries:
        if axis != 1:
            raise NotImplementedError("Only axis=1 (row-wise) supported in SimpleDataFrame.apply")
        results = [func(row) for row in self._records]
        return SimpleSeries(results)

    def to_csv(self, path_or_buf: Union[str, Path, io.StringIO], index: bool = False) -> Optional[str]:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=self._columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(self._records)
        content = output.getvalue()
        if isinstance(path_or_buf, (str, Path)):
            with open(path_or_buf, "w", encoding="utf-8") as f:
                f.write(content)
            return None
        elif hasattr(path_or_buf, "write"):
            path_or_buf.write(content)
            return None
        return content

    def head(self, n: int = 5) -> SimpleDataFrame:
        return SimpleDataFrame(self._records[:n])

    def __repr__(self) -> str:
        rows_preview = "\n".join(str(r) for r in self._records[:3])
        return f"SimpleDataFrame(rows={len(self._records)}, cols={self._columns}):\n{rows_preview}"


# ==============================================================================
# 5. Syntactic Category Classifier
# ==============================================================================

# Linguistic lexicons for English (Track 1)
_EN_PREPOSITIONS = {
    "in", "on", "at", "with", "by", "from", "to", "into", "onto", "under", "over",
    "through", "between", "among", "behind", "near", "about", "against", "without",
    "during", "before", "after", "toward", "towards", "down", "up", "off", "across",
    "around", "out", "along", "past", "throughout", "beneath", "beside", "upon",
}

_EN_QUANTIFIERS = {
    "all", "every", "each", "both", "either", "neither", "some", "any", "no", "none",
    "many", "few", "several", "much", "little", "everyone", "everybody", "someone",
    "somebody", "anyone", "anybody", "nobody", "no one", "one", "two", "three",
    "four", "five", "six", "seven", "eight", "nine", "ten", "zero",
}

_EN_AUXILIARIES = {
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "do", "does", "did", "will", "would", "can", "could", "should", "may", "might", "must",
}

_EN_DEGREE_ADVERBS = {
    "very", "extremely", "quite", "almost", "nearly", "too", "completely",
    "partially", "barely", "hardly", "rapidly", "carefully", "slowly", "fast",
    "well", "badly", "together", "apart", "downhill", "uphill", "soon", "late",
}

# Linguistic lexicons for Spanish (Track 2)
_ES_PREPOSITIONS = {
    "a", "al", "ante", "bajo", "con", "contra", "de", "del", "desde", "durante",
    "en", "entre", "hacia", "hasta", "mediante", "para", "por", "según", "sin",
    "so", "sobre", "tras", "versus", "vía",
}

_ES_QUANTIFIERS = {
    "todos", "todas", "todo", "toda", "cada", "ambos", "ambas", "algunos", "algunas",
    "algún", "alguna", "alguno", "ningún", "ninguna", "ninguno", "pocos", "pocas",
    "poco", "poca", "muchos", "muchas", "mucho", "mucha", "varios", "varias",
    "alguien", "nadie", "nada", "uno", "una", "dos", "tres", "cuatro", "cinco",
    "seis", "siete", "ocho", "nueve", "diez", "cero",
}

_ES_AUXILIARIES = {
    "es", "son", "era", "eran", "fue", "fueron", "sea", "sean", "ser", "sido",
    "está", "están", "estaba", "estaban", "estuvo", "estuvieron", "estar", "estado",
    "ha", "han", "había", "habían", "hubo", "haber",
}

_ES_DEGREE_ADVERBS = {
    "muy", "tan", "bastante", "casi", "extremadamente", "completamente",
    "rápidamente", "lentamente", "cuidadosamente", "bien", "mal", "apenas",
    "siempre", "nunca", "jamás", "quizás", "tal vez",
}


def _tokenize_lower(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r"\b\w+\b", text.lower())


def extract_syntactic_category(text1: str, text2: str, lang: str = "en") -> str:
    """
    Deterministically extracts the syntactic category of a phrase pair (A, B).

    The PhrasIS corpus evaluates 5 syntactic categories:
    - quantificational: numerical, universal, existential modifiers.
    - prepositional: spatial, locative, instrumental adjuncts starting with prepositions.
    - verbal: verb chains, aspectual auxiliaries, gerunds, infinitive clauses.
    - adverbial: manner, degree, or clausal adverbial modifiers.
    - nominal: noun phrases, named entities, appositions, compound nouns (default).

    Args:
        text1: First phrase (A).
        text2: Second phrase (B).
        lang: Language code ('en' or 'es').

    Returns:
        One of 'quantificational', 'prepositional', 'verbal', 'adverbial', 'nominal'.
    """
    lang_clean = lang.lower().strip()
    is_spanish = lang_clean.startswith("es")

    tokens1 = _tokenize_lower(text1)
    tokens2 = _tokenize_lower(text2)

    if not tokens1 and not tokens2:
        return SyntacticCategory.NOMINAL.value

    first_word_1 = tokens1[0] if tokens1 else ""
    first_word_2 = tokens2[0] if tokens2 else ""
    all_words = set(tokens1 + tokens2)

    # 1. Check Quantificational (Numbers, numerals, generalized quantifiers)
    if any(re.search(r"\d", w) for w in all_words):
        return SyntacticCategory.QUANTIFICATIONAL.value

    quantifier_lexicon = _ES_QUANTIFIERS if is_spanish else _EN_QUANTIFIERS
    if first_word_1 in quantifier_lexicon or first_word_2 in quantifier_lexicon:
        return SyntacticCategory.QUANTIFICATIONAL.value
    if len(all_words.intersection(quantifier_lexicon)) >= 2:
        return SyntacticCategory.QUANTIFICATIONAL.value

    # 2. Check Verbal Chains (VP)
    # Verbal chains include: gerunds (-ing, -ando, -iendo), infinitives (to + V, -ar/-er/-ir),
    # copula/auxiliary constructions, and preposition + gerund/infinitive (e.g., 'in destroying', 'al destruir').
    aux_lexicon = _ES_AUXILIARIES if is_spanish else _EN_AUXILIARIES
    if first_word_1 in aux_lexicon or first_word_2 in aux_lexicon:
        return SyntacticCategory.VERBAL.value

    if is_spanish:
        # Spanish gerunds (-ando, -iendo, -yendo)
        if any(w.endswith(("ando", "iendo", "yendo")) for w in all_words):
            return SyntacticCategory.VERBAL.value
        # Spanish 'al + infinitivo' or standalone infinitive at start
        if first_word_1 == "al" and len(tokens1) > 1 and tokens1[1].endswith(("ar", "er", "ir")):
            return SyntacticCategory.VERBAL.value
        if any(w.endswith(("ar", "er", "ir")) for w in [first_word_1, first_word_2]):
            return SyntacticCategory.VERBAL.value
    else:
        # English gerunds (-ing) e.g. "destroying", "in destroying", "are playing"
        if any(w.endswith("ing") for w in all_words):
            return SyntacticCategory.VERBAL.value
        # English infinitive "to <verb>"
        if first_word_1 == "to" or first_word_2 == "to":
            return SyntacticCategory.VERBAL.value

    # 3. Check Prepositional Phrases (PP)
    prep_lexicon = _ES_PREPOSITIONS if is_spanish else _EN_PREPOSITIONS
    if first_word_1 in prep_lexicon or first_word_2 in prep_lexicon:
        return SyntacticCategory.PREPOSITIONAL.value

    # 4. Check Adverbial Expressions (ADVP)
    adv_lexicon = _ES_DEGREE_ADVERBS if is_spanish else _EN_DEGREE_ADVERBS
    if is_spanish:
        if any(w.endswith("mente") for w in all_words) or first_word_1 in adv_lexicon or first_word_2 in adv_lexicon:
            return SyntacticCategory.ADVERBIAL.value
        if first_word_1 == "como" or first_word_2 == "como":
            return SyntacticCategory.ADVERBIAL.value
    else:
        if any(w.endswith("ly") for w in all_words) or first_word_1 in adv_lexicon or first_word_2 in adv_lexicon:
            return SyntacticCategory.ADVERBIAL.value
        if first_word_1 == "as" or first_word_2 == "as":
            return SyntacticCategory.ADVERBIAL.value

    # 5. Default to Nominal Phrases (NP)
    return SyntacticCategory.NOMINAL.value


# ==============================================================================
# 6. File Resolution and Normalization
# ==============================================================================

def normalize_track(track: Union[str, int]) -> str:
    """
    Normalizes diverse track representations to canonical 'track1' or 'track2'.

    Supports: 1, '1', 'track1', 'en', 'english', 'track1_en' -> 'track1'
              2, '2', 'track2', 'es', 'spanish', 'track2_es' -> 'track2'
    """
    clean = str(track).strip().lower()
    if clean in {"1", "track1", "en", "english", "track1_en", "t1"}:
        return "track1"
    elif clean in {"2", "track2", "es", "spanish", "track2_es", "t2"}:
        return "track2"
    raise ValueError(f"Unsupported track identifier '{track}'. Must be track 1 (English) or 2 (Spanish).")


def normalize_split(split: str) -> str:
    """
    Normalizes split aliases to 'train', 'dev', or 'test'.
    """
    clean = split.strip().lower()
    if clean in {"train", "training", "tr"}:
        return "train"
    elif clean in {"dev", "development", "val", "validation", "valid"}:
        return "dev"
    elif clean in {"test", "testing", "eval", "evaluation"}:
        return "test"
    raise ValueError(f"Unsupported split identifier '{split}'. Must be 'train', 'dev', or 'test'.")


def find_phrasis_file(
    track: Union[str, int],
    split: str,
    data_dir: Union[str, Path] = "data",
) -> Optional[Path]:
    """
    Searches standard project and benchmark locations for a PhrasIS split CSV.

    Search precedence:
    1. Direct participant labeled CSV: dico_nli_<split>_<track>_participant_labeled.csv
    2. Direct reference CSV: dico_nli_<split>_<track>_reference.csv
    3. Nested subdirectories: data_dir/<split>/..., data_dir/final_data/<split>/..., data_dir/raw/...
    4. Generic aliases: <track>_<split>.csv, dico_nli_<track>_<split>.csv
    5. Glob fallback matching *<split>*<track>*.csv

    Returns:
        Path to file if found, None otherwise.
    """
    track_norm = normalize_track(track)
    split_norm = normalize_split(split)
    base_dir = Path(data_dir)

    if not base_dir.exists():
        return None

    candidate_patterns = [
        # Official benchmark filenames
        f"dico_nli_{split_norm}_{track_norm}_participant_labeled.csv",
        f"dico_nli_{split_norm}_{track_norm}_reference.csv",
        f"dico_nli_{split_norm}_{track_norm}_submission_template.csv",
        f"dico_nli_{split_norm}_{track_norm}_unlabeled.csv",
        # Shorthand aliases
        f"{track_norm}_{split_norm}.csv",
        f"{split_norm}_{track_norm}.csv",
        f"dico_nli_{split_norm}_{track_norm}.csv",
    ]

    subdirs = [
        base_dir,
        base_dir / split_norm,
        base_dir / "final_data" / split_norm,
        base_dir / "final_data",
        base_dir / "raw",
        base_dir / "processed",
    ]

    for sdir in subdirs:
        for fname in candidate_patterns:
            target = sdir / fname
            if target.is_file() and target.stat().st_size > 0:
                return target

    # Glob fallback
    for p in base_dir.rglob("*.csv"):
        name_lower = p.name.lower()
        if split_norm in name_lower and track_norm in name_lower:
            if p.stat().st_size > 0:
                return p

    return None


# ==============================================================================
# 7. CSV Schema Parsing and Validation
# ==============================================================================

def read_phrasis_csv(
    file_path: Union[str, Path],
    allow_missing_label: bool = False,
) -> List[InstanceRecord]:
    """
    Reads and validates a DiCo-NLI PhrasIS CSV file into a list of InstanceRecord objects.

    Enforces:
    - UTF-8 / UTF-8 BOM encoding.
    - Required columns: instance_id, pair_id, text1_lang, text2_lang, text1, text2.
    - Label column presence (unless allow_missing_label is True).
    - Label enum membership in ALL_LABELS.
    - Syntactic category extraction if missing from input file.
    - Non-empty, non-whitespace instance_id and pair_id.
    - Uniqueness of instance_id within the split.

    Args:
        file_path: Path to CSV file.
        allow_missing_label: Set True when loading unlabeled test split.

    Returns:
        List of validated InstanceRecord objects.

    Raises:
        SchemaValidationError: If columns are missing or validation constraints fail.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PhrasIS file not found at '{path}'.")

    records: List[InstanceRecord] = []
    seen_instance_ids: Set[str] = set()

    with open(path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SchemaValidationError(f"File '{path}' is empty or has no header row.")

        fieldnames = [col.strip() for col in reader.fieldnames]
        required_cols = {"instance_id", "pair_id", "text1_lang", "text2_lang", "text1", "text2"}
        missing_cols = required_cols - set(fieldnames)
        if missing_cols:
            raise SchemaValidationError(
                f"Missing required columns in '{path}': {sorted(missing_cols)}"
            )

        has_label_col = "label" in fieldnames
        if not has_label_col and not allow_missing_label:
            raise SchemaValidationError(f"Column 'label' is missing in labeled split file '{path}'.")

        has_rev_col = "reverse_pair_id" in fieldnames
        has_cat_col = "category" in fieldnames or "syntactic_category" in fieldnames
        cat_key = "category" if "category" in fieldnames else "syntactic_category"

        for row_idx, row in enumerate(reader, start=2):
            # Clean values
            instance_id = (row.get("instance_id") or "").strip()
            pair_id = (row.get("pair_id") or "").strip()
            text1_lang = (row.get("text1_lang") or "").strip()
            text2_lang = (row.get("text2_lang") or "").strip()
            text1 = (row.get("text1") or "").strip()
            text2 = (row.get("text2") or "").strip()

            if not instance_id:
                raise SchemaValidationError(f"Empty instance_id at row {row_idx} in '{path}'.")
            if " " in instance_id:
                raise SchemaValidationError(f"Whitespace in instance_id '{instance_id}' at row {row_idx}.")
            if instance_id in seen_instance_ids:
                raise SchemaValidationError(f"Duplicate instance_id '{instance_id}' at row {row_idx}.")
            seen_instance_ids.add(instance_id)

            if not pair_id:
                raise SchemaValidationError(f"Empty pair_id at row {row_idx} in '{path}'.")
            if " " in pair_id:
                raise SchemaValidationError(f"Whitespace in pair_id '{pair_id}' at row {row_idx}.")

            if not text1 or not text2:
                raise SchemaValidationError(f"Empty text1 or text2 at row {row_idx} for instance '{instance_id}'.")

            # Parse label
            raw_label = (row.get("label") or "").strip() if has_label_col else ""
            label: Optional[str] = None
            if raw_label:
                if raw_label not in ALL_LABELS:
                    raise SchemaValidationError(
                        f"Invalid label '{raw_label}' at row {row_idx}; expected one of {sorted(ALL_LABELS)}."
                    )
                label = raw_label
            elif not allow_missing_label:
                raise SchemaValidationError(f"Missing label at row {row_idx} in labeled file '{path}'.")

            # Parse reverse_pair_id
            raw_rev = (row.get("reverse_pair_id") or "").strip() if has_rev_col else ""
            reverse_pair_id = raw_rev if raw_rev else None

            # Parse or extract category
            category: Optional[str] = None
            if has_cat_col:
                raw_cat = (row.get(cat_key) or "").strip().lower()
                if raw_cat in ALL_CATEGORIES:
                    category = raw_cat

            if not category:
                category = extract_syntactic_category(text1, text2, lang=text1_lang)

            record = InstanceRecord(
                instance_id=instance_id,
                pair_id=pair_id,
                text1_lang=text1_lang,
                text2_lang=text2_lang,
                text1=text1,
                text2=text2,
                label=label,
                category=category,
                reverse_pair_id=reverse_pair_id,
            )
            records.append(record)

    return records


# ==============================================================================
# 8. Synthetic Mock Data Generation
# ==============================================================================

_MOCK_PAIRS_TRACK1 = [
    # Round 1 (All 5 categories)
    ("Pope Francis", "pope", DiCoLabel.FORWARD_ENTAILMENT.value, SyntacticCategory.NOMINAL.value),
    ("is sprinting across the field", "is running", DiCoLabel.FORWARD_ENTAILMENT.value, SyntacticCategory.VERBAL.value),
    ("in the center of Madrid", "in Spain", DiCoLabel.FORWARD_ENTAILMENT.value, SyntacticCategory.PREPOSITIONAL.value),
    ("extremely rapidly", "very fast", DiCoLabel.EQUIVALENCE.value, SyntacticCategory.ADVERBIAL.value),
    ("nine energetic children", "several children", DiCoLabel.FORWARD_ENTAILMENT.value, SyntacticCategory.QUANTIFICATIONAL.value),

    # Round 2 (All 5 categories)
    ("the prime minister", "the head of government", DiCoLabel.EQUIVALENCE.value, SyntacticCategory.NOMINAL.value),
    ("to destroy", "to demolish", DiCoLabel.EQUIVALENCE.value, SyntacticCategory.VERBAL.value),
    ("with a tennis racket", "with sports equipment", DiCoLabel.FORWARD_ENTAILMENT.value, SyntacticCategory.PREPOSITIONAL.value),
    ("with great care", "carefully", DiCoLabel.EQUIVALENCE.value, SyntacticCategory.ADVERBIAL.value),
    ("all registered voters", "every voter", DiCoLabel.EQUIVALENCE.value, SyntacticCategory.QUANTIFICATIONAL.value),

    # Round 3 (All 5 categories)
    ("a young boy", "a child", DiCoLabel.FORWARD_ENTAILMENT.value, SyntacticCategory.NOMINAL.value),
    ("are dancing joyfully", "are moving", DiCoLabel.FORWARD_ENTAILMENT.value, SyntacticCategory.VERBAL.value),
    ("at the present moment", "right now", DiCoLabel.EQUIVALENCE.value, SyntacticCategory.PREPOSITIONAL.value),
    ("walking quite slowly", "walking slowly", DiCoLabel.FORWARD_ENTAILMENT.value, SyntacticCategory.ADVERBIAL.value),
    ("everyone attended", "someone attended", DiCoLabel.FORWARD_ENTAILMENT.value, SyntacticCategory.QUANTIFICATIONAL.value),

    # Round 4 (Negatives & edge cases across categories)
    ("a wild tiger", "a domestic feline", DiCoLabel.NEGATIVE_OTHER.value, SyntacticCategory.NOMINAL.value),
    ("was sleeping peacefully", "is singing loud", DiCoLabel.NEGATIVE_OTHER.value, SyntacticCategory.VERBAL.value),
    ("under the bridge", "above the blue sky", DiCoLabel.NEGATIVE_OTHER.value, SyntacticCategory.PREPOSITIONAL.value),
    ("completely silent", "deafeningly loud", DiCoLabel.NEGATIVE_OTHER.value, SyntacticCategory.ADVERBIAL.value),
    ("three golden retrievers", "two black cats", DiCoLabel.NEGATIVE_OTHER.value, SyntacticCategory.QUANTIFICATIONAL.value),
]

_MOCK_PAIRS_TRACK2 = [
    # Round 1 (All 5 categories)
    ("el papa Francisco", "el papa", DiCoLabel.FORWARD_ENTAILMENT.value, SyntacticCategory.NOMINAL.value),
    ("está corriendo a toda prisa", "se está desplazando", DiCoLabel.FORWARD_ENTAILMENT.value, SyntacticCategory.VERBAL.value),
    ("en el centro de Madrid", "en España", DiCoLabel.FORWARD_ENTAILMENT.value, SyntacticCategory.PREPOSITIONAL.value),
    ("extremadamente rápido", "muy velozmente", DiCoLabel.EQUIVALENCE.value, SyntacticCategory.ADVERBIAL.value),
    ("nueve niños entusiastas", "varios niños", DiCoLabel.FORWARD_ENTAILMENT.value, SyntacticCategory.QUANTIFICATIONAL.value),

    # Round 2 (All 5 categories)
    ("el presidente del gobierno", "el mandatario", DiCoLabel.EQUIVALENCE.value, SyntacticCategory.NOMINAL.value),
    ("destruir por completo", "demoler", DiCoLabel.EQUIVALENCE.value, SyntacticCategory.VERBAL.value),
    ("con una raqueta de tenis", "con equipo deportivo", DiCoLabel.FORWARD_ENTAILMENT.value, SyntacticCategory.PREPOSITIONAL.value),
    ("con sumo cuidado", "cuidadosamente", DiCoLabel.EQUIVALENCE.value, SyntacticCategory.ADVERBIAL.value),
    ("todos los votantes censados", "cada votante", DiCoLabel.EQUIVALENCE.value, SyntacticCategory.QUANTIFICATIONAL.value),

    # Round 3 (All 5 categories)
    ("un niño pequeño", "un menor de edad", DiCoLabel.FORWARD_ENTAILMENT.value, SyntacticCategory.NOMINAL.value),
    ("están bailando con alegría", "se están moviendo", DiCoLabel.FORWARD_ENTAILMENT.value, SyntacticCategory.VERBAL.value),
    ("en este preciso instante", "ahora mismo", DiCoLabel.EQUIVALENCE.value, SyntacticCategory.PREPOSITIONAL.value),
    ("caminando muy despacio", "caminando despacio", DiCoLabel.FORWARD_ENTAILMENT.value, SyntacticCategory.ADVERBIAL.value),
    ("todos asistieron", "alguien asistió", DiCoLabel.FORWARD_ENTAILMENT.value, SyntacticCategory.QUANTIFICATIONAL.value),

    # Round 4 (Negatives & edge cases across categories)
    ("un tigre salvaje", "un felino doméstico", DiCoLabel.NEGATIVE_OTHER.value, SyntacticCategory.NOMINAL.value),
    ("estaba durmiendo plácidamente", "está cantando fuerte", DiCoLabel.NEGATIVE_OTHER.value, SyntacticCategory.VERBAL.value),
    ("debajo del puente", "sobre el cielo despejado", DiCoLabel.NEGATIVE_OTHER.value, SyntacticCategory.PREPOSITIONAL.value),
    ("completamente en silencio", "ensordecedoramente ruidoso", DiCoLabel.NEGATIVE_OTHER.value, SyntacticCategory.ADVERBIAL.value),
    ("tres perros dorados", "dos gatos negros", DiCoLabel.NEGATIVE_OTHER.value, SyntacticCategory.QUANTIFICATIONAL.value),
]


def _invert_mock_label(label: str) -> str:
    """Helper to invert label for reverse mock pair generation."""
    if label == DiCoLabel.FORWARD_ENTAILMENT.value:
        return DiCoLabel.BACKWARD_ENTAILMENT.value
    elif label == DiCoLabel.BACKWARD_ENTAILMENT.value:
        return DiCoLabel.FORWARD_ENTAILMENT.value
    return label  # EQUIVALENCE and NEGATIVE_OTHER remain unchanged


def generate_mock_phrasis_data(
    num_pairs: int = 16,
    track: Union[str, int] = "track1",
    split: str = "dev",
) -> List[InstanceRecord]:
    """
    Generates synthetic, structurally valid DiCo-NLI PhrasIS instance records.

    Guarantees:
    - Balanced reciprocal pairs for reversible relations (EQUIV, FWD, BWD).
    - Unaligned instances for NEGATIVE_OTHER.
    - All 5 syntactic categories represented (nominal, verbal, prepositional, adverbial, quantificational).
    - Correct ISO-639 language tags ('en-en' or 'es-es').
    - Strict instance_id formatting: <pair_id>__<l1>-<l2>__<variant>.

    Args:
        num_pairs: Approximate number of source phrase pairs to generate.
        track: 'track1' (English) or 'track2' (Spanish).
        split: 'train', 'dev', or 'test'.

    Returns:
        List of InstanceRecord objects.
    """
    track_norm = normalize_track(track)
    split_norm = normalize_split(split)
    is_spanish = (track_norm == "track2")
    lang_code = "es" if is_spanish else "en"
    lang_pair = f"{lang_code}-{lang_code}"

    templates = _MOCK_PAIRS_TRACK2 if is_spanish else _MOCK_PAIRS_TRACK1
    selected_templates = (templates * ((num_pairs // len(templates)) + 1))[:num_pairs]

    split_prefix = 1000000 if split_norm == "train" else (2000000 if split_norm == "dev" else 3000000)
    records: List[InstanceRecord] = []

    for idx, (t1, t2, fwd_label, category) in enumerate(selected_templates, start=1):
        pair_num = split_prefix + idx
        pair_id = f"dico_phrasis_{pair_num:07d}"

        id_original = f"{pair_id}__{lang_pair}__original"
        id_flipped = f"{pair_id}__{lang_pair}__flipped"

        # Test split omits gold labels
        orig_label = None if split_norm == "test" else fwd_label

        if fwd_label == DiCoLabel.NEGATIVE_OTHER.value:
            # NEGATIVE_OTHER pairs are typically unaligned with empty reverse pointer
            rec = InstanceRecord(
                instance_id=id_original,
                pair_id=pair_id,
                text1_lang=lang_code,
                text2_lang=lang_code,
                text1=t1,
                text2=t2,
                label=orig_label,
                category=category,
                reverse_pair_id=None,
            )
            records.append(rec)
        else:
            # Reversible pair: generate both forward and flipped instances
            rev_label = None if split_norm == "test" else _invert_mock_label(fwd_label)

            rec_fwd = InstanceRecord(
                instance_id=id_original,
                pair_id=pair_id,
                text1_lang=lang_code,
                text2_lang=lang_code,
                text1=t1,
                text2=t2,
                label=orig_label,
                category=category,
                reverse_pair_id=id_flipped,
            )
            rec_rev = InstanceRecord(
                instance_id=id_flipped,
                pair_id=pair_id,
                text1_lang=lang_code,
                text2_lang=lang_code,
                text1=t2,
                text2=t1,
                label=rev_label,
                category=category,
                reverse_pair_id=id_original,
            )
            records.extend([rec_fwd, rec_rev])

    return records


# ==============================================================================
# 9. Main Public API: load_phrasis_split
# ==============================================================================

def load_phrasis_split(
    track: Union[str, int] = "track1",
    split: str = "train",
    data_dir: Union[str, Path] = "data",
    allow_mock_fallback: bool = True,
) -> Any:
    """
    Loads a PhrasIS dataset partition for Track 1 (English) or Track 2 (Spanish).

    Returns a pandas.DataFrame (or SimpleDataFrame fallback if pandas is not installed)
    containing columns:
        ['instance_id', 'pair_id', 'text1_lang', 'text2_lang', 'text1', 'text2', 'label', 'category']

    Args:
        track: Track identifier (1, 2, 'track1', 'track2', 'en', 'es').
        split: Split partition ('train', 'dev', 'test').
        data_dir: Base directory to search for dataset files (default: 'data').
        allow_mock_fallback: If True, generates synthetic mock data if raw CSV is not found.

    Returns:
        DataFrame containing parsed and validated PhrasIS records.

    Raises:
        DatasetNotFoundError: If file is not found and allow_mock_fallback is False.
    """
    track_norm = normalize_track(track)
    split_norm = normalize_split(split)

    csv_path = find_phrasis_file(track=track_norm, split=split_norm, data_dir=data_dir)

    if csv_path is not None:
        records = read_phrasis_csv(
            file_path=csv_path,
            allow_missing_label=(split_norm == "test"),
        )
    elif allow_mock_fallback:
        records = generate_mock_phrasis_data(
            num_pairs=32 if split_norm == "train" else 16,
            track=track_norm,
            split=split_norm,
        )
    else:
        raise DatasetNotFoundError(
            f"PhrasIS dataset file for track='{track_norm}', split='{split_norm}' not found in '{data_dir}'."
        )

    dict_records = [r.to_dict() for r in records]

    # Convert to pandas.DataFrame if available, else SimpleDataFrame
    if HAS_PANDAS and pd is not None:
        df = pd.DataFrame(dict_records)
    else:
        df = SimpleDataFrame.from_records(dict_records)

    return df


# ==============================================================================
# 10. Split Isolation & Leakage Verification
# ==============================================================================

def verify_split_leakage(train_data: Any, eval_data: Any) -> Tuple[bool, List[str]]:
    """
    Verifies that no source pair IDs or phrase pairs from training leak into evaluation.

    Args:
        train_data: DataFrame or list of InstanceRecord for training split.
        eval_data: DataFrame or list of InstanceRecord for dev/test split.

    Returns:
        Tuple (is_leak_free, list_of_leaked_pair_ids).
    """
    if isinstance(train_data, (SimpleDataFrame, getattr(pd, "DataFrame", SimpleDataFrame))):
        train_pairs = set(train_data["pair_id"])
        eval_pairs = set(eval_data["pair_id"])
    elif isinstance(train_data, list):
        train_pairs = {r.pair_id for r in train_data}
        eval_pairs = {r.pair_id for r in eval_data}
    else:
        raise TypeError("Unsupported data type for split leakage verification.")

    leaked = sorted(train_pairs.intersection(eval_pairs))
    return (len(leaked) == 0, leaked)


# ==============================================================================
# 11. Mock Dataset Disk Serializer
# ==============================================================================

def save_mock_dataset(
    output_dir: Union[str, Path] = "data/raw",
    num_train_pairs: int = 32,
    num_dev_pairs: int = 16,
) -> Dict[str, Path]:
    """
    Generates and saves standard PhrasIS CSVs for Track 1 and Track 2 to disk.
    Creates participant labeled CSVs and reference CSVs.

    Args:
        output_dir: Destination folder (e.g. 'data/raw').
        num_train_pairs: Number of training pairs per track.
        num_dev_pairs: Number of dev pairs per track.

    Returns:
        Dictionary mapping split/track keys to created Path objects.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created_paths: Dict[str, Path] = {}

    configs = [
        ("track1", "train", num_train_pairs),
        ("track1", "dev", num_dev_pairs),
        ("track2", "train", num_train_pairs),
        ("track2", "dev", num_dev_pairs),
    ]

    for track, split, num_pairs in configs:
        records = generate_mock_phrasis_data(num_pairs=num_pairs, track=track, split=split)

        # 1. Participant labeled CSV: instance_id,pair_id,text1_lang,text2_lang,text1,text2,label
        part_fname = f"dico_nli_{split}_{track}_participant_labeled.csv"
        part_path = out_dir / part_fname
        with open(part_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["instance_id", "pair_id", "text1_lang", "text2_lang", "text1", "text2", "label"])
            for r in records:
                writer.writerow([r.instance_id, r.pair_id, r.text1_lang, r.text2_lang, r.text1, r.text2, r.label])
        created_paths[f"{track}_{split}_participant"] = part_path

        # 2. Reference CSV: instance_id,pair_id,text1_lang,text2_lang,text1,text2,reverse_pair_id,label
        ref_fname = f"dico_nli_{split}_{track}_reference.csv"
        ref_path = out_dir / ref_fname
        with open(ref_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["instance_id", "pair_id", "text1_lang", "text2_lang", "text1", "text2", "reverse_pair_id", "label"])
            for r in records:
                writer.writerow([r.instance_id, r.pair_id, r.text1_lang, r.text2_lang, r.text1, r.text2, r.reverse_pair_id or "", r.label])
        created_paths[f"{track}_{split}_reference"] = ref_path

    return created_paths
