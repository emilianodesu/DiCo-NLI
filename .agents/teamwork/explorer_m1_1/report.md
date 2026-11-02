# Architectural Blueprint & Specification: Data Loading & Ingestion (`src/data/loader.py`)

**Milestone:** M1 — Data Loading & Ingestion  
**Agent:** Explorer M1.1 (`explorer_m1_1`)  
**Target Module:** `src/data/loader.py` (and `src/data/__init__.py`)  
**Target Tracks:** Track 1 (English Monolingual) & Track 2 (Spanish Monolingual)  
**Date:** 2026-11-01  
**Project:** SemEval-2027 Task 2 (DiCo-NLI) — Licenciatura en Ciencia de Datos (LCD), UNAM  

---

## 1. Executive Summary & Mission Alignment

In Natural Language Inference (NLI), standard sentence-level benchmarks (e.g. SNLI, MNLI) evaluate entailment over complete discourses where lexical overlap or narrative heuristics can mask underlying reasoning flaws. SemEval-2027 Task 2 (**DiCo-NLI**) evaluates **directional consistency** in fine-grained NLI over ordered phrase pairs $(A, B)$ derived from the **PhrasIS** benchmark (*Phrase Inference and Similarity benchmark*, Lopez-Gazpio et al., 2024).

The fundamental challenge in DiCo-NLI is the **Reversal Curse**: when a model is trained or queried on directed phrase pairs, it frequently assigns incompatible predictions when presented with the transposed inverse pair $(B, A)$ (for example, predicting `FORWARD_ENTAILMENT` for both $(A, B)$ and $(B, A)$).

This report establishes the complete architectural blueprint for `src/data/loader.py`, responsible for:
1. Ingesting PhrasIS dataset files for **Track 1 (English)** and **Track 2 (Spanish)** across **train**, **dev**, and **test** partitions.
2. Parsing and strictly validating 7-column participant labeled CSVs, 8-column gold reference CSVs, and unlabeled test split CSVs.
3. Deterministically extracting the **5 core syntactic categories** evaluated in PhrasIS:
   - **Nominal Phrases (NP)**
   - **Verbal Chains (VP)**
   - **Prepositional Phrases (PP)**
   - **Adverbial Expressions (ADVP)**
   - **Quantificational Expressions (QP)**
4. Providing a **self-contained synthetic mock data generator** enabling offline, zero-network, ultra-fast ($<1$s) execution on CPU environments without pre-downloaded dataset files.
5. Providing a **pure-Python DataFrame fallback (`SimpleDataFrame`)** ensuring 100% test compatibility even when heavy external packages (e.g. `pandas`) are not installed.
6. Enforcing **source-pair split isolation** to prevent train-dev/test leakage.

---

## 2. PhrasIS Corpus Specifications & Partition Layout

### 2.1 Partition Counts and Balance Properties
The official PhrasIS corpus release partitions for Track 1 and Track 2 are structured with exact reciprocal balance:

| Partition | Track 1 (English) | Track 2 (Spanish) | Reversible Pairs ($|\mathcal{P}|$) | Negative Noise Pairs | Total Instances |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 3,042 | 3,042 | 1,282 ($2,564$ inst.) | 478 ($478$ inst.) | **3,042** |
| **Dev** | 660 | 660 | 277 ($554$ inst.) | 106 ($106$ inst.) | **660** |
| **Test** | Unlabeled | Unlabeled | TBD | TBD | Evaluation Set |

#### Label Distribution Breakdown (Train & Dev):
- `FORWARD_ENTAILMENT`: 881 (28.96% Train) | 190 (28.79% Dev)
- `BACKWARD_ENTAILMENT`: 881 (28.96% Train) | 190 (28.79% Dev)
- `EQUIVALENCE`: 802 (26.36% Train) | 174 (26.36% Dev)
- `NEGATIVE_OTHER`: 478 (15.71% Train) | 106 (16.06% Dev)

*Linguistic & Structural Property:* `FORWARD_ENTAILMENT` and `BACKWARD_ENTAILMENT` have **identical sample frequencies** because for every directed source pair $(A, B)$ labeled `FORWARD_ENTAILMENT`, PhrasIS incorporates the transposed pair $(B, A)$ labeled `BACKWARD_ENTAILMENT`.

### 2.2 Canonical Identifier Architecture
Identifiers in PhrasIS strictly follow structured semantic patterns:
1. **Source Pair ID (`pair_id`):** `dico_phrasis_<07d>` (e.g. `dico_phrasis_0000002`). This canonical identifier indexes the underlying semantic phrase pair regardless of presentation order or language.
2. **Evaluation Instance ID (`instance_id`):** `<pair_id>__<text1_lang>-<text2_lang>__<variant>`
   - `text1_lang-text2_lang`: `en-en` for Track 1; `es-es` for Track 2.
   - `variant`:
     - `original`: Canonical forward presentation $(A, B)$.
     - `flipped`: Reciprocal inverted presentation $(B, A)$.
3. **Reciprocal Pointer (`reverse_pair_id`):**
   - For reversible instances (`EQUIVALENCE`, `FORWARD_ENTAILMENT`, `BACKWARD_ENTAILMENT`), points to the corresponding reverse `instance_id`.
   - For `NEGATIVE_OTHER`, points to `None` / empty string `""` (non-reversible noise).

### 2.3 File Discovery & Precedence Hierarchy
To support flexible local development, GPU training clusters, and CI runners, `find_phrasis_file` searches candidate paths in the following priority order:

```
[Search Root: data_dir (default: 'data')]
  ├── 1. dico_nli_{split}_{track}_participant_labeled.csv (in data_dir/, data_dir/{split}/, or data_dir/final_data/{split}/)
  ├── 2. dico_nli_{split}_{track}_reference.csv (gold reference fallback)
  ├── 3. dico_nli_{split}_{track}_unlabeled.csv (test split)
  ├── 4. dico_nli_{split}_{track}_submission_template.csv
  ├── 5. raw/dico_nli_{split}_{track}_participant_labeled.csv
  ├── 6. {track}_{split}.csv (shorthand alias)
  └── 7. Recursive glob match: *{split}*{track}*.csv
```

---

## 3. Schema Parsing, Ingestion & Validation Engine

### 3.1 Input CSV Schemas

#### A. Participant Labeled CSV (7 columns)
Used for model training and local development:
```csv
instance_id,pair_id,text1_lang,text2_lang,text1,text2,label
dico_phrasis_0000002__en-en__flipped,dico_phrasis_0000002,en,en,'s war shrine visit,'s shrine visit,FORWARD_ENTAILMENT
dico_phrasis_0000002__en-en__original,dico_phrasis_0000002,en,en,'s shrine visit,'s war shrine visit,BACKWARD_ENTAILMENT
```

#### B. Gold Reference CSV (8 columns)
Contains explicit reciprocal pointers used for competition scoring:
```csv
instance_id,pair_id,text1_lang,text2_lang,text1,text2,reverse_pair_id,label
dico_phrasis_0000002__en-en__flipped,dico_phrasis_0000002,en,en,'s war shrine visit,'s shrine visit,dico_phrasis_0000002__en-en__original,FORWARD_ENTAILMENT
dico_phrasis_0001949__en-en__original,dico_phrasis_0001949,en,en,Syria opposition unity,Syria divided opposition,,NEGATIVE_OTHER
```

#### C. Unlabeled Test Split CSV (6 columns)
```csv
instance_id,pair_id,text1_lang,text2_lang,text1,text2
dico_phrasis_0003001__en-en__original,dico_phrasis_0003001,en,en,a brown dog,an animal
```

### 3.2 Strict Validation Checklist
`read_phrasis_csv` enforces comprehensive data integrity:
1. **Encoding:** Automatically strips UTF-8 Byte Order Marks (`utf-8-sig`) and standard UTF-8.
2. **Column Completeness:** Ensures all required columns (`instance_id`, `pair_id`, `text1_lang`, `text2_lang`, `text1`, `text2`) are present.
3. **No Whitespace in Identifiers:** Detects and rejects any leading, trailing, or embedded spaces in `instance_id` or `pair_id`.
4. **Duplicate Prevention:** Tracks seen `instance_id` values and raises `SchemaValidationError` on any collision.
5. **Categorical Adherence:** Validates that labels strictly belong to `{"EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"}`.
6. **Graceful Test Split Handling:** Sets `label = None` without raising errors when `allow_missing_label=True` or when loading test splits.

---

## 4. Syntactic Category Extraction Engine

### 4.1 Linguistic Motivation
PhrasIS decomposes sentences from Flickr30k and news headlines into aligned semantic chunks. An essential research requirement (ORIGINAL_REQUEST §R4.3) is **syntactic breakdown error analysis**, evaluating whether DeBERTa and BETO models exhibit differential vulnerability to the Reversal Curse depending on grammatical structure.

The five evaluated categories represent distinct semantic composition mechanisms:
1. **Nominal Phrases (NP / Sintagmas Nominales):** Hypernymy/hyponymy, entity modification, noun compounds (`"Pope Francis"` vs `"pope"`).
2. **Verbal Chains (VP / Cadenas Verbales):** Aspectual verbs, telicity, gerunds, progressive chains (`"is sprinting across the field"` vs `"is running"`). Predicates have high directional asymmetry.
3. **Prepositional Phrases (PP / Sintagmas Preposicionales):** Spatial, temporal, instrumental inclusion hierarchies (`"in the center of Madrid"` vs `"in Spain"`).
4. **Adverbial Expressions (ADVP / Expresiones Adverbiales):** Manner, degree, trajectory modifiers (`"extremely rapidly"` vs `"very fast"`).
5. **Quantificational Expressions (QP / Expresiones Cuantificacionales):** Aristotelian quantifiers, numerals, cardinality (`"nine children"` vs `"several children"`).

### 4.2 Deterministic Multi-Lingual Extraction Algorithm
Because raw participant CSVs do not always contain an explicit `category` column, `loader.py` implements a zero-dependency, deterministic heuristic parser for English (`en`) and Spanish (`es`):

```python
def extract_syntactic_category(text1: str, text2: str, lang: str = "en") -> str:
    ...
```

#### Priority Rule Hierarchy:
```
┌─────────────────────────────────────────────────────────────┐
│ 1. Quantificational Check (QP)                              │
│    - Contains digits (\d+) OR initial quantifier word       │
│    - EN: all, every, each, some, no, nine, three, everyone  │
│    - ES: todos, cada, algunos, ningún, nueve, tres, alguien │
└──────────────────────────────┬──────────────────────────────┘
                               │ (if false)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Verbal Chain Check (VP)                                  │
│    - Copula/Auxiliary: is, are, was, está, están, ha        │
│    - Gerunds: -ing (EN), -ando, -iendo, -yendo (ES)         │
│    - Infinitives: "to <verb>", -ar, -er, -ir (ES)           │
│    - Gerundival adjuncts: "in destroying", "al destruir"    │
└──────────────────────────────┬──────────────────────────────┘
                               │ (if false)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Prepositional Phrase Check (PP)                          │
│    - Initial preposition token                              │
│    - EN: in, on, at, with, through, under, behind, near     │
│    - ES: en, con, de, del, por, para, hacia, sobre, entre   │
└──────────────────────────────┬──────────────────────────────┘
                               │ (if false)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Adverbial Expression Check (ADVP)                        │
│    - Manner suffix: -ly (EN), -mente (ES)                   │
│    - Degree words: extremely, very, quite, muy, casi        │
│    - Comparative/role adjuncts: "as CEO", "como presidente" │
└──────────────────────────────┬──────────────────────────────┘
                               │ (if false)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Default: Nominal Phrase (NP)                             │
│    - Entity phrases, nouns, determiners + noun compounds    │
└─────────────────────────────────────────────────────────────┘
```

#### Crucial Linguistic Disambiguation:
- **`"in destroying"` vs `"in Spain"`:** A phrase starting with `"in"` followed by a gerund (`-ing`) is a **verbal chain**, whereas `"in"` followed by a noun is a **prepositional phrase**. The rule engine checks verbal morphology before prepositional categorization.
- **`"to destroy"` vs `"to the store"`:** `"to"` paired with a verb is categorized as **verbal**.
- **`"al destruir"` in Spanish:** `"al"` followed by an infinitive (`-ar/-er/-ir`) is classified as **verbal**.

---

## 5. Fallback & Synthetic Mock Data Generation

### 5.1 Sandbox & Offline Execution Constraints
During sandboxed execution or fast CPU verification (`tests/test_pipeline.py`), external network access is blocked, and downloading full HuggingFace datasets or gigabyte-scale embeddings is prohibited. 

`generate_mock_phrasis_data` generates synthetic, semantically realistic PhrasIS instances on demand:
- **Balanced Label Distribution:** Ensures exact equality between `FORWARD_ENTAILMENT` and `BACKWARD_ENTAILMENT`.
- **Reciprocal Inversion:** Every forward pair $(A, B)$ has a paired $(B, A)$ instance with inverted labels.
- **Syntactic Diversity:** Round-robin interleaving ensures that even small batches ($N=5$) contain instances from all 5 categories.
- **Bilingual Coverage:** Supports Track 1 (English) and Track 2 (Spanish) with native phrasing.
- **Unlabeled Test Mode:** When `split="test"`, sets `label = None` while preserving all text and identifier structures.

### 5.2 Disk Serializer (`save_mock_dataset`)
`save_mock_dataset(output_dir="data/raw")` writes valid CSV files conforming to official Codabench standards, allowing offline integration tests to populate `data/` instantly:
- `dico_nli_train_track1_participant_labeled.csv`
- `dico_nli_dev_track1_reference.csv`
- `dico_nli_train_track2_participant_labeled.csv`
- `dico_nli_dev_track2_reference.csv`

---

## 6. Split Isolation & Leakage Auditor

### 6.1 The Data Leakage Threat in Directional NLI
If a source pair $(A, B)$ appears in the training set and its flipped counterpart $(B, A)$ appears in the dev or test set, models with memorization capacity will exhibit artificially inflated `SoftCons` and `HardCons` scores.

### 6.2 The Auditor (`verify_split_leakage`)
`verify_split_leakage(train_data, eval_data)` performs set intersection over canonical `pair_id`s:
$$\mathcal{I}_{\text{leak}} = \text{pair\_ids}(\text{Train}) \cap \text{pair\_ids}(\text{Eval})$$
- Returns `(True, [])` if $\mathcal{I}_{\text{leak}} = \emptyset$.
- Returns `(False, leaked_ids)` if any leakage is detected, logging the offending identifiers for immediate rectification.

---

## 7. Data Structures & Pure-Python Fallback (`SimpleDataFrame`)

### 7.1 Data Structures & Enums
```python
@dataclass(frozen=True)
class InstanceRecord:
    instance_id: str
    pair_id: str
    text1_lang: str
    text2_lang: str
    text1: str
    text2: str
    label: Optional[str] = None
    category: Optional[str] = None
    reverse_pair_id: Optional[str] = None

class DiCoLabel(str, Enum):
    EQUIVALENCE = "EQUIVALENCE"
    FORWARD_ENTAILMENT = "FORWARD_ENTAILMENT"
    BACKWARD_ENTAILMENT = "BACKWARD_ENTAILMENT"
    NEGATIVE_OTHER = "NEGATIVE_OTHER"

class SyntacticCategory(str, Enum):
    NOMINAL = "nominal"
    VERBAL = "verbal"
    PREPOSITIONAL = "prepositional"
    ADVERBIAL = "adverbial"
    QUANTIFICATIONAL = "quantificational"
    UNKNOWN = "unknown"
```

### 7.2 Zero-Dependency `SimpleDataFrame`
When running in environments where `pandas` is not installed (e.g., standard Python sandboxes), `loader.py` provides `SimpleDataFrame`. It implements the essential pandas API:
- Slicing and column access: `df["instance_id"]`, `df[["text1", "text2"]]`
- Length and shape: `len(df)`, `df.shape`
- Boolean filtering: `df[df["label"] == "EQUIVALENCE"]`
- Column assignment: `df["category"] = categories`
- Aggregations: `df["label"].value_counts()`, `df["category"].unique()`
- Row iteration: `df.iterrows()`, `df.apply(fn, axis=1)`
- Exporting: `df.to_csv(path, index=False)`, `df.to_dict(orient="records")`

When `pandas` IS installed, `load_phrasis_split` seamlessly returns a native `pandas.DataFrame`.

---

## 8. Verification and Empirical Validation

The complete proposed implementation was thoroughly validated locally on macOS (`Darwin 27.2.0`, Python 3.14.3) using `.agents/teamwork/explorer_m1_1/test_proposed_loader.py`.

### Execution Summary:
```
=== RUNNING PROPOSED LOADER LOCAL TEST SUITE ===
[TEST] Normalization...
 -> PASS
[TEST] Syntactic Category Extraction...
 -> PASS
[TEST] Mock Data Generation...
 -> PASS
[TEST] Split Leakage Auditor...
 -> PASS
[TEST] File I/O and Schema Validation...
 -> PASS
[TEST] SimpleDataFrame Fallback Operations...
 -> PASS
=== ALL TESTS PASSED SUCCESSFULLY! ===
```

### Verified Scenarios:
1. **Multilingual Syntactic Category Accuracy:**
   - English: `"Pope Francis"` $\to$ `nominal`, `"is sprinting"` $\to$ `verbal`, `"in the center of Madrid"` $\to$ `prepositional`, `"extremely rapidly"` $\to$ `adverbial`, `"nine children"` $\to$ `quantificational`.
   - Spanish: `"el papa Francisco"` $\to$ `nominal`, `"está corriendo"` $\to$ `verbal`, `"en España"` $\to$ `prepositional`, `"extremadamente rápido"` $\to$ `adverbial`, `"todos los perros"` $\to$ `quantificational`.
2. **Schema Integrity:** Successfully trapped missing required columns, whitespace inside instance identifiers, invalid label strings, and duplicate instance IDs.
3. **Mock Data Generation:** Successfully verified that mock partitions produce balanced reciprocal pairs, unaligned negative instances, and round-robin syntactic distributions.
4. **Leakage Detection:** Successfully verified zero leakage on clean splits and correctly caught deliberately injected overlapping pairs.
5. **SimpleDataFrame Operations:** Verified 100% equivalence with standard pandas DataFrame slicing, filtering, and dict export operations.

---

## 9. Downstream Interface Contracts

`src/data/loader.py` fulfills the exact interface contracts specified in `PROJECT.md`:

```python
# Primary Loader Interface
def load_phrasis_split(
    track: Union[str, int] = "track1",
    split: str = "train",
    data_dir: Union[str, Path] = "data",
    allow_mock_fallback: bool = True,
) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
    ['instance_id', 'pair_id', 'text1_lang', 'text2_lang', 'text1', 'text2', 'label', 'category']
    """
```

### Downstream Consumers:
1. **`src/data/reversal.py` (M1.2):** Receives the DataFrame from `load_phrasis_split`, indexes pairs by `pair_id`, and builds the Bi-directional Symmetrical Augmentation (BSA) dataset.
2. **`src/data/dataset.py` / `collator.py` (M2):** Receives `InstanceRecord` or DataFrame rows for standard and Siamese batch collation.
3. **`experiments/train.py` & `evaluate.py` (M3 / M4):** Calls `load_phrasis_split(track, split)` to ingest train, dev, and test splits.
4. **`src/evaluation/analysis.py` (M4):** Groups evaluation metrics by the `category` column to generate the syntactic breakdown table and plots.
5. **`tests/test_pipeline.py` (M_TEST):** Executes fast end-to-end CPU smoke tests using `allow_mock_fallback=True` in $<1$ second.

---

## 10. File Artifacts Created in this Investigation

1. `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m1_1/proposed_loader.py`:
   Full, verified source code ready to be copied directly into `src/data/loader.py`.
2. `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m1_1/test_proposed_loader.py`:
   Self-contained automated unit test script confirming 100% test pass rate.
3. `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m1_1/report.md`:
   This architectural specification report.
4. `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m1_1/handoff.md`:
   Formal 5-component handoff report.
