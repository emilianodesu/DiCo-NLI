# DiCo-NLI (SemEval-2027 Task 2): Data & Metrics Specification Report

**Author:** Specification Miner (`survey_miner`)  
**Target Tracks:** Track 1 (English Monolingual) & Track 2 (Spanish Monolingual)  
**Authoritative Sources Inspected:**
1. DiCo-NLI Official Task Website & Repository (`https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI`, `https://inigolopezgazpio.net/SemEval-2027-Task-2-DiCo-NLI/`)
2. Official Evaluation Package: `evaluation_functions/` (`metrics.py`, `labels.py`, `schemas.py`, `validation.py`, `io.py`, `scorer.py`, `cli.py`)
3. Official LaTeX Proposal & Evaluation Plan: `latex/task_proposal_v2/dico_nli_proposal_v2.tex`, `item4_evaluation_plan/4.Evaluation_Plan.tex`, `dico_nli_biblio.bib`
4. Official Data Releases: `final_data/` (`dev/`, `train/` participant labeled CSVs, reference CSVs, submission templates)
5. Starter Kit: `starter_kit/` (`main.py`, `src/trainer/helpers/data_reader.py`)
6. Course Guidelines: `Guia Proyectos - PLN-2027.pdf` and Project Specifications: `.agents/teamwork/ORIGINAL_REQUEST.md`

---

## Features Discovered

| # | Category | Feature | Description | Inputs | Outputs | Error Behavior | Discovered Via |
|---|----------|---------|-------------|--------|---------|----------------|----------------|
| 1 | Corpus Schema | Participant Labeled CSV | Public phrase-pair dataset for model training and local development | File path to CSV with columns `instance_id,pair_id,text1_lang,text2_lang,text1,text2,label` | Dataframe/Records of phrase pairs and labels | `ValueError` if required columns missing | `final_data/dev/dico_nli_dev_track1_participant_labeled.csv`, `starter_kit/src/trainer/helpers/data_reader.py` |
| 2 | Corpus Schema | Official Reference CSV | Gold reference for local and competition scoring | File path to CSV with columns `instance_id,pair_id,text1_lang,text2_lang,text1,text2,reverse_pair_id,label` | Validated `dict[str, GoldInstance]` | `ValidationError` on missing columns, broken reciprocity, or invalid reverse labels | `final_data/dev/dico_nli_dev_track1_reference.csv`, `evaluation_functions/io.py` |
| 3 | Corpus Schema | Submission Template CSV | 2-column baseline template for system predictions | File path to CSV with columns `instance_id,label` | Flat CSV template | Rejected if structure differs | `final_data/dev/dico_nli_dev_track1_submission_template.csv` |
| 4 | Identifiers | Source Pair ID (`pair_id`) | Unique canonical identifier for underlying phrase pair | `dico_phrasis_<7-digit-integer>` (e.g. `dico_phrasis_0000002`) | Canonical grouping key | Whitespace or empty string triggers `ValidationError` | `final_data/dev/`, `evaluation_functions/validation.py` |
| 5 | Identifiers | Evaluation Instance ID (`instance_id`) | Unique identifier for ordered instance in specific language and direction | `<pair_id>__<l1>-<l2>__<variant>` (e.g., `dico_phrasis_0000002__en-en__original`, `dico_phrasis_0000002__en-en__flipped`) | Unique string instance key | Duplicate or whitespace triggers `ValidationError` | `final_data/`, `evaluation_functions/validation.py` |
| 6 | Identifiers | Reciprocal Pointer (`reverse_pair_id`) | Explicit pointer linking $(A, B)$ to $(B, A)$ | Matching `instance_id` of reverse direction (or empty string/None for `NEGATIVE_OTHER`) | String pointer or None | Self-referencing, missing reverse, or asymmetric linkage triggers `ValidationError` | `evaluation_functions/validation.py`, `evaluation_functions/schemas.py` |
| 7 | Label Algebra | Reversible Labels Subset | Closed set of labels participating in directional inversion | `{"EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT"}` | Set $\mathcal{R}$ | Non-reversible labels cannot be inverted in consistency metrics | `evaluation_functions/labels.py` |
| 8 | Label Algebra | Full Label Inventory | Official 4-class classification target | `{"EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"}` | Set $\mathcal{L}$ | Any label outside $\mathcal{L}$ triggers `ValidationError` | `evaluation_functions/labels.py` |
| 9 | Reversal Logic | Deterministic Reversal Operator ($\Rev$) | Strict involutive mapping for directional inference | $\Rev(\text{EQUIV}) = \text{EQUIV}$; $\Rev(\text{FORWARD}) = \text{BACKWARD}$; $\Rev(\text{BACKWARD}) = \text{FORWARD}$ | Mapped label string | `ValidationError` if called on `NEGATIVE_OTHER` | `evaluation_functions/labels.py` |
| 10 | Reversal Logic | Symmetric Augmentation Reversal | Inversion logic for training data augmentation (BSA) | Inverts $\mathcal{R}$ as above, plus maps $\Rev(\text{NEGATIVE\_OTHER}) = \text{NEGATIVE\_OTHER}$ | Augmented $(B, A)$ training instances | N/A (applied during offline or online training augmentation) | `ORIGINAL_REQUEST.md` (R1.1), `task_proposal_v2.tex` |
| 11 | Metric | Weighted F1 ($F_w$) | Item-level multiclass F1 weighted by support across all 4 classes | Dicts of gold and predicted labels for all instances | Scalar $F_w \in [0.0, 1.0]$ | Undefined division mapped to $0.0$; total instances = 0 raises `ValueError` | `evaluation_functions/metrics.py:compute_classification_report` |
| 12 | Metric | Soft Consistency Rate (`SoftCons`) | Percentage of reversible source pairs where predicted direction $(A, B)$ matches $\Rev(\text{predicted } (B, A))$ | List of unique `ReversiblePair` and participant predictions | Scalar $\SoftCons \in [0.0, 1.0]$ | Zero reversible pairs raises `ValueError`; predicting `NEGATIVE_OTHER` gives $s_i = 0$ | `evaluation_functions/metrics.py:compute_consistency_report` |
| 13 | Metric | Hard Consistency Rate (`HardCons`) | Percentage of reversible source pairs where BOTH directions match gold truth | List of unique `ReversiblePair` and participant predictions | Scalar $\HardCons \in [0.0, 1.0]$ | Requires joint correctness: $h_i = 1 \implies s_i = 1$ | `evaluation_functions/metrics.py:compute_consistency_report` |
| 14 | Scorer Output | Diagnostic Metrics | Granular evaluation diagnostics | Classification & consistency reports | Macro F1, Per-label P/R/F1, Confusion Matrix, Error lists | Formatted in JSON and text | `evaluation_functions/metrics.py`, `evaluation_functions/scorer.py` |
| 15 | Submission Format | Codabench CSV/TSV | Prediction submission file for competition | 2-column CSV/TSV: `instance_id,label` | Scored result (`scores.json`, `scores.txt`) | Rejected if extra columns, missing/duplicate IDs, or empty rows exist | `evaluation_functions/io.py`, `evaluation_functions/scorer.py` |
| 16 | File Parsing | CSV/TSV Sniffer & Validator | Dialect and integrity validator for input files | File path, max bytes (64MB), max rows (5,000,000) | Parsed records | Rejects NUL bytes, blank lines, parser overflow, encoding faults | `evaluation_functions/io.py` |

---

## Edge Cases

| # | Feature | Input | Observed Behavior |
|---|---------|-------|-------------------|
| 1 | SoftCons | Predicts `NEGATIVE_OTHER` for $(A, B)$ and `NEGATIVE_OTHER` for $(B, A)$ on a reversible pair | Scorer assigns $s_i = 0$ because `first_prediction not in REVERSIBLE_LABELS`. `SoftCons` strictly requires predictions to be within $\mathcal{R}$. |
| 2 | SoftCons | Predicts `EQUIVALENCE` for both $(A, B)$ and $(B, A)$ when gold is `FORWARD_ENTAILMENT` / `BACKWARD_ENTAILMENT` | Scorer assigns $s_i = 1$ (SoftCons does NOT inspect gold labels; $\Rev(\text{EQUIV}) = \text{EQUIV}$). However, $h_i = 0$ for HardCons. |
| 3 | SoftCons | Predicts `FORWARD_ENTAILMENT` for both $(A, B)$ and $(B, A)$ | Scorer assigns $s_i = 0$ because $\Rev(\text{FORWARD}) = \text{BACKWARD} \neq \text{FORWARD}$. |
| 4 | HardCons | System predicts $(A, B)$ correctly, but $(B, A)$ incorrectly | Scorer assigns $h_i = 0$. HardCons is strict joint correctness: both directions must be gold-correct. |
| 5 | HardCons | System predicts both $(A, B)$ and $(B, A)$ correctly | Scorer assigns $h_i = 1$. By algebraic necessity, $s_i = 1$ as well ($\HardCons \le \SoftCons$ always holds). |
| 6 | Submission Validation | Prediction file contains an extra column (e.g. `instance_id,label,confidence`) | `evaluation_functions/io.py` raises `ValidationError: unexpected column(s): ['confidence']`. Codabench submission fails immediately. |
| 7 | Submission Validation | Prediction file has a trailing blank line at the end of the file | `evaluation_functions/io.py` raises `ValidationError: blank row at line X`. |
| 8 | Submission Validation | Prediction file has predictions sorted in different order than gold reference | Validator indexes predictions into a map by `instance_id`; order permutation is allowed as long as the ID set matches bijectively. |
| 9 | Submission Validation | Prediction file is missing 1 instance ID from the evaluation set | Validator detects `missing = sorted(gold_ids - prediction_ids)` and raises `ValidationError: missing predictions: 'dico_phrasis_...'.` |
| 10 | Submission Validation | Prediction file contains an unknown label (e.g. `CONTRADICTION`, `NEUTRAL`, lowercase `equivalence`) | Validator raises `ValidationError: Invalid label at row X '...'; expected one of: EQUIVALENCE, FORWARD_ENTAILMENT, BACKWARD_ENTAILMENT, NEGATIVE_OTHER.` |
| 11 | Reference Validation | A `NEGATIVE_OTHER` record has a non-empty `reverse_pair_id` | `evaluation_functions/validation.py` raises `ValidationError: '...' has reverse_pair_id but non-reversible label 'NEGATIVE_OTHER'.` |
| 12 | Metric Zero-Division | A label $\ell$ has 0 support and 0 predictions ($TP=FP=FN=0$) | `_safe_divide` returns $0.0$; precision, recall, and F1 are safely $0.0$. Macro F1 averages only over labels with support $> 0$. |

---

## 1. PhrasIS Corpus Specification & Track Details

### 1.1 Origin and Compositional Lineage
The DiCo-NLI benchmark is built upon **PhrasIS** (*Phrase Inference and Similarity benchmark*, Lopez-Gazpio et al., 2024; *Logic Journal of the IGPL*), which itself inherits from the **Interpretable Semantic Textual Similarity (iSTS)** task (SemEval-2015 Task 2, SemEval-2016 Task 2; Lopez-Gazpio et al., 2017).
Unlike standard sentence-level NLI benchmarks (e.g., SNLI, MNLI) or lexical relation benchmarks (e.g., WordNet, BLESS), PhrasIS decomposes sentence pairs extracted from image descriptions (Flickr30k, MSR-VTT) and news headlines into aligned semantic chunks.
In DiCo-NLI, the units of inference are ordered phrase pairs $(A, B)$ that evaluate compositionality over local syntactic structures rather than isolated single words or full discursive sentences.

### 1.2 Syntactic Categories
The corpus covers five main categories of syntactic chunks:
1. **Nominal Phrases (NP):**
   - Entities, modifiers, compound nouns, and appositions.
   - *English examples:* `"Pope Francis"` vs `"pope"`, `"A man and a woman"` vs `"A man"`, `"Wikileaks soldier Bradley Manning"` vs `"Bradley Manning"`.
   - *Spanish examples:* `"papa Francisco"` vs `"papa"`, `"Un hombre y una mujer"` vs `"Un hombre"`, `"el soldado de WikiLeaks Bradley Manning"` vs `"Bradley Manning"`.
2. **Verb Chains (VP):**
   - Complex predicates, aspectual verbs, gerunds, and modal verb chains.
   - *English examples:* `"in destroying"` vs `"destroy"`, `"are playing"` vs `"run"`, `"shaving"` vs `"is carving"`.
   - *Spanish examples:* `"al destruir"` vs `"destruir"`, `"están jugando"` vs `"corren"`.
3. **Prepositional Phrases (PP):**
   - Spatial, temporal, instrumental, and locative adjuncts.
   - *English examples:* `"with a tennis ball"` vs `"with a ball"`, `"in Afghan city"` vs `"in Afghanistan"`, `"through the water"` vs `"through the grass"`.
   - *Spanish examples:* `"con una pelota de tenis"` vs `"con una pelota"`, `"en una ciudad afgana"` vs `"en Afganistán"`.
4. **Adverbial Expressions (ADVP) & Prepositional Modifiers:**
   - Manner, degree, trajectory, and clausal adjuncts.
   - *English examples:* `"as new CEO"` vs `"as CEO"`, `"down a snowy hill"` vs `"sledding"`, `"onto another"` vs `"to another"`.
   - *Spanish examples:* `"como nuevo CEO"` vs `"como CEO"`.
5. **Quantificational Expressions & Numerical Modifiers:**
   - Aristotelian and generalized quantifiers, cardinality changes, and determiners.
   - *English examples:* `"Everyone is hungry"` vs `"Someone is hungry"`, `"Nine children"` vs `"Five children"`, `"Three dogs"` vs `"The dog"`.
   - *Spanish examples:* `"Nueve niños"` vs `"Cinco niños"`, `"Tres perros"` vs `"El perro"`.

### 1.3 Splits and Instance Counts
The benchmark defines four tracks:
- **Track 1:** English Monolingual
- **Track 2:** Spanish Monolingual
- **Track 3:** Basque Monolingual
- **Track 4:** Mixed Multilingual (Cross-lingual combinations of EN/ES/EU)

The official release partition counts are structured as follows:

| Partition | Track 1 (English) | Track 2 (Spanish) | Track 3 (Basque) | Track 4 (Mixed Multilingual) |
|-----------|------------------:|------------------:|-----------------:|-----------------------------:|
| **Train** | 3,042 | 3,042 | 3,042 | 18,252 |
| **Dev** | 660 | 660 | 660 | 3,960 |
| **Total Public** | **3,702** | **3,702** | **3,702** | **22,212** |

#### Label Distribution (Tracks 1, 2, and 3):
| Label | Train Count | Train % | Dev Count | Dev % | Reversibility Status |
|---|---:|---:|---:|---:|---|
| `FORWARD_ENTAILMENT` | 881 | 28.96% | 190 | 28.79% | Reversible (maps to `BACKWARD`) |
| `BACKWARD_ENTAILMENT` | 881 | 28.96% | 190 | 28.79% | Reversible (maps to `FORWARD`) |
| `EQUIVALENCE` | 802 | 26.36% | 174 | 26.36% | Reversible (maps to `EQUIVALENCE`) |
| `NEGATIVE_OTHER` | 478 | 15.71% | 106 | 16.06% | Non-reversible noise / negative subset |
| **Total** | **3,042** | **100.0%** | **660** | **100.0%** | |

*Noticeable Structural Property:* `FORWARD_ENTAILMENT` and `BACKWARD_ENTAILMENT` have **identical frequencies** (881 in Train, 190 in Dev) because the dataset contains the reciprocal flipped pair for every directed source pair.

#### Source-Pair Grouping & Leakage Prevention:
The fundamental splitting unit is the **source phrase pair** (`pair_id`), not the isolated ordered instance.
For any source pair $(A, B)$:
$$\{(A, B), (B, A), \text{Spanish}(A, B), \text{Spanish}(B, A), \text{Basque}(A, B), \dots\} \subset \text{Split}_k$$
This guarantees **zero directional leakage** (no $(B, A)$ in test if $(A, B)$ was in train) and **zero cross-lingual leakage** (no Spanish translation in test if English counterpart was in train).

### 1.4 File Naming and Schema Conventions
For each split (`train`, `dev`) and track (`track1`, `track2`):
1. **`dico_nli_<split>_<track>_participant_labeled.csv`**:
   - Columns: `instance_id,pair_id,text1_lang,text2_lang,text1,text2,label`
2. **`dico_nli_<split>_<track>_reference.csv`**:
   - Columns: `instance_id,pair_id,text1_lang,text2_lang,text1,text2,reverse_pair_id,label`
3. **`dico_nli_<split>_<track>_submission_template.csv`**:
   - Columns: `instance_id,label`

### 1.5 Instance Identifier Conventions
The identifiers adhere to strict canonical formats:
- **`pair_id`:** `dico_phrasis_<07d>` (e.g. `dico_phrasis_0000002`, `dico_phrasis_0001949`)
- **`instance_id`:** `<pair_id>__<text1_lang>-<text2_lang>__<variant>`
  - `<text1_lang>-<text2_lang>`: `en-en` for Track 1; `es-es` for Track 2.
  - `<variant>`:
    - `original`: Canonical forward presentation of the phrase pair.
    - `flipped`: Inverted presentation $(B, A)$.
    - For `NEGATIVE_OTHER` instances: instances typically appear with `__original`, and their `reverse_pair_id` is empty string (`""` / `None`).

---

## 2. Deterministic Reversal Logic & Pair Algebra

### 2.1 The Reversal Operator $\Rev(\cdot)$
In natural logic and fine-grained NLI, the relation between two phrases $A$ and $B$ under inversion of argument order obeys a strict deterministic algebra:

$$\Rev(y) = \begin{cases}
\text{EQUIVALENCE}, & \text{if } y = \text{EQUIVALENCE} \\
\text{BACKWARD\_ENTAILMENT}, & \text{if } y = \text{FORWARD\_ENTAILMENT} \\
\text{FORWARD\_ENTAILMENT}, & \text{if } y = \text{BACKWARD\_ENTAILMENT}
\end{cases}$$

Notice that the operator is an **involution**:
$$\Rev(\Rev(y)) = y \quad \forall y \in \mathcal{R}$$
where $\mathcal{R} = \{\text{EQUIVALENCE}, \text{FORWARD\_ENTAILMENT}, \text{BACKWARD\_ENTAILMENT}\}$.

### 2.2 Distinguishing Evaluation Algebra vs. Data Augmentation
A vital distinction discovered in the specification:
1. **In Official Evaluation (`evaluation_functions/labels.py` & `metrics.py`):**
   - $\Rev$ is defined **only** on $\mathcal{R}$.
   - Calling $\Rev(\text{NEGATIVE\_OTHER})$ raises a `ValidationError`.
   - The set of reversible pairs $\mathcal{P}$ consists exclusively of pairs where the gold label belongs to $\mathcal{R}$ and `reverse_pair_id` is non-null.
   - If a model predicts `NEGATIVE_OTHER` for an item in $\mathcal{P}$, it receives $s_i = 0$ (not soft-consistent) and $h_i = 0$ (not hard-correct).
2. **In Bidirectional Symmetric Data Augmentation (BSA, R1.1 & R2.2):**
   - When preparing training sets or augmenting the training distribution with reverse pairs:
     - $\text{EQUIVALENCE} \leftrightarrow \text{EQUIVALENCE}$
     - $\text{FORWARD\_ENTAILMENT} \leftrightarrow \text{BACKWARD\_ENTAILMENT}$
     - $\text{BACKWARD\_ENTAILMENT} \leftrightarrow \text{FORWARD\_ENTAILMENT}$
     - $\text{NEGATIVE\_OTHER} \leftrightarrow \text{NEGATIVE\_OTHER}$
   - If an unaligned or non-reversible $(A, B)$ phrase pair is classified as `NEGATIVE_OTHER` (unrelated/noise), presenting $(B, A)$ also constitutes `NEGATIVE_OTHER`.

### 2.3 Linking, Matching, and Indexing Pairs
In reference data and evaluation pipelines, pairs $(A, B)$ and $(B, A)$ are processed as follows:
- **Pair Formation:** Given instances $x_1$ with ID $id_1$ and $x_2$ with ID $id_2$, they form a reciprocal pair iff:
  $$id_2 = \text{reverse\_pair\_id}(id_1) \quad \text{and} \quad id_1 = \text{reverse\_pair\_id}(id_2)$$
- **Canonical Deduplication:** To evaluate consistency metrics without double counting, each pair is indexed using an unordered set (or `frozenset`):
  $$k = \{id_1, id_2\}$$
  Each unique $k$ corresponds to exactly one `ReversiblePair(first_id, second_id, first_label, second_label)`.
- **Count Validation:**
  - For Track 1 / Track 2 Dev:
    $$N_{\text{reversible instances}} = 190 (\text{FWD}) + 190 (\text{BWD}) + 174 (\text{EQUIV}) = 554$$
    $$|\mathcal{P}| = \frac{554}{2} = 277 \text{ reversible source pairs.}$$
  - For Track 1 / Track 2 Train:
    $$N_{\text{reversible instances}} = 881 (\text{FWD}) + 881 (\text{BWD}) + 802 (\text{EQUIV}) = 2,564$$
    $$|\mathcal{P}| = \frac{2,564}{2} = 1,282 \text{ reversible source pairs.}$$

---

## 3. Mathematical and Programmatic Metric Definitions

DiCo-NLI reports a three-score profile for each track: $(F_w, \SoftCons, \HardCons)$.

### 3.1 Weighted F1 ($F_w$)
Measures standard item-level prediction quality across all $N$ instances and all 4 classes in $\mathcal{L}$.

#### Mathematical Formulation:
Let $\mathcal{L} = \{1, 2, 3, 4\}$ denote the classes: `BACKWARD_ENTAILMENT`, `EQUIVALENCE`, `FORWARD_ENTAILMENT`, `NEGATIVE_OTHER`.
Let $C_{\ell, k}$ be the confusion matrix entry representing instances with gold class $\ell$ predicted as class $k$.
For each label $\ell \in \mathcal{L}$:
- Support: $n_\ell = \sum_{k \in \mathcal{L}} C_{\ell, k}$
- Total instances: $N = \sum_{\ell \in \mathcal{L}} n_\ell$
- True Positives: $TP_\ell = C_{\ell, \ell}$
- False Positives: $FP_\ell = \sum_{k \neq \ell} C_{k, \ell}$
- False Negatives: $FN_\ell = \sum_{k \neq \ell} C_{\ell, k}$
- Precision: $P_\ell = \frac{TP_\ell}{TP_\ell + FP_\ell}$ (with $0.0$ if $TP_\ell + FP_\ell = 0$)
- Recall: $R_\ell = \frac{TP_\ell}{TP_\ell + FN_\ell}$ (with $0.0$ if $TP_\ell + FN_\ell = 0$)
- F1-Score: $F_\ell = \frac{2 P_\ell R_\ell}{P_\ell + R_\ell}$ (with $0.0$ if $P_\ell + R_\ell = 0$)

The weighted F1 is the support-weighted average of per-class F1:
$$F_w = \sum_{\ell \in \mathcal{L}} \frac{n_\ell}{N} F_\ell$$

### 3.2 Soft Consistency Rate ($\SoftCons$)
Measures whether the model's predictions in forward and reverse directions are logically compatible with each other, completely independent of whether they match the ground truth.

#### Mathematical Formulation:
Let $\mathcal{P}$ be the set of unique reversible source pairs.
For pair $i \in \mathcal{P}$, let $x_i$ be the first instance and $x_i^r$ be its reciprocal reverse instance.
Let $\hat{y}_i = f(x_i)$ and $\hat{y}_i^r = f(x_i^r)$ be the model predictions.
The directional consistency indicator $s_i \in \{0, 1\}$ is defined as:
$$s_i = \begin{cases}
1, & \text{if } \hat{y}_i \in \mathcal{R} \;\land\; \hat{y}_i^r = \Rev(\hat{y}_i) \\
0, & \text{otherwise}
\end{cases}$$

The overall SoftCons metric is:
$$\SoftCons = \frac{1}{|\mathcal{P}|} \sum_{i \in \mathcal{P}} s_i$$

*Key Properties:*
- Invariant to gold label.
- If $f(x_i) = \text{EQUIVALENCE}$ and $f(x_i^r) = \text{EQUIVALENCE}$, $s_i = 1$ (even if gold is `FORWARD_ENTAILMENT`).
- If $f(x_i) = \text{FORWARD_ENTAILMENT}$ and $f(x_i^r) = \text{FORWARD_ENTAILMENT}$, $s_i = 0$ (the classic Reversal Curse failure mode).
- If either $\hat{y}_i = \text{NEGATIVE_OTHER}$ or $\hat{y}_i^r = \text{NEGATIVE_OTHER}$, $s_i = 0$.

### 3.3 Hard Consistency Rate ($\HardCons$)
Measures strict paired correctness over complete reversible units: both directions must be predicted correctly according to the ground truth.

#### Mathematical Formulation:
Let $y_i$ be the gold label for $x_i$, and $y_i^r = \Rev(y_i)$ be the gold label for $x_i^r$.
The paired correctness indicator $h_i \in \{0, 1\}$ is:
$$h_i = \begin{cases}
1, & \text{if } \hat{y}_i = y_i \;\land\; \hat{y}_i^r = y_i^r \\
0, & \text{otherwise}
\end{cases}$$

The overall HardCons metric is:
$$\HardCons = \frac{1}{|\mathcal{P}|} \sum_{i \in \mathcal{P}} h_i$$

*Mathematical Relationships and Theorems:*
1. **Correctness Entails Consistency:**
   Because gold reference labels strictly satisfy $y_i^r = \Rev(y_i)$, if $\hat{y}_i = y_i$ and $\hat{y}_i^r = y_i^r$, then:
   $$\hat{y}_i^r = \Rev(y_i) = \Rev(\hat{y}_i)$$
   Therefore:
   $$h_i = 1 \implies s_i = 1$$
2. **Metric Inequality:**
   $$\HardCons \le \SoftCons$$
   Any run where $\HardCons > \SoftCons$ violates mathematical definitions and indicates a scoring bug.
3. **Upper Bound by Directional Accuracies:**
   $$\HardCons \le \min\left(\text{Acc}_{\text{forward}}(\mathcal{P}), \text{Acc}_{\text{reverse}}(\mathcal{P})\right) \le \text{Acc}(\mathcal{P})$$

### 3.4 Official Scorer Logic Reference
The exact algorithmic implementation from `evaluation_functions/metrics.py`:
```python
def compute_consistency_report(
    pairs: list[ReversiblePair],
    predicted_labels: Mapping[str, str],
) -> ConsistencyReport:
    soft_count = 0
    hard_count = 0

    for pair in pairs:
        first_prediction = predicted_labels[pair.first_id]
        second_prediction = predicted_labels[pair.second_id]

        soft_ok = (
            first_prediction in REVERSIBLE_LABELS
            and second_prediction == reverse_label(first_prediction)
        )
        hard_ok = (
            first_prediction == pair.first_label
            and second_prediction == pair.second_label
        )

        if soft_ok:
            soft_count += 1
        if hard_ok:
            hard_count += 1

    total = len(pairs)
    return ConsistencyReport(
        soft_cons=soft_count / total,
        hard_cons=hard_count / total,
        reversible_pairs=total,
        soft_consistent_pairs=soft_count,
        hard_correct_pairs=hard_count,
    )
```

---

## 4. Codabench Submission Format & Validation Rules

### 4.1 Strict Two-Column File Specification
Prediction files must be flat CSV or TSV files formatted with UTF-8 encoding (BOM `utf-8-sig` accepted) containing exactly two columns:
```csv
instance_id,label
dico_phrasis_0000002__en-en__flipped,FORWARD_ENTAILMENT
dico_phrasis_0000002__en-en__original,BACKWARD_ENTAILMENT
dico_phrasis_0000004__en-en__flipped,FORWARD_ENTAILMENT
dico_phrasis_0000004__en-en__original,BACKWARD_ENTAILMENT
```

### 4.2 Comprehensive Validator Checklist (from `io.py` and `validation.py`)
To prevent Codabench submission rejection, submissions must strictly satisfy:

1. **Header Row:**
   - First line must be exactly `instance_id,label` (or tab-separated).
   - Extra columns are strictly rejected when `allow_extra_fields=False` (e.g. `confidence`, `prob_0`, `text1` trigger `ValidationError: unexpected column(s)`).
2. **Completeness & Row Count:**
   - File must contain exactly one prediction per test instance (e.g., 660 rows for Dev).
   - Missing instance IDs or extra unrecognized IDs raise `ValidationError: missing predictions: ...; unknown predictions: ...`.
3. **Uniqueness:**
   - No duplicate `instance_id` allowed (`Duplicate prediction for instance_id '...'`).
4. **No Empty or Whitespace Values:**
   - Neither `instance_id` nor `label` may be empty, null, or contain whitespace (`_validate_identifier`).
   - Blank rows (even empty lines at EOF) trigger `ValidationError: blank row at line X`.
5. **Exact Categorical Label Values:**
   - Every label must match verbatim one of the four tokens:
     - `EQUIVALENCE`
     - `FORWARD_ENTAILMENT`
     - `BACKWARD_ENTAILMENT`
     - `NEGATIVE_OTHER`
   - Case-sensitive; no leading/trailing spaces.
6. **File Limits & Sanitization:**
   - Max file size: 64 MB.
   - Max lines: 5,000,000.
   - No NUL bytes (`\x00`).
7. **CodaBench Output Formats:**
   - `scores.txt`:
     ```text
     weighted_f1: 0.750000000000
     soft_cons: 0.840000000000
     hard_cons: 0.790000000000
     ```
   - `scores.json`: Comprehensive report containing `weighted_f1`, `macro_f1`, `accuracy`, `confusion_matrix`, `per_label` stats, `soft_cons`, `hard_cons`, and sample `pair_errors`.

---

## 5. Architectural Recommendation for `src/data/` and `src/metrics/`

To cleanly decouple dataset handling, model training routines, and evaluation metrics while remaining fully compliant with Codabench and the project's requirements, the following module architecture is recommended:

```
src/
├── data/
│   ├── __init__.py
│   ├── schemas.py           # Dataclasses: InstanceRecord, GoldRecord, ReversiblePair
│   ├── dataset.py           # PyTorch Dataset for single and paired instances
│   ├── collator.py          # Dynamic padding, tokenizer collation, paired batching
│   ├── augmentation.py      # Bidirectional Symmetric Augmentation (BSA) logic
│   └── loader.py            # CSV loaders, split auditors, and leak verification
│
├── metrics/
│   ├── __init__.py
│   ├── scorer.py            # Local official evaluator matching evaluation_functions
│   ├── classification.py    # Weighted F1, Macro F1, Per-label P/R/F1, Confusion Matrix
│   ├── consistency.py       # Vectorized SoftCons and HardCons computation
│   ├── decoding.py          # Symmetrized Decoding (SD) post-hoc projection
│   └── submission.py        # Validated Codabench CSV exporter and format auditor
```

### 5.1 Data Module Specifications (`src/data/`)

1. **`loader.py`**:
   - `load_dico_csv(path: Path) -> list[InstanceRecord]`: Safe reading supporting participant labeled CSVs, reference CSVs, and prediction templates.
   - `verify_split_isolation(train_records, val_records) -> bool`: Verifies that no `pair_id` present in training appears in validation (guarantees zero leakage).
2. **`augmentation.py` (BSA - Symmetric Augmentation)**:
   - Takes a dataset $D = \{(A_i, B_i, y_i)\}$ and produces $D_{\text{BSA}} = D \cup \{(B_i, A_i, \Rev_{\text{BSA}}(y_i))\}$.
   - Inverts labels deterministically including `NEGATIVE_OTHER -> NEGATIVE_OTHER`.
   - Generates synthetic reciprocal IDs (e.g., adding `__flipped` suffix).
3. **`dataset.py` & `collator.py`**:
   - Support both **Standard Unidirectional Batching** (for naive baseline and BSA) and **Paired Siamese Batching** (for Consistency Regularization Loss / CRL).
   - Paired batch collation groups $(A_i, B_i)$ and $(B_i, A_i)$ in adjacent or parallel tensor slots:
     `batch["input_ids_fwd"]`, `batch["input_ids_rev"]`, allowing vectorized divergence computation $\mathcal{D}_{\text{KL}}(P(A, B) \parallel \Rev(P(B, A)))$.

### 5.2 Metrics Module Specifications (`src/metrics/`)

1. **`scorer.py` & `consistency.py`**:
   - Provides pure, dependency-minimal metric calculations matching `evaluation_functions/` exactly to 12 decimal places.
   - Vectorized evaluation for fast validation epochs during GPU training.
2. **`decoding.py` (Symmetrized Decoding - SD)**:
   - Post-hoc probability symmetrization operator.
   - Given logits or predicted distributions $P(A, B) \in \Delta^3$ and $P(B, A) \in \Delta^3$:
     - Reversal permutation matrix $\Pi_{\Rev}$:
       $$\Pi_{\Rev} = \begin{bmatrix}
       0 & 0 & 1 & 0 \\
       0 & 1 & 0 & 0 \\
       1 & 0 & 0 & 0 \\
       0 & 0 & 0 & 1
       \end{bmatrix}$$
       (where indices are `[BWD, EQUIV, FWD, NEG]`).
     - Symmetrized probability:
       $$P_{\text{sym}}(A, B) = \frac{1}{2}\left(P(A, B) + \Pi_{\Rev} P(B, A)\right)$$
       $$P_{\text{sym}}(B, A) = \Pi_{\Rev} P_{\text{sym}}(A, B)$$
     - **Mathematical Guarantee:** Symmetrized decoding provably achieves $\SoftCons = 1.0$ ($100\%$) across all reversible pairs!
3. **`submission.py`**:
   - `export_codabench_submission(predictions, output_path)`: Exports strictly two columns `instance_id,label`.
   - `validate_submission_file(file_path, expected_reference_path)`: Re-runs official validation checks from `evaluation_functions/io.py` before submission to prevent disqualification.

---

## Summary of Discovered Metrics & Targets for Model Development

| Metric | Target Track 1 (English DeBERTa) | Target Track 2 (Spanish BETO / RoBERTa) | Key Methodological Leverage |
|---|---|---|---|
| **Weighted F1** | $\ge 0.75$ (Pilot benchmark: 0.75) | $\ge 0.72$ | Pretrained NLI encoders, balanced cross-entropy, warmup |
| **SoftCons** | $1.00$ ($100\%$ with SD; $\ge 0.84$ baseline) | $1.00$ ($100\%$ with SD; $\ge 0.80$ baseline) | Symmetrized Decoding (SD) post-hoc projection & CRL |
| **HardCons** | $\ge 0.79$ (Pilot benchmark: 0.79) | $\ge 0.75$ | Joint Siamese Regularization Loss (CRL) + BSA |
