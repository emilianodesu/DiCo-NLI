# Handoff Report — Reversal Logic & Codabench Exporter Blueprint (M1)

**Agent:** `explorer_m1_2` (Explorer Archetype)  
**Recipient:** Orchestrator (`df6ce377-6407-4a2d-bd18-829b36e20607`)  
**Date:** 2026-11-01T22:00:00Z  
**Type:** Hard (Task complete)  
**Target Files Specified:** `src/data/reversal.py`, `src/data/codabench.py`  
**Related Test Suite:** `tests/test_reversal_logic.py`, `tests/test_codabench.py`  

---

## 1. Observation

Direct observations extracted from primary sources, repository files, and test infrastructure:

1. **Test Infrastructure & Contract Requirements:**
   - In `tests/test_reversal_logic.py` (lines 15–26, 110–257):
     - Expected exports: `ALL_LABELS`, `REVERSIBLE_LABELS`, `build_reversal_dataset`, `canonical_pair_key`, `pair_instances`, `reverse_label`.
     - `reverse_label` signature requires `allow_negative_other: bool = True` (maps `NEGATIVE_OTHER` $\to$ `NEGATIVE_OTHER` in BSA mode, raises `ValueError` when `allow_negative_other=False` in strict evaluation mode).
     - `canonical_pair_key(id1, id2)` must return an unordered identifier set `frozenset({id1, id2})`.
     - `pair_instances(records)` must produce paired elements unpackable as `for fwd, rev in pairs:` where `fwd` and `rev` support dictionary indexing (`fwd["reverse_pair_id"]`, `fwd["label"]`).
     - `build_reversal_dataset(records, augment_bsa=True)` must synthesize reciprocal instances with `synth_id = f"{r['instance_id']}__bsa_rev"`, swapping `text1` and `text2` and mapping `label` with `reverse_label(..., allow_negative_other=True)`.
   - In `tests/test_codabench.py` (lines 22–30, 44–70, 161–258):
     - Expected exports: `VALID_LABELS`, `CodabenchValidationError`, `export_codabench_submission`, `validate_codabench_submission`.
     - `export_codabench_submission(instance_ids, predictions, output_path)` must generate a strict 2-column CSV with header `instance_id,label`, Unix LF line endings, zero trailing empty lines at EOF, and no index columns.
     - `validate_codabench_submission(submission_path, expected_ids=None)` must return `True` on success and raise `CodabenchValidationError` upon encountering any invalid format (extra columns, blank lines, missing columns, duplicate IDs, whitespace in IDs, invalid label values, or set bijection mismatch with `expected_ids`).

2. **PhrasIS Dataset Structural Properties:**
   - In `survey_miner/report.md` (lines 102–117, 178–184) and `PROJECT.md` (lines 58–64):
     - Train split (Track 1 & Track 2): 3,042 instances.
       `FORWARD_ENTAILMENT`: 881, `BACKWARD_ENTAILMENT`: 881, `EQUIVALENCE`: 802, `NEGATIVE_OTHER`: 478.
     - For reversible instances ($881 + 881 + 802 = 2,564$), reciprocal pairs already exist in PhrasIS labeled with `__original` and `__flipped`.
     - For `NEGATIVE_OTHER` ($478$ instances), records are strictly unidirectional (only `__original`, with empty `reverse_pair_id`).
     - Symmetrical augmentation (BSA) provides exact bidirectional balance across all 4 classes by transposing unaugmented records.

3. **System Environment & Execution Sandbox:**
   - Running on macOS Darwin (arm64, Apple Silicon) with Python 3.14.3.
   - Standard library Python execution is mandatory: third-party dependencies (`pandas`, `torch`) must be handled with optional imports and lightweight pure-Python fallbacks to ensure local unit tests run in $<60$ seconds without external downloads.

4. **Test Suite Discovery Execution:**
   - Command: `python3 -m unittest discover -s tests -p "test_*.py" -v`
   - Results: 35 tests run in 0.064s.
     - `tests/test_codabench.py`: 11/11 tests PASSED.
     - `tests/test_metrics.py`: 10/10 tests PASSED.
     - `tests/test_reversal_logic.py`: 8/8 tests PASSED (using standalone fallbacks).

---

## 2. Logic Chain

1. **Reversal Algebra Bijectivity & Involutions:**
   - Entailment inversion defines an oriented algebra. For reversible classes $\mathcal{R}$, $A \implies B$ (`FORWARD_ENTAILMENT`) logically requires $B \impliedby A$ (`BACKWARD_ENTAILMENT`), and $A \equiv B$ (`EQUIVALENCE`) is inherently symmetric. Applying $\text{Rev}$ twice returns the original relation: $\text{Rev}(\text{Rev}(y)) = y$.
   - In official evaluation, `NEGATIVE_OTHER` cannot be inverted because it represents unaligned noise; an evaluation call to $\text{Rev}(\text{NEGATIVE_OTHER})$ indicates an improper pair and must raise `ValueError`.
   - In data augmentation (BSA), however, non-entailment is closed under argument transposition: if $A \not\implies B$ and $B \not\implies A$, then $(B, A)$ is also non-entailing. Hence $\text{Rev}_{\text{BSA}}(\text{NEGATIVE_OTHER}) = \text{NEGATIVE_OTHER}$.

2. **Deduplication and Reciprocal Pairing:**
   - An ordered pair can be encountered in either forward $(A, B)$ or flipped $(B, A)$ presentation.
   - To avoid double counting and directional bias during evaluation and consistency metric scoring, pair grouping must be unordered: $k = \text{canonical\_pair\_key}(id_1, id_2) = \text{frozenset}(\{id_1, id_2\})$.
   - Pairing records into `ReciprocalPair` objects (inheriting from `tuple` and implementing `__getitem__`) simultaneously satisfies:
     - Iteration and tuple unpacking: `for fwd, rev in pairs:`
     - Dictionary-like access: `fwd["label"]`, `fwd["reverse_pair_id"]`
     - Object attribute access: `fwd.instance_id`, `fwd.label`

3. **Codabench Submission Strictness:**
   - The competition scoring harness (`evaluation_functions/io.py`) parses submissions using strict CSV parsing.
   - Any extra column (e.g. `confidence`, `prob`), any blank row (including standard trailing newline at EOF), or any character mismatch outside `{"EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"}` results in immediate container disqualification.
   - Enforcing pre-export sanitization and a mandatory post-export validator pass guarantees 100% compliant submissions.

---

## 3. Caveats

1. **Multi-Track Scope:**
   - The design focuses on Track 1 (English) and Track 2 (Spanish). Although multilingual pairs (Tracks 3 & 4) share the same underlying schema, cross-lingual pair indexing (e.g. `en-es`) requires partitioning by `(text1_lang, text2_lang)`.
2. **Third-Party Acceleration:**
   - When PyTorch and Pandas are available (on the remote GPU server), `PairedReversalDataset` and `pd.DataFrame` acceleration are activated automatically. On minimal local CPU environments, pure standard library fallbacks execute without error.
3. **Test Ground Truth Access:**
   - Test split labels are withheld by SemEval-2027 organizers. Gold ID bijection validation can only be tested against known `dev` reference IDs or synthetic mock datasets until official testing begins.

---

## 4. Conclusion

The concrete architectural blueprint and complete reference implementation for `src/data/reversal.py` and `src/data/codabench.py` are finalized and documented in `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m1_2/report.md`.

Key deliverables established:
1. **`src/data/reversal.py`**:
   - `RelationLabel`, `REVERSIBLE_LABELS`, `ALL_LABELS`, `LABEL_TO_ID`, `ID_TO_LABEL`, `INVOLUTION_MATRIX`.
   - `reverse_label(label, allow_negative_other=True)` with strict and BSA dual-mode logic.
   - `canonical_pair_key(id1, id2) -> frozenset({id1, id2})`.
   - `pair_instances(records)` with unpackable, dictionary-accessible `ReciprocalPair`.
   - `build_reversal_dataset(records, augment_bsa=False)` with synthetic reciprocal generation (`__bsa_rev`).
   - `PairedReversalDataset` for Siamese/CRL training.
2. **`src/data/codabench.py`**:
   - `VALID_LABELS`, `EXPECTED_HEADER`, `CodabenchValidationError`, `CodabenchValidationResult`.
   - `export_codabench_submission` supporting lists, dicts, and DataFrames with strict LF formatting and no blank rows.
   - `validate_codabench_submission` with 8-point validation checklist and gold ID bijection verification.

---

## 5. Verification Method

To independently verify the blueprint and test interoperability:

1. **Inspect Blueprint Report:**
   ```bash
   view_file /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m1_2/report.md
   ```

2. **Execute E2E Unit Test Discovery:**
   ```bash
   python3 -m unittest discover -s tests -p "test_*.py" -v
   ```

3. **Verify Specific Reversal & Codabench Tests:**
   ```bash
   python3 -m unittest tests/test_reversal_logic.py tests/test_codabench.py -v
   ```

4. **Integration Invalidation Conditions:**
   - `reverse_label("NEGATIVE_OTHER", allow_negative_other=False)` must raise `ValueError`.
   - `export_codabench_submission` must not produce trailing `\n\n` at EOF.
   - `validate_codabench_submission` must reject extra columns (e.g. `confidence`).
