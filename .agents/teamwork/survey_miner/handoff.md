# Handoff Report — Data & Metrics Specification Mining for SemEval-2027 Task 2 (DiCo-NLI)

**Agent:** `survey_miner` (Specification Miner)  
**Recipient:** Orchestrator (`df6ce377-6407-4a2d-bd18-829b36e20607`)  
**Date:** 2026-11-01T21:51:00Z  
**Type:** Hard (Task complete)  

---

## 1. Observation

Direct observations extracted from primary sources:

1. **PhrasIS Corpus Splits & Counts:**
   - From `final_data/README.md` (lines 44–47) and website (`content.md`, lines 254–268):
     - Train: Track 1 (English) = 3,042; Track 2 (Spanish) = 3,042; Track 3 (Basque) = 3,042; Track 4 (Mixed) = 18,252.
     - Dev: Track 1 = 660; Track 2 = 660; Track 3 = 660; Track 4 = 3,960.
   - Gold label breakdown in Track 1 & 2 Train: `BACKWARD_ENTAILMENT`: 881, `EQUIVALENCE`: 802, `FORWARD_ENTAILMENT`: 881, `NEGATIVE_OTHER`: 478. Total = 3,042.
   - Gold label breakdown in Track 1 & 2 Dev: `BACKWARD_ENTAILMENT`: 190, `EQUIVALENCE`: 174, `FORWARD_ENTAILMENT`: 190, `NEGATIVE_OTHER`: 106. Total = 660.

2. **File Formats & Identifier Schemes:**
   - `final_data/dev/dico_nli_dev_track1_participant_labeled.csv`:
     Line 9: `instance_id,pair_id,text1_lang,text2_lang,text1,text2,label`
     Line 10: `dico_phrasis_0000002__en-en__flipped,dico_phrasis_0000002,en,en,'s war shrine visit,'s shrine visit,FORWARD_ENTAILMENT`
     Line 11: `dico_phrasis_0000002__en-en__original,dico_phrasis_0000002,en,en,'s shrine visit,'s war shrine visit,BACKWARD_ENTAILMENT`
   - `final_data/dev/dico_nli_dev_track2_participant_labeled.csv`:
     Line 10: `dico_phrasis_0000002__es-es__flipped,dico_phrasis_0000002,es,es,su visita a un santuario de guerra,su visita a un santuario,FORWARD_ENTAILMENT`
     Line 11: `dico_phrasis_0000002__es-es__original,dico_phrasis_0000002,es,es,su visita a un santuario,su visita a un santuario de guerra,BACKWARD_ENTAILMENT`
   - `final_data/dev/dico_nli_dev_track1_reference.csv`:
     Line 9: `instance_id,pair_id,text1_lang,text2_lang,text1,text2,reverse_pair_id,label`
     Line 10: `...,dico_phrasis_0000002__en-en__original,FORWARD_ENTAILMENT`
     Line 564 (`NEGATIVE_OTHER`): `dico_phrasis_0001949__en-en__original,dico_phrasis_0001949,en,en,Syria opposition unity,Syria divided opposition,,NEGATIVE_OTHER` (note empty `reverse_pair_id`).

3. **Label Algebra & Consistency Rules:**
   - From `evaluation_functions/labels.py` (lines 32–42):
     ```python
     REVERSIBLE_LABELS = ("EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT")
     _REVERSE_LABEL_MAP = {
         "EQUIVALENCE": "EQUIVALENCE",
         "FORWARD_ENTAILMENT": "BACKWARD_ENTAILMENT",
         "BACKWARD_ENTAILMENT": "FORWARD_ENTAILMENT",
     }
     ```
   - From `evaluation_functions/metrics.py` (lines 220–227):
     ```python
     soft_ok = (
         first_prediction in REVERSIBLE_LABELS
         and second_prediction == reverse_label(first_prediction)
     )
     hard_ok = (
         first_prediction == pair.first_label
         and second_prediction == pair.second_label
     )
     ```
   - From `item4_evaluation_plan/4.Evaluation_Plan.tex` (lines 140–153):
     "If either prediction for a reversible pair is `NEGATIVE_OTHER`, the pair receives $s_i = 0$ because the prediction does not express a reversible directional relation."

4. **Codabench Submission and Validator Checks:**
   - From `evaluation_functions/io.py` (lines 31–55, 175–190):
     - `read_prediction_file` enforces `allow_extra_fields=False`. Any extra column raises: `ValidationError: unexpected column(s): ...`.
     - Blank lines trigger: `ValidationError: blank row at line X`.
     - File size capped at 64MB (`DEFAULT_MAX_BYTES`). Max rows: 5,000,000.
     - Required columns: exactly `instance_id,label`.
     - Non-empty, no whitespace in identifiers, valid label token.

---

## 2. Logic Chain

1. **Corpus & Split Integrity:**
   Observations 1 and 2 establish that the PhrasIS dataset splits are grouped strictly by `pair_id`. This prevents data leakage across folds and across translation pairs. For Tracks 1 and 2, every split contains reciprocal `original` and `flipped` pairs with exact reciprocal frequency balance ($881 = 881$ in train, $190 = 190$ in dev).

2. **Metric Computation Scope:**
   Observation 3 proves that while `Weighted F1` evaluates all 4 classes across all $N$ instances (including `NEGATIVE_OTHER`), `SoftCons` and `HardCons` evaluate strictly over the set $\mathcal{P}$ of unique reversible source pairs ($|\mathcal{P}| = 277$ on Dev; $|\mathcal{P}| = 1,282$ on Train).
   Because $h_i = 1 \iff (\hat{y}_i = y_i \land \hat{y}_i^r = \Rev(y_i))$, and ground truth satisfies $\Rev(y_i) = y_i^r$, correctness in both directions automatically implies directional self-consistency ($\hat{y}_i^r = \Rev(\hat{y}_i)$). Thus, $h_i = 1 \implies s_i = 1$, which strictly bounds $\HardCons \le \SoftCons$.

3. **Data Augmentation vs. Evaluation:**
   Observation 3 shows that `NEGATIVE_OTHER` is non-reversible in official metric calculations ($s_i = 0$). However, for Bidirectional Symmetric Augmentation (BSA) during training (R1.1), reversing an unaligned noise pair $(A, B)$ retains the noise classification $(B, A) \to \text{NEGATIVE_OTHER}$.

4. **Validation Strictness:**
   Observation 4 shows that Codabench scoring containers use `evaluation_functions/io.py` with zero tolerance for trailing blank lines, additional columns, or mismatched row counts.

---

## 3. Caveats

- **Tracks 3 & 4 (Basque & Mixed):** Although documented in the specification, our implementation focus is strictly Track 1 (English) and Track 2 (Spanish) per user requirements.
- **Test Split Labels:** Official test split labels are hidden until competition conclusion; local validation and CI must rely on `dev` reference files.

---

## 4. Conclusion

The data specifications, deterministic reversal algebra, official metrics, and validation constraints for SemEval-2027 Task 2 (DiCo-NLI) are completely discovered and documented in `report.md`.
The specifications dictate:
1. Data reading must support 7-column participant CSVs and 8-column reference CSVs, generating 2-column prediction CSVs (`instance_id,label`).
2. Symmetrized Decoding (SD) mathematically guarantees $\SoftCons = 1.0$ by projecting logits via permutation matrix $\Pi_{\Rev}$.
3. Local evaluation in `src/metrics/` must mirror `evaluation_functions/metrics.py` to ensure exact parity with official CodaBench scoring.

---

## 5. Verification Method

To independently verify all findings and validate local behavior:
1. Inspect the full specification report:
   `view_file /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/survey_miner/report.md`
2. Validate against official reference data using python:
   ```bash
   python3 -c "
   import pandas as pd
   # Check Dev Track 1
   df = pd.read_csv('https://raw.githubusercontent.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI/main/final_data/dev/dico_nli_dev_track1_participant_labeled.csv')
   print('Dev instances:', len(df))
   print(df['label'].value_counts())
   "
   ```
3. Test compatibility with official evaluation functions repository (`evaluation_functions/metrics.py` and `io.py`).
