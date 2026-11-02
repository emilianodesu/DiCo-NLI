# Syntactic Category Error Analysis: CRL_SD (TRACK1)

- **Method:** `crl_sd`
- **Benchmark Track:** `track1`
- **Total Instances:** 31
- **Total Reciprocal Pairs:** 15
- **Overall Weighted F1:** 0.2535
- **Overall SoftCons:** 1.0000
- **Overall HardCons:** 0.4000

## 1. Disaggregated Performance by Syntactic Chunk Category

| Category | Instances | Pairs | Weighted F1 | SoftCons | HardCons | Entailment Loops | Equiv Asym | RCSI (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Adverbial** | 4 | 2 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0.0% |
| **Nominal** | 7 | 3 | 0.2857 | 1.0000 | 0.3333 | 0 | 0 | 0.0% |
| **Prepositional** | 6 | 3 | 0.1667 | 1.0000 | 0.3333 | 0 | 0 | 0.0% |
| **Quantificational** | 6 | 3 | 0.1667 | 1.0000 | 0.3333 | 0 | 0 | 0.0% |
| **Verbal** | 8 | 4 | 0.1000 | 1.0000 | 0.2500 | 0 | 0 | 0.0% |

### Key Linguistic Vulnerability Findings
- **Most Vulnerable Category to Reversal Curse:** `adverbial` (RCSI: 0.0%).
- **Secondary Vulnerability:** `nominal` (RCSI: 0.0%).

## 3. Qualitative Error Case Diagnoses

Selected representative failure examples demonstrating directional inconsistency mechanisms:

### Case 1: [NOMINAL] Pair `dico_phrasis_2000001`
- **Phrase A:** "Pope Francis"
- **Phrase B:** "pope"
- **Ground Truth:** Direct: `FORWARD_ENTAILMENT` | Reverse: `BACKWARD_ENTAILMENT`
- **Prediction (crl_sd):** Direct: `EQUIVALENCE` | Reverse: `EQUIVALENCE`
- **Consistency:** SoftCons: ✅ (`True`) | HardCons: ❌ (`False`)
- **Error Taxonomy:** `GENERAL_MISCLASSIFICATION`
- **Diagnosis:** General Misclassification in Nominal Chunk: Expected FORWARD_ENTAILMENT / BACKWARD_ENTAILMENT, predicted EQUIVALENCE / EQUIVALENCE.

### Case 2: [NOMINAL] Pair `dico_phrasis_2000011`
- **Phrase A:** "a young boy"
- **Phrase B:** "a child"
- **Ground Truth:** Direct: `FORWARD_ENTAILMENT` | Reverse: `BACKWARD_ENTAILMENT`
- **Prediction (crl_sd):** Direct: `EQUIVALENCE` | Reverse: `EQUIVALENCE`
- **Consistency:** SoftCons: ✅ (`True`) | HardCons: ❌ (`False`)
- **Error Taxonomy:** `GENERAL_MISCLASSIFICATION`
- **Diagnosis:** General Misclassification in Nominal Chunk: Expected FORWARD_ENTAILMENT / BACKWARD_ENTAILMENT, predicted EQUIVALENCE / EQUIVALENCE.

### Case 3: [PREPOSITIONAL] Pair `dico_phrasis_2000003`
- **Phrase A:** "in the center of Madrid"
- **Phrase B:** "in Spain"
- **Ground Truth:** Direct: `FORWARD_ENTAILMENT` | Reverse: `BACKWARD_ENTAILMENT`
- **Prediction (crl_sd):** Direct: `EQUIVALENCE` | Reverse: `EQUIVALENCE`
- **Consistency:** SoftCons: ✅ (`True`) | HardCons: ❌ (`False`)
- **Error Taxonomy:** `GENERAL_MISCLASSIFICATION`
- **Diagnosis:** General Misclassification in Prepositional Chunk: Expected FORWARD_ENTAILMENT / BACKWARD_ENTAILMENT, predicted EQUIVALENCE / EQUIVALENCE.

### Case 4: [PREPOSITIONAL] Pair `dico_phrasis_2000008`
- **Phrase A:** "with a tennis racket"
- **Phrase B:** "with sports equipment"
- **Ground Truth:** Direct: `FORWARD_ENTAILMENT` | Reverse: `BACKWARD_ENTAILMENT`
- **Prediction (crl_sd):** Direct: `EQUIVALENCE` | Reverse: `EQUIVALENCE`
- **Consistency:** SoftCons: ✅ (`True`) | HardCons: ❌ (`False`)
- **Error Taxonomy:** `GENERAL_MISCLASSIFICATION`
- **Diagnosis:** General Misclassification in Prepositional Chunk: Expected FORWARD_ENTAILMENT / BACKWARD_ENTAILMENT, predicted EQUIVALENCE / EQUIVALENCE.

### Case 5: [QUANTIFICATIONAL] Pair `dico_phrasis_2000005`
- **Phrase A:** "nine energetic children"
- **Phrase B:** "several children"
- **Ground Truth:** Direct: `FORWARD_ENTAILMENT` | Reverse: `BACKWARD_ENTAILMENT`
- **Prediction (crl_sd):** Direct: `EQUIVALENCE` | Reverse: `EQUIVALENCE`
- **Consistency:** SoftCons: ✅ (`True`) | HardCons: ❌ (`False`)
- **Error Taxonomy:** `GENERAL_MISCLASSIFICATION`
- **Diagnosis:** General Misclassification in Quantificational Chunk: Expected FORWARD_ENTAILMENT / BACKWARD_ENTAILMENT, predicted EQUIVALENCE / EQUIVALENCE.

### Case 6: [QUANTIFICATIONAL] Pair `dico_phrasis_2000015`
- **Phrase A:** "everyone attended"
- **Phrase B:** "someone attended"
- **Ground Truth:** Direct: `FORWARD_ENTAILMENT` | Reverse: `BACKWARD_ENTAILMENT`
- **Prediction (crl_sd):** Direct: `EQUIVALENCE` | Reverse: `EQUIVALENCE`
- **Consistency:** SoftCons: ✅ (`True`) | HardCons: ❌ (`False`)
- **Error Taxonomy:** `GENERAL_MISCLASSIFICATION`
- **Diagnosis:** General Misclassification in Quantificational Chunk: Expected FORWARD_ENTAILMENT / BACKWARD_ENTAILMENT, predicted EQUIVALENCE / EQUIVALENCE.

### Case 7: [VERBAL] Pair `dico_phrasis_2000002`
- **Phrase A:** "is sprinting across the field"
- **Phrase B:** "is running"
- **Ground Truth:** Direct: `FORWARD_ENTAILMENT` | Reverse: `BACKWARD_ENTAILMENT`
- **Prediction (crl_sd):** Direct: `EQUIVALENCE` | Reverse: `EQUIVALENCE`
- **Consistency:** SoftCons: ✅ (`True`) | HardCons: ❌ (`False`)
- **Error Taxonomy:** `GENERAL_MISCLASSIFICATION`
- **Diagnosis:** General Misclassification in Verbal Chunk: Expected FORWARD_ENTAILMENT / BACKWARD_ENTAILMENT, predicted EQUIVALENCE / EQUIVALENCE.

### Case 8: [VERBAL] Pair `dico_phrasis_2000012`
- **Phrase A:** "are dancing joyfully"
- **Phrase B:** "are moving"
- **Ground Truth:** Direct: `FORWARD_ENTAILMENT` | Reverse: `BACKWARD_ENTAILMENT`
- **Prediction (crl_sd):** Direct: `EQUIVALENCE` | Reverse: `EQUIVALENCE`
- **Consistency:** SoftCons: ✅ (`True`) | HardCons: ❌ (`False`)
- **Error Taxonomy:** `GENERAL_MISCLASSIFICATION`
- **Diagnosis:** General Misclassification in Verbal Chunk: Expected FORWARD_ENTAILMENT / BACKWARD_ENTAILMENT, predicted EQUIVALENCE / EQUIVALENCE.

### Case 9: [VERBAL] Pair `dico_phrasis_2000014`
- **Phrase A:** "walking quite slowly"
- **Phrase B:** "walking slowly"
- **Ground Truth:** Direct: `FORWARD_ENTAILMENT` | Reverse: `BACKWARD_ENTAILMENT`
- **Prediction (crl_sd):** Direct: `EQUIVALENCE` | Reverse: `EQUIVALENCE`
- **Consistency:** SoftCons: ✅ (`True`) | HardCons: ❌ (`False`)
- **Error Taxonomy:** `GENERAL_MISCLASSIFICATION`
- **Diagnosis:** General Misclassification in Verbal Chunk: Expected FORWARD_ENTAILMENT / BACKWARD_ENTAILMENT, predicted EQUIVALENCE / EQUIVALENCE.
