# Syntactic Category Error Analysis: CRL (TRACK2)

- **Method:** `crl`
- **Benchmark Track:** `track2`
- **Total Instances:** 31
- **Total Reciprocal Pairs:** 15
- **Overall Weighted F1:** 0.8260
- **Overall SoftCons:** 0.7333
- **Overall HardCons:** 0.7333

## 1. Disaggregated Performance by Syntactic Chunk Category

| Category | Instances | Pairs | Weighted F1 | SoftCons | HardCons | Entailment Loops | Equiv Asym | RCSI (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Adverbial** | 4 | 2 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0.0% |
| **Nominal** | 7 | 3 | 0.8000 | 1.0000 | 1.0000 | 0 | 0 | 0.0% |
| **Prepositional** | 6 | 3 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0.0% |
| **Quantificational** | 6 | 3 | 0.5556 | 0.3333 | 0.3333 | 2 | 0 | 100.0% |
| **Verbal** | 8 | 4 | 0.7500 | 0.5000 | 0.5000 | 2 | 0 | 66.7% |

### Key Linguistic Vulnerability Findings
- **Most Vulnerable Category to Reversal Curse:** `quantificational` (RCSI: 100.0%).
- **Secondary Vulnerability:** `verbal` (RCSI: 66.7%).

## 3. Qualitative Error Case Diagnoses

Selected representative failure examples demonstrating directional inconsistency mechanisms:

### Case 1: [QUANTIFICATIONAL] Pair `dico_phrasis_2000005`
- **Phrase A:** "nueve niños entusiastas"
- **Phrase B:** "varios niños"
- **Ground Truth:** Direct: `FORWARD_ENTAILMENT` | Reverse: `BACKWARD_ENTAILMENT`
- **Prediction (crl):** Direct: `FORWARD_ENTAILMENT` | Reverse: `FORWARD_ENTAILMENT`
- **Consistency:** SoftCons: ❌ (`False`) | HardCons: ❌ (`False`)
- **Error Taxonomy:** `REVERSAL_CURSE_LOOP`
- **Diagnosis:** Reversal Curse Entailment Loop in Quantificational Chunk: Monotonicity entailment from specific to general bound correctly identified, but model fails to reverse the quantifier entailment arrow under argument reversal.

### Case 2: [QUANTIFICATIONAL] Pair `dico_phrasis_2000015`
- **Phrase A:** "todos asistieron"
- **Phrase B:** "alguien asistió"
- **Ground Truth:** Direct: `FORWARD_ENTAILMENT` | Reverse: `BACKWARD_ENTAILMENT`
- **Prediction (crl):** Direct: `FORWARD_ENTAILMENT` | Reverse: `FORWARD_ENTAILMENT`
- **Consistency:** SoftCons: ❌ (`False`) | HardCons: ❌ (`False`)
- **Error Taxonomy:** `REVERSAL_CURSE_LOOP`
- **Diagnosis:** Reversal Curse Entailment Loop in Quantificational Chunk: Monotonicity entailment from specific to general bound correctly identified, but model fails to reverse the quantifier entailment arrow under argument reversal.

### Case 3: [VERBAL] Pair `dico_phrasis_2000012`
- **Phrase A:** "están bailando con alegría"
- **Phrase B:** "se están moviendo"
- **Ground Truth:** Direct: `FORWARD_ENTAILMENT` | Reverse: `BACKWARD_ENTAILMENT`
- **Prediction (crl):** Direct: `FORWARD_ENTAILMENT` | Reverse: `FORWARD_ENTAILMENT`
- **Consistency:** SoftCons: ❌ (`False`) | HardCons: ❌ (`False`)
- **Error Taxonomy:** `REVERSAL_CURSE_LOOP`
- **Diagnosis:** Reversal Curse Entailment Loop in Verbal Chunk: Model detects event subsumption ('están bailando con alegría' -> 'se están moviendo'), but high lexical overlap causes spurious forward entailment under reversal ('se están moviendo' -> 'están bailando con alegría'), failing to recognize that hyponymic verbs do not reverse.

### Case 4: [VERBAL] Pair `dico_phrasis_2000014`
- **Phrase A:** "caminando muy despacio"
- **Phrase B:** "caminando despacio"
- **Ground Truth:** Direct: `FORWARD_ENTAILMENT` | Reverse: `BACKWARD_ENTAILMENT`
- **Prediction (crl):** Direct: `BACKWARD_ENTAILMENT` | Reverse: `BACKWARD_ENTAILMENT`
- **Consistency:** SoftCons: ❌ (`False`) | HardCons: ❌ (`False`)
- **Error Taxonomy:** `REVERSAL_CURSE_LOOP`
- **Diagnosis:** Reversal Curse Entailment Loop in Verbal Chunk: Model detects event subsumption ('caminando muy despacio' -> 'caminando despacio'), but high lexical overlap causes spurious forward entailment under reversal ('caminando despacio' -> 'caminando muy despacio'), failing to recognize that hyponymic verbs do not reverse.
