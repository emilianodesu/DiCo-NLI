# Syntactic Category Error Analysis: BSA (TRACK2)

- **Method:** `bsa`
- **Benchmark Track:** `track2`
- **Total Instances:** 31
- **Total Reciprocal Pairs:** 15
- **Overall Weighted F1:** 0.8569
- **Overall SoftCons:** 0.8000
- **Overall HardCons:** 0.8000

## 1. Disaggregated Performance by Syntactic Chunk Category

| Category | Instances | Pairs | Weighted F1 | SoftCons | HardCons | Entailment Loops | Equiv Asym | RCSI (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Adverbial** | 4 | 2 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0.0% |
| **Nominal** | 7 | 3 | 0.6667 | 0.6667 | 0.6667 | 1 | 0 | 50.0% |
| **Prepositional** | 6 | 3 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0.0% |
| **Quantificational** | 6 | 3 | 0.8222 | 0.6667 | 0.6667 | 1 | 0 | 50.0% |
| **Verbal** | 8 | 4 | 0.8714 | 0.7500 | 0.7500 | 1 | 0 | 33.3% |

### Key Linguistic Vulnerability Findings
- **Most Vulnerable Category to Reversal Curse:** `nominal` (RCSI: 50.0%).
- **Secondary Vulnerability:** `quantificational` (RCSI: 50.0%).

## 3. Qualitative Error Case Diagnoses

Selected representative failure examples demonstrating directional inconsistency mechanisms:

### Case 1: [NOMINAL] Pair `dico_phrasis_2000011`
- **Phrase A:** "un niño pequeño"
- **Phrase B:** "un menor de edad"
- **Ground Truth:** Direct: `FORWARD_ENTAILMENT` | Reverse: `BACKWARD_ENTAILMENT`
- **Prediction (bsa):** Direct: `BACKWARD_ENTAILMENT` | Reverse: `BACKWARD_ENTAILMENT`
- **Consistency:** SoftCons: ❌ (`False`) | HardCons: ❌ (`False`)
- **Error Taxonomy:** `REVERSAL_CURSE_LOOP`
- **Diagnosis:** Reversal Curse Entailment Loop in Nominal Chunk: Hypernymy hierarchy between 'un niño pequeño' and 'un menor de edad' recognized unidirectionally, but inverted relation fails to invert to BACKWARD_ENTAILMENT.

### Case 2: [QUANTIFICATIONAL] Pair `dico_phrasis_2000015`
- **Phrase A:** "todos asistieron"
- **Phrase B:** "alguien asistió"
- **Ground Truth:** Direct: `FORWARD_ENTAILMENT` | Reverse: `BACKWARD_ENTAILMENT`
- **Prediction (bsa):** Direct: `BACKWARD_ENTAILMENT` | Reverse: `BACKWARD_ENTAILMENT`
- **Consistency:** SoftCons: ❌ (`False`) | HardCons: ❌ (`False`)
- **Error Taxonomy:** `REVERSAL_CURSE_LOOP`
- **Diagnosis:** Reversal Curse Entailment Loop in Quantificational Chunk: Monotonicity entailment from specific to general bound correctly identified, but model fails to reverse the quantifier entailment arrow under argument reversal.

### Case 3: [VERBAL] Pair `dico_phrasis_2000014`
- **Phrase A:** "caminando muy despacio"
- **Phrase B:** "caminando despacio"
- **Ground Truth:** Direct: `FORWARD_ENTAILMENT` | Reverse: `BACKWARD_ENTAILMENT`
- **Prediction (bsa):** Direct: `BACKWARD_ENTAILMENT` | Reverse: `BACKWARD_ENTAILMENT`
- **Consistency:** SoftCons: ❌ (`False`) | HardCons: ❌ (`False`)
- **Error Taxonomy:** `REVERSAL_CURSE_LOOP`
- **Diagnosis:** Reversal Curse Entailment Loop in Verbal Chunk: Model detects event subsumption ('caminando muy despacio' -> 'caminando despacio'), but high lexical overlap causes spurious forward entailment under reversal ('caminando despacio' -> 'caminando muy despacio'), failing to recognize that hyponymic verbs do not reverse.
