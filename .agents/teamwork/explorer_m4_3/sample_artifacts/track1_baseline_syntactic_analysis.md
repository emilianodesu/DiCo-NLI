# Syntactic Category Error Analysis: BASELINE (TRACK1)

- **Method:** `baseline`
- **Benchmark Track:** `track1`
- **Total Instances:** 31
- **Total Reciprocal Pairs:** 15
- **Overall Weighted F1:** 0.1306
- **Overall SoftCons:** 0.0000
- **Overall HardCons:** 0.0000

## 1. Disaggregated Performance by Syntactic Chunk Category

| Category | Instances | Pairs | Weighted F1 | SoftCons | HardCons | Entailment Loops | Equiv Asym | RCSI (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Adverbial** | 4 | 2 | 0.0000 | 0.0000 | 0.0000 | 0 | 2 | 0.0% |
| **Nominal** | 7 | 3 | 0.1270 | 0.0000 | 0.0000 | 2 | 1 | 100.0% |
| **Prepositional** | 6 | 3 | 0.1667 | 0.0000 | 0.0000 | 2 | 1 | 100.0% |
| **Quantificational** | 6 | 3 | 0.1667 | 0.0000 | 0.0000 | 2 | 1 | 100.0% |
| **Verbal** | 8 | 4 | 0.2045 | 0.0000 | 0.0000 | 3 | 1 | 100.0% |

### Key Linguistic Vulnerability Findings
- **Most Vulnerable Category to Reversal Curse:** `nominal` (RCSI: 100.0%).
- **Secondary Vulnerability:** `prepositional` (RCSI: 100.0%).

## 2. Cross-Method Contrast Across Syntactic Categories

Comparative evaluation contrasting SoftCons across Reversal Curse mitigation methods:

| Category | **BASELINE SoftCons** | **BSA SoftCons** | **CRL SoftCons** | **SD SoftCons** | **CRL_SD SoftCons** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Adverbial** | 0.0000 | 0.5000 | 0.5000 | 1.0000 | 1.0000 |
| **Nominal** | 0.0000 | 0.6667 | 0.6667 | 1.0000 | 1.0000 |
| **Prepositional** | 0.0000 | 0.3333 | 1.0000 | 1.0000 | 1.0000 |
| **Quantificational** | 0.0000 | 0.3333 | 1.0000 | 1.0000 | 1.0000 |
| **Verbal** | 0.0000 | 0.2500 | 1.0000 | 1.0000 | 1.0000 |

- **Most improved category under Symmetrized Decoding:** `adverbial`.

## 3. Qualitative Error Case Diagnoses

Selected representative failure examples demonstrating directional inconsistency mechanisms:

### Case 1: [ADVERBIAL] Pair `dico_phrasis_2000004`
- **Phrase A:** "extremely rapidly"
- **Phrase B:** "very fast"
- **Ground Truth:** Direct: `EQUIVALENCE` | Reverse: `EQUIVALENCE`
- **Prediction (baseline):** Direct: `FORWARD_ENTAILMENT` | Reverse: `FORWARD_ENTAILMENT`
- **Consistency:** SoftCons: ❌ (`False`) | HardCons: ❌ (`False`)
- **Error Taxonomy:** `EQUIVALENCE_ASYMMETRY`
- **Diagnosis:** Equivalence Asymmetry in Adverbial Chunk: Ground truth relation is symmetric equivalence ('extremely rapidly' <=> 'very fast'), but model predicts asymmetric 'FORWARD_ENTAILMENT' forward and 'FORWARD_ENTAILMENT' backward, violating bidirectional equivalence invariance.

### Case 2: [ADVERBIAL] Pair `dico_phrasis_2000009`
- **Phrase A:** "with great care"
- **Phrase B:** "carefully"
- **Ground Truth:** Direct: `EQUIVALENCE` | Reverse: `EQUIVALENCE`
- **Prediction (baseline):** Direct: `FORWARD_ENTAILMENT` | Reverse: `FORWARD_ENTAILMENT`
- **Consistency:** SoftCons: ❌ (`False`) | HardCons: ❌ (`False`)
- **Error Taxonomy:** `EQUIVALENCE_ASYMMETRY`
- **Diagnosis:** Equivalence Asymmetry in Adverbial Chunk: Ground truth relation is symmetric equivalence ('with great care' <=> 'carefully'), but model predicts asymmetric 'FORWARD_ENTAILMENT' forward and 'FORWARD_ENTAILMENT' backward, violating bidirectional equivalence invariance.

### Case 3: [NOMINAL] Pair `dico_phrasis_2000001`
- **Phrase A:** "Pope Francis"
- **Phrase B:** "pope"
- **Ground Truth:** Direct: `FORWARD_ENTAILMENT` | Reverse: `BACKWARD_ENTAILMENT`
- **Prediction (baseline):** Direct: `FORWARD_ENTAILMENT` | Reverse: `FORWARD_ENTAILMENT`
- **Consistency:** SoftCons: ❌ (`False`) | HardCons: ❌ (`False`)
- **Error Taxonomy:** `REVERSAL_CURSE_LOOP`
- **Diagnosis:** Reversal Curse Entailment Loop in Nominal Chunk: Hypernymy hierarchy between 'Pope Francis' and 'pope' recognized unidirectionally, but inverted relation fails to invert to BACKWARD_ENTAILMENT.

### Case 4: [NOMINAL] Pair `dico_phrasis_2000006`
- **Phrase A:** "the prime minister"
- **Phrase B:** "the head of government"
- **Ground Truth:** Direct: `EQUIVALENCE` | Reverse: `EQUIVALENCE`
- **Prediction (baseline):** Direct: `FORWARD_ENTAILMENT` | Reverse: `FORWARD_ENTAILMENT`
- **Consistency:** SoftCons: ❌ (`False`) | HardCons: ❌ (`False`)
- **Error Taxonomy:** `EQUIVALENCE_ASYMMETRY`
- **Diagnosis:** Equivalence Asymmetry in Nominal Chunk: Ground truth relation is symmetric equivalence ('the prime minister' <=> 'the head of government'), but model predicts asymmetric 'FORWARD_ENTAILMENT' forward and 'FORWARD_ENTAILMENT' backward, violating bidirectional equivalence invariance.

### Case 5: [NOMINAL] Pair `dico_phrasis_2000011`
- **Phrase A:** "a young boy"
- **Phrase B:** "a child"
- **Ground Truth:** Direct: `FORWARD_ENTAILMENT` | Reverse: `BACKWARD_ENTAILMENT`
- **Prediction (baseline):** Direct: `FORWARD_ENTAILMENT` | Reverse: `FORWARD_ENTAILMENT`
- **Consistency:** SoftCons: ❌ (`False`) | HardCons: ❌ (`False`)
- **Error Taxonomy:** `REVERSAL_CURSE_LOOP`
- **Diagnosis:** Reversal Curse Entailment Loop in Nominal Chunk: Hypernymy hierarchy between 'a young boy' and 'a child' recognized unidirectionally, but inverted relation fails to invert to BACKWARD_ENTAILMENT.

### Case 6: [PREPOSITIONAL] Pair `dico_phrasis_2000003`
- **Phrase A:** "in the center of Madrid"
- **Phrase B:** "in Spain"
- **Ground Truth:** Direct: `FORWARD_ENTAILMENT` | Reverse: `BACKWARD_ENTAILMENT`
- **Prediction (baseline):** Direct: `FORWARD_ENTAILMENT` | Reverse: `FORWARD_ENTAILMENT`
- **Consistency:** SoftCons: ❌ (`False`) | HardCons: ❌ (`False`)
- **Error Taxonomy:** `REVERSAL_CURSE_LOOP`
- **Diagnosis:** Reversal Curse Entailment Loop in Prepositional Chunk: Model correctly captures spatial/containment hierarchy in forward direction ('in the center of Madrid' -> 'in Spain'), but repeats 'FORWARD_ENTAILMENT' under reversal ('in Spain' -> 'in the center of Madrid') due to unpenalized directional bias, failing to invert locative inclusion.

### Case 7: [PREPOSITIONAL] Pair `dico_phrasis_2000008`
- **Phrase A:** "with a tennis racket"
- **Phrase B:** "with sports equipment"
- **Ground Truth:** Direct: `FORWARD_ENTAILMENT` | Reverse: `BACKWARD_ENTAILMENT`
- **Prediction (baseline):** Direct: `FORWARD_ENTAILMENT` | Reverse: `FORWARD_ENTAILMENT`
- **Consistency:** SoftCons: ❌ (`False`) | HardCons: ❌ (`False`)
- **Error Taxonomy:** `REVERSAL_CURSE_LOOP`
- **Diagnosis:** Reversal Curse Entailment Loop in Prepositional Chunk: Model correctly captures spatial/containment hierarchy in forward direction ('with a tennis racket' -> 'with sports equipment'), but repeats 'FORWARD_ENTAILMENT' under reversal ('with sports equipment' -> 'with a tennis racket') due to unpenalized directional bias, failing to invert locative inclusion.

### Case 8: [PREPOSITIONAL] Pair `dico_phrasis_2000013`
- **Phrase A:** "at the present moment"
- **Phrase B:** "right now"
- **Ground Truth:** Direct: `EQUIVALENCE` | Reverse: `EQUIVALENCE`
- **Prediction (baseline):** Direct: `FORWARD_ENTAILMENT` | Reverse: `FORWARD_ENTAILMENT`
- **Consistency:** SoftCons: ❌ (`False`) | HardCons: ❌ (`False`)
- **Error Taxonomy:** `EQUIVALENCE_ASYMMETRY`
- **Diagnosis:** Equivalence Asymmetry in Prepositional Chunk: Ground truth relation is symmetric equivalence ('at the present moment' <=> 'right now'), but model predicts asymmetric 'FORWARD_ENTAILMENT' forward and 'FORWARD_ENTAILMENT' backward, violating bidirectional equivalence invariance.

### Case 9: [QUANTIFICATIONAL] Pair `dico_phrasis_2000005`
- **Phrase A:** "nine energetic children"
- **Phrase B:** "several children"
- **Ground Truth:** Direct: `FORWARD_ENTAILMENT` | Reverse: `BACKWARD_ENTAILMENT`
- **Prediction (baseline):** Direct: `FORWARD_ENTAILMENT` | Reverse: `FORWARD_ENTAILMENT`
- **Consistency:** SoftCons: ❌ (`False`) | HardCons: ❌ (`False`)
- **Error Taxonomy:** `REVERSAL_CURSE_LOOP`
- **Diagnosis:** Reversal Curse Entailment Loop in Quantificational Chunk: Monotonicity entailment from specific to general bound correctly identified, but model fails to reverse the quantifier entailment arrow under argument reversal.

### Case 10: [QUANTIFICATIONAL] Pair `dico_phrasis_2000010`
- **Phrase A:** "all registered voters"
- **Phrase B:** "every voter"
- **Ground Truth:** Direct: `EQUIVALENCE` | Reverse: `EQUIVALENCE`
- **Prediction (baseline):** Direct: `FORWARD_ENTAILMENT` | Reverse: `FORWARD_ENTAILMENT`
- **Consistency:** SoftCons: ❌ (`False`) | HardCons: ❌ (`False`)
- **Error Taxonomy:** `EQUIVALENCE_ASYMMETRY`
- **Diagnosis:** Equivalence Asymmetry in Quantificational Chunk: Ground truth relation is symmetric equivalence ('all registered voters' <=> 'every voter'), but model predicts asymmetric 'FORWARD_ENTAILMENT' forward and 'FORWARD_ENTAILMENT' backward, violating bidirectional equivalence invariance.

### Case 11: [QUANTIFICATIONAL] Pair `dico_phrasis_2000015`
- **Phrase A:** "everyone attended"
- **Phrase B:** "someone attended"
- **Ground Truth:** Direct: `FORWARD_ENTAILMENT` | Reverse: `BACKWARD_ENTAILMENT`
- **Prediction (baseline):** Direct: `FORWARD_ENTAILMENT` | Reverse: `FORWARD_ENTAILMENT`
- **Consistency:** SoftCons: ❌ (`False`) | HardCons: ❌ (`False`)
- **Error Taxonomy:** `REVERSAL_CURSE_LOOP`
- **Diagnosis:** Reversal Curse Entailment Loop in Quantificational Chunk: Monotonicity entailment from specific to general bound correctly identified, but model fails to reverse the quantifier entailment arrow under argument reversal.

### Case 12: [VERBAL] Pair `dico_phrasis_2000002`
- **Phrase A:** "is sprinting across the field"
- **Phrase B:** "is running"
- **Ground Truth:** Direct: `FORWARD_ENTAILMENT` | Reverse: `BACKWARD_ENTAILMENT`
- **Prediction (baseline):** Direct: `FORWARD_ENTAILMENT` | Reverse: `FORWARD_ENTAILMENT`
- **Consistency:** SoftCons: ❌ (`False`) | HardCons: ❌ (`False`)
- **Error Taxonomy:** `REVERSAL_CURSE_LOOP`
- **Diagnosis:** Reversal Curse Entailment Loop in Verbal Chunk: Model detects event subsumption ('is sprinting across the field' -> 'is running'), but high lexical overlap causes spurious forward entailment under reversal ('is running' -> 'is sprinting across the field'), failing to recognize that hyponymic verbs do not reverse.

### Case 13: [VERBAL] Pair `dico_phrasis_2000007`
- **Phrase A:** "to destroy"
- **Phrase B:** "to demolish"
- **Ground Truth:** Direct: `EQUIVALENCE` | Reverse: `EQUIVALENCE`
- **Prediction (baseline):** Direct: `FORWARD_ENTAILMENT` | Reverse: `FORWARD_ENTAILMENT`
- **Consistency:** SoftCons: ❌ (`False`) | HardCons: ❌ (`False`)
- **Error Taxonomy:** `EQUIVALENCE_ASYMMETRY`
- **Diagnosis:** Equivalence Asymmetry in Verbal Chunk: Ground truth relation is symmetric equivalence ('to destroy' <=> 'to demolish'), but model predicts asymmetric 'FORWARD_ENTAILMENT' forward and 'FORWARD_ENTAILMENT' backward, violating bidirectional equivalence invariance.

### Case 14: [VERBAL] Pair `dico_phrasis_2000012`
- **Phrase A:** "are dancing joyfully"
- **Phrase B:** "are moving"
- **Ground Truth:** Direct: `FORWARD_ENTAILMENT` | Reverse: `BACKWARD_ENTAILMENT`
- **Prediction (baseline):** Direct: `FORWARD_ENTAILMENT` | Reverse: `FORWARD_ENTAILMENT`
- **Consistency:** SoftCons: ❌ (`False`) | HardCons: ❌ (`False`)
- **Error Taxonomy:** `REVERSAL_CURSE_LOOP`
- **Diagnosis:** Reversal Curse Entailment Loop in Verbal Chunk: Model detects event subsumption ('are dancing joyfully' -> 'are moving'), but high lexical overlap causes spurious forward entailment under reversal ('are moving' -> 'are dancing joyfully'), failing to recognize that hyponymic verbs do not reverse.
