# Syntactic Category Error Analysis: CRL_SD (TRACK2)

- **Method:** `crl_sd`
- **Benchmark Track:** `track2`
- **Total Instances:** 31
- **Total Reciprocal Pairs:** 15
- **Overall Weighted F1:** 0.8896
- **Overall SoftCons:** 1.0000
- **Overall HardCons:** 0.9333

## 1. Disaggregated Performance by Syntactic Chunk Category

| Category | Instances | Pairs | Weighted F1 | SoftCons | HardCons | Entailment Loops | Equiv Asym | RCSI (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Adverbial** | 4 | 2 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0.0% |
| **Nominal** | 7 | 3 | 0.8000 | 1.0000 | 1.0000 | 0 | 0 | 0.0% |
| **Prepositional** | 6 | 3 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0.0% |
| **Quantificational** | 6 | 3 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0.0% |
| **Verbal** | 8 | 4 | 0.7500 | 1.0000 | 0.7500 | 0 | 0 | 0.0% |

### Key Linguistic Vulnerability Findings
- **Most Vulnerable Category to Reversal Curse:** `adverbial` (RCSI: 0.0%).
- **Secondary Vulnerability:** `nominal` (RCSI: 0.0%).

## 3. Qualitative Error Case Diagnoses

Selected representative failure examples demonstrating directional inconsistency mechanisms:

### Case 1: [VERBAL] Pair `dico_phrasis_2000014`
- **Phrase A:** "caminando muy despacio"
- **Phrase B:** "caminando despacio"
- **Ground Truth:** Direct: `FORWARD_ENTAILMENT` | Reverse: `BACKWARD_ENTAILMENT`
- **Prediction (crl_sd):** Direct: `BACKWARD_ENTAILMENT` | Reverse: `FORWARD_ENTAILMENT`
- **Consistency:** SoftCons: ✅ (`True`) | HardCons: ❌ (`False`)
- **Error Taxonomy:** `DIRECTIONAL_FLIP`
- **Diagnosis:** Directional Entailment Flip in Verbal Chunk: Model inverted the directional entailment arrow (predicted 'BACKWARD_ENTAILMENT' for gold 'FORWARD_ENTAILMENT').
