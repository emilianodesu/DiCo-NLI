# BRIEFING — 2026-11-01T22:50:00Z

## Mission
Design the implementation blueprint for `src/models/losses.py` and `src/models/siamese_crl.py` (losses, involution permutation, dual-stream forward/eval).

## 🔒 My Identity
- Archetype: explorer
- Roles: investigator, architect, synthesizer
- Working directory: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m2_2
- Original parent: df6ce377-6407-4a2d-bd18-829b36e20607
- Milestone: M2 - Modeling & CRL

## 🔒 Key Constraints
- Read-only investigation — do NOT implement directly in `src/`, produce comprehensive blueprint report and handoff.
- Write only to `.agents/teamwork/explorer_m2_2/`.
- Ensure mathematical rigor and exact PyTorch tensor implementations for involution matrix, KL/JS divergence, numerical stability (log_softmax, clamp, epsilon), and batch handling.

## Current Parent
- Conversation ID: df6ce377-6407-4a2d-bd18-829b36e20607
- Updated: 2026-11-01T22:50:00Z

## Investigation State
- **Explored paths**:
  - `ORIGINAL_REQUEST.md`, `PROJECT.md`, `survey_modeling/report.md`
  - `src/data/reversal.py` (examined `PairedReversalDataset` keys: `dir_input_ids`, `inv_input_ids`, etc.)
  - `src/metrics/metrics.py` (examined `LABEL_TO_IDX`, `INVOLUTION_PERMUTATION_MATRIX`)
  - `tests/test_symmetrized_decoding.py`, `tests/test_pipeline.py`
  - Executed mathematical verification in `.agents/teamwork/explorer_m2_2/test_blueprint.py` (all 7 tests passed).
- **Key findings**:
  - Class index encoding: 0: BACKWARD_ENTAILMENT, 1: EQUIVALENCE, 2: FORWARD_ENTAILMENT, 3: NEGATIVE_OTHER.
  - Permutation matrix $P$ is an orthogonal symmetric involution ($P = P^T = P^{-1}, P^2 = I$) mapping $0 \leftrightarrow 2$, preserving 1 and 3.
  - Symmetrized Decoding strictly achieves SoftCons = 100% across all probability distributions.
  - JSD is bounded in $[0, \ln 2]$, strictly symmetric, and robust against zero probabilities.
  - CRL loss creates a 300x penalty surge when both directions predict forward entailment (reversal curse).
- **Unexplored areas**: None for M2 Explorer 2 scope.

## Key Decisions Made
- `src/models/losses.py`:
  - Provide `P_MATRIX` and `TranspositionOperator` (registering buffer $P$).
  - Provide `SymmetricKLDivergenceLoss` with `scale=0.5`, log-space calculation, and `eps=1e-8`.
  - Provide `JensenShannonDivergenceLoss` with numerical clamping.
  - Provide `ConsistencyRegularizationLoss` with dynamic linear/cosine lambda warmup, supervised CE, and diagnostic metrics dictionary.
  - Provide pure-Python fallbacks for mathematical operations when PyTorch is not available.
- `src/models/siamese_crl.py`:
  - `SiameseCRLModel` wrapping shared encoder.
  - Dual stream forward pass with both sequential and concatenated ($2B$) options.
  - Supports input keys matching `PairedReversalDataset` (`dir_*`, `inv_*`) and standard NLI names (`*_dir`, `*_rev`).
  - Evaluates with Symmetrized Decoding for SoftCons = 100%.
  - Supports both Transformer backbones and fast CPU mock backbones.

## Artifact Index
- DISPATCH.md — incoming dispatch instructions
- BRIEFING.md — persistent situational awareness
- progress.md — liveness heartbeat
- test_blueprint.py — verified pure-Python/PyTorch mathematical proof and simulation script
- report.md — comprehensive blueprint report
- handoff.md — 5-component handoff report
