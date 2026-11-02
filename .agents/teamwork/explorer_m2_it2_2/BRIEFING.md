# BRIEFING — 2026-11-01T23:12:40Z

## Mission
Investigate Siamese CRL ablation and Symmetrized Decoding control in `src/models/siamese_crl.py`, ensuring clean separation between pure CRL evaluation (raw probabilities) and test-time Symmetrized Decoding, and blueprint a precise code patch for Worker.

## 🔒 My Identity
- Archetype: Teamwork explorer
- Roles: Read-only investigator, synthesizer
- Working directory: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m2_it2_2
- Original parent: df6ce377-6407-4a2d-bd18-829b36e20607
- Milestone: M2 Iteration 2

## 🔒 Key Constraints
- Read-only investigation — do NOT implement directly in source tree (Worker implements).
- Adhere strictly to project architecture and Handoff protocol.
- Respect `apply_symmetrized_decoding` flag in `SiameseCRLModel.forward` and `predict_pair`.

## Current Parent
- Conversation ID: df6ce377-6407-4a2d-bd18-829b36e20607
- Updated: 2026-11-01T23:12:40Z

## Investigation State
- **Explored paths**:
  - `src/models/siamese_crl.py` (lines 95-274 PyTorch & lines 286-407 Pure-Python)
  - `src/models/factory.py` (`build_model`, `SymmetrizedModelWrapper`)
  - `src/models/cross_encoder.py` (`CrossEncoderNLI`, `MockCrossEncoder`)
  - `src/models/decoding.py` (`symmetrized_decoding`, `SymmetrizedDecoder`)
  - `tests/test_models.py` (unit tests for CRL, SD, Factory)
  - `challenger_m2_2/handoff.md`, `reviewer_m2_1/handoff.md`
- **Key findings**:
  - Bug in PyTorch forward (line 239): `apply_symmetrized_decoding or not self.training` unconditionally forced SD during `model.eval()`.
  - Bug in Pure-Python forward (line 391): unconditional execution of `symmetrized_decoding` without checking `apply_symmetrized_decoding`.
  - Missing methods on `SiameseCRLModel`: lacks `predict_pair`, `predict_proba`, `predict_logits`, and `predict`. This caused `build_model(method='crl_sd').predict_pair(...)` to crash with `NotImplementedError`, and prevented direct inference on `SiameseCRLModel`.
  - Verified prototype: pure CRL returns `sym_probs_dir = None`, `sym_probs_rev = None`, and raw unprojected predictions; SD is active only when requested.
- **Unexplored areas**:
  - None within Explorer 2 scope. Explorer 1 handles Mock `eval`/`train` and wrapper delegation. Explorer 3 handles `cross_entropy_pure` `target=-100`.

## Key Decisions Made
- `SiameseCRLModel.__init__` will accept `apply_symmetrized_decoding: bool = False` (defaulting to pure CRL evaluation) and `mask_negative_other: bool = False`.
- `SiameseCRLModel.forward` will evaluate `should_apply_sd = apply_symmetrized_decoding if apply_symmetrized_decoding is not None else getattr(self, "apply_symmetrized_decoding", False)`.
- When `should_apply_sd` is `False`, `sym_probs_dir` and `sym_probs_rev` will be strictly `None`, and predictions are standard argmax over logits/probabilities.
- Add `predict_logits`, `predict_proba`, `predict`, `predict_pair`, `decode_pair`, `decode_pairs` to both PyTorch and Pure-Python versions of `SiameseCRLModel`.
- Configure `build_model` in `factory.py` so `method="crl"` explicitly creates `SiameseCRLModel(..., apply_symmetrized_decoding=False)` and `method="crl_sd"` creates `SiameseCRLModel(..., apply_symmetrized_decoding=True)`.

## Artifact Index
- DISPATCH.md — incoming dispatch instructions
- progress.md — liveness heartbeat
- report.md — comprehensive investigation report and patch blueprint
- handoff.md — formal 5-component handoff report
