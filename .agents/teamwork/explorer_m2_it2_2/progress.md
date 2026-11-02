# Progress — M2 Iteration 2 Explorer 2

- **Status**: Investigation Complete & Prototype Verified
- **Last visited**: 2026-11-01T23:12:30Z
- **Current Step**: Authoring detailed report and handoff
- **Summary**:
  1. Identified root cause in `src/models/siamese_crl.py`:
     - Line 239: `if apply_symmetrized_decoding or not self.training:` forced Symmetrized Decoding during evaluation mode regardless of `apply_symmetrized_decoding=False`.
     - Line 391 (pure Python): unconditionally ran Symmetrized Decoding without checking `apply_symmetrized_decoding`.
     - `SiameseCRLModel` lacked `predict_pair`, `predict_proba`, `predict_logits`, and `predict` methods, breaking both standalone pair inference and `build_model(method='crl_sd')` wrapper delegation.
  2. Verified proposed solution via dynamic Python prototype:
     - Pure CRL evaluation returns `sym_probs_dir = None`, `sym_probs_rev = None`, and raw argmax discrete predictions.
     - Symmetrized Decoding projection is only activated when `apply_symmetrized_decoding=True` (or model config).
     - Full high-level inference methods added to both PyTorch and Pure-Python versions of `SiameseCRLModel`.
     - Tested compatibility with factory methods `method="crl"` and `method="crl_sd"`.
