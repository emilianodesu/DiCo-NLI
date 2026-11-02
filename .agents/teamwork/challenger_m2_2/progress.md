# Progress — Challenger 2 (Milestone 2)

Last visited: 2026-11-01T23:05:00Z
Status: Completed all empirical challenges. Documenting findings and writing handoff report.

- [x] Initialized DISPATCH.md, BRIEFING.md, progress.md
- [x] Read ORIGINAL_REQUEST.md, PROJECT.md, and worker_m2/handoff.md
- [x] Inspect source code of strategies, factory, and tests
- [x] Run full project test suite (`python3 -m unittest discover -s tests -p "test_*.py" -v` -> 73 tests passed in 0.064s)
- [x] Write and run empirical test: 25,000 random Dirichlet probability distributions for SoftCons = 1.0 (100% verified)
- [x] Write and run stress tests: extreme logits, zeros, identical distributions, tied maximum probabilities, equivariant tie-breaking (100% verified)
- [x] Write and run factory method loading tests across baseline, bsa, crl, sd, crl_sd:
  - Discovered Bug 1: `AttributeError` on `model.eval()` / `model.train()` for `CrossEncoderNLI` without PyTorch.
  - Discovered Bug 2: `NotImplementedError` on `predict_pair()` for `build_model(method='crl_sd')`.
  - Discovered Bug 3: `SiameseCRLModel.forward()` cannot disable Symmetrized Decoding during evaluation (`apply_symmetrized_decoding=False` ignored).
  - Discovered Bug 4: `mask_negative_other=False` default causes `SoftCons = 0.0` on reversible pairs if `NEGATIVE_OTHER` is predicted.
- [ ] Compile challenge report and handoff.md with verdict REQUEST_CHANGES
- [ ] Send verdict to orchestrator
