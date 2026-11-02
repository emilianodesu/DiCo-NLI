# Progress — explorer_m2_2

- Last visited: 2026-11-01T22:50:00Z
- Status: Investigation and mathematical verification complete. Blueprint report and handoff drafting in progress.
- Findings:
  1. Involution permutation matrix P is an orthogonal symmetric involution ($P^T = P = P^{-1}, P^2 = I$) swapping indices 0 and 2.
  2. Pure-Python and PyTorch implementations verified for KL, JSD, CRL loss, and Symmetrized Decoding across 1000 configurations with 100% SoftCons invariance.
  3. Reversal curse penalty confirmed: inconsistent predictions trigger >300x larger consistency regularization loss compared to consistent predictions.
  4. Integration contracts with `src/data/reversal.py` (`PairedReversalDataset`) and `src/metrics/metrics.py` verified.
