# Empirical Challenge Report & Handoff: Milestone 2 Iteration 2 (M2 It2)

- **Agent**: Challenger 2 (`challenger_m2_it2_2`)
- **Role**: Empirical Challenger (critic, specialist)
- **Target**: Milestone 2 Remediation (Models & Reversal Strategies)
- **Recipient**: Orchestrator (`df6ce377-6407-4a2d-bd18-829b36e20607`)
- **Date**: 2026-11-01T23:30:00Z
- **Verdict**: **APPROVE**

---

## 1. Observation

All tests, empirical harnesses, and stress tests were executed directly in the runtime environment (`Python 3.14.3`, CPU mode).

### 1.1 Full Test Suite Execution
Executed command:
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```
**Result**: 83 tests ran in 0.072s (well within the <60s CPU budget). 0 errors, 0 failures (`OK`).

### 1.2 Re-Test of Iteration 1 Defects

#### Defect 1 Re-Test: `CrossEncoderNLI.eval()` and `.train()` on CPU Mock
Executed test script:
```python
from src.models.factory import build_model
m_baseline = build_model('mock', method='baseline')
ret_eval = m_baseline.eval()
assert ret_eval is m_baseline
assert m_baseline.training is False
assert m_baseline.mock_engine.training is False

ret_train = m_baseline.train()
assert ret_train is m_baseline
assert m_baseline.training is True
assert m_baseline.mock_engine.training is True
```
- **Result**: `CrossEncoderNLI` and `_ModuleBase` (lines 472-488 and 660-685 in `src/models/cross_encoder.py`) cleanly support `.eval()` and `.train()`, synchronize `self.training` and `self.mock_engine.training`, and return `self` for chaining. Zero `AttributeError` raised.

#### Defect 2 Re-Test: `build_model(method='crl_sd').predict_pair()` Delegation & 100% SoftCons
Executed test script:
```python
from src.models import ID_TO_LABEL
from src.models.factory import build_model
from src.models.losses import REV_INDEX_MAP
from src.metrics.metrics import compute_official_metrics

m_crl_sd = build_model('mock', method='crl_sd', mask_negative_other=True)
p1, p2 = m_crl_sd.predict_pair('the red car', 'a vehicle')
assert p2 == REV_INDEX_MAP[p1]

# Evaluated 500 random text pairs
metrics = compute_official_metrics(y_true_dir=y_true_dir, y_pred_dir=y_pred_dir,
                                   y_true_rev=y_true_rev, y_pred_rev=y_pred_rev,
                                   is_reversible_mask=[True] * 500)
```
- **Result**:
  - `SymmetrizedModelWrapper._resolve_predict_target` (lines 210-216 in `src/models/factory.py`) inspects `self.base_model.encoder`, successfully finding `predict_proba`.
  - Calling `m.predict_pair()`, `m.predict_proba()`, and `m.predict()` executed with zero `NotImplementedError`.
  - Tested 500 arbitrary pairs under official SemEval metrics: `SoftCons: 1.000000 (100.0%)`, `HardCons: 1.000000 (100.0%)`.

#### Defect 3 Re-Test: Pure CRL Ablation Decoupled from Symmetrized Decoding in Eval Mode
Executed test script:
```python
from src.models.siamese_crl import SiameseCRLModel
# Mock encoder returning asymmetric predictions (index 2 for both dir and rev)
enc = AsymmetricMockEncoder()
model = SiameseCRLModel(encoder=enc, apply_symmetrized_decoding=False)
model.eval()

# 1. Forward with apply_symmetrized_decoding=False
out_no_sd = model.forward(input_ids_dir=[[1, 10, 20]], input_ids_rev=[[2, 20, 10]], apply_symmetrized_decoding=False)
assert out_no_sd.sym_probs_dir is None
assert out_no_sd.sym_probs_rev is None
assert out_no_sd.preds_dir == [2] and out_no_sd.preds_rev == [2]

# 2. Forward with apply_symmetrized_decoding=True
out_sd = model.forward(input_ids_dir=[[1, 10, 20]], input_ids_rev=[[2, 20, 10]], apply_symmetrized_decoding=True)
assert out_sd.sym_probs_dir is not None
assert out_sd.preds_dir == [1] and out_sd.preds_rev == [1]
```
- **Result**:
  - In `src/models/siamese_crl.py` (lines 249-293 for PyTorch and lines 556-577 for pure Python), `should_apply_sd` is strictly decoupled from `self.training`.
  - When `apply_symmetrized_decoding=False`, `sym_probs_dir` and `sym_probs_rev` are strictly `None`, preserving raw unprojected asymmetric predictions (`[2]` and `[2]`).
  - When `apply_symmetrized_decoding=True`, SD is applied, projecting the distribution and resolving the symmetric tie to `[1]` (`EQUIVALENCE`). Pure CRL ablation in Milestone 4 is unblocked.

#### Defect 4 Re-Test: `mask_negative_other` Behavior & Configurability
Executed test script:
```python
p_dir = [0.10, 0.10, 0.20, 0.60]  # NEGATIVE_OTHER (index 3) is dominant
p_rev = [0.20, 0.10, 0.10, 0.60]

dec_unmasked = SymmetrizedDecoder(mask_negative_other=False)
assert dec_unmasked.decode_pair(p_dir, p_rev) == (3, 3)

dec_masked = SymmetrizedDecoder(mask_negative_other=True)
assert dec_masked.decode_pair(p_dir, p_rev) == (2, 0)
assert dec_masked.decode_pair(p_dir, p_rev, mask_negative_other=False) == (3, 3)
```
- **Result**:
  - `SymmetrizedDecoder` (in `src/models/decoding.py`), `SymmetrizedModelWrapper` (in `src/models/factory.py`), and `SiameseCRLModel` (in `src/models/siamese_crl.py`) all provide complete configurability via `__init__`, dynamic attribute mutation, and per-call overrides (`mask_negative_other: Optional[bool] = None`).
  - When `mask_negative_other=True`, class 3 is zeroed, probabilities over `{0, 1, 2}` are renormalized, and official SemEval `SoftCons` evaluates to `1.000000 (100.0%)`.

### 1.3 Adversarial Stress Testing

1. **Empirical Dirichlet Stress Test (25,000 Distributions)**:
   - Evaluated 25,000 independent pairs across 5 distinct Dirichlet regimes (`Uniform`, `Sparse/Spiky`, `Ultra-Sparse`, `High Concentration`, `Asymmetric Skewed`) with `mask_negative_other=True`.
   - Result: 25,000 / 25,000 pairs achieved `SoftCons == 1.000000 (100.0%)` under `compute_official_metrics` in 0.177s.
2. **Extreme Negative Boundary `[0.0, 0.0, 0.0, 1.0]`**:
   - When masked, all 3 reversible classes are 0.0 (sum = 0.0).
   - The decoder safely clamps the denominator, prevents division by zero or NaN, and uses involutory tie-breaking to output `(1, 1)` (`EQUIVALENCE`), preserving coordinate equivariance.
3. **Robust Categorical Cross-Entropy with `ignore_index=-100`**:
   - In `src/models/losses.py`, `cross_entropy_pure` correctly masks single and batched samples with `target=-100` (returning 0.0 for masked samples without IndexError or dilution).
   - In `ConsistencyRegularizationLoss`, auto-deriving `yr` handles `-100` without KeyError, and task loss normalizes strictly by valid samples ($N_{valid}$).
   - Consistency loss correctly evaluates to `0.0` for perfectly symmetric distributions and `> 0.0` for asymmetric distributions.

---

## 2. Logic Chain

1. **Step 1 — Resolution of Defect 1 (Mock API Parity)**:
   - Direct empirical execution confirms `CrossEncoderNLI.eval()` and `train()` function cleanly in CPU mock mode, toggling `self.training` and chaining without `AttributeError`. (Observation 1.2, Defect 1).
2. **Step 2 — Resolution of Defect 2 (Wrapper Delegation)**:
   - Direct empirical execution confirms `build_model(method='crl_sd').predict_pair()` delegates inference to the underlying encoder's `predict_proba`, returns valid paired predictions, and guarantees 100% `SoftCons`. (Observation 1.2, Defect 2).
3. **Step 3 — Resolution of Defect 3 (Ablation Decoupling)**:
   - Direct empirical execution confirms that setting `apply_symmetrized_decoding=False` in `SiameseCRLModel.forward` in eval mode leaves `sym_probs_dir` and `sym_probs_rev` as `None` and produces unprojected predictions, enabling pure CRL ablation. (Observation 1.2, Defect 3).
4. **Step 4 — Resolution of Defect 4 (Configurable Masking)**:
   - Direct empirical execution confirms `mask_negative_other` is fully configurable across all classes and methods, and guarantees `SoftCons = 1.000000` on reversible subsets under official SemEval metrics. (Observation 1.2, Defect 4).
5. **Step 5 — Stress Testing & Non-Regression**:
   - 25,000 Dirichlet distributions, extreme boundary vectors, and `ignore_index=-100` loss edge cases all pass without failure. All 83 automated tests pass in 0.072s. (Observations 1.1 & 1.3).
6. **Step 6 — Final Assessment**:
   - All criteria have been empirically verified with zero regressions. The implementation satisfies all contractual requirements for Milestone 2.

---

## 3. Caveats

- Testing was conducted in Python 3.14 on macOS CPU; full multi-GPU distributed training scripts will be exercised during Milestone 3 on remote GPU infrastructure.
- All tests completed in < 0.2 seconds total, well under the 60-second CPU budget.

---

## 4. Conclusion

### Explicit Verdict: **APPROVE**

All 4 defects from Iteration 1 have been completely resolved with clean, mathematically principled logic. Milestone 2 (Models & Reversal Strategies) is fully validated, robustly stress-tested, and ready to unblock Milestone 3 (CLI Training Pipeline & GPU Scripts).

---

## 5. Verification Method

To independently reproduce this verification:

1. **Run full automated test suite**:
   ```bash
   python3 -m unittest discover -s tests -p "test_*.py" -v
   ```
   *Expected Result*: 83 tests passing, 0 failures, 0 errors in $< 0.1$s.

2. **Verify CPU Mock `eval()` / `train()`**:
   ```bash
   python3 -c "
   from src.models.factory import build_model
   m = build_model('mock', method='baseline')
   m.eval()
   assert not m.training
   m.train()
   assert m.training
   print('CrossEncoderNLI eval/train OK')
   "
   ```

3. **Verify `crl_sd.predict_pair`**:
   ```bash
   python3 -c "
   from src.models.factory import build_model
   from src.models.losses import REV_INDEX_MAP
   m = build_model('mock', method='crl_sd', mask_negative_other=True)
   pd, pr = m.predict_pair('the red car', 'a vehicle')
   assert pr == REV_INDEX_MAP[pd]
   print('crl_sd predict_pair OK')
   "
   ```

4. **Verify Pure CRL Ablation Decoupling**:
   ```bash
   python3 -c "
   from src.models.siamese_crl import SiameseCRLModel
   from src.models.cross_encoder import MockCrossEncoder
   m = SiameseCRLModel(encoder=MockCrossEncoder(), apply_symmetrized_decoding=False)
   m.eval()
   out = m.forward(input_ids_dir=[[1, 2, 3]], input_ids_rev=[[3, 2, 1]], apply_symmetrized_decoding=False)
   assert out.sym_probs_dir is None
   print('Pure CRL ablation decoupling OK')
   "
   ```

5. **Verify `mask_negative_other` Configurability**:
   ```bash
   python3 -c "
   from src.models.decoding import SymmetrizedDecoder
   p = [0.1, 0.1, 0.1, 0.7]
   dec = SymmetrizedDecoder(mask_negative_other=False)
   assert dec.decode_pair(p, p) == (3, 3)
   assert dec.decode_pair(p, p, mask_negative_other=True) == (1, 1)
   print('mask_negative_other configurability OK')
   "
   ```
