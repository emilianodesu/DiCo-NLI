# Empirical Challenge Report & Handoff: Milestone 2 (M2)

- **Agent**: Challenger 2 (`challenger_m2_2`)
- **Role**: Empirical Challenger (critic, specialist)
- **Target**: Milestone 2 (Models & Reversal Strategies)
- **Recipient**: Orchestrator (`df6ce377-6407-4a2d-bd18-829b36e20607`)
- **Date**: 2026-11-01T23:06:00Z
- **Verdict**: **REQUEST_CHANGES**

---

## 1. Observation

All tests and empirical stress scripts were executed directly in the runtime environment (`Python 3.14.3`, CPU mode).

### 1.1 Full Test Suite Verification
Executed command:
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```
**Result**: 73 tests passed in 0.064s with 0 errors and 0 failures.

### 1.2 Empirical Dirichlet Stress Test (25,000 Distributions)
Tested 25,000 independent pairs across 5 distinct Dirichlet regimes:
1. Uniform Dirichlet ($\alpha = [1.0, 1.0, 1.0, 1.0]$): 5,000 pairs -> Official SoftCons = 1.000000 (100.0%)
2. Sparse/Spiky Dirichlet ($\alpha = [0.1, 0.1, 0.1, 0.1]$): 5,000 pairs -> Official SoftCons = 1.000000 (100.0%)
3. Ultra-Sparse Dirichlet ($\alpha = [0.01, 0.01, 0.01, 0.01]$): 5,000 pairs -> Official SoftCons = 1.000000 (100.0%)
4. High Concentration Centroid ($\alpha = [50.0, 50.0, 50.0, 50.0]$): 5,000 pairs -> Official SoftCons = 1.000000 (100.0%)
5. Asymmetric Skewed ($\alpha = [10.0, 0.2, 2.0, 0.5]$): 5,000 pairs -> Official SoftCons = 1.000000 (100.0%)

**Grand Total**: 25,000 / 25,000 pairs achieved `SoftCons == 1.000000` under the official SemEval metrics implementation (`compute_official_metrics`).
Raw coordinate equivariance ($\operatorname{argmax}(p_{sym,rev}) == \operatorname{Rev}(\operatorname{argmax}(p_{sym,dir}))$) held in 25,000 / 25,000 trials.

### 1.3 Boundary, Tie-Breaking, and Extreme Value Stress Tests
- **All 15 Tie-Breaking Subsets under Involution**:
  Every non-empty subset $S \subseteq \{0, 1, 2, 3\}$ of tied maximal values satisfies $\operatorname{argmax}^*(v \cdot P) == \operatorname{Rev}(\operatorname{argmax}^*(v))$. Passed 15/15.
- **Cross-Tie Combinations**:
  All $15 \times 15 = 225$ pairwise combinations of direct and reverse tie configurations passed with 100% equivariance.
- **Extreme Boundary Vectors**:
  Tested 144 combinations of one-hot spikes, all zeros ($[0, 0, 0, 0]$), extreme subnormals ($10^{-300}$), and float overflow extremes ($10^{300}$). Zero crashes, zero NaN outputs, 100% equivariance.
- **Softmax Stability**:
  Tested 25 extreme logit combinations up to float exp overflow boundary ($709.0$). All produced valid probability simplices and preserved equivariance.
- **Vectorized vs 1D Parity**:
  Verified 5,000 random samples comparing `decode_pairs(batch_dir, batch_rev)` against itemwise `decode_pair(d, r)`. Predictions were 100% identical.

---

### 1.4 Empirical Vulnerabilities & Defects Identified

#### Defect 1 (High): `CrossEncoderNLI` raises `AttributeError` on `.eval()` and `.train()` when `HAS_TORCH=False`
**Verbatim Error**:
```python
from src.models.factory import build_model
model = build_model('mock', method='baseline')
model.eval()
# -> AttributeError: 'CrossEncoderNLI' object has no attribute 'eval'
```
- In `src/models/cross_encoder.py` (lines 470-476), when `HAS_TORCH=False`, `_ModuleBase` only implements `__init__` and `__call__`.
- `CrossEncoderNLI` implements `to()`, but fails to forward `eval()` and `train()` to `self.mock_engine`.
- By contrast, `MockCrossEncoder` (lines 307-313), `MockTransformerModel` (lines 119-124), and `SiameseCRLModel` (lines 314-320) all implement `eval()` and `train()`.
- Any standard evaluation or test script calling `model.eval()` on `baseline` or `bsa` fails with `AttributeError`.

#### Defect 2 (High): `build_model(method='crl_sd')` raises `NotImplementedError` on `predict_pair()`
**Verbatim Error**:
```python
from src.models.factory import build_model
model = build_model('mock', method='crl_sd')
model.predict_pair('the red car', 'a vehicle')
# -> NotImplementedError: Base model does not implement predict_proba.
```
- In `src/models/factory.py` (lines 209-216), `SymmetrizedModelWrapper.predict_pair` calls `self.base_model.predict_proba(text1, text2)`.
- For `method="sd"`, `base_model` is `CrossEncoderNLI`, which defines `predict_proba`.
- For `method="crl_sd"`, `base_model` is `SiameseCRLModel`, which does NOT define `predict_proba`.
- As a result, `model.predict_pair` on `crl_sd` is completely broken.

#### Defect 3 (Medium/High): Pure CRL ablation is impossible because `SiameseCRLModel.forward` forces Symmetrized Decoding during evaluation
**Verbatim Execution**:
```python
from src.models.siamese_crl import SiameseCRLModel
from src.models.cross_encoder import MockCrossEncoder

enc = MockCrossEncoder()
model = SiameseCRLModel(encoder=enc)
out = model.forward(
    input_ids_dir=[[1, 2, 3]],
    input_ids_rev=[[3, 2, 1]],
    apply_symmetrized_decoding=False
)
print(out.sym_probs_dir is not None)
# -> True (Symmetrized decoding was executed anyway!)
```
- In `src/models/siamese_crl.py`:
  - Line 239 (PyTorch): `if apply_symmetrized_decoding or not self.training:` forces Symmetrized Decoding whenever `self.training == False` (evaluation mode), even if the caller explicitly passes `apply_symmetrized_decoding=False`.
  - Line 391 (Pure Python): `sym_p_dir, sym_p_rev = symmetrized_decoding(probs_dir, probs_rev)` unconditionally runs Symmetrized Decoding without ever checking `apply_symmetrized_decoding`.
- Consequence: In Milestone 4 (Evaluation & Ablations), it will be impossible to measure the standalone performance of Consistency Regularization Loss (CRL) without post-hoc Symmetrized Decoding, as CRL will always output SD predictions.

#### Defect 4 (Medium): Default `mask_negative_other=False` causes `SoftCons = 0.0` under official SemEval metrics when `NEGATIVE_OTHER` is predicted on reversible pairs
**Verbatim Execution**:
```python
from src.models.decoding import SymmetrizedDecoder
from src.metrics.metrics import compute_official_metrics

p_dir = [0.1, 0.1, 0.2, 0.6]  # predicts NEGATIVE_OTHER (index 3)
p_rev = [0.2, 0.1, 0.1, 0.6]  # predicts NEGATIVE_OTHER (index 3)

decoder = SymmetrizedDecoder(mask_negative_other=False)
idx_d, idx_r = decoder.decode_pair(p_dir, p_rev)
# idx_d = 3, idx_r = 3

metrics = compute_official_metrics(
    y_true_dir=['FORWARD_ENTAILMENT'],
    y_pred_dir=['NEGATIVE_OTHER'],
    y_true_rev=['BACKWARD_ENTAILMENT'],
    y_pred_rev=['NEGATIVE_OTHER'],
    is_reversible_mask=[True]
)
# -> {'weighted_f1': 0.0, 'soft_cons': 0.0, 'hard_cons': 0.0}
```
- In `src/metrics/metrics.py` (lines 370-373), `soft_ok = (pred_1 in REVERSIBLE_LABELS and pred_2 == REVERSE_LABEL_MAP.get(pred_1))`.
- If `pred_1 == 'NEGATIVE_OTHER'`, `soft_ok` evaluates to `False`.
- In `src/models/decoding.py` line 239 and `src/models/factory.py` line 252, `mask_negative_other` defaults to `False`.
- For reversible pair evaluation, Symmetrized Decoding only guarantees `SoftCons = 1.0` if `mask_negative_other=True` is enabled.

---

## 2. Logic Chain

1. **Step 1 — Mathematical Core Robustness**:
   - Observations 1.2 and 1.3 confirm that the algebraic foundation ($P = P^T, P^2 = I$, equivariant tie-breaking, and probability projection) is mathematically sound, numerically stable, and free of precision leaks across 25,000 Dirichlet distributions and edge-case permutations.
2. **Step 2 — Interface Inconsistency in `CrossEncoderNLI`**:
   - Observation 1.4 (Defect 1) shows that while `MockCrossEncoder` and `SiameseCRLModel` support `.eval()` and `.train()`, `CrossEncoderNLI` does not forward these methods in CPU fallback mode (`HAS_TORCH=False`). This breaks the polymorphic interface contract expected by standard training and evaluation loops.
3. **Step 3 — Factory Method Breakdown for `crl_sd`**:
   - Observation 1.4 (Defect 2) shows that `build_model(method='crl_sd')` returns a `SymmetrizedModelWrapper` wrapping `SiameseCRLModel`, but `SiameseCRLModel` does not implement `predict_proba`. Calling `predict_pair` raises `NotImplementedError`.
4. **Step 4 — Invalidation of CRL Ablation Study**:
   - Observation 1.4 (Defect 3) shows that `SiameseCRLModel.forward` cannot disable Symmetrized Decoding during evaluation. This prevents the M4 ablation study from disentangling representation learning via CRL loss from post-hoc decoding via SD.
5. **Step 5 — Verdict Determination**:
   - While the unit tests pass (Observation 1.1) and the core algorithm is mathematically proven (Observations 1.2 & 1.3), Defects 1, 2, and 3 will cause failures in Milestone 3 (CLI Training) and Milestone 4 (Evaluation & Ablations) if not addressed now. Therefore, a verdict of **REQUEST_CHANGES** is necessary.

---

## 3. Caveats

- **PyTorch/CUDA Execution**: Testing was conducted in Python 3.14 on macOS without PyTorch/CUDA. PyTorch execution paths were analyzed statically and via pure-Python parity equivalents.
- **Performance Budget**: All tests ran in under 0.1 seconds, well within the 60-second CPU budget.

---

## 4. Conclusion & Actionable Fix Recommendations

### Explicit Verdict: **REQUEST_CHANGES**

To unblock Milestone 3 and Milestone 4, `worker_m2` should implement the following minimal, targeted fixes:

1. **Fix `CrossEncoderNLI.eval()` and `CrossEncoderNLI.train()`** in `src/models/cross_encoder.py`:
   ```python
   def eval(self):
       if self.is_mock and hasattr(self, "mock_engine"):
           self.mock_engine.eval()
       elif hasattr(self, "backbone") and self.backbone is not None:
           self.backbone.eval()
       return self

   def train(self, mode: bool = True):
       if self.is_mock and hasattr(self, "mock_engine"):
           self.mock_engine.train(mode)
       elif hasattr(self, "backbone") and self.backbone is not None:
           self.backbone.train(mode)
       return self
   ```

2. **Fix `crl_sd.predict_pair` support** in `src/models/factory.py` (or `src/models/siamese_crl.py`):
   In `SymmetrizedModelWrapper.predict_pair` (`src/models/factory.py`):
   ```python
   def predict_pair(self, text1: str, text2: str) -> Tuple[int, int]:
       """Predict directional pair with 100% SoftCons guarantee."""
       target = self.base_model
       if hasattr(target, "encoder") and hasattr(target.encoder, "predict_proba"):
           target = target.encoder
       if hasattr(target, "predict_proba"):
           p_dir = target.predict_proba(text1, text2)
           p_rev = target.predict_proba(text2, text1)
           return self.decoder.decode_pair(p_dir, p_rev)
       raise NotImplementedError("Base model does not implement predict_proba.")
   ```

3. **Honor `apply_symmetrized_decoding=False` in `SiameseCRLModel.forward`** in `src/models/siamese_crl.py`:
   - In PyTorch mode (line 239):
     ```python
     # Only apply SD if explicitly enabled or if evaluating and not explicitly disabled
     apply_sd = apply_symmetrized_decoding if apply_symmetrized_decoding is not None else (not self.training)
     ```
     (Or use `if apply_symmetrized_decoding:` directly).
   - In pure-Python mode (line 391):
     ```python
     if apply_symmetrized_decoding:
         sym_p_dir, sym_p_rev = symmetrized_decoding(probs_dir, probs_rev)
         preds_dir = [_argmax_with_rev_tie_break(r) for r in sym_p_dir]
         preds_rev = [_argmax_with_rev_tie_break(r) for r in sym_p_rev]
     else:
         sym_p_dir, sym_p_rev = None, None
         preds_dir = [r.index(max(r)) for r in probs_dir]
         preds_rev = [r.index(max(r)) for r in probs_rev]
     ```

4. **Default / Document `mask_negative_other=True` for Reversible Pair Decoding**:
   Ensure `build_model(method='sd', mask_negative_other=True)` is used or set as default when evaluating on reversible test sets to guarantee official `SoftCons = 1.000000`.

---

## 5. Verification Method

Once `worker_m2` applies the fixes, run the following verification commands:

1. **Verify Full Unit Test Suite**:
   ```bash
   python3 -m unittest discover -s tests -p "test_*.py" -v
   ```
   *Expected Result*: 73+ tests pass with `OK`.

2. **Verify `model.eval()` on `CrossEncoderNLI`**:
   ```bash
   python3 -c "
   from src.models.factory import build_model
   model = build_model('mock', method='baseline')
   model.eval()
   print('CrossEncoderNLI.eval() OK')
   "
   ```

3. **Verify `crl_sd.predict_pair`**:
   ```bash
   python3 -c "
   from src.models.factory import build_model
   model = build_model('mock', method='crl_sd')
   p_d, p_r = model.predict_pair('the red car', 'a vehicle')
   print(f'crl_sd predict_pair OK: dir={p_d}, rev={p_r}')
   "
   ```

4. **Verify Pure CRL Disables Symmetrized Decoding**:
   ```bash
   python3 -c "
   from src.models.siamese_crl import SiameseCRLModel
   from src.models.cross_encoder import MockCrossEncoder
   model = SiameseCRLModel(encoder=MockCrossEncoder())
   out = model.forward(input_ids_dir=[[1, 2, 3]], input_ids_rev=[[3, 2, 1]], apply_symmetrized_decoding=False)
   assert out.sym_probs_dir is None, 'SD should not be applied when apply_symmetrized_decoding=False'
   print('Pure CRL ablation toggle OK')
   "
   ```
