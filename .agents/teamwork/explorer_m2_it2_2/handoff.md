# Handoff Report: Siamese CRL Ablation & Symmetrized Decoding Control

- **Agent**: Explorer 2 (`explorer_m2_it2_2`)
- **Role**: Read-only Investigator & Architecture Specialist
- **Target**: Milestone 2 Iteration 2 (Siamese CRL Ablation & Symmetrized Decoding Control)
- **Recipient**: Orchestrator (`df6ce377-6407-4a2d-bd18-829b36e20607`), Worker (`worker_m2`)
- **Date**: 2026-11-01T23:14:00Z
- **Type**: **Hard Handoff** (Task complete; blueprints, prototypes, and verification scripts fully specified)

---

## 1. Observation

### 1.1 Direct Observation of the Evaluation Bug in `src/models/siamese_crl.py`
In `src/models/siamese_crl.py`:
- **PyTorch mode** (line 239):
  ```python
  if apply_symmetrized_decoding or not self.training:
  ```
  When the model is in evaluation mode (`model.eval()`), `self.training` is `False`, so `not self.training` is `True`. This unconditionally forces Symmetrized Decoding during evaluation even when `apply_symmetrized_decoding=False` is passed.
- **Pure-Python fallback mode** (lines 391-394):
  ```python
  sym_p_dir, sym_p_rev = symmetrized_decoding(probs_dir, probs_rev)
  preds_dir = [_argmax_with_rev_tie_break(r) for r in sym_p_dir]
  preds_rev = [_argmax_with_rev_tie_break(r) for r in sym_p_rev]
  ```
  The function `symmetrized_decoding` was executed unconditionally without ever inspecting the `apply_symmetrized_decoding` argument.
- **Verbatim execution reproducing the bug**:
  Command:
  ```bash
  python3 -c "
  from src.models.cross_encoder import MockCrossEncoder
  from src.models.siamese_crl import SiameseCRLModel
  model = SiameseCRLModel(encoder=MockCrossEncoder())
  out1 = model.forward(input_ids_dir=[[1, 2, 3]], input_ids_rev=[[3, 2, 1]], apply_symmetrized_decoding=False)
  print('sym_probs_dir with apply_symmetrized_decoding=False:', out1.sym_probs_dir is not None)
  model.eval()
  out2 = model.forward(input_ids_dir=[[1, 2, 3]], input_ids_rev=[[3, 2, 1]])
  print('sym_probs_dir in eval mode:', out2.sym_probs_dir is not None)
  "
  ```
  Output:
  ```
  sym_probs_dir with apply_symmetrized_decoding=False: True
  sym_probs_dir in eval mode: True
  ```

### 1.2 Direct Observation of Missing Inference Methods on `SiameseCRLModel`
In `src/models/siamese_crl.py`:
- `SiameseCRLModel` implemented only `_encode_stream`, `forward`, and `__call__` (PyTorch mode), and `train`, `eval`, `to`, `_encode`, `forward`, `__call__` (Pure-Python mode).
- It lacked `predict_pair`, `predict_proba`, `predict_logits`, and `predict`.
- **Verbatim execution reproducing the factory and inference failure**:
  Command:
  ```bash
  python3 -c "
  from src.models.factory import build_model
  m_crl_sd = build_model('mock', method='crl_sd')
  try:
      m_crl_sd.predict_pair('premise', 'hypothesis')
  except Exception as e:
      print('crl_sd error:', type(e).__name__, e)

  m_crl = build_model('mock', method='crl')
  try:
      m_crl.predict_pair('premise', 'hypothesis')
  except Exception as e:
      print('crl error:', type(e).__name__, e)
  "
  ```
  Output:
  ```
  crl_sd error: NotImplementedError Base model does not implement predict_proba.
  crl error: AttributeError 'SiameseCRLModel' object has no attribute 'predict_pair'
  ```

### 1.3 Direct Observation of Dynamic Prototype Success
A prototype of the proposed solution was executed in the runtime environment:
- `out.sym_probs_dir is None` held true when `apply_symmetrized_decoding=False`.
- `model.eval()` did not force Symmetrized Decoding when `apply_symmetrized_decoding=False`.
- `model.predict_pair("the red car", "a vehicle", apply_symmetrized_decoding=False)` returned raw unprojected class predictions.
- `model.predict_pair("the red car", "a vehicle", apply_symmetrized_decoding=True)` returned equivariant predictions satisfying `pred_rev == REV_INDEX_MAP[pred_dir]`.
- `build_model("mock", method="crl_sd").predict_pair("the red car", "a vehicle")` succeeded with 100% equivariance.
- All 5 assertions passed without error.

---

## 2. Logic Chain

1. **Step 1 — Separation of Learning vs Post-Hoc Projection**:
   Observation 1.1 reveals that `SiameseCRLModel` conflated evaluation mode (`model.eval()`) with Symmetrized Decoding post-hoc projection. CRL is an intrinsic representation learning method trained via divergence loss $\mathcal{L}_{CRL}$, whereas Symmetrized Decoding is an extrinsic test-time projection $p_{sym} = 0.5(p + p_{rev} P)$. To isolate CRL's contribution to directional consistency in Milestone 4 and the ACL paper, the model must support pure CRL evaluation with unprojected predictions ($\hat{y} = \operatorname{argmax}(p)$).
2. **Step 2 — Evaluation Flag Hierarchy**:
   By introducing `self.apply_symmetrized_decoding: bool = False` as an instance attribute (defaulting to pure CRL) and allowing call-time override via `forward(..., apply_symmetrized_decoding=...)` and `predict_pair(..., apply_symmetrized_decoding=...)`, callers have total control:
   $$\text{should\_apply\_sd} = \text{apply\_symmetrized\_decoding} \text{ if not None else } \text{self.apply\_symmetrized\_decoding}$$
   Removing `or not self.training` from line 239 decouples evaluation mode from projection.
3. **Step 3 — Clean Output Contract**:
   When `should_apply_sd == False`:
   `sym_probs_dir = None`, `sym_probs_rev = None`, `preds_dir = argmax(logits_dir)`, `preds_rev = argmax(logits_rev)`.
   When `should_apply_sd == True`:
   `sym_probs_dir` and `sym_probs_rev` contain the projected distributions, and `preds_dir`, `preds_rev` use `_argmax_with_rev_tie_break`.
4. **Step 4 — Unified Inference Interface**:
   Observation 1.2 shows that `SymmetrizedModelWrapper.predict_pair` failed on `crl_sd` because `SiameseCRLModel` lacked `predict_proba`. Adding `predict_logits`, `predict_proba`, `predict`, `predict_pair`, `decode_pair`, and `decode_pairs` directly to `SiameseCRLModel` simultaneously fixes `build_model(method='crl_sd')` wrapper delegation AND allows standalone pair prediction on `method='crl'`.
5. **Step 5 — Dual-Stream Architecture Parity**:
   Implementing these fixes symmetrically in both `if HAS_TORCH:` and `else:` blocks guarantees 100% parity between remote GPU training (CUDA/Torch) and local rapid testing (CPU/No-Torch).

---

## 3. Caveats

- **Scope Coordination**:
  - `CrossEncoderNLI.eval()` and `CrossEncoderNLI.train()` when `HAS_TORCH=False` is handled by Explorer 1 (`explorer_m2_it2_1`).
  - `cross_entropy_pure` `target=-100` handling is handled by Explorer 3 (`explorer_m2_it2_3`).
  - This report strictly covers `SiameseCRLModel` forward decoding control, CRL ablation separation, and inference methods (`predict_pair`, `predict_proba`).
- **Default Parameter Value**:
  Existing test `test_siamese_crl_model_forward` in `tests/test_models.py` explicitly passed `apply_symmetrized_decoding=True`, so it will continue to pass without modification. New unit tests will verify that omitting or setting `apply_symmetrized_decoding=False` correctly keeps `sym_probs_dir is None`.

---

## 4. Conclusion & Actionable Blueprint

The solution is fully designed and prototyped. `worker_m2` should apply the following blueprint:

### 4.1 In `src/models/siamese_crl.py`:
1. Add `_argmax_scalar(p)` helper function.
2. In PyTorch `SiameseCRLModel`:
   - Add `apply_symmetrized_decoding: bool = False` and `mask_negative_other: bool = False` to `__init__`.
   - Update `forward` signature with `apply_symmetrized_decoding: Optional[bool] = None` and `mask_negative_other: Optional[bool] = None`.
   - Replace line 239 `if apply_symmetrized_decoding or not self.training:` with `if should_apply_sd:`.
   - When `should_apply_sd` is `False`, set `sym_probs_dir = None`, `sym_probs_rev = None`, and use `torch.argmax(logits, dim=-1)`.
   - Add `predict_logits`, `predict_proba`, `predict`, `predict_pair`, `decode_pair`, and `decode_pairs`.
3. In Pure-Python `SiameseCRLModel`:
   - Mirror all changes from PyTorch version: update `__init__`, update `forward` to check `should_apply_sd`, return `None` for `sym_probs_dir` and `sym_probs_rev` when `False`, and add all 6 inference methods.

### 4.2 In `src/models/factory.py`:
In `build_model`:
- When `clean_method == "crl"`: pass `apply_symmetrized_decoding=False`.
- When `clean_method == "crl_sd"`: pass `apply_symmetrized_decoding=True`.

### 4.3 In `tests/test_models.py`:
Add:
- `test_siamese_crl_pure_ablation`: confirms `sym_probs_dir is None` during eval with `apply_symmetrized_decoding=False`.
- `test_siamese_crl_predict_pair_and_proba`: validates `predict_proba` and `predict_pair` with SD toggle.
- `test_crl_sd_factory_predict_pair`: validates `build_model("mock", method="crl_sd").predict_pair(...)`.

---

## 5. Verification Method

Once `worker_m2` applies the patch, execute the following commands:

1. **Verify Pure CRL Forward Ablation**:
   ```bash
   python3 -c "
   from src.models.siamese_crl import SiameseCRLModel
   from src.models.cross_encoder import MockCrossEncoder
   model = SiameseCRLModel(encoder=MockCrossEncoder())
   model.eval()
   out = model.forward(input_ids_dir=[[1, 2, 3]], input_ids_rev=[[3, 2, 1]], apply_symmetrized_decoding=False)
   assert out.sym_probs_dir is None, 'Failed: sym_probs_dir should be None'
   assert out.sym_probs_rev is None, 'Failed: sym_probs_rev should be None'
   print('Pure CRL forward ablation: OK')
   "
   ```

2. **Verify `predict_pair` on `crl` and `crl_sd`**:
   ```bash
   python3 -c "
   from src.models.factory import build_model
   from src.models.losses import REV_INDEX_MAP
   m_crl = build_model('mock', method='crl')
   p_d, p_r = m_crl.predict_pair('the red car', 'a vehicle')
   print(f'crl predict_pair: {p_d}, {p_r}')

   m_crl_sd = build_model('mock', method='crl_sd')
   p_d_sd, p_r_sd = m_crl_sd.predict_pair('the red car', 'a vehicle')
   print(f'crl_sd predict_pair: {p_d_sd}, {p_r_sd}')
   assert p_r_sd == REV_INDEX_MAP[p_d_sd], 'crl_sd symmetry failed'
   print('predict_pair checks: OK')
   "
   ```

3. **Run Full Test Suite via Unittest**:
   ```bash
   python3 -m unittest discover -s tests -p "test_*.py" -v
   ```
   *Expected Result*: All tests pass with code 0 (`OK`).
