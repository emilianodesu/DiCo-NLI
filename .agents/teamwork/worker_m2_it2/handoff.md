# Handoff Report: Milestone 2 Iteration 2 (Models & Reversal Strategies Remediation)

- **Agent**: Implementation Worker (`worker_m2_it2`)
- **Recipient**: Orchestrator (`df6ce377-6407-4a2d-bd18-829b36e20607`)
- **Date**: 2026-11-01T23:26:00Z
- **Working Directory**: `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/worker_m2_it2/`

---

## 1. Observation

During Milestone 2 review and empirical stress testing, the following defects and edge cases were identified:
1. **AttributeError on CPU Mock `.eval()` / `.train()`**:
   In `src/models/cross_encoder.py`, `_ModuleBase` lacked `.train()` and `.eval()` methods. Calling `m.eval()` on models instantiated via `build_model("mock", method="baseline")` raised `AttributeError: 'CrossEncoderNLI' object has no attribute 'eval'`.
2. **Wrapper Delegation Breakdown for `crl_sd`**:
   In `src/models/factory.py`, `SymmetrizedModelWrapper.predict_pair` called `self.base_model.predict_proba()`. For `method="crl_sd"`, `self.base_model` is `SiameseCRLModel`, which lacked `predict_proba`, causing `NotImplementedError: Base model does not implement predict_proba`.
3. **Pure CRL Ablation Conflation with Symmetrized Decoding**:
   In `src/models/siamese_crl.py`, `SiameseCRLModel.forward` in PyTorch mode checked `if apply_symmetrized_decoding or not self.training:` (line 239) and pure Python mode applied `symmetrized_decoding` unconditionally (line 391). Consequently, calling `model.eval()` always forced Symmetrized Decoding, outputting projected probabilities `sym_probs_dir` even when `apply_symmetrized_decoding=False`, precluding pure CRL ablation.
4. **PyTorch `ignore_index=-100` Crash in Pure Python Cross-Entropy**:
   In `src/models/losses.py`, `cross_entropy_pure(logits, -100)` performed `probs[-100]`, raising `IndexError: list index out of range`. In `ConsistencyRegularizationLoss`, auto-deriving `yr` when `yd` contained `-100` raised `KeyError: -100`. Furthermore, dividing batch loss by batch size diluted loss when masked instances were present instead of dividing strictly by $N_{valid}$.
5. **Configurability of `mask_negative_other` in `SymmetrizedDecoder`**:
   `SymmetrizedDecoder` methods did not allow per-call overriding (`mask_negative_other: Optional[bool] = None`), and `SymmetrizedModelWrapper` did not propagate the parameter.

Verbatim execution prior to fixes:
```
AttributeError: 'CrossEncoderNLI' object has no attribute 'eval'
NotImplementedError: Base model does not implement predict_proba.
IndexError: list index out of range
```

---

## 2. Logic Chain

1. **Parity for Non-PyTorch Fallback (`src/models/cross_encoder.py`)**:
   - Observation 1 showed that `_ModuleBase` and `CrossEncoderNLI` lacked `.eval()` and `.train()`.
   - We implemented `train(self, mode: bool = True)` and `eval(self)` on `_ModuleBase` to maintain `self.training` and return `self`.
   - In `CrossEncoderNLI`, `train()` and `eval()` synchronize `self.training` and recursively invoke `self.mock_engine.train(mode)` or `self.backbone.train(mode)`.
2. **Wrapper Inference Resolution & Pair Prediction (`src/models/factory.py`)**:
   - Observation 2 showed that `SymmetrizedModelWrapper` failed when wrapping architectures whose prediction logic resided in an underlying encoder.
   - We implemented `_resolve_predict_target(self)` in `SymmetrizedModelWrapper`: if `self.base_model` does not directly have `predict_proba`, it inspects `self.base_model.encoder`.
   - We expanded `predict_pair` to support both modern (`text_a`, `text_b`) and legacy (`text1`, `text2`) parameters, optional `mask_negative_other` per-call overrides, and added `predict_proba` and `predict` methods.
   - In `build_model`, we explicitly configured `apply_symmetrized_decoding=False` for `clean_method == "crl"` and `apply_symmetrized_decoding=True` for `clean_method == "crl_sd"`.
3. **Decoupled Evaluation & Symmetrized Decoding Control (`src/models/siamese_crl.py`)**:
   - Observation 3 showed that coupling `self.training` to post-hoc symmetrization prevented evaluating pure CRL models without Symmetrized Decoding.
   - In `SiameseCRLModel.__init__`, we added `apply_symmetrized_decoding: bool = False` and `mask_negative_other: bool = False`.
   - In `SiameseCRLModel.forward`, we decoupled `should_apply_sd` from `self.training`. When `should_apply_sd` is `False`, `sym_probs_dir` and `sym_probs_rev` are set to `None`, and discrete predictions are derived directly from unprojected stream distributions.
   - We added complete inference APIs (`predict_logits`, `predict_proba`, `predict`, `predict_pair`, `decode_pair`, `decode_pairs`) to `SiameseCRLModel` in both PyTorch and Pure-Python classes.
4. **Robust Categorical Cross-Entropy with `ignore_index=-100` (`src/models/losses.py`)**:
   - Observation 4 showed that masked targets crashed or corrupted pure-Python categorical loss computations.
   - In `cross_entropy_pure`, if `target == ignore_index` (default -100) or is out of bounds, the sample is ignored. For 2D batches, the mean divides strictly by non-ignored count $N_{valid}$ (returning 0.0 if all entries are masked).
   - In `ConsistencyRegularizationLoss` fallback, deriving `yr` uses `REV_INDEX_MAP.get(y, y)`, safely keeping `-100` intact, and normalizes task loss strictly by valid samples.
5. **Decoder Masking Control & Metric Alignment (`src/models/decoding.py`)**:
   - Observation 5 showed that while `mask_negative_other=True` is required to guarantee 100% `SoftCons` under SemEval-2027 consistency metrics, users needed flexible overriding.
   - We added `mask_negative_other: Optional[bool] = None` per-call overriding across `project_probabilities`, `decode_pair`, `decode_pairs`, `decode_batch`, and `decode`.
   - Comprehensive docstrings explaining the mathematical rationale and SemEval metric alignment were added.

---

## 3. Caveats

- In PyTorch mode (`HAS_TORCH=True`), `torch.nn.CrossEntropyLoss` already ignores index -100 by default; our fixes ensure identical mathematical behavior in pure-Python/CPU-mock mode.
- When `mask_negative_other=True`, class index 3 (`NEGATIVE_OTHER`) probability is zeroed and indices 0, 1, 2 are renormalized. On datasets where `NEGATIVE_OTHER` is a valid ground-truth relation, `mask_negative_other=False` should be retained.

---

## 4. Conclusion

All 5 defects identified in Milestone 2 review have been remedied with genuine, mathematically rigorous logic:
- Mock interface parity is complete (`eval()` and `train()` chain correctly on all mock and wrapper models).
- `crl_sd` and `sd` wrappers correctly delegate inference.
- Pure CRL ablation is fully decoupled from Symmetrized Decoding.
- PyTorch `ignore_index=-100` parity is achieved in pure Python cross-entropy and CRL loss functions.
- `mask_negative_other` is configurable at instance, attribute, and call levels.
- Full test suite expanded from 73 to 83 tests, achieving a 100% pass rate in 0.063s on CPU (well within the <60s budget).

---

## 5. Verification Method

To independently verify the implementation:

1. **Run full unit test suite**:
   ```bash
   python3 -m unittest discover -s tests -p "test_*.py" -v
   ```
   *Expected*: 83 tests passing, 0 errors, 0 failures in $<1.0$s.

2. **Verify CPU Mock `.eval()` / `.train()`**:
   ```bash
   python3 -c "from src.models.factory import build_model; m = build_model('mock', method='baseline'); m.eval(); assert not m.training; m.train(); assert m.training; print('CrossEncoderNLI eval/train OK')"
   ```

3. **Verify Pure CRL Ablation disables Symmetrized Decoding in `eval()` mode**:
   ```bash
   python3 -c "
   from src.models.siamese_crl import SiameseCRLModel
   from src.models.cross_encoder import MockCrossEncoder
   model = SiameseCRLModel(encoder=MockCrossEncoder())
   model.eval()
   out = model.forward(input_ids_dir=[[1, 2, 3]], input_ids_rev=[[3, 2, 1]], apply_symmetrized_decoding=False)
   assert out.sym_probs_dir is None, 'sym_probs_dir should be None'
   assert out.sym_probs_rev is None, 'sym_probs_rev should be None'
   print('Pure CRL ablation check: PASSED')
   "
   ```

4. **Verify `crl_sd` and `crl` `predict_pair` inference**:
   ```bash
   python3 -c "
   from src.models.factory import build_model
   from src.models.losses import REV_INDEX_MAP

   m_crl = build_model('mock', method='crl')
   p_d, p_r = m_crl.predict_pair('the red car', 'a vehicle')
   print(f'crl predict_pair: {p_d}, {p_r}')

   m_crl_sd = build_model('mock', method='crl_sd')
   p_d_sd, p_r_sd = m_crl_sd.predict_pair('the red car', 'a vehicle')
   assert p_r_sd == REV_INDEX_MAP[p_d_sd]
   print(f'crl_sd predict_pair: {p_d_sd}, {p_r_sd}')
   print('predict_pair checks: PASSED')
   "
   ```

5. **Verify `cross_entropy_pure` `target=-100` and batch normalization**:
   ```bash
   python3 -c "
   from src.models.losses import cross_entropy_pure
   l1 = [0.1, 0.2, 2.5, 0.1]
   assert cross_entropy_pure(l1, -100) == 0.0
   assert abs(cross_entropy_pure([l1, l1], [2, -100]) - cross_entropy_pure(l1, 2)) < 1e-9
   print('cross_entropy_pure ignore_index OK')
   "
   ```

6. **Verify `SymmetrizedDecoder` configurability**:
   ```bash
   python3 -c "
   from src.models.decoding import SymmetrizedDecoder
   p = [0.1, 0.1, 0.1, 0.7]
   dec = SymmetrizedDecoder(mask_negative_other=False)
   assert dec.decode_pair(p, p) == (3, 3)
   assert dec.decode_pair(p, p, mask_negative_other=True) == (1, 1)
   print('SymmetrizedDecoder configurability OK')
   "
   ```
