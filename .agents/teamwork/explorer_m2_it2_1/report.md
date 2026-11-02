# Explorer 1 Technical Report: Mock Interface Parity & Wrapper Delegation (M2 Iteration 2)

- **Agent**: Explorer 1 (`explorer_m2_it2_1`)
- **Role**: Mock Interface & Wrapper Delegation Specialist
- **Target**: Milestone 2 (Models & Reversal Strategies)
- **Recipient**: Orchestrator (`df6ce377-6407-4a2d-bd18-829b36e20607`), Worker (`worker_m2_it2`)
- **Date**: 2026-11-01T23:12:00Z

---

## 1. Executive Summary

During Milestone 2 review and empirical stress testing by `challenger_m2_2`, two high-priority interface defects were detected that prevent the execution of standard training and evaluation loops on CPU and break inference for the `crl_sd` reversal mitigation method:

1. **Defect 1 (Mock Interface Parity)**: When `HAS_TORCH=False` (CPU fallback environment), `CrossEncoderNLI` inherits from `_ModuleBase`, which fails to implement `.eval()` and `.train()`. Calling `model.eval()` on models created via `build_model('mock', method='baseline')` raises `AttributeError: 'CrossEncoderNLI' object has no attribute 'eval'`.
2. **Defect 2 (Wrapper Delegation for `crl_sd`)**: `build_model('mock', method='crl_sd')` returns a `SymmetrizedModelWrapper` wrapping a `SiameseCRLModel`. When `model.predict_pair(text_a, text_b)` is called, the wrapper tries to call `self.base_model.predict_proba()`. Because `SiameseCRLModel` does not implement `predict_proba`, it raises `NotImplementedError: Base model does not implement predict_proba`.

This report provides the full architectural diagnosis and an exact, drop-in code patch blueprint for `worker_m2_it2`.

---

## 2. Root Cause Analysis

### 2.1 Defect 1: Interface Breakage in `CrossEncoderNLI` under CPU Fallback

- **Source File**: `src/models/cross_encoder.py` (lines 469–478, 479–514, 641–646)
- **Mechanism**:
  - In `cross_encoder.py`:
    ```python
    if HAS_TORCH:
        _ModuleBase = torch.nn.Module
    else:
        class _ModuleBase:
            def __init__(self, *args, **kwargs):
                pass
            def __call__(self, *args, **kwargs):
                return self.forward(*args, **kwargs)
    ```
  - When `HAS_TORCH=False` (e.g., Python 3.14 on macOS without PyTorch), `_ModuleBase` has only `__init__` and `__call__`.
  - `CrossEncoderNLI` inherits from `_ModuleBase`. While it overrides `to(self, device)`, it never implements `train(self, mode)` or `eval(self)`.
  - In addition, if `HAS_TORCH=True` with `is_mock=True`, `torch.nn.Module.eval()` would set `self.training = False` but would fail to set `self.mock_engine.training = False` because `self.mock_engine` is a `MockCrossEncoder` (not registered as an `nn.Module` child module).
- **Required Fix**:
  1. Add `.train(mode: bool = True)` and `.eval()` to `_ModuleBase` when `HAS_TORCH=False`, setting `self.training = mode` and returning `self`.
  2. Implement `.train(mode: bool = True)` and `.eval()` on `CrossEncoderNLI` to synchronize `self.training`, recursively update `self.mock_engine` (when `self.is_mock`), or forward to `self.backbone` (when running real PyTorch weights).
  3. Ensure `MockCrossEncoder.train` and `MockCrossEncoder.eval` have explicit docstrings and return `self`.

---

### 2.2 Defect 2: Wrapper Delegation Failure for `crl_sd` in `SymmetrizedModelWrapper`

- **Source File**: `src/models/factory.py` (lines 194–216, 300–302, 325–327)
- **Mechanism**:
  - In `factory.py`, `build_model(..., method='crl_sd')` instantiates:
    ```python
    crl_model = SiameseCRLModel(encoder=base_encoder)
    return SymmetrizedModelWrapper(crl_model, mask_negative_other=mask_negative_other)
    ```
  - In `SymmetrizedModelWrapper.predict_pair`:
    ```python
    def predict_pair(self, text1: str, text2: str) -> Tuple[int, int]:
        if hasattr(self.base_model, "predict_proba"):
            p_dir = self.base_model.predict_proba(text1, text2)
            p_rev = self.base_model.predict_proba(text2, text1)
            return self.decoder.decode_pair(p_dir, p_rev)
        raise NotImplementedError("Base model does not implement predict_proba.")
    ```
  - `self.base_model` is `SiameseCRLModel`. `SiameseCRLModel` does not implement `predict_proba`.
  - The underlying sequence classification model is stored in `self.base_model.encoder`.
- **Required Fix**:
  1. In `SymmetrizedModelWrapper.predict_pair`: Resolve the prediction target by checking whether `self.base_model` has `predict_proba`. If not, but `hasattr(self.base_model, "encoder")` has `predict_proba`, delegate to `self.base_model.encoder`.
  2. Parameter Flexibility: Support both positional and keyword arguments `text_a, text_b` (SemEval / modern convention) and `text1, text2` (legacy alias).
  3. Add `predict_proba(text1, text2)` and `predict(text1, text2)` to `SymmetrizedModelWrapper` to provide complete interface parity with standard models.
  4. (Complementary robustness) Also recommend adding `predict_proba` to `SiameseCRLModel` so it forwards calls directly to `self.encoder`.

---

## 3. Detailed Code Blueprint for Worker

### 3.1 Modification 1: `src/models/cross_encoder.py`

#### A. Update `_ModuleBase` (Lines 469–478)
```python
<<<<
if HAS_TORCH:
    _ModuleBase = torch.nn.Module
else:
    class _ModuleBase:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)
====
if HAS_TORCH:
    _ModuleBase = torch.nn.Module
else:
    class _ModuleBase:  # type: ignore
        def __init__(self, *args, **kwargs):
            self.training = True

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def train(self, mode: bool = True):
            self.training = mode
            return self

        def eval(self):
            return self.train(False)

        def to(self, device: Any):
            return self
>>>>
```

#### B. Update `CrossEncoderNLI` (Lines 641–647)
Add `train` and `eval` methods to `CrossEncoderNLI`:
```python
<<<<
    def to(self, device: Any):
        if self.is_mock:
            return self
        if hasattr(self, "backbone") and self.backbone is not None:
            self.backbone.to(device)
        return self
====
    def to(self, device: Any):
        if self.is_mock:
            return self
        if hasattr(self, "backbone") and self.backbone is not None:
            self.backbone.to(device)
        return self

    def train(self, mode: bool = True) -> CrossEncoderNLI:
        """Set model to training (mode=True) or evaluation (mode=False) mode.

        Returns:
            self for method chaining.
        """
        self.training = mode
        if HAS_TORCH and hasattr(super(), "train"):
            super().train(mode)
        if self.is_mock and hasattr(self, "mock_engine"):
            self.mock_engine.train(mode)
        elif hasattr(self, "backbone") and self.backbone is not None:
            if hasattr(self.backbone, "train"):
                self.backbone.train(mode)
        return self

    def eval(self) -> CrossEncoderNLI:
        """Set model to evaluation mode.

        Equivalent to self.train(False).

        Returns:
            self for method chaining.
        """
        return self.train(False)
>>>>
```

---

### 3.2 Modification 2: `src/models/factory.py`

#### Update `SymmetrizedModelWrapper` (Lines 209–237)
```python
<<<<
    def predict_pair(self, text1: str, text2: str) -> Tuple[int, int]:
        """Predict directional pair with 100% SoftCons guarantee."""
        if hasattr(self.base_model, "predict_proba"):
            p_dir = self.base_model.predict_proba(text1, text2)
            p_rev = self.base_model.predict_proba(text2, text1)
            return self.decoder.decode_pair(p_dir, p_rev)
        raise NotImplementedError("Base model does not implement predict_proba.")

    def decode_pair(self, p_dir: Sequence[float], p_rev: Sequence[float]) -> Tuple[int, int]:
        return self.decoder.decode_pair(p_dir, p_rev)

    def decode_pairs(self, probs_dir: Any, probs_rev: Any) -> Tuple[Any, Any]:
        return self.decoder.decode_pairs(probs_dir, probs_rev)

    def to(self, device: Any):
        if hasattr(self.base_model, "to"):
            self.base_model.to(device)
        return self

    def eval(self):
        if hasattr(self.base_model, "eval"):
            self.base_model.eval()
        return self

    def train(self, mode: bool = True):
        if hasattr(self.base_model, "train"):
            self.base_model.train(mode)
        return self
====
    def _resolve_predict_target(self) -> Any:
        """Resolve the model or encoder implementing predict_proba."""
        target = self.base_model
        if not hasattr(target, "predict_proba") and hasattr(target, "encoder"):
            target = target.encoder
        return target

    def predict_pair(
        self,
        text_a: Optional[str] = None,
        text_b: Optional[str] = None,
        text1: Optional[str] = None,
        text2: Optional[str] = None,
    ) -> Tuple[int, int]:
        """Predict directional pair with 100% SoftCons guarantee.

        Delegates to wrapped model (or its underlying encoder) to obtain
        directional probabilities for (text_a, text_b) and (text_b, text_a),
        and applies post-hoc Symmetrized Decoding.

        Args:
            text_a: Premise text. Also accepts alias 'text1'.
            text_b: Hypothesis text. Also accepts alias 'text2'.
            text1: Backward compatibility alias for text_a.
            text2: Backward compatibility alias for text_b.

        Returns:
            Tuple of discrete predicted class indices (pred_dir, pred_rev).
        """
        first = text_a if text_a is not None else text1
        second = text_b if text_b is not None else text2

        if first is None or second is None:
            raise ValueError(
                "predict_pair requires two text inputs: text_a (or text1) and text_b (or text2)."
            )

        target = self._resolve_predict_target()
        if hasattr(target, "predict_proba"):
            p_dir = target.predict_proba(first, second)
            p_rev = target.predict_proba(second, first)
            return self.decoder.decode_pair(p_dir, p_rev)

        raise NotImplementedError(
            f"Neither base model {type(self.base_model).__name__} nor its encoder implements predict_proba."
        )

    def predict_proba(self, text1: str, text2: str) -> Any:
        """Generate probabilities delegating to wrapped model or its encoder."""
        target = self._resolve_predict_target()
        if hasattr(target, "predict_proba"):
            return target.predict_proba(text1, text2)
        raise NotImplementedError(
            f"Neither base model {type(self.base_model).__name__} nor its encoder implements predict_proba."
        )

    def predict(self, text1: str, text2: str) -> str:
        """Predict discrete label string delegating to wrapped model or its encoder."""
        target = self._resolve_predict_target()
        if hasattr(target, "predict"):
            return target.predict(text1, text2)
        probs = self.predict_proba(text1, text2)
        idx = probs.index(max(probs))
        return ID_TO_LABEL[idx]

    def decode_pair(self, p_dir: Sequence[float], p_rev: Sequence[float]) -> Tuple[int, int]:
        return self.decoder.decode_pair(p_dir, p_rev)

    def decode_pairs(self, probs_dir: Any, probs_rev: Any) -> Tuple[Any, Any]:
        return self.decoder.decode_pairs(probs_dir, probs_rev)

    def to(self, device: Any):
        if hasattr(self.base_model, "to"):
            self.base_model.to(device)
        return self

    def eval(self):
        if hasattr(self.base_model, "eval"):
            self.base_model.eval()
        return self

    def train(self, mode: bool = True):
        if hasattr(self.base_model, "train"):
            self.base_model.train(mode)
        return self
>>>>
```

---

### 3.3 Complementary Robustness: `src/models/siamese_crl.py`

In `SiameseCRLModel` (both PyTorch and Pure-Python classes):
Add `.predict_proba(text1, text2)` so that `SiameseCRLModel` can directly serve probability queries without requiring an unwrapping step:
```python
    def predict_proba(self, text1: str, text2: str) -> Any:
        """Forward predict_proba query to the underlying sequence encoder."""
        if hasattr(self.encoder, "predict_proba"):
            return self.encoder.predict_proba(text1, text2)
        raise NotImplementedError("Underlying encoder does not implement predict_proba.")
```

---

### 3.4 Unit Test Additions: `tests/test_models.py`

Add the following targeted tests to `tests/test_models.py`:

```python
    def test_cross_encoder_nli_eval_and_train_cpu(self):
        """Verify .eval() and .train() work on CrossEncoderNLI and MockCrossEncoder without PyTorch."""
        mock_enc = MockCrossEncoder()
        self.assertIs(mock_enc.eval(), mock_enc)
        self.assertFalse(mock_enc.training)
        self.assertIs(mock_enc.train(True), mock_enc)
        self.assertTrue(mock_enc.training)

        nli = CrossEncoderNLI("mock", is_mock=True)
        self.assertIs(nli.eval(), nli)
        self.assertFalse(nli.training)
        self.assertFalse(nli.mock_engine.training)
        self.assertIs(nli.train(True), nli)
        self.assertTrue(nli.training)
        self.assertTrue(nli.mock_engine.training)

        m_base = build_model("mock", method="baseline")
        self.assertIs(m_base.eval(), m_base)
        self.assertFalse(m_base.training)

    def test_symmetrized_model_wrapper_predict_pair_delegation(self):
        """Verify SymmetrizedModelWrapper delegates predict_pair for both sd and crl_sd methods."""
        # 1. method='sd'
        model_sd = build_model("mock", method="sd", mask_negative_other=True)
        p_dir, p_rev = model_sd.predict_pair("the red car", "a vehicle")
        self.assertIsInstance(p_dir, int)
        self.assertIsInstance(p_rev, int)
        self.assertEqual(p_rev, REV_INDEX_MAP[p_dir])

        # 2. method='crl_sd'
        model_crl_sd = build_model("mock", method="crl_sd", mask_negative_other=True)
        p_d, p_r = model_crl_sd.predict_pair("the red car", "a vehicle")
        self.assertIsInstance(p_d, int)
        self.assertIsInstance(p_r, int)
        self.assertEqual(p_r, REV_INDEX_MAP[p_d])

        # 3. Keyword argument aliases
        p_d_named, p_r_named = model_crl_sd.predict_pair(text_a="the red car", text_b="a vehicle")
        self.assertEqual((p_d, p_r), (p_d_named, p_r_named))

        p_d_legacy, p_r_legacy = model_crl_sd.predict_pair(text1="the red car", text2="a vehicle")
        self.assertEqual((p_d, p_r), (p_d_legacy, p_r_legacy))

        # 4. Predict proba & discrete predict
        probs = model_crl_sd.predict_proba("the red car", "a vehicle")
        self.assertEqual(len(probs), 4)
        pred_label = model_crl_sd.predict("the red car", "a vehicle")
        self.assertIn(pred_label, ID_TO_LABEL.values())
```

---

## 4. Verification & Validation Protocol

Execute the following bash commands:
1. `python3 -m unittest discover -s tests -p "test_*.py" -v`
   - Expect: All 73+ tests pass with `OK`.
2. Verify `CrossEncoderNLI.eval()`:
   ```bash
   python3 -c "from src.models.factory import build_model; m = build_model('mock', method='baseline'); m.eval(); print('eval OK')"
   ```
3. Verify `crl_sd.predict_pair()`:
   ```bash
   python3 -c "from src.models.factory import build_model; m = build_model('mock', method='crl_sd'); d, r = m.predict_pair('the red car', 'a vehicle'); print(f'crl_sd predict_pair OK: {d}, {r}')"
   ```
