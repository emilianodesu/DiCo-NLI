# Investigation Report: Siamese CRL Ablation & Symmetrized Decoding Control

- **Agent**: Explorer 2 (`explorer_m2_it2_2`)
- **Role**: Read-only Investigation & Blueprint Architecture
- **Target**: Milestone 2 Iteration 2 (Siamese CRL Ablation & Symmetrized Decoding Control)
- **Target Files**:
  - `src/models/siamese_crl.py`
  - `src/models/factory.py`
  - `tests/test_models.py`
- **Recipient**: Orchestrator (`df6ce377-6407-4a2d-bd18-829b36e20607`), Worker (`worker_m2`)
- **Date**: 2026-11-01T23:14:00Z

---

## 1. Executive Summary

During Milestone 2 review, `challenger_m2_2` and `reviewer_m2_1` identified critical architectural defects in `SiameseCRLModel`:
1. **Inability to perform Pure CRL Ablation (Finding F-01 / Defect 3)**:
   In `src/models/siamese_crl.py`, `SiameseCRLModel.forward` unconditionally applied Symmetrized Decoding during evaluation (`eval()` mode) because of `or not self.training` in PyTorch mode (line 239) and unconditional execution in pure Python mode (line 391). Consequently, pure CRL evaluation was impossible: it always returned 100% `SoftCons` through post-hoc projection, preventing researchers from isolating the consistency learned by neural representations under Consistency Regularization Loss (CRL).
2. **Missing Pair & Probability Inference API on `SiameseCRLModel` (Defect 2 / Finding F-02)**:
   `SiameseCRLModel` did not implement `predict_proba`, `predict_pair`, `predict_logits`, or `predict`. When `build_model(method='crl_sd')` returned `SymmetrizedModelWrapper(SiameseCRLModel)`, calling `.predict_pair()` raised `NotImplementedError: Base model does not implement predict_proba`. Furthermore, users could not call `.predict_pair()` directly on `SiameseCRLModel`.

This report provides the mathematical rationale, architectural design, and exact line-by-line patch blueprint for `worker_m2` to cleanly resolve these issues across both PyTorch and Pure-Python execution paths.

---

## 2. Problem Analysis & Evidence Chain

### 2.1 Flawed Evaluation Condition in `forward()`

#### PyTorch Execution Path (`HAS_TORCH=True`):
In `src/models/siamese_crl.py` lines 238-240:
```python
if apply_symmetrized_decoding or not self.training:
    # Symmetrized decoding projection applied!
```
- When a model is placed in evaluation mode (`model.eval()`), PyTorch sets `self.training = False`.
- Therefore, `not self.training` evaluates to `True`.
- Even when the caller explicitly passes `apply_symmetrized_decoding=False`, the boolean condition `False or True` evaluates to `True`.
- As observed by `challenger_m2_2`:
  ```python
  model.eval()
  out = model.forward(..., apply_symmetrized_decoding=False)
  print(out.sym_probs_dir is not None) # -> True! (SD was forced!)
  ```

#### Pure-Python Execution Path (`HAS_TORCH=False`):
In `src/models/siamese_crl.py` lines 391-394:
```python
sym_p_dir, sym_p_rev = symmetrized_decoding(probs_dir, probs_rev)
preds_dir = [_argmax_with_rev_tie_break(r) for r in sym_p_dir]
preds_rev = [_argmax_with_rev_tie_break(r) for r in sym_p_rev]
```
The parameter `apply_symmetrized_decoding` was present in the function signature but completely ignored in the body. Symmetrized Decoding was executed unconditionally.

### 2.2 Impact on Research & ACL Paper (SemEval-2027 Task 2)

Requirements R2 and R4 in `PROJECT.md` require evaluating:
1. **Baseline**: Standard Cross-Encoder (unidirectional).
2. **BSA**: Bi-directional Symmetrical Augmentation.
3. **Pure CRL**: Siamese dual-stream with Consistency Regularization Loss (no post-hoc projection).
4. **SD**: Baseline + Symmetrized Decoding.
5. **CRL + SD**: Siamese CRL + Symmetrized Decoding.

Under the flawed implementation:
- Pure CRL in `eval()` mode always output projected probabilities `sym_probs_dir`.
- Pure CRL and CRL + SD produced identical 100% `SoftCons`.
- Reviewers and readers of the ACL paper would conclude that CRL loss had no measurable impact, or that the experimental setup was fraudulent/conflated.

### 2.3 Broken Model Factory & Missing Inference APIs

In `src/models/factory.py` lines 209-216:
```python
def predict_pair(self, text1: str, text2: str) -> Tuple[int, int]:
    """Predict directional pair with 100% SoftCons guarantee."""
    if hasattr(self.base_model, "predict_proba"):
        p_dir = self.base_model.predict_proba(text1, text2)
        p_rev = self.base_model.predict_proba(text2, text1)
        return self.decoder.decode_pair(p_dir, p_rev)
    raise NotImplementedError("Base model does not implement predict_proba.")
```
For `method="crl_sd"`, `self.base_model` is `SiameseCRLModel`. Since `SiameseCRLModel` lacked `predict_proba`:
```python
m = build_model('mock', method='crl_sd')
m.predict_pair("premise", "hypothesis")
# -> NotImplementedError: Base model does not implement predict_proba.
```
And for `method="crl"`:
```python
m = build_model('mock', method='crl')
m.predict_pair("premise", "hypothesis")
# -> AttributeError: 'SiameseCRLModel' object has no attribute 'predict_pair'
```

---

## 3. Architectural Design

### 3.1 Symmetrized Decoding Control Principle

We establish a clear hierarchy of control:
1. **Instance Configuration**: `SiameseCRLModel(..., apply_symmetrized_decoding=False)` sets default evaluation behavior. Pure CRL models default to `False`. Models designated for CRL+SD default to `True`.
2. **Explicit Call Override**: `forward(..., apply_symmetrized_decoding=...)` and `predict_pair(..., apply_symmetrized_decoding=...)` take precedence when not `None`:
   $$\text{should\_apply\_sd} = \begin{cases} \text{apply\_symmetrized\_decoding} & \text{if } \text{apply\_symmetrized\_decoding is not None} \\ \text{self.apply\_symmetrized\_decoding} & \text{otherwise (default False)} \end{cases}$$
3. **Decoupling from `training` mode**: The `self.training` flag controls model regularization (dropout, batchnorm updates), **NOT** the algebraic projection of inference predictions.

### 3.2 Output Contract Separation

| Output Field | Pure CRL (`apply_symmetrized_decoding=False`) | CRL + SD (`apply_symmetrized_decoding=True`) |
|---|---|---|
| `sym_probs_dir` | `None` | Projected tensor/list $0.5(p_{dir} + p_{rev} P)$ |
| `sym_probs_rev` | `None` | Projected tensor/list $0.5(p_{rev} + p_{dir} P)$ |
| `probs_dir` | Softmax distribution $p_{dir}$ | Softmax distribution $p_{dir}$ |
| `probs_rev` | Softmax distribution $p_{rev}$ | Softmax distribution $p_{rev}$ |
| `preds_dir` | Independent argmax $\operatorname{argmax}(p_{dir})$ | Equivariant tie-break $\operatorname{argmax}^*(p_{sym,dir})$ |
| `preds_rev` | Independent argmax $\operatorname{argmax}(p_{rev})$ | Equivariant tie-break $\operatorname{argmax}^*(p_{sym,rev})$ |
| `loss` | CRL Loss (if labels provided) | CRL Loss (if labels provided) |

### 3.3 Complete High-Level Inference Methods

`SiameseCRLModel` will implement:
1. `predict_logits(text1, text2)`: Delegates to underlying encoder.
2. `predict_proba(text1, text2)`: Delegates to underlying encoder, returning probability simplex.
3. `predict(text1, text2)`: Returns predicted class name string (`ID_TO_LABEL[argmax]`).
4. `predict_pair(text1, text2, apply_symmetrized_decoding=None, mask_negative_other=None)`:
   - Evaluates `p_dir = self.predict_proba(text1, text2)` and `p_rev = self.predict_proba(text2, text1)`.
   - If `should_apply_sd`: applies `symmetrized_decoding` and equivariant tie-breaking.
   - If not `should_apply_sd`: returns raw independent discrete predictions `(argmax(p_dir), argmax(p_rev))`.
5. `decode_pair(p_dir, p_rev, apply_symmetrized_decoding=None, mask_negative_other=None)`:
   - Direct probability vector pair decoding.
6. `decode_pairs(probs_dir, probs_rev, apply_symmetrized_decoding=None, mask_negative_other=None)`:
   - Vectorized batch probability decoding.

---

## 4. Exact Code Patch Blueprint for Worker

### Target 1: `src/models/siamese_crl.py`

#### Edit 1.1: Add `_argmax_scalar` helper function (around line 43)
```python
def _argmax_scalar(p: Any) -> int:
    """Extract scalar integer argmax from 1D sequence or tensor."""
    if hasattr(p, "argmax"):
        v = p.argmax()
        return int(v.item() if hasattr(v, "item") else v)
    seq = list(p)
    return int(seq.index(max(seq)))
```

#### Edit 1.2: Update PyTorch `SiameseCRLModel.__init__` (around line 100)
```python
<<<<
        def __init__(
            self,
            encoder: Any,
            divergence: str = "sym_kl",
            lambda_cons: float = 1.0,
            warmup_epochs: int = 1,
            total_epochs: int = 5,
            concatenate_streams: bool = False,
            label_smoothing: float = 0.0,
        ) -> None:
            super().__init__()
            self.encoder = encoder
            self.concatenate_streams = concatenate_streams
            self.crl_loss_fn = ConsistencyRegularizationLoss(
====
        def __init__(
            self,
            encoder: Any,
            divergence: str = "sym_kl",
            lambda_cons: float = 1.0,
            warmup_epochs: int = 1,
            total_epochs: int = 5,
            concatenate_streams: bool = False,
            label_smoothing: float = 0.0,
            apply_symmetrized_decoding: bool = False,
            mask_negative_other: bool = False,
        ) -> None:
            super().__init__()
            self.encoder = encoder
            self.concatenate_streams = concatenate_streams
            self.apply_symmetrized_decoding = apply_symmetrized_decoding
            self.mask_negative_other = mask_negative_other
            self.crl_loss_fn = ConsistencyRegularizationLoss(
>>>>
```

#### Edit 1.3: Update PyTorch `SiameseCRLModel.forward` (around lines 163-165 and 239-261)
```python
<<<<
            epoch: float = 1.0,
            apply_symmetrized_decoding: bool = True,
            # Batch mapping dictionary
            batch: Optional[Dict[str, Any]] = None,
        ) -> SiameseCRLOutput:
====
            epoch: float = 1.0,
            apply_symmetrized_decoding: Optional[bool] = None,
            mask_negative_other: Optional[bool] = None,
            # Batch mapping dictionary
            batch: Optional[Dict[str, Any]] = None,
        ) -> SiameseCRLOutput:
>>>>
```

And in Step 4 of PyTorch `forward`:
```python
<<<<
            # 4. Symmetrized Decoding (SD) projection
            sym_probs_dir = None
            sym_probs_rev = None
            preds_dir = None
            preds_rev = None

            if apply_symmetrized_decoding or not self.training:
                # p_sym_dir = 0.5 * (p_dir + p_rev @ P)
                # p_sym_rev = 0.5 * (p_rev + p_dir @ P)
                p_rev_trans = self.transposition(probs_rev)
                p_dir_trans = self.transposition(probs_dir)

                sym_probs_dir = 0.5 * (probs_dir + p_rev_trans)
                sym_probs_rev = 0.5 * (probs_rev + p_dir_trans)

                preds_dir = torch.tensor(
                    [_argmax_with_rev_tie_break(row.tolist()) for row in sym_probs_dir],
                    device=sym_probs_dir.device,
                    dtype=torch.long,
                )
                preds_rev = torch.tensor(
                    [_argmax_with_rev_tie_break(row.tolist()) for row in sym_probs_rev],
                    device=sym_probs_rev.device,
                    dtype=torch.long,
                )
            else:
                preds_dir = torch.argmax(logits_dir, dim=-1)
                preds_rev = torch.argmax(logits_rev, dim=-1)
====
            # 4. Symmetrized Decoding (SD) projection or raw unprojected predictions
            should_apply_sd = (
                apply_symmetrized_decoding
                if apply_symmetrized_decoding is not None
                else getattr(self, "apply_symmetrized_decoding", False)
            )
            should_mask_neg = (
                mask_negative_other
                if mask_negative_other is not None
                else getattr(self, "mask_negative_other", False)
            )

            sym_probs_dir = None
            sym_probs_rev = None

            if should_apply_sd:
                # p_sym_dir = 0.5 * (p_dir + p_rev @ P)
                # p_sym_rev = 0.5 * (p_rev + p_dir @ P)
                p_rev_trans = self.transposition(probs_rev)
                p_dir_trans = self.transposition(probs_dir)

                sym_probs_dir = 0.5 * (probs_dir + p_rev_trans)
                sym_probs_rev = 0.5 * (probs_rev + p_dir_trans)

                if should_mask_neg:
                    sym_probs_dir = sym_probs_dir.clone()
                    sym_probs_rev = sym_probs_rev.clone()
                    sym_probs_dir[:, 3] = 0.0
                    sym_probs_rev[:, 3] = 0.0
                    sd = sym_probs_dir.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                    sr = sym_probs_rev.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                    sym_probs_dir = sym_probs_dir / sd
                    sym_probs_rev = sym_probs_rev / sr

                preds_dir = torch.tensor(
                    [_argmax_with_rev_tie_break(row.tolist()) for row in sym_probs_dir],
                    device=sym_probs_dir.device,
                    dtype=torch.long,
                )
                preds_rev = torch.tensor(
                    [_argmax_with_rev_tie_break(row.tolist()) for row in sym_probs_rev],
                    device=sym_probs_rev.device,
                    dtype=torch.long,
                )
            else:
                preds_dir = torch.argmax(logits_dir, dim=-1)
                preds_rev = torch.argmax(logits_rev, dim=-1)
>>>>
```

#### Edit 1.4: Add inference APIs to PyTorch `SiameseCRLModel`
```python
        def predict_logits(self, text1: str, text2: str) -> Any:
            """Generate logits for an individual text pair using the underlying encoder."""
            if hasattr(self.encoder, "predict_logits"):
                return self.encoder.predict_logits(text1, text2)
            raise NotImplementedError("Underlying encoder does not implement predict_logits.")

        def predict_proba(self, text1: str, text2: str) -> Any:
            """Generate probabilities for an individual text pair using the underlying encoder."""
            if hasattr(self.encoder, "predict_proba"):
                return self.encoder.predict_proba(text1, text2)
            raise NotImplementedError("Underlying encoder does not implement predict_proba.")

        def predict(self, text1: str, text2: str) -> str:
            """Predict discrete label string for an individual text pair."""
            if hasattr(self.encoder, "predict"):
                return self.encoder.predict(text1, text2)
            probs = self.predict_proba(text1, text2)
            if hasattr(probs, "tolist"):
                probs = probs.tolist()
            idx = probs.index(max(probs))
            from .cross_encoder import ID_TO_LABEL
            return ID_TO_LABEL[idx]

        def predict_pair(
            self,
            text1: str,
            text2: str,
            apply_symmetrized_decoding: Optional[bool] = None,
            mask_negative_other: Optional[bool] = None,
        ) -> Tuple[int, int]:
            """Predict directional discrete class indices for (text1, text2) and (text2, text1).

            Args:
                text1: Premise text.
                text2: Hypothesis text.
                apply_symmetrized_decoding: If True, apply Symmetrized Decoding projection.
                    If False, return raw unprojected direct and reverse predictions (argmax).
                    If None, defaults to self.apply_symmetrized_decoding (default False).
                mask_negative_other: Whether to mask NEGATIVE_OTHER during symmetrized decoding.

            Returns:
                Tuple[int, int]: (pred_dir, pred_rev)
            """
            should_apply_sd = (
                apply_symmetrized_decoding
                if apply_symmetrized_decoding is not None
                else getattr(self, "apply_symmetrized_decoding", False)
            )
            should_mask_neg = (
                mask_negative_other
                if mask_negative_other is not None
                else getattr(self, "mask_negative_other", False)
            )
            p_dir = self.predict_proba(text1, text2)
            p_rev = self.predict_proba(text2, text1)

            if should_apply_sd:
                p_s_dir, p_s_rev = symmetrized_decoding(
                    p_dir, p_rev, mask_negative_other=should_mask_neg
                )
                pred_dir = _argmax_with_rev_tie_break(p_s_dir)
                pred_rev = _argmax_with_rev_tie_break(p_s_rev)
            else:
                pred_dir = _argmax_scalar(p_dir)
                pred_rev = _argmax_scalar(p_rev)

            return pred_dir, pred_rev

        def decode_pair(
            self,
            p_dir: Sequence[float],
            p_rev: Sequence[float],
            apply_symmetrized_decoding: Optional[bool] = None,
            mask_negative_other: Optional[bool] = None,
        ) -> Tuple[int, int]:
            """Decode directional probability vectors into discrete class indices."""
            should_apply_sd = (
                apply_symmetrized_decoding
                if apply_symmetrized_decoding is not None
                else getattr(self, "apply_symmetrized_decoding", False)
            )
            should_mask_neg = (
                mask_negative_other
                if mask_negative_other is not None
                else getattr(self, "mask_negative_other", False)
            )
            if should_apply_sd:
                p_s_dir, p_s_rev = symmetrized_decoding(
                    p_dir, p_rev, mask_negative_other=should_mask_neg
                )
                return _argmax_with_rev_tie_break(p_s_dir), _argmax_with_rev_tie_break(p_s_rev)
            else:
                return _argmax_scalar(p_dir), _argmax_scalar(p_rev)

        def decode_pairs(
            self,
            probs_dir: Any,
            probs_rev: Any,
            apply_symmetrized_decoding: Optional[bool] = None,
            mask_negative_other: Optional[bool] = None,
        ) -> Tuple[Any, Any]:
            """Decode batch of direct and reverse probability arrays into discrete class indices."""
            should_apply_sd = (
                apply_symmetrized_decoding
                if apply_symmetrized_decoding is not None
                else getattr(self, "apply_symmetrized_decoding", False)
            )
            should_mask_neg = (
                mask_negative_other
                if mask_negative_other is not None
                else getattr(self, "mask_negative_other", False)
            )
            if should_apply_sd:
                from .decoding import SymmetrizedDecoder
                decoder = SymmetrizedDecoder(mask_negative_other=should_mask_neg)
                return decoder.decode_pairs(probs_dir, probs_rev)
            else:
                if HAS_TORCH and torch is not None and isinstance(probs_dir, torch.Tensor):
                    return torch.argmax(probs_dir, dim=-1), torch.argmax(probs_rev, dim=-1)
                try:
                    import numpy as np
                    if isinstance(probs_dir, np.ndarray):
                        return np.argmax(probs_dir, axis=-1), np.argmax(probs_rev, axis=-1)
                except ImportError:
                    pass
                preds_dir = [r.index(max(r)) for r in probs_dir]
                preds_rev = [r.index(max(r)) for r in probs_rev]
                return preds_dir, preds_rev
```

#### Edit 1.5: Update Pure-Python `SiameseCRLModel.__init__` and `forward`
In `SiameseCRLModel.__init__` (Pure Python, around line 290):
```python
<<<<
        def __init__(
            self,
            encoder: Any,
            divergence: str = "sym_kl",
            lambda_cons: float = 1.0,
            warmup_epochs: int = 1,
            total_epochs: int = 5,
            concatenate_streams: bool = False,
            label_smoothing: float = 0.0,
        ) -> None:
            self.encoder = encoder
            self.divergence = divergence
            self.lambda_cons = lambda_cons
            self.warmup_epochs = warmup_epochs
            self.total_epochs = total_epochs
            self.concatenate_streams = concatenate_streams
            self.training = True
====
        def __init__(
            self,
            encoder: Any,
            divergence: str = "sym_kl",
            lambda_cons: float = 1.0,
            warmup_epochs: int = 1,
            total_epochs: int = 5,
            concatenate_streams: bool = False,
            label_smoothing: float = 0.0,
            apply_symmetrized_decoding: bool = False,
            mask_negative_other: bool = False,
        ) -> None:
            self.encoder = encoder
            self.divergence = divergence
            self.lambda_cons = lambda_cons
            self.warmup_epochs = warmup_epochs
            self.total_epochs = total_epochs
            self.concatenate_streams = concatenate_streams
            self.apply_symmetrized_decoding = apply_symmetrized_decoding
            self.mask_negative_other = mask_negative_other
            self.training = True
>>>>
```

In `SiameseCRLModel.forward` (Pure Python, around line 357 and 391):
```python
<<<<
            epoch: float = 1.0,
            apply_symmetrized_decoding: bool = True,
            batch: Optional[Dict[str, Any]] = None,
        ) -> SiameseCRLOutput:
====
            epoch: float = 1.0,
            apply_symmetrized_decoding: Optional[bool] = None,
            mask_negative_other: Optional[bool] = None,
            batch: Optional[Dict[str, Any]] = None,
        ) -> SiameseCRLOutput:
>>>>
```
And around line 391:
```python
<<<<
            sym_p_dir, sym_p_rev = symmetrized_decoding(probs_dir, probs_rev)
            preds_dir = [_argmax_with_rev_tie_break(r) for r in sym_p_dir]
            preds_rev = [_argmax_with_rev_tie_break(r) for r in sym_p_rev]

            return SiameseCRLOutput(
                loss=loss_val,
                logits_dir=logits_dir,
                logits_rev=logits_rev,
                probs_dir=probs_dir,
                probs_rev=probs_rev,
                sym_probs_dir=sym_p_dir,
                sym_probs_rev=sym_p_rev,
                preds_dir=preds_dir,
                preds_rev=preds_rev,
                metrics=metrics,
            )
====
            should_apply_sd = (
                apply_symmetrized_decoding
                if apply_symmetrized_decoding is not None
                else getattr(self, "apply_symmetrized_decoding", False)
            )
            should_mask_neg = (
                mask_negative_other
                if mask_negative_other is not None
                else getattr(self, "mask_negative_other", False)
            )

            if should_apply_sd:
                sym_p_dir, sym_p_rev = symmetrized_decoding(
                    probs_dir, probs_rev, mask_negative_other=should_mask_neg
                )
                preds_dir = [_argmax_with_rev_tie_break(r) for r in sym_p_dir]
                preds_rev = [_argmax_with_rev_tie_break(r) for r in sym_p_rev]
            else:
                sym_p_dir, sym_p_rev = None, None
                preds_dir = [r.index(max(r)) for r in probs_dir]
                preds_rev = [r.index(max(r)) for r in probs_rev]

            return SiameseCRLOutput(
                loss=loss_val,
                logits_dir=logits_dir,
                logits_rev=logits_rev,
                probs_dir=probs_dir,
                probs_rev=probs_rev,
                sym_probs_dir=sym_p_dir,
                sym_probs_rev=sym_p_rev,
                preds_dir=preds_dir,
                preds_rev=preds_rev,
                metrics=metrics,
            )
>>>>
```

Add the identical inference methods (`predict_logits`, `predict_proba`, `predict`, `predict_pair`, `decode_pair`, `decode_pairs`) to Pure-Python `SiameseCRLModel`.

---

### Target 2: `src/models/factory.py`

In `build_model`:
```python
<<<<
        elif clean_method == "crl":
            return SiameseCRLModel(encoder=base_encoder)
        elif clean_method == "sd":
            return SymmetrizedModelWrapper(base_encoder, mask_negative_other=mask_negative_other)
        elif clean_method == "crl_sd":
            crl_model = SiameseCRLModel(encoder=base_encoder)
            return SymmetrizedModelWrapper(crl_model, mask_negative_other=mask_negative_other)
====
        elif clean_method == "crl":
            return SiameseCRLModel(
                encoder=base_encoder,
                apply_symmetrized_decoding=False,
                mask_negative_other=mask_negative_other,
            )
        elif clean_method == "sd":
            return SymmetrizedModelWrapper(base_encoder, mask_negative_other=mask_negative_other)
        elif clean_method == "crl_sd":
            crl_model = SiameseCRLModel(
                encoder=base_encoder,
                apply_symmetrized_decoding=True,
                mask_negative_other=mask_negative_other,
            )
            return SymmetrizedModelWrapper(crl_model, mask_negative_other=mask_negative_other)
>>>>
```
And repeat the same clean configuration in the live backbone branch (lines 321-327).

---

### Target 3: Unit Tests in `tests/test_models.py`

Add the following unit test cases to `TestSiameseCRLModel` and `TestModelFactory`:

```python
    def test_siamese_crl_pure_ablation(self):
        """Verify that SiameseCRLModel with apply_symmetrized_decoding=False disables SD."""
        mock_enc = MockCrossEncoder()
        siamese = SiameseCRLModel(
            encoder=mock_enc,
            divergence="sym_kl",
            apply_symmetrized_decoding=False,
        )
        siamese.eval()

        tok = MockTokenizer()
        b_dir = tok(["Pope Francis"], ["the pope"])
        b_rev = tok(["the pope"], ["Pope Francis"])

        # Forward in eval mode without SD
        output = siamese.forward(
            input_ids_dir=b_dir["input_ids"],
            input_ids_rev=b_rev["input_ids"],
            apply_symmetrized_decoding=False,
        )
        self.assertIsNone(output.sym_probs_dir, "Pure CRL must not compute sym_probs_dir")
        self.assertIsNone(output.sym_probs_rev, "Pure CRL must not compute sym_probs_rev")
        self.assertIsNotNone(output.probs_dir)
        self.assertIsNotNone(output.preds_dir)

    def test_siamese_crl_predict_pair_and_proba(self):
        """Verify predict_pair and predict_proba on SiameseCRLModel."""
        mock_enc = MockCrossEncoder()
        siamese = SiameseCRLModel(encoder=mock_enc)

        # 1. predict_proba
        probs = siamese.predict_proba("the red car", "a vehicle")
        self.assertEqual(len(probs), 4)
        self.assertAlmostEqual(sum(probs), 1.0, places=4)

        # 2. predict_pair with pure CRL
        p_d_raw, p_r_raw = siamese.predict_pair("the red car", "a vehicle", apply_symmetrized_decoding=False)
        self.assertIsInstance(p_d_raw, int)
        self.assertIsInstance(p_r_raw, int)

        # 3. predict_pair with SD enabled
        p_d_sd, p_r_sd = siamese.predict_pair("the red car", "a vehicle", apply_symmetrized_decoding=True)
        self.assertEqual(p_r_sd, REV_INDEX_MAP[p_d_sd])

    def test_crl_sd_factory_predict_pair(self):
        """Verify build_model(method='crl_sd') can invoke predict_pair without error."""
        model = build_model("mock", method="crl_sd")
        p_dir, p_rev = model.predict_pair("Pope Francis", "the pope")
        self.assertEqual(p_rev, REV_INDEX_MAP[p_dir])
```

---

## 5. Verification Protocol

The implementer (`worker_m2`) can verify the complete patch using these deterministic checks:

1. **Verify pure CRL ablation disabling SD**:
   ```bash
   python3 -c "
   from src.models.siamese_crl import SiameseCRLModel
   from src.models.cross_encoder import MockCrossEncoder
   model = SiameseCRLModel(encoder=MockCrossEncoder())
   model.eval()
   out = model.forward(input_ids_dir=[[1, 2, 3]], input_ids_rev=[[3, 2, 1]], apply_symmetrized_decoding=False)
   assert out.sym_probs_dir is None, 'Failed: sym_probs_dir should be None'
   assert out.sym_probs_rev is None, 'Failed: sym_probs_rev should be None'
   print('Pure CRL ablation check: PASSED')
   "
   ```

2. **Verify `predict_pair` on `crl` and `crl_sd`**:
   ```bash
   python3 -c "
   from src.models.factory import build_model
   m_crl = build_model('mock', method='crl')
   p_d, p_r = m_crl.predict_pair('the red car', 'a vehicle')
   print(f'crl predict_pair: {p_d}, {p_r}')

   m_crl_sd = build_model('mock', method='crl_sd')
   p_d_sd, p_r_sd = m_crl_sd.predict_pair('the red car', 'a vehicle')
   print(f'crl_sd predict_pair: {p_d_sd}, {p_r_sd}')
   from src.models.losses import REV_INDEX_MAP
   assert p_r_sd == REV_INDEX_MAP[p_d_sd], 'crl_sd did not satisfy Rev symmetry'
   print('crl and crl_sd predict_pair checks: PASSED')
   "
   ```

3. **Verify Full Unit Test Suite**:
   ```bash
   python3 -m unittest discover -s tests -p "test_*.py" -v
   ```
   *Expected Result*: All 73+ tests pass with 0 errors and 0 failures in $<0.1$s on CPU.
