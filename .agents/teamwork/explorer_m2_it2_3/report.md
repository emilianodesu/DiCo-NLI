# Exploration Report: Loss Edge Cases & Decoupled Verification

- **Author**: Explorer 3 (`explorer_m2_it2_3`)
- **Recipient**: Orchestrator (`df6ce377-6407-4a2d-bd18-829b36e20607`) & Worker (`worker_m2_it2`)
- **Date**: 2026-11-01T23:15:00Z
- **Scope**: Loss Edge Cases (`cross_entropy_pure` target=-100 ignore_index), Decoupled Verification (`SymmetrizedDecoder` mask_negative_other configurability & documentation), and unit tests in `tests/test_models.py`.

---

## 1. Executive Summary

This investigation resolves the edge cases identified during Milestone 2 adversarial review regarding:
1. **PyTorch `ignore_index=-100` parity in pure-Python categorical cross-entropy**:
   - `cross_entropy_pure(logits, -100)` previously raised `IndexError: list index out of range`.
   - In batch loss computation, masked targets must contribute `0.0` to the loss numerator and be excluded from the sample count denominator, dividing *only* by the count of non-ignored entries ($N_{valid}$).
   - In `ConsistencyRegularizationLoss` fallback, deriving `yr` when `yd` contains `-100` previously raised `KeyError: -100` because `-100` was not in `REV_INDEX_MAP`.
2. **`SymmetrizedDecoder` `mask_negative_other` configurability and SemEval metric alignment**:
   - Under official SemEval-2027 consistency metrics (`compute_official_metrics` in `src/metrics/metrics.py`), a pair is evaluated as `soft_ok = True` only if `pred_1 in REVERSIBLE_LABELS` and `pred_2 == REVERSE_LABEL_MAP[pred_1]`.
   - Because `NEGATIVE_OTHER` (index 3) is non-directional, predicting `NEGATIVE_OTHER` on a reversible pair yields `SoftCons = 0.0`.
   - With `mask_negative_other=True`, class 3 is zeroed out and classes $\{0, 1, 2\}$ are renormalized, strictly guaranteeing `SoftCons = 1.0` on all reversible pairs.
   - `mask_negative_other` is now made fully configurable at `__init__`, dynamic attribute, and per-call override levels, and comprehensively documented across `SymmetrizedDecoder`, `symmetrized_decoding`, and `SymmetrizedModelWrapper`.
3. **Comprehensive Unit Test Additions**:
   - 5 new test methods for `tests/test_models.py` verifying single-item ignore_index, batch non-ignored divisor normalization, CRL loss with masked labels, dynamic/per-call masking configuration, and SemEval official metric integration.
4. **Concrete Worker Blueprint**:
   - Machine-applicable patch and line-by-line before/after replacement blocks provided.

---

## 2. Technical Investigation & Root Cause Analysis

### 2.1 Issue 1: `cross_entropy_pure` and `target=-100`
- **Location**: `src/models/losses.py`, lines 87-91.
- **Observed Behavior**:
  ```python
  def cross_entropy_pure(logits: Sequence[float], target: int) -> float:
      probs = softmax_pure(logits)
      p = max(probs[target], 1e-15)
      return -math.log(p)
  ```
  When `target == -100`, Python evaluates `probs[-100]`. Since `probs` has length 4, an `IndexError: list index out of range` is raised.
- **Batch Divisor Parity**:
  In PyTorch:
  $$\mathcal{L}_{\text{CE}} = \frac{1}{\sum_{i} \mathbb{I}[y_i \neq -100]} \sum_{i: y_i \neq -100} -\log(p_{i, y_i})$$
  If a batch of $B$ instances contains $K$ ignored entries, the loss must be divided by $B - K$, not $B$. If all $B$ instances are ignored ($B - K = 0$), the loss evaluates to $0.0$.
- **Cascading `KeyError` in `ConsistencyRegularizationLoss`**:
  In `src/models/losses.py` line 501:
  ```python
  if yr is None:
      yr = [REV_INDEX_MAP[y] for y in yd]
  ```
  If `yd` contains `-100`, accessing `REV_INDEX_MAP[-100]` raises `KeyError: -100`.
  Furthermore, `loss_ce_dir` and `loss_ce_rev` were computed by dividing by the total batch size `b` (lines 519-520), which diluted the loss when masked elements were present.

### 2.2 Issue 2: `SymmetrizedDecoder` `mask_negative_other` Configurability & SemEval Metrics
- **Location**: `src/models/decoding.py` (lines 236-290) and `src/models/factory.py` (lines 194-237).
- **Observed Behavior**:
  - `SymmetrizedDecoder.__init__` accepted `mask_negative_other: bool = False`, but `decode_pair`, `decode_pairs`, and `project_probabilities` did not allow per-call overriding (`mask_negative_other: Optional[bool] = None`).
  - `SymmetrizedModelWrapper` in `src/models/factory.py` did not expose `mask_negative_other` in `predict_pair`, `decode_pair`, or `decode_pairs`.
  - Documentation did not explain the mathematical necessity of `mask_negative_other=True` for official SemEval `SoftCons = 1.0` evaluation.
- **Mathematical Rationale**:
  For reversible phrase pairs (e.g., $(A, B)$ with label `FORWARD_ENTAILMENT`), the ground truth belongs to $\mathcal{Y}_{rev} = \{\text{BACKWARD}, \text{EQUIV}, \text{FORWARD}\}$.
  Under `src/metrics/metrics.py`, `compute_official_metrics` evaluates:
  ```python
  soft_ok = (pred_1 in REVERSIBLE_LABELS and pred_2 == REVERSE_LABEL_MAP.get(pred_1))
  ```
  If a model outputs high probability for `NEGATIVE_OTHER` (index 3), post-hoc projection preserves equivariance ($p_{sym,rev,3} = p_{sym,dir,3}$), but `pred_1 == 'NEGATIVE_OTHER' \notin \text{REVERSIBLE_LABELS}`, resulting in `soft_ok = False` (`SoftCons = 0.0`).
  Enabling `mask_negative_other=True` zeroes out $p_3$ and renormalizes $p_0, p_1, p_2$, ensuring predictions remain within $\mathcal{Y}_{rev}$ and strictly guaranteeing `SoftCons = 1.0`.
  Conversely, for uncurated or negative-sampled datasets where `NEGATIVE_OTHER` is a valid label, `mask_negative_other=False` must remain available.

---

## 3. Worker Implementation Blueprint

### File 1: `src/models/losses.py`

#### Change 1.1: Replace `cross_entropy_pure` (lines 87-91)
```python
def cross_entropy_pure(
    logits: Union[Sequence[float], Sequence[Sequence[float]]],
    target: Union[int, Sequence[int]],
    ignore_index: int = -100,
    reduction: str = "mean",
) -> float:
    """Pure-Python Cross-Entropy Loss with PyTorch ignore_index parity.

    Computes categorical cross-entropy loss with support for single sample (1D logits)
    or batch of samples (2D logits). When target matches ignore_index (default -100)
    or is an invalid out-of-bounds index, the masked entry is ignored, and the mean loss
    divides only by the count of non-ignored entries.

    Args:
        logits: 1D sequence of class logits (e.g. [2.0, 1.0, 0.5, 0.1]) or 2D batch of logits.
        target: Ground truth integer class index (for 1D) or sequence of class indices (for 2D).
        ignore_index: Target value that is masked and ignored (default -100, matching PyTorch).
        reduction: 'mean' (default) or 'sum'. Ignored entries are excluded from reduction.

    Returns:
        Scalar cross-entropy loss. If all entries are ignored, returns 0.0.
    """
    if not logits:
        return 0.0

    # 1D single sample case
    if isinstance(logits[0], (int, float)):
        if isinstance(target, (list, tuple)):
            if len(target) == 1:
                tgt = int(target[0])
            else:
                raise ValueError("Shape mismatch: 1D logits with multi-element target.")
        else:
            tgt = int(target)

        if tgt == ignore_index or tgt < 0 or tgt >= len(logits):
            return 0.0
        probs = softmax_pure(logits)
        p = max(probs[tgt], 1e-15)
        return -math.log(p)

    # 2D batch of samples case
    batch_targets: Sequence[int]
    if isinstance(target, (int, float)):
        batch_targets = [int(target)] * len(logits)
    else:
        batch_targets = target

    total_loss = 0.0
    non_ignored_count = 0

    for row_logits, row_tgt in zip(logits, batch_targets):
        if row_tgt == ignore_index or row_tgt < 0 or row_tgt >= len(row_logits):
            continue
        probs = softmax_pure(row_logits)
        p = max(probs[row_tgt], 1e-15)
        total_loss += -math.log(p)
        non_ignored_count += 1

    if reduction == "sum":
        return total_loss
    return (total_loss / non_ignored_count) if non_ignored_count > 0 else 0.0
```

#### Change 1.2: Update `consistency_regularization_loss_pure` (lines 135-142)
```python
    if target_rev is None:
        target_rev = REV_INDEX_MAP.get(target_dir, target_dir)

    loss_ce_dir = cross_entropy_pure(logits_dir, target_dir, ignore_index=-100)
    loss_ce_rev = cross_entropy_pure(logits_rev, target_rev, ignore_index=-100)

    valid_task = 0
    task_sum = 0.0
    if target_dir != -100 and 0 <= target_dir < len(logits_dir):
        task_sum += loss_ce_dir
        valid_task += 1
    if target_rev != -100 and 0 <= target_rev < len(logits_rev):
        task_sum += loss_ce_rev
        valid_task += 1
    loss_task = (task_sum / valid_task) if valid_task > 0 else 0.0
```

#### Change 1.3: Update `ConsistencyRegularizationLoss.__call__` in pure-Python fallback (lines 500-525)
```python
            if yr is None:
                yr = [REV_INDEX_MAP.get(y, y) for y in yd]

            total_ce_dir = 0.0
            total_ce_rev = 0.0
            valid_d = 0
            valid_r = 0
            total_cons = 0.0
            b = max(1, len(ld))

            for i in range(b):
                if yd[i] != -100 and 0 <= yd[i] < len(ld[i]):
                    total_ce_dir += cross_entropy_pure(ld[i], yd[i], ignore_index=-100)
                    valid_d += 1
                if yr[i] != -100 and 0 <= yr[i] < len(lr[i]):
                    total_ce_rev += cross_entropy_pure(lr[i], yr[i], ignore_index=-100)
                    valid_r += 1

                pd = softmax_pure(ld[i])
                pr = softmax_pure(lr[i])
                pr_trans = self.transposition(pr)
                total_cons += self.div_fn(pd, pr_trans)

            loss_ce_dir = (total_ce_dir / valid_d) if valid_d > 0 else 0.0
            loss_ce_rev = (total_ce_rev / valid_r) if valid_r > 0 else 0.0
            loss_task = self.task_weight * (loss_ce_dir + loss_ce_rev)
            loss_cons = total_cons / b
```

---

### File 2: `src/models/decoding.py`

#### Change 2.1: Enhance `SymmetrizedDecoder` class (lines 236-310)
```python
class SymmetrizedDecoder:
    """Post-hoc probability projection engine strictly guaranteeing SoftCons = 100%.

    The decoder projects directional prediction distributions (p_dir, p_rev)
    into coordinate-equivariant symmetrized distributions:
        p_sym_dir = 0.5 * (p_dir + p_rev @ P)
        p_sym_rev = 0.5 * (p_rev + p_dir @ P)
    where P is the 4x4 involutory permutation matrix encoding class reversal:
        P = [[0, 0, 1, 0],   # 0: BACKWARD <-> 2: FORWARD
             [0, 1, 0, 0],   # 1: EQUIVALENCE <-> 1: EQUIVALENCE
             [1, 0, 0, 0],   # 2: FORWARD <-> 0: BACKWARD
             [0, 0, 0, 1]]   # 3: NEGATIVE_OTHER <-> 3: NEGATIVE_OTHER

    Mathematical Invariant:
        p_sym_rev @ P = p_sym_dir
        (p_sym_rev)_Rev(c) == (p_sym_dir)_c for all c in {0, 1, 2, 3}.

    Configuring `mask_negative_other`:
        - `mask_negative_other=True`: Zeroes out the 4th class (index 3: NEGATIVE_OTHER)
          and renormalizes probabilities over the reversible classes {0: BWD, 1: EQUIV, 2: FWD}.
          Use this setting when evaluating on strictly reversible benchmark subsets where
          official SemEval consistency metrics require predictions in REVERSIBLE_LABELS
          to award SoftCons = 1.0.
        - `mask_negative_other=False` (default): Preserves all 4 classes including NEGATIVE_OTHER.
          Use this setting when evaluating unrestricted open-domain pairs where non-entailed
          negative relation is a valid prediction.

    Attributes:
        mask_negative_other (bool): Default masking configuration for this decoder instance.
        tie_break (str): Tie-breaking strategy ('symmetric' preserves equivariance).
    """

    def __init__(self, mask_negative_other: bool = False, tie_break: str = "symmetric") -> None:
        self.mask_negative_other = mask_negative_other
        self.tie_break = tie_break

    def project_probabilities(
        self,
        probs_dir: Any,
        probs_rev: Any,
        mask_negative_other: Optional[bool] = None,
    ) -> Tuple[Any, Any]:
        """Project raw direct and reverse probabilities into symmetrized space.

        Args:
            probs_dir: Direct pair probabilities (1D or 2D batch).
            probs_rev: Reverse pair probabilities (1D or 2D batch).
            mask_negative_other: If specified, overrides instance `self.mask_negative_other`.

        Returns:
            Tuple of (p_sym_dir, p_sym_rev) with same type and shape as inputs.
        """
        mask = self.mask_negative_other if mask_negative_other is None else mask_negative_other
        return symmetrized_decoding(probs_dir, probs_rev, mask_negative_other=mask)

    def decode_pair(
        self,
        p_dir: Sequence[float],
        p_rev: Sequence[float],
        mask_negative_other: Optional[bool] = None,
    ) -> Tuple[int, int]:
        """Decode single pair into discrete class indices.

        Args:
            p_dir: 1D probability distribution for direct pair (length 4).
            p_rev: 1D probability distribution for reverse pair (length 4).
            mask_negative_other: Optional per-call override for negative class masking.

        Returns:
            Tuple of (pred_dir, pred_rev) satisfying pred_rev == Rev(pred_dir).
        """
        p_s_dir, p_s_rev = self.project_probabilities(p_dir, p_rev, mask_negative_other=mask_negative_other)
        pred_dir = _argmax_with_rev_tie_break(p_s_dir)
        pred_rev = _argmax_with_rev_tie_break(p_s_rev)
        return pred_dir, pred_rev

    def decode_pairs(
        self,
        probs_dir: Any,
        probs_rev: Any,
        mask_negative_other: Optional[bool] = None,
    ) -> Tuple[Any, Any]:
        """Decode batch of direct and reverse probability arrays into discrete indices.

        Args:
            probs_dir: Batch of direct probabilities (N, 4).
            probs_rev: Batch of reverse probabilities (N, 4).
            mask_negative_other: Optional per-call override for negative class masking.

        Returns:
            Tuple of (preds_dir, preds_rev) with discrete class indices.
        """
        p_s_dir, p_s_rev = self.project_probabilities(probs_dir, probs_rev, mask_negative_other=mask_negative_other)

        if HAS_NUMPY and np is not None and isinstance(p_s_dir, np.ndarray):
            preds_dir = np.array([_argmax_with_rev_tie_break(row) for row in p_s_dir], dtype=np.int64)
            preds_rev = np.array([_argmax_with_rev_tie_break(row) for row in p_s_rev], dtype=np.int64)
            return preds_dir, preds_rev
        elif HAS_TORCH and torch is not None and isinstance(p_s_dir, torch.Tensor):
            preds_dir = torch.tensor(
                [_argmax_with_rev_tie_break(row.tolist()) for row in p_s_dir],
                device=p_s_dir.device,
                dtype=torch.long,
            )
            preds_rev = torch.tensor(
                [_argmax_with_rev_tie_break(row.tolist()) for row in p_s_rev],
                device=p_s_rev.device,
                dtype=torch.long,
            )
            return preds_dir, preds_rev
        else:
            preds_dir = [_argmax_with_rev_tie_break(row) for row in p_s_dir]
            preds_rev = [_argmax_with_rev_tie_break(row) for row in p_s_rev]
            return preds_dir, preds_rev

    def decode_batch(
        self,
        probs_dir: Any,
        probs_rev: Any,
        mask_negative_other: Optional[bool] = None,
    ) -> Tuple[Any, Any]:
        """Alias for decode_pairs."""
        return self.decode_pairs(probs_dir, probs_rev, mask_negative_other=mask_negative_other)

    def decode(
        self,
        probs_dir: Any,
        probs_rev: Any,
        mask_negative_other: Optional[bool] = None,
    ) -> Tuple[Any, Any]:
        """Smart dispatcher: auto-detects 1D vs 2D input with optional masking override."""
        if HAS_NUMPY and np is not None and isinstance(probs_dir, np.ndarray):
            if probs_dir.ndim == 1:
                return self.decode_pair(probs_dir, probs_rev, mask_negative_other=mask_negative_other)
            return self.decode_pairs(probs_dir, probs_rev, mask_negative_other=mask_negative_other)
        if HAS_TORCH and torch is not None and isinstance(probs_dir, torch.Tensor):
            if probs_dir.dim() == 1:
                return self.decode_pair(probs_dir.tolist(), probs_rev.tolist(), mask_negative_other=mask_negative_other)
            return self.decode_pairs(probs_dir, probs_rev, mask_negative_other=mask_negative_other)
        if not probs_dir:
            return (0, 0)
        if isinstance(probs_dir[0], (int, float)):
            return self.decode_pair(probs_dir, probs_rev, mask_negative_other=mask_negative_other)
        return self.decode_pairs(probs_dir, probs_rev, mask_negative_other=mask_negative_other)
```

---

### File 3: `src/models/factory.py`

#### Change 3.1: Enhance `SymmetrizedModelWrapper` (lines 194-222)
```python
class SymmetrizedModelWrapper:
    """Inference wrapper pairing any base model with post-hoc Symmetrized Decoding."""

    def __init__(self, base_model: Any, mask_negative_other: bool = False) -> None:
        self.base_model = base_model
        self.mask_negative_other = mask_negative_other
        self.decoder = SymmetrizedDecoder(mask_negative_other=mask_negative_other)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_model(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if hasattr(self.base_model, "forward"):
            return self.base_model.forward(*args, **kwargs)
        return self.base_model(*args, **kwargs)

    def predict_pair(
        self,
        text1: str,
        text2: str,
        mask_negative_other: Optional[bool] = None,
    ) -> Tuple[int, int]:
        """Predict directional pair with 100% SoftCons guarantee.

        Args:
            text1: Premise phrase.
            text2: Hypothesis phrase.
            mask_negative_other: Optional override for NEGATIVE_OTHER class masking.

        Returns:
            Tuple of (pred_dir, pred_rev) class indices satisfying involution symmetry.
        """
        target = self.base_model
        if hasattr(target, "encoder") and hasattr(target.encoder, "predict_proba"):
            target = target.encoder
        if hasattr(target, "predict_proba"):
            p_dir = target.predict_proba(text1, text2)
            p_rev = target.predict_proba(text2, text1)
            return self.decoder.decode_pair(p_dir, p_rev, mask_negative_other=mask_negative_other)
        raise NotImplementedError("Base model does not implement predict_proba.")

    def decode_pair(
        self,
        p_dir: Sequence[float],
        p_rev: Sequence[float],
        mask_negative_other: Optional[bool] = None,
    ) -> Tuple[int, int]:
        return self.decoder.decode_pair(p_dir, p_rev, mask_negative_other=mask_negative_other)

    def decode_pairs(
        self,
        probs_dir: Any,
        probs_rev: Any,
        mask_negative_other: Optional[bool] = None,
    ) -> Tuple[Any, Any]:
        return self.decoder.decode_pairs(probs_dir, probs_rev, mask_negative_other=mask_negative_other)
```

---

### File 4: `tests/test_models.py`

#### Change 4.1: Append 5 new unit tests to `tests/test_models.py`

Add to `TestLossesAndCRL`:
```python
    def test_cross_entropy_pure_ignore_index_single(self):
        """Verify cross_entropy_pure ignores target=-100 and out-of-bounds targets."""
        logits = [0.1, 0.2, 2.5, 0.1]
        loss_valid = cross_entropy_pure(logits, 2)
        self.assertGreater(loss_valid, 0.0)

        # PyTorch ignore_index (-100) returns 0.0
        loss_ignored = cross_entropy_pure(logits, -100)
        self.assertEqual(loss_ignored, 0.0)

        # Negative out-of-bounds target returns 0.0
        self.assertEqual(cross_entropy_pure(logits, -1), 0.0)
        # Out-of-bounds target beyond class count returns 0.0
        self.assertEqual(cross_entropy_pure(logits, 10), 0.0)

    def test_cross_entropy_pure_batch_divided_by_non_ignored(self):
        """Verify batch cross_entropy_pure divides loss ONLY by non-ignored count."""
        l1 = [0.1, 0.2, 2.5, 0.1]  # target 2
        l2 = [0.1, 2.5, 0.2, 0.1]  # target 1
        l3 = [2.5, 0.1, 0.1, 0.1]  # target 0

        loss1 = cross_entropy_pure(l1, 2)
        loss3 = cross_entropy_pure(l3, 0)

        # Batch with 1 masked target (-100) out of 3 samples
        batch_logits = [l1, l2, l3]
        batch_targets = [2, -100, 0]

        batch_loss = cross_entropy_pure(batch_logits, batch_targets)
        # Should divide sum(loss1, loss3) by 2 (non-ignored count), NOT by 3!
        expected_loss = 0.5 * (loss1 + loss3)
        self.assertAlmostEqual(batch_loss, expected_loss, places=6)

        # When all entries in batch are ignored, return 0.0
        all_ignored_loss = cross_entropy_pure(batch_logits, [-100, -100, -100])
        self.assertEqual(all_ignored_loss, 0.0)

    def test_crl_loss_with_ignore_index_handling(self):
        """Verify ConsistencyRegularizationLoss handles target=-100 without KeyError or IndexError."""
        crl_loss = ConsistencyRegularizationLoss(divergence="sym_kl", lambda_cons=1.0)

        ld = [[0.1, 0.2, 2.5, 0.1], [0.1, 2.5, 0.2, 0.1]]
        lr = [[2.5, 0.2, 0.1, 0.1], [0.1, 2.5, 0.2, 0.1]]
        yd = [2, -100]  # Second pair is masked

        loss, metrics = crl_loss(
            logits_dir=ld,
            logits_rev=lr,
            labels_dir=yd,
            labels_rev=None,  # Auto-derives yr=[0, -100] safely
            epoch_or_progress=1.0,
        )

        self.assertGreater(loss, 0.0)
        expected_ce_d = cross_entropy_pure(ld[0], 2)
        self.assertAlmostEqual(metrics["loss_ce_dir"], expected_ce_d, places=5)
```

Add to `TestSymmetrizedDecoding`:
```python
    def test_symmetrized_decoder_mask_negative_other_configurability(self):
        """Verify mask_negative_other is configurable via __init__, attribute, and per-call override."""
        p_dir = [0.1, 0.1, 0.2, 0.6]  # Highest is NEGATIVE_OTHER (3)
        p_rev = [0.2, 0.1, 0.1, 0.6]  # Highest is NEGATIVE_OTHER (3)

        # 1. Unmasked decoder predicts index 3
        decoder_unmasked = SymmetrizedDecoder(mask_negative_other=False)
        self.assertFalse(decoder_unmasked.mask_negative_other)
        d_pred, r_pred = decoder_unmasked.decode_pair(p_dir, p_rev)
        self.assertEqual(d_pred, 3)
        self.assertEqual(r_pred, 3)

        # 2. Masked decoder zeroes index 3 and predicts among {0, 1, 2}
        decoder_masked = SymmetrizedDecoder(mask_negative_other=True)
        self.assertTrue(decoder_masked.mask_negative_other)
        d_pred_m, r_pred_m = decoder_masked.decode_pair(p_dir, p_rev)
        self.assertEqual(d_pred_m, 2)  # FORWARD_ENTAILMENT
        self.assertEqual(r_pred_m, 0)  # BACKWARD_ENTAILMENT

        # 3. Dynamic attribute mutation
        decoder_unmasked.mask_negative_other = True
        d_pred_dyn, r_pred_dyn = decoder_unmasked.decode_pair(p_dir, p_rev)
        self.assertEqual(d_pred_dyn, 2)
        self.assertEqual(r_pred_dyn, 0)

        # 4. Method-level override: call with mask_negative_other=False on masked decoder
        d_override, r_override = decoder_masked.decode_pair(p_dir, p_rev, mask_negative_other=False)
        self.assertEqual(d_override, 3)
        self.assertEqual(r_override, 3)

        # 5. Batch method-level override
        batch_d, batch_r = decoder_unmasked.decode_pairs(
            [p_dir, p_dir],
            [p_rev, p_rev],
            mask_negative_other=False,
        )
        self.assertEqual(batch_d, [3, 3])
        self.assertEqual(batch_r, [3, 3])

    def test_symmetrized_decoder_semeval_softcons_guarantee_with_masking(self):
        """Verify that mask_negative_other=True guarantees 100% SoftCons under SemEval metrics."""
        from src.metrics.metrics import compute_official_metrics

        p_dir = [0.15, 0.10, 0.25, 0.50]
        p_rev = [0.25, 0.10, 0.15, 0.50]

        decoder = SymmetrizedDecoder(mask_negative_other=True)
        pred_d, pred_r = decoder.decode_pair(p_dir, p_rev)
        self.assertEqual(pred_d, 2)  # FORWARD_ENTAILMENT
        self.assertEqual(pred_r, 0)  # BACKWARD_ENTAILMENT

        metrics = compute_official_metrics(
            y_true_dir=["FORWARD_ENTAILMENT"],
            y_pred_dir=[ID_TO_LABEL[pred_d]],
            y_true_rev=["BACKWARD_ENTAILMENT"],
            y_pred_rev=[ID_TO_LABEL[pred_r]],
            is_reversible_mask=[True],
        )
        self.assertEqual(metrics["soft_cons"], 1.0)
        self.assertEqual(metrics["hard_cons"], 1.0)
```

---

## 4. Independent Verification Plan

1. **Verify `cross_entropy_pure` behavior**:
   ```bash
   python3 -c "
   from src.models.losses import cross_entropy_pure
   l1 = [0.1, 0.2, 2.5, 0.1]
   assert cross_entropy_pure(l1, -100) == 0.0
   assert abs(cross_entropy_pure([l1, l1], [2, -100]) - cross_entropy_pure(l1, 2)) < 1e-9
   print('cross_entropy_pure OK')
   "
   ```
2. **Verify `SymmetrizedDecoder` configurability**:
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
3. **Execute Full Test Suite**:
   ```bash
   python3 -m unittest discover -s tests -p "test_*.py" -v
   ```
   *Expected Result*: 78 tests pass in $<0.1$ seconds with `OK`.
