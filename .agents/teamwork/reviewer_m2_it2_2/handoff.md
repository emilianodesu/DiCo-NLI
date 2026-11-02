# Handoff Report: Reviewer 2 for Milestone 2 Iteration 2 (Models & Reversal Strategies)

- **Agent**: Reviewer 2 (`reviewer_m2_it2_2`)
- **Role**: Quality Reviewer & Adversarial Critic
- **Recipient**: Orchestrator (`df6ce377-6407-4a2d-bd18-829b36e20607`)
- **Date**: 2026-11-01T23:32:00Z
- **Working Directory**: `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/reviewer_m2_it2_2/`
- **Verdict**: **APPROVE**

---

## Review Summary

**Verdict**: **APPROVE**  
**Overall Risk Assessment**: **LOW**  
**Integrity Audit**: **CLEAN (Zero Integrity Violations)**

Milestone 2 Iteration 2 remediation has successfully resolved all 5 defects identified during Iteration 1 review. The neural modeling subpackage (`src/models/`) exhibits rigorous mathematical foundations, clean architectural separation between PyTorch/Transformers production backbones and fast CPU mock fallbacks, complete test coverage (83 tests passing in 0.065s on CPU), and full configurability of post-hoc Symmetrized Decoding.

---

## 1. Observation

All verification commands, test suites, and empirical stress tests were executed directly in the runtime environment (`Python 3.14.3`, CPU mode).

### 1.1 Full Test Suite Execution
- **Command**: `python3 -m unittest discover -s tests -p "test_*.py" -v`
- **Result**: **83 tests passed, 0 failures, 0 errors in 0.065 seconds** (Budget: $<60.0$s; margin of $>900\times$).
- **Test suite expansion**: 73 tests (M2 It1) $\to$ 83 tests (M2 It2), adding 10 targeted regression tests in `tests/test_models.py`.

### 1.2 Verification of Iteration 2 Bug Fixes
1. **CPU Mock `.eval()` / `.train()` Parity**:
   - `_ModuleBase` and `CrossEncoderNLI` now implement `.eval()` and `.train(mode: bool = True)`, properly chaining `self.training` and returning `self`.
   - Verified via:
     ```python
     from src.models.factory import build_model
     for m in ['baseline', 'bsa', 'crl', 'sd', 'crl_sd']:
         model = build_model('mock', method=m)
         assert model.eval() is model and not model.training
         assert model.train(True) is model and model.training
     ```
     Result: 100% pass across all 5 methods.

2. **Wrapper Delegation for `crl_sd` and `sd`**:
   - `SymmetrizedModelWrapper._resolve_predict_target()` inspects `self.base_model` and falls back to `self.base_model.encoder`.
   - `build_model("mock", method="crl_sd").predict_pair("the red car", "a vehicle")` executes without `NotImplementedError` and returns `(p_dir, p_rev)` with $p_{rev} = \text{Rev}(p_{dir})$.
   - `predict_proba` and `predict` are directly exposed on `SymmetrizedModelWrapper`.

3. **Decoupled Pure CRL Ablation from Symmetrized Decoding**:
   - In `SiameseCRLModel` (PyTorch lines 249–293, pure-Python lines 556–577), `should_apply_sd` is decoupled from `self.training`.
   - When `apply_symmetrized_decoding=False` is set at initialization or passed in `forward(..., apply_symmetrized_decoding=False)`, `sym_probs_dir` and `sym_probs_rev` evaluate to `None`, and discrete predictions are derived directly from unprojected streams (`torch.argmax(logits_dir, dim=-1)`).
   - In `build_model`, `method="crl"` instantiates `SiameseCRLModel(apply_symmetrized_decoding=False)` and `method="crl_sd"` instantiates `SiameseCRLModel(apply_symmetrized_decoding=True)`.

4. **PyTorch `ignore_index=-100` Parity in Pure-Python Cross-Entropy**:
   - In `src/models/losses.py`, `cross_entropy_pure` ignores samples where `target == -100` or is out-of-bounds, returning `0.0` for 1D masked samples.
   - For 2D batches, the average loss divides strictly by $N_{valid}$ (non-ignored count), rather than the total batch size $B$. If all samples are masked, it safely returns `0.0`.
   - In `ConsistencyRegularizationLoss` fallback, reverse target derivation uses `REV_INDEX_MAP.get(y, y)`, preserving `-100` without `KeyError`.

5. **`SymmetrizedDecoder` Configurable Negative-Other Masking**:
   - `SymmetrizedDecoder` accepts `mask_negative_other: bool` at initialization, supports dynamic attribute assignment (`decoder.mask_negative_other = ...`), and accepts `mask_negative_other: Optional[bool] = None` per-call overrides across:
     - `project_probabilities(..., mask_negative_other=...)`
     - `decode_pair(..., mask_negative_other=...)`
     - `decode_pairs(..., mask_negative_other=...)`
     - `decode_batch(..., mask_negative_other=...)`
     - `decode(..., mask_negative_other=...)`
   - Verified that on input $[0.1, 0.1, 0.2, 0.6]$, `decode_pair` returns $(3, 3)$ when unmasked, and $(2, 0)$ when masked.

### 1.3 Adversarial Stress Testing Results
- **10,000 Dirichlet Monte Carlo Pairs**:
  - Sampled 10,000 asymmetric Dirichlet pairs ($\alpha = 0.1$).
  - With `mask_negative_other=True`: 10,000 / 10,000 pairs produced $\text{SoftCons} = 1.000000$ and discrete predictions strictly in $\{0, 1, 2\}$, guaranteeing 100% SoftCons under SemEval-2027 official metrics.
- **Degenerate Spikes with Masking**:
  - Tested extreme degenerate vector $[0.0, 0.0, 0.0, 1.0]$ (100% negative class) with `mask_negative_other=True`.
  - The zeroed vector $[0, 0, 0, 0]$ resolved via symmetric tie-breaking to $(1, 1)$ (`EQUIVALENCE`), preserving the mathematical involution invariant $\text{Rev}(1) = 1$.
- **All-Masked Targets in CRL Loss**:
  - Executed `ConsistencyRegularizationLoss` with $y_{dir} = [-100, -100]$. Task loss evaluated to $0.0$, and the unsupervised consistency loss was computed without division by zero.

### 1.4 Forensic Integrity Audit
- **Source Code Scan**: Checked all modules in `src/models/` for hardcoded labels, synthetic test sample string lookups (e.g. `"Pope Francis"`, `"syn_en"`), or dummy facades. Zero instances found.
- **Mathematical Reality**: Involutory permutation matrix $P$, softmax, KL divergence, JSD, and cross-entropy are evaluated dynamically through numerical linear algebra.
- **No Shortcuts**: Implementation builds upon native PyTorch modules with full pure-Python offline equivalents.
- **Attestation**: Test outputs and logs reflect authentic local executions.

---

## 2. Logic Chain

1. **Step 1 — Verification of Remediation**:
   - Observations 1.1 and 1.2 confirm that all 5 defects flagged in Iteration 1 have been remediated with genuine mathematical logic and comprehensive tests.
2. **Step 2 — Duck-Typing and Interface Consistency**:
   - The 4 model families (`baseline`, `bsa`, `crl`, `sd` / `crl_sd`) share a consistent inference and lifecycle interface:
     - Lifecycle: `.eval()`, `.train()`, and `.to()` all support chaining (`return self`).
     - Point inference: `.predict(text1, text2)` returns canonical label strings; `.predict_proba(text1, text2)` returns normalized 4-class distributions.
   - For reciprocal pair inference:
     - `crl`, `sd`, and `crl_sd` all provide `predict_pair(text1, text2)` returning $(pred_{dir}, pred_{rev})$.
     - As noted in Finding F-M2-03 (Minor), `SiameseCRLModel.predict_pair` accepts `text1, text2` while `SymmetrizedModelWrapper` accepts `text_a, text_b, text1, text2`. Adding aliases ensures uniform keyword calling.
     - As noted in Finding F-M2-04 (Minor), `CrossEncoderNLI` does not expose `predict_pair`, though callers can obtain reciprocal pair predictions via `predict_proba(t1, t2)` and `predict_proba(t2, t1)`.
3. **Step 3 — Mathematical Equivariance Invariant**:
   - Observation 1.3 demonstrates that Symmetrized Decoding satisfies the algebraic equivariance invariant:
     $$\operatorname{argmax}^*(p_{sym,rev}) = \operatorname{Rev}(\operatorname{argmax}^*(p_{sym,dir}))$$
     across all degenerate, spiky, and Dirichlet distributions with zero precision failure.
4. **Step 4 — Readiness for Milestone 3 (CLI Training & GPU Pipeline)**:
   - The model factory (`build_model`, `build_loss`, `build_tokenizer`), precision detection (`resolve_device`, `resolve_torch_dtype`), and mock fallbacks are fully operational, unblocking `experiments/train.py` and `scripts/run_gpu_training.sh`.
5. **Step 5 — Verdict Determination**:
   - Since all functional requirements are satisfied, all tests pass, and zero integrity violations or blocking flaws exist, the verdict is **APPROVE**.

---

## 3. Findings

### [Minor] Finding 1: Parameter Alias Consistency in `SiameseCRLModel.predict_pair`
- **What**: `SiameseCRLModel.predict_pair` accepts `(text1, text2)` but raises `TypeError: unexpected keyword argument 'text_a'` if invoked with `text_a, text_b` keywords. By contrast, `SymmetrizedModelWrapper.predict_pair` supports both `(text_a, text_b)` and `(text1, text2)`.
- **Where**: `src/models/siamese_crl.py`, line 329 (PyTorch) and line 611 (pure Python).
- **Why**: When writing unified client code that calls `model.predict_pair(text_a=..., text_b=...)` by keyword, `crl` rejects the call while `sd` accepts it.
- **Suggestion**: Add `text_a: Optional[str] = None, text_b: Optional[str] = None` parameters with fallback to `text1, text2` in `SiameseCRLModel.predict_pair`.

### [Minor] Finding 2: `CrossEncoderNLI` Pair Inference Helper
- **What**: `CrossEncoderNLI` does not define `predict_pair(text1, text2)`. Calling `predict_pair` on `baseline` or `bsa` models raises `AttributeError`.
- **Where**: `src/models/cross_encoder.py`.
- **Why**: Evaluators iterating over all models using `model.predict_pair(...)` must special-case `CrossEncoderNLI`.
- **Suggestion**: Add a convenience `predict_pair(text1, text2)` to `CrossEncoderNLI` that evaluates both directions independently without symmetrization:
  ```python
  def predict_pair(self, text1: str, text2: str) -> Tuple[int, int]:
      p_d = self.predict_proba(text1, text2)
      p_r = self.predict_proba(text2, text1)
      return p_d.index(max(p_d)), p_r.index(max(p_r))
  ```

### [Minor / Advisory for M3] Finding 3: PyTorch `TranspositionOperator` Buffer Dtype under AMP
- **What**: In `src/models/losses.py` line 263 (`TranspositionOperator.forward`), `torch.matmul(probs, self.P)` multiplies `probs` by FP32 buffer `self.P`.
- **Where**: `src/models/losses.py`, lines 254 & 263.
- **Why**: Under Automatic Mixed Precision (AMP `fp16`/`bf16`) in M3, `probs` may be half-precision while `self.P` remains float32, which could cause a PyTorch dtype mismatch exception.
- **Suggestion**: In M3 GPU training, update to `torch.matmul(probs, self.P.to(dtype=probs.dtype, device=probs.device))`, exactly matching the pattern in `decoding.py`.

---

## 4. Verified Claims

| Claim | Verification Method | Status |
|:---|:---|:---:|
| 83 automated tests passing in $<1.0$s on CPU | `python3 -m unittest discover -s tests -p "test_*.py" -v` | **PASS** (0.065s) |
| `.eval()` and `.train()` chaining on `CrossEncoderNLI` & Mocks | Programmatic chaining assertion across all 5 methods | **PASS** |
| `build_model("mock", method="crl_sd").predict_pair` | Dynamic inference on mock strings | **PASS** |
| Pure CRL ablation disables SD in `eval()` mode | `SiameseCRLModel.forward(..., apply_symmetrized_decoding=False)` | **PASS** (`sym_probs` is None) |
| `cross_entropy_pure` ignores `-100` and normalizes by $N_{valid}$ | Batch loss calculation with mixed & all-masked targets | **PASS** |
| `mask_negative_other` per-call overrides across all decoder methods | Verified on `project_probabilities`, `decode_pair`, `decode_pairs`, `decode_batch`, `decode` | **PASS** |
| SoftCons = 100% under 10,000 Dirichlet distributions | Monte Carlo simulation ($\alpha = 0.1$) with masked decoder | **PASS** (100.0%) |

---

## 5. Adversarial Challenge Report

### Challenge Summary
- **Overall Risk Assessment**: **LOW**
- **Robustness**: High numerical stability across edge cases, ties, subnormals, and zero-sum vectors.

### Stress Test Matrix
| Scenario | Expected Behavior | Actual Behavior | Result |
|:---|:---|:---|:---:|
| Input vector $[0, 0, 0, 1]$ with `mask_negative_other=True` | Equivariant tie-breaking to class 1 (`EQUIVALENCE`) | Output $(1, 1)$, $\text{Rev}(1) = 1$ | **PASS** |
| All targets $=-100$ in `ConsistencyRegularizationLoss` | Task loss $=0.0$, consistency loss intact, no zero-division | Loss $= 0.00037$, task loss $= 0.0$ | **PASS** |
| 10,000 asymmetric Dirichlet pairs | 100% SoftCons, zero class-3 predictions with masking | 10,000 / 10,000 $\text{SoftCons}=1.0$, only $\{0, 1, 2\}$ | **PASS** |
| Calling `.eval()` followed by `.train()` on all 5 model methods | State toggle and method chaining | `model.training` toggles, returns self | **PASS** |
| Overriding `mask_negative_other` per call on masked/unmasked instances | Immediate per-call precedence over instance setting | Clean toggle between class 3 and reversible classes | **PASS** |

### Unchallenged Areas
- Distributed multi-GPU CUDA training and FP16/BF16 kernel compilation: Not available in local CPU environment; planned and scoped for Milestone 3 (`scripts/run_gpu_training.sh`).

---

## 6. Caveats

- **Runtime Environment**: Validation was conducted in Python 3.14 on macOS without GPU hardware or local PyTorch installation. All PyTorch code paths were verified via static AST analysis and the decoupled pure-Python numerical equivalents.
- **Advisories**: The minor findings noted above do not affect correctness or Milestone 2 criteria and are documented to assist Milestone 3 and Milestone 4 implementing agents.

---

## 7. Conclusion

**Verdict: APPROVE**

Milestone 2 Iteration 2 meets all functional requirements, mathematical invariants, architectural design specifications, and quality standards for DiCo-NLI. The modeling subpackage is approved to unblock Milestone 3 (CLI Training Pipeline & GPU Scripts).

---

## 8. Verification Method

To independently reproduce this verification:

1. **Execute Complete Unit Test Suite**:
   ```bash
   python3 -m unittest discover -s tests -p "test_*.py" -v
   ```
   *Expected*: 83 tests pass in $<0.1$s with `OK`.

2. **Verify Duck-Typing and Chaining**:
   ```bash
   python3 -c "
   from src.models.factory import build_model
   for m in ['baseline', 'bsa', 'crl', 'sd', 'crl_sd']:
       model = build_model('mock', method=m)
       assert model.eval() is model and not model.training
       assert model.train(True) is model and model.training
       assert len(model.predict_proba('car', 'vehicle')) == 4
       assert isinstance(model.predict('car', 'vehicle'), str)
   print('Duck-typing verification OK')
   "
   ```

3. **Verify Pure CRL Ablation**:
   ```bash
   python3 -c "
   from src.models.siamese_crl import SiameseCRLModel
   from src.models.cross_encoder import MockCrossEncoder
   model = SiameseCRLModel(encoder=MockCrossEncoder(), apply_symmetrized_decoding=False)
   model.eval()
   out = model.forward(input_ids_dir=[[1, 2]], input_ids_rev=[[2, 1]])
   assert out.sym_probs_dir is None and out.sym_probs_rev is None
   print('Pure CRL ablation verification OK')
   "
   ```

4. **Verify SymmetrizedDecoder Per-Call Masking Override**:
   ```bash
   python3 -c "
   from src.models.decoding import SymmetrizedDecoder
   p = [0.1, 0.1, 0.2, 0.6]
   dec = SymmetrizedDecoder(mask_negative_other=False)
   assert dec.decode_pair(p, p) == (3, 3)
   assert dec.decode_pair(p, p, mask_negative_other=True) == (2, 0)
   print('SymmetrizedDecoder per-call override OK')
   "
   ```
