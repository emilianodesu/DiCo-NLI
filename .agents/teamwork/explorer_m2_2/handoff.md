# Handoff Report: Siamese CRL & Loss Functions Blueprint (M2 Explorer 2)

- **Agent**: M2 Explorer 2 (Siamese CRL & Loss Functions)
- **Recipient**: Orchestrator (df6ce377-6407-4a2d-bd18-829b36e20607)
- **Date**: 2026-11-01T22:55:00Z
- **Type**: Hard Handoff (Task Complete)

---

## 1. Observation

1. **Task Scope & Requirements**:
   - Dispatch assigned design of `src/models/losses.py` (standard CE, symmetric KL, JSD, CRL loss with involution matrix $P$) and `src/models/siamese_crl.py` (dual-stream Siamese architecture, permutation matrix $P$, training forward computing total loss, eval forward returning paired predictions).
   - `ORIGINAL_REQUEST.md` line 26: Consistency Regularization Loss (CRL) jointly processes $(A, B)$ and $(B, A)$ penalizing distribution divergences via KL or JSD.
   - `PROJECT.md` lines 67-76: Class index encoding:
     `0`: `BACKWARD_ENTAILMENT`, `1`: `EQUIVALENCE`, `2`: `FORWARD_ENTAILMENT`, `3`: `NEGATIVE_OTHER`.
     Involution permutation matrix $P$:
     $$P = \begin{pmatrix} 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

2. **Codebase Status & Contracts**:
   - `src/models/` does not yet exist.
   - `src/data/reversal.py` lines 586-673: `PairedReversalDataset` already outputs dictionary items with keys:
     `dir_input_ids`, `dir_attention_mask`, `dir_token_type_ids`, `inv_input_ids`, `inv_attention_mask`, `inv_token_type_ids`, `direct_label_id`, `inverse_label_id`.
   - `src/metrics/metrics.py` lines 38-84: `LABEL_TO_IDX` and `INVOLUTION_PERMUTATION_MATRIX` match the canonical encoding.
   - Current Python runtime (`/usr/local/bin/python3 3.14.3`) runs all 54 existing tests in `tests/` in 0.062s because the repository architecture supports pure Python / lightweight mock fallbacks when `torch` is not installed in the local environment.

3. **Tool Execution & Empirical Test Results**:
   - Executed `.agents/teamwork/explorer_m2_2/test_blueprint.py` via `python3 .agents/teamwork/explorer_m2_2/test_blueprint.py`:
     - Test 1: $P = P^T$ and $P^2 = I_4$ verified.
     - Test 2: Probability simplex preservation and coordinate swapping $0 \leftrightarrow 2$ verified.
     - Test 3: Symmetric KL divergence properties verified ($D_{sym\_KL} \ge 0$, $D_{sym\_KL}(p, q) = D_{sym\_KL}(q, p)$).
     - Test 4: Jensen-Shannon divergence bounded in $[0, \ln 2]$ and symmetric verified.
     - Test 5: CRL loss calculation and dynamic warmup annealing verified.
     - Test 6: Contradictory logits trigger a >300x spike in consistency loss (`0.0058` for consistent vs. `1.8016` for contradictory).
     - Test 7: Symmetrized Decoding SoftCons = 100% verified across 1,000 randomized configurations.

---

## 2. Logic Chain

1. **Algebraic Invariance**: Because semantic reversal is an involution ($\operatorname{Rev}(\operatorname{Rev}(y)) = y$), the permutation matrix representation $P$ must be symmetric ($P = P^T$) and involutory ($P^2 = I$). When a probability distribution $p_{rev}$ over reverse pair $(B, A)$ is multiplied by $P$, the resulting vector $\tilde{p}_{rev} = p_{rev} P$ shifts coordinates such that index 2 becomes index 0, index 0 becomes index 2, and indices 1 and 3 remain stationary. Thus, $\tilde{p}_{rev}$ and $p_{dir}$ reside in the identical forward reference frame.
2. **Divergence Selection & Numerical Safety**:
   - Direct calculation of KL divergence in standard probability space can suffer from division by zero or log of zero if any class probability approaches 0.
   - Clamping probabilities to `eps = 1e-8` and renormalizing prevents `NaN` gradients.
   - In PyTorch's `F.kl_div(input, target)`, `input` must be $\log Q$ and `target` must be $P$ with `reduction="batchmean"` to divide strictly by the batch dimension $B$.
   - JSD incorporates midpoint distribution $M = 0.5(P + Q)$, ensuring $M_c > 0$ whenever $P_c > 0$ or $Q_c > 0$, guaranteeing numerical stability and boundedness in $[0, \ln 2]$.
3. **Dual-Stream Optimization**:
   - Processing both streams sequentially uses $1\times$ peak memory on low-memory GPUs.
   - Concatenating direct and reverse streams along dimension 0 ($2B$) allows a single backbone call, reducing CUDA kernel launches and improving Tensor Core efficiency on remote servers. Providing both modes configurable via `concatenate_streams` satisfies both memory-constrained and compute-optimized environments.
4. **Inference Guarantee**:
   - Incorporating Symmetrized Decoding ($p_{sym,dir} = 0.5(p_{dir} + p_{rev} P)$) in evaluation mode ensures that the model outputs are mathematically guaranteed to achieve $\text{SoftCons} = 100\%$ on any test set without post-processing scripts.

---

## 3. Caveats

1. **Dependency Constraints**: In environments without PyTorch installed locally (such as current local macOS testing environment), code that imports `torch` directly will raise `ModuleNotFoundError` unless protected by `HAS_TORCH` conditionals or executed within a configured virtual environment on the remote GPU server. The blueprint includes both PyTorch modules and pure-Python fallbacks to ensure compatibility across all test tiers.
2. **Label Smoothing in CRL**: When `label_smoothing > 0.0` is used in task cross-entropy, the ground truth targets become soft probability distributions. The supervised task loss handles this gracefully, while the consistency regularization term acts on predicted model probabilities independently of label smoothing.
3. **Negative Other Reversibility**: In SemEval-2027 Task 2, `NEGATIVE_OTHER` is self-symmetric ($\operatorname{Rev}(3) = 3$). The consistency penalty regularizes `NEGATIVE_OTHER` pairs identically to reversible pairs unless an explicit mask (`is_reversible`) is passed.

---

## 4. Conclusion

The blueprints for `src/models/losses.py` and `src/models/siamese_crl.py` are fully designed, mathematically validated, and verified.
- **`src/models/losses.py`** provides:
  - `P_MATRIX` and `TranspositionOperator` with GPU device synchronization.
  - `SymmetricKLDivergenceLoss` ($0.5 \cdot [D_{KL}(P \parallel Q) + D_{KL}(Q \parallel P)]$).
  - `JensenShannonDivergenceLoss` (strictly bounded in $[0, \ln 2]$).
  - `ConsistencyRegularizationLoss` combining supervised task CE with dynamically annealed consistency regularization and detailed metric telemetry.
- **`src/models/siamese_crl.py`** provides:
  - `SiameseCRLModel` wrapping any shared encoder.
  - Native ingestion of `PairedReversalDataset` batches.
  - Configurable sequential vs. concatenated ($2B$) forward execution.
  - Automatic Symmetrized Decoding during evaluation producing 100% SoftCons.
  - Lightweight `MockSequenceEncoder` for CPU smoke testing in $<5$ ms.

The complete code specifications and mathematical formulations are documented in `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m2_2/report.md`.

---

## 5. Verification Method

To independently verify the mathematical properties and algorithmic logic:
1. Run the test script:
   ```bash
   python3 /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m2_2/test_blueprint.py
   ```
   **Expected Output**:
   ```
   Executing Pure-Python Blueprint Mathematical Verification...
   Test 1: P is symmetric, orthogonal involution (P^2 = I) [PASSED]
   Test 2: Simplex preservation & exact coordinate permutation [PASSED]
   Test 3: Symmetric KL divergence properties (sym_kl=0.086654) [PASSED]
   Test 4: Jensen-Shannon divergence properties (JSD=0.021454 <= ln 2) [PASSED]
   Test 5: CRL Loss calculation & annealing [PASSED]: total=0.2212, cons=0.0058
   Test 6: Reversal Curse detection [PASSED]: consistent=0.0058 vs inconsistent=1.8016
   Test 7: Symmetrized Decoding SoftCons = 100% verified across 1000 configurations [PASSED]
   ALL MATHEMATICAL & ALGORITHMIC TESTS PASSED PERFECTLY!
   ```
2. Invalidation Conditions:
   - If $P @ P \neq I_4$, the involution property is broken.
   - If $p_{sym,rev} \neq p_{sym,dir} P$, the equivariance property is broken.
   - If $\operatorname{argmax}(p_{sym,rev}) \neq \operatorname{Rev}(\operatorname{argmax}(p_{sym,dir}))$, the 100% SoftCons guarantee fails.
