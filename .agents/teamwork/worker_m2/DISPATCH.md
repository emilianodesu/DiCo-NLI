## 2026-11-01T22:51:50Z

You are the Implementation Worker for Milestone 2 (M2: Models & Reversal Strategies) for SemEval-2027 Task 2 (DiCo-NLI).
Your working directory is: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/worker_m2/
Orchestrator conversation ID: df6ce377-6407-4a2d-bd18-829b36e20607

Read:
- /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/ORIGINAL_REQUEST.md
- /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/PROJECT.md
- /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m2_1/report.md
- /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m2_2/report.md
- /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m2_3/report.md

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

File write ownership (you own exclusively):
- src/models/__init__.py
- src/models/cross_encoder.py
- src/models/siamese_crl.py
- src/models/losses.py
- src/models/decoding.py
- src/models/factory.py
- tests/test_models.py (create model unit tests)
- tests/test_pipeline.py (verify live binding)

Implementation goals:
1. `src/models/cross_encoder.py`:
   - Standard unidirectional cross-encoder fine-tuning architecture.
   - 4-class classification head (0: BACKWARD_ENTAILMENT, 1: EQUIVALENCE, 2: FORWARD_ENTAILMENT, 3: NEGATIVE_OTHER).
   - DeBERTa-v3 (with cls_dropout), BETO, RoBERTa-bne support.
   - Fast mock transformer fallback (MockCrossEncoder, MockTokenizer) for offline CPU tests.
2. `src/models/losses.py`:
   - Cross-entropy loss.
   - Symmetric KL divergence loss.
   - Jensen-Shannon divergence loss.
   - Dynamic warmup scheduling for consistency regularization.
3. `src/models/siamese_crl.py`:
   - Siamese / dual-stream architecture processing direct pair (A, B) and reverse pair (B, A).
   - Involution permutation matrix P (swapping indices 0 and 2).
   - Loss combination: L_total = L_CE(A,B) + L_CE(B,A) + lambda * D_KL(p_A || P^T p_B).
4. `src/models/decoding.py`:
   - Post-hoc Symmetrized Decoding projection:
     p_sym,dir = 0.5 * (p_dir + p_rev @ P)
     p_sym,rev = 0.5 * (p_rev + p_dir @ P)
   - SymmetrizedDecoder supporting tensor/array/list operations and equivariant tie-breaking.
   - Guaranteed SoftCons = 100%.
5. `src/models/factory.py`:
   - Unified build_model(model_name_or_path, method="baseline", ...) supporting baseline, bsa, crl, sd.
   - build_loss, build_tokenizer, and fast mock fallbacks.
6. `src/models/__init__.py`: Clean public exports.
7. Unit tests in `tests/test_models.py` verifying all 4 methods.
8. Execute test suite:
   python3 -m unittest discover -s tests -p "test_*.py" -v
   Verify 100% test pass rate in <60s on CPU.
9. Deliver handoff report to:
   /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/worker_m2/handoff.md
   and message orchestrator (df6ce377-6407-4a2d-bd18-829b36e20607) when complete.
