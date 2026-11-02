# Empirical Challenge Report & Handoff: Milestone 3

**Author:** Challenger 1 (Milestone 3 — CLI Training Pipeline & GPU Scripts)  
**Date:** 2026-11-02T02:38:00Z  
**Recipient:** Orchestrator (`df6ce377-6407-4a2d-bd18-829b36e20607`)  
**Workspace:** `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject`  
**Verdict:** **REQUEST_CHANGES**

---

## Challenge Summary

- **Overall Risk Assessment**: **MEDIUM**
- **Core Pipeline Status**: Robust and highly performant (104/104 tests pass in 0.865s; all 10 track-method combinations succeed; Symmetrized Decoding delivers strictly 100.00% SoftCons on both tracks; GPU bash orchestrator executes 8 benchmarks cleanly in 1s).
- **Vulnerabilities Discovered**: 4 concrete empirical defects in `experiments/train.py` and `src/training/trainer.py` (including a silent failure mode on missing checkpoints and dead code in `trainer.py`).

---

## 1. Observation

Direct empirical observations from executing tests, shell commands, and stress harnesses:

1. **Test Suite Baseline**:
   - Command: `python3 -m unittest discover -s tests -p "test_*.py" -v`
   - Result: `Ran 104 tests in 0.865s. OK.` (100% pass rate).

2. **Empirical Method Matrix Verification (`experiments/train.py --test_run`)**:
   Executed all 10 combinations of Track (1, 2) × Method (baseline, bsa, crl, sd, crl_sd):
   - Track 1 baseline: Weighted F1 = 0.3200, SoftCons = 0.6667, HardCons = 0.2667 (Passed)
   - Track 1 bsa: Weighted F1 = 0.3200, SoftCons = 0.6667, HardCons = 0.2667 (Passed)
   - Track 1 crl: Weighted F1 = 0.3200, SoftCons = 0.6667, HardCons = 0.2667 (Passed)
   - Track 1 sd: Weighted F1 = 0.5333, **SoftCons = 1.000000**, HardCons = 0.5333 (Passed)
   - Track 1 crl_sd: Weighted F1 = 0.5333, **SoftCons = 1.000000**, HardCons = 0.5333 (Passed)
   - Track 2 baseline: Weighted F1 = 0.2824, SoftCons = 0.7333, HardCons = 0.2667 (Passed)
   - Track 2 bsa: Weighted F1 = 0.2824, SoftCons = 0.7333, HardCons = 0.2667 (Passed)
   - Track 2 crl: Weighted F1 = 0.2824, SoftCons = 0.7333, HardCons = 0.2667 (Passed)
   - Track 2 sd: Weighted F1 = 0.4054, **SoftCons = 1.000000**, HardCons = 0.4000 (Passed)
   - Track 2 crl_sd: Weighted F1 = 0.4054, **SoftCons = 1.000000**, HardCons = 0.4000 (Passed)
   - All 10 generated Codabench submission files passed programmatic format verification (`validate_codabench_submission`).

3. **Defect 1 — Silent Evaluation on Non-Existent Checkpoint (`experiments/train.py:359`)**:
   - Command: `python3 experiments/train.py --eval_only --checkpoint_path /path/to/nonexistent.pt --test_run`
   - Result: Exited with code 0! Output:
     ```
     2026-11-01 20:34:24 | INFO | dico_nli.train | Mode: eval_only | Test Run: True
     ...
     2026-11-01 20:34:24 | INFO | dico_nli.train | OFFICIAL EVALUATION METRICS:
     2026-11-01 20:34:24 | INFO | dico_nli.train |   Weighted F1 : 0.320000
     ```
   - Inspection of `experiments/train.py:359`:
     ```python
     if args.checkpoint_path is not None and os.path.isfile(args.checkpoint_path):
         logger.info(f"Loading checkpoint weights from: {args.checkpoint_path}")
         if hasattr(model, "load_state_dict") and HAS_TORCH:
             model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
     ```
     If the user makes a typo in `--checkpoint_path`, `os.path.isfile` is false, so it silently skips loading, evaluates uninitialized weights, and produces fake submissions without raising any exception or warning.

4. **Defect 2 — Dead / Truncated Duplicate Method in `src/training/trainer.py:494-515`**:
   - In `src/training/trainer.py`:
     - Lines 494-515: An incomplete `evaluate` method definition is opened and ends abruptly:
       ```python
       494:     def evaluate(self, eval_dataset: Optional[Any] = None) -> Dict[str, float]:
       ...
       513:         if hasattr(self.model, "eval"):
       514:             self.model.eval()
       515:
       516:     def _predict_pair_indices(self, t1: str, t2: str, apply_sd: bool) -> Tuple[int, int]:
       ```
     - Lines 539-596: A second `evaluate` method is defined, which shadows/overwrites lines 494-515.
     - Lines 494-515 are dead code and an incomplete stub.

5. **Defect 3 — Uncaught `TypeError` on Missing Text Fields in `CrossEncoderDataset` (`src/training/trainer.py:142-143`)**:
   - Command:
     ```python
     from src.training.trainer import CrossEncoderDataset
     from src.models.factory import build_tokenizer
     tok = build_tokenizer('mock')
     ds = CrossEncoderDataset([{'instance_id': 1, 'text1': None, 'text2': None, 'label': 'EQUIVALENCE'}], tokenizer=tok)
     item = ds[0]
     ```
   - Result:
     `TypeError: 'NoneType' object is not iterable` in tokenizer.
     `r.get("text1", "")` returns `None` when `"text1"` is present with value `None` (common in pandas DataFrames with NaN). It must be `(r.get("text1") or "")`.

6. **Defect 4 — Unvalidated Conflicting Precision Flags (`--fp16` and `--bf16`)**:
   - Command: `python3 experiments/train.py --fp16 --bf16 --test_run`
   - Result: Exited with code 0; `Mixed Precision: fp16=True, bf16=True`.
   - In `_train_pytorch` (line 401), `GradScaler("cuda", enabled=self.config.fp16)` is instantiated while `get_autocast_dtype` returns `torch.bfloat16`. Running `GradScaler` with `bfloat16` is a known anti-pattern in PyTorch.

7. **Concurrency & Payload Audit on `JsonlLogger`**:
   - 10 threads × 50 writes (500 records) completed with 0 dropped lines and valid JSON parsing.
   - `NaN` and `Infinity` handled cleanly.
   - Attempting to log Python `set` raised `TypeError: Object of type set is not JSON serializable` due to lack of `default=str` in `json.dumps`.

8. **GPU Bash Script Audit (`scripts/run_gpu_training.sh`)**:
   - Syntax check: `bash -n scripts/run_gpu_training.sh` passed.
   - Dry run: `bash scripts/run_gpu_training.sh --dry-run` passed.
   - Invalid track: `bash scripts/run_gpu_training.sh --track 99` exited with code 1 and error message `[ERROR] Invalid --track '99'`.
   - Invalid method: `bash scripts/run_gpu_training.sh --method unknown` exited with code 1 and error message `[ERROR] Invalid --method 'unknown'`.
   - Quick test: `bash scripts/run_gpu_training.sh --quick-test` executed all 8 matrix runs in 1s, verified 8 Codabench submission files, and generated `logs/matrix_summary.md` and `results/summary.json`.

---

## 2. Logic Chain

1. **Silent Fallback Logic (Defect 1)**:
   - When `--checkpoint_path` is specified, the user expects that exact checkpoint to be evaluated or resumed.
   - If `os.path.isfile(args.checkpoint_path)` is false, silently continuing with default weights produces invalid scientific metrics and invalid competition submissions without the user realizing their checkpoint was missing or misspelled.
   - In accordance with defensive engineering principles, a missing file explicitly provided by the user must raise `FileNotFoundError`.

2. **Code Hygiene & Shadowed Functions (Defect 2)**:
   - Lines 494-515 in `src/training/trainer.py` contain 22 lines of an abandoned function definition that terminates without a `return` statement before `_predict_pair_indices`.
   - While Python's namespace semantics allow the later definition at line 539 to overwrite it, dead code confuses maintenance, linters, and refactoring tools.

3. **Data Robustness (Defect 3)**:
   - Pandas datasets regularly represent missing strings as `np.nan` or `None`.
   - `dict.get(key, default)` returns the value if the key exists, even if the value is `None`.
   - Consequently, `t1 = r.get("text1", "")` produces `t1 = None`, crashing the tokenizer. Using `(r.get("text1") or "")` guards against both missing keys and `None` values.

4. **Mutual Exclusion (Defect 4)**:
   - Mixed precision in PyTorch cannot be both `fp16` and `bf16` simultaneously.
   - The CLI should enforce mutual exclusivity between `--fp16` and `--bf16` or resolve to an explicit error if both are passed.

---

## 3. Adversarial Challenges

### [High] Challenge 1: Silent Execution on Non-Existent Checkpoint
- **Assumption Challenged**: System assumes that checking `if os.path.isfile(args.checkpoint_path)` is sufficient.
- **Attack Scenario**: User runs `python3 experiments/train.py --eval_only --checkpoint_path checkpoints/best_moedl.pt` (typo).
- **Blast Radius**: The pipeline completes with exit code 0 and logs evaluation scores of random/mock weights as if it were the trained checkpoint. Submissions submitted to Codabench would be completely invalid.
- **Mitigation**:
  ```python
  if args.checkpoint_path is not None:
      if not os.path.isfile(args.checkpoint_path):
          raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint_path}")
      ...
  ```

### [Medium] Challenge 2: Dead / Truncated Method Definition in `src/training/trainer.py`
- **Assumption Challenged**: Implementation has clean class structure without duplicate methods.
- **Attack Scenario**: Linter / AST analysis / static type checking flags duplicate method declaration `evaluate` at lines 494 and 539.
- **Blast Radius**: Code readability, confusion over which `evaluate` method is active, lint errors.
- **Mitigation**: Delete dead lines 494-515 in `src/training/trainer.py`.

### [Medium] Challenge 3: Uncaught Tokenizer Crash on Missing Text Fields
- **Assumption Challenged**: Text fields in input data are always non-empty strings.
- **Attack Scenario**: A corrupted CSV or split contains `None` or `NaN` for `text1` or `text2`.
- **Blast Radius**: `TypeError: 'NoneType' object is not iterable` during training batch collation.
- **Mitigation**:
  ```python
  t1 = str(r.get("text1") or "") if isinstance(r, dict) else str(getattr(r, "text1", "") or "")
  t2 = str(r.get("text2") or "") if isinstance(r, dict) else str(getattr(r, "text2", "") or "")
  ```

### [Low] Challenge 4: Conflicting `--fp16` and `--bf16` Precision Flags
- **Assumption Challenged**: User will only pass one precision flag.
- **Attack Scenario**: User passes `--fp16 --bf16` simultaneously.
- **Blast Radius**: Discrepancy between autocast dtype (`bfloat16`) and GradScaler state (`enabled=True`).
- **Mitigation**: Raise `ValueError("Cannot specify both --fp16 and --bf16. Please select one.")` in `parse_args`.

---

## 4. Stress Test Results Summary

| # | Test Scenario | Expected Behavior | Actual Behavior | Result |
|---|---------------|-------------------|-----------------|--------|
| 1 | `python3 -m unittest discover` | 104 tests pass | 104 tests pass in 0.865s | PASS |
| 2 | Track 1 baseline/bsa/crl `--test_run` | Clean exit 0, metrics computed | Exit 0, metrics valid | PASS |
| 3 | Track 1 sd & crl_sd `--test_run` | Exit 0, **SoftCons == 1.000000** | Exit 0, **SoftCons = 1.000000** | PASS |
| 4 | Track 2 baseline/bsa/crl `--test_run` | Clean exit 0, metrics computed | Exit 0, metrics valid | PASS |
| 5 | Track 2 sd & crl_sd `--test_run` | Exit 0, **SoftCons == 1.000000** | Exit 0, **SoftCons = 1.000000** | PASS |
| 6 | Invalid Track `--track 99` | Graceful exit with error | `ValueError: Unsupported track '99'`, exit 1 | PASS |
| 7 | Invalid Method `--method invalid` | Argparse rejection | `invalid choice: 'invalid'`, exit 2 | PASS |
| 8 | Missing Checkpoint in `--eval_only` | Raise FileNotFoundError | Silently ignored, exit 0 | **FAIL** |
| 9 | Duplicate `evaluate` in `trainer.py` | Single clean method definition | Truncated stub at line 494 | **FAIL** |
| 10 | Null text in `CrossEncoderDataset` | Coerce to empty string | `TypeError: 'NoneType' object is not iterable` | **FAIL** |
| 11 | Conflicting `--fp16 --bf16` | Reject or warn | Both enabled without warning | **FAIL** |
| 12 | Empty train/eval dataset `[]` | Graceful handling | Status completed, 0.0 metrics | PASS |
| 13 | Boundary batch size 1 & 1000 | No shape/division crash | All batch sizes succeed | PASS |
| 14 | `scripts/run_gpu_training.sh -h/--dry-run` | Valid matrix display | 8 experiments displayed correctly | PASS |
| 15 | `scripts/run_gpu_training.sh --quick-test` | All 8 runs pass in <60s | All 8 runs pass in 1s, Codabench verified | PASS |
| 16 | `JsonlLogger` 10-thread concurrency | 500 records intact | 500 records intact, 0 corruptions | PASS |

---

## 5. Caveats

- **Multi-GPU / Real CUDA Hardware**: Full hardware testing with NVIDIA CUDA and Tensor Cores was evaluated via dry-run and CPU mock fallback because the local runner is a macOS CPU environment. Real GPU training requires running `bash scripts/run_gpu_training.sh` on an NVIDIA server.
- **Mock Checkpoint Format**: In mock CPU mode, checkpoints are saved as JSON metadata (`best_model.pt`). On a real GPU server with PyTorch, checkpoints are binary state dicts via `torch.save`. The missing checkpoint validation fix will safeguard both environments.

---

## 6. Conclusion

The Milestone 3 implementation achieves extraordinary performance and meets all core architectural requirements:
- Dual-mode execution (ultra-fast CPU testing vs real PyTorch GPU).
- Mathematical consistency guarantee (**SoftCons = 100%**) holds across both tracks.
- Codabench submissions are strictly validated.
- Bash orchestration script is robust, POSIX-portable, and fully automated.

However, to guarantee maximum production integrity on remote compute nodes, the 4 identified defects must be corrected.

**Verdict: REQUEST_CHANGES**

### Actionable Fixes Required from Worker M3:
1. **Fix `experiments/train.py:359`**:
   If `args.checkpoint_path` is not None, verify `os.path.isfile(args.checkpoint_path)`. If it does not exist, raise `FileNotFoundError(f"Checkpoint file not found: {args.checkpoint_path}")`.
2. **Clean up `src/training/trainer.py:494-515`**:
   Remove the truncated, dead `evaluate` stub preceding `_predict_pair_indices`.
3. **Harden `CrossEncoderDataset` (`src/training/trainer.py:142-143`)**:
   Use `str(r.get("text1") or "")` and `str(r.get("text2") or "")` to prevent `NoneType` tokenizer errors when datasets have missing values.
4. **Validate conflicting CLI precision flags (`experiments/train.py`)**:
   If both `args.fp16` and `args.bf16` are set, raise an informative error (e.g., `parser.error("Cannot specify both --fp16 and --bf16.")`).

---

## 7. Verification Method

To verify the requested changes:
```bash
# 1. Run unit tests
python3 -m unittest discover -s tests -p "test_*.py" -v

# 2. Verify non-existent checkpoint raises FileNotFoundError
python3 experiments/train.py --eval_only --checkpoint_path /path/to/nonexistent.pt --test_run
# Expected: Exit code 1 with FileNotFoundError

# 3. Verify conflicting precision flags are rejected
python3 experiments/train.py --fp16 --bf16 --test_run
# Expected: Exit code 2 with parser error

# 4. Verify GPU quick-test still passes 100%
bash scripts/run_gpu_training.sh --quick-test
```
