# Empirical Challenge Report: Milestone 3 Iteration 2 (Defect Re-Testing)

**Challenger**: Challenger 1 (`challenger_m3_it2_1`)  
**Role**: Empirical Challenger (Critic, Specialist)  
**Recipient**: Orchestrator (`df6ce377-6407-4a2d-bd18-829b36e20607`)  
**Date**: 2026-11-02T03:08:00Z  
**Verdict**: **APPROVE**  

---

## 1. Observation

All 6 defect categories and boundary conditions identified in Milestone 3 Iteration 1 were independently and empirically re-tested using direct CLI invocations, synthetic generators, and stress harnesses:

### 1.1 ReciprocalPair Pre-Pairing Re-Test (Trainer & Dataset Alignment)
- **Tool Command**:
  ```bash
  python3 -c "
  from src.data.loader import load_phrasis_split
  from src.data.reversal import pair_instances, ReciprocalPair
  from src.training.trainer import Trainer, TrainerConfig
  from src.models.factory import build_model

  df = load_phrasis_split('track1', 'train')
  pairs = pair_instances(df)
  trainer = Trainer(
      model=build_model('mock', method='crl', is_mock=True),
      config=TrainerConfig(method='crl', is_mock=True),
      train_dataset=pairs,
  )
  prepared = trainer._prepare_dataset(trainer.train_dataset_raw, is_paired=True)
  assert len(prepared) == 27
  assert len(prepared) == len(pairs)
  "
  ```
- **Observed Result**:
  - `len(pairs) == 27` in Track 1 train split.
  - `len(prepared) == 27` (exact 1:1 preservation of input reciprocal pairs, zero pairs dropped).
  - Calling `trainer.evaluate(eval_dataset=dev_pairs)` evaluated all 15 dev pairs, returning valid metrics: `weighted_f1: 0.3200`, `soft_cons: 0.6667`, `hard_cons: 0.2667`.

### 1.2 `SymmetrizedModelWrapper` Parameter and State Delegation
- **Tool Command**:
  ```bash
  python3 -c "
  from src.models.factory import build_model, SymmetrizedModelWrapper

  for m_name in ['sd', 'crl_sd']:
      m = build_model('mock', method=m_name)
      assert isinstance(m, SymmetrizedModelWrapper)
      assert hasattr(m, 'parameters') and hasattr(m, 'named_parameters')
      assert hasattr(m, 'state_dict') and hasattr(m, 'load_state_dict')
      list(m.parameters())
      list(m.named_parameters())
      sd = m.state_dict()
      m.load_state_dict(sd)
  "
  ```
- **Observed Result**:
  - Successfully instantiated both `sd` and `crl_sd` wrapped models.
  - Calling `m.parameters()`, `m.named_parameters()`, `m.state_dict()`, and `m.load_state_dict()` completed with zero exceptions.
  - Tested delegation with custom model having weights and state dictionary: `wrap.parameters()`, `wrap.named_parameters()`, `wrap.state_dict()`, `wrap.load_state_dict()`, `wrap.to('cuda:0')`, `wrap.train(True)`, and `wrap.eval()` all delegated transparently to the underlying base model.
  - Verified non-existent attributes raise `AttributeError` via `__getattr__` fallback as expected.

### 1.3 Missing Checkpoint Validation in `--eval_only` Mode
- **Tool Command**:
  ```bash
  python3 experiments/train.py --eval_only --checkpoint_path non_existent_checkpoint_xyz.pt --test_run
  ```
- **Observed Result**:
  - Exited with status code **1**.
  - Logged verbatim error: `2026-11-01 21:04:06 | ERROR | dico_nli.train | Checkpoint file not found: non_existent_checkpoint_xyz.pt` followed by `FileNotFoundError: Checkpoint file not found: non_existent_checkpoint_xyz.pt`.
  - Tested directory path `--checkpoint_path checkpoints/`: also exited with status code **1** and raised `FileNotFoundError`.
  - Tested valid existing checkpoint `checkpoints/t1_baseline_deberta/t1_baseline_deberta/best_model.pt`: loaded successfully and completed evaluation with exit code **0**.

### 1.4 Mutual Exclusion of `--fp16` and `--bf16` Precision Flags
- **Tool Command**:
  ```bash
  python3 experiments/train.py --fp16 --bf16 --test_run
  ```
- **Observed Result**:
  - Exited with status code **2**.
  - Emitted standard parser error: `train.py: error: Cannot specify both --fp16 and --bf16. Please select one.`
  - Stress-tested conflicting matrix combinations:
    1. `--precision fp16 --bf16` -> Exits **2** (`Cannot specify both --fp16 and --bf16. Please select one.`)
    2. `--precision bf16 --fp16` -> Exits **2** (`Cannot specify both --fp16 and --bf16. Please select one.`)
    3. `--precision fp32 --fp16` -> Exits **2** (`Cannot specify --fp16 or --bf16 when --precision fp32 is set.`)
    4. `--precision fp32 --bf16` -> Exits **2** (`Cannot specify --fp16 or --bf16 when --precision fp32 is set.`)

### 1.5 None/NaN Robustness in CrossEncoderDataset
- **Tool Command**:
  ```bash
  python3 -c "
  import math
  from src.training.trainer import CrossEncoderDataset
  from src.models.factory import build_tokenizer
  tok = build_tokenizer('mock')
  records = [
      {'instance_id': 0, 'text1': None, 'text2': None, 'label': 'EQUIVALENCE'},
      {'instance_id': 1, 'text1': float('nan'), 'text2': 'Valid text', 'label': 'FORWARD_ENTAILMENT'},
      {'instance_id': 2, 'text1': 'Valid text', 'text2': float('nan'), 'label': 'BACKWARD_ENTAILMENT'},
      {'instance_id': None, 'text1': None, 'text2': None, 'label': None},
      {'instance_id': float('nan'), 'text1': float('nan'), 'text2': float('nan'), 'label': float('nan')},
      {'instance_id': 999, 'text1': 12345, 'text2': 67890, 'label': 'EQUIVALENCE'},
      {},
  ]
  ds = CrossEncoderDataset(records=records, tokenizer=tok)
  for i in range(len(records)):
      item = ds[i]
      assert isinstance(item['text1'], str) and isinstance(item['text2'], str)
  "
  ```
- **Observed Result**:
  - All 7 pathological records were sanitized to valid strings (`""` or stringified integers) without `TypeError`.
  - Missing/NaN `instance_id` values were converted to deterministic fallback string index IDs.

### 1.6 Master GPU Training Launcher (`scripts/run_gpu_training.sh`)
- **Tool Commands & Results**:
  - `bash scripts/run_gpu_training.sh --python "/usr/bin/false" --continue-on-error --track 1` -> Exits **1** with `Benchmark matrix finished with 5 failed or invalid experiment(s)`.
  - `bash scripts/run_gpu_training.sh --method crl_sd --dry-run` -> Plans exactly 2 experiments (`t1_crl_sd_deberta` and `t2_crl_sd_beto`).
  - `bash scripts/run_gpu_training.sh --quick-test` -> Executes 10 benchmark experiments (`baseline`, `bsa`, `crl`, `sd`, `crl_sd` across `track1` and `track2`) in **2.0s**. All 10 Codabench submission files passed verification, and `results/codabench_submissions.zip` was generated with all 10 files.
  - Full unittest suite (`python3 -m unittest discover -s tests -p "test_*.py" -v`): **114/114 tests passed** in **1.365s**.

---

## 2. Logic Chain

1. **Pre-pairing preservation**: In `src/training/trainer.py`, `_prepare_dataset` checks `if records and isinstance(records[0], ReciprocalPair): pairs = records else: pairs = pair_instances(records)`. Because pre-formed reciprocal pairs are retained directly, length is invariant (27 train pairs in Track 1, 15 dev pairs in Track 1).
2. **Optimizer compatibility**: By implementing explicit duck-typing delegation on `SymmetrizedModelWrapper` (`parameters`, `named_parameters`, `state_dict`, `load_state_dict`, `named_buffers`, `buffers`, `zero_grad`, `save_pretrained`, `to`, `eval`, `train`, `__getattr__`), wrapping a model with Symmetrized Decoding does not break `torch.optim.AdamW`, gradient norm clipping, or checkpoint serialization.
3. **Execution safety**: In `experiments/train.py`, verifying `os.path.isfile(args.checkpoint_path)` prevents uninitialized weights from silently being evaluated in production mode, guaranteeing that missing checkpoints raise `FileNotFoundError` (exit code 1).
4. **CLI argument mutual exclusion**: Parser validation rejects conflicting mixed precision configurations (`--fp16` alongside `--bf16` or `--precision fp32`), raising exit code 2 before any runtime initialization occurs.
5. **Data input sanitization**: In `CrossEncoderDataset.__getitem__`, coercing `None` and `NaN` values to `""` via `_clean_str()` eliminates `TypeError: 'NoneType' object is not iterable` in HuggingFace/mock tokenizers.
6. **Failure visibility**: In `scripts/run_gpu_training.sh`, `FAILED_COUNT` ensures non-zero pipeline termination (exit 1) whenever any experiment fails, preventing silent green builds on CI/CD and remote GPU compute nodes.

---

## 3. Caveats & Adversarial Finding (Tier 5 Recommendation)

### Adversarial Finding: `PairedReversalDataset` None/NaN Sanitization Gap
- **Observation**:
  While `CrossEncoderDataset` in `src/training/trainer.py` was sanitized with `_clean_str()`, `PairedReversalDataset` in `src/data/reversal.py` (lines 619–622) retrieves text fields using:
  ```python
  d_text1 = d.get("text1") if hasattr(d, "get") else getattr(d, "text1", "")
  ```
  If `d["text1"]` is `None` or `float('nan')`, `d.get("text1")` returns `None` or `float('nan')`. When passed to `self.tokenizer(d_text1, d_text2)`, `MockTokenizer` or HF `PreTrainedTokenizer` raises:
  `TypeError: 'NoneType' object is not iterable` or `TypeError: 'float' object is not iterable`.
- **Severity**: LOW / NON-BLOCKING for M3.
  - The PhrasIS dataset provided in `data/` contains clean, non-null strings across all rows, meaning standard training and all test suites run without encountering this issue.
  - In `Trainer.evaluate()` and `Trainer.predict()`, text inputs are already defended by `str(fwd.get("text1") or "")`.
- **Recommended Remediation (for Tier 5 / Final Acceptance)**:
  Apply `_clean_str()` in `PairedReversalDataset.__getitem__` in `src/data/reversal.py`:
  ```python
  def _clean_str(val: Any) -> str:
      if val is None or (isinstance(val, float) and math.isnan(val)):
          return ""
      return str(val)
  ```

---

## 4. Conclusion

All 5 defects from Milestone 3 Iteration 1 have been completely remediated, empirically re-tested, and verified:
1. `ReciprocalPair` pre-pairing preserved (`len == 27` train pairs).
2. `SymmetrizedModelWrapper` parameter and state delegation verified.
3. Missing checkpoint validation verified (exits 1 with `FileNotFoundError`).
4. Mutual exclusion of precision flags verified (exits 2 across all combinations).
5. `CrossEncoderDataset` `None`/`NaN` input immunity verified.
6. `scripts/run_gpu_training.sh` 10-experiment matrix and failure exit accounting verified.
7. Full unit test suite passes: 114/114 tests in 1.365s.

**Verdict**: **APPROVE**

---

## 5. Verification Method

To independently reproduce all challenger test results:

```bash
# 1. Full Unit Test Suite (114 tests)
python3 -m unittest discover -s tests -p "test_*.py" -v

# 2. ReciprocalPair Pre-Pairing Verification (len == 27)
python3 -c "
from src.data.loader import load_phrasis_split
from src.data.reversal import pair_instances
from src.training.trainer import Trainer, TrainerConfig
from src.models.factory import build_model
pairs = pair_instances(load_phrasis_split('track1', 'train'))
trainer = Trainer(model=build_model('mock', method='crl'), config=TrainerConfig(method='crl'), train_dataset=pairs)
prepared = trainer._prepare_dataset(trainer.train_dataset_raw, is_paired=True)
assert len(prepared) == 27, f'Expected 27, got {len(prepared)}'
print('Verified: Pre-paired dataset preserved 27 pairs.')
"

# 3. SymmetrizedModelWrapper Delegation Verification
python3 -c "
from src.models.factory import build_model
for m_name in ['sd', 'crl_sd']:
    m = build_model('mock', method=m_name)
    list(m.parameters()); list(m.named_parameters()); m.load_state_dict(m.state_dict())
print('Verified: Wrapper parameter delegation operational.')
"

# 4. Checkpoint Existence Validation (Exits 1)
python3 experiments/train.py --eval_only --checkpoint_path non_existent_checkpoint_xyz.pt --test_run || echo "Exited with code $?"

# 5. Conflicting Precision Flags Validation (Exits 2)
python3 experiments/train.py --fp16 --bf16 --test_run || echo "Exited with code $?"

# 6. GPU Script Quick Test (10/10 Runs)
bash scripts/run_gpu_training.sh --quick-test

# 7. GPU Script Failure Accounting (Exits 1)
bash scripts/run_gpu_training.sh --python "/usr/bin/false" --continue-on-error --track 1 || echo "Exited with code $?"
```
