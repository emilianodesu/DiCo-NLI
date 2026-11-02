# Review & Adversarial Stress-Test Report: Milestone 3 (CLI Training Pipeline & GPU Scripts)

**Reviewer:** Reviewer 1 (Archetype: `reviewer_critic`)  
**Date:** 2026-11-02T02:40:00Z  
**Target:** Milestone 3 Deliverables (`src/training/trainer.py`, `src/training/utils.py`, `src/training/__init__.py`, `experiments/train.py`, `scripts/run_gpu_training.sh`, `tests/test_training.py`)  
**Verdict:** **REQUEST_CHANGES**  
**Overall Risk Assessment:** **HIGH**  

---

## 1. Observation

Direct observations from codebase inspection, empirical stress-tests, and tool executions:

### Obs 1: Pre-Pairing Double-Inversion Bug Leaves 0 Training Batches for `crl` and `crl_sd`
In `experiments/train.py` (lines 331–336):
```python
elif args.method in ["crl", "crl_sd"]:
    logger.info("Extracting reciprocal pairs for Siamese CRL training...")
    train_pairs = pair_instances(df_train)
    logger.info(f"Formed {len(train_pairs)} aligned reciprocal pairs.")
    train_data = train_pairs
```
In `src/training/trainer.py` (lines 247–264):
```python
def _prepare_dataset(self, data: Any, is_paired: bool) -> Any:
    """Adapts raw records or DataFrame into appropriate Dataset."""
    if data is None:
        return None
    if hasattr(data, "to_dict") and callable(getattr(data, "to_dict")):
        records = data.to_dict("records")
    elif isinstance(data, (list, tuple)):
        records = list(data)
    else:
        return data

    if is_paired:
        pairs = pair_instances(records)
        return PairedReversalDataset(
            pairs=pairs,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
        )
```
When `train_data` (already a list of `ReciprocalPair` objects) is passed to `_prepare_dataset(..., is_paired=True)`, `pair_instances(records)` tries to extract `instance_id` on each `ReciprocalPair`. Since `ReciprocalPair` tuples do not have a top-level `instance_id` attribute, `pair_instances` returns an empty list (`[]`).
Verification command executed:
```bash
python3 -c "
from src.data.loader import load_phrasis_split
from src.data.reversal import pair_instances
from src.training.trainer import Trainer, TrainerConfig
from src.models.factory import build_model

df = load_phrasis_split('track1', 'train')
pairs = pair_instances(df)
trainer = Trainer(model=build_model('mock', method='crl', is_mock=True),
                  config=TrainerConfig(method='crl', is_mock=True),
                  train_dataset=pairs)
ds = trainer._prepare_dataset(trainer.train_dataset_raw, is_paired=True)
print('Length of prepared dataset:', len(ds))
"
```
Output:
`Length of prepared dataset: 0`  
Result: In PyTorch mode (`_train_pytorch`), the DataLoader contains 0 batches, `total_steps = 0`, and the model trains for 0 iterations.

---

### Obs 2: `SymmetrizedModelWrapper` Missing Parameter and State Delegation for Real PyTorch Training
In `src/models/factory.py` (lines 194–209):
`SymmetrizedModelWrapper` does not inherit from `torch.nn.Module`, nor does it implement `parameters()`, `named_parameters()`, `state_dict()`, or `load_state_dict()`.
In `src/training/trainer.py` (lines 361–380):
```python
named_params = list(self.model.named_parameters()) if hasattr(self.model, "named_parameters") else []
if named_params:
    optimizer_grouped_parameters = [ ... ]
else:
    optimizer_grouped_parameters = self.model.parameters() if hasattr(self.model, "parameters") else []

optimizer = self.optimizer or torch.optim.AdamW(
    optimizer_grouped_parameters,
    lr=self.config.learning_rate,
    eps=1e-8,
)
```
When `method="sd"` or `method="crl_sd"`, `build_model()` returns `SymmetrizedModelWrapper`.
Because `self.model` has neither `named_parameters` nor `parameters`, `optimizer_grouped_parameters` evaluates to `[]`.
Calling `torch.optim.AdamW([], lr=...)` raises `ValueError: optimizer got an empty parameter list`.
Additionally:
- Line 440: `if hasattr(self.model, "parameters"): torch.nn.utils.clip_grad_norm_(...)` skips gradient clipping entirely.
- Line 665: `"model_state_dict": self.model.state_dict() if hasattr(self.model, "state_dict") else {}` saves an empty dictionary `{}`.
- `experiments/train.py` line 361: `if hasattr(model, "load_state_dict") and HAS_TORCH:` skips loading checkpoint weights entirely.

---

### Obs 3: Dead Code / Duplicate Incomplete Method Definition in `trainer.py`
In `src/training/trainer.py` (lines 494–515):
```python
494:     def evaluate(self, eval_dataset: Optional[Any] = None) -> Dict[str, float]:
495:         """Evaluates model against official metrics (Weighted F1, SoftCons, HardCons)."""
496:         data = eval_dataset if eval_dataset is not None else self.eval_dataset_raw
497:         if data is None:
498:             return {}
499: 
500:         records = (
501:             data.to_dict("records")
502:             if hasattr(data, "to_dict") and callable(getattr(data, "to_dict"))
503:             else list(data)
504:         )
505:         pairs = pair_instances(records)
506: 
507:         y_true_dir: List[str] = []
508:         y_pred_dir: List[str] = []
509:         y_true_rev: List[str] = []
510:         y_pred_rev: List[str] = []
511:         is_rev_mask: List[bool] = []
512: 
513:         if hasattr(self.model, "eval"):
514:             self.model.eval()
515: 
516:     def _predict_pair_indices(self, t1: str, t2: str, apply_sd: bool) -> Tuple[int, int]:
...
538:             return idx_d, idx_r
539: 
540:     def evaluate(self, eval_dataset: Optional[Any] = None) -> Dict[str, float]:
```
Lines 494–515 define an incomplete `evaluate` method that stops abruptly at `self.model.eval()`. Python overwrites this stub at line 540 when the real `evaluate` is defined, but lines 494–515 remain dead code.

---

### Obs 4: PyTorch Version Incompatibility in GradScaler Instantiation
In `src/training/trainer.py` (line 401):
```python
scaler = torch.amp.GradScaler("cuda", enabled=(self.config.fp16 and str(self.device).startswith("cuda")))
```
Passing the device string `"cuda"` as the first positional argument to `torch.amp.GradScaler` was introduced in PyTorch 2.3. On GPU compute nodes running PyTorch 1.12 through 2.2, `torch.cuda.amp.GradScaler(enabled=...)` is required; calling `torch.amp.GradScaler("cuda", ...)` raises `TypeError` or `AttributeError`. (In contrast, `src/training/utils.py` lines 192–199 correctly implements fallbacks between `torch.amp.autocast` and `torch.cuda.amp.autocast`).

---

### Obs 5: Test Discovery & Quick-Test Execution
- Command `python3 -m unittest discover -s tests -p "test_*.py" -v`:
  Output: `Ran 104 tests in 0.840s. OK.` (100% pass on pure CPU mock mode).
- Command `bash scripts/run_gpu_training.sh --quick-test`:
  Output: Completed all 8 benchmark runs in `00m01s` on CPU. Generated valid submissions in `results/`, valid JSON summaries in `logs/`, valid checkpoint directories in `checkpoints/`, and verified Codabench submission compliance (`PASSED`).
- `test_training.py` only tests with `is_mock=True` or `--test_run`. The real PyTorch code pathway `_train_pytorch` was never exercised by any unit test.

---

## 2. Logic Chain

1. From Obs 1, `experiments/train.py` calls `pair_instances(df_train)` for CRL methods and passes the resulting list of `ReciprocalPair` objects to `Trainer`.
2. From Obs 1, `Trainer._prepare_dataset` assumes its input is raw records and invokes `pair_instances(records)` again. Because `pair_instances` is not idempotent on `ReciprocalPair` tuples, it discards all pairs and produces a dataset of length 0.
3. Therefore, any remote GPU training run for `crl` and `crl_sd` will fail to train on any samples (0 batches per epoch).
4. From Obs 2, for methods `sd` and `crl_sd`, `build_model` wraps the underlying backbone in `SymmetrizedModelWrapper`.
5. From Obs 2, `SymmetrizedModelWrapper` exposes neither `named_parameters()` nor `parameters()`, and does not implement `__getattr__`.
6. From Obs 2, `Trainer._train_pytorch` directly queries `self.model.named_parameters()` and `self.model.parameters()`. Both return empty lists, causing `torch.optim.AdamW` to throw `ValueError: optimizer got an empty parameter list`.
7. Therefore, remote GPU training for `sd` and `crl_sd` will crash immediately at epoch initialization.
8. From Obs 5, the unit tests and the `--quick-test` bash flag only execute `_train_mock()`. In `_train_mock()`, `_prepare_dataset()` is bypassed and `self.model.base_model` is explicitly checked, masking both Obs 1 and Obs 2 from the local test suite.
9. From Obs 3 and 4, `src/training/trainer.py` contains redundant dead code (lines 494–515) and lacks backward-compatible GradScaler instantiation for PyTorch < 2.3.
10. Based on steps 3, 7, and 9, Milestone 3 deliverables cannot be approved until these defects are resolved.

---

## 3. Caveats

- Testing was performed on a host macOS environment without CUDA hardware. The PyTorch CUDA training path was analyzed via static analysis, attribute introspection, and mocked dependency tracing.
- Data ingestion, label reversal algebra, Codabench CSV formatting, and mock execution pipelines are solid and adhere to specification.
- No integrity violations (hardcoded test results or fake implementations) were found; the issues are genuine architectural integration bugs between the model wrappers, the data pipeline, and the PyTorch training loop.

---

## 4. Conclusion & Required Changes

The implementation exhibits high quality in CLI design, POSIX shell automation, and Codabench compliance, but contains **two Critical defects** that will cause runtime failure on remote GPU servers for methods `crl`, `sd`, and `crl_sd`.

### Required Fixes:
1. **Fix ReciprocalPair Ingestion in `_prepare_dataset`**:
   In `src/training/trainer.py` (`_prepare_dataset`):
   ```python
   if is_paired:
       if records and isinstance(records[0], ReciprocalPair):
           pairs = records
       else:
           pairs = pair_instances(records)
       return PairedReversalDataset(pairs=pairs, tokenizer=self.tokenizer, max_length=self.config.max_length)
   ```
2. **Enable Parameter and State Delegation in `SymmetrizedModelWrapper` & `Trainer`**:
   In `src/models/factory.py` (`SymmetrizedModelWrapper`), implement attribute delegation:
   ```python
   def __getattr__(self, name: str) -> Any:
       return getattr(self.base_model, name)
   ```
   Or explicitly implement `parameters()`, `named_parameters()`, `state_dict()`, and `load_state_dict()`.
   In `src/training/trainer.py` (`_train_pytorch`), resolve the trainable model:
   ```python
   target_model = getattr(self.model, "base_model", self.model)
   named_params = list(target_model.named_parameters()) if hasattr(target_model, "named_parameters") else []
   ```
3. **Remove Dead Incomplete Stub in `trainer.py`**:
   Delete lines 494–515 of `src/training/trainer.py`.
4. **Add PyTorch < 2.3 Fallback for GradScaler**:
   In `src/training/trainer.py` (line 401), instantiate GradScaler with fallback between `torch.amp.GradScaler` and `torch.cuda.amp.GradScaler`.
5. **Add Integration Unit Test for `_prepare_dataset` with `crl`**:
   Add a test in `tests/test_training.py` verifying that passing `train_pairs` into `Trainer(method='crl')` yields `len(ds) > 0`.

---

## 5. Verification Method

To verify the requested fixes after implementation:
1. **Verify `_prepare_dataset` with Pre-Paired Data**:
   ```bash
   python3 -c "
   from src.data.loader import load_phrasis_split
   from src.data.reversal import pair_instances
   from src.training.trainer import Trainer, TrainerConfig
   from src.models.factory import build_model
   df = load_phrasis_split('track1', 'train')
   pairs = pair_instances(df)
   trainer = Trainer(model=build_model('mock', method='crl', is_mock=True),
                     config=TrainerConfig(method='crl', is_mock=True),
                     train_dataset=pairs)
   ds = trainer._prepare_dataset(trainer.train_dataset_raw, is_paired=True)
   assert len(ds) == 27, f'Expected 27 pairs, got {len(ds)}'
   print('PASSED: Aligned reciprocal dataset has length', len(ds))
   "
   ```
2. **Verify Parameter Delegation on SymmetrizedModelWrapper**:
   ```bash
   python3 -c "
   from src.models.factory import SymmetrizedModelWrapper
   class MockModule:
       def parameters(self): return [1]
       def named_parameters(self): return [('w', 1)]
       def state_dict(self): return {'w': 1}
   wrapped = SymmetrizedModelWrapper(MockModule())
   assert hasattr(wrapped, 'parameters') and len(list(wrapped.parameters())) == 1
   assert hasattr(wrapped, 'named_parameters') and len(list(wrapped.named_parameters())) == 1
   assert hasattr(wrapped, 'state_dict') and wrapped.state_dict() == {'w': 1}
   print('PASSED: SymmetrizedModelWrapper delegates PyTorch Module methods')
   "
   ```
3. **Run Full Test Suite**:
   ```bash
   python3 -m unittest discover -s tests -p "test_*.py" -v
   bash scripts/run_gpu_training.sh --quick-test
   ```
