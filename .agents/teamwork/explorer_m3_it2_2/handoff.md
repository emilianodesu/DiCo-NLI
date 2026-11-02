# Investigation & Code Remediation Blueprint: SymmetrizedModelWrapper Delegation

**Agent:** Explorer 2 (`explorer_m3_it2_2`)  
**Mission:** Analyze defects in `src/models/factory.py` regarding `SymmetrizedModelWrapper` parameter and state delegation and produce an exact code remediation blueprint for the remediation Worker.  
**Date:** 2026-11-02T02:46:00Z  
**Target:** `src/models/factory.py` (and test additions in `tests/test_models.py`)  

---

## 1. Observation

### Obs 1: Missing Parameter & State Delegation on SymmetrizedModelWrapper
In `src/models/factory.py` (lines 194–310):
`SymmetrizedModelWrapper` wraps an underlying `base_model` (`CrossEncoderNLI` for `method="sd"`, or `SiameseCRLModel` for `method="crl_sd"`).
Inspecting the class attributes directly:
```bash
python3 -c "
from src.models.factory import build_model, SymmetrizedModelWrapper
m_sd = build_model('mock', method='sd')
print('hasattr parameters:', hasattr(m_sd, 'parameters'))
print('hasattr named_parameters:', hasattr(m_sd, 'named_parameters'))
print('hasattr state_dict:', hasattr(m_sd, 'state_dict'))
print('hasattr load_state_dict:', hasattr(m_sd, 'load_state_dict'))
"
```
Output:
```text
hasattr parameters: False
hasattr named_parameters: False
hasattr state_dict: False
hasattr load_state_dict: False
```

### Obs 2: Crash in PyTorch Training Loop (`_train_pytorch`)
In `src/training/trainer.py` (lines 360–380):
```python
no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "RMSNorm.weight"]
named_params = list(self.model.named_parameters()) if hasattr(self.model, "named_parameters") else []
if named_params:
    optimizer_grouped_parameters = [
        {"params": [p for n, p in named_params if not any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": self.config.weight_decay},
        {"params": [p for n, p in named_params if any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0},
    ]
else:
    optimizer_grouped_parameters = self.model.parameters() if hasattr(self.model, "parameters") else []

optimizer = self.optimizer or torch.optim.AdamW(
    optimizer_grouped_parameters,
    lr=self.config.learning_rate,
    eps=1e-8,
)
```
When `method="sd"` or `method="crl_sd"`, `self.model` is an instance of `SymmetrizedModelWrapper`.
Because neither `named_parameters` nor `parameters` exists on `self.model`:
`named_params` evaluates to `[]`, `optimizer_grouped_parameters` evaluates to `[]`, and `torch.optim.AdamW([], lr=...)` throws:
`ValueError: optimizer got an empty parameter list`.

### Obs 3: Downstream Silent Failures in Gradient Clipping and Checkpointing
1. In `src/training/trainer.py` (lines 439–440, 448–449):
   `if hasattr(self.model, "parameters"): torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)`
   Because `hasattr(self.model, "parameters")` is `False`, gradient clipping is completely bypassed during GPU training.
2. In `src/training/trainer.py` (line 665):
   `"model_state_dict": self.model.state_dict() if hasattr(self.model, "state_dict") else {}`
   Because `hasattr(self.model, "state_dict")` is `False`, checkpoints save an empty dictionary `{}` instead of model weights.
3. In `experiments/train.py` (line 361):
   `if hasattr(model, "load_state_dict") and HAS_TORCH:`
   Because `hasattr(model, "load_state_dict")` is `False`, resuming or loading from a checkpoint silently skips weight loading.
4. In `src/models/factory.py` (lines 296–299):
   ```python
   def to(self, device: Any):
       if hasattr(self.base_model, "to"):
           self.base_model.to(device)
       return self
   ```
   `to()` only accepts `device: Any`. PyTorch's `to()` method accepts `(*args, **kwargs)` (e.g. `model.to(device, dtype=torch.float16, non_blocking=True)`). Calling with keyword args raises `TypeError: to() got an unexpected keyword argument 'dtype'`.

### Obs 4: `nn.Module` Subclassing Pitfall vs Delegation Pattern
If `SymmetrizedModelWrapper` were changed to inherit from `torch.nn.Module`:
1. `self.base_model` would be registered as a child submodule (`self._modules["base_model"]`).
2. Every parameter name in `named_parameters()` would be prepended with `"base_model."` (e.g., `"base_model.transformer.embeddings..."`).
3. Checkpoint keys in `state_dict()` would be prepended with `"base_model."`, breaking standard Hugging Face weight loading, Codabench inference scripts, and standalone evaluator scripts.
4. On CPU environments without PyTorch installed (`HAS_TORCH = False`), `nn.Module` is not available.
5. If `base_model` is a pure-Python mock (`MockCrossEncoder`), PyTorch `nn.Module` refuses non-`nn.Module` child assignment or fails to track parameters.
In contrast, a transparent delegation wrapper pattern with fallback `__getattr__` perfectly preserves parameter names, state dictionary keys, device transfers, and works identically across PyTorch GPU models and CPU mock objects.

---

## 2. Logic Chain

1. From Obs 1 and Obs 2, when `build_model(..., method="sd")` or `build_model(..., method="crl_sd")` is initialized, `Trainer._train_pytorch` checks for `named_parameters` and `parameters` on the outer model.
2. Because `SymmetrizedModelWrapper` does not expose these methods, `optimizer_grouped_parameters` is empty, causing immediate runtime abortion with `ValueError: optimizer got an empty parameter list`.
3. From Obs 3, checkpoint saving (`state_dict`) and checkpoint loading (`load_state_dict`) silently fail, producing empty checkpoints and breaking model resumption.
4. From Obs 4, subclassing `nn.Module` introduces dangerous parameter key prefixes (`"base_model."`) that corrupt model checkpoints and fails when `torch` is not installed or when mock objects are wrapped.
5. Therefore, the optimal and robust remediation is to equip `SymmetrizedModelWrapper` with explicit, argument-tolerant delegation methods for:
   - `parameters(*args, **kwargs)`
   - `named_parameters(*args, **kwargs)`
   - `state_dict(*args, **kwargs)`
   - `load_state_dict(state_dict, *args, **kwargs)`
   - `named_buffers(*args, **kwargs)`
   - `buffers(*args, **kwargs)`
   - `zero_grad(*args, **kwargs)`
   - `save_pretrained(save_directory, *args, **kwargs)`
   - `to(*args, **kwargs)`
   - `eval()`
   - `train(mode=True)`
   - `__getattr__(name)` for general attribute forwarding (e.g. `device`, `config`).
6. Additionally, equipping `MockTransformerModel` in `src/models/factory.py` with parameter stubs (`parameters`, `named_parameters`, `state_dict`, `load_state_dict`) ensures seamless duck-typing consistency across all mock unit tests.

---

## 3. Caveats

- Testing was executed on a macOS host environment without CUDA hardware. Full PyTorch CUDA execution paths were validated using mocked PyTorch modules and static dependency analysis.
- `Trainer._train_pytorch` in `src/training/trainer.py` also benefits from resolving `target_model = getattr(self.model, "base_model", self.model)` as defense in depth (handled by Worker 1). The changes to `SymmetrizedModelWrapper` ensure that even without changes to `trainer.py`, any optimizer, clipper, or checkpoint serializer directly inspecting `self.model` functions flawlessly.
- No other model classes require delegation modifications: `CrossEncoderNLI` and `SiameseCRLModel` already expose full `nn.Module` APIs in PyTorch mode.

---

## 4. Conclusion & Exact Remediation Blueprint

### Target 1: `src/models/factory.py`

#### Part A: Add parameter stubs to `MockTransformerModel`
At lines 124–127 in `src/models/factory.py`:
```python
<<<< BEFORE (lines 124-127):
    def train(self, mode: bool = True) -> MockTransformerModel:
        self.training = mode
        return self

==== AFTER:
    def train(self, mode: bool = True) -> MockTransformerModel:
        self.training = mode
        return self

    def parameters(self, *args: Any, **kwargs: Any) -> Any:
        return iter([])

    def named_parameters(self, *args: Any, **kwargs: Any) -> Any:
        return iter([])

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Any, *args: Any, **kwargs: Any) -> Any:
        return None
>>>>
```

#### Part B: Add `__getattr__` and delegation methods to `SymmetrizedModelWrapper`
1. Below `__init__` (line 201), add `__getattr__`:
```python
<<<< BEFORE (lines 197-204):
    def __init__(self, base_model: Any, mask_negative_other: bool = False) -> None:
        self.base_model = base_model
        self.mask_negative_other = mask_negative_other
        self.decoder = SymmetrizedDecoder(mask_negative_other=mask_negative_other)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_model(*args, **kwargs)

==== AFTER:
    def __init__(self, base_model: Any, mask_negative_other: bool = False) -> None:
        self.base_model = base_model
        self.mask_negative_other = mask_negative_other
        self.decoder = SymmetrizedDecoder(mask_negative_other=mask_negative_other)

    def __getattr__(self, name: str) -> Any:
        """Forward attribute lookup to base model for PyTorch & HF interoperability."""
        try:
            base = self.__dict__["base_model"]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(base, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_model(*args, **kwargs)
>>>>
```

2. Replace `to`, `eval`, and `train` (lines 296–309) with the full delegation suite:
```python
<<<< BEFORE (lines 296-309):
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

==== AFTER:
    def parameters(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate parameter iterator to base model."""
        if hasattr(self.base_model, "parameters"):
            try:
                return self.base_model.parameters(*args, **kwargs)
            except TypeError:
                return self.base_model.parameters()
        return iter([])

    def named_parameters(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate named parameter iterator to base model."""
        if hasattr(self.base_model, "named_parameters"):
            try:
                return self.base_model.named_parameters(*args, **kwargs)
            except TypeError:
                return self.base_model.named_parameters()
        return iter([])

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Delegate state dictionary retrieval to base model."""
        if hasattr(self.base_model, "state_dict"):
            return self.base_model.state_dict(*args, **kwargs)
        return {}

    def load_state_dict(self, state_dict: Any, *args: Any, **kwargs: Any) -> Any:
        """Delegate state dictionary loading to base model."""
        if hasattr(self.base_model, "load_state_dict"):
            return self.base_model.load_state_dict(state_dict, *args, **kwargs)
        return None

    def named_buffers(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate named buffer iterator to base model."""
        if hasattr(self.base_model, "named_buffers"):
            try:
                return self.base_model.named_buffers(*args, **kwargs)
            except TypeError:
                return self.base_model.named_buffers()
        return iter([])

    def buffers(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate buffer iterator to base model."""
        if hasattr(self.base_model, "buffers"):
            try:
                return self.base_model.buffers(*args, **kwargs)
            except TypeError:
                return self.base_model.buffers()
        return iter([])

    def zero_grad(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate gradient zeroing to base model."""
        if hasattr(self.base_model, "zero_grad"):
            return self.base_model.zero_grad(*args, **kwargs)
        return None

    def save_pretrained(self, save_directory: str, *args: Any, **kwargs: Any) -> Any:
        """Delegate save_pretrained to base model."""
        if hasattr(self.base_model, "save_pretrained"):
            return self.base_model.save_pretrained(save_directory, *args, **kwargs)
        return None

    def to(self, *args: Any, **kwargs: Any) -> SymmetrizedModelWrapper:
        """Forward device or dtype transfer to base model and return self."""
        if hasattr(self.base_model, "to"):
            try:
                self.base_model.to(*args, **kwargs)
            except TypeError:
                if args:
                    self.base_model.to(args[0])
                elif "device" in kwargs:
                    self.base_model.to(kwargs["device"])
        return self

    def eval(self) -> SymmetrizedModelWrapper:
        """Set base model to evaluation mode, returning self."""
        if hasattr(self.base_model, "eval"):
            self.base_model.eval()
        return self

    def train(self, mode: bool = True) -> SymmetrizedModelWrapper:
        """Set base model training mode, returning self."""
        if hasattr(self.base_model, "train"):
            self.base_model.train(mode)
        return self
>>>>
```

---

### Target 2: `tests/test_models.py`

Add a dedicated unit test in `TestModels` after line 479:
```python
    def test_symmetrized_model_wrapper_parameter_and_state_delegation(self):
        """Verify SymmetrizedModelWrapper delegates PyTorch module methods (parameters, state_dict, to)."""
        class MockTrainableModule:
            def __init__(self):
                self.device = "cpu"
                self.training = True
                self.zeroed = False
            def parameters(self, recurse: bool = True):
                return ["param_w", "param_b"]
            def named_parameters(self, prefix: str = "", recurse: bool = True):
                return [("weight", "param_w"), ("bias", "param_b")]
            def state_dict(self, *args, **kwargs):
                return {"weight": [1.0, 2.0], "bias": [0.0]}
            def load_state_dict(self, state_dict, strict: bool = True):
                return "loaded_successfully"
            def to(self, *args, **kwargs):
                self.device = args[0] if args else kwargs.get("device", "cpu")
                return self
            def eval(self):
                self.training = False
                return self
            def train(self, mode: bool = True):
                self.training = mode
                return self
            def zero_grad(self):
                self.zeroed = True

        base = MockTrainableModule()
        wrapped = SymmetrizedModelWrapper(base)

        # 1. Parameter delegation
        self.assertTrue(hasattr(wrapped, "parameters"))
        self.assertEqual(list(wrapped.parameters()), ["param_w", "param_b"])
        self.assertEqual(list(wrapped.parameters(recurse=False)), ["param_w", "param_b"])

        # 2. Named parameter delegation
        self.assertTrue(hasattr(wrapped, "named_parameters"))
        self.assertEqual(list(wrapped.named_parameters()), [("weight", "param_w"), ("bias", "param_b")])

        # 3. State dict delegation
        self.assertTrue(hasattr(wrapped, "state_dict"))
        self.assertEqual(wrapped.state_dict(), {"weight": [1.0, 2.0], "bias": [0.0]})

        # 4. Load state dict delegation
        self.assertTrue(hasattr(wrapped, "load_state_dict"))
        self.assertEqual(wrapped.load_state_dict({"weight": [1.0]}), "loaded_successfully")

        # 5. Device transfer delegation
        self.assertIs(wrapped.to("cuda:0"), wrapped)
        self.assertEqual(base.device, "cuda:0")

        # 6. Mode switching
        self.assertIs(wrapped.eval(), wrapped)
        self.assertFalse(base.training)
        self.assertIs(wrapped.train(True), wrapped)
        self.assertTrue(base.training)

        # 7. Reviewer test contract (zero-argument methods)
        class ReviewerMock:
            def parameters(self): return [1]
            def named_parameters(self): return [('w', 1)]
            def state_dict(self): return {'w': 1}

        r_wrapped = SymmetrizedModelWrapper(ReviewerMock())
        self.assertTrue(hasattr(r_wrapped, 'parameters') and len(list(r_wrapped.parameters())) == 1)
        self.assertTrue(hasattr(r_wrapped, 'named_parameters') and len(list(r_wrapped.named_parameters())) == 1)
        self.assertTrue(hasattr(r_wrapped, 'state_dict') and r_wrapped.state_dict() == {'w': 1})
```

---

## 5. Verification Method

To verify the implementation independently after applying the patch:

### Test 1: Reviewer Verification Snippet
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

### Test 2: Factory Build Model Attribute Inspection for SD and CRL+SD
```bash
python3 -c "
from src.models.factory import build_model
for m in ['sd', 'crl_sd']:
    model = build_model('mock', method=m)
    assert hasattr(model, 'parameters'), f'{m} missing parameters'
    assert hasattr(model, 'named_parameters'), f'{m} missing named_parameters'
    assert hasattr(model, 'state_dict'), f'{m} missing state_dict'
    assert hasattr(model, 'load_state_dict'), f'{m} missing load_state_dict'
    assert hasattr(model, 'to'), f'{m} missing to'
    assert hasattr(model, 'eval'), f'{m} missing eval'
    assert hasattr(model, 'train'), f'{m} missing train'
    assert model.to('cpu') is model
print('PASSED: build_model builds wrapper exposing all PyTorch delegation methods')
"
```

### Test 3: Run Full Test Suite
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
bash scripts/run_gpu_training.sh --quick-test
```
All unit tests must pass with 0 failures, and the quick-test smoke runner must complete with all 8 benchmark runs passing.
