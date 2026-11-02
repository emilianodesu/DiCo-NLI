# Handoff Report: Explorer 3 — Metrics Defense & Pipeline Decoupling (M1 Iteration 2)

- **Agent:** `explorer_m1_it2_3`
- **Archetype / Roles:** explorer / investigator, synthesizer
- **Milestone:** M1 (Data Pipeline & Consistency Metrics) — Iteration 2
- **Recipient:** Orchestrator (`df6ce377-6407-4a2d-bd18-829b36e20607`)
- **Date:** 2026-11-01T22:22:00Z

---

## 1. Observation

### Observation 1: Exception on Slices with Zero Reversible Pairs
Direct execution of `src.metrics.metrics.compute_official_metrics` with non-reversible items:
```bash
python3 -c "
from src.metrics.metrics import compute_official_metrics
res = compute_official_metrics(
    ['NEGATIVE_OTHER'], ['NEGATIVE_OTHER'],
    ['NEGATIVE_OTHER'], ['NEGATIVE_OTHER'],
    [False]
)
"
```
Verbatim Traceback:
```
Traceback (most recent call last):
  File "<string>", line 3, in <module>
    res = compute_official_metrics(
  File "/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/src/metrics/metrics.py", line 419, in compute_official_metrics
    cons_report = compute_consistency_report(pairs, pred_map)
  File "/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/src/metrics/metrics.py", line 336, in compute_consistency_report
    raise ValueError("Cannot compute consistency metrics without reversible pairs.")
ValueError: Cannot compute consistency metrics without reversible pairs.
```
Inspecting `src/metrics/metrics.py:408-425`:
```python
    for g_d, p_d, g_r, p_r, is_rev in zip(
        y_true_dir, y_pred_dir, y_true_rev, y_pred_rev, is_reversible_mask
    ):
        if is_rev:
            id_d = f"inst_{pair_idx}_dir"
            id_r = f"inst_{pair_idx}_rev"
            pairs.append(ReversiblePair(id_d, id_r, g_d, g_r))
            pred_map[id_d] = p_d
            pred_map[id_r] = p_r
            pair_idx += 1

    cons_report = compute_consistency_report(pairs, pred_map)
```
When `is_reversible_mask` has no `True` elements, `pairs` is empty (`len(pairs) == 0`). Calling `compute_consistency_report(pairs, pred_map)` unconditionally crashes with `ValueError`.

### Observation 2: `tests.test_pipeline.SRC_AVAILABLE` Evaluates to `False`
Direct execution in the project workspace:
```bash
python3 -c "import tests.test_pipeline as tp; print('tests.test_pipeline.SRC_AVAILABLE:', tp.SRC_AVAILABLE)"
```
Output:
```
tests.test_pipeline.SRC_AVAILABLE: False
```
Inspecting `tests/test_pipeline.py:23-35`:
```python
try:
    from src.data.codabench import export_codabench_submission, validate_codabench_submission
    from src.data.reversal import build_reversal_dataset, pair_instances, reverse_label
    from src.metrics.metrics import compute_official_metrics
    from src.models.decoding import SymmetrizedDecoder
    SRC_AVAILABLE = True
except ImportError:
    SRC_AVAILABLE = False
    from tests.test_codabench import export_codabench_submission, validate_codabench_submission
    from tests.test_metrics import compute_official_metrics
    from tests.test_reversal_logic import build_reversal_dataset, pair_instances, reverse_label
    from tests.test_symmetrized_decoding import SymmetrizedDecoder
```
Testing the imports individually:
```bash
python3 -c "
from src.data.codabench import export_codabench_submission, validate_codabench_submission
from src.data.reversal import build_reversal_dataset, pair_instances, reverse_label
from src.metrics.metrics import compute_official_metrics
print('M1 imports: OK')
"
# Output: M1 imports: OK

python3 -c "from src.models.decoding import SymmetrizedDecoder"
# Output: ModuleNotFoundError: No module named 'src.models'
```
Because `src/models/` is a Milestone 2 deliverable, line 27 raised `ModuleNotFoundError`, which triggered `except ImportError:`, causing `test_pipeline.py` to overwrite all M1 imports with fallback mocks from `tests/`.

### Observation 3: Duck-Typing Failure in `compute_consistency_report` with `ReciprocalPair`
In `src/metrics/metrics.py:343`:
```python
if pair.first_id not in predicted_labels or pair.second_id not in predicted_labels:
```
Testing `compute_consistency_report` with `ReciprocalPair` objects from `src.data.reversal.pair_instances`:
```bash
python3 -c "
from src.metrics.metrics import compute_consistency_report
from src.data.reversal import pair_instances
pairs = pair_instances([{'instance_id': '1', 'pair_id': 'p1', 'label': 'EQUIVALENCE', 'text1': 'a', 'text2': 'b', 'reverse_pair_id': '2'}, {'instance_id': '2', 'pair_id': 'p1', 'label': 'EQUIVALENCE', 'text1': 'b', 'text2': 'a', 'reverse_pair_id': '1'}])
compute_consistency_report(pairs, {'1': 'EQUIVALENCE', '2': 'EQUIVALENCE'})
"
```
Output:
```
AttributeError: 'ReciprocalPair' object has no attribute 'first_id'
```
`ReciprocalPair` stores items as a 2-tuple `(direct_instance, inverse_instance)`, so accessing `.first_id` raises `AttributeError`.

---

## 2. Logic Chain

1. **Defense on Zero Reversible Pairs:**
   - From Observation 1: During training, batches or slices containing only `NEGATIVE_OTHER` have `is_reversible_mask = [False, ...]`.
   - The consistency rate over 0 reversible instances is mathematically undefined ($0/0$). Under standard numeric library conventions (matching `safe_divide(0, 0)`), empty sets must return `0.0`.
   - In `compute_official_metrics`, checking `if not pairs: return {"weighted_f1": weighted_f1, "soft_cons": 0.0, "hard_cons": 0.0}` cleanly prevents `ValueError` while accurately preserving the multiclass `weighted_f1` for the slice.
   - Enforcing `len(y_true_dir) == len(y_pred_dir) == len(y_true_rev) == len(y_pred_rev) == len(is_reversible_mask)` prevents silent truncation by `zip()`.

2. **Decoupling Pipeline Tests Across Milestones:**
   - From Observation 2: `tests/test_pipeline.py` coupled M1 deliverables (`src.data`, `src.metrics`) with M2 deliverables (`src.models.decoding`).
   - Bundling both in a single `try` block caused `test_pipeline.py` to completely decouple from `src/`, executing only mock test oracles.
   - Decoupling the M1 import block from M2 model import allows `test_pipeline.py` to test M1 code live (`SRC_AVAILABLE: True`), while using a stub/oracle for `SymmetrizedDecoder` until M2 is implemented.
   - Adding an explicit unit test `test_m1_source_available` guarantees that `SRC_AVAILABLE` is enforced programmatically by `unittest`.
   - Adding DataFrame ingestion validation into `test_pipeline.py` ensures that `build_reversal_dataset` is exercised against real DataFrames in the end-to-end pipeline test.

3. **Duck-Typing Interoperability:**
   - From Observation 3: In `src/metrics/consistency.py:208-218`, pair extraction accepts `hasattr(pair, "first_id")`, `isinstance(pair, (tuple, list))`, and `isinstance(pair, dict)`.
   - Adopting this same polymorphic extraction pattern in `src/metrics/metrics.py:compute_consistency_report` resolves Reviewer 2 Finding 3 and allows `pair_instances()` outputs to be passed directly to `compute_consistency_report()`.

---

## 3. Caveats

- **No Caveats.** All observations were confirmed by exact Python commands and source inspection in the repository workspace. No speculative assumptions were made.

---

## 4. Conclusion

The blueprints provided in `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m1_it2_3/report.md` provide complete, drop-in solutions:
1. `src/metrics/metrics.py`:
   - `compute_official_metrics` checks sequence lengths and returns `soft_cons: 0.0, hard_cons: 0.0` when `pairs` is empty.
   - `compute_consistency_report` adds polymorphic duck typing for `ReciprocalPair`, tuples, and dicts.
2. `tests/test_pipeline.py`:
   - Separates M1 imports (`SRC_AVAILABLE`) from M2 model imports (`M2_MODELS_AVAILABLE`).
   - Adds `test_m1_source_available` and DataFrame ingestion verification.
3. `tests/test_metrics.py`:
   - Adds unit tests for zero reversible pairs slice, length mismatch, and duck typing.

---

## 5. Verification Method

To independently verify the reported blueprint and fixes:

1. **Verify M1 Decoupling and `SRC_AVAILABLE: True`:**
   ```bash
   python3 -c "
   import tests.test_codabench as tc
   import tests.test_metrics as tm
   import tests.test_reversal_logic as tr
   import tests.test_pipeline as tp
   assert tc.SRC_AVAILABLE and tm.SRC_AVAILABLE and tr.SRC_AVAILABLE and tp.SRC_AVAILABLE, 'All test modules must have SRC_AVAILABLE: True'
   print('VERIFICATION PASS: All modules test live src')
   "
   ```

2. **Verify Zero Reversible Pairs Slice:**
   ```bash
   python3 -c "
   from src.metrics.metrics import compute_official_metrics
   res = compute_official_metrics(
       ['NEGATIVE_OTHER'], ['NEGATIVE_OTHER'],
       ['NEGATIVE_OTHER'], ['NEGATIVE_OTHER'],
       [False]
   )
   assert res['soft_cons'] == 0.0 and res['hard_cons'] == 0.0
   print('VERIFICATION PASS: Zero reversible slice handled safely:', res)
   "
   ```

3. **Verify Sequence Length Mismatch Guard:**
   ```bash
   python3 -c "
   from src.metrics.metrics import compute_official_metrics
   try:
       compute_official_metrics(['A'], ['A'], ['A'], ['A'], [True, False])
       assert False, 'Should have raised ValueError'
   except ValueError:
       print('VERIFICATION PASS: Length mismatch caught')
   "
   ```

4. **Run Full Test Suite:**
   ```bash
   python3 -m unittest discover -s tests -p "test_*.py" -v
   ```
