# Investigation & Architecture Report: Metrics Defense & Pipeline Decoupling (M1 Iteration 2)

**Agent:** `explorer_m1_it2_3`  
**Milestone:** M1 (Data Pipeline & Consistency Metrics) — Iteration 2  
**Focus:** Metrics Defensive Handling (Zero Reversible Slices) & Pipeline Test Decoupling  
**Date:** 2026-11-01T22:21:00Z  

---

## Executive Summary

During Milestone 1 Iteration 1 review, two critical defects and one test architecture flaw were identified:
1. **Unhandled Exception in `compute_official_metrics` (Finding F-03):** When evaluated on data slices with zero reversible pairs (e.g. all `NEGATIVE_OTHER` samples, or arbitrary batch slices during training), `compute_official_metrics` unconditionally forwards an empty list `pairs = []` to `compute_consistency_report()`, which raises `ValueError("Cannot compute consistency metrics without reversible pairs.")`.
2. **False Verification Claim & Mock Bypass in `tests/test_pipeline.py` (Finding F-01 & F-04):** `test_pipeline.py` grouped M1 imports (`src.data.*`, `src.metrics.*`) with unwritten Milestone 2 model code (`from src.models.decoding import SymmetrizedDecoder`) in a single `try` block. Because `src/models/` is not yet implemented, `test_pipeline.py` threw `ImportError`, set `SRC_AVAILABLE = False`, and loaded mock fallbacks from `tests/`. This prevented the end-to-end smoke test from ever exercising `src/` code, thereby concealing severe bugs like the DataFrame ingestion crash in `build_reversal_dataset`.
3. **Implicit Silent Truncation & Duck-Typing Brittleness:** In `compute_official_metrics`, `zip()` silently truncated if `is_reversible_mask` had a different length from the label sequences, and `compute_consistency_report` crashed with `AttributeError` if passed `ReciprocalPair` objects from `src.data.reversal.pair_instances`.

This report provides complete root cause analysis and a production-grade, drop-in blueprint for Worker M1 to resolve these issues deterministically.

---

## 1. Defensive Handling in `compute_official_metrics`

### 1.1 Failure Mode Analysis
In `src/metrics/metrics.py:405-425`:
```python
    pairs: List[ReversiblePair] = []
    pred_map: Dict[str, str] = {}
    pair_idx = 0
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
When `is_reversible_mask` contains no `True` values:
- `pairs` is empty (`len(pairs) == 0`).
- Line 419 calls `compute_consistency_report(pairs, pred_map)`.
- In `compute_consistency_report(pairs, ...)` (lines 334-336):
  ```python
  total = len(pairs)
  if total == 0:
      raise ValueError("Cannot compute consistency metrics without reversible pairs.")
  ```
- Result: Unhandled `ValueError` crashing evaluation loops.

### 1.2 Mathematical & Architectural Rationale for 0.0
- **Support-based justification:** The consistency metrics $\text{SoftCons} = \frac{\sum \text{soft\_ok}}{N_{\text{rev}}}$ and $\text{HardCons} = \frac{\sum \text{hard\_ok}}{N_{\text{rev}}}$ are defined over the set of reversible pairs $R$. When $|R| = 0$, the denominator is 0.
- Under IEEE 754 float arithmetic conventions and standard library zero-division handling (as implemented in `safe_divide(0, 0) == 0.0`), empty sets must return `0.0`.
- Returning `0.0` prevents training/evaluation loop aborts on mini-batches or category slices (e.g., negative-only slices) while preserving the valid `weighted_f1` computed across all items.

### 1.3 Strict Sequence Length Validation
The standard library `zip()` stops at the shortest input sequence. If `is_reversible_mask` has a different length than `y_true_dir`, metrics would be silently evaluated on only a prefix of the dataset. An explicit guard must enforce:
```python
if not (
    len(y_true_dir)
    == len(y_pred_dir)
    == len(y_true_rev)
    == len(y_pred_rev)
    == len(is_reversible_mask)
):
    raise ValueError(...)
```

### 1.4 Universal Pair Duck-Typing in `compute_consistency_report`
Currently, `compute_consistency_report` strictly accesses `pair.first_id` and `pair.second_id`. Passing `ReciprocalPair` from `src.data.reversal.pair_instances` causes `AttributeError: 'ReciprocalPair' object has no attribute 'first_id'`.
Enhancing `compute_consistency_report` with polymorphic duck-typing (matching `src/metrics/consistency.py:208-218`) enables seamless interoperability with:
- `ReversiblePair` (attributes `first_id`, `second_id`, `first_label`, `second_label`)
- `ReciprocalPair` / 2-tuple of dicts or `InstanceRecord`
- Standard dictionary specifications (`{"first_id": ..., "second_id": ...}`)

---

## 2. Decoupling of `tests/test_pipeline.py`

### 2.1 Root Cause of `SRC_AVAILABLE: False`
`tests/test_pipeline.py:23-35` bundled all imports together:
```python
try:
    from src.data.codabench import export_codabench_submission, validate_codabench_submission
    from src.data.reversal import build_reversal_dataset, pair_instances, reverse_label
    from src.metrics.metrics import compute_official_metrics
    from src.models.decoding import SymmetrizedDecoder   # <--- M2 module!
    SRC_AVAILABLE = True
except ImportError:
    SRC_AVAILABLE = False
    from tests.test_codabench import export_codabench_submission, validate_codabench_submission
    from tests.test_metrics import compute_official_metrics
    from tests.test_reversal_logic import build_reversal_dataset, pair_instances, reverse_label
    from tests.test_symmetrized_decoding import SymmetrizedDecoder
```
Because `src/models/decoding.py` is an unwritten Milestone 2 deliverable, `from src.models.decoding import SymmetrizedDecoder` raised `ModuleNotFoundError: No module named 'src.models'`. This triggered the `except ImportError:` branch, which:
1. Set `SRC_AVAILABLE = False`.
2. Overwrote the real `src.data` and `src.metrics` imports with test fallbacks.
3. Completely bypassed `src/` code during pipeline test execution.

### 2.2 Clean Decoupling Architecture
To guarantee that M1 code is tested live while gracefully supporting progressive milestone implementation:
1. **M1 Live Source Import Block:** Import `src.data.codabench`, `src.data.reversal`, and `src.metrics.metrics`. If any M1 import fails, `SRC_AVAILABLE = False` (or raise immediately).
2. **M2 Decoupled Import Block:** Attempt to import `src.models.decoding.SymmetrizedDecoder`. If unavailable, cleanly fall back to `tests.test_symmetrized_decoding.SymmetrizedDecoder` and flag `M2_MODELS_AVAILABLE = False`.
3. **Explicit Assertion Test:** Add `test_m1_source_available()` in `TestPipeline` asserting `self.assertTrue(SRC_AVAILABLE)`. This guarantees no future regression can silently mask `SRC_AVAILABLE: False`.
4. **DataFrame Ingestion Integration:** Add an explicit test step inside `test_fast_cpu_smoke_pipeline` verifying `build_reversal_dataset` against `load_phrasis_split('track1', 'dev')` (both unaugmented and BSA augmented), preventing regression of Finding F-02.

---

## 3. Exact Code Patch Blueprint for Worker

### Patch 1: `src/metrics/metrics.py`

#### Change 1.1: Universal Pair Duck-Typing in `compute_consistency_report`
**File:** `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/src/metrics/metrics.py`  
**Location:** Lines 326–390

```python
def compute_consistency_report(
    pairs: Sequence[Any],
    predicted_labels: Mapping[str, str],
    max_pair_errors: int = 50,
) -> ConsistencyReport:
    """
    Compute official SoftCons and HardCons over reversible source pairs.
    Accepts ReversiblePair, ReciprocalPair, 2-tuples of records/dicts, or pair dicts.
    """
    total = len(pairs)
    if total == 0:
        raise ValueError("Cannot compute consistency metrics without reversible pairs.")

    soft_count = 0
    hard_count = 0
    pair_errors: List[Dict[str, str]] = []

    for pair in pairs:
        # Polymorphic ID and ground-truth label extraction
        if hasattr(pair, "first_id") and hasattr(pair, "second_id"):
            id1 = pair.first_id
            id2 = pair.second_id
            gold1 = getattr(pair, "first_label", None)
            gold2 = getattr(pair, "second_label", None)
        elif isinstance(pair, (tuple, list)) and len(pair) == 2:
            p1, p2 = pair[0], pair[1]
            id1 = p1.get("instance_id") if hasattr(p1, "get") else getattr(p1, "instance_id", str(p1))
            id2 = p2.get("instance_id") if hasattr(p2, "get") else getattr(p2, "instance_id", str(p2))
            gold1 = p1.get("label") if hasattr(p1, "get") else getattr(p1, "label", None)
            gold2 = p2.get("label") if hasattr(p2, "get") else getattr(p2, "label", None)
        elif isinstance(pair, dict):
            id1 = pair["first_id"]
            id2 = pair["second_id"]
            gold1 = pair.get("first_label")
            gold2 = pair.get("second_label")
        else:
            raise TypeError(f"Unsupported pair type for consistency evaluation: {type(pair)}")

        if id1 not in predicted_labels or id2 not in predicted_labels:
            raise KeyError(f"Missing predictions for pair ({id1}, {id2})")

        pred_1 = predicted_labels[id1]
        pred_2 = predicted_labels[id2]

        soft_ok = (
            pred_1 in REVERSIBLE_LABELS
            and pred_2 == REVERSE_LABEL_MAP.get(pred_1)
        )
        hard_ok = (
            pred_1 == gold1
            and pred_2 == gold2
        ) if (gold1 is not None and gold2 is not None) else False

        if soft_ok:
            soft_count += 1
        if hard_ok:
            hard_count += 1
        elif len(pair_errors) < max_pair_errors:
            pair_errors.append({
                "first_id": id1,
                "second_id": id2,
                "first_gold": str(gold1),
                "second_gold": str(gold2),
                "first_prediction": pred_1,
                "second_prediction": pred_2,
                "soft_consistent": str(soft_ok).lower(),
                "hard_correct": str(hard_ok).lower(),
            })

    soft_cons = soft_count / total
    hard_cons = hard_count / total

    assert hard_cons <= soft_cons + 1e-12, (
        f"Mathematical invariant violated: HardCons ({hard_cons}) > SoftCons ({soft_cons})"
    )

    return ConsistencyReport(
        soft_cons=soft_cons,
        hard_cons=hard_cons,
        reversible_pairs=total,
        soft_consistent_pairs=soft_count,
        hard_correct_pairs=hard_count,
        pair_errors=pair_errors,
    )
```

#### Change 1.2: Defensive Handling & Length Check in `compute_official_metrics`
**File:** `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/src/metrics/metrics.py`  
**Location:** Lines 391–425

```python
def compute_official_metrics(
    y_true_dir: Sequence[str],
    y_pred_dir: Sequence[str],
    y_true_rev: Sequence[str],
    y_pred_rev: Sequence[str],
    is_reversible_mask: Sequence[bool],
) -> Dict[str, float]:
    """
    Compute the official metric trio: Weighted F1, SoftCons, HardCons.

    Defensive behaviors:
    - Enforces identical lengths across all 5 input sequences to prevent silent zip truncation.
    - If is_reversible_mask contains zero True values (no reversible pairs in slice),
      returns 0.0 for soft_cons and hard_cons instead of raising ValueError.
    """
    n_dir = len(y_true_dir)
    if not (
        len(y_pred_dir) == n_dir
        and len(y_true_rev) == n_dir
        and len(y_pred_rev) == n_dir
        and len(is_reversible_mask) == n_dir
    ):
        raise ValueError(
            f"Input length mismatch in compute_official_metrics: "
            f"y_true_dir={len(y_true_dir)}, y_pred_dir={len(y_pred_dir)}, "
            f"y_true_rev={len(y_true_rev)}, y_pred_rev={len(y_pred_rev)}, "
            f"is_reversible_mask={len(is_reversible_mask)}"
        )

    all_gold = list(y_true_dir) + list(y_true_rev)
    all_pred = list(y_pred_dir) + list(y_pred_rev)
    weighted_f1 = compute_weighted_f1(all_gold, all_pred)

    pairs: List[ReversiblePair] = []
    pred_map: Dict[str, str] = {}
    pair_idx = 0
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

    if not pairs:
        # Zero reversible pairs in this evaluation slice: return 0.0 without exception
        return {
            "weighted_f1": weighted_f1,
            "soft_cons": 0.0,
            "hard_cons": 0.0,
        }

    cons_report = compute_consistency_report(pairs, pred_map)
    return {
        "weighted_f1": weighted_f1,
        "soft_cons": cons_report.soft_cons,
        "hard_cons": cons_report.hard_cons,
    }
```

---

### Patch 2: `tests/test_pipeline.py`

#### Change 2.1: Decoupled Imports by Milestone
**File:** `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/tests/test_pipeline.py`  
**Location:** Lines 22–36

```python
# Milestone 1: Live source code imports (Data pipeline and consistency metrics)
try:
    from src.data.codabench import export_codabench_submission, validate_codabench_submission
    from src.data.reversal import build_reversal_dataset, pair_instances, reverse_label
    from src.metrics.metrics import compute_official_metrics
    SRC_AVAILABLE = True
except ImportError:
    SRC_AVAILABLE = False
    from tests.test_codabench import export_codabench_submission, validate_codabench_submission
    from tests.test_metrics import compute_official_metrics
    from tests.test_reversal_logic import build_reversal_dataset, pair_instances, reverse_label

# Milestone 2: Neural modeling & decoding (Decoupled: uses live src.models if available; fallback to test oracle)
try:
    from src.models.decoding import SymmetrizedDecoder
    M2_MODELS_AVAILABLE = True
except ImportError:
    M2_MODELS_AVAILABLE = False
    from tests.test_symmetrized_decoding import SymmetrizedDecoder
```

#### Change 2.2: Add Explicit `test_m1_source_available` and DataFrame Pipeline Step
**File:** `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/tests/test_pipeline.py`  
**Location:** Inside `class TestPipeline(unittest.TestCase)` (around lines 160–180)

```python
    def test_m1_source_available(self):
        """Verify that M1 source modules are imported directly from src/ without fallback."""
        self.assertTrue(
            SRC_AVAILABLE,
            "M1 source modules (src.data, src.metrics) must be imported live from src/."
        )

    def test_fast_cpu_smoke_pipeline(self):
        """Execute complete end-to-end training, inference, export, validation, and scoring in <60s."""
        start_time = time.time()

        # Step 1: Generate synthetic corpus
        corpus = generate_synthetic_phrasis_benchmark()
        self.assertGreaterEqual(len(corpus), 48, "Synthetic benchmark should contain at least 48 instances")

        # Step 2: Build reversible pairing and BSA dataset (lists of dicts)
        paired = pair_instances(corpus)
        self.assertGreater(len(paired), 0, "Must identify canonical reciprocal pairs")
        bsa_corpus = build_reversal_dataset(corpus, augment_bsa=False)

        # Step 2b: Verify DataFrame ingestion via load_phrasis_split (Finding F-02 regression guard)
        from src.data.loader import load_phrasis_split
        df_dev = load_phrasis_split("track1", "dev")
        df_unaugmented = build_reversal_dataset(df_dev, augment_bsa=False)
        self.assertEqual(len(df_unaugmented), len(df_dev), "Unaugmented DataFrame must preserve row count")
        df_augmented = build_reversal_dataset(df_dev, augment_bsa=True)
        self.assertGreater(len(df_augmented), len(df_dev), "BSA augmented DataFrame must add synthetic pairs")

        # Step 3: Fast CPU Training Simulation
        ...
```

---

### Patch 3: `tests/test_metrics.py`

#### Change 3.1: Synchronize Fallback `compute_official_metrics`
**File:** `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/tests/test_metrics.py`  
**Location:** Lines 154–172

```python
        if not pairs:
            return {
                "weighted_f1": weighted_f1,
                "soft_cons": 0.0,
                "hard_cons": 0.0,
            }

        cons_report = compute_consistency_report(pairs, pred_map)
        return {
            "weighted_f1": weighted_f1,
            "soft_cons": cons_report.soft_cons,
            "hard_cons": cons_report.hard_cons,
        }
```

#### Change 3.2: Add Unit Tests for Edge Cases & Duck-Typing
**File:** `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/tests/test_metrics.py`  
**Location:** Append inside `class TestMetrics(unittest.TestCase)` (after line 344)

```python
    def test_compute_official_metrics_zero_reversible_slice(self):
        """When is_reversible_mask has no True values, soft_cons and hard_cons return 0.0 without error."""
        y_true_dir = ["NEGATIVE_OTHER", "NEGATIVE_OTHER"]
        y_pred_dir = ["NEGATIVE_OTHER", "NEGATIVE_OTHER"]
        y_true_rev = ["NEGATIVE_OTHER", "NEGATIVE_OTHER"]
        y_pred_rev = ["NEGATIVE_OTHER", "NEGATIVE_OTHER"]
        is_rev = [False, False]

        metrics = compute_official_metrics(
            y_true_dir, y_pred_dir, y_true_rev, y_pred_rev, is_rev
        )
        self.assertIn("weighted_f1", metrics)
        self.assertIn("soft_cons", metrics)
        self.assertIn("hard_cons", metrics)
        self.assertEqual(metrics["soft_cons"], 0.0)
        self.assertEqual(metrics["hard_cons"], 0.0)
        self.assertAlmostEqual(metrics["weighted_f1"], 1.0)

    def test_compute_official_metrics_length_mismatch(self):
        """Mismatched sequence lengths in compute_official_metrics must raise ValueError."""
        with self.assertRaises(ValueError):
            compute_official_metrics(
                ["EQUIVALENCE"],
                ["EQUIVALENCE"],
                ["EQUIVALENCE"],
                ["EQUIVALENCE"],
                [True, False],  # Length 2 vs 1
            )

    def test_compute_consistency_report_duck_typing(self):
        """compute_consistency_report must accept ReciprocalPair objects and 2-tuples."""
        from src.data.reversal import InstanceRecord, ReciprocalPair

        p = ReciprocalPair(
            InstanceRecord(instance_id="d1", pair_id="pair_1", label="FORWARD_ENTAILMENT"),
            InstanceRecord(instance_id="r1", pair_id="pair_1", label="BACKWARD_ENTAILMENT"),
        )
        preds = {"d1": "FORWARD_ENTAILMENT", "r1": "BACKWARD_ENTAILMENT"}
        report = compute_consistency_report([p], preds)
        self.assertEqual(report.soft_cons, 1.0)
        self.assertEqual(report.hard_cons, 1.0)
```

---

## 4. Verification & Acceptance Commands

Worker M1 should run these exact commands to verify implementation:

1. **Verify `SRC_AVAILABLE` is True across all test suites:**
   ```bash
   python3 -c "
   import tests.test_codabench as tc
   import tests.test_metrics as tm
   import tests.test_reversal_logic as tr
   import tests.test_pipeline as tp
   assert tc.SRC_AVAILABLE, 'test_codabench fallback active'
   assert tm.SRC_AVAILABLE, 'test_metrics fallback active'
   assert tr.SRC_AVAILABLE, 'test_reversal_logic fallback active'
   assert tp.SRC_AVAILABLE, 'test_pipeline fallback active'
   print('ALL TEST MODULES CONFIRMED SRC_AVAILABLE: True')
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
   print('ZERO REVERSIBLE SLICE PASS:', res)
   "
   ```

3. **Verify Length Mismatch Detection:**
   ```bash
   python3 -c "
   from src.metrics.metrics import compute_official_metrics
   try:
       compute_official_metrics(['A'], ['A'], ['A'], ['A'], [True, False])
       assert False, 'Failed to catch length mismatch'
   except ValueError:
       print('LENGTH MISMATCH DETECTION PASS')
   "
   ```

4. **Verify Duck-Typing with `ReciprocalPair`:**
   ```bash
   python3 -c "
   from src.metrics.metrics import compute_consistency_report
   from src.data.reversal import ReciprocalPair, InstanceRecord
   p = ReciprocalPair(
       InstanceRecord(instance_id='d1', pair_id='p1', label='FORWARD_ENTAILMENT'),
       InstanceRecord(instance_id='r1', pair_id='p1', label='BACKWARD_ENTAILMENT')
   )
   rep = compute_consistency_report([p], {'d1': 'FORWARD_ENTAILMENT', 'r1': 'BACKWARD_ENTAILMENT'})
   assert rep.soft_cons == 1.0 and rep.hard_cons == 1.0
   print('DUCK TYPING PASS')
   "
   ```

5. **Run Complete Automated Test Suite:**
   ```bash
   python3 -m unittest discover -s tests -p "test_*.py" -v
   ```
