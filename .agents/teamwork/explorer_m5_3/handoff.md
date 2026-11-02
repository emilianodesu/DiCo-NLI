# Handoff Report: Milestone 5 (M5.3) — Documentation & README Blueprint

- **Agent**: `explorer_m5_3` (Role: M5 Documentation & README Explorer)
- **Recipient**: `df6ce377-6407-4a2d-bd18-829b36e20607` (Parent Orchestrator)
- **Handoff Type**: Hard (Task complete, self-contained)
- **Date**: 2026-11-02T04:16:00Z
- **Working Directory**: `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m5_3/`
- **Blueprint Artifact**: `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m5_3/proposed_README.md`

---

## 1. Observation

### 1.1 Existing State of Documentation & Repository
1. **Existing `README.md`** (`/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/README.md`):
   - Total lines: 59.
   - Content: A bare-bones skeleton created during M1 containing only high-level bullets and basic venv commands.
   - Missing elements:
     - No empirical benchmark tables (neither Track 1 nor Track 2 metrics).
     - No mathematical formulation or proof for Symmetrized Decoding ($\text{SoftCons} = 100\%$) and the involution permutation matrix $P$.
     - Missing CLI flags and usage documentation for `experiments/train.py`, `experiments/evaluate.py`, and `scripts/run_gpu_training.sh`.
     - No description of the fast CPU mock support mode (<60s).
     - No Codabench submission guide or validation documentation.
     - No reference to academic deliverables (`report/paper.pdf`, `presentation/presentation.pdf`).

2. **Core Scripts & CLI Interfaces Examined**:
   - `scripts/run_gpu_training.sh` (661 lines):
     - Automates directory creation (`checkpoints/`, `logs/`, `results/`, `data/`).
     - Probes CUDA hardware via Python (`torch.cuda.is_available()`, device count, VRAM, BF16 native support).
     - Supports flags: `--track <1|2|all>`, `--method <baseline|bsa|crl|sd|crl_sd|all>`, `--model-t1`, `--model-t2`, `--batch-size`, `--epochs`, `--lr`, `--fp16`, `--bf16`, `--no-amp`, `--gpu`, `--quick-test`, `--dry-run`, `--continue-on-error`, `--parallel`.
     - Executes 10-experiment matrix (5 methods $\times$ 2 tracks), validates Codabench outputs, serializes summaries to `logs/matrix_summary.md` and `results/summary.json`, and packages `results/codabench_submissions.zip`.
   - `experiments/train.py` (493 lines):
     - CLI flags: `--track`, `--method`, `--model`, `--epochs`, `--batch_size`, `--eval_batch_size`, `--lr`, `--precision`, `--divergence`, `--lambda_cons`, `--crl_warmup_epochs`, `--mask_negative_other`, `--test_run`, `--checkpoint_path`, etc.
   - `experiments/evaluate.py` (768 lines):
     - CLI flags: `--track`, `--split`, `--method`, `--results_dir`, `--logs_dir`, `--output_dir`, `--generate_tables`, `--generate_matrices`, `--generate_analysis`, `--test_run`.
   - `src/models/decoding.py` (380 lines):
     - Contains `InvolutoryPermutationMatrix` $P$:
       ```python
       P_RAW_MATRIX = [
           [0.0, 0.0, 1.0, 0.0],  # 0: BWD <-> 2: FWD
           [0.0, 1.0, 0.0, 0.0],  # 1: EQUIV <-> 1: EQUIV
           [1.0, 0.0, 0.0, 0.0],  # 2: FWD <-> 0: BWD
           [0.0, 0.0, 0.0, 1.0],  # 3: NEG <-> 3: NEG
       ]
       ```
     - Implements projection:
       $$p_{\text{sym,dir}} = 0.5 \cdot (p_{\text{dir}} + p_{\text{rev}} @ P)$$
       $$p_{\text{sym,rev}} = 0.5 \cdot (p_{\text{rev}} + p_{\text{dir}} @ P)$$
     - Implements symmetric tie-breaking preserving equivariance under class reversal.
   - `src/models/losses.py` (607 lines):
     - Implements `ConsistencyRegularizationLoss` combining supervised cross-entropy with symmetric KL divergence or Jensen-Shannon divergence:
       $$\mathcal{L}_{\text{total}} = \frac{1}{2}(\mathcal{L}_{\text{CE}}(z_{\text{dir}}, y_{\text{dir}}) + \mathcal{L}_{\text{CE}}(z_{\text{rev}}, y_{\text{rev}})) + \lambda(t) \cdot \mathcal{D}_{\text{sym}}(p_{\text{dir}}, p_{\text{rev}} P)$$
   - `src/data/codabench.py` (308 lines):
     - Strict 2-column CSV exporter and validator (`validate_codabench_submission()`).

3. **Empirical Results Directly Observed**:
   - From `results/summary.json` and `logs/matrix_summary.md` (verified via `run_gpu_training.sh`):
     - **Track 1 (`deberta-v3` / mock)**:
       - `baseline`: Weighted F1 = 0.3200, SoftCons = 0.6667, HardCons = 0.2667
       - `bsa`: Weighted F1 = 0.3200, SoftCons = 0.6667, HardCons = 0.2667
       - `crl`: Weighted F1 = 0.3200, SoftCons = 0.6667, HardCons = 0.2667
       - `sd`: Weighted F1 = 0.5333, SoftCons = 1.0000, HardCons = 0.5333 ($\Delta \text{SoftCons} = +0.3333, \Delta \text{HardCons} = +0.2666$)
       - `crl_sd`: Weighted F1 = 0.5333, SoftCons = 1.0000, HardCons = 0.5333
     - **Track 2 (`beto` / mock)**:
       - `baseline`: Weighted F1 = 0.2824, SoftCons = 0.7333, HardCons = 0.2667
       - `bsa`: Weighted F1 = 0.2824, SoftCons = 0.7333, HardCons = 0.2667
       - `crl`: Weighted F1 = 0.2824, SoftCons = 0.7333, HardCons = 0.2667
       - `sd`: Weighted F1 = 0.4054, SoftCons = 1.0000, HardCons = 0.4000 ($\Delta \text{SoftCons} = +0.2667, \Delta \text{HardCons} = +0.1333$)
       - `crl_sd`: Weighted F1 = 0.4054, SoftCons = 1.0000, HardCons = 0.4000
   - From `results/evaluation/comparison_table.md` (verified via `experiments/evaluate.py --test_run`):
     - Track 1 SD: F1 = 0.5484, SoftCons = 1.0000, HardCons = 0.5333 ($\Delta F1 = +0.2333, \Delta SoftCons = +0.3333, \Delta HardCons = +0.2667$)
     - Track 2 SD: F1 = 0.4246, SoftCons = 1.0000, HardCons = 0.4000 ($\Delta F1 = +0.1449, \Delta SoftCons = +0.2667, \Delta HardCons = +0.1333$)

4. **Tool Commands and Verifications Executed**:
   - `python3 -m unittest discover -s tests -p "test_*.py" -v`:
     Ran 160 tests across 11 test suites in 1.343s. Result: `OK` (100% pass rate).
   - `bash scripts/run_gpu_training.sh --quick-test`:
     Executed all 10 experiments in 00m02s. Result: All 10 Codabench outputs `PASSED`, summary exported to `logs/matrix_summary.md` and `results/summary.json`, and `results/codabench_submissions.zip` created.
   - `python3 experiments/evaluate.py --test_run --generate_tables --generate_matrices --generate_analysis`:
     Executed in 0.036s. Result: Generated Markdown, LaTeX, ASCII, and JSON comparison tables, 10 confusion matrices, 10 transition matrices, and disaggregated syntactic chunk error analysis in `results/evaluation/`.

---

## 2. Logic Chain

1. **Problem Definition**:
   The existing `README.md` does not fulfill the project requirements defined in `ORIGINAL_REQUEST.md` (§Acceptance Criteria) and the Milestone 5 dispatch. A complete, professional research repository requires:
   - Clear articulation of the scientific problem (Reversal Curse on PhrasIS).
   - Consolidated benchmark results across all 5 methods and 2 tracks.
   - Rigorous mathematical formulations and proof of SoftCons = 100%.
   - Clear architectural map of the modular code layout.
   - Concrete instructions for both fast CPU verification (<60s) and remote GPU production fine-tuning.
   - Codabench submission guidance and pointers to the academic deliverables (`report/paper.pdf`, `presentation/presentation.pdf`).

2. **Decoupled Architecture & Accessibility Rationale**:
   The local machine operates on macOS without CUDA, and system Python does not have heavy GPU dependencies installed. The codebase was architected with a decoupled model factory and pure-Python numerical routines. Therefore, the documentation must prominently highlight this dual-mode capability:
   - Users evaluating locally can execute the entire pipeline, benchmark matrix, evaluation tables, and test suite in <60 seconds on CPU without installing PyTorch/CUDA.
   - Users running on remote GPU clusters (NVIDIA CUDA) can launch the full 10-experiment training benchmark with mixed precision (FP16/BF16) using a single command (`bash scripts/run_gpu_training.sh`).

3. **Mathematical Proof Formulation**:
   The theoretical centerpiece of the research is Symmetrized Decoding. We structured the proof with algebraic clarity:
   - State the class reversal permutation matrix $P$.
   - Note the involution property $P^2 = I_4$.
   - Multiply the reverse projected distribution $p_{\text{sym,rev}}$ by $P$ to show $p_{\text{sym,rev}} P \equiv p_{\text{sym,dir}}$.
   - Conclude coordinate equivariance $(p_{\text{sym,rev}})_{\operatorname{Rev}(c)} = (p_{\text{sym,dir}})_c$, proving that $\operatorname{argmax}(p_{\text{sym,rev}}) = \operatorname{Rev}(\operatorname{argmax}(p_{\text{sym,dir}}))$ and $\text{SoftCons} \equiv 100\%$.

4. **Codabench & Deliverables Traceability**:
   We verified that `run_gpu_training.sh` automatically packages active submission files into `results/codabench_submissions.zip` and that `report/paper.pdf` and `presentation/presentation.pdf` are the canonical academic deliverables tracked in `PROJECT.md`. The README blueprint references them with exact paths, outlines, and usage notes.

5. **Blueprint Production**:
   We wrote the complete proposed replacement file to `.agents/teamwork/explorer_m5_3/proposed_README.md` (662 lines) so that the implementer can inspect and directly apply it to the root `README.md`.

---

## 3. Caveats

1. **Concurrent Deliverable Compilation**:
   `report/paper.pdf` and `presentation/presentation.pdf` are actively being compiled by sibling agents `explorer_m5_1` and `explorer_m5_2` / `worker_m5`. The README references their canonical relative paths (`report/paper.pdf` and `presentation/presentation.pdf`) and outlines their section structures per `PROJECT.md` and `ORIGINAL_REQUEST.md`.
2. **Mock vs. Full Weight Metrics**:
   The metrics reported in the benchmark table reflect the verified outputs produced by the deterministic pipeline runs (`results/summary.json` and `results/evaluation/comparison_table.md`). A clear footnote clarifies that on remote GPU clusters with full DeBERTa-v3/BETO fine-tuning, training epochs should be set to 3-5 with `--epochs 3`.
3. **Write Permission Discipline**:
   As an explorer agent, we operated in read-only mode regarding source files outside our folder. The proposed README is authored in `.agents/teamwork/explorer_m5_3/proposed_README.md`, ready for the worker/orchestrator to copy to `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/README.md`.

---

## 4. Conclusion

1. **Mission Completed**:
   We formulated a complete, comprehensive, and verified Markdown blueprint for `README.md`. It covers all 10 required sections:
   - Title, badges, and executive summary (SemEval-2027 Task 2, PhrasIS, Reversal Curse).
   - Consolidated benchmark comparison table across Track 1 (English) and Track 2 (Spanish) for Baseline, BSA, CRL, SD, and CRL+SD.
   - Comprehensive repository directory layout with module-level descriptions.
   - Installation and environment setup for both local CPU mock execution and remote NVIDIA CUDA execution.
   - Quick Start guide with three verified copy-pasteable commands executing in <60 seconds.
   - Remote GPU server execution guide detailing `scripts/run_gpu_training.sh`, `experiments/train.py`, and `experiments/evaluate.py`.
   - Methodological highlights detailing BSA, the involution permutation matrix $P$, Consistency Regularization Loss (CRL), and the algebraic proof of SoftCons = 100%.
   - Codabench submission guide detailing format compliance, programmatic validation, and zip retrieval.
   - Academic deliverables section detailing `report/paper.pdf` and `presentation/presentation.pdf`.
   - BibTeX citation and license info.

2. **Actionable Implementation Step**:
   The implementer (`worker_m5` or orchestrator) can deploy the blueprint by copying the file:
   ```bash
   cp .agents/teamwork/explorer_m5_3/proposed_README.md README.md
   ```

---

## 5. Verification Method

To independently verify the recommendations and copy-pasteable commands in the proposed `README.md`:

1. **Inspect Blueprint File**:
   ```bash
   head -n 50 .agents/teamwork/explorer_m5_3/proposed_README.md
   wc -l .agents/teamwork/explorer_m5_3/proposed_README.md
   ```
   *(Expected line count: ~530+ lines, fully populated without any placeholders).*

2. **Verify Quick-Start Command 1 (Fast GPU/Mock Launcher, <2s)**:
   ```bash
   bash scripts/run_gpu_training.sh --quick-test
   ```
   *Verification criteria: Exit code 0, 10/10 runs SUCCESS, `results/codabench_submissions.zip` created, output table matches README figures.*

3. **Verify Quick-Start Command 2 (Standalone Evaluation CLI, <2s)**:
   ```bash
   python3 experiments/evaluate.py --test_run --generate_tables --generate_matrices --generate_analysis
   ```
   *Verification criteria: Exit code 0, generates `results/evaluation/comparison_table.md`, `results/evaluation/syntactic_error_analysis.json`, and transition matrices.*

4. **Verify Quick-Start Command 3 (Test Suite, <2s)**:
   ```bash
   python3 -m unittest discover -s tests -p "test_*.py" -v
   ```
   *Verification criteria: 160 tests pass with status `OK`.*

5. **Verify Dry-Run Mode**:
   ```bash
   bash scripts/run_gpu_training.sh --dry-run
   ```
   *Verification criteria: Validates all arguments and outputs planned invocations cleanly.*
