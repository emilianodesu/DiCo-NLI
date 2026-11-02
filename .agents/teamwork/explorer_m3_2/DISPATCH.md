# Dispatch: Explorer 2 Milestone 3 (Training Engine, Mixed Precision & Checkpointing)

## Mission
Analyze requirements and technical design for `src/training/trainer.py` and `src/training/utils.py`.

## Key Documents
- Read `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/ORIGINAL_REQUEST.md` (Authoritative user requirements §R3)
- Read `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/PROJECT.md` (Project blueprint and interface contracts)
- Inspect `src/models/` (losses, cross_encoder, siamese_crl, factory) and `src/metrics/`

## Scope
1. Design the `Trainer` class in `src/training/trainer.py`:
   - Dual execution support: real PyTorch training loop (AdamW, linear scheduler with warmup, gradient clipping, AMP with `torch.cuda.amp.autocast` / `GradScaler` or `torch.amp`) AND fast deterministic pure-Python / NumPy CPU mock training loop (<60s execution for unit testing).
   - Support for both standard cross-encoder training (Baseline, BSA) and Siamese CRL training (pairs with consistency regularization loss).
   - Evaluation step: evaluating validation set after every epoch, computing official metrics (Weighted F1, SoftCons, HardCons), tracking best checkpoint based on validation score.
   - Checkpointing: saving model weights / state dict, optimizer state, scheduler state, and training metadata to `checkpoints/<run_name>/best_model.pt` and `checkpoint_last.pt`.
   - Logging: structured JSON logs / metrics history to `logs/<run_name>.jsonl`.
2. Design `src/training/utils.py`:
   - `set_seed(seed: int)`
   - Hardware detection: `get_device(preferred: Optional[str] = None) -> torch.device` (handles `cuda`, `mps`, `cpu`)
   - Mixed precision detection: `can_use_amp() -> bool`, `get_autocast_dtype(fp16: bool, bf16: bool)`
3. Provide concrete implementation blueprint, edge cases (e.g. AMP dtype for transposition matrix $P$, zero division in metrics), and test strategies for the Worker.
4. Write your findings to `/Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m3_2/handoff.md`.


## 2026-11-01T23:32:21Z
You are Explorer 2 for Milestone 3 (CLI Training Pipeline & GPU Scripts).
Your working directory is: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m3_2/
Please read your assignment in: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m3_2/DISPATCH.md
Read the authoritative requirements in: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/ORIGINAL_REQUEST.md
Read the project architecture in: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/PROJECT.md

Investigate requirements and technical design for `src/training/trainer.py` and `src/training/utils.py`. Cover both real PyTorch training loop (with AMP, AdamW, scheduler, checkpointing, and validation callbacks) and fast CPU mock training loop (<60s budget), along with device and precision utilities. Write a comprehensive analysis and implementation blueprint to /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject/.agents/teamwork/explorer_m3_2/handoff.md following the Handoff Protocol. Notify the orchestrator via send_message when done.
