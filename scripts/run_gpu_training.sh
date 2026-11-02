#!/usr/bin/env bash
# ==============================================================================
# SemEval-2027 Task 2 (DiCo-NLI) - Master GPU Training & Benchmark Orchestrator
#
# Production execution script for remote NVIDIA GPU servers (CUDA) and local
# CPU/mock verification. Automates environment tuning, dependency checks,
# directory creation, the 8-experiment benchmark matrix, real-time logging,
# and official Codabench format verification.
#
# Usage:
#   bash scripts/run_gpu_training.sh [OPTIONS]
#
# Examples:
#   bash scripts/run_gpu_training.sh                      # Run full 8-experiment matrix on GPU
#   bash scripts/run_gpu_training.sh --quick-test         # Run fast CPU smoke test (<60s)
#   bash scripts/run_gpu_training.sh --dry-run            # Validate args & print commands
#   bash scripts/run_gpu_training.sh --track 1 --method sd # Run single experiment
#   bash scripts/run_gpu_training.sh --gpu 0 --bf16       # Specify GPU and precision
# ==============================================================================

set -euo pipefail

# ------------------------------------------------------------------------------
# 1. Project Directory Resolution & Shell Setup
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Global Execution Tracking
START_WALL_TIME=$(date +%s)
SUMMARY_RECORDS=()
CLEANUP_PIDS=()
FAILED_COUNT=0
ACTIVE_SUBMISSION_FILES=()

# Color Codes (disabled if not in a terminal or NO_COLOR is set)
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    C_RESET='\033[0m'
    C_BOLD='\033[1m'
    C_RED='\033[0;31m'
    C_GREEN='\033[0;32m'
    C_YELLOW='\033[0;33m'
    C_BLUE='\033[0;34m'
    C_CYAN='\033[0;36m'
else
    C_RESET=''
    C_BOLD=''
    C_RED=''
    C_GREEN=''
    C_YELLOW=''
    C_BLUE=''
    C_CYAN=''
fi

log_info()    { echo -e "${C_CYAN}[INFO]${C_RESET} $*"; }
log_warn()    { echo -e "${C_YELLOW}[WARN]${C_RESET} $*" >&2; }
log_error()   { echo -e "${C_RED}[ERROR]${C_RESET} $*" >&2; }
log_success() { echo -e "${C_GREEN}[SUCCESS]${C_RESET} $*"; }
log_section() {
    echo ""
    echo -e "${C_BOLD}${C_BLUE}======================================================================${C_RESET}"
    echo -e "${C_BOLD}${C_BLUE} $*${C_RESET}"
    echo -e "${C_BOLD}${C_BLUE}======================================================================${C_RESET}"
}

# ------------------------------------------------------------------------------
# 2. Trap Handlers & Clean Exit
# ------------------------------------------------------------------------------
cleanup() {
    local exit_code=$?
    if [ ${#CLEANUP_PIDS[@]} -gt 0 ]; then
        log_warn "Terminating background processes: ${CLEANUP_PIDS[*]}"
        kill "${CLEANUP_PIDS[@]}" 2>/dev/null || true
    fi
    if [ $exit_code -ne 0 ] && [ $exit_code -ne 130 ]; then
        log_error "Pipeline failed with exit code $exit_code."
    fi
}
trap cleanup EXIT
trap 'log_warn "Interrupted by user (SIGINT)"; exit 130' INT TERM

# ------------------------------------------------------------------------------
# 3. Default Configuration Values
# ------------------------------------------------------------------------------
TRACK_FILTER="all"
METHOD_FILTER="all"
MODEL_T1="microsoft/deberta-v3-base"
MODEL_T2="dccuchile/bert-base-spanish-wwm-cased"
BATCH_SIZE=16
EPOCHS=3
LEARNING_RATE="2e-5"
PRECISION="auto"
GPU_ID=""
QUICK_TEST=0
DRY_RUN=0
CONTINUE_ON_ERROR=0
PARALLEL_TRACKS=0
ALLOW_CPU=0
PYTHON_BIN=""

# ------------------------------------------------------------------------------
# 4. Command Line Argument Parsing
# ------------------------------------------------------------------------------
print_usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Master training orchestrator for SemEval-2027 Task 2 (DiCo-NLI).

Options:
  -t, --track <1|2|track1|track2|all>
                         Track to train (default: all)
  -m, --method <baseline|bsa|crl|sd|crl_sd|all>
                         Reversal mitigation method to train (default: all)
  --model-t1 <name>      HuggingFace backbone for Track 1 (default: microsoft/deberta-v3-base)
  --model-t2 <name>      HuggingFace backbone for Track 2 (default: dccuchile/bert-base-spanish-wwm-cased)
                         Alternative: PlanTL-GOB-ES/roberta-base-bne
  -b, --batch-size <n>   Training batch size (default: 16)
  -e, --epochs <n>       Training epochs (default: 3)
  --lr <rate>            Learning rate (default: 2e-5)
  --fp16                 Force FP16 mixed precision
  --bf16                 Force BF16 mixed precision
  --no-amp               Disable automatic mixed precision (use FP32)
  -g, --gpu <id>         Specify CUDA device ID (e.g. 0 or 0,1)
  -q, --quick-test       Run fast CPU mock smoke test in <60s (all 8 experiments)
  -d, --dry-run          Print planned execution matrix and commands without running
  --continue-on-error    Continue running remaining matrix experiments if one fails
  --parallel             Run Track 1 and Track 2 concurrently if multi-GPU available
  --allow-cpu            Permit CPU execution for full training without prompt/warning
  --python <path>        Custom python executable path
  -h, --help             Show this help message and exit
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--track)
            TRACK_FILTER="$2"
            shift 2
            ;;
        -m|--method)
            METHOD_FILTER="$2"
            shift 2
            ;;
        --model-t1)
            MODEL_T1="$2"
            shift 2
            ;;
        --model-t2)
            MODEL_T2="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --fp16)
            PRECISION="fp16"
            shift 1
            ;;
        --bf16)
            PRECISION="bf16"
            shift 1
            ;;
        --no-amp)
            PRECISION="fp32"
            shift 1
            ;;
        -g|--gpu)
            GPU_ID="$2"
            shift 2
            ;;
        -q|--quick-test)
            QUICK_TEST=1
            shift 1
            ;;
        -d|--dry-run)
            DRY_RUN=1
            shift 1
            ;;
        --continue-on-error)
            CONTINUE_ON_ERROR=1
            shift 1
            ;;
        --parallel)
            PARALLEL_TRACKS=1
            shift 1
            ;;
        --allow-cpu)
            ALLOW_CPU=1
            shift 1
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Normalize track filter
case "${TRACK_FILTER}" in
    1|track1|en) TRACK_FILTER="track1" ;;
    2|track2|es) TRACK_FILTER="track2" ;;
    all)         TRACK_FILTER="all" ;;
    *) log_error "Invalid --track '${TRACK_FILTER}'. Valid: 1, 2, track1, track2, all"; exit 1 ;;
esac

# Normalize method filter
case "${METHOD_FILTER}" in
    baseline|bsa|crl|sd|crl_sd|all) ;;
    *) log_error "Invalid --method '${METHOD_FILTER}'. Valid: baseline, bsa, crl, sd, crl_sd, all"; exit 1 ;;
esac

# Quick-test overrides
if [ "$QUICK_TEST" -eq 1 ]; then
    log_info "Quick-test mode enabled: overriding backbones to mock and epochs to 1."
    MODEL_T1="mock"
    MODEL_T2="mock"
    EPOCHS=1
    BATCH_SIZE=4
fi

# ------------------------------------------------------------------------------
# 5. Environment Variables & Cache Directory Configuration
# ------------------------------------------------------------------------------
log_section "Configuring Execution Environment & Cache Hierarchies"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

# Set GPU device if provided
if [ -n "$GPU_ID" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    log_info "CUDA_VISIBLE_DEVICES set to: ${CUDA_VISIBLE_DEVICES}"
fi

# Ensure localized caching to avoid filling remote user home quotas
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

# Directory hierarchy creation
mkdir -p checkpoints logs results data/raw data/processed "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"
log_info "Directory hierarchy verified: checkpoints/, logs/, results/, data/"

# ------------------------------------------------------------------------------
# 6. Python & Hardware Dependency Probe
# ------------------------------------------------------------------------------
log_section "Hardware & Dependency Audit"

# Resolve Python binary
if [ -n "$PYTHON_BIN" ]; then
    PYTHON="$PYTHON_BIN"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    PYTHON="${VIRTUAL_ENV}/bin/python"
elif [ -x "${PROJECT_ROOT}/.venv/bin/python" ]; then
    PYTHON="${PROJECT_ROOT}/.venv/bin/python"
else
    PYTHON="$(command -v python3 || command -v python)"
fi

log_info "Using Python interpreter: ${PYTHON} ($(${PYTHON} --version 2>&1))"

# Check core python dependencies
${PYTHON} -c "
import sys
missing = []
for pkg in ['numpy']:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)
if missing:
    print(f'Missing core packages: {missing}', file=sys.stderr)
    sys.exit(1)
" || {
    log_warn "Some core packages are missing. Running pip install -r requirements.txt is recommended."
}

# Probe CUDA Hardware Capabilities via Python
PROBE_RESULT=$(${PYTHON} -c "
import sys
try:
    import torch
    has_cuda = torch.cuda.is_available()
    dev_count = torch.cuda.device_count() if has_cuda else 0
    dev_name = torch.cuda.get_device_name(0) if has_cuda else 'None'
    bf16_sup = torch.cuda.is_bf16_supported() if has_cuda and hasattr(torch.cuda, 'is_bf16_supported') else False
    cap = torch.cuda.get_device_capability(0) if has_cuda else (0, 0)
    vram_gb = (torch.cuda.get_device_properties(0).total_memory / (1024**3)) if has_cuda else 0.0
    print(f'{has_cuda}|{dev_count}|{dev_name}|{bf16_sup}|{cap[0]}.{cap[1]}|{vram_gb:.1f}')
except Exception:
    print('False|0|None|False|0.0|0.0')
" 2>/dev/null || echo "False|0|None|False|0.0|0.0")

IFS='|' read -r HAS_CUDA DEV_COUNT DEV_NAME BF16_SUP COMPUTE_CAP VRAM_GB <<< "$PROBE_RESULT"

log_info "CUDA Available:       ${HAS_CUDA}"
log_info "Device Count:         ${DEV_COUNT}"
log_info "Device Name:          ${DEV_NAME}"
log_info "Compute Capability:   ${COMPUTE_CAP}"
log_info "Total VRAM:           ${VRAM_GB} GB"
log_info "BF16 Native Support:  ${BF16_SUP}"

# Precision Auto-Detection
if [ "$PRECISION" = "auto" ]; then
    if [ "$HAS_CUDA" = "True" ]; then
        if [ "$BF16_SUP" = "True" ]; then
            PRECISION="bf16"
            log_info "Auto-resolved precision: bf16 (native Ampere/Hopper acceleration)"
        else
            PRECISION="fp16"
            log_info "Auto-resolved precision: fp16 (Volta/Turing acceleration)"
        fi
    else
        PRECISION="fp32"
        log_info "Auto-resolved precision: fp32 (CPU execution)"
    fi
fi

# Sanity Check for GPU server requirement
if [ "$HAS_CUDA" != "True" ] && [ "$QUICK_TEST" -eq 0 ] && [ "$DRY_RUN" -eq 0 ] && [ "$ALLOW_CPU" -eq 0 ]; then
    log_warn "WARNING: No CUDA-capable GPU detected by PyTorch!"
    log_warn "Full transformer fine-tuning on CPU will be extremely slow."
    log_warn "Use --quick-test for fast local testing, or pass --allow-cpu to continue."
fi

# Check parallel execution capabilities
if [ "$PARALLEL_TRACKS" -eq 1 ]; then
    if [ "$DEV_COUNT" -lt 2 ]; then
        log_warn "Multi-GPU parallel execution requested (--parallel), but only ${DEV_COUNT} device(s) detected."
        log_warn "Falling back to sequential execution queue to prevent resource contention."
        PARALLEL_TRACKS=0
    else
        log_info "Multi-device environment detected (${DEV_COUNT} devices). Parallel tracks enabled."
    fi
fi

# Log nvidia-smi if command is present
if command -v nvidia-smi &>/dev/null; then
    log_info "nvidia-smi summary:"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,utilization.gpu --format=csv,noheader || true
fi

# ------------------------------------------------------------------------------
# 7. Define Benchmark Matrix (10 Experiments)
# ------------------------------------------------------------------------------
# Format: TRACK|METHOD|BACKBONE|RUN_NAME
EXPERIMENT_MATRIX=(
    "track1|baseline|${MODEL_T1}|t1_baseline_deberta"
    "track1|bsa|${MODEL_T1}|t1_bsa_deberta"
    "track1|crl|${MODEL_T1}|t1_crl_deberta"
    "track1|sd|${MODEL_T1}|t1_sd_deberta"
    "track1|crl_sd|${MODEL_T1}|t1_crl_sd_deberta"
    "track2|baseline|${MODEL_T2}|t2_baseline_beto"
    "track2|bsa|${MODEL_T2}|t2_bsa_beto"
    "track2|crl|${MODEL_T2}|t2_crl_beto"
    "track2|sd|${MODEL_T2}|t2_sd_beto"
    "track2|crl_sd|${MODEL_T2}|t2_crl_sd_beto"
)

# Filter matrix according to user flags
PLANNED_EXPERIMENTS=()
for exp in "${EXPERIMENT_MATRIX[@]}"; do
    IFS='|' read -r exp_track exp_method exp_model exp_name <<< "$exp"
    
    if [ "$TRACK_FILTER" != "all" ] && [ "$TRACK_FILTER" != "$exp_track" ]; then
        continue
    fi
    if [ "$METHOD_FILTER" != "all" ] && [ "$METHOD_FILTER" != "$exp_method" ]; then
        continue
    fi
    PLANNED_EXPERIMENTS+=("$exp")
done

TOTAL_EXPERIMENTS=${#PLANNED_EXPERIMENTS[@]}
if [ "$TOTAL_EXPERIMENTS" -eq 0 ]; then
    log_warn "No experiments matched the specified filters: --track ${TRACK_FILTER} --method ${METHOD_FILTER}"
    exit 0
fi

log_section "Planned Benchmark Matrix (${TOTAL_EXPERIMENTS} Experiments)"
printf "${C_BOLD}%-3s %-8s %-12s %-36s %-20s${C_RESET}\n" "#" "TRACK" "METHOD" "BACKBONE" "RUN NAME"
echo "--------------------------------------------------------------------------------"
idx=1
for exp in "${PLANNED_EXPERIMENTS[@]}"; do
    IFS='|' read -r exp_track exp_method exp_model exp_name <<< "$exp"
    printf "%-3s %-8s %-12s %-36s %-20s\n" "$idx" "$exp_track" "$exp_method" "$exp_model" "$exp_name"
    ((idx++))
done
echo "--------------------------------------------------------------------------------"

if [ "$DRY_RUN" -eq 1 ]; then
    log_section "Dry-Run: Generated Invocations"
    for exp in "${PLANNED_EXPERIMENTS[@]}"; do
        IFS='|' read -r exp_track exp_method exp_model exp_name <<< "$exp"
        cmd="${PYTHON} experiments/train.py --track ${exp_track} --method ${exp_method} --model ${exp_model} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS} --lr ${LEARNING_RATE} --precision ${PRECISION} --output_dir checkpoints/${exp_name} --log_file logs/${exp_name}.log --predictions_file results/${exp_name}_submission.csv"
        [ "$QUICK_TEST" -eq 1 ] && cmd="${cmd} --test_run"
        echo "[DRY RUN] ${cmd}"
    done
    log_success "Dry run complete. All argument paths and variables valid."
    exit 0
fi

# ------------------------------------------------------------------------------
# 8. Matrix Execution Engine & Codabench Export Verification
# ------------------------------------------------------------------------------
log_section "Launching Execution Matrix"

curr_idx=1
for exp in "${PLANNED_EXPERIMENTS[@]}"; do
    IFS='|' read -r exp_track exp_method exp_model exp_name <<< "$exp"
    
    EXP_START_TIME=$(date +%s)
    log_section "Experiment [${curr_idx}/${TOTAL_EXPERIMENTS}]: ${exp_name}"
    log_info "Track:       ${exp_track}"
    log_info "Method:      ${exp_method}"
    log_info "Model:       ${exp_model}"
    log_info "Batch Size:  ${BATCH_SIZE}"
    log_info "Epochs:      ${EPOCHS}"
    log_info "Precision:   ${PRECISION}"
    
    CHECKPOINT_DIR="checkpoints/${exp_name}"
    LOG_FILE="logs/${exp_name}.log"
    JSONL_LOG="logs/${exp_name}.jsonl"
    PRED_FILE="results/${exp_name}_submission.csv"
    
    # Assemble CLI command
    CMD_ARGS=(
        "--track" "${exp_track}"
        "--method" "${exp_method}"
        "--model" "${exp_model}"
        "--batch_size" "${BATCH_SIZE}"
        "--epochs" "${EPOCHS}"
        "--lr" "${LEARNING_RATE}"
        "--precision" "${PRECISION}"
        "--output_dir" "${CHECKPOINT_DIR}"
        "--log_file" "${LOG_FILE}"
        "--predictions_file" "${PRED_FILE}"
        "--run_name" "${exp_name}"
    )
    if [ "$QUICK_TEST" -eq 1 ]; then
        CMD_ARGS+=("--test_run")
    fi
    
    log_info "Executing: ${PYTHON} experiments/train.py ${CMD_ARGS[*]}"
    
    # Execute training and tee output
    RUN_STATUS="SUCCESS"
    set +e
    ${PYTHON} experiments/train.py "${CMD_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
    RUN_EXIT_CODE=${PIPESTATUS[0]}
    set -e
    
    EXP_END_TIME=$(date +%s)
    EXP_DURATION=$((EXP_END_TIME - EXP_START_TIME))
    EXP_DUR_STR=$(printf "%02dm%02ds" $((EXP_DURATION / 60)) $((EXP_DURATION % 60)))
    
    if [ $RUN_EXIT_CODE -ne 0 ]; then
        log_error "Training failed for ${exp_name} with exit code ${RUN_EXIT_CODE}."
        RUN_STATUS="FAILED"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        if [ "$CONTINUE_ON_ERROR" -eq 0 ]; then
            log_error "Aborting matrix execution. (Use --continue-on-error to skip failing runs)."
            exit $RUN_EXIT_CODE
        fi
    fi
    
    # Codabench Export Verification Step
    VALID_STATUS="SKIPPED"
    if [ "$RUN_STATUS" = "SUCCESS" ]; then
        log_info "Verifying official Codabench submission file: ${PRED_FILE}"
        if [ ! -f "${PRED_FILE}" ]; then
            log_error "Expected prediction file ${PRED_FILE} was not created!"
            RUN_STATUS="INVALID_OUTPUT"
            VALID_STATUS="MISSING_FILE"
            FAILED_COUNT=$((FAILED_COUNT + 1))
            if [ "$CONTINUE_ON_ERROR" -eq 0 ]; then
                log_error "Aborting matrix execution due to missing submission file in ${exp_name}."
                exit 1
            fi
        else
            # Run programmatic validation via src.data.codabench with strict exit code check
            VAL_EXIT=0
            VAL_OUTPUT=$(${PYTHON} -c "
import sys
from pathlib import Path
from src.data.codabench import validate_codabench_submission
from src.data.loader import load_phrasis_split

sub_path = '${PRED_FILE}'
track = '${exp_track}'
try:
    df_dev = load_phrasis_split(track, 'dev')
    raw_ids = df_dev['instance_id']
    expected_ids = set(raw_ids.tolist() if hasattr(raw_ids, 'tolist') else raw_ids)
    validate_codabench_submission(sub_path, expected_ids=expected_ids, raise_on_error=True)
    print('VALIDATION_PASSED')
except Exception as e:
    print(f'VALIDATION_FAILED: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1) || VAL_EXIT=$?

            if [ $VAL_EXIT -eq 0 ] && [[ "$VAL_OUTPUT" == *"VALIDATION_PASSED"* ]]; then
                log_success "Codabench submission format verification: PASSED"
                VALID_STATUS="PASSED"
                ACTIVE_SUBMISSION_FILES+=("${exp_name}_submission.csv")
            else
                log_error "Codabench submission format verification FAILED: ${VAL_OUTPUT}"
                RUN_STATUS="INVALID_OUTPUT"
                VALID_STATUS="FAILED"
                FAILED_COUNT=$((FAILED_COUNT + 1))
                if [ "$CONTINUE_ON_ERROR" -eq 0 ]; then
                    log_error "Aborting matrix execution due to invalid Codabench submission in ${exp_name}."
                    exit 1
                fi
            fi
        fi
    fi
    
    # Parse Metrics from Summary JSON or JSONL if available
    METRIC_W_F1="-"
    METRIC_SOFT_CONS="-"
    METRIC_HARD_CONS="-"
    SUMMARY_JSON="logs/${exp_name}_summary.json"
    if [ -f "${SUMMARY_JSON}" ]; then
        PARSED_METRICS=$(${PYTHON} -c "
import json
try:
    with open('${SUMMARY_JSON}', 'r') as f:
        data = json.load(f)
    m = data.get('metrics', {})
    f1 = m.get('weighted_f1', '-')
    scons = m.get('soft_cons', '-')
    hcons = m.get('hard_cons', '-')
    f1_str = f'{f1:.4f}' if isinstance(f1, (int, float)) else str(f1)
    s_str = f'{scons:.4f}' if isinstance(scons, (int, float)) else str(scons)
    h_str = f'{hcons:.4f}' if isinstance(hcons, (int, float)) else str(hcons)
    print(f'{f1_str}|{s_str}|{h_str}')
except Exception:
    print('-|-|-')
" 2>/dev/null || echo "-|-|-")
        IFS='|' read -r METRIC_W_F1 METRIC_SOFT_CONS METRIC_HARD_CONS <<< "$PARSED_METRICS"
    fi
    
    # Record Summary
    # Format: RUN_NAME|TRACK|METHOD|STATUS|DURATION|W_F1|SOFT_CONS|HARD_CONS|VALID_SUB
    SUMMARY_RECORDS+=("${exp_name}|${exp_track}|${exp_method}|${RUN_STATUS}|${EXP_DUR_STR}|${METRIC_W_F1}|${METRIC_SOFT_CONS}|${METRIC_HARD_CONS}|${VALID_STATUS}")
    
    ((curr_idx++))
done

# ------------------------------------------------------------------------------
# 9. Summary Reporting & Results Serialization
# ------------------------------------------------------------------------------
TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - START_WALL_TIME))
TOTAL_DUR_STR=$(printf "%02dm%02ds" $((TOTAL_DURATION / 60)) $((TOTAL_DURATION % 60)))

log_section "Benchmark Matrix Execution Summary"

SUMMARY_MD_FILE="logs/matrix_summary.md"
SUMMARY_JSON_FILE="results/summary.json"

# Print terminal summary table
printf "${C_BOLD}%-22s %-8s %-10s %-14s %-8s %-8s %-10s %-10s %-10s${C_RESET}\n" \
    "RUN NAME" "TRACK" "METHOD" "STATUS" "TIME" "W-F1" "SOFTCONS" "HARDCONS" "CODABENCH"
echo "----------------------------------------------------------------------------------------------------------"

for record in "${SUMMARY_RECORDS[@]}"; do
    IFS='|' read -r r_name r_track r_method r_status r_time r_f1 r_scons r_hcons r_val <<< "$record"
    
    STATUS_COLOR="${C_RESET}"
    if [ "$r_status" = "SUCCESS" ]; then
        STATUS_COLOR="${C_GREEN}"
    elif [ "$r_status" = "FAILED" ] || [ "$r_status" = "INVALID_OUTPUT" ]; then
        STATUS_COLOR="${C_RED}"
    fi
    
    printf "%-22s %-8s %-10s ${STATUS_COLOR}%-14s${C_RESET} %-8s %-8s %-10s %-10s %-10s\n" \
        "$r_name" "$r_track" "$r_method" "$r_status" "$r_time" "$r_f1" "$r_scons" "$r_hcons" "$r_val"
done
echo "----------------------------------------------------------------------------------------------------------"
log_info "Total Suite Runtime: ${TOTAL_DUR_STR}"

# Write Markdown Summary Table
{
    echo "# DiCo-NLI Benchmark Matrix Execution Summary"
    echo ""
    echo "**Date**: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "**Total Wall Time**: ${TOTAL_DUR_STR}"
    echo ""
    echo "| Run Name | Track | Method | Status | Time | Weighted F1 | SoftCons | HardCons | Codabench Submission |"
    echo "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    for record in "${SUMMARY_RECORDS[@]}"; do
        IFS='|' read -r r_name r_track r_method r_status r_time r_f1 r_scons r_hcons r_val <<< "$record"
        echo "| \`${r_name}\` | ${r_track} | ${r_method} | **${r_status}** | ${r_time} | ${r_f1} | ${r_scons} | ${r_hcons} | \`${r_val}\` |"
    done
} > "${SUMMARY_MD_FILE}"
log_info "Markdown report written to: ${SUMMARY_MD_FILE}"

# Write Machine-Readable JSON Summary
${PYTHON} -c "
import json, sys
records = []
for r in sys.argv[2:]:
    parts = r.split('|')
    records.append({
        'run_name': parts[0],
        'track': parts[1],
        'method': parts[2],
        'status': parts[3],
        'duration': parts[4],
        'weighted_f1': parts[5],
        'soft_cons': parts[6],
        'hard_cons': parts[7],
        'codabench_valid': parts[8],
        'submission_file': f'results/{parts[0]}_submission.csv'
    })
with open('${SUMMARY_JSON_FILE}', 'w') as f:
    json.dump({'total_runtime': sys.argv[1], 'experiments': records}, f, indent=2)
" "${TOTAL_DUR_STR}" "${SUMMARY_RECORDS[@]}" 2>/dev/null || true
log_info "JSON report written to: ${SUMMARY_JSON_FILE}"

# Package only active submissions from current run for Codabench upload
if command -v zip &>/dev/null && [ ${#ACTIVE_SUBMISSION_FILES[@]} -gt 0 ]; then
    (cd results && rm -f codabench_submissions.zip && zip -q codabench_submissions.zip "${ACTIVE_SUBMISSION_FILES[@]}" 2>/dev/null || true)
    log_info "Packaged ${#ACTIVE_SUBMISSION_FILES[@]} active predictions to results/codabench_submissions.zip"
fi

# Exit with appropriate status code based on failures
if [ "$FAILED_COUNT" -gt 0 ]; then
    log_error "Benchmark matrix finished with ${FAILED_COUNT} failed or invalid experiment(s)."
    exit 1
fi

log_success "All benchmark runs completed successfully."
exit 0
