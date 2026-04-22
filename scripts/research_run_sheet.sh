#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/research_run_sheet.sh [options]

Purpose:
  Run the current research-grade NExT-GQA experiment blocks on top of an
  already materialized full-data cache run.

Default Omkar-machine paths:
  --cache-run      /share/data/drive_1/omkar/grounded_videoqa_runs/nextgqa_full_multigpu_seed13
  --research-root  /share/data/drive_1/omkar/grounded_videoqa_runs/research_nextgqa_full
  --conda-env      /home/omkar/ritesh/grounded_videoqa/conda-env
  --gpu            0
  --model-name     openai/clip-vit-base-patch32

Options:
  --cache-run PATH           Full-data cache run directory to reuse.
  --research-root PATH       Root directory for new research experiments.
  --conda-env SPEC          Conda env name or full prefix path.
  --gpu ID                  GPU id to expose via CUDA_VISIBLE_DEVICES.
  --model-name NAME         Frozen multimodal model name.
  --main-seeds CSV          Seeds for learned-policy main runs (default: 13,21,34).
  --policy-epochs N         Policy epochs (default: 20).
  --fixed-max-k N           Largest fixed budget K for frame=K, segment=K sweep (default: 4).
  --oracle-mode MODE        Oracle mode (default: correctness_plus_sufficiency).
  --oracle-min-sufficiency X  Oracle minimum sufficiency (default: 0.8).
  --oracle-min-temporal-iou X Oracle minimum temporal IoU (default: 0.0).
  --max-items N             Sequential acquisition cap (default: 6).
  --min-items-before-stop N Minimum items before stop (default: 1).

Block controls:
  --skip-main               Skip 3-seed learned-policy block.
  --skip-budget-sweep       Skip fixed-budget sweep.
  --skip-ablations          Skip frame-only and segment-only ablations.
  --skip-model-relative     Skip model-relative analysis.
  --aggregate-only          Only run aggregation over existing outputs.

Notes:
  - This script assumes the cache run already contains:
      candidates/train.visual_features.jsonl
      candidates/val.visual_features.jsonl
      outputs/train_oracle_traces_frozen.jsonl
      outputs/val_oracle_traces_frozen.jsonl
  - Model-relative analysis also expects:
      models/answerer_hybrid_linear/
  - The current codebase does not yet include random/frame-first/segment-first
    sequential baselines, so this run sheet focuses on the implemented blocks.
EOF
}

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Required file not found: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "Required directory not found: $path" >&2
    exit 1
  fi
}

activate_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not available on PATH." >&2
    exit 1
  fi
  eval "$(conda shell.bash hook)"
  if [[ -d "$CONDA_ENV_SPEC" ]]; then
    conda activate "$CONDA_ENV_SPEC"
  else
    conda activate "$CONDA_ENV_SPEC"
  fi
}

run_logged() {
  local log_path="$1"
  shift
  log "Running: $*"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$@" >"$log_path" 2>&1
}

run_main_seed() {
  local seed="$1"
  local run="$RESEARCH_ROOT/main/policy_frozen_seed${seed}"
  mkdir -p "$run"/{models,outputs,logs}

  run_logged \
    "$run/logs/policy.train.log" \
    python scripts/train_policy.py \
    --train-traces-path "$CACHE_RUN/outputs/train_oracle_traces_frozen.jsonl" \
    --validation-traces-path "$CACHE_RUN/outputs/val_oracle_traces_frozen.jsonl" \
    --model-dir "$run/models/policy_hybrid_frozen" \
    --answerer frozen_multimodal \
    --answerer-device cuda \
    --answerer-model-name "$MODEL_NAME" \
    --epochs "$POLICY_EPOCHS" \
    --seed "$seed" \
    --min-items-before-stop "$MIN_ITEMS_BEFORE_STOP"

  run_logged \
    "$run/logs/policy.eval.log" \
    python scripts/run_sequential_policy.py \
    --input-path "$CACHE_RUN/candidates/val.visual_features.jsonl" \
    --summary-output "$run/outputs/sequential_policy_frozen.summary.json" \
    --predictions-output "$run/outputs/sequential_policy_frozen.predictions.jsonl" \
    --answerer frozen_multimodal \
    --answerer-device cuda \
    --answerer-model-name "$MODEL_NAME" \
    --policy linear \
    --policy-model-dir "$run/models/policy_hybrid_frozen" \
    --retriever hybrid_clip \
    --visual-device cuda \
    --visual-model-name "$MODEL_NAME" \
    --subtitle-k 0 \
    --frame-k 3 \
    --segment-k 3 \
    --max-items "$MAX_ITEMS" \
    --min-items-before-stop "$MIN_ITEMS_BEFORE_STOP"
}

run_budget_sweep() {
  local k
  for ((k=1; k<=FIXED_MAX_K; k++)); do
    local run="$RESEARCH_ROOT/budget_sweep/f${k}_s${k}"
    mkdir -p "$run"/{outputs,logs}

    run_logged \
      "$run/logs/fixed_budget.log" \
      python scripts/run_fixed_budget_baseline.py \
      --input-path "$CACHE_RUN/candidates/val.visual_features.jsonl" \
      --summary-output "$run/outputs/fixed_budget_frozen.summary.json" \
      --predictions-output "$run/outputs/fixed_budget_frozen.predictions.jsonl" \
      --answerer frozen_multimodal \
      --answerer-device cuda \
      --answerer-model-name "$MODEL_NAME" \
      --retriever hybrid_clip \
      --visual-device cuda \
      --visual-model-name "$MODEL_NAME" \
      --subtitle-k 0 \
      --frame-k "$k" \
      --segment-k "$k" \
      --oracle-mode "$ORACLE_MODE" \
      --oracle-min-sufficiency "$ORACLE_MIN_SUFFICIENCY" \
      --oracle-min-temporal-iou "$ORACLE_MIN_TEMPORAL_IOU"
  done
}

run_ablation() {
  local label="$1"
  local frame_k="$2"
  local segment_k="$3"
  local run="$RESEARCH_ROOT/ablations/$label"
  mkdir -p "$run"/{models,outputs,logs}

  run_logged \
    "$run/logs/train.oracle.log" \
    python scripts/export_oracle_traces.py \
    --input-path "$CACHE_RUN/candidates/train.visual_features.jsonl" \
    --output-path "$run/outputs/train_oracle_traces_frozen.jsonl" \
    --answerer frozen_multimodal \
    --answerer-device cuda \
    --answerer-model-name "$MODEL_NAME" \
    --retriever hybrid_clip \
    --visual-device cuda \
    --visual-model-name "$MODEL_NAME" \
    --subtitle-k 0 \
    --frame-k "$frame_k" \
    --segment-k "$segment_k" \
    --oracle-mode "$ORACLE_MODE" \
    --oracle-min-sufficiency "$ORACLE_MIN_SUFFICIENCY" \
    --oracle-min-temporal-iou "$ORACLE_MIN_TEMPORAL_IOU"

  run_logged \
    "$run/logs/val.oracle.log" \
    python scripts/export_oracle_traces.py \
    --input-path "$CACHE_RUN/candidates/val.visual_features.jsonl" \
    --output-path "$run/outputs/val_oracle_traces_frozen.jsonl" \
    --answerer frozen_multimodal \
    --answerer-device cuda \
    --answerer-model-name "$MODEL_NAME" \
    --retriever hybrid_clip \
    --visual-device cuda \
    --visual-model-name "$MODEL_NAME" \
    --subtitle-k 0 \
    --frame-k "$frame_k" \
    --segment-k "$segment_k" \
    --oracle-mode "$ORACLE_MODE" \
    --oracle-min-sufficiency "$ORACLE_MIN_SUFFICIENCY" \
    --oracle-min-temporal-iou "$ORACLE_MIN_TEMPORAL_IOU"

  run_logged \
    "$run/logs/policy.train.log" \
    python scripts/train_policy.py \
    --train-traces-path "$run/outputs/train_oracle_traces_frozen.jsonl" \
    --validation-traces-path "$run/outputs/val_oracle_traces_frozen.jsonl" \
    --model-dir "$run/models/policy_hybrid_frozen" \
    --answerer frozen_multimodal \
    --answerer-device cuda \
    --answerer-model-name "$MODEL_NAME" \
    --epochs "$POLICY_EPOCHS" \
    --seed "${ABLATION_SEED}" \
    --min-items-before-stop "$MIN_ITEMS_BEFORE_STOP"

  run_logged \
    "$run/logs/fixed_budget.log" \
    python scripts/run_fixed_budget_baseline.py \
    --input-path "$CACHE_RUN/candidates/val.visual_features.jsonl" \
    --summary-output "$run/outputs/fixed_budget_frozen.summary.json" \
    --predictions-output "$run/outputs/fixed_budget_frozen.predictions.jsonl" \
    --answerer frozen_multimodal \
    --answerer-device cuda \
    --answerer-model-name "$MODEL_NAME" \
    --retriever hybrid_clip \
    --visual-device cuda \
    --visual-model-name "$MODEL_NAME" \
    --subtitle-k 0 \
    --frame-k "$frame_k" \
    --segment-k "$segment_k" \
    --oracle-mode "$ORACLE_MODE" \
    --oracle-min-sufficiency "$ORACLE_MIN_SUFFICIENCY" \
    --oracle-min-temporal-iou "$ORACLE_MIN_TEMPORAL_IOU"

  run_logged \
    "$run/logs/policy.eval.log" \
    python scripts/run_sequential_policy.py \
    --input-path "$CACHE_RUN/candidates/val.visual_features.jsonl" \
    --summary-output "$run/outputs/sequential_policy_frozen.summary.json" \
    --predictions-output "$run/outputs/sequential_policy_frozen.predictions.jsonl" \
    --answerer frozen_multimodal \
    --answerer-device cuda \
    --answerer-model-name "$MODEL_NAME" \
    --policy linear \
    --policy-model-dir "$run/models/policy_hybrid_frozen" \
    --retriever hybrid_clip \
    --visual-device cuda \
    --visual-model-name "$MODEL_NAME" \
    --subtitle-k 0 \
    --frame-k "$frame_k" \
    --segment-k "$segment_k" \
    --max-items "$MAX_ITEMS" \
    --min-items-before-stop "$MIN_ITEMS_BEFORE_STOP"
}

run_model_relative() {
  local run="$RESEARCH_ROOT/model_relative/linear_vs_frozen"
  mkdir -p "$run"/{outputs,logs}

  if [[ ! -d "$CACHE_RUN/models/answerer_hybrid_linear" ]]; then
    log "Skipping model-relative analysis because $CACHE_RUN/models/answerer_hybrid_linear is missing."
    return
  fi

  run_logged \
    "$run/logs/model_relative.log" \
    python scripts/run_model_relative_study.py \
    --input-path "$CACHE_RUN/candidates/val.visual_features.jsonl" \
    --output-dir "$run/outputs" \
    --answerer-a linear \
    --answerer-a-model-dir "$CACHE_RUN/models/answerer_hybrid_linear" \
    --answerer-a-label linear \
    --answerer-b frozen_multimodal \
    --answerer-b-model-name "$MODEL_NAME" \
    --answerer-b-device cuda \
    --answerer-b-label frozen \
    --retriever hybrid_clip \
    --visual-device cuda \
    --visual-model-name "$MODEL_NAME" \
    --subtitle-k 0 \
    --frame-k 3 \
    --segment-k 3 \
    --oracle-mode "$ORACLE_MODE" \
    --oracle-min-sufficiency "$ORACLE_MIN_SUFFICIENCY" \
    --oracle-min-temporal-iou "$ORACLE_MIN_TEMPORAL_IOU"
}

run_aggregate() {
  local learned_roots=()
  local seed
  for seed in "${MAIN_SEEDS[@]}"; do
    learned_roots+=("$RESEARCH_ROOT/main/policy_frozen_seed${seed}")
  done

  python scripts/aggregate_run_summaries.py \
    --run-roots "${learned_roots[@]}" \
    --summary-files sequential_policy_frozen.summary.json \
    --output-json "$RESEARCH_ROOT/aggregate/learned_policy_3seed.json" \
    --output-markdown "$RESEARCH_ROOT/aggregate/learned_policy_3seed.md"
}

CACHE_RUN="/share/data/drive_1/omkar/grounded_videoqa_runs/nextgqa_full_multigpu_seed13"
RESEARCH_ROOT="/share/data/drive_1/omkar/grounded_videoqa_runs/research_nextgqa_full"
CONDA_ENV_SPEC="/home/omkar/ritesh/grounded_videoqa/conda-env"
GPU_ID="0"
MODEL_NAME="openai/clip-vit-base-patch32"
MAIN_SEEDS_CSV="13,21,34"
POLICY_EPOCHS="20"
FIXED_MAX_K="4"
ORACLE_MODE="correctness_plus_sufficiency"
ORACLE_MIN_SUFFICIENCY="0.8"
ORACLE_MIN_TEMPORAL_IOU="0.0"
MAX_ITEMS="6"
MIN_ITEMS_BEFORE_STOP="1"
RUN_MAIN="1"
RUN_BUDGET_SWEEP="1"
RUN_ABLATIONS="1"
RUN_MODEL_RELATIVE="1"
AGGREGATE_ONLY="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cache-run) CACHE_RUN="$2"; shift 2 ;;
    --research-root) RESEARCH_ROOT="$2"; shift 2 ;;
    --conda-env) CONDA_ENV_SPEC="$2"; shift 2 ;;
    --gpu) GPU_ID="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --main-seeds) MAIN_SEEDS_CSV="$2"; shift 2 ;;
    --policy-epochs) POLICY_EPOCHS="$2"; shift 2 ;;
    --fixed-max-k) FIXED_MAX_K="$2"; shift 2 ;;
    --oracle-mode) ORACLE_MODE="$2"; shift 2 ;;
    --oracle-min-sufficiency) ORACLE_MIN_SUFFICIENCY="$2"; shift 2 ;;
    --oracle-min-temporal-iou) ORACLE_MIN_TEMPORAL_IOU="$2"; shift 2 ;;
    --max-items) MAX_ITEMS="$2"; shift 2 ;;
    --min-items-before-stop) MIN_ITEMS_BEFORE_STOP="$2"; shift 2 ;;
    --skip-main) RUN_MAIN="0"; shift 1 ;;
    --skip-budget-sweep) RUN_BUDGET_SWEEP="0"; shift 1 ;;
    --skip-ablations) RUN_ABLATIONS="0"; shift 1 ;;
    --skip-model-relative) RUN_MODEL_RELATIVE="0"; shift 1 ;;
    --aggregate-only)
      AGGREGATE_ONLY="1"
      RUN_MAIN="0"
      RUN_BUDGET_SWEEP="0"
      RUN_ABLATIONS="0"
      RUN_MODEL_RELATIVE="0"
      shift 1
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

require_dir "$CACHE_RUN"
require_file "$CACHE_RUN/candidates/train.visual_features.jsonl"
require_file "$CACHE_RUN/candidates/val.visual_features.jsonl"
require_file "$CACHE_RUN/outputs/train_oracle_traces_frozen.jsonl"
require_file "$CACHE_RUN/outputs/val_oracle_traces_frozen.jsonl"

activate_conda

export PYTHONUNBUFFERED=1

mkdir -p "$RESEARCH_ROOT"/{main,budget_sweep,ablations,model_relative,aggregate,qualitative}

IFS=',' read -r -a MAIN_SEEDS <<< "$MAIN_SEEDS_CSV"
if [[ "${#MAIN_SEEDS[@]}" -eq 0 ]]; then
  echo "At least one main seed is required." >&2
  exit 1
fi
ABLATION_SEED="${MAIN_SEEDS[0]}"

log "Cache run: $CACHE_RUN"
log "Research root: $RESEARCH_ROOT"
log "GPU: $GPU_ID"
log "Main seeds: ${MAIN_SEEDS[*]}"

if [[ "$AGGREGATE_ONLY" == "0" && "$RUN_MAIN" == "1" ]]; then
  log "Phase 1: main learned-policy runs"
  for seed in "${MAIN_SEEDS[@]}"; do
    run_main_seed "$seed"
  done
fi

if [[ "$AGGREGATE_ONLY" == "0" && "$RUN_BUDGET_SWEEP" == "1" ]]; then
  log "Phase 2: fixed-budget sweep"
  run_budget_sweep
fi

if [[ "$AGGREGATE_ONLY" == "0" && "$RUN_ABLATIONS" == "1" ]]; then
  log "Phase 3: evidence-type ablations"
  run_ablation "frame_only" "3" "0"
  run_ablation "segment_only" "0" "3"
fi

if [[ "$AGGREGATE_ONLY" == "0" && "$RUN_MODEL_RELATIVE" == "1" ]]; then
  log "Phase 4: model-relative analysis"
  run_model_relative
fi

log "Phase 5: aggregation"
run_aggregate

log "Research run sheet complete."
printf '  %s\n' \
  "$RESEARCH_ROOT/aggregate/learned_policy_3seed.json" \
  "$RESEARCH_ROOT/aggregate/learned_policy_3seed.md"
