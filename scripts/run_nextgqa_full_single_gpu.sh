#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_nextgqa_full_single_gpu.sh \
    [--data-root /path/to/nextgqa] \
    [--video-root /path/to/nextgqa/videos] \
    [--run-root /path/to/output/run] \
    [--conda-env /path/to/conda-env-or-env-name]

Defaults are prefilled for the earlier Omkar single-GPU machine:
  --data-root   /home/omkar/ritesh/grounded_videoqa/data/nextgqa
  --video-root  /home/omkar/ritesh/grounded_videoqa/data/nextgqa/videos
  --run-root    /share/data/drive_1/omkar/grounded_videoqa_runs/nextgqa_full_seed13
  --conda-env   /home/omkar/ritesh/grounded_videoqa/conda-env
  --gpu         0
  --cache-root  /share/data/drive_1/omkar/model_cache

Optional:
  --gpu                    GPU id to use via CUDA_VISIBLE_DEVICES (default: 0)
  --seed                   Policy seed (default: 13)
  --cache-root             Root directory for HF/Torch caches
  --feature-batch-size     CLIP feature batch size (default: 32)
  --model-name             CLIP model name (default: openai/clip-vit-base-patch32)
  --subtitle-k             Retrieved subtitle items (default: 0)
  --frame-k                Retrieved frame items (default: 3)
  --segment-k              Retrieved segment items (default: 3)
  --oracle-mode            Oracle mode (default: correctness_plus_sufficiency)
  --oracle-min-sufficiency Minimum sufficiency threshold (default: 0.8)
  --max-items              Policy acquisition cap (default: 6)
  --min-items-before-stop  Minimum items before stop is allowed (default: 1)
EOF
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

DATA_ROOT="/home/omkar/ritesh/grounded_videoqa/data/nextgqa"
VIDEO_ROOT="/home/omkar/ritesh/grounded_videoqa/data/nextgqa/videos"
RUN_ROOT="/share/data/drive_1/omkar/grounded_videoqa_runs/nextgqa_full_seed13"
CONDA_ENV_SPEC="/home/omkar/ritesh/grounded_videoqa/conda-env"
GPU_ID="0"
SEED="13"
CACHE_ROOT="/share/data/drive_1/omkar/model_cache"
FEATURE_BATCH_SIZE="32"
MODEL_NAME="openai/clip-vit-base-patch32"
SUBTITLE_K="0"
FRAME_K="3"
SEGMENT_K="3"
ORACLE_MODE="correctness_plus_sufficiency"
ORACLE_MIN_SUFFICIENCY="0.8"
MAX_ITEMS="6"
MIN_ITEMS_BEFORE_STOP="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --video-root) VIDEO_ROOT="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --conda-env) CONDA_ENV_SPEC="$2"; shift 2 ;;
    --gpu) GPU_ID="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --cache-root) CACHE_ROOT="$2"; shift 2 ;;
    --feature-batch-size) FEATURE_BATCH_SIZE="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --subtitle-k) SUBTITLE_K="$2"; shift 2 ;;
    --frame-k) FRAME_K="$2"; shift 2 ;;
    --segment-k) SEGMENT_K="$2"; shift 2 ;;
    --oracle-mode) ORACLE_MODE="$2"; shift 2 ;;
    --oracle-min-sufficiency) ORACLE_MIN_SUFFICIENCY="$2"; shift 2 ;;
    --max-items) MAX_ITEMS="$2"; shift 2 ;;
    --min-items-before-stop) MIN_ITEMS_BEFORE_STOP="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

require_dir "$DATA_ROOT"
require_dir "$VIDEO_ROOT"
require_file "$DATA_ROOT/train.csv"
require_file "$DATA_ROOT/val.csv"
require_file "$DATA_ROOT/map_vid_vidorID.json"
require_file "$DATA_ROOT/gsub_val.json"
require_file "$DATA_ROOT/frame2time_val.json"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p "$RUN_ROOT"

if [[ -n "$CACHE_ROOT" ]]; then
  mkdir -p "$CACHE_ROOT/hf" "$CACHE_ROOT/torch"
  export HF_HOME="$CACHE_ROOT/hf"
  export TRANSFORMERS_CACHE="$CACHE_ROOT/hf"
  export TORCH_HOME="$CACHE_ROOT/torch"
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"

CONDA_ARGS=()
if [[ -d "$CONDA_ENV_SPEC" ]]; then
  CONDA_ARGS+=(--conda-prefix "$CONDA_ENV_SPEC")
else
  CONDA_ARGS+=(--conda-env "$CONDA_ENV_SPEC")
fi

bash "$REPO_ROOT/scripts/run_nextgqa_experiment.sh" \
  --data-root "$DATA_ROOT" \
  --video-root "$VIDEO_ROOT" \
  --run-root "$RUN_ROOT" \
  "${CONDA_ARGS[@]}" \
  --train-limit 999999 \
  --val-limit 999999 \
  --seed "$SEED" \
  --device cuda \
  --feature-batch-size "$FEATURE_BATCH_SIZE" \
  --model-name "$MODEL_NAME" \
  --subtitle-k "$SUBTITLE_K" \
  --frame-k "$FRAME_K" \
  --segment-k "$SEGMENT_K" \
  --oracle-mode "$ORACLE_MODE" \
  --oracle-min-sufficiency "$ORACLE_MIN_SUFFICIENCY" \
  --max-items "$MAX_ITEMS" \
  --min-items-before-stop "$MIN_ITEMS_BEFORE_STOP"
