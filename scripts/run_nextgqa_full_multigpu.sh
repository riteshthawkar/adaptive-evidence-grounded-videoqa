#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_nextgqa_full_multigpu.sh \
    --data-root /path/to/nextgqa \
    --video-root /path/to/nextgqa/videos \
    --run-root /path/to/output/run \
    --conda-env /path/to/conda-env-or-env-name \
    --gpus 0,1

Required:
  --data-root      Directory containing train.csv, val.csv, map_vid_vidorID.json,
                   gsub_val.json, and frame2time_val.json.
  --video-root     Directory containing NExT-GQA videos.
  --run-root       Output directory for this full-data run.
  --conda-env      Conda environment name or full prefix path.

Optional:
  --gpus                   Comma-separated GPU ids (default: 0,1)
  --cache-root             Root directory for HF/Torch caches
  --feature-batch-size     CLIP feature batch size (default: 32)
  --feature-model          CLIP model name (default: openai/clip-vit-base-patch32)
  --subtitle-k             Retrieved subtitle items (default: 0)
  --frame-k                Retrieved frame items (default: 3)
  --segment-k              Retrieved segment items (default: 3)
  --oracle-mode            Oracle mode (default: correctness_plus_sufficiency)
  --oracle-min-sufficiency Minimum sufficiency threshold (default: 0.8)
  --oracle-min-temporal-iou Minimum temporal IoU threshold (default: 0.0)
  --max-items              Policy acquisition cap (default: 6)
  --answerer-epochs        Linear answerer epochs (default: 20)
  --policy-epochs          Policy epochs (default: 20)
  --skip-model-relative    Skip model-relative study
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

DATA_ROOT=""
VIDEO_ROOT=""
RUN_ROOT=""
CONDA_ENV_SPEC=""
GPU_IDS="0,1"
CACHE_ROOT=""
FEATURE_BATCH_SIZE="32"
FEATURE_MODEL="openai/clip-vit-base-patch32"
SUBTITLE_K="0"
FRAME_K="3"
SEGMENT_K="3"
ORACLE_MODE="correctness_plus_sufficiency"
ORACLE_MIN_SUFFICIENCY="0.8"
ORACLE_MIN_TEMPORAL_IOU="0.0"
MAX_ITEMS="6"
ANSWERER_EPOCHS="20"
POLICY_EPOCHS="20"
SKIP_MODEL_RELATIVE="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --video-root) VIDEO_ROOT="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --conda-env) CONDA_ENV_SPEC="$2"; shift 2 ;;
    --gpus) GPU_IDS="$2"; shift 2 ;;
    --cache-root) CACHE_ROOT="$2"; shift 2 ;;
    --feature-batch-size) FEATURE_BATCH_SIZE="$2"; shift 2 ;;
    --feature-model) FEATURE_MODEL="$2"; shift 2 ;;
    --subtitle-k) SUBTITLE_K="$2"; shift 2 ;;
    --frame-k) FRAME_K="$2"; shift 2 ;;
    --segment-k) SEGMENT_K="$2"; shift 2 ;;
    --oracle-mode) ORACLE_MODE="$2"; shift 2 ;;
    --oracle-min-sufficiency) ORACLE_MIN_SUFFICIENCY="$2"; shift 2 ;;
    --oracle-min-temporal-iou) ORACLE_MIN_TEMPORAL_IOU="$2"; shift 2 ;;
    --max-items) MAX_ITEMS="$2"; shift 2 ;;
    --answerer-epochs) ANSWERER_EPOCHS="$2"; shift 2 ;;
    --policy-epochs) POLICY_EPOCHS="$2"; shift 2 ;;
    --skip-model-relative) SKIP_MODEL_RELATIVE="1"; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$DATA_ROOT" || -z "$VIDEO_ROOT" || -z "$RUN_ROOT" || -z "$CONDA_ENV_SPEC" ]]; then
  usage
  exit 1
fi

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

CMD=(
  bash "$REPO_ROOT/scripts/run_multigpu_pipeline.sh"
  --dataset nextgqa
  --train-qa "$DATA_ROOT/train.csv"
  --val-qa "$DATA_ROOT/val.csv"
  --video-root "$VIDEO_ROOT"
  --video-map "$DATA_ROOT/map_vid_vidorID.json"
  --val-gsub "$DATA_ROOT/gsub_val.json"
  --val-frame-times "$DATA_ROOT/frame2time_val.json"
  --run-root "$RUN_ROOT"
  --conda-env "$CONDA_ENV_SPEC"
  --gpus "$GPU_IDS"
  --feature-model "$FEATURE_MODEL"
  --feature-batch-size "$FEATURE_BATCH_SIZE"
  --subtitle-k "$SUBTITLE_K"
  --frame-k "$FRAME_K"
  --segment-k "$SEGMENT_K"
  --answerer-epochs "$ANSWERER_EPOCHS"
  --policy-epochs "$POLICY_EPOCHS"
  --oracle-mode "$ORACLE_MODE"
  --oracle-min-sufficiency "$ORACLE_MIN_SUFFICIENCY"
  --oracle-min-temporal-iou "$ORACLE_MIN_TEMPORAL_IOU"
  --max-items "$MAX_ITEMS"
)

if [[ "$SKIP_MODEL_RELATIVE" == "1" ]]; then
  CMD+=(--skip-model-relative)
fi

"${CMD[@]}"
