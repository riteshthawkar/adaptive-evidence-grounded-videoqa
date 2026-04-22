#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_research_single_gpu.sh [options]

Purpose:
  Launch the research experiment blocks on a single-GPU machine after the
  full-data cache run has been created by scripts/run_nextgqa_full_single_gpu.sh.

Default Omkar-machine paths:
  --cache-run      /share/data/drive_1/omkar/grounded_videoqa_runs/nextgqa_full_seed13
  --research-root  /share/data/drive_1/omkar/grounded_videoqa_runs/research_nextgqa_full_single_gpu
  --conda-env      /home/omkar/ritesh/grounded_videoqa/conda-env
  --gpu            0

Notes:
  - If the cache run does not exist yet, run:
      bash scripts/run_nextgqa_full_single_gpu.sh
    and wait for it to finish first.
  - This wrapper delegates to scripts/research_run_sheet.sh and supports the
    same phase-control flags such as:
      --skip-main
      --skip-budget-sweep
      --skip-ablations
      --skip-model-relative
      --aggregate-only
EOF
}

CACHE_RUN="/share/data/drive_1/omkar/grounded_videoqa_runs/nextgqa_full_seed13"
RESEARCH_ROOT="/share/data/drive_1/omkar/grounded_videoqa_runs/research_nextgqa_full_single_gpu"
CONDA_ENV_SPEC="/home/omkar/ritesh/grounded_videoqa/conda-env"
GPU_ID="0"

PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cache-run) CACHE_RUN="$2"; shift 2 ;;
    --research-root) RESEARCH_ROOT="$2"; shift 2 ;;
    --conda-env) CONDA_ENV_SPEC="$2"; shift 2 ;;
    --gpu) GPU_ID="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      PASSTHROUGH+=("$1")
      shift 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ ! -f "$CACHE_RUN/candidates/train.visual_features.jsonl" || ! -f "$CACHE_RUN/candidates/val.visual_features.jsonl" ]]; then
  echo "Cache run is incomplete or missing at: $CACHE_RUN" >&2
  echo "Run 'bash scripts/run_nextgqa_full_single_gpu.sh' first, or override --cache-run." >&2
  exit 1
fi

bash "$REPO_ROOT/scripts/research_run_sheet.sh" \
  --cache-run "$CACHE_RUN" \
  --research-root "$RESEARCH_ROOT" \
  --conda-env "$CONDA_ENV_SPEC" \
  --gpu "$GPU_ID" \
  "${PASSTHROUGH[@]}"
