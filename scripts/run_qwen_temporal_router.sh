#!/usr/bin/env bash
set -euo pipefail

SPLIT="${1:-all}"
ROOT="runs/nextgqa_full_seed13"
SEL_ROOT="${ROOT}/outputs/followup_strengthening/temporal_router/selections"
OUT_ROOT="${ROOT}/outputs/followup_strengthening/temporal_router/qwen"
BASE_QWEN_ROOT="${ROOT}/outputs/vlm_qwen25vl_3b_full"
FOLLOW_QWEN_ROOT="${ROOT}/outputs/followup_strengthening/qwen"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-3B-Instruct}"

export PYTHONPATH="${PYTHONPATH:-src:.}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUT_ROOT}/predictions" "${OUT_ROOT}/summaries" "${OUT_ROOT}/metrics" "${OUT_ROOT}/logs" "${OUT_ROOT}/frame_cache"

candidate_path_for_split() {
  local split="$1"
  echo "${ROOT}/candidates/${split}.visual_features.jsonl"
}

run_vlm() {
  local split="$1"
  local label="temporal_router_top2_nms0p0"
  local candidate_path
  candidate_path="$(candidate_path_for_split "${split}")"

  python scripts/run_vlm_answerer.py \
    --candidate-path "${candidate_path}" \
    --selection-path "${SEL_ROOT}/${split}_${label}.jsonl" \
    --predictions-output "${OUT_ROOT}/predictions/${split}_${label}.jsonl" \
    --summary-output "${OUT_ROOT}/summaries/${split}_${label}.summary.json" \
    --image-cache-dir "${OUT_ROOT}/frame_cache" \
    --method-label "${label}" \
    --model-name "${MODEL_NAME}" \
    --segment-frames 1 \
    --max-images 6 \
    --max-pixels 65536 \
    --batch-size "${BATCH_SIZE_TEMPORAL:-8}" \
    --progress-every 100 \
    --resume
}

summarize_split() {
  local split="$1"
  local candidate_path
  candidate_path="$(candidate_path_for_split "${split}")"

  python scripts/summarize_nextgqa_paper_metrics.py \
    --candidate-path "${candidate_path}" \
    --predictions \
      "${BASE_QWEN_ROOT}/predictions/${split}_learned_policy_min2.jsonl" \
      "${BASE_QWEN_ROOT}/predictions/${split}_router_mlp_top2.jsonl" \
      "${FOLLOW_QWEN_ROOT}/predictions/${split}_router_mlp_top2_nms0p0.jsonl" \
      "${OUT_ROOT}/predictions/${split}_temporal_router_top2_nms0p0.jsonl" \
      "${FOLLOW_QWEN_ROOT}/predictions/${split}_oracle_iop_top2.jsonl" \
      "${BASE_QWEN_ROOT}/predictions/${split}_fixed_f3_s3.jsonl" \
    --labels \
      qwen_linear_min2 \
      qwen_router_mlp_top2 \
      qwen_router_mlp_top2_nms0 \
      qwen_temporal_router_top2_nms0 \
      qwen_oracle_iop_top2 \
      qwen_fixed_f3_s3 \
    --summary-json "${OUT_ROOT}/metrics/${split}_temporal_router_compare.json" \
    --summary-md "${OUT_ROOT}/metrics/${split}_temporal_router_compare.md" \
    --bootstrap-samples "${BOOTSTRAP_SAMPLES:-1000}" \
    --seed "${BOOTSTRAP_SEED:-13}"
}

run_split() {
  local split="$1"
  run_vlm "${split}"
  summarize_split "${split}"
}

case "${SPLIT}" in
  val)
    run_split val
    ;;
  test)
    run_split test
    ;;
  all)
    run_split val
    run_split test
    ;;
  *)
    echo "Usage: $0 [val|test|all]" >&2
    exit 2
    ;;
esac
