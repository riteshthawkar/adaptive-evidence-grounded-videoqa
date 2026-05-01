#!/usr/bin/env bash
set -euo pipefail

SPLIT="${1:-val}"
ROOT="runs/nextgqa_full_seed13"
OUT_ROOT="${ROOT}/outputs/vlm_qwen25vl_3b_full"
SMOKE_ROOT="${ROOT}/outputs/vlm_qwen25vl_3b_smoke"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-3B-Instruct}"

export PYTHONPATH="${PYTHONPATH:-src:.}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUT_ROOT}/predictions" "${OUT_ROOT}/summaries" "${OUT_ROOT}/paper_metrics" "${OUT_ROOT}/logs"

seed_prediction() {
  local source_path="$1"
  local output_path="$2"
  if [[ ! -s "${output_path}" && -s "${source_path}" ]]; then
    cp "${source_path}" "${output_path}"
  fi
}

run_vlm() {
  local split="$1"
  local label="$2"
  local candidate_path="$3"
  local selection_path="$4"
  local batch_size="$5"
  local output_path="${OUT_ROOT}/predictions/${split}_${label}.jsonl"
  local summary_path="${OUT_ROOT}/summaries/${split}_${label}.summary.json"

  python scripts/run_vlm_answerer.py \
    --candidate-path "${candidate_path}" \
    --selection-path "${selection_path}" \
    --predictions-output "${output_path}" \
    --summary-output "${summary_path}" \
    --image-cache-dir "${OUT_ROOT}/frame_cache" \
    --method-label "${label}" \
    --model-name "${MODEL_NAME}" \
    --segment-frames 1 \
    --max-images 6 \
    --max-pixels 65536 \
    --batch-size "${batch_size}" \
    --progress-every 100 \
    --resume
}

summarize_split() {
  local split="$1"
  local candidate_path="$2"
  local fixed_segment1_path="$3"
  local policy_min2_path="$4"
  local fixed_f3_s3_path="$5"
  local bootstrap_samples="${BOOTSTRAP_SAMPLES:-1000}"

  python scripts/summarize_nextgqa_paper_metrics.py \
    --candidate-path "${candidate_path}" \
    --predictions \
      "${fixed_segment1_path}" \
      "${OUT_ROOT}/predictions/${split}_fixed_segment1.jsonl" \
      "${policy_min2_path}" \
      "${OUT_ROOT}/predictions/${split}_learned_policy_min2.jsonl" \
      "${fixed_f3_s3_path}" \
      "${OUT_ROOT}/predictions/${split}_fixed_f3_s3.jsonl" \
    --labels \
      clip_fixed_segment1 \
      qwen25vl_fixed_segment1 \
      clip_policy_min2 \
      qwen25vl_policy_min2 \
      clip_fixed_f3_s3 \
      qwen25vl_fixed_f3_s3 \
    --summary-json "${OUT_ROOT}/paper_metrics/${split}_qwen25vl_controls.json" \
    --summary-md "${OUT_ROOT}/paper_metrics/${split}_qwen25vl_controls.md" \
    --bootstrap-samples "${bootstrap_samples}" \
    --seed "${BOOTSTRAP_SEED:-13}"
}

run_split() {
  local split="$1"
  local candidate_path
  local fixed_segment1_path
  local policy_min2_path
  local fixed_f3_s3_path

  if [[ "${split}" == "val" ]]; then
    candidate_path="${ROOT}/candidates/val.visual_features.jsonl"
    fixed_segment1_path="${ROOT}/outputs/research_controls/predictions/fixed_segment1.jsonl"
    policy_min2_path="${ROOT}/outputs/submission_followups/predictions/val_policy_min2.jsonl"
    fixed_f3_s3_path="${ROOT}/outputs/research_controls/predictions/fixed_f3_s3.jsonl"

    seed_prediction \
      "${SMOKE_ROOT}/predictions/val_fixed_segment1_limit100.jsonl" \
      "${OUT_ROOT}/predictions/val_fixed_segment1.jsonl"
    seed_prediction \
      "${SMOKE_ROOT}/predictions/val_policy_min2_limit100.jsonl" \
      "${OUT_ROOT}/predictions/val_learned_policy_min2.jsonl"
    seed_prediction \
      "${SMOKE_ROOT}/predictions/val_fixed_f3_s3_limit100.jsonl" \
      "${OUT_ROOT}/predictions/val_fixed_f3_s3.jsonl"
  elif [[ "${split}" == "test" ]]; then
    candidate_path="${ROOT}/candidates/test.visual_features.jsonl"
    fixed_segment1_path="${ROOT}/outputs/submission_followups/test_controls/predictions/fixed_segment1.jsonl"
    policy_min2_path="${ROOT}/outputs/submission_followups/predictions/test_policy_min2.jsonl"
    fixed_f3_s3_path="${ROOT}/outputs/submission_followups/test_controls/predictions/fixed_f3_s3.jsonl"
  else
    echo "Unknown split: ${split}" >&2
    exit 2
  fi

  run_vlm "${split}" "fixed_segment1" "${candidate_path}" "${fixed_segment1_path}" "${BATCH_SIZE_SEGMENT1:-8}"
  run_vlm "${split}" "learned_policy_min2" "${candidate_path}" "${policy_min2_path}" "${BATCH_SIZE_MIN2:-4}"
  run_vlm "${split}" "fixed_f3_s3" "${candidate_path}" "${fixed_f3_s3_path}" "${BATCH_SIZE_F3S3:-2}"
  summarize_split "${split}" "${candidate_path}" "${fixed_segment1_path}" "${policy_min2_path}" "${fixed_f3_s3_path}"
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
