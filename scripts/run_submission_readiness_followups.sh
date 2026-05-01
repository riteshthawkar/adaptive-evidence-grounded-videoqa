#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ROOT="${RUN_ROOT:-runs/nextgqa_full_seed13}"
DATA_ROOT="${DATA_ROOT:-data/nextgqa_hf/datasets/nextgqa}"
VIDEO_ROOT="${VIDEO_ROOT:-data/nextgqa_hf/NExTVideo}"
MODEL_NAME="${MODEL_NAME:-openai/clip-vit-base-patch32}"
DEVICE="${DEVICE:-cuda}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-32}"
VISUAL_MATERIALIZE_WORKERS="${VISUAL_MATERIALIZE_WORKERS:-12}"
VISUAL_MATERIALIZE_CHUNKSIZE="${VISUAL_MATERIALIZE_CHUNKSIZE:-4}"
AEVQA_FFPROBE_WORKERS="${AEVQA_FFPROBE_WORKERS:-8}"

export PYTHONPATH="${PYTHONPATH:-src}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-data/model_cache/hf}"
export TORCH_HOME="${TORCH_HOME:-data/model_cache/torch}"
export PYTHONUNBUFFERED=1
export AEVQA_FFPROBE_WORKERS

FOLLOWUP_DIR="$RUN_ROOT/outputs/submission_followups"
mkdir -p \
  "$FOLLOWUP_DIR"/{logs,predictions,paper_metrics,oracle,models} \
  "$RUN_ROOT"/{normalized,candidates,artifacts/frames,artifacts/segments,features/clip}

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_logged() {
  local log_path="$1"
  shift
  log "Running: $*"
  "$@" >"$log_path" 2>&1
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

require_dir "$DATA_ROOT"
require_dir "$VIDEO_ROOT"
require_file "$DATA_ROOT/test.csv"
require_file "$DATA_ROOT/gsub_test.json"
require_file "$DATA_ROOT/frame2time_test.json"
require_file "$DATA_ROOT/map_vid_vidorID.json"
require_file "$RUN_ROOT/candidates/train.visual_features.jsonl"
require_file "$RUN_ROOT/candidates/val.visual_features.jsonl"
require_file "$RUN_ROOT/outputs/train_oracle_traces_frozen.jsonl"
require_file "$RUN_ROOT/outputs/val_oracle_traces_frozen.jsonl"
require_file "$RUN_ROOT/models/policy_hybrid_frozen/weights.npz"

log "Summarizing existing validation controls with paper-style NExT-GQA metrics"
python scripts/summarize_nextgqa_paper_metrics.py \
  --candidate-path "$RUN_ROOT/candidates/val.visual_features.jsonl" \
  --predictions \
    "$RUN_ROOT/outputs/fixed_budget_frozen.predictions.jsonl" \
    "$RUN_ROOT/outputs/sequential_policy_frozen.predictions.jsonl" \
    "$RUN_ROOT/outputs/research_controls/predictions/fixed_segment1.jsonl" \
    "$RUN_ROOT/outputs/research_controls/predictions/fixed_f6_s6.jsonl" \
  --labels fixed_f3_s3 learned_policy fixed_segment1 fixed_f6_s6 \
  --summary-json "$FOLLOWUP_DIR/paper_metrics/val_existing_controls.json" \
  --summary-md "$FOLLOWUP_DIR/paper_metrics/val_existing_controls.md"

if [[ ! -f "$RUN_ROOT/candidates/test.visual_features.jsonl" ]]; then
  log "Preparing NExT-GQA test split"
  run_logged \
    "$FOLLOWUP_DIR/logs/test.prepare.log" \
    python scripts/prepare_nextgqa.py \
      --qa-path "$DATA_ROOT/test.csv" \
      --gsub-path "$DATA_ROOT/gsub_test.json" \
      --frame-times-path "$DATA_ROOT/frame2time_test.json" \
      --video-map-path "$DATA_ROOT/map_vid_vidorID.json" \
      --video-root "$VIDEO_ROOT" \
      --output-path "$RUN_ROOT/normalized/test.jsonl"

  log "Building test candidate pool"
  run_logged \
    "$FOLLOWUP_DIR/logs/test.candidate_pool.log" \
    python scripts/build_candidate_pool.py \
      --input-path "$RUN_ROOT/normalized/test.jsonl" \
      --output-path "$RUN_ROOT/candidates/test.jsonl"

  log "Materializing test visual evidence"
  run_logged \
    "$FOLLOWUP_DIR/logs/test.materialize.log" \
    python scripts/materialize_visual_evidence_parallel.py \
      --input-path "$RUN_ROOT/candidates/test.jsonl" \
      --video-root "$VIDEO_ROOT" \
      --output-path "$RUN_ROOT/candidates/test.visual.jsonl" \
      --frames-dir "$RUN_ROOT/artifacts/frames" \
      --segments-dir "$RUN_ROOT/artifacts/segments" \
      --workers "$VISUAL_MATERIALIZE_WORKERS" \
      --chunksize "$VISUAL_MATERIALIZE_CHUNKSIZE" \
      --overwrite

  log "Extracting test CLIP features"
  run_logged \
    "$FOLLOWUP_DIR/logs/test.features.log" \
    python scripts/extract_clip_features.py \
      --input-path "$RUN_ROOT/candidates/test.visual.jsonl" \
      --output-path "$RUN_ROOT/candidates/test.visual_features.jsonl" \
      --feature-dir "$RUN_ROOT/features/clip" \
      --device "$DEVICE" \
      --batch-size "$FEATURE_BATCH_SIZE" \
      --model-name "$MODEL_NAME"
else
  log "Test visual features already exist; reusing $RUN_ROOT/candidates/test.visual_features.jsonl"
fi

log "Running fixed-budget and deterministic policy controls on test"
run_logged \
  "$FOLLOWUP_DIR/logs/test.controls.log" \
  python scripts/run_research_controls.py \
    --input-path "$RUN_ROOT/candidates/test.visual_features.jsonl" \
    --output-dir "$FOLLOWUP_DIR/test_controls" \
    --write-predictions \
    --answerer frozen_multimodal \
    --answerer-device "$DEVICE" \
    --answerer-model-name "$MODEL_NAME" \
    --retriever hybrid_clip \
    --visual-device "$DEVICE" \
    --visual-model-name "$MODEL_NAME" \
    --subtitle-k 0 \
    --frame-k 3 \
    --segment-k 3 \
    --fixed-budget-ks 1,2,3,4,5,6 \
    --max-items 6 \
    --oracle-mode correctness_plus_sufficiency \
    --oracle-min-sufficiency 0.8

log "Evaluating current learned policy on test"
run_logged \
  "$FOLLOWUP_DIR/logs/test.learned_policy.log" \
  python scripts/run_sequential_policy.py \
    --input-path "$RUN_ROOT/candidates/test.visual_features.jsonl" \
    --summary-output "$FOLLOWUP_DIR/test_learned_policy.summary.json" \
    --predictions-output "$FOLLOWUP_DIR/predictions/test_learned_policy.jsonl" \
    --answerer frozen_multimodal \
    --answerer-device "$DEVICE" \
    --answerer-model-name "$MODEL_NAME" \
    --policy linear \
    --policy-model-dir "$RUN_ROOT/models/policy_hybrid_frozen" \
    --retriever hybrid_clip \
    --visual-device "$DEVICE" \
    --visual-model-name "$MODEL_NAME" \
    --subtitle-k 0 \
    --frame-k 3 \
    --segment-k 3 \
    --max-items 6 \
    --min-items-before-stop 1

log "Training stricter min-2-stop policy from existing oracle traces"
run_logged \
  "$FOLLOWUP_DIR/logs/policy_min2.train.log" \
  python scripts/train_policy.py \
    --train-traces-path "$RUN_ROOT/outputs/train_oracle_traces_frozen.jsonl" \
    --validation-traces-path "$RUN_ROOT/outputs/val_oracle_traces_frozen.jsonl" \
    --model-dir "$FOLLOWUP_DIR/models/policy_hybrid_frozen_min2" \
    --answerer frozen_multimodal \
    --answerer-device "$DEVICE" \
    --answerer-model-name "$MODEL_NAME" \
    --epochs 10 \
    --seed 13 \
    --min-items-before-stop 2

for split in val test; do
  log "Evaluating min-2-stop policy on ${split}"
  run_logged \
    "$FOLLOWUP_DIR/logs/${split}.policy_min2.eval.log" \
    python scripts/run_sequential_policy.py \
      --input-path "$RUN_ROOT/candidates/${split}.visual_features.jsonl" \
      --summary-output "$FOLLOWUP_DIR/${split}_policy_min2.summary.json" \
      --predictions-output "$FOLLOWUP_DIR/predictions/${split}_policy_min2.jsonl" \
      --answerer frozen_multimodal \
      --answerer-device "$DEVICE" \
      --answerer-model-name "$MODEL_NAME" \
      --policy linear \
      --policy-model-dir "$FOLLOWUP_DIR/models/policy_hybrid_frozen_min2" \
      --retriever hybrid_clip \
      --visual-device "$DEVICE" \
      --visual-model-name "$MODEL_NAME" \
      --subtitle-k 0 \
      --frame-k 3 \
      --segment-k 3 \
      --max-items 6 \
      --min-items-before-stop 2
done

log "Exporting grounding-aware oracle diagnostics for val/test"
for split in val test; do
  run_logged \
    "$FOLLOWUP_DIR/logs/${split}.oracle_grounded_iou01.log" \
    python scripts/export_oracle_traces.py \
      --input-path "$RUN_ROOT/candidates/${split}.visual_features.jsonl" \
      --output-path "$FOLLOWUP_DIR/oracle/${split}_oracle_grounded_iou01.jsonl" \
      --answerer frozen_multimodal \
      --answerer-device "$DEVICE" \
      --answerer-model-name "$MODEL_NAME" \
      --retriever hybrid_clip \
      --visual-device "$DEVICE" \
      --visual-model-name "$MODEL_NAME" \
      --subtitle-k 0 \
      --frame-k 3 \
      --segment-k 3 \
      --oracle-mode correctness_plus_sufficiency_plus_grounding \
      --oracle-min-sufficiency 0.8 \
      --oracle-min-temporal-iou 0.1
done

log "Summarizing paper-style metrics for val and test follow-ups"
python scripts/summarize_nextgqa_paper_metrics.py \
  --candidate-path "$RUN_ROOT/candidates/val.visual_features.jsonl" \
  --predictions \
    "$RUN_ROOT/outputs/fixed_budget_frozen.predictions.jsonl" \
    "$RUN_ROOT/outputs/sequential_policy_frozen.predictions.jsonl" \
    "$FOLLOWUP_DIR/predictions/val_policy_min2.jsonl" \
  --labels fixed_f3_s3 learned_policy_min1 learned_policy_min2 \
  --summary-json "$FOLLOWUP_DIR/paper_metrics/val_policy_comparison.json" \
  --summary-md "$FOLLOWUP_DIR/paper_metrics/val_policy_comparison.md"

python scripts/summarize_nextgqa_paper_metrics.py \
  --candidate-path "$RUN_ROOT/candidates/test.visual_features.jsonl" \
  --predictions \
    "$FOLLOWUP_DIR/test_controls/predictions/fixed_main.jsonl" \
    "$FOLLOWUP_DIR/test_controls/predictions/fixed_segment1.jsonl" \
    "$FOLLOWUP_DIR/test_controls/predictions/fixed_f6_s6.jsonl" \
    "$FOLLOWUP_DIR/predictions/test_learned_policy.jsonl" \
    "$FOLLOWUP_DIR/predictions/test_policy_min2.jsonl" \
  --labels fixed_f3_s3 fixed_segment1 fixed_f6_s6 learned_policy_min1 learned_policy_min2 \
  --summary-json "$FOLLOWUP_DIR/paper_metrics/test_policy_comparison.json" \
  --summary-md "$FOLLOWUP_DIR/paper_metrics/test_policy_comparison.md"

log "Submission follow-up run complete"
cat "$FOLLOWUP_DIR/paper_metrics/val_policy_comparison.md"
cat "$FOLLOWUP_DIR/paper_metrics/test_policy_comparison.md"
