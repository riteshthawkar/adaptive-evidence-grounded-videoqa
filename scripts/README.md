# Scripts

This directory contains command-line entry points for preprocessing, evidence construction, model training, routing, evaluation, and analysis.

## Data Preparation

- `prepare_tvqa.py`: normalize TVQA annotations and subtitles into the shared JSONL schema.
- `prepare_tvqa_plus.py`: normalize TVQA+ annotations and subtitles.
- `prepare_nextgqa.py`: normalize NExT-GQA CSV annotations, temporal grounding files, frame-time files, and video-id maps.
- `build_candidate_pool.py`: build subtitle, frame, and segment candidates from normalized examples.
- `materialize_visual_evidence.py`: extract frame artifacts and optional segment clips with `ffmpeg`.
- `materialize_visual_evidence_parallel.py`: parallel visual materialization for large video collections.
- `extract_clip_features.py`: extract CLIP frame features and aggregate segment features.

## Baselines And Policies

- `run_fixed_budget_baseline.py`: evaluate fixed frame/segment allocations.
- `train_answerer.py`: train the linear evidence-conditioned multiple-choice answerer.
- `train_calibrated_multimodal_answerer.py`: train the calibrated frozen-feature answerer.
- `export_oracle_traces.py`: export oracle acquisition traces for imitation learning.
- `train_policy.py`: train the sequential acquisition policy.
- `run_sequential_policy.py`: evaluate learned or heuristic sequential policies.
- `train_evidence_router.py`: train the MLP evidence router.
- `train_temporal_router_from_pool.py`: train the temporal-supervised router.

## Qwen Evaluation

- `run_vlm_answerer.py`: run Qwen2.5-VL or another supported VLM over selected evidence.
- `create_followup_evidence_selections.py`: create compact selector files for Qwen follow-up runs.
- `run_qwen_followup_strengthening.sh`: run Qwen evaluations for one-segment, MLP, MLP+NMS, oracle, and fixed references.
- `run_qwen_temporal_router.sh`: run the temporal-supervised router diagnostic with Qwen.

## Experiment Wrappers

- `run_nextgqa_experiment.sh`: end-to-end NExT-GQA experiment flow.
- `run_nextgqa_full_single_gpu.sh`: full-data cache construction on one GPU.
- `run_nextgqa_full_multigpu.sh`: full-data cache construction across multiple GPUs.
- `research_run_sheet.sh`: run extended research experiment blocks on top of a completed cache.
- `run_research_single_gpu.sh`: single-GPU wrapper around the research run sheet.
- `run_multigpu_pipeline.sh`: generic multi-GPU pipeline wrapper.
- `run_research_controls.py`: fixed-budget sweeps and low-budget controls.

## Analysis

- `aggregate_run_summaries.py`: aggregate metrics across run directories.
- `summarize_nextgqa_paper_metrics.py`: compute tables, bootstrap intervals, and question-type breakdowns.
- `analyze_followup_strengthening.py`: analyze follow-up selector behavior.
- `plot_metric_tradeoffs.py`: plot accuracy-cost-grounding tradeoffs.
- `select_qualitative_cases.py`: select representative qualitative examples.
- `select_paper_qualitative_cases.py`: select report-quality qualitative examples.
- `build_paper_qualitative_figure.py`: build qualitative figure panels.
- `run_model_relative_study.py`: compare oracle evidence across answerers.
- `run_ablation.py`: run configured ablation blocks.

## Common Environment

Most scripts should be run from the repository root:

```bash
export PYTHONPATH=src:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Full-data scripts write experiment outputs under `runs/` by default. Raw datasets, extracted video artifacts, and full prediction caches are intentionally excluded from version control.
