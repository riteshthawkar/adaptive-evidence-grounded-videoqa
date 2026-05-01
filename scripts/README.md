This directory contains dataset preprocessing, feature extraction, model training,
evaluation, and paper-analysis entry points.

Core scripts:

- `prepare_tvqa.py`
- `prepare_tvqa_plus.py`
- `prepare_nextgqa.py`
- `build_candidate_pool.py`
- `materialize_visual_evidence.py`
- `extract_clip_features.py`
- `run_fixed_budget_baseline.py`
- `train_answerer.py`
- `export_oracle_traces.py`
- `train_policy.py`
- `run_sequential_policy.py`
- `run_model_relative_study.py`
- `run_research_controls.py`
- `select_qualitative_cases.py`
- `run_multigpu_pipeline.sh`
- `run_nextgqa_experiment.sh`
- `run_nextgqa_full_single_gpu.sh`
- `run_nextgqa_full_multigpu.sh`
- `research_run_sheet.sh`
- `run_research_single_gpu.sh`
- `aggregate_run_summaries.py`
- `run_ablation.py`
- `run_vlm_answerer.py`
- `run_vlm_qwen25_submission_followups.sh`
- `create_followup_evidence_selections.py`
- `train_evidence_router.py`
- `train_temporal_router_from_pool.py`
- `run_qwen_followup_strengthening.sh`
- `run_qwen_temporal_router.sh`
- `summarize_nextgqa_paper_metrics.py`
- `plot_metric_tradeoffs.py`
- `select_paper_qualitative_cases.py`
- `build_paper_qualitative_figure.py`

The dataset normalization scripts now convert raw TVQA, TVQA+, and NExT-GQA annotations into
the repository's unified JSONL format. The NExT-GQA path supports official CSV annotations plus optional
grounded-span JSON, frame-time JSON, and video-id maps.
The candidate-pool builder now turns normalized records into chunked subtitle, segment, and frame candidates.
For visual-only datasets such as NExT-GQA, it can use frame timestamps or full-video duration metadata instead of subtitles.
The visual materialization script now extracts real frame images and optional segment clips from source videos.
The CLIP feature script now encodes those frame artifacts and aggregates segment embeddings.
The fixed-budget baseline runner now evaluates lexical or BM25 retrieval plus either the lexical answerer
or the trained linear answerer and writes summary metrics and per-example outputs.
The answerer trainer now fits a reproducible linear multiple-choice scorer on fixed-budget retrieved evidence.
The oracle trace exporter now converts retrieved seed evidence into offline acquisition targets for later policy training,
including explicit oracle modes for correctness, sufficiency, and temporal-grounding constraints.
The policy trainer now fits a trainable sequential acquisition policy from those oracle traces.
The sequential policy runner now evaluates keyword or learned acquisition policies on retriever-built candidate pools,
using the same retrieval allocation controls as trace export.
It also exposes deterministic low-budget controls such as frame-only, segment-only, top-evidence-only, and
frame-plus-segment one-shot policies.
The model-relative study runner compares oracle subsets across two answerers and records overlap, transfer, and grounding metrics.
The research controls runner evaluates these low-budget controls plus fixed-budget sweeps and writes a compact
paper-table JSON/Markdown summary.
The qualitative case selector joins fixed-budget and learned-policy prediction files and extracts examples for
efficient successes, grounding tradeoffs, early-stop failures, and cases where the policy fixes the fixed budget.
The multi-GPU bash runner orchestrates the current end-to-end experiment on HPC-style machines by parallelizing
CLIP feature extraction and retrieval-heavy evaluation across available GPUs.
The focused NExT-GQA bash runner orchestrates the validated single-GPU experiment flow used for larger-subset
and full-run experiments on HPC machines: preprocessing, candidate generation, visual materialization,
CLIP feature extraction, fixed-budget baseline, keyword baseline, oracle export, learned-policy training,
and learned-policy evaluation.
The full-data single-GPU and multi-GPU wrappers package the validated NExT-GQA cache-building flow.
They are intended to create reusable full-data artifacts that later experiments can consume without
repeating raw-video preprocessing.
The research run sheet assumes such a full-data cache already exists and then launches the experiment blocks
that matter most for a research-grade paper: multi-seed learned-policy runs, fixed-budget sweeps, low-budget
one-shot controls, evidence-type ablations, and model-relative analysis.
The single-GPU research wrapper is the easiest entry point for a machine with one CUDA device and around
`24 GB` VRAM. It delegates to the research run sheet after confirming that the single-GPU full-data cache
exists.
The summary aggregation script reads multiple run directories and prints a paper-ready mean/std table for
selected summary files and metrics.

Final submission entry points:

```bash
export PYTHONPATH=src:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash scripts/run_qwen_followup_strengthening.sh test
bash scripts/run_qwen_temporal_router.sh test
```

The final paper metrics are written under:

- `runs/nextgqa_full_seed13/outputs/followup_strengthening/qwen/metrics/`
- `runs/nextgqa_full_seed13/outputs/followup_strengthening/temporal_router/qwen/metrics/`

Lightweight copies for the GitHub artifact package are stored in:

- `docs/artifacts/final_metrics/`
