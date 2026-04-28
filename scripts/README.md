This directory is reserved for dataset preprocessing, feature extraction, and training entry points.

As the implementation matures, expected scripts are:

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
- `run_multigpu_pipeline.sh`
- `run_nextgqa_experiment.sh`
- `run_nextgqa_full_single_gpu.sh`
- `run_nextgqa_full_multigpu.sh`
- `research_run_sheet.sh`
- `run_research_single_gpu.sh`
- `aggregate_run_summaries.py`
- `run_ablation.py`

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
The model-relative study runner compares oracle subsets across two answerers and records overlap, transfer, and grounding metrics.
The multi-GPU bash runner orchestrates the current end-to-end experiment on HPC-style machines by parallelizing
CLIP feature extraction and retrieval-heavy evaluation across available GPUs.
The focused NExT-GQA bash runner orchestrates the validated single-GPU experiment flow used for larger-subset
and full-run experiments on HPC machines: preprocessing, candidate generation, visual materialization,
CLIP feature extraction, fixed-budget baseline, keyword baseline, oracle export, learned-policy training,
and learned-policy evaluation.
The full-data single-GPU and multi-GPU wrappers package the validated NExT-GQA cache-building flow for the
Omkar machines used during the project. They are intended to create reusable full-data artifacts that later
experiments can consume without repeating raw-video preprocessing.
The research run sheet assumes such a full-data cache already exists and then launches the experiment blocks
that matter most for a research-grade paper: multi-seed learned-policy runs, fixed-budget sweeps, evidence-type
ablations, and model-relative analysis.
The single-GPU research wrapper is the easiest entry point for a machine with one CUDA device and around
`24 GB` VRAM. It delegates to the research run sheet after confirming that the single-GPU full-data cache
exists.
The summary aggregation script reads multiple run directories and prints a paper-ready mean/std table for
selected summary files and metrics.
