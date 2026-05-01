# Research Status And Agent Handoff

Last updated: May 1, 2026.

This document is a concise handoff for anyone joining the repository after the
final experiment pass. For the submission package, treat
`docs/SUBMISSION_READINESS.md` and `docs/report/final_paper.pdf` as the canonical
summary.

## Research Question

The work studies grounded VideoQA as adaptive visual evidence acquisition:

> How much visual evidence can a grounded VideoQA system avoid consuming before
> answer accuracy and temporal grounding begin to degrade?

The final paper focuses on NExT-GQA visual evidence: keyframes and short temporal
segments. Subtitle support exists in the codebase, but subtitle-heavy claims are
not part of the final empirical story.

## Final Method

The final proposed operating point is `Qwen MLP+NMS (Our)`:

1. build a candidate pool of frames and temporal segments;
2. score candidate evidence with a compact MLP router over frozen visual/question
   features;
3. apply temporal non-maximum suppression to reduce redundant neighboring
   selections;
4. pass the selected evidence to Qwen2.5-VL-3B-Instruct for multiple-choice
   answering;
5. evaluate answer accuracy, evidence cost, temporal overlap, and grounded answer
   accuracy.

The temporal-supervised router is reported as a diagnostic ablation. Oracle rows
measure candidate-pool headroom rather than a deployable method.

## Completed Full-Test Results

All final Qwen2.5-VL test results are complete on the `5,553`-example NExT-GQA
test split.

| Method | Acc@QA | Cost | Count | mIoP | IoP@0.5 | mIoU | IoU@0.5 | Acc@GQA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen one segment | `0.668` | `1.500` | `1.000` | `0.301` | `0.311` | `0.165` | `0.133` | `0.221` |
| Qwen MLP+NMS (Our) | `0.689` | `2.935` | `2.000` | `0.440` | `0.453` | `0.246` | `0.210` | `0.321` |
| Qwen temporal-supervised router | `0.696` | `2.273` | `2.000` | `0.433` | `0.432` | `0.088` | `0.075` | `0.312` |
| Qwen oracle top-2 | `0.701` | `2.493` | `2.000` | `0.618` | `0.615` | `0.224` | `0.207` | `0.442` |
| Qwen fixed 3+3 | `0.724` | `7.500` | `6.000` | `0.618` | `0.615` | `0.296` | `0.287` | `0.454` |

Interpretation:

- `MLP+NMS` is the strongest learned grounded selector and is the main method.
- The temporal-supervised router is cheaper and slightly higher on Acc@QA, but
  its much lower IoU makes it a diagnostic rather than the final method.
- Oracle top-2 nearly matches fixed 3+3 grounded accuracy at much lower cost,
  showing that routing quality is the main remaining bottleneck.
- Fixed 3+3 is a high-cost coverage reference, not the proposed method.

## Canonical Artifacts

- Final report: `docs/report/final_paper.pdf`
- Report source: `docs/report/final_paper.tex`
- Final readiness log: `docs/SUBMISSION_READINESS.md`
- Demo file: `demo/nextgqa_qwen_sample.md`
- Final Qwen metrics:
  `docs/artifacts/final_metrics/qwen_test_followup_compare.md`
- Temporal-router metrics:
  `docs/artifacts/final_metrics/qwen_test_temporal_router_compare.md`
- Frozen controls, candidate-pool ablations, NMS ablations, and qualitative case
  summaries: `docs/artifacts/final_metrics/`
- Main qualitative figure: `docs/report/qualitative_cases.pdf`
- Appendix qualitative figure: `docs/report/appendix_qualitative_cases.pdf`

## Reproduction Entry Points

```bash
export PYTHONPATH=src:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash scripts/run_qwen_followup_strengthening.sh test
bash scripts/run_qwen_temporal_router.sh test
```

The full cached run uses `data/nextgqa_hf/` for annotations and
`data/nextgqa_hf/NExTVideo/` for videos. If the dataset is obtained from a gated
Hugging Face mirror, provide `HF_TOKEN` through a local `.env` file or the shell
environment. Do not commit `.env`.
The large local `data/` and `runs/` directories are intentionally ignored for the
GitHub artifact package.

## Remaining Research Caveats

The work is submission-ready for the class final report, but the limitations
should remain explicit:

- the answerer is frozen rather than fine-tuned end to end;
- the router is compact and uses frozen features;
- selected-evidence temporal overlap is an evidence-faithfulness metric, not a
  dense localization benchmark;
- the final empirical scope is NExT-GQA visual evidence acquisition.
