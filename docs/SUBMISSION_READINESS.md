# Submission Readiness Log

Last updated: May 1, 2026

## Final Research Claim

The work studies grounded VideoQA as adaptive evidence acquisition. The main claim
is an accuracy-cost-grounding tradeoff: compact learned evidence can improve
grounded accuracy over low-cost single-segment evidence, but answer accuracy alone
does not imply faithful temporal grounding.

The final proposed learned operating point is `Qwen MLP+NMS (Our)`. The temporal
supervised router is reported as a diagnostic ablation, while oracle rows measure
candidate-pool headroom.

## Completed Final Results

- Full NExT-GQA test split contains 5,553 examples.
- Full Qwen2.5-VL-3B-Instruct test results are complete for:
  - one segment,
  - linear min-2 policy,
  - MLP top-2,
  - MLP+NMS,
  - temporal-supervised router,
  - oracle top-2/top-3,
  - fixed 3 frames + 3 segments.
- Completed main test results:
  - `Qwen one segment`: 0.668 Acc@QA, 0.221 Acc@GQA, cost 1.500.
  - `Qwen MLP+NMS (Our)`: 0.689 Acc@QA, 0.321 Acc@GQA, cost 2.935.
  - `Qwen temporal-supervised router`: 0.696 Acc@QA, 0.312 Acc@GQA, cost 2.273.
  - `Qwen oracle top-2`: 0.701 Acc@QA, 0.442 Acc@GQA, cost 2.493.
  - `Qwen fixed 3+3`: 0.724 Acc@QA, 0.454 Acc@GQA, cost 7.500.

## Submission Interpretation

- `MLP+NMS` is the strongest learned grounded selector and should be the main
  method in the report.
- The temporal-supervised router is cheaper and slightly higher in Acc@QA than
  MLP+NMS, but its IoU is much lower, so it should remain an ablation.
- Oracle top-2 nearly matches fixed 3+3 Acc@GQA at about one third of the cost,
  showing that the candidate pool is strong and routing is the main bottleneck.
- Fixed 3+3 is a high-cost coverage reference, not the proposed method.

## Final Artifacts

- Main report source: `docs/report/final_paper.tex`
- Main report PDF: `docs/report/final_paper.pdf`
- Presentation-values report copy: `docs/report/final_paper_presentation_values.pdf`
- Main qualitative figure: `docs/report/qualitative_cases.pdf`
- Appendix qualitative figure: `docs/report/appendix_qualitative_cases.pdf`
- Final tradeoff plot: `docs/report/qwen_current_tradeoff_presentation.png`
- Final Qwen metrics:
  - `docs/artifacts/final_metrics/qwen_test_followup_compare.md`
  - `docs/artifacts/final_metrics/qwen_test_temporal_router_compare.md`
- Frozen controls, candidate-pool ablations, NMS ablations, and qualitative case
  summaries: `docs/artifacts/final_metrics/`

## Verification

- `docs/report/final_paper.pdf` compiles successfully.
- Main content ends on page 9; references start on page 10 and appendix follows.
- Remaining LaTeX messages are harmless underfull boxes and a PDF-version warning
  from including `figure_1.pdf`.
