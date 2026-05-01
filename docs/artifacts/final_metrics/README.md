# Final Metric Artifacts

This directory contains lightweight copies of the final metric summaries used by
the report. The full local `runs/` directory contains large prediction files,
frame caches, and intermediate experiment outputs, so it is intentionally not
part of the GitHub artifact package.

Files:

- `qwen_test_followup_compare.md` / `.json`: final Qwen2.5-VL test comparison,
  including one-segment, linear policy, MLP, MLP+NMS, oracle, and fixed 3+3 rows.
- `qwen_test_temporal_router_compare.md` / `.json`: temporal-supervised router
  diagnostic comparison.
- `frozen_test_policy_comparison_bootstrap.md` / `.json`: frozen-answerer test
  controls with bootstrap reporting.
- `test_candidate_pool_recall.md`: candidate-pool temporal support analysis.
- `test_nms_ablation_selection_summary.md`: temporal NMS selector ablation.
- `paper_qualitative_cases.md`: qualitative examples used to build the paper
  figures.
