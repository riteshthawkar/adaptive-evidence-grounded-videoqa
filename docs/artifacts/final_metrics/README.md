# Final Metric Artifacts

This directory contains lightweight copies of the final metric summaries used by
the report. The full local `runs/` directory contains large prediction files,
frame caches, and intermediate experiment outputs, so it is intentionally not
part of the GitHub artifact package.

Files:

- `qwen_test_followup_compare.json`: final Qwen2.5-VL test comparison,
  including one-segment, linear policy, MLP, MLP+NMS, oracle, and fixed 3+3 rows.
- `qwen_test_temporal_router_compare.json`: temporal-supervised router
  diagnostic comparison.
- `frozen_test_policy_comparison_bootstrap.json`: frozen-answerer test
  controls with bootstrap reporting.
