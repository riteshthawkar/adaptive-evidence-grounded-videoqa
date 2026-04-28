# Research Status And Agent Handoff

This document is the shortest accurate handoff for a new coding agent joining the project.
Read this before changing the method, running large experiments, or rewriting the paper scope.

## 1. Research Question

The project is not a generic VideoQA benchmark reproduction.
The core research question is:

> How much visual evidence can a grounded VideoQA system avoid consuming before answer accuracy and temporal grounding begin to degrade?

The project studies **adaptive visual evidence acquisition** for grounded VideoQA.
The evidence types in the current validated setup are:

- keyframes
- short temporal segments

Subtitle-aware acquisition is part of the broader design, but it is not yet the strongest validated story.
The paper should therefore be framed around **visual evidence acquisition** unless stronger subtitle experiments are completed later.

## 2. Why This Project Is Worth Doing

Most VideoQA systems still consume a fixed amount of retrieved context regardless of question difficulty.
That is wasteful and it confounds answer quality with evidence quantity.
Our approach makes evidence consumption itself part of the model:

- retrieve a candidate evidence pool
- acquire evidence sequentially under a budget
- stop once extra evidence is predicted to be low-value

The goal is not only to improve answer accuracy.
The real scientific target is the tradeoff between:

- answer accuracy
- evidence cost
- evidence count
- temporal grounding quality

That tradeoff is already visible in the current results, which is why the direction is research-worthy.

## 3. What Is Implemented

The codebase currently supports:

- dataset normalization for TVQA, TVQA+, and NExT-GQA
- candidate-pool generation
- visual evidence materialization from videos
- CLIP-based frame and segment features
- lexical, BM25, and hybrid CLIP retrieval
- fixed-budget baselines
- frozen multimodal answerer evaluation
- linear answerer training
- oracle trace export
- sequential policy training and evaluation
- model-relative minimal-evidence comparison
- run aggregation across seeds
- HPC-oriented single-GPU and multi-GPU wrappers

## 4. What Has Been Scientifically Validated

The following points are confirmed and should be treated as real results, not aspirations:

1. The full NExT-GQA pipeline works end to end on real data.
2. The degenerate immediate-stop policy failure was identified and fixed by enforcing `min_items_before_stop = 1`.
3. A controlled NExT-GQA subset experiment (`500` train / `200` validation) was completed across `3` seeds.

The confirmed subset result is:

- fixed budget: accuracy `0.360`, evidence cost `7.5`, evidence count `6.0`, temporal IoU `0.273`
- keyword sequential baseline: same operating point as fixed budget in the current setup
- learned policy: accuracy `0.365`, evidence cost `1.5`, evidence count `1.0`, temporal IoU `0.168`

The correct interpretation is:

- the learned policy **matches** fixed-budget accuracy on the current setup
- it uses much less evidence
- it yields weaker temporal grounding overlap

This is already enough to support a meaningful efficiency-grounding tradeoff story.

## 5. What Is Not Yet Strong Enough

The project is promising, but it is not yet complete as research-grade work.
The main gaps are:

- the strongest result is still on a controlled subset rather than a finished full-data study
- the heuristic baseline suite is too weak
- the temporal-IoU drop needs deeper analysis
- seed invariance has to be explained more rigorously at larger scale
- the current learned policy is still simple, so the evaluation must be strong enough to justify it

Do not overclaim this as a solved “minimal sufficient evidence” problem.
Right now, the evidence supports an **efficiency claim first**.

## 6. Confirmed Reviewer Signal From The Interim Report

The midway report received strong feedback overall, but the key critiques were scientifically useful:

- some tables were too similar, so the effect size needed clearer interpretation
- different seeds showed no change, which needed explanation
- the temporal grounding tradeoff deserved more attention

These are not signs of a weak project.
They tell us exactly what the final paper has to strengthen.

## 7. Recommended Final Paper Scope

Unless much stronger multimodal evidence experiments are completed, the final paper should be framed as:

**Adaptive visual evidence acquisition for grounded VideoQA**

The main claim should be:

> A learned acquisition policy can preserve answer accuracy while reducing evidence cost substantially, but the gain comes with a measurable temporal-grounding tradeoff.

That claim is narrower than the original broad ambition, but it is scientifically cleaner and already supported by real results.

## 8. What To Run Next

The next serious experiments should be done in this order:

1. full-data NExT-GQA main run
2. learned-policy runs over `3` seeds on full data
3. fixed-budget sweep (`1+1`, `2+2`, `3+3`, `4+4`)
4. frame-only and segment-only ablations
5. model-relative comparison across answerers
6. qualitative case studies

The new launchers for this are:

- `scripts/research_run_sheet.sh`
- `scripts/run_research_single_gpu.sh`

The research run sheet assumes a completed full-data cache run and then reuses cached artifacts instead of redoing video materialization for every seed.

## 9. What A Research-Grade Final Result Must Include

At minimum, the final paper should report:

- full-data main comparison
- `3`-seed learned-policy robustness
- budget sweep
- ablation by evidence granularity
- temporal IoU, sufficiency, comprehensiveness, and oracle-validity analysis
- qualitative successes and failures

The final paper tables should cover:

- main full-data result
- budget sweep
- ablation table
- robustness table

## 10. What New Agents Should Not Waste Time On

Avoid these unless the main full-data story is already complete:

- broadening the scope to too many datasets
- rewriting the answerer into a large end-to-end model
- spending time on subtitle-heavy claims without supporting experiments
- polishing the report wording before the full-data evidence is stronger
- adding many method variants before the baseline suite is rigorous

## 11. Immediate Priorities For A New Agent

If you are picking up this project now, do this first:

1. verify whether the latest full-data cache run completed
2. reuse cached `train.visual_features.jsonl` and `val.visual_features.jsonl`
3. run `scripts/run_research_single_gpu.sh` or `scripts/research_run_sheet.sh`
4. summarize the resulting metrics into the planned paper tables
5. only then decide whether the policy model itself needs to be upgraded

## 12. Bottom Line

This is no longer just a class project.
It already has a legitimate research question, a working codebase, and a real empirical signal.
The remaining work is to make the evaluation strong enough that the final paper is judged by the tradeoff it uncovers, not by the fact that it uses a sequential policy.
