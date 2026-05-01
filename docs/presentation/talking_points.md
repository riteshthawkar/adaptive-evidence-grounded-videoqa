# Final Presentation Talking Points

Target time: 7 minutes plus Q&A.

## Slide 1: Title, 20 seconds

State the project in one sentence: we study how to answer grounded VideoQA
questions using adaptive evidence instead of a fixed context budget.

## Slide 2: Motivation, 45 seconds

Video evidence is sparse. Many questions depend on one short event, but standard
systems process a fixed number of frames or segments. That wastes compute and
does not prove the answer came from the right time span.

## Slide 3: Related Work and Gap, 50 seconds

Prior work gives strong VideoQA benchmarks, pretrained vision-language models,
and NExT-GQA for grounded answering. Our gap is the budget question: once we
have candidate evidence, how much should we actually acquire before answering?

## Slide 4: Problem Formulation, 45 seconds

Define the example, candidate evidence pool, selected subset, cost, and target
grounding span. The goal is not only high answer accuracy, but high accuracy
with low evidence cost and temporal support.

## Slide 5: Method, 60 seconds

Walk through the pipeline: normalize NExT-GQA, build frame and segment
candidates, select evidence, answer with either the frozen answerer or Qwen, then
evaluate answer accuracy, cost, and grounding.

## Slide 6: Evaluation, 45 seconds

Mention validation and test split sizes. Explain the three evidence settings:
one segment, learned two-item policy, and fixed 3+3. Define Acc@QA and Acc@GQA
briefly.

## Slide 7: Frozen Results, 50 seconds

The frozen answerer is the bottleneck. The learned policy improves grounding
over one segment, but accuracy remains around 0.38. This motivated replacing the
answerer while keeping evidence selections fixed.

## Slide 8: Qwen Results, 70 seconds

This is the main result. Qwen improves answer accuracy sharply. The learned
policy reaches 0.695 validation Acc@QA at cost 2.785, while fixed 3+3 reaches
0.724 at cost 7.5. So compact evidence keeps most answer accuracy, but fixed
evidence still wins in Acc@GQA.

## Slide 9: Tradeoff Figure, 45 seconds

Use the figure to emphasize that this is not a single winner story. The learned
policy is an efficient operating point; fixed 3+3 is a grounding-focused
operating point.

## Slide 10: Qualitative Lessons, 45 seconds

Use the three examples to show why answer accuracy alone is insufficient:
correct and grounded, correct but weakly grounded, grounded but wrong.

## Slide 11: Conclusion, 45 seconds

End with the final message: grounded VideoQA should be evaluated as evidence
selection plus answering. The result is an accuracy-cost-grounding tradeoff.

## Likely Q&A

**Why not use the official NExT-GQA evaluator directly?**

The official grounding evaluator expects one predicted temporal span per
question. Our method selects an evidence set. We report evidence-set support
using best selected overlap, and we explicitly state that it is not identical to
single-span leaderboard evaluation.

**Why did Qwen improve accuracy but not grounding overlap?**

The answerer and evidence selector solve different parts of the problem. Qwen
reasons better from the selected images, but the evidence policy still determines
whether those images cover the annotated support moment.

**What is the next research step?**

Train the evidence policy using feedback from the stronger VLM answerer, and
represent segments with multiple frames or video input rather than one sampled
frame.
