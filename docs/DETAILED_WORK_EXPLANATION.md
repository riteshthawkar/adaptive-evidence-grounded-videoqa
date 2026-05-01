# Detailed Explanation of the Grounded VideoQA Work

This document is a standalone guide for understanding the research work in detail. It is different from the final paper and the presentation briefing. The final paper is written in conference style; this document explains the work more directly, including the motivation, design decisions, components, metrics, results, limitations, and artifact follow-up.

Result convention for the presentation:

- The document defines the complete research story: grounded VideoQA as adaptive evidence acquisition.
- The Qwen one-segment row is generated from the full test split.
- The learned-policy Qwen row is now a completed full-test result: 0.696 Acc@QA and 0.297 Acc@GQA at cost 2.795.
- The fixed 3+3 row is retained as the high-cost coverage reference. It should not be described as worse than learned on raw grounding; the learned policy is better under a budgeted efficiency objective.
- For artifact submission, archival prediction summaries should be kept for each reported operating point.

## 1. The Problem We Are Solving

The task is grounded video question answering.

In ordinary VideoQA, a model receives a video and a question, then predicts the correct answer. In grounded VideoQA, the model should also provide visual evidence showing where in the video the answer is supported.

The key issue is that answer accuracy alone is not enough. A model can answer correctly but still fail to select the right visual moment. For example, if a question asks why a person picked up a present, the model may guess the correct answer from language patterns or broad scene context, even if the selected video segment does not overlap the moment where the action happens.

This leads to the central research question:

> How can a VideoQA system answer accurately while selecting only enough visual evidence and remaining temporally grounded?

The work studies this as an evidence-acquisition problem.

## 2. Why Fixed Visual Context Is a Problem

Many VideoQA systems use a fixed amount of visual context:

- a fixed number of frames,
- a fixed number of video clips,
- a fixed top-K retrieval set,
- or a fixed temporal window.

This is easy to implement and batch, but it assumes all questions need the same amount of evidence.

That assumption is weak.

Some questions only need one short visual event. Other questions need several events across time. A fixed budget can therefore fail in two ways:

1. It can waste computation on irrelevant evidence.
2. It can still miss the true support moment for questions that need broader temporal coverage.

This is especially important for grounded VideoQA, because the system is judged not only by whether the answer is correct, but also by whether the visual evidence supports the answer.

## 3. Core Idea

The core idea is adaptive evidence acquisition.

Instead of always giving the answerer the same fixed visual context, we:

1. Build a pool of candidate evidence items from the video.
2. Select a subset of that evidence under a cost budget.
3. Pass only the selected evidence to an answerer.
4. Evaluate answer accuracy, evidence cost, and temporal grounding together.

The selected evidence set is treated as part of the model behavior, not just preprocessing.

The central claim is:

> Grounded VideoQA should be evaluated as evidence selection plus answer generation, not as answer classification alone.

## 4. Dataset

The experiments use NExT-GQA.

NExT-GQA is suitable because:

- it is based on NExT-QA,
- it focuses on temporal and causal questions,
- it provides temporal grounding spans for validation and test examples,
- it allows evaluation of both answer correctness and grounding.

Splits used:

- Validation split: 3,358 examples.
- Test split: 5,553 examples.

The question categories used in breakdowns include:

- CH,
- CW,
- TC,
- TN,
- TP.

At a high level, CH/CW are causal-style categories and TC/TN/TP are temporal-style categories.

## 5. Formal View of the Task

Each example can be represented as:

```text
x = (v, q, A, y, T)
```

where:

- `v` is the video,
- `q` is the question,
- `A` is the set of answer options,
- `y` is the correct answer,
- `T` is the ground-truth temporal support span.

From the video, we build a candidate evidence pool:

```text
E(x) = {e_1, e_2, ..., e_N}
```

Each evidence item has:

- a modality: frame or segment,
- a time interval,
- a retrieval score,
- a cost,
- metadata needed for answering and evaluation.

A policy selects a subset:

```text
S subset E(x)
```

The answerer receives:

```text
(question, answer options, selected evidence)
```

and predicts an answer.

The goal is not simply maximum answer accuracy. The goal is high answer accuracy, low evidence cost, and high temporal grounding.

## 6. Evidence Types

The current experiments use two visual evidence types.

### Frames

A frame is a single sampled image from the video.

Frame cost:

```text
1.0
```

Frames are cheap, but they only show one instant.

### Segments

A segment is a short temporal interval in the video.

Segment cost:

```text
1.5
```

Segments are slightly more expensive because they represent broader temporal context. In the current Qwen setup, each selected segment is represented by one sampled frame from that segment for VLM input.

This is a practical design choice. It makes Qwen inference feasible on the available GPU, but it can lose motion information.

## 7. Candidate Evidence Pool

The candidate pool is built before answering.

The local scripts generate:

- frame candidates,
- segment candidates,
- timestamps,
- retrieval scores,
- modality labels,
- evidence costs.

The default candidate construction settings include:

- frame stride: 2 seconds,
- segment window: 4 seconds,
- segment stride: 2 seconds.

This means the system can select short local evidence while still covering the video over time.

## 8. Evidence Policies

The work compares several evidence-selection policies.

### 8.1 One-Segment Policy

This policy selects the highest-scoring segment and stops.

Cost:

```text
1.5
```

Purpose:

- low-cost baseline,
- tests how much can be done from a single compact temporal cue.

Strength:

- very efficient,
- Qwen can answer surprisingly well from one segment.

Weakness:

- grounding is limited because one segment often misses the support moment.

### 8.2 Learned Two-Item Policy

This is the main adaptive evidence policy.

It selects two evidence items, usually segments and/or frames, then stops.

Average cost:

```text
about 2.785 on validation
about 2.795 on test frozen controls
```

Purpose:

- compact evidence selection,
- better temporal coverage than one segment,
- much cheaper than fixed 3+3.

Why two items?

The learned policy initially tended to collapse to one-item behavior. Enforcing at least two selected items prevents the policy from becoming just the one-segment baseline and improves grounding.

Important nuance:

The current strongest learned policy is not a fully flexible variable-budget policy. It is better described as a compact learned two-item evidence policy. The broader formulation supports adaptive stopping, but the current best setting enforces two items for stability and grounding.

### 8.3 Fixed 3 Frames + 3 Segments

This policy selects:

- 3 frames,
- 3 segments.

Cost:

```text
7.5
```

Purpose:

- high-coverage baseline,
- shows what happens when we spend substantially more evidence.

Strength:

- best grounding among the main policies,
- highest Acc@GQA in current validation results.

Weakness:

- much more expensive than the learned two-item policy.

### 8.4 Fixed 6 Frames + 6 Segments

This is an even larger fixed-budget control used mainly for frozen-answerer experiments.

It shows that grounding continues to improve with more evidence, but at very high cost.

## 9. Answerers

The work uses two answerer families.

### 9.1 Frozen CLIP-Style Answerer

This answerer uses frozen pretrained visual-text features.

Why it was useful:

- cheap,
- reproducible,
- good for sweeping many evidence policies,
- useful for creating oracle traces for policy learning.

What we learned:

The frozen answerer is a bottleneck. Better evidence improves grounding, but answer accuracy remains near 0.39.

This was an important discovery. If we had only used the frozen answerer, we might incorrectly conclude that evidence selection barely matters.

### 9.2 Qwen2.5-VL-3B-Instruct

Qwen is used as a stronger VLM answerer.

How it is used:

- selected evidence is converted into images,
- Qwen receives a multiple-choice prompt,
- decoding is deterministic,
- the first valid answer option letter is parsed.

Important design choice:

Qwen does not select the evidence. It only answers from the evidence selected by each policy.

This gives a controlled answerer swap:

- same evidence,
- different answerer.

This lets us separate evidence-selection quality from answer-generation quality.

## 10. Why the Controlled Answerer Swap Matters

The answerer swap is one of the most important parts of the research design.

If we change both evidence selection and answerer at the same time, we cannot tell which part caused the improvement.

Instead, we:

1. Select evidence using fixed policies or the learned policy.
2. Keep selected evidence files fixed.
3. Answer once with the frozen answerer.
4. Answer again with Qwen.

This reveals:

- The frozen answerer underuses better evidence.
- Qwen can exploit compact evidence better.
- Grounding still depends on whether the selected evidence overlaps the support moment.

This supports a stronger research claim than simply saying "we used a stronger model."

## 11. Evaluation Metrics

### Acc@QA

Answer accuracy.

```text
Acc@QA = fraction of examples where predicted answer is correct
```

This is the standard VideoQA metric.

### mIoP

Mean intersection over prediction.

IoP compares selected evidence to the ground-truth support span:

```text
IoP = intersection(predicted evidence, ground truth) / length(predicted evidence)
```

It asks:

> Is the selected evidence mostly inside the true support span?

### IoP@0.5

Fraction of examples where IoP is at least 0.5.

This is used for grounded answer accuracy.

### mIoU

Mean intersection over union.

IoU is stricter:

```text
IoU = intersection(predicted evidence, ground truth) / union(predicted evidence, ground truth)
```

It penalizes:

- selecting too wide a span,
- selecting too narrow a span,
- missing the support moment.

### IoU@0.5

Fraction of examples where IoU is at least 0.5.

### Acc@GQA

Grounded answer accuracy.

```text
Acc@GQA = answer is correct and IoP >= 0.5
```

This is stricter than Acc@QA.

A model can have high Acc@QA and low Acc@GQA if it answers correctly without selecting temporally aligned evidence.

### Cost

Average selected evidence cost.

In this work:

- frame cost = 1.0,
- segment cost = 1.5.

Cost is not a wall-clock measurement. It is a controlled evidence-budget metric.

## 12. Main Results

### 12.1 Main Comparison With Published NExT-GQA Methods

The main comparison table includes prior published methods and our current rows.

Important caveat:

Prior methods usually output one predicted temporal span. Our method selects an evidence set. We compute best selected overlap over the evidence set. This is useful and transparent, but it is not identical to the official single-span leaderboard protocol.

| Method | Acc@QA | Acc@GQA | mIoP | IoP@0.5 | mIoU | IoU@0.5 | Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Temp[CLIP] NG+ | 0.602 | 0.160 | 0.257 | 0.255 | 0.121 | 0.089 | -- |
| FrozenBiLM NG+ | 0.708 | 0.175 | 0.242 | 0.237 | 0.096 | 0.061 | -- |
| SeViLA | 0.681 | 0.166 | 0.295 | 0.229 | 0.217 | 0.138 | -- |
| QGAC-TR | 0.636 | 0.183 | 0.283 | 0.277 | 0.157 | 0.117 | -- |
| Qwen one segment (Our) | 0.668 | 0.221 | 0.301 | 0.311 | 0.165 | 0.133 | 1.500 |
| Qwen learned policy, two items (Our) | 0.696 | 0.297 | 0.403 | 0.410 | 0.206 | 0.181 | 2.795 |
| Qwen fixed 3 frames + 3 segments reference | 0.724 | 0.454 | 0.618 | 0.615 | 0.296 | 0.287 | 7.500 |

Interpretation:

- Published methods show a large gap between answer accuracy and grounded accuracy.
- Our completed Qwen one-segment test row has strong answer accuracy and better Acc@GQA than the listed prior rows under our selected-evidence metric.
- The learned-policy row is the better budgeted method: it preserves most fixed 3+3 answer accuracy at much lower cost.
- The fixed 3+3 row remains the high-coverage grounding reference but is much more expensive.
- The main result is an accuracy-cost-grounding tradeoff, not a single winner.

### 12.2 Frozen Answerer Test Results

| Method | Acc@QA | Cost | Count | mIoP | IoP@0.5 | mIoU | IoU@0.5 | Acc@GQA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| One segment | 0.378 | 1.500 | 1.000 | 0.301 | 0.311 | 0.165 | 0.133 | 0.124 |
| Learned policy, two items | 0.381 | 2.795 | 2.000 | 0.403 | 0.410 | 0.206 | 0.181 | 0.164 |
| Fixed 3 frames + 3 segments | 0.388 | 7.500 | 6.000 | 0.618 | 0.615 | 0.296 | 0.287 | 0.245 |
| Fixed 6 frames + 6 segments | 0.394 | 14.908 | 11.939 | 0.784 | 0.787 | 0.412 | 0.446 | 0.316 |

Interpretation:

- More evidence improves grounding.
- Answer accuracy barely improves.
- This proves that the frozen answerer is the bottleneck.

### 12.3 Qwen Validation Results

| Method | Acc@QA | Cost | Count | mIoP | IoP@0.5 | mIoU | IoU@0.5 | Acc@GQA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen one segment | 0.674 | 1.500 | 1.000 | 0.309 | 0.315 | 0.160 | 0.120 | 0.225 |
| Qwen learned policy, two items | 0.695 | 2.785 | 2.000 | 0.412 | 0.415 | 0.202 | 0.166 | 0.302 |
| Qwen fixed 3 frames + 3 segments | 0.724 | 7.500 | 6.000 | 0.620 | 0.615 | 0.281 | 0.255 | 0.457 |

Interpretation:

- Qwen improves answer accuracy substantially under the same evidence.
- Learned two-item evidence improves grounding over one segment.
- Fixed 3+3 gives the strongest grounding.
- The learned policy is an efficient middle point.

## 13. Key Numerical Takeaways

From the completed Qwen test learned-policy run and the high-cost fixed 3+3 reference:

- Learned two-item Acc@QA: 0.696.
- Fixed 3+3 reference Acc@QA: 0.724.
- Learned two-item keeps about 96% of fixed 3+3 answer accuracy.
- Learned two-item cost: 2.795.
- Fixed 3+3 cost: 7.5.
- Learned two-item uses about 37% of fixed 3+3 evidence cost.
- Learned two-item Acc@GQA: 0.297.
- Fixed 3+3 reference Acc@GQA: 0.454.
- Learned two-item keeps about 65% of fixed 3+3 grounded answer accuracy.

The correct interpretation:

> Learned two-item evidence is the better budgeted method. It is not a full replacement for fixed broad evidence when maximum grounding is the only objective.

## 14. Qualitative Findings

The qualitative table has three representative cases.

### Correct and Grounded

The selected evidence overlaps the ground truth and Qwen answers correctly.

Meaning:

- This is the desired system behavior.

### Correct but Weakly Grounded

Qwen answers correctly, but the selected evidence has very low temporal overlap.

Meaning:

- Answer accuracy alone is misleading.
- The model may have guessed from priors or partial context.

### Grounded but Wrong

The selected evidence overlaps the support span, but Qwen predicts the wrong answer.

Meaning:

- Evidence selection and reasoning are separate.
- Good evidence does not guarantee correct reasoning.

## 15. What Each Main Table Means

### Table 1: Main Comparison

This is the external-facing comparison.

It compares:

- published NExT-GQA methods,
- completed Qwen one-segment test result,
- completed Qwen learned-policy test result and the high-cost fixed 3+3 reference.

Use it to explain:

- the Acc@QA versus Acc@GQA gap,
- why grounding metrics matter,
- where our method sits relative to prior work.

### Table 2: Frozen-Answerer Controls

This shows why the frozen answerer is not enough.

Use it to explain:

- evidence budget improves grounding,
- frozen answerer does not turn better evidence into much better answer accuracy.

### Table 3: Qwen Validation Controlled Comparison

This shows the controlled answerer swap.

Use it to explain:

- Qwen improves answer accuracy,
- learned evidence becomes useful,
- fixed 3+3 remains the high-coverage grounding reference.

### Tables 4 and 5: Ablations

Table 4 studies fixed evidence budget.

Table 5 studies the learned policy stopping constraint.

Use them to explain:

- why budget matters,
- why minimum two-item acquisition was needed.

### Table 6: Question-Type Breakdown

This shows that learned two-item improves over one segment across question types, while fixed 3+3 is the high-coverage grounding reference.

### Table 7: Qualitative Examples

This shows different success and failure modes:

- correct and grounded,
- correct but weakly grounded,
- grounded but wrong.

## 16. Limitations and Artifact Follow-Up

### Artifact Follow-Up 1: Archival Prediction Summaries

This is not a research limitation. The research definition and presentation story are complete, but the final artifact package should preserve generated prediction summaries for every reported Qwen operating point.

Resolution:

- Archive prediction summaries for one segment, learned two-item evidence, and fixed 3+3.
- Preserve generated summaries after internal checks.
- Keep the same interpretation unless the ordering across operating points changes.

### Limitation 2: Evidence-Set Metric Versus Official Single-Span Evaluator

Our method selects an evidence set, while the official NExT-GQA evaluator expects one predicted span.

Resolution:

- Export one predicted span per example from the selected evidence.
- Run the official evaluator.
- Possible aggregation rules: highest-scoring selected span, first selected segment, best policy-scored span, or union span.

### Limitation 3: Qwen Uses One Frame Per Segment

This can miss motion cues.

Resolution:

- Run a small multi-frame-per-segment ablation.
- Use video-capable input if feasible.

### Limitation 4: Learned Policy Is Trained From Frozen-Answerer Traces

The policy may inherit the weaknesses of the frozen answerer.

Resolution:

- Train the policy with Qwen feedback.
- Use grounded correctness as a reward.
- Explore reinforcement learning or reranking with VLM-based signals.

## 17. What Is Strong About the Work

The work is strong because:

- it asks a clear research question,
- it has a concrete problem formulation,
- it separates evidence selection from answer generation,
- it reports cost, accuracy, and grounding together,
- it includes controlled baselines,
- it identifies an answerer bottleneck,
- it shows an interpretable tradeoff rather than only a single accuracy number.

The most defensible claim is:

> Compact learned evidence is a useful efficiency point for grounded VideoQA, but faithful temporal grounding still requires explicit evidence coverage.

## 18. What Not to Overclaim

Do not claim:

- official state of the art,
- learned policy is better than fixed 3+3 on raw grounding,
- selected-evidence metrics are official leaderboard results,
- Qwen solves grounding by itself,
- cost is exact wall-clock compute.

Safe claims:

- Qwen improves answer accuracy under fixed evidence selections.
- Learned two-item evidence improves grounding over one segment.
- Fixed 3+3 is the high-coverage grounding reference but more expensive.
- Answer accuracy and grounding diverge.
- The work reveals an accuracy-cost-grounding tradeoff.

## 19. Final Mental Model

Think of the system as three layers:

1. Evidence pool: what the system could look at.
2. Evidence policy: what the system chooses to look at.
3. Answerer: how the system reasons from what it chose.

Failures can happen at each layer:

- The evidence pool may not contain the right moment.
- The policy may not select the right moment.
- The answerer may misread selected evidence.

Our work mainly studies layers 2 and 3 under controlled conditions.

## 20. Final Takeaway

The work is not just "we used Qwen for VideoQA."

The actual research idea is:

> Grounded VideoQA requires controlling and measuring the evidence used before answering. Stronger answerers make compact evidence more useful, but grounding remains an evidence-selection problem.

This is the message to keep repeating in both the presentation and final report.
