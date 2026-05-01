# Presentation Briefing: Adaptive Evidence Acquisition for Grounded VideoQA

Use this as the detailed preparation document for the final presentation. It is written as a presenter guide, not as a paper section. The goal is to make sure every team member can explain the idea, pipeline, experiments, results, and likely questions clearly.

Result convention for today's presentation:

- Present the work as a complete evidence-acquisition study with three final operating points: one segment, learned two-item evidence, and fixed 3 frames + 3 segments.
- The one-segment and learned-policy Qwen rows are generated from the full test split.
- The learned-policy result is now a completed test result: 0.696 Acc@QA and 0.297 Acc@GQA at cost 2.795.
- Frame the learned policy as the better budgeted method: it keeps about 96% of the high-cost fixed 3+3 answer accuracy while using about 37% of the evidence cost.
- Do not claim the learned policy beats fixed 3+3 on raw grounding; fixed 3+3 remains the high-coverage reference.

## One-Sentence Summary

We study grounded video question answering as an adaptive evidence-acquisition problem: instead of always giving the answerer a fixed number of frames or clips, we select a compact set of visual evidence under a cost budget and evaluate answer accuracy, evidence cost, and temporal grounding together.

## 30-Second Pitch

Most VideoQA systems answer from a fixed visual context, such as a fixed number of frames or a fixed number of clips. That is convenient, but it ignores that different questions require different amounts of evidence. Some questions can be answered from one short event, while others need broader temporal context. In grounded VideoQA, answer correctness alone is not enough; the selected evidence should overlap the annotated support moment. Our work formulates this as adaptive evidence acquisition. We compare low-cost one-segment evidence, a learned two-item policy, and larger fixed-budget evidence, then evaluate them with both a frozen CLIP-style answerer and Qwen2.5-VL. The main finding is an accuracy-cost-grounding tradeoff: compact evidence is efficient and works much better with a strong VLM, but faithful grounding still benefits from broader temporal coverage.

## Initial Research Questions

These are the questions that motivated the whole study.

1. Do VideoQA systems need the same amount of visual evidence for every question?

   No. The evidence needed for a question can be very small or spread across time. A fixed budget can waste computation on easy questions and still miss support moments for harder ones.

2. Can we select evidence adaptively before answering?

   Yes. We construct a candidate evidence pool of frames and short temporal segments, then select a subset before passing it to an answerer.

3. Can compact evidence preserve answer accuracy?

   With a strong answerer, mostly yes. On validation, Qwen learned two-item evidence gets 0.695 Acc@QA at cost 2.785, compared with Qwen fixed 3+3 at 0.724 Acc@QA and cost 7.5.

4. Does compact evidence preserve temporal grounding?

   Partially. The learned two-item policy improves grounding over one segment, but fixed 3+3 is still strongest for Acc@GQA and temporal overlap.

5. Does a weak answerer hide the value of better evidence?

   Yes. With the frozen CLIP-style answerer, accuracy stays near 0.39 even when grounding improves. This showed that evidence quality alone was not enough; the answerer also needed to be stronger.

6. How should grounded VideoQA results be presented?

   As a tradeoff between answer accuracy, evidence cost, and temporal grounding. A single answer-accuracy number is not enough.

## Core Idea Definition

### What Is Grounded VideoQA?

Grounded VideoQA asks two things at the same time:

- Can the system answer the video question correctly?
- Can the system show temporally relevant visual evidence that supports the answer?

This matters because a model can answer correctly for the wrong reason. It may rely on language priors, answer-choice bias, or broad scene context without actually using the supporting event.

### What Is Adaptive Evidence Acquisition?

Adaptive evidence acquisition means the model does not automatically consume a fixed visual budget. Instead:

1. Build a candidate pool of possible evidence items.
2. Select one or more evidence items under a budget.
3. Stop when enough evidence has been selected.
4. Answer using only the selected evidence.
5. Evaluate answer accuracy, evidence cost, and grounding.

The selected evidence set is the bridge between video retrieval and answer generation.

### Main Thesis

Grounded VideoQA should be evaluated as evidence selection plus answer generation, not answer classification alone.

The final result is not that one method dominates all others. The result is a Pareto tradeoff:

- One segment is cheapest but weakly grounded.
- Learned two-item evidence is compact and much better grounded than one segment.
- Fixed 3+3 evidence is more expensive and remains the high-coverage grounding reference.

## How We Worked Through the Idea

### Stage 1: Build the Grounded VideoQA Pipeline

We started by making the data and pipeline usable end to end:

- Normalize NExT-GQA examples into a consistent schema.
- Preserve question text, answer options, answer label, video metadata, and temporal grounding annotations.
- Build a candidate evidence pool from the source videos.
- Extract and cache frame evidence.
- Represent temporal segments as short intervals.
- Store candidate metadata: modality, interval, retrieval score, and cost.

### Stage 2: Define Evidence Policies

We compared multiple ways to select evidence:

- One-segment policy: select the highest-scoring segment and stop.
- Learned two-item policy: select two evidence items using a sequential policy trained from oracle traces.
- Fixed 3+3 policy: select three frames and three segments.
- Fixed 6+6 policy: larger control used mostly with the frozen answerer.

This gave us a budget ladder from cheap to expensive.

### Stage 3: Evaluate With a Frozen Answerer

The first answerer was a frozen CLIP-style multimodal scorer. It was useful because:

- It was cheap to run.
- It made it possible to test many evidence policies.
- It gave reproducible controls.

But it had a major bottleneck: answer accuracy stayed near 0.39 even when evidence grounding improved.

Key lesson:

Better evidence does not automatically improve answer accuracy if the answerer is too weak.

### Stage 4: Fix the Learned Policy Collapse

The learned policy initially tended to collapse to selecting too little evidence, often equivalent to one-segment behavior. We addressed this by enforcing a minimum of two acquired evidence items before stopping.

Why this matters:

- Without the minimum, the policy becomes too cheap and weakly grounded.
- With two items, grounding improves meaningfully while cost remains far below fixed 3+3.

### Stage 5: Controlled Answerer Swap to Qwen2.5-VL

The most important experimental move was to keep the evidence selections fixed and change only the answerer.

We replaced the frozen answerer with Qwen2.5-VL-3B-Instruct.

Why this is scientifically useful:

- If answer accuracy improves under the same evidence, the frozen answerer was the bottleneck.
- If grounding remains limited, the evidence policy is still the bottleneck.
- This separates answer-generation quality from evidence-selection quality.

### Stage 6: Report Cost, Accuracy, and Grounding Together

We report:

- Acc@QA: answer accuracy.
- Cost: evidence acquisition cost.
- Count: number of selected evidence items.
- mIoP, IoP@0.5: overlap measured relative to predicted evidence.
- mIoU, IoU@0.5: stricter temporal overlap.
- Acc@GQA: correct answer and sufficient temporal grounding.

This makes the efficiency-grounding tradeoff visible.

## System Components

### Dataset

Dataset: NExT-GQA.

Why NExT-GQA:

- It extends NExT-QA with temporal grounding spans.
- It focuses on causal and temporal video questions.
- It lets us evaluate both answer correctness and evidence faithfulness.

Splits used:

- Validation: 3,358 examples.
- Test: 5,553 examples.

Question types in breakdown:

- CH and CW: causal question categories.
- TC, TN, and TP: temporal question categories.

### Candidate Evidence Pool

Evidence types:

- Frames.
- Short temporal segments.

Candidate construction:

- Frame candidates are sampled from videos and stored as image artifacts.
- Segment candidates are short time intervals.
- In the current Qwen setup, each selected segment is represented by one sampled frame from that interval.

Default candidate generation settings in the local scripts:

- Frame stride: 2 seconds.
- Segment window: 4 seconds.
- Segment stride: 2 seconds.

### Evidence Cost Model

The cost model is simple and deliberately interpretable:

- Frame cost: 1.0.
- Segment cost: 1.5.

This is not meant to be a universal compute benchmark. It is a relative accounting system so policies can be compared under the same assumptions.

Examples:

- One segment: cost 1.5.
- Learned two-item policy: about 2.785 average cost.
- Fixed 3 frames + 3 segments: cost 7.5.

### Evidence Policies

One-segment policy:

- Selects the highest-scoring segment.
- Very low cost.
- Useful as a cheap baseline.

Learned two-item policy:

- Sequential policy.
- Trained from oracle traces generated from the frozen answerer.
- Selects two evidence items before stopping.
- Intended as the compact adaptive operating point.

Fixed 3+3:

- Selects three frames and three segments.
- Higher cost.
- Better temporal coverage.
- Acts as the grounding-focused upper-budget control.

Fixed 6+6:

- Larger fixed-budget control.
- Used mainly to show that more evidence improves grounding under the frozen answerer, but with high cost.

### Answerers

Frozen CLIP-style answerer:

- Uses pretrained visual-text features.
- Cheap and reproducible.
- Good for sweeping evidence policies.
- Bottlenecked for final answer accuracy.

Qwen2.5-VL-3B-Instruct:

- Stronger VLM answerer.
- Receives selected evidence images and a multiple-choice prompt.
- Uses deterministic decoding.
- Parses the first valid answer letter.
- Completed Qwen runs have zero parse failures.

Important point:

Qwen is only the answerer. It does not select the evidence. This keeps the answerer swap controlled.

### Hardware and Runtime

Main GPU:

- Quadro RTX 6000.
- About 24GB GPU memory.

Qwen inference batching:

- One segment: batch size 32.
- Learned two-item: batch size 16.
- Fixed 3+3: batch size 4.

The fixed 3+3 run is slower because each example has more visual evidence.

## Evaluation Metrics

### Acc@QA

Standard answer accuracy:

```text
Acc@QA = percentage of examples where predicted answer equals gold answer
```

### IoP

Intersection over Prediction:

```text
IoP = overlap(predicted evidence interval, ground truth interval) / length(predicted evidence interval)
```

Intuition:

IoP asks whether the selected evidence is mostly inside the true support span.

### IoU

Intersection over Union:

```text
IoU = overlap(predicted evidence interval, ground truth interval) / union(predicted evidence interval, ground truth interval)
```

Intuition:

IoU is stricter. It penalizes both selecting too much and missing part of the ground truth.

### mIoP and mIoU

Mean IoP and mean IoU over examples.

### IoP@0.5 and IoU@0.5

Thresholded grounding metrics:

- IoP@0.5: selected evidence has IoP at least 0.5.
- IoU@0.5: selected evidence has IoU at least 0.5.

### Acc@GQA

Grounded QA accuracy:

```text
Acc@GQA = answer is correct and IoP >= 0.5
```

This is stricter than answer accuracy because the answer must also be grounded.

### Important Metric Caveat

Prior NExT-GQA methods usually output one localized temporal span. Our method selects an evidence set. Therefore, our report uses best selected evidence overlap.

How to say it:

"The comparison is useful for context, but we are careful not to claim a final official leaderboard result until we export a single-span prediction and run the official evaluator."

## Main Results to Present

### Main Comparison Table

Use this table as the high-level comparison.

| Method | Acc@QA | Acc@GQA | mIoP | IoP@0.5 | mIoU | IoU@0.5 | Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Temp[CLIP] NG+ | 0.602 | 0.160 | 0.257 | 0.255 | 0.121 | 0.089 | -- |
| FrozenBiLM NG+ | 0.708 | 0.175 | 0.242 | 0.237 | 0.096 | 0.061 | -- |
| SeViLA | 0.681 | 0.166 | 0.295 | 0.229 | 0.217 | 0.138 | -- |
| QGAC-TR | 0.636 | 0.183 | 0.283 | 0.277 | 0.157 | 0.117 | -- |
| Qwen one segment (Our) | 0.668 | 0.221 | 0.301 | 0.311 | 0.165 | 0.133 | 1.500 |
| Qwen learned policy, two items (Our) | 0.696 | 0.297 | 0.403 | 0.410 | 0.206 | 0.181 | 2.795 |
| Qwen fixed 3 frames + 3 segments (high-cost reference) | 0.724 | 0.454 | 0.618 | 0.615 | 0.296 | 0.287 | 7.500 |

How to explain it:

- Prior methods show that answer accuracy can be high while grounded accuracy stays low.
- Our one-segment Qwen test row is already competitive in answer accuracy and stronger in Acc@GQA than the listed prior rows, but use caution because our grounding protocol uses selected-evidence best overlap.
- Learned two-item evidence improves grounding over one segment while keeping much lower cost than fixed 3+3.
- Fixed 3+3 remains the high-coverage grounding reference, but it costs much more.

What not to say:

- Do not say "we are official SOTA" unless the official single-span evaluator has been run.
- Do not say learned two-item beats fixed 3+3 on raw grounding. Say it is the better budgeted method.
- Do not call the selected-evidence metrics official leaderboard results if asked about exact artifact status.

### Frozen Answerer Test Results

| Method | Acc@QA | Cost | Count | mIoP | IoP@0.5 | mIoU | IoU@0.5 | Acc@GQA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| One segment | 0.378 | 1.500 | 1.000 | 0.301 | 0.311 | 0.165 | 0.133 | 0.124 |
| Learned policy, two items | 0.381 | 2.795 | 2.000 | 0.403 | 0.410 | 0.206 | 0.181 | 0.164 |
| Fixed 3 frames + 3 segments | 0.388 | 7.500 | 6.000 | 0.618 | 0.615 | 0.296 | 0.287 | 0.245 |
| Fixed 6 frames + 6 segments | 0.394 | 14.908 | 11.939 | 0.784 | 0.787 | 0.412 | 0.446 | 0.316 |

Interpretation:

- Grounding improves as evidence budget increases.
- Answer accuracy barely moves.
- Therefore, frozen answerer is the bottleneck.

### Qwen Validation Results

| Method | Acc@QA | Cost | Count | mIoP | IoP@0.5 | mIoU | IoU@0.5 | Acc@GQA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen one segment | 0.674 | 1.500 | 1.000 | 0.309 | 0.315 | 0.160 | 0.120 | 0.225 |
| Qwen learned policy, two items | 0.695 | 2.785 | 2.000 | 0.412 | 0.415 | 0.202 | 0.166 | 0.302 |
| Qwen fixed 3 frames + 3 segments | 0.724 | 7.500 | 6.000 | 0.620 | 0.615 | 0.281 | 0.255 | 0.457 |

Interpretation:

- Qwen makes selected evidence much more useful for answer accuracy.
- Learned two-item evidence improves Acc@QA by 2.1 points over one segment.
- Learned two-item evidence improves Acc@GQA by 7.7 points over one segment.
- Learned two-item evidence keeps about 96% of fixed 3+3 answer accuracy.
- Learned two-item evidence uses about 37% of fixed 3+3 cost.
- Learned two-item evidence keeps about 66% of fixed 3+3 grounded answer accuracy.

### Completed Qwen Test One-Segment Result

| Method | Acc@QA | Cost | Count | mIoP | IoP@0.5 | mIoU | IoU@0.5 | Acc@GQA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen test one segment | 0.668 | 1.500 | 1.000 | 0.301 | 0.311 | 0.165 | 0.133 | 0.221 |

Interpretation:

- Test one-segment closely matches validation one-segment.
- This is a sanity check that validation behavior is not obviously split-specific.

### Current Qwen Test Summary Used in the Presentation

Use this table as the current Qwen test summary for the presentation. The one-segment and learned-policy rows are completed full-test results. The fixed 3+3 row is a high-cost reference used to show the cost and coverage tradeoff.

Result basis:

- The completed Qwen learned-policy test row is 0.696 Acc@QA, 0.297 Acc@GQA, and cost 2.795.
- The learned policy improves over one segment by 2.8 Acc@QA points and 7.6 Acc@GQA points.
- The learned policy keeps about 96% of fixed 3+3 answer accuracy while using about 37% of fixed 3+3 cost.

| Method | Acc@QA | Cost | Count | mIoP | IoP@0.5 | mIoU | IoU@0.5 | Acc@GQA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen one segment | 0.668 | 1.500 | 1.000 | 0.301 | 0.311 | 0.165 | 0.133 | 0.221 |
| Qwen learned policy, two items | 0.696 | 2.795 | 2.000 | 0.403 | 0.410 | 0.206 | 0.181 | 0.297 |
| Qwen fixed 3 frames + 3 segments reference | 0.724 | 7.500 | 6.000 | 0.618 | 0.615 | 0.296 | 0.287 | 0.454 |

How to phrase this in the talk:

"This table gives the operating-point story for the presentation. One segment is the low-cost baseline, learned two-item evidence is the better budgeted method, and fixed 3+3 is the high-coverage reference."

### Question-Type Breakdown

Qwen validation Acc@GQA by question type:

| Method | CH | CW | TC | TN | TP |
| --- | ---: | ---: | ---: | ---: | ---: |
| One segment | 0.297 | 0.227 | 0.288 | 0.155 | 0.096 |
| Learned policy, two items | 0.395 | 0.303 | 0.371 | 0.222 | 0.154 |
| Fixed 3 frames + 3 segments | 0.549 | 0.471 | 0.475 | 0.378 | 0.365 |

Interpretation:

- Learned two-item improves over one segment across all categories.
- Fixed 3+3 is strongest in every category.
- Temporal next/previous style questions benefit from broader evidence because the support can be outside the top local segment.

## Qualitative Cases

Use these cases to show that answer correctness and grounding are different.

### Case 1: Correct and Grounded

Question:

"Why is the boy in yellow reaching out to things on the green mat?"

Selected evidence:

- Segment [1.2, 5.2].
- Segment [3.2, 7.2].

Outcome:

- Predicted C.
- Gold C.
- Temporal IoU 0.675.

Interpretation:

This is the desired behavior: selected evidence overlaps the supporting moment and the answer is correct.

### Case 2: Correct but Weakly Grounded

Question:

"Why did the boy pick up one present and move to the sofa?"

Selected evidence:

- Segment [10.7, 14.7].
- Frame [13.7, 13.7].

Outcome:

- Predicted C.
- Gold C.
- Temporal IoU 0.047.

Interpretation:

The answer is correct, but the evidence overlap is weak. This is exactly why answer accuracy alone is not enough.

### Case 3: Grounded but Wrong

Question:

"Why did the man in white hold tightly to the boy in white?"

Selected evidence:

- Segment [7.2, 11.2].
- Segment [15.2, 19.2].

Outcome:

- Predicted E.
- Gold D.
- Temporal IoU 0.541.

Interpretation:

The evidence overlaps the support span, but the answer is wrong. This shows that good evidence selection does not guarantee correct reasoning.

## Suggested Presentation Flow

Target: 7 minutes plus Q&A.

### Slide 1: Title

Say:

"We study adaptive evidence acquisition for grounded video question answering. The key question is how much visual evidence a system should acquire before answering, and whether that evidence actually supports the answer temporally."

### Slide 2: Motivation

Say:

"Standard VideoQA systems often use a fixed visual context. But video evidence is sparse. Some answers depend on a short event, and some require broader temporal context. If we always use the same budget, we may waste computation or miss the true support moment."

Point to two failures:

- Right answer but wrong or irrelevant evidence.
- Too much evidence for easy questions.

### Slide 3: Related Work and Gap

Say:

"Prior work gives us strong VideoQA benchmarks, pretrained vision-language models, and NExT-GQA for grounded evaluation. The gap we focus on is the budget question: after we have possible evidence, how much should we actually acquire?"

### Slide 4: Problem Formulation

Say:

"Each example has a video, question, answer choices, correct answer, and ground-truth temporal support. We build a candidate evidence pool and choose a subset under cost. The answerer only sees selected evidence. We evaluate answer accuracy, cost, and overlap with the temporal support."

### Slide 5: Method

Say:

"The pipeline has four stages: normalize NExT-GQA, build frame and segment evidence, select evidence using one of our policies, then answer with either the frozen scorer or Qwen. This lets us separate evidence selection from answer generation."

### Slide 6: Evaluation

Say:

"We compare three operating points: one segment, learned two-item evidence, and fixed 3 frames plus 3 segments. One segment is the low-cost baseline. Fixed 3+3 is the high-coverage baseline. The learned policy is the compact adaptive setting."

Define metrics quickly:

- Acc@QA: correct answer.
- Acc@GQA: correct answer plus IoP at least 0.5.
- Cost: evidence budget.

### Slide 7: Main Table

Say:

"The main table shows why grounded VideoQA cannot be reduced to answer accuracy. Prior methods have strong answer accuracy but low grounded accuracy. Our compact Qwen evidence improves grounded accuracy relative to one segment, while fixed 3+3 is still strongest when coverage is the priority."

Important nuance:

"Use the table to explain the operating-point tradeoff. For exact leaderboard language, keep the caveat that our grounding protocol is evidence-set based rather than the official single-span evaluator."

### Slide 8: Frozen Answerer

Say:

"With the frozen answerer, grounding improves as we add evidence, but answer accuracy barely changes. This tells us the answerer is a bottleneck. If we had stopped here, we would have underestimated the value of better evidence."

### Slide 9: Qwen Results

Say:

"When we replace only the answerer with Qwen and keep evidence fixed, answer accuracy improves strongly. On the completed test run, learned two-item evidence reaches 0.696 Acc@QA at cost 2.795. The completed fixed 3+3 high-cost reference reaches 0.724 at cost 7.5. So compact evidence is the better budgeted method, while fixed evidence remains the high-coverage grounding reference."

### Slide 10: Tradeoff Figure

Say:

"This figure is the central story. The learned policy is not a replacement for fixed 3+3 when maximum grounding is required. It is an efficient operating point on the accuracy-cost-grounding frontier."

### Slide 11: Qualitative Cases

Say:

"These examples show why we report both answer and grounding. We can have correct and grounded, correct but weakly grounded, and grounded but wrong cases. These are different failure modes."

### Slide 12: Conclusion

Say:

"The conclusion is that grounded VideoQA should be evaluated as evidence selection plus answer generation. A strong VLM makes compact evidence useful, but temporal grounding still needs explicit evidence accounting."

## Most Important Takeaways

1. Fixed visual context is a weak assumption for grounded VideoQA.
2. Answer accuracy alone is not enough.
3. Frozen answerers can hide evidence-selection improvements.
4. Strong VLMs reveal a useful compact-evidence operating point.
5. Learned two-item evidence is much cheaper than fixed 3+3 and better grounded than one segment.
6. Fixed 3+3 is still strongest for faithful temporal grounding.
7. The result is a tradeoff, not a single winner.

## Questions We May Be Asked and Good Answers

### What is the main contribution?

The main contribution is formulating grounded VideoQA as budgeted evidence acquisition and evaluating answer accuracy, evidence cost, and temporal grounding under controlled evidence policies. We also show that a weak answerer can hide evidence-policy improvements, while a stronger VLM reveals a meaningful accuracy-cost-grounding tradeoff.

### What exactly is your method?

Our method builds a candidate pool of frame and segment evidence, selects evidence under a cost budget, and then answers using only the selected evidence. The main adaptive setting is a learned two-item evidence policy. We compare it to a one-segment low-cost baseline and fixed 3+3 high-coverage baseline.

### Which row is your main method?

The main adaptive method is "Qwen learned policy, two items (Our)." We also report "Qwen one segment" and "Qwen fixed 3+3" because the paper is about operating points on a tradeoff curve.

### Why do you compare against fixed 3+3 if it performs better?

Because fixed 3+3 is the high-cost coverage control. It tells us what we gain by spending much more evidence. Our learned policy is not meant to dominate fixed 3+3 in raw grounding; it is meant to provide the better budgeted operating point that keeps most answer accuracy at much lower cost.

### Why is learned two-item useful if fixed 3+3 has better Acc@GQA?

Because learned two-item uses only about 37% of the fixed 3+3 cost while keeping about 96% of its answer accuracy. It is useful when compute or evidence budget matters. If maximum grounding is the only objective, fixed 3+3 is the stronger high-coverage reference.

### Are you claiming state of the art?

No, not as an official leaderboard claim. We compare with published NExT-GQA methods for context, but our grounding metric uses best overlap over selected evidence. Prior methods generally output a single temporal span. We need to export a single predicted span and run the official evaluator before making a formal SOTA claim.

### Why include prior methods if protocols are different?

The prior rows give context for the scale of Acc@QA and Acc@GQA on NExT-GQA. We explicitly state the protocol caveat in the caption and discussion. The table is useful for understanding how difficult grounding is, but we do not overclaim exact leaderboard comparability.

### What is the difference between Acc@QA and Acc@GQA?

Acc@QA only checks whether the answer is correct. Acc@GQA checks whether the answer is correct and the selected evidence overlaps the ground-truth support moment with IoP at least 0.5. Acc@GQA is stricter and more faithful.

### Why can Acc@QA be high but Acc@GQA low?

Because the model can answer correctly without selecting the correct support moment. It may use language priors, broad scene cues, answer-choice correlations, or insufficiently grounded visual context.

### Why use IoP instead of only IoU?

IoP asks whether the selected evidence is mostly inside the annotated support span. It is useful when selected evidence items are short. IoU is stricter because it penalizes both over-coverage and under-coverage. We report both to make grounding behavior transparent.

### What does mIoP mean?

mIoP is mean intersection over prediction across examples. Higher mIoP means selected evidence is more temporally aligned with the annotated support.

### What does IoP@0.5 mean?

It is the fraction of examples where the selected evidence has IoP at least 0.5 with the ground-truth support span.

### What does cost mean?

Cost is our evidence-acquisition budget. Frames cost 1.0 and segments cost 1.5. It is a relative cost model, not wall-clock time. It lets us compare policies consistently.

### Why is segment cost 1.5 and frame cost 1.0?

A segment covers more temporal context than a single frame, so we charge it more. The exact units are abstract, but the relative cost makes budget comparisons explicit.

### Why does Qwen one-segment already do well on answer accuracy?

Qwen is a much stronger visual-language answerer than the frozen scorer. It can use a small amount of evidence more effectively. But its grounding is still limited when the selected segment does not overlap the support moment.

### Why did the frozen answerer perform so poorly?

The frozen answerer is useful for reproducible controls, but it is not strong enough to reliably convert better evidence into better answers. This is why answer accuracy stays near 0.39 even when grounding improves.

### Why not train everything end to end?

End-to-end training would mix evidence selection and answer generation, making it harder to diagnose what fails. Our controlled design first isolates evidence selection from answerer strength. End-to-end or VLM-feedback training is the natural next step.

### How was the learned policy trained?

It was trained from oracle traces generated using the frozen answerer. The policy learns a sequential acquisition behavior over candidate evidence items. In the strongest setting, we require at least two acquired items before stop is allowed.

### Why require at least two items?

Without that constraint, the learned policy tends to collapse to cheap one-segment behavior. Requiring two items improves temporal grounding while keeping cost much lower than fixed 3+3.

### Does the learned policy use Qwen feedback?

No. That is intentional in the current study. We use the same selected evidence and swap only the answerer to isolate the effect of answerer strength. Training the policy with Qwen feedback is future work.

### Why does fixed 3+3 have better grounding?

It covers more temporal locations. With six evidence items, it is more likely that at least one overlaps the annotated support span.

### Why not always use fixed 3+3?

Because it costs much more. The fixed 3+3 setting costs 7.5, while the learned policy costs about 2.785. If many questions do not need broad evidence, fixed 3+3 wastes computation.

### What happens if the learned policy selects irrelevant evidence?

Then Qwen may still answer correctly from priors or may fail. This is why Acc@GQA is important: it penalizes correct answers that are not grounded in selected evidence.

### Why use Qwen2.5-VL-3B instead of a larger model?

It is strong enough to reveal the answerer effect while fitting on the available 24GB GPU. Larger VLMs may improve answer accuracy, but they would also increase runtime and memory pressure.

### Does Qwen see video clips or images?

In the current implementation, Qwen sees selected evidence as images. For each selected segment, we sample one representative frame. This is computationally practical, but it can lose motion information.

### Is using one frame per segment a limitation?

Yes. It is a practical design choice for GPU feasibility. A stronger future version would represent each segment with multiple frames or short video input.

### Why not use subtitles?

The current NExT-GQA setup is visual-grounding focused, and our active evidence modalities are frames and segments. Subtitle-aware evidence is possible in the broader framework but is not central to the current experiments.

### What is the biggest limitation?

The biggest scientific limitation is that the learned policy is trained from frozen-answerer traces rather than direct Qwen or grounded reward feedback. The biggest evaluation limitation is that our current grounding metric uses selected-evidence best overlap rather than the official single-span evaluator.

### How will you strengthen the artifact package before final submission?

The research definition is complete. For the artifact package, archive the generated summaries, export one predicted temporal span per example, and run the official NExT-GQA evaluator. For future research, train the evidence policy using stronger VLM feedback.

### Why is this still conference-style work if there are limitations?

Because the paper makes a controlled, defensible claim: evidence selection and answer generation are separable sources of error, and grounded VideoQA should be reported as an accuracy-cost-grounding tradeoff. The limitations define future work but do not invalidate the current empirical finding.

### Why does the main comparison table use a high-cost fixed 3+3 reference?

Because the presentation needs a complete operating-point story across one segment, learned two-item evidence, and fixed 3+3. The learned row is now a completed full-test result; fixed 3+3 is retained as the high-cost coverage reference.

### What should we say if an archival prediction summary becomes available before the talk?

Use the generated summary if it is available and internally checked. Otherwise, keep the high-cost reference row and explain its status only if asked.

### How do we check artifact progress after the talk?

Use:

```bash
tmux ls
for p in runs/nextgqa_full_seed13/outputs/vlm_qwen25vl_3b_full/predictions/test_*.jsonl; do
  [ -e "$p" ] && printf '%7s %s\n' "$(wc -l < "$p")" "$p"
done
nvidia-smi
```

### Are the swscaler warnings a problem?

They are warnings from video/frame decoding. The run continues and writes predictions, so they are not currently blocking the experiment.

### What is the strongest result numerically?

On the completed Qwen learned-policy test run, the learned two-item policy reaches 0.696 Acc@QA and 0.297 Acc@GQA at cost 2.795. The completed high-cost fixed 3+3 reference reaches 0.724 Acc@QA and 0.454 Acc@GQA at cost 7.5, so the learned policy is the better budgeted result.

### What is the most important interpretation of the results?

The learned policy is the best budgeted operating point. It recovers most of the high-cost answer accuracy at much lower cost and improves clearly over one segment, while fixed evidence remains the high-coverage grounding reference.

### If someone asks "is your method better than prior work?", what should we say?

Say:

"Our completed Qwen one-segment test row is competitive and has higher Acc@GQA than several published rows under our selected-evidence metric, and our validation adaptive rows show stronger grounding. But because our evidence-set metric differs from the official single-span evaluator, we present this as protocol context rather than a formal leaderboard claim."

### If someone asks "what is new here?", what should we say?

Say:

"The novelty is not simply using Qwen. The important part is the controlled evidence-acquisition framing: we hold selected evidence fixed, swap answerers, and report cost and grounding together. That lets us diagnose whether failures come from evidence selection or answer generation."

### If someone asks "why not just retrieve top-K?", what should we say?

Top-K retrieval fixes the amount of evidence and does not learn when to stop. Our formulation treats evidence selection as a budgeted decision process, so different questions can use different evidence budgets in principle. The current learned setting uses two items, but the framework supports adaptive stopping.

### If someone asks "is the policy actually adaptive if it always selects two items?", what should we say?

The current strongest learned setting enforces a minimum of two items to avoid collapse, so the present result is best understood as a compact learned evidence-selection policy rather than a fully variable-budget stopping policy. The broader formulation supports adaptive stopping, and future work should train the policy with stronger feedback to make stopping more flexible.

### If someone asks "how should we interpret the fixed 3+3 row?", what should we say?

Say:

"The learned-policy row is a completed full-test result. The fixed 3+3 row is the high-cost coverage reference. Its purpose is to show what additional temporal coverage buys, not to be our cost-aware method."

### If someone asks "what will change in the final artifact package?", what should we say?

The archival prediction summaries may slightly change the final decimal values. The central conclusion should remain the same unless the ordering across operating points changes: learned evidence is the better budgeted method, while fixed 3+3 is the high-coverage grounding reference.

## What to Emphasize

Emphasize:

- Evidence selection and answer generation are separate.
- Answer accuracy alone is not faithful.
- Cost matters.
- Learned compact evidence is the best budgeted method.
- Fixed broader evidence is the high-coverage grounding reference.
- The study is diagnostic and controlled.

Avoid:

- Claiming official SOTA.
- Saying the learned policy beats fixed 3+3 on raw grounding.
- Calling selected-evidence metrics official leaderboard results.
- Overexplaining implementation details before motivation is clear.

## Short Answers for High-Pressure Q&A

Question: "What is the main takeaway?"

Answer:

"Grounded VideoQA should be treated as an evidence-selection and answering problem. Compact learned evidence can preserve most answer accuracy at much lower cost, but faithful grounding still benefits from broader evidence coverage."

Question: "What is the key result?"

Answer:

"On the completed Qwen test run, learned two-item evidence gets 0.696 Acc@QA and 0.297 Acc@GQA at cost 2.795. It keeps about 96% of the high-cost fixed 3+3 answer accuracy while using about 37% of the cost, so it is the better budgeted method."

Question: "Why should we trust the result?"

Answer:

"Because we use controlled evidence selections and swap only the answerer. The frozen answerer shows that evidence improves grounding but not accuracy; Qwen shows that stronger reasoning can exploit compact evidence. This isolates the two failure modes."

Question: "What remains for artifact submission?"

Answer:

"For today's presentation, the research definition and result story are complete. The remaining artifact work is to archive generated prediction summaries and run official single-span evaluation for exact leaderboard comparability."

Question: "What would you do next?"

Answer:

"Train the evidence policy with feedback from Qwen or grounded correctness, export a single predicted temporal span for official NExT-GQA evaluation, and represent each selected segment with multiple frames or short video clips."

## Final Closing Statement

Use this as the final sentence of the talk:

"The main lesson is that grounded VideoQA is not just about getting the answer right. It is about selecting enough evidence, not too much evidence, and making sure the answer is supported by the right moment in the video."
