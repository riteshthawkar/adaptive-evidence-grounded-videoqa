# Slide-by-Slide Voiceover Script

Use this as the spoken script for `research_presentation.pptx` or `research_presentation_8min.pptx`. It is written for an approximately 8-minute talk. Speak naturally and use the Q&A section only if asked.

## Slide 1: Title

**Target time: 20 seconds**

Good afternoon. Our work is titled **Adaptive Evidence Acquisition for Grounded Video Question Answering**.

The central question is: when answering a question about a video, how much visual evidence is enough? We argue that a VideoQA system should not only answer correctly; it should also use evidence from the right moment in the video. So we evaluate answer accuracy, evidence cost, and temporal grounding together.

## Slide 2: Three Questions We Answer

**Target time: 30 seconds**

We organize the talk around three questions.

First, do all video questions need the same amount of visual evidence? Second, can a compact learned evidence policy preserve answer accuracy while using less evidence? Third, when performance is weak, is the bottleneck evidence selection or the answerer?

I will define the task and method, show the experiments, and then return to these questions at the end.

## Slide 3: Motivation

**Target time: 30 seconds**

VideoQA is different from image QA because the answer may depend on a short event inside a longer clip. A model can still choose the right option without using the true supporting moment, because of language priors, answer-choice bias, or broad scene context.

That is why grounded VideoQA matters. We ask two things: is the answer correct, and does the selected evidence overlap the annotated support moment? A correct answer with wrong evidence is not truly grounded.

## Slide 4: Related Work and Gap

**Target time: 30 seconds**

This work builds on VideoQA benchmarks such as MovieQA, TGIF-QA, TVQA, and NExT-QA, and grounded benchmarks such as TVQA+ and NExT-GQA.

It also builds on pretrained vision-language systems such as CLIP, FrozenBiLM, SeViLA, and Qwen2.5-VL, plus retrieval-augmented video reasoning.

The gap is that evidence amount, answer accuracy, and temporal grounding are often not evaluated together. We make that tradeoff explicit.

## Slide 5: Task Formulation

**Target time: 35 seconds**

Each example contains a video, question, answer options, the correct answer, and a temporal support span.

From the video, we build an evidence pool. Each evidence item is a frame or short temporal segment with a time interval and acquisition cost. The policy selects a subset of that evidence, and the answerer only sees the selected evidence.

The goal is not just high answer accuracy. We want high answer accuracy, low evidence cost, and strong overlap between selected evidence and the ground-truth support span.

## Slide 6: Method Pipeline

**Target time: 45 seconds**

The pipeline has four main stages.

First, we normalize NExT-GQA examples. Second, we build candidate evidence from frames and sliding short segments. In the final setup, segment candidates use 4-second windows with 2-second stride, while frame candidates come from available frame timestamps or regular sampling.

Third, we rank candidates using CLIP-style similarity between the question-plus-options text and the visual evidence. The policy sees the top ranked frames and segments, not the whole video.

Fourth, the selected evidence is passed to the answerer. We then evaluate accuracy, cost, count, and grounding. The key design is the controlled answerer swap: we keep selected evidence fixed and change only the answerer, so evidence selection and reasoning quality can be diagnosed separately.

## Slide 7: Evidence Policies and Cost Ladder

**Target time: 45 seconds**

We compare three main operating points.

One segment selects the highest-scoring segment and costs 1.5. This is the low-cost baseline.

The learned policy is a lightweight linear router trained from oracle acquisition traces. At each step, it chooses whether to acquire a frame, acquire a segment, or stop. In the final setting, stop is not allowed until at least two evidence items are selected, so this becomes our learned two-item operating point with average cost 2.795.

Fixed 3+3 selects three frames and three segments, with cost 7.5. This is the high-coverage reference. Fixed 6+6 is used only as a larger frozen-answerer control.

## Slide 8: Experimental Setup and Metrics

**Target time: 45 seconds**

We evaluate on NExT-GQA. The validation split has 3,358 examples, and the test split has 5,553 examples.

We use two answerers: a frozen CLIP-style scorer for cheap controlled analysis, and Qwen2.5-VL-3B-Instruct as a stronger vision-language answerer.

The metrics are Acc@QA for answer accuracy, cost and count for evidence usage, IoP and IoU for temporal overlap, and Acc@GQA for grounded answer accuracy.

IoP asks whether the selected evidence lies inside the annotated support moment, which is useful for compact evidence. IoU is stricter and asks how well the selected evidence covers the full support interval. Acc@GQA requires a correct answer and IoP of at least 0.5.

## Slide 9: Main Comparison

**Target time: 35 seconds**

The main comparison shows that answer accuracy and grounded accuracy diverge.

Prior methods can have strong answer accuracy while grounded accuracy remains much lower. Our Qwen one-segment result is the low-cost point. The learned two-item policy improves grounded accuracy while keeping cost low. Fixed 3+3 gives the strongest grounding, but uses much more evidence.

So this is not a single-winner result. It is an accuracy-cost-grounding tradeoff. We treat published-method comparison as context, because our grounding metric uses best overlap over selected evidence rather than the official single-span evaluator.

## Slide 10: Frozen Answerer Result

**Target time: 35 seconds**

The frozen answerer result is diagnostic.

As we increase evidence, grounding improves clearly. IoP@0.5 rises from 0.311 with one segment to 0.787 with fixed 6+6, and Acc@GQA rises from 0.124 to 0.316.

But answer accuracy stays around 0.38 to 0.39. This means the frozen answerer is the bottleneck: better evidence is being selected, but the weak answerer does not translate it into much better answers.

## Slide 11: Qwen Result

**Target time: 45 seconds**

When we switch to Qwen2.5-VL, the tradeoff becomes clearer.

With one segment, Qwen reaches 0.668 Acc@QA and 0.221 Acc@GQA. The completed learned two-item test run improves this to 0.696 Acc@QA and 0.297 Acc@GQA at cost 2.795. Fixed 3+3 reaches 0.724 Acc@QA and 0.454 Acc@GQA, but at cost 7.5.

The learned policy keeps about 96 percent of fixed 3+3 answer accuracy while using about 37 percent of the evidence cost. It is the better budgeted method, while fixed 3+3 remains the high-coverage grounding reference.

## Slide 12: Accuracy-Cost-Grounding Tradeoff

**Target time: 35 seconds**

This figure summarizes the central result.

On answer accuracy, learned two-item evidence is close to fixed 3+3 despite much lower cost. On grounded accuracy, broader evidence still helps, so fixed 3+3 remains strongest.

The learned policy is therefore an efficient operating point, not a replacement for the high-coverage reference. The result is a Pareto tradeoff between accuracy, cost, and grounding.

## Slide 13: Qualitative Lessons

**Target time: 30 seconds**

The qualitative cases show why answer accuracy alone is not enough.

One case is correct and grounded: selected evidence overlaps the support moment and the answer is correct. Another is correct but weakly grounded, showing that answer accuracy can overstate faithfulness. A third is grounded but wrong, showing that good evidence does not guarantee correct reasoning.

Evidence selection and answer generation are separate components.

## Slide 14: Answers to the Opening Questions

**Target time: 35 seconds**

Now we can answer the opening questions.

Do all questions need the same evidence budget? No. Evidence need varies, so fixed context can waste computation or miss support.

Can compact learned evidence preserve answer accuracy? Yes. The completed learned two-item test result keeps about 96 percent of fixed 3+3 answer accuracy while using about 37 percent of the cost.

Is weak performance an evidence problem or an answerer problem? It is both. The frozen answerer hides evidence improvements, while Qwen reveals the accuracy-cost-grounding frontier.

## Slide 15: Final Takeaway

**Target time: 25 seconds**

The final takeaway is that grounded VideoQA should be evaluated as evidence selection plus answer generation.

A strong VLM makes compact evidence more useful, but faithful temporal grounding still requires explicit evidence accounting.

So the goal is to select enough evidence, not too much evidence, and verify that the answer is supported by the right moment in the video.

## If There Is Extra Q&A Time

Use these short answers for likely questions.

**Is learned two-item evidence better than fixed 3+3?**
Under a budgeted objective, yes: it keeps almost all answer accuracy at much lower cost. Fixed 3+3 remains stronger for raw grounding.

**Why use both Acc@QA and Acc@GQA?**
Acc@QA measures answer correctness only. Acc@GQA requires correctness plus temporal support, so it catches correct-but-ungrounded answers.

**What is the main novelty?**
The novelty is not only the framing. We implement a cost-aware evidence-acquisition pipeline with a learned linear router that selects frame or segment evidence before answering, then evaluate accuracy, cost, and grounding under controlled answerer swaps.

**How is the learned policy trained?**
It is trained as a supervised action classifier from oracle traces. The input state includes the question, selected evidence, remaining candidates, retrieval scores, cost, and answerer confidence. The target action is acquire frame, acquire segment, or stop.

**Why use both IoP and IoU?**
IoP rewards compact evidence inside the annotated support span. IoU is stricter and measures coverage of the full support interval. Together they distinguish compact faithful evidence from broad coverage.

**How is the candidate pool built?**
We sample frame candidates and sliding segment candidates, extract CLIP visual features, rank candidates against the question and answer options, and pass the top frames and segments to the policy.

**Why does Qwen help?**
Qwen is a stronger answerer, so it uses compact selected evidence better. But it cannot recover missing temporal evidence.

**What would improve the work next?**
Train the evidence policy with stronger VLM feedback, export a single predicted span for official NExT-GQA evaluation, and represent selected segments with richer multi-frame or video evidence.
