# 8-Minute Speaker Notes

Use these notes to keep the 15-slide deck within roughly 8 minutes. The deck is structured as: questions first, work description in the middle, direct answers at the end.

| Slide | Time | Main point |
| --- | ---: | --- |
| 1. Title | 20s | State the problem: how much evidence is enough for grounded VideoQA? |
| 2. Questions | 35s | Read the three research questions and promise to answer them at the end. |
| 3. Motivation | 35s | Explain why correct answers can still be ungrounded. |
| 4. Related Work | 35s | Position the work against VideoQA, grounded VideoQA, VLMs, and retrieval. |
| 5. Formulation | 40s | Define evidence pool, selected evidence, cost, answerer, and grounding. |
| 6. Pipeline | 40s | Walk left to right: normalize data, build evidence, select, answer, evaluate. |
| 7. Policies | 35s | Explain one segment, learned two-item, fixed 3+3, and why each exists. |
| 8. Setup | 35s | State dataset, answerers, and the key metrics. |
| 9. Main Comparison | 40s | Emphasize that Acc@QA and Acc@GQA diverge. |
| 10. Frozen Answerer | 40s | Grounding improves, but answer accuracy stays flat. This diagnoses answerer bottleneck. |
| 11. Qwen Summary | 45s | Strong VLM confirms learned two-item evidence as the best budgeted operating point. |
| 12. Tradeoff Figure | 40s | Explain the frontier: learned is the cost-aware method, fixed 3+3 is the high-coverage reference. |
| 13. Qualitative Lessons | 35s | Correct/grounded, correct/weakly grounded, grounded/wrong are different failures. |
| 14. Answers | 45s | Directly answer the three opening questions. |
| 15. Takeaway | 25s | Close with evidence selection plus answer generation as the core message. |

Target total: about 8 minutes. If time is short, spend less time on Slides 4, 8, and 13; do not skip Slides 2, 11, 12, or 14.
