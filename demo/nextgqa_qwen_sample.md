# NExT-GQA Qwen Evidence-Answering Demo

This demo records one concrete test example from the final `Qwen MLP+NMS (Our)`
run. It is a lightweight sample input/output artifact; full reproduction commands
are in the main README.

## Sample Input

- Example id: `nextgqa:5600915537:9`
- Video id: `1019/5600915537`
- Question type: `TN`
- Ground-truth temporal evidence: `[8.2, 11.8]` seconds and `[33.1, 37.2]` seconds
- Question: `what does the man do after stopping half way`
- Options:
  - A. `exit stage`
  - B. `adjust the tricycle`
  - C. `changes direction`
  - D. `lower the camera on his shirt`
  - E. `jumps to cross the water`

Selected `MLP+NMS` evidence:

| Evidence | Modality | Time span | Cost |
| --- | --- | --- | --- |
| `nextgqa:5600915537:9:segment:19` | segment | `[39.2, 41.7]` seconds | `1.5` |
| `nextgqa:5600915537:9:segment:16` | segment | `[33.2, 37.2]` seconds | `1.5` |

## Sample Output

Qwen2.5-VL output:

```text
C
```

Parsed prediction:

```json
{
  "predicted_index": 2,
  "predicted_answer": "changes direction",
  "gold_index": 2,
  "correct": 1.0,
  "selected_evidence_count": 2,
  "selected_evidence_cost": 3.0,
  "prompt_version": "vlm_mcq_letter_v1"
}
```

The selected evidence overlaps the annotated event span and preserves the correct
answer with two evidence items instead of the six-item fixed 3+3 reference.

Local source prediction file from the full experiment cache:

```text
runs/nextgqa_full_seed13/outputs/followup_strengthening/qwen/predictions/test_router_mlp_top2_nms0p0.jsonl
```

The full prediction JSONL is large and is not required in the lightweight GitHub
artifact package; aggregate metrics are copied to
`docs/artifacts/final_metrics/qwen_test_followup_compare.md`.
