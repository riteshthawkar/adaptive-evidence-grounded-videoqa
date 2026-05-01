# NExT-GQA Router And Qwen Demo

This demo records one concrete NExT-GQA test example evaluated with the `Qwen MLP+NMS` evidence selector. It includes a runnable learned-router selection path and a recorded Qwen answer for the selected evidence.

The runnable part uses:

- router checkpoint: `artifacts/router_mlp_oracle_top2/`
- candidate record: `demo/nextgqa_sample_candidates.jsonl`
- fixed 3+3 evidence pool: `demo/nextgqa_sample_fixed_f3_s3_pool.jsonl`
- CLIP feature bundle: `demo/features/nextgqa_5600915537_9.npz`

Run the router demo from the repository root:

```bash
PYTHONPATH=src:. python scripts/create_followup_evidence_selections.py \
  --candidate-path demo/nextgqa_sample_candidates.jsonl \
  --pool-selection-path demo/nextgqa_sample_fixed_f3_s3_pool.jsonl \
  --output-dir demo/outputs \
  --split sample \
  --router-model-dir artifacts/router_mlp_oracle_top2 \
  --nms-thresholds 0.0
```

The command writes `demo/outputs/sample_router_mlp_top2_nms0p0.jsonl`. It should select:

```text
nextgqa:5600915537:9:segment:19
nextgqa:5600915537:9:segment:16
```

The recorded Qwen answer below requires the source visual inputs and Qwen2.5-VL runtime, so it is provided as a compact expected output rather than as a fully self-contained video-generation artifact.

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

The full prediction JSONL is omitted from version control because full runs produce large prediction and frame-cache artifacts. Aggregate metrics are reported in `docs/report/final_paper.pdf`.
