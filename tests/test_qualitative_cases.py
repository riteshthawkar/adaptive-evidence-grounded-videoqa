from scripts.select_qualitative_cases import select_cases


def record(example_id, *, correct, cost, iou):
    return {
        "example_id": example_id,
        "video_id": "video",
        "gold_index": 0,
        "predicted_index": 0 if correct else 1,
        "correct": 1.0 if correct else 0.0,
        "prediction_confidence": 0.8,
        "selected_evidence_cost": cost,
        "selected_evidence_count": 1,
        "selected_temporal_iou": iou,
        "selected_evidence": [],
    }


def test_select_qualitative_cases_buckets_tradeoffs() -> None:
    fixed_records = [
        record("grounded", correct=True, cost=7.5, iou=0.5),
        record("tradeoff", correct=True, cost=7.5, iou=0.6),
        record("failure", correct=True, cost=7.5, iou=0.5),
        record("fix", correct=False, cost=7.5, iou=0.1),
    ]
    policy_records = [
        record("grounded", correct=True, cost=1.5, iou=0.48),
        record("tradeoff", correct=True, cost=1.5, iou=0.1),
        record("failure", correct=False, cost=1.5, iou=0.1),
        record("fix", correct=True, cost=1.5, iou=0.2),
    ]

    result = select_cases(
        fixed_records,
        policy_records,
        limit_per_bucket=2,
        min_cost_saving=1.0,
        min_iou_drop=0.05,
    )

    assert result["num_joined_records"] == 4
    assert result["buckets"]["efficient_and_grounded"][0]["example_id"] == "grounded"
    assert result["buckets"]["efficient_but_less_grounded"][0]["example_id"] == "tradeoff"
    assert result["buckets"]["policy_saves_cost_but_fails"][0]["example_id"] == "failure"
    assert result["buckets"]["policy_fixes_fixed_budget"][0]["example_id"] == "fix"
