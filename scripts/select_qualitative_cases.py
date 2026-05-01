import argparse
import json
from pathlib import Path

from adaptive_evidence_vqa.data.base import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select qualitative success/failure cases from two prediction JSONL files."
    )
    parser.add_argument("--fixed-predictions", required=True, help="Fixed-budget prediction JSONL.")
    parser.add_argument("--policy-predictions", required=True, help="Policy prediction JSONL.")
    parser.add_argument("--output-json", required=True, help="Output JSON with selected cases.")
    parser.add_argument("--limit-per-bucket", type=int, default=5)
    parser.add_argument(
        "--min-cost-saving",
        type=float,
        default=1.0,
        help="Minimum fixed-policy cost difference used to call a policy example efficient.",
    )
    parser.add_argument(
        "--min-iou-drop",
        type=float,
        default=0.05,
        help="Minimum fixed-policy temporal IoU gap for grounding-tradeoff examples.",
    )
    return parser.parse_args()


def by_example_id(records: list[dict]) -> dict[str, dict]:
    return {str(record["example_id"]): record for record in records}


def numeric(record: dict, key: str, default: float = 0.0) -> float:
    value = record.get(key)
    if value is None:
        return default
    return float(value)


def is_correct(record: dict) -> bool:
    return numeric(record, "correct") >= 0.5


def compact_record(fixed: dict, policy: dict) -> dict:
    return {
        "example_id": fixed["example_id"],
        "video_id": fixed.get("video_id"),
        "gold_index": fixed.get("gold_index"),
        "fixed": {
            "predicted_index": fixed.get("predicted_index"),
            "correct": fixed.get("correct"),
            "confidence": fixed.get("prediction_confidence"),
            "cost": fixed.get("selected_evidence_cost"),
            "count": fixed.get("selected_evidence_count"),
            "temporal_iou": fixed.get("selected_temporal_iou"),
            "selected_evidence": fixed.get("selected_evidence", []),
        },
        "policy": {
            "predicted_index": policy.get("predicted_index"),
            "correct": policy.get("correct"),
            "confidence": policy.get("prediction_confidence"),
            "cost": policy.get("selected_evidence_cost"),
            "count": policy.get("selected_evidence_count"),
            "temporal_iou": policy.get("selected_temporal_iou"),
            "trace": policy.get("trace", []),
            "selected_evidence": policy.get("selected_evidence", []),
        },
    }


def append_case(
    buckets: dict[str, list[dict]],
    bucket_name: str,
    fixed: dict,
    policy: dict,
    limit: int,
) -> None:
    if len(buckets[bucket_name]) >= limit:
        return
    buckets[bucket_name].append(compact_record(fixed, policy))


def select_cases(
    fixed_records: list[dict],
    policy_records: list[dict],
    *,
    limit_per_bucket: int,
    min_cost_saving: float,
    min_iou_drop: float,
) -> dict:
    policy_by_id = by_example_id(policy_records)
    buckets: dict[str, list[dict]] = {
        "efficient_and_grounded": [],
        "efficient_but_less_grounded": [],
        "policy_saves_cost_but_fails": [],
        "policy_fixes_fixed_budget": [],
    }

    joined = []
    for fixed in fixed_records:
        policy = policy_by_id.get(str(fixed["example_id"]))
        if policy is None:
            continue
        fixed_cost = numeric(fixed, "selected_evidence_cost")
        policy_cost = numeric(policy, "selected_evidence_cost")
        fixed_iou = numeric(fixed, "selected_temporal_iou")
        policy_iou = numeric(policy, "selected_temporal_iou")
        joined.append(
            (
                fixed_cost - policy_cost,
                fixed_iou - policy_iou,
                fixed,
                policy,
            )
        )

    joined.sort(key=lambda item: (-item[0], -abs(item[1]), str(item[2]["example_id"])))
    for cost_saving, iou_drop, fixed, policy in joined:
        fixed_correct = is_correct(fixed)
        policy_correct = is_correct(policy)
        if policy_correct and fixed_correct and cost_saving >= min_cost_saving and iou_drop <= min_iou_drop:
            append_case(buckets, "efficient_and_grounded", fixed, policy, limit_per_bucket)
        if policy_correct and fixed_correct and cost_saving >= min_cost_saving and iou_drop > min_iou_drop:
            append_case(buckets, "efficient_but_less_grounded", fixed, policy, limit_per_bucket)
        if fixed_correct and not policy_correct and cost_saving >= min_cost_saving:
            append_case(buckets, "policy_saves_cost_but_fails", fixed, policy, limit_per_bucket)
        if not fixed_correct and policy_correct:
            append_case(buckets, "policy_fixes_fixed_budget", fixed, policy, limit_per_bucket)

    return {
        "num_fixed_records": len(fixed_records),
        "num_policy_records": len(policy_records),
        "num_joined_records": len(joined),
        "selection_config": {
            "limit_per_bucket": limit_per_bucket,
            "min_cost_saving": min_cost_saving,
            "min_iou_drop": min_iou_drop,
        },
        "buckets": buckets,
    }


def main() -> None:
    args = parse_args()
    result = select_cases(
        fixed_records=load_jsonl(args.fixed_predictions),
        policy_records=load_jsonl(args.policy_predictions),
        limit_per_bucket=args.limit_per_bucket,
        min_cost_saving=args.min_cost_saving,
        min_iou_drop=args.min_iou_drop,
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote qualitative cases to {output_path}")


if __name__ == "__main__":
    main()
