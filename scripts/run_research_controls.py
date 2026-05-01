import argparse
import json
from pathlib import Path

from adaptive_evidence_vqa.data.base import save_jsonl
from adaptive_evidence_vqa.data.evidence_records import serialize_evidence
from adaptive_evidence_vqa.data.normalized import load_normalized_examples
from adaptive_evidence_vqa.eval.metrics import (
    accuracy,
    comprehensiveness,
    evidence_cost,
    max_temporal_iou_for_target_spans,
    sufficiency,
    temporal_target_spans,
)
from adaptive_evidence_vqa.models.answerer import Answerer, build_answerer
from adaptive_evidence_vqa.models.oracle import ORACLE_MODES, MinimalEvidenceOracle, OracleConfig
from adaptive_evidence_vqa.models.policy import build_policy
from adaptive_evidence_vqa.retrieval.base import (
    CandidatePoolBuilder,
    FixedBudgetRetriever,
    RetrievalAllocation,
    build_named_retriever,
)
from adaptive_evidence_vqa.schemas import EvidenceItem, ModelPrediction, QuestionExample


DEFAULT_CONTROL_POLICIES = (
    "frame_once",
    "segment_once",
    "frame_segment_once",
    "segment_frame_once",
    "top_once",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run low-budget controls and fixed-budget sweeps for the "
            "adaptive evidence-acquisition study."
        )
    )
    parser.add_argument("--input-path", required=True, help="Path to candidate-pool JSONL.")
    parser.add_argument("--output-dir", required=True, help="Directory for summary outputs.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on examples.")
    parser.add_argument(
        "--write-predictions",
        action="store_true",
        help="Write one JSONL prediction file per method.",
    )
    parser.add_argument(
        "--answerer",
        choices=("lexical", "linear", "frozen_multimodal", "calibrated_multimodal"),
        default="frozen_multimodal",
        help="Answerer used for every evaluated method.",
    )
    parser.add_argument("--answerer-model-dir", help="Model directory for the linear answerer.")
    parser.add_argument(
        "--answerer-model-name",
        default="openai/clip-vit-base-patch32",
        help="Model name for the frozen multimodal answerer.",
    )
    parser.add_argument("--answerer-device", help="Device override for the answerer.")
    parser.add_argument(
        "--retriever",
        choices=("lexical", "bm25", "hybrid_clip"),
        default="hybrid_clip",
        help="Retriever used to construct candidate pools.",
    )
    parser.add_argument(
        "--visual-model-name",
        default="openai/clip-vit-base-patch32",
        help="Visual-text encoder used by hybrid CLIP retrieval.",
    )
    parser.add_argument("--visual-device", help="Device override for hybrid retrieval.")
    parser.add_argument("--subtitle-k", type=int, default=0, help="Main-run subtitle budget.")
    parser.add_argument("--frame-k", type=int, default=3, help="Main-run frame budget.")
    parser.add_argument("--segment-k", type=int, default=3, help="Main-run segment budget.")
    parser.add_argument(
        "--fixed-budget-ks",
        default="1,2,3,4",
        help="Comma-separated K values for fixed frame=K, segment=K sweeps.",
    )
    parser.add_argument(
        "--policies",
        default=",".join(DEFAULT_CONTROL_POLICIES),
        help="Comma-separated control policies to evaluate.",
    )
    parser.add_argument("--max-items", type=int, default=6, help="Maximum acquisitions per policy rollout.")
    parser.add_argument(
        "--oracle-mode",
        choices=ORACLE_MODES,
        default="correctness_plus_sufficiency",
        help="Oracle mode used for validity diagnostics.",
    )
    parser.add_argument("--oracle-min-sufficiency", type=float, default=0.8)
    parser.add_argument("--oracle-min-temporal-iou", type=float, default=0.0)
    return parser.parse_args()


def parse_csv_ints(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_csv_strings(value: str) -> tuple[str, ...]:
    if not value.strip():
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def remaining_evidence(
    example: QuestionExample,
    selected_items: tuple[EvidenceItem, ...],
) -> tuple[EvidenceItem, ...]:
    selected_ids = {item.evidence_id for item in selected_items}
    return tuple(item for item in example.evidence_pool if item.evidence_id not in selected_ids)


def prediction_record(
    *,
    example: QuestionExample,
    method_name: str,
    method_type: str,
    selected_evidence: tuple[EvidenceItem, ...],
    prediction: ModelPrediction,
    answerer: Answerer,
    oracle: MinimalEvidenceOracle,
    trace: list[dict] | None = None,
) -> dict:
    oracle_valid = oracle.seed_satisfies_constraints(example, selected_evidence)
    oracle_subset = oracle.minimal_subset(example, selected_evidence)
    result = {
        "method_name": method_name,
        "method_type": method_type,
        "example_id": example.example_id,
        "video_id": example.video_id,
        "predicted_index": prediction.predicted_index,
        "prediction_confidence": prediction.confidence,
        "selected_evidence": serialize_evidence(selected_evidence),
        "selected_evidence_count": len(selected_evidence),
        "selected_evidence_cost": evidence_cost(selected_evidence),
        "oracle_valid": oracle_valid,
        "oracle_subset_count": len(oracle_subset),
        "oracle_subset_cost": evidence_cost(oracle_subset),
    }
    if trace is not None:
        result["trace"] = trace

    if example.answer_index is None:
        return result

    full_prediction = answerer.predict(example, example.evidence_pool)
    reduced_prediction = answerer.predict(example, remaining_evidence(example, selected_evidence))
    result.update(
        {
            "gold_index": example.answer_index,
            "correct": accuracy(prediction.predicted_index, example.answer_index),
            "sufficiency": sufficiency(full_prediction, prediction, example.answer_index),
            "comprehensiveness": comprehensiveness(full_prediction, reduced_prediction, example.answer_index),
        }
    )

    if example.temporal_grounding is not None:
        result["selected_temporal_iou"] = max_temporal_iou_for_target_spans(
            selected_evidence,
            temporal_target_spans(example.temporal_grounding, example.metadata),
        )
        result["oracle_temporal_iou"] = max_temporal_iou_for_target_spans(
            oracle_subset,
            temporal_target_spans(example.temporal_grounding, example.metadata),
        )

    return result


def summarize_records(records: list[dict]) -> dict[str, float]:
    return {
        "accuracy": average([record["correct"] for record in records if "correct" in record]),
        "selected_evidence_cost": average([record["selected_evidence_cost"] for record in records]),
        "selected_evidence_count": average([record["selected_evidence_count"] for record in records]),
        "selected_temporal_iou": average(
            [record["selected_temporal_iou"] for record in records if "selected_temporal_iou" in record]
        ),
        "sufficiency": average([record["sufficiency"] for record in records if "sufficiency" in record]),
        "comprehensiveness": average(
            [record["comprehensiveness"] for record in records if "comprehensiveness" in record]
        ),
        "oracle_valid_rate": average([1.0 if record["oracle_valid"] else 0.0 for record in records]),
        "oracle_subset_cost": average([record["oracle_subset_cost"] for record in records]),
        "oracle_subset_count": average([record["oracle_subset_count"] for record in records]),
    }


def evaluate_fixed_budget(
    *,
    examples: list[QuestionExample],
    method_name: str,
    allocation: RetrievalAllocation,
    answerer: Answerer,
    retriever: FixedBudgetRetriever,
    oracle: MinimalEvidenceOracle,
) -> list[dict]:
    records = []
    for example in examples:
        selected_evidence = retriever.retrieve(example, allocation)
        prediction = answerer.predict(example, selected_evidence)
        records.append(
            prediction_record(
                example=example,
                method_name=method_name,
                method_type="fixed_budget",
                selected_evidence=selected_evidence,
                prediction=prediction,
                answerer=answerer,
                oracle=oracle,
            )
        )
    return records


def evaluate_policy(
    *,
    examples: list[QuestionExample],
    method_name: str,
    policy_name: str,
    allocation: RetrievalAllocation,
    max_items: int,
    answerer: Answerer,
    pool_builder: CandidatePoolBuilder,
    oracle: MinimalEvidenceOracle,
) -> list[dict]:
    policy = build_policy(policy_name, answerer=answerer)
    records = []
    for example in examples:
        candidate_pool = pool_builder.build(example, top_k_per_modality=allocation)
        trace = policy.run(example, candidate_pool, max_items=max_items)
        selected_evidence = tuple(step.selected_item for step in trace.steps if step.selected_item is not None)
        records.append(
            prediction_record(
                example=example,
                method_name=method_name,
                method_type="policy_control",
                selected_evidence=selected_evidence,
                prediction=trace.final_prediction,
                answerer=answerer,
                oracle=oracle,
                trace=[
                    {
                        "step_index": step.step_index,
                        "action": step.action,
                        "selected_evidence_id": step.selected_item.evidence_id if step.selected_item else None,
                        "confidence_after_step": step.confidence_after_step,
                    }
                    for step in trace.steps
                ],
            )
        )
    return records


def markdown_table(summary: dict[str, dict]) -> str:
    metrics = (
        "accuracy",
        "selected_evidence_cost",
        "selected_evidence_count",
        "selected_temporal_iou",
        "sufficiency",
        "oracle_valid_rate",
    )
    headers = ["Method", "Type", "Allocation"] + [metric.replace("_", " ").title() for metric in metrics]
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for method_name, payload in summary["methods"].items():
        row = [
            method_name,
            payload["method_type"],
            payload.get("allocation_label", ""),
        ]
        method_metrics = payload["metrics"]
        for metric in metrics:
            row.append(f"{float(method_metrics.get(metric, 0.0)):.3f}")
        rows.append("| " + " | ".join(row) + " |")
    return "\n".join(rows) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    predictions_dir = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.write_predictions:
        predictions_dir.mkdir(parents=True, exist_ok=True)

    examples = load_normalized_examples(args.input_path)
    if args.limit is not None:
        examples = examples[: args.limit]

    answerer = build_answerer(
        args.answerer,
        args.answerer_model_dir,
        model_name=args.answerer_model_name,
        device=args.answerer_device,
    )
    retriever_backend = build_named_retriever(
        args.retriever,
        visual_model_name=args.visual_model_name,
        visual_device=args.visual_device,
    )
    fixed_retriever = FixedBudgetRetriever(retriever_backend)
    pool_builder = CandidatePoolBuilder(retriever_backend)
    oracle_config = OracleConfig.from_mode(
        args.oracle_mode,
        min_sufficiency=args.oracle_min_sufficiency,
        min_temporal_iou=args.oracle_min_temporal_iou,
    )
    oracle = MinimalEvidenceOracle(answerer, config=oracle_config)

    methods: list[tuple[str, str, RetrievalAllocation, str | None]] = [
        (
            "fixed_main",
            "fixed_budget",
            RetrievalAllocation(args.subtitle_k, args.frame_k, args.segment_k),
            None,
        ),
        ("fixed_frame1", "fixed_budget", RetrievalAllocation(0, 1, 0), None),
        ("fixed_segment1", "fixed_budget", RetrievalAllocation(0, 0, 1), None),
        ("fixed_frame1_segment1", "fixed_budget", RetrievalAllocation(0, 1, 1), None),
    ]
    for k in parse_csv_ints(args.fixed_budget_ks):
        methods.append((f"fixed_f{k}_s{k}", "fixed_budget", RetrievalAllocation(0, k, k), None))
    for policy_name in parse_csv_strings(args.policies):
        methods.append(
            (
                f"policy_{policy_name}",
                "policy_control",
                RetrievalAllocation(args.subtitle_k, args.frame_k, args.segment_k),
                policy_name,
            )
        )

    seen = set()
    method_summaries = {}
    for method_name, method_type, allocation, policy_name in methods:
        if method_name in seen:
            continue
        seen.add(method_name)
        if method_type == "fixed_budget":
            records = evaluate_fixed_budget(
                examples=examples,
                method_name=method_name,
                allocation=allocation,
                answerer=answerer,
                retriever=fixed_retriever,
                oracle=oracle,
            )
        else:
            if policy_name is None:
                raise ValueError(f"Missing policy name for method {method_name}.")
            records = evaluate_policy(
                examples=examples,
                method_name=method_name,
                policy_name=policy_name,
                allocation=allocation,
                max_items=args.max_items,
                answerer=answerer,
                pool_builder=pool_builder,
                oracle=oracle,
            )

        if args.write_predictions:
            save_jsonl(records, predictions_dir / f"{method_name}.jsonl")
        method_summaries[method_name] = {
            "method_type": method_type,
            "allocation": allocation.to_dict(),
            "allocation_label": f"s{allocation.subtitle}/f{allocation.frame}/g{allocation.segment}",
            "num_examples": len(records),
            "metrics": summarize_records(records),
        }

    summary = {
        "input_path": args.input_path,
        "num_examples": len(examples),
        "answerer": args.answerer,
        "answerer_model_name": args.answerer_model_name if args.answerer == "frozen_multimodal" else None,
        "retriever": args.retriever,
        "visual_model_name": args.visual_model_name if args.retriever == "hybrid_clip" else None,
        "oracle_config": oracle_config.to_dict(),
        "methods": method_summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(markdown_table(summary), encoding="utf-8")

    print(f"Wrote control summary to {output_dir / 'summary.json'}")
    print(f"Wrote Markdown table to {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
