import argparse
import json
from pathlib import Path

from adaptive_evidence_vqa.data.base import load_jsonl, save_jsonl
from adaptive_evidence_vqa.data.evidence_records import parse_evidence_record, serialize_evidence
from adaptive_evidence_vqa.data.normalized import load_normalized_examples
from adaptive_evidence_vqa.eval.metrics import evidence_cost, temporal_target_spans
from adaptive_evidence_vqa.models.answerer import build_answerer
from adaptive_evidence_vqa.models.evidence_router import MLPEvidenceRouterPolicy, temporal_iop_target
from adaptive_evidence_vqa.schemas import EvidenceItem, QuestionExample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create follow-up evidence-selection files from an existing fixed 3+3 "
            "retrieved pool. The outputs can be passed directly to run_vlm_answerer.py."
        )
    )
    parser.add_argument("--candidate-path", required=True, help="Candidate JSONL with question metadata.")
    parser.add_argument(
        "--pool-selection-path",
        required=True,
        help="Prediction JSONL whose selected_evidence field is the retrieved pool, e.g. fixed 3+3.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for generated selection JSONL files.")
    parser.add_argument("--split", required=True, help="Split label used in output filenames.")
    parser.add_argument("--router-model-dir", help="Optional trained MLP router directory.")
    parser.add_argument(
        "--router-method-prefix",
        default="router_mlp_top2",
        help="Prefix for generated router selection methods.",
    )
    parser.add_argument(
        "--nms-thresholds",
        nargs="*",
        type=float,
        default=(0.0, 0.3, 0.5),
        help="Temporal IoU thresholds for router+NMS top-2 variants.",
    )
    return parser.parse_args()


def item_interval(item: EvidenceItem) -> tuple[float, float] | None:
    if item.start_time is None and item.end_time is None:
        return None
    start = item.start_time if item.start_time is not None else item.end_time
    end = item.end_time if item.end_time is not None else item.start_time
    if start is None or end is None:
        return None
    return tuple(sorted((float(start), float(end))))


def interval_iou(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float:
    if a is None or b is None:
        return 0.0
    intersection = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    if union <= 0.0:
        return 1.0 if a[0] == b[0] else 0.0
    return intersection / union


def temporal_iou_target(item: EvidenceItem, target_spans: list[tuple[float, float]]) -> float:
    interval = item_interval(item)
    if interval is None or not target_spans:
        return 0.0
    best = 0.0
    for target in target_spans:
        best = max(best, interval_iou(interval, target))
    return best


def make_candidate_pool(items: list[EvidenceItem]) -> dict[str, tuple[EvidenceItem, ...]]:
    return {
        "subtitle": tuple(item for item in items if item.modality.value == "subtitle"),
        "frame": tuple(item for item in items if item.modality.value == "frame"),
        "segment": tuple(item for item in items if item.modality.value == "segment"),
    }


def select_oracle(
    example: QuestionExample,
    items: list[EvidenceItem],
    *,
    top_k: int,
    target_kind: str,
) -> list[EvidenceItem]:
    target_spans = temporal_target_spans(example.temporal_grounding, example.metadata)
    if target_kind == "iop":
        score_fn = lambda item: temporal_iop_target(item, target_spans)
    elif target_kind == "iou":
        score_fn = lambda item: temporal_iou_target(item, target_spans)
    else:
        raise ValueError(f"Unsupported target kind: {target_kind}")
    ranked = sorted(
        items,
        key=lambda item: (
            -score_fn(item),
            -float(item.retrieval_score),
            item.modality.value,
            item.evidence_id,
        ),
    )
    return ranked[: min(top_k, len(ranked))]


def select_with_nms(
    scored_items: list[tuple[float, EvidenceItem]],
    *,
    top_k: int,
    threshold: float,
) -> list[EvidenceItem]:
    selected: list[EvidenceItem] = []
    for _, item in scored_items:
        candidate_interval = item_interval(item)
        if all(interval_iou(candidate_interval, item_interval(existing)) <= threshold for existing in selected):
            selected.append(item)
        if len(selected) >= top_k:
            return selected

    selected_ids = {item.evidence_id for item in selected}
    for _, item in scored_items:
        if item.evidence_id not in selected_ids:
            selected.append(item)
        if len(selected) >= top_k:
            return selected
    return selected


def trace_for_selection(selected: list[EvidenceItem]) -> list[dict]:
    trace = []
    for index, item in enumerate(selected):
        trace.append(
            {
                "step_index": index,
                "action": f"acquire_{item.modality.value}",
                "selected_evidence_id": item.evidence_id,
                "confidence_after_step": 0.0,
            }
        )
    trace.append(
        {
            "step_index": len(trace),
            "action": "stop",
            "selected_evidence_id": None,
            "confidence_after_step": 0.0,
        }
    )
    return trace


def result_record(example: QuestionExample, selected: list[EvidenceItem], *, method: str) -> dict:
    return {
        "example_id": example.example_id,
        "video_id": example.video_id,
        "method": method,
        "predicted_index": -1,
        "prediction_confidence": 0.0,
        "trace": trace_for_selection(selected),
        "selected_evidence": serialize_evidence(tuple(selected)),
        "selected_evidence_count": len(selected),
        "selected_evidence_cost": evidence_cost(tuple(selected)),
        "gold_index": example.answer_index,
    }


def load_pool_records(path: str | Path) -> dict[str, list[EvidenceItem]]:
    pools = {}
    for record in load_jsonl(path):
        pools[record["example_id"]] = [
            parse_evidence_record(item) for item in record.get("selected_evidence", [])
        ]
    return pools


def write_method(output_dir: Path, split: str, method: str, records: list[dict]) -> Path:
    path = output_dir / f"{split}_{method}.jsonl"
    save_jsonl(records, path)
    return path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_normalized_examples(args.candidate_path)
    pools_by_id = load_pool_records(args.pool_selection_path)
    examples = [example for example in examples if example.example_id in pools_by_id]

    methods: dict[str, list[dict]] = {}
    for top_k in (1, 2, 3):
        for target_kind in ("iop", "iou"):
            method = f"oracle_{target_kind}_top{top_k}"
            methods[method] = [
                result_record(
                    example,
                    select_oracle(
                        example,
                        pools_by_id[example.example_id],
                        top_k=top_k,
                        target_kind=target_kind,
                    ),
                    method=method,
                )
                for example in examples
            ]

    if args.router_model_dir:
        router = MLPEvidenceRouterPolicy.load(
            args.router_model_dir,
            answerer=build_answerer("lexical"),
        )
        for threshold in args.nms_thresholds:
            threshold_label = str(threshold).replace(".", "p")
            method = f"{args.router_method_prefix}_nms{threshold_label}"
            records = []
            for example in examples:
                pool_items = pools_by_id[example.example_id]
                scored = sorted(
                    router.score_items(example, make_candidate_pool(pool_items)),
                    key=lambda pair: pair[0],
                    reverse=True,
                )
                selected = select_with_nms(scored, top_k=2, threshold=threshold)
                records.append(result_record(example, selected, method=method))
            methods[method] = records

    manifest = {}
    for method, records in methods.items():
        path = write_method(output_dir, args.split, method, records)
        manifest[method] = {
            "path": str(path),
            "num_records": len(records),
            "mean_cost": sum(record["selected_evidence_cost"] for record in records) / max(len(records), 1),
            "mean_count": sum(record["selected_evidence_count"] for record in records) / max(len(records), 1),
        }

    (output_dir / f"{args.split}_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
