import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize NExT-GQA-style QA, grounding, and grounded-QA metrics."
    )
    parser.add_argument(
        "--candidate-path",
        required=True,
        help="Candidate-pool JSONL with temporal grounding metadata.",
    )
    parser.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        help="Prediction JSONL files to summarize.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional method labels matching --predictions.",
    )
    parser.add_argument("--summary-json", required=True, help="Path for JSON summary.")
    parser.add_argument("--summary-md", required=True, help="Path for Markdown table.")
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=0,
        help="Optional number of bootstrap samples for 95%% confidence intervals.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Bootstrap random seed.")
    return parser.parse_args()


def load_example_metadata(path: str | Path) -> dict[str, dict]:
    metadata = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            spans = []
            raw_spans = record.get("metadata", {}).get("temporal_grounding_spans")
            if isinstance(raw_spans, list):
                for span in raw_spans:
                    if isinstance(span, (list, tuple)) and len(span) == 2:
                        start, end = sorted((float(span[0]), float(span[1])))
                        spans.append((start, end))
            if not spans and record.get("temporal_grounding") is not None:
                start, end = sorted(
                    (
                        float(record["temporal_grounding"][0]),
                        float(record["temporal_grounding"][1]),
                    )
                )
                spans.append((start, end))
            metadata[record["example_id"]] = {
                "target_spans": spans,
                "question_type": record.get("metadata", {}).get("question_type", ""),
            }
    return metadata


def evidence_interval(item: dict) -> tuple[float, float] | None:
    start = item.get("start_time")
    end = item.get("end_time")
    if start is None and end is None:
        return None
    if start is None:
        start = end
    if end is None:
        end = start
    if start is None or end is None:
        return None
    start, end = sorted((float(start), float(end)))
    return start, end


def interval_overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def temporal_iou(prediction: tuple[float, float], target: tuple[float, float]) -> float:
    intersection = interval_overlap(prediction, target)
    union = max(prediction[1], target[1]) - min(prediction[0], target[0])
    if union <= 0.0:
        return 0.0
    return intersection / union


def temporal_iop(prediction: tuple[float, float], target: tuple[float, float]) -> float:
    length = prediction[1] - prediction[0]
    if length <= 0.0:
        return 1.0 if target[0] <= prediction[0] <= target[1] else 0.0
    return interval_overlap(prediction, target) / length


def best_overlap(
    selected_evidence: list[dict],
    target_spans: list[tuple[float, float]],
    metric,
) -> float:
    if not selected_evidence or not target_spans:
        return 0.0
    best = 0.0
    for item in selected_evidence:
        interval = evidence_interval(item)
        if interval is None:
            continue
        best = max(best, max(metric(interval, target) for target in target_spans))
    return best


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def summarize_rows(rows: list[dict]) -> dict[str, float]:
    return {
        "num_examples": float(len(rows)),
        "acc_qa": average([row["correct"] for row in rows]),
        "selected_evidence_cost": average([row["cost"] for row in rows]),
        "selected_evidence_count": average([row["count"] for row in rows]),
        "m_iop": average([row["iop"] for row in rows]),
        "iop_at_0_3": average([1.0 if row["iop"] >= 0.3 else 0.0 for row in rows]),
        "iop_at_0_5": average([1.0 if row["iop"] >= 0.5 else 0.0 for row in rows]),
        "m_iou": average([row["iou"] for row in rows]),
        "iou_at_0_3": average([1.0 if row["iou"] >= 0.3 else 0.0 for row in rows]),
        "iou_at_0_5": average([1.0 if row["iou"] >= 0.5 else 0.0 for row in rows]),
        "acc_gqa_iop_0_5": average(
            [1.0 if row["correct"] and row["iop"] >= 0.5 else 0.0 for row in rows]
        ),
        "acc_gqa_iou_0_5": average(
            [1.0 if row["correct"] and row["iou"] >= 0.5 else 0.0 for row in rows]
        ),
    }


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_intervals(
    rows: list[dict],
    *,
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if samples <= 0 or not rows:
        return {}
    rng = random.Random(seed)
    metric_values: dict[str, list[float]] = {}
    for _ in range(samples):
        sample_rows = [rows[rng.randrange(len(rows))] for _ in rows]
        sample_metrics = summarize_rows(sample_rows)
        for metric, value in sample_metrics.items():
            if metric == "num_examples":
                continue
            metric_values.setdefault(metric, []).append(float(value))
    return {
        metric: {
            "low": percentile(values, 0.025),
            "high": percentile(values, 0.975),
        }
        for metric, values in metric_values.items()
    }


def summarize_prediction_file(
    path: str | Path,
    metadata: dict[str, dict],
    *,
    bootstrap_samples: int = 0,
    seed: int = 13,
) -> dict:
    rows = []
    by_question_type: dict[str, list[dict]] = {}
    trace_counts: dict[str, int] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            example_meta = metadata.get(record["example_id"], {})
            target_spans = example_meta.get("target_spans", [])
            selected_evidence = record.get("selected_evidence", [])
            row = {
                "correct": float(record.get("correct", 0.0)),
                "cost": float(record.get("selected_evidence_cost", 0.0)),
                "count": float(
                    record.get("selected_evidence_count", len(selected_evidence))
                ),
                "iop": best_overlap(selected_evidence, target_spans, temporal_iop),
                "iou": best_overlap(selected_evidence, target_spans, temporal_iou),
            }
            rows.append(row)
            question_type = str(example_meta.get("question_type", ""))
            by_question_type.setdefault(question_type, []).append(row)
            if record.get("trace"):
                actions = " -> ".join(step["action"] for step in record["trace"])
                trace_counts[actions] = trace_counts.get(actions, 0) + 1

    payload = {
        "metrics": summarize_rows(rows),
        "by_question_type": {
            question_type: summarize_rows(question_rows)
            for question_type, question_rows in sorted(by_question_type.items())
        },
        "top_traces": dict(
            sorted(trace_counts.items(), key=lambda item: item[1], reverse=True)[:10]
        ),
    }
    if bootstrap_samples > 0:
        payload["bootstrap_95ci"] = bootstrap_intervals(
            rows,
            samples=bootstrap_samples,
            seed=seed,
        )
    return payload


def markdown_table(summary: dict[str, dict]) -> str:
    headers = [
        "Method",
        "Acc@QA",
        "Cost",
        "Count",
        "mIoP",
        "IoP@0.5",
        "mIoU",
        "IoU@0.5",
        "Acc@GQA",
    ]
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for label, payload in summary.items():
        metrics = payload["metrics"]
        rows.append(
            "| "
            + " | ".join(
                [
                    label,
                    f"{metrics['acc_qa']:.3f}",
                    f"{metrics['selected_evidence_cost']:.3f}",
                    f"{metrics['selected_evidence_count']:.3f}",
                    f"{metrics['m_iop']:.3f}",
                    f"{metrics['iop_at_0_5']:.3f}",
                    f"{metrics['m_iou']:.3f}",
                    f"{metrics['iou_at_0_5']:.3f}",
                    f"{metrics['acc_gqa_iop_0_5']:.3f}",
                ]
            )
            + " |"
        )
    sections = ["\n".join(rows)]

    if any("bootstrap_95ci" in payload for payload in summary.values()):
        sections.append(confidence_interval_table(summary))

    sections.append(question_type_table(summary))
    return "\n\n".join(section for section in sections if section) + "\n"


def format_ci(payload: dict, metric: str) -> str:
    interval = payload.get("bootstrap_95ci", {}).get(metric)
    if not interval:
        return ""
    return f"[{interval['low']:.3f}, {interval['high']:.3f}]"


def confidence_interval_table(summary: dict[str, dict]) -> str:
    headers = [
        "Method",
        "Acc@QA 95% CI",
        "mIoP 95% CI",
        "IoP@0.5 95% CI",
        "Acc@GQA 95% CI",
    ]
    rows = [
        "### Bootstrap confidence intervals",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for label, payload in summary.items():
        rows.append(
            "| "
            + " | ".join(
                [
                    label,
                    format_ci(payload, "acc_qa"),
                    format_ci(payload, "m_iop"),
                    format_ci(payload, "iop_at_0_5"),
                    format_ci(payload, "acc_gqa_iop_0_5"),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def question_type_table(summary: dict[str, dict]) -> str:
    headers = [
        "Method",
        "Question type",
        "N",
        "Acc@QA",
        "mIoP",
        "IoP@0.5",
        "Acc@GQA",
    ]
    rows = [
        "### Question-type breakdown",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    added = False
    for label, payload in summary.items():
        for question_type, metrics in payload.get("by_question_type", {}).items():
            added = True
            rows.append(
                "| "
                + " | ".join(
                    [
                        label,
                        question_type or "unknown",
                        f"{int(metrics['num_examples'])}",
                        f"{metrics['acc_qa']:.3f}",
                        f"{metrics['m_iop']:.3f}",
                        f"{metrics['iop_at_0_5']:.3f}",
                        f"{metrics['acc_gqa_iop_0_5']:.3f}",
                    ]
                )
                + " |"
            )
    return "\n".join(rows) if added else ""


def main() -> None:
    args = parse_args()
    if args.labels is not None and args.labels and len(args.labels) != len(args.predictions):
        raise ValueError("--labels must have the same length as --predictions.")

    labels = args.labels or [Path(path).stem for path in args.predictions]
    metadata = load_example_metadata(args.candidate_path)
    summary = {
        label: summarize_prediction_file(
            path,
            metadata,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        for label, path in zip(labels, args.predictions, strict=True)
    }

    summary_json = Path(args.summary_json)
    summary_md = Path(args.summary_md)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md.write_text(markdown_table(summary), encoding="utf-8")
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
