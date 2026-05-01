import argparse
import json
import textwrap
from pathlib import Path


LETTERS = "ABCDE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select strong qualitative examples for the grounded VideoQA paper."
    )
    parser.add_argument("--root", default="runs/nextgqa_full_seed13")
    parser.add_argument(
        "--output-dir",
        default="runs/nextgqa_full_seed13/outputs/followup_strengthening/analysis/paper_qualitative",
    )
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_by_id(path: str | Path) -> dict[str, dict]:
    return {record["example_id"]: record for record in load_jsonl(path)}


def evidence_interval(item: dict) -> tuple[float, float] | None:
    start = item.get("start_time")
    end = item.get("end_time")
    if start is None and "time" in item:
        start = item["time"]
    if end is None and "time" in item:
        end = item["time"]
    if start is None and end is None:
        return None
    if start is None:
        start = end
    if end is None:
        end = start
    if start is None or end is None:
        return None
    return tuple(sorted((float(start), float(end))))


def target_spans(example: dict) -> list[tuple[float, float]]:
    spans = []
    raw = example.get("metadata", {}).get("temporal_grounding_spans")
    if isinstance(raw, list):
        for span in raw:
            if isinstance(span, list) and len(span) == 2:
                spans.append(tuple(sorted((float(span[0]), float(span[1])))))
    if not spans and example.get("temporal_grounding") is not None:
        start, end = example["temporal_grounding"]
        spans.append(tuple(sorted((float(start), float(end)))))
    return spans


def interval_overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def interval_iop(prediction: tuple[float, float], target: tuple[float, float]) -> float:
    length = prediction[1] - prediction[0]
    if length <= 0.0:
        return 1.0 if target[0] <= prediction[0] <= target[1] else 0.0
    return interval_overlap(prediction, target) / length


def interval_iou(prediction: tuple[float, float], target: tuple[float, float]) -> float:
    intersection = interval_overlap(prediction, target)
    union = max(prediction[1], target[1]) - min(prediction[0], target[0])
    return intersection / union if union > 0.0 else 0.0


def best_metric(selected: list[dict], spans: list[tuple[float, float]], metric) -> float:
    best = 0.0
    for item in selected:
        interval = evidence_interval(item)
        if interval is None:
            continue
        for span in spans:
            best = max(best, metric(interval, span))
    return best


def pairwise_iou(selected: list[dict]) -> float:
    intervals = [evidence_interval(item) for item in selected]
    intervals = [interval for interval in intervals if interval is not None]
    if len(intervals) < 2:
        return 0.0
    values = []
    for index, left in enumerate(intervals):
        for right in intervals[index + 1 :]:
            intersection = interval_overlap(left, right)
            union = max(left[1], right[1]) - min(left[0], right[0])
            values.append(intersection / union if union > 0.0 else 0.0)
    return sum(values) / len(values) if values else 0.0


def answer_letter(index: int | float | None) -> str:
    if index is None:
        return "?"
    index = int(index)
    return LETTERS[index] if 0 <= index < len(LETTERS) else "?"


def span_text(interval: tuple[float, float] | None) -> str:
    if interval is None:
        return "?"
    if abs(interval[1] - interval[0]) < 1e-6:
        return f"{interval[0]:.1f}s"
    return f"{interval[0]:.1f}-{interval[1]:.1f}s"


def evidence_text(selected: list[dict]) -> str:
    parts = []
    for item in selected:
        modality = item.get("modality", "?")
        evidence_id = str(item.get("evidence_id", ""))
        index = evidence_id.split(":")[-1] if ":" in evidence_id else ""
        parts.append(f"{modality[0].upper()}{index}@{span_text(evidence_interval(item))}")
    return ", ".join(parts)


def compact_question(text: str, width: int = 82) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def record_metrics(record: dict, example: dict) -> dict:
    selected = list(record.get("selected_evidence", []))
    spans = target_spans(example)
    return {
        "pred": int(record.get("predicted_index", -1)),
        "correct": bool(float(record.get("correct", 0.0))),
        "iop": best_metric(selected, spans, interval_iop),
        "iou": best_metric(selected, spans, interval_iou),
        "pair_iou": pairwise_iou(selected),
        "evidence": evidence_text(selected),
        "selected_evidence": selected,
    }


def build_cases(examples: dict[str, dict], records: dict[str, dict[str, dict]]) -> list[dict]:
    rows = []
    for example_id, example in examples.items():
        if not all(example_id in method_records for method_records in records.values()):
            continue
        metrics = {
            method: record_metrics(method_records[example_id], example)
            for method, method_records in records.items()
        }
        rows.append(
            {
                "example_id": example_id,
                "video_id": example.get("video_id"),
                "question_type": example.get("metadata", {}).get("question_type", ""),
                "target_spans": [span_text(span) for span in target_spans(example)],
                "question": example["question"],
                "options": example["options"],
                "gold": int(example["answer_index"]),
                "metrics": metrics,
            }
        )
    return rows


def choose_cases(cases: list[dict]) -> list[dict]:
    selected = []
    used_ids = set()

    def pick(title: str, predicate, score_fn, takeaway: str):
        candidates = [
            case for case in cases
            if case["example_id"] not in used_ids and predicate(case)
        ]
        if not candidates:
            return
        chosen = max(candidates, key=score_fn)
        chosen = dict(chosen)
        chosen["title"] = title
        chosen["takeaway"] = takeaway
        selected.append(chosen)
        used_ids.add(chosen["example_id"])

    pick(
        "Diversity Removes Redundant Evidence",
        lambda c: (
            c["metrics"]["mlp_nms"]["correct"]
            and c["metrics"]["mlp_nms"]["iop"] >= 0.95
            and c["metrics"]["mlp_plain"]["iop"] <= 0.10
            and c["metrics"]["mlp_plain"]["pair_iou"] >= 0.20
        ),
        lambda c: (
            c["metrics"]["mlp_nms"]["iop"] - c["metrics"]["mlp_plain"]["iop"]
            + c["metrics"]["mlp_plain"]["pair_iou"]
        ),
        "NMS keeps the answer correct while replacing a redundant nearby segment with temporally grounded evidence.",
    )
    pick(
        "Oracle Exposes the Router Gap",
        lambda c: (
            c["metrics"]["oracle_top2"]["correct"]
            and c["metrics"]["oracle_top2"]["iop"] >= 0.95
            and (
                not c["metrics"]["mlp_nms"]["correct"]
                or c["metrics"]["mlp_nms"]["iop"] <= 0.20
            )
        ),
        lambda c: (
            c["metrics"]["oracle_top2"]["iop"] - c["metrics"]["mlp_nms"]["iop"]
            + (1.0 if not c["metrics"]["mlp_nms"]["correct"] else 0.0)
        ),
        "The retrieved pool already contains compact supporting evidence, but the learned router does not always select it.",
    )
    pick(
        "High-Coverage Evidence Still Helps",
        lambda c: (
            c["metrics"]["fixed_f3_s3"]["correct"]
            and c["metrics"]["fixed_f3_s3"]["iop"] >= 0.95
            and not c["metrics"]["mlp_nms"]["correct"]
            and not c["metrics"]["oracle_top2"]["correct"]
        ),
        lambda c: (
            c["metrics"]["fixed_f3_s3"]["iop"]
            - max(c["metrics"]["mlp_nms"]["iop"], c["metrics"]["oracle_top2"]["iop"])
        ),
        "Some questions still need broader visual context or redundant views even when compact oracle evidence overlaps the target span.",
    )
    pick(
        "Compact Oracle Nearly Matches Fixed Budget",
        lambda c: (
            c["metrics"]["oracle_top2"]["correct"]
            and c["metrics"]["oracle_top2"]["iop"] >= 0.95
            and c["metrics"]["fixed_f3_s3"]["correct"]
            and c["metrics"]["fixed_f3_s3"]["iop"] >= 0.95
            and c["metrics"]["mlp_nms"]["iop"] < 0.50
        ),
        lambda c: c["metrics"]["oracle_top2"]["iop"] - c["metrics"]["mlp_nms"]["iop"],
        "Two well-chosen items can be enough; the large fixed budget mainly compensates for imperfect learned selection.",
    )
    return selected


def choose_appendix_cases(cases: list[dict], excluded_ids: set[str]) -> list[dict]:
    selected = []
    used_ids = set(excluded_ids)

    def pick(title: str, predicate, score_fn, takeaway: str):
        candidates = [
            case for case in cases
            if case["example_id"] not in used_ids and predicate(case)
        ]
        if not candidates:
            return
        chosen = max(candidates, key=score_fn)
        chosen = dict(chosen)
        chosen["title"] = title
        chosen["takeaway"] = takeaway
        selected.append(chosen)
        used_ids.add(chosen["example_id"])

    pick(
        "High-Coverage Context",
        lambda c: (
            c["metrics"]["fixed_f3_s3"]["correct"]
            and c["metrics"]["fixed_f3_s3"]["iop"] >= 0.95
            and not c["metrics"]["mlp_nms"]["correct"]
            and not c["metrics"]["oracle_top2"]["correct"]
        ),
        lambda c: (
            c["metrics"]["fixed_f3_s3"]["iop"]
            - max(c["metrics"]["mlp_nms"]["iop"], c["metrics"]["oracle_top2"]["iop"])
        ),
        "A larger fixed evidence budget can rescue cases where compact evidence overlaps the event but lacks enough context for the answer.",
    )
    pick(
        "Compact Oracle Support",
        lambda c: (
            c["metrics"]["mlp_nms"]["correct"]
            and c["metrics"]["mlp_nms"]["iop"] < 0.20
            and c["metrics"]["oracle_top2"]["correct"]
            and c["metrics"]["oracle_top2"]["iop"] >= 0.95
            and c["metrics"]["fixed_f3_s3"]["correct"]
        ),
        lambda c: c["metrics"]["oracle_top2"]["iop"] - c["metrics"]["mlp_nms"]["iop"],
        "The answer can be correct even when selected evidence is ungrounded, while oracle evidence shows compact support was available.",
    )
    pick(
        "Oracle Recovers Support",
        lambda c: (
            not c["metrics"]["mlp_nms"]["correct"]
            and c["metrics"]["mlp_nms"]["iop"] <= 0.20
            and c["metrics"]["oracle_top2"]["correct"]
            and c["metrics"]["oracle_top2"]["iop"] >= 0.95
            and c["metrics"]["fixed_f3_s3"]["correct"]
        ),
        lambda c: (
            c["metrics"]["oracle_top2"]["iop"]
            - c["metrics"]["mlp_nms"]["iop"]
            + c["metrics"]["fixed_f3_s3"]["iop"]
        ),
        "Oracle evidence recovers the annotated support and answer when the learned selector routes to the wrong moment.",
    )
    pick(
        "Grounded Learned Success",
        lambda c: (
            c["metrics"]["mlp_nms"]["correct"]
            and c["metrics"]["mlp_nms"]["iop"] >= 0.95
            and c["metrics"]["oracle_top2"]["correct"]
            and c["metrics"]["fixed_f3_s3"]["correct"]
        ),
        lambda c: c["metrics"]["mlp_nms"]["iop"] + c["metrics"]["fixed_f3_s3"]["iop"],
        "The learned selector can find compact grounded evidence that is sufficient for Qwen to answer correctly.",
    )
    pick(
        "Grounded Answer Error",
        lambda c: (
            not c["metrics"]["mlp_nms"]["correct"]
            and c["metrics"]["mlp_nms"]["iop"] >= 0.95
            and c["metrics"]["oracle_top2"]["correct"]
            and c["metrics"]["fixed_f3_s3"]["correct"]
        ),
        lambda c: c["metrics"]["mlp_nms"]["iop"] + c["metrics"]["oracle_top2"]["iop"],
        "High temporal overlap is necessary but not sufficient; the answerer can still misinterpret grounded evidence.",
    )
    return selected


def method_line(label: str, metrics: dict) -> str:
    return (
        f"- {label}: pred {answer_letter(metrics['pred'])}, "
        f"correct {int(metrics['correct'])}, "
        f"IoP {metrics['iop']:.3f}, IoU {metrics['iou']:.3f}, "
        f"evidence {metrics['evidence']}"
    )


def markdown(cases: list[dict]) -> str:
    sections = ["# Paper Qualitative Cases\n\n"]
    for case in cases:
        sections.append(f"## {case['title']}\n\n")
        sections.append(f"**Example:** `{case['example_id']}`; type `{case['question_type']}`; target `{', '.join(case['target_spans'])}`\n\n")
        sections.append(f"**Question:** {case['question']}\n\n")
        options = "; ".join(
            f"{answer_letter(index)}. {option}"
            for index, option in enumerate(case["options"])
        )
        sections.append(f"**Options:** {options}\n\n")
        sections.append(f"**Gold:** {answer_letter(case['gold'])}. {case['options'][case['gold']]}\n\n")
        for key, label in [
            ("linear_min2", "Linear min-2"),
            ("mlp_plain", "MLP top-2"),
            ("mlp_nms", "MLP top-2 + NMS"),
            ("temporal_router_val_or_partial", "Temporal-supervised router"),
            ("oracle_top2", "Oracle IoP top-2"),
            ("fixed_f3_s3", "Fixed 3+3"),
        ]:
            if key in case["metrics"]:
                sections.append(method_line(label, case["metrics"][key]) + "\n")
        sections.append(f"\n**Takeaway:** {case['takeaway']}\n\n")
    return "".join(sections)


def latex_table(cases: list[dict]) -> str:
    rows = [
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Qualitative examples from NExT-GQA test. Evidence entries use F/S for frame/segment and show timestamps.}",
        "\\label{tab:qualitative-cases}",
        "\\begin{tabular}{p{0.18\\linewidth}p{0.30\\linewidth}p{0.22\\linewidth}p{0.22\\linewidth}}",
        "\\toprule",
        "Case & Question / Gold & Learned evidence & Diagnostic evidence \\\\",
        "\\midrule",
    ]
    for case in cases:
        question = case["question"].replace("&", "\\&")
        gold = f"{answer_letter(case['gold'])}. {case['options'][case['gold']]}".replace("&", "\\&")
        learned = (
            f"MLP+NMS: {answer_letter(case['metrics']['mlp_nms']['pred'])}, "
            f"IoP {case['metrics']['mlp_nms']['iop']:.2f}; "
            f"{case['metrics']['mlp_nms']['evidence']}"
        ).replace("&", "\\&")
        diagnostic = (
            f"Oracle: {answer_letter(case['metrics']['oracle_top2']['pred'])}, "
            f"IoP {case['metrics']['oracle_top2']['iop']:.2f}; "
            f"Fixed: {answer_letter(case['metrics']['fixed_f3_s3']['pred'])}, "
            f"IoP {case['metrics']['fixed_f3_s3']['iop']:.2f}"
        ).replace("&", "\\&")
        title = case["title"].replace("&", "\\&")
        rows.append(
            f"{title} & {question} \\newline Gold: {gold} & {learned} & {diagnostic} \\\\"
        )
    rows.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_by_id(root / "candidates/test.visual_features.jsonl")
    records = {
        "linear_min2": load_by_id(root / "outputs/vlm_qwen25vl_3b_full/predictions/test_learned_policy_min2.jsonl"),
        "mlp_plain": load_by_id(root / "outputs/vlm_qwen25vl_3b_full/predictions/test_router_mlp_top2.jsonl"),
        "mlp_nms": load_by_id(root / "outputs/followup_strengthening/qwen/predictions/test_router_mlp_top2_nms0p0.jsonl"),
        "oracle_top2": load_by_id(root / "outputs/followup_strengthening/qwen/predictions/test_oracle_iop_top2.jsonl"),
        "fixed_f3_s3": load_by_id(root / "outputs/vlm_qwen25vl_3b_full/predictions/test_fixed_f3_s3.jsonl"),
    }
    temporal_path = root / "outputs/followup_strengthening/temporal_router/qwen/predictions/test_temporal_router_top2_nms0p0.jsonl"
    if temporal_path.exists():
        temporal_records = load_by_id(temporal_path)
        if len(temporal_records) == len(examples):
            records["temporal_router_val_or_partial"] = temporal_records

    all_cases = build_cases(examples, records)
    cases = choose_cases(all_cases)
    appendix_cases = choose_appendix_cases(
        all_cases,
        {case["example_id"] for case in cases[:2]},
    )
    payload = {
        "selection_note": (
            "Cases are selected from held-out test examples using completed Qwen "
            "outputs. Temporal-router rows are included only if the full test run has completed."
        ),
        "cases": cases,
        "appendix_cases": appendix_cases,
    }
    (output_dir / "paper_qualitative_cases.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (output_dir / "paper_qualitative_cases.md").write_text(markdown(cases), encoding="utf-8")
    (output_dir / "paper_qualitative_table.tex").write_text(latex_table(cases), encoding="utf-8")
    print(f"Wrote {len(cases)} qualitative cases to {output_dir}")


if __name__ == "__main__":
    main()
