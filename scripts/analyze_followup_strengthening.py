import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze follow-up grounded VideoQA experiments for research tables."
    )
    parser.add_argument("--root", default="runs/nextgqa_full_seed13")
    parser.add_argument(
        "--output-dir",
        default="runs/nextgqa_full_seed13/outputs/followup_strengthening/analysis",
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


def overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def interval_iop(prediction: tuple[float, float], target: tuple[float, float]) -> float:
    length = prediction[1] - prediction[0]
    if length <= 0.0:
        return 1.0 if target[0] <= prediction[0] <= target[1] else 0.0
    return overlap(prediction, target) / length


def interval_iou(prediction: tuple[float, float], target: tuple[float, float]) -> float:
    intersection = overlap(prediction, target)
    union = max(prediction[1], target[1]) - min(prediction[0], target[0])
    if union <= 0.0:
        return 0.0
    return intersection / union


def best_metric(selected: list[dict], spans: list[tuple[float, float]], metric) -> float:
    best = 0.0
    for item in selected:
        interval = evidence_interval(item)
        if interval is None:
            continue
        for span in spans:
            best = max(best, metric(interval, span))
    return best


def pairwise_interval_iou(selected: list[dict]) -> float:
    intervals = [evidence_interval(item) for item in selected]
    intervals = [interval for interval in intervals if interval is not None]
    if len(intervals) < 2:
        return 0.0
    values = []
    for left_index, left in enumerate(intervals):
        for right in intervals[left_index + 1 :]:
            intersection = overlap(left, right)
            union = max(left[1], right[1]) - min(left[0], right[0])
            values.append(intersection / union if union > 0.0 else 0.0)
    return sum(values) / len(values) if values else 0.0


def selected_cost(selected: list[dict]) -> float:
    return sum(float(item.get("acquisition_cost", 1.0)) for item in selected)


def selected_combo(selected: list[dict]) -> str:
    return "+".join(str(item.get("modality", "")) for item in selected)


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_selection(records: list[dict], examples: dict[str, dict]) -> dict:
    rows = []
    combos = Counter()
    for record in records:
        selected = list(record.get("selected_evidence", []))
        spans = target_spans(examples[record["example_id"]])
        iop = best_metric(selected, spans, interval_iop)
        iou = best_metric(selected, spans, interval_iou)
        combos[selected_combo(selected)] += 1
        rows.append(
            {
                "cost": float(record.get("selected_evidence_cost", selected_cost(selected))),
                "count": float(record.get("selected_evidence_count", len(selected))),
                "iop": iop,
                "iou": iou,
                "pair_iou": pairwise_interval_iou(selected),
            }
        )
    return {
        "n": len(rows),
        "cost": average([row["cost"] for row in rows]),
        "count": average([row["count"] for row in rows]),
        "m_iop": average([row["iop"] for row in rows]),
        "iop_at_0_3": average([1.0 if row["iop"] >= 0.3 else 0.0 for row in rows]),
        "iop_at_0_5": average([1.0 if row["iop"] >= 0.5 else 0.0 for row in rows]),
        "m_iou": average([row["iou"] for row in rows]),
        "iou_at_0_3": average([1.0 if row["iou"] >= 0.3 else 0.0 for row in rows]),
        "iou_at_0_5": average([1.0 if row["iou"] >= 0.5 else 0.0 for row in rows]),
        "mean_pairwise_iou": average([row["pair_iou"] for row in rows]),
        "combos": dict(combos.most_common()),
    }


def candidate_pool_recall(pool_records: list[dict], examples: dict[str, dict]) -> dict[str, dict]:
    output = {}
    for k in (1, 2, 3, 6):
        rows = []
        for record in pool_records:
            selected = list(record.get("selected_evidence", []))[:k]
            spans = target_spans(examples[record["example_id"]])
            iop = best_metric(selected, spans, interval_iop)
            iou = best_metric(selected, spans, interval_iou)
            rows.append((iop, iou))
        output[f"recall_at_{k}"] = {
            "m_iop": average([row[0] for row in rows]),
            "iop_at_0_3": average([1.0 if row[0] >= 0.3 else 0.0 for row in rows]),
            "iop_at_0_5": average([1.0 if row[0] >= 0.5 else 0.0 for row in rows]),
            "m_iou": average([row[1] for row in rows]),
            "iou_at_0_3": average([1.0 if row[1] >= 0.3 else 0.0 for row in rows]),
            "iou_at_0_5": average([1.0 if row[1] >= 0.5 else 0.0 for row in rows]),
        }
    return output


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def format_float(value: float) -> str:
    return f"{value:.3f}"


def write_candidate_recall(output_dir: Path, split: str, recall: dict[str, dict]) -> str:
    rows = []
    for label, metrics in recall.items():
        rows.append(
            [
                label.replace("recall_at_", "@"),
                format_float(metrics["m_iop"]),
                format_float(metrics["iop_at_0_3"]),
                format_float(metrics["iop_at_0_5"]),
                format_float(metrics["m_iou"]),
                format_float(metrics["iou_at_0_3"]),
                format_float(metrics["iou_at_0_5"]),
            ]
        )
    table = markdown_table(
        ["Pool Prefix", "mIoP", "IoP@0.3", "IoP@0.5", "mIoU", "IoU@0.3", "IoU@0.5"],
        rows,
    )
    (output_dir / f"{split}_candidate_pool_recall.md").write_text(table, encoding="utf-8")
    return table


def write_selection_summary(output_dir: Path, split: str, summaries: dict[str, dict]) -> str:
    rows = []
    for label, metrics in summaries.items():
        top_combo = next(iter(metrics["combos"].items())) if metrics["combos"] else ("", 0)
        rows.append(
            [
                label,
                str(metrics["n"]),
                format_float(metrics["cost"]),
                format_float(metrics["count"]),
                format_float(metrics["m_iop"]),
                format_float(metrics["iop_at_0_5"]),
                format_float(metrics["m_iou"]),
                format_float(metrics["iou_at_0_5"]),
                format_float(metrics["mean_pairwise_iou"]),
                f"{top_combo[0]} ({top_combo[1]})",
            ]
        )
    table = markdown_table(
        [
            "Method",
            "N",
            "Cost",
            "Count",
            "mIoP",
            "IoP@0.5",
            "mIoU",
            "IoU@0.5",
            "Pair IoU",
            "Top Combo",
        ],
        rows,
    )
    (output_dir / f"{split}_selection_summary.md").write_text(table, encoding="utf-8")
    return table


def nms_identity_report(root: Path, split: str) -> dict[str, object]:
    selection_dir = root / "outputs/followup_strengthening/selections"
    plain_path = root / f"outputs/router_mlp/predictions/{split}_router_mlp.jsonl"
    plain = load_by_id(plain_path)
    report = {}
    for suffix in ("0p0", "0p3", "0p5"):
        path = selection_dir / f"{split}_router_mlp_top2_nms{suffix}.jsonl"
        records = load_by_id(path)
        same_as_plain = 0
        changed = 0
        for example_id, record in records.items():
            selected = [item["evidence_id"] for item in record["selected_evidence"]]
            base = [item["evidence_id"] for item in plain[example_id]["selected_evidence"]]
            if selected == base:
                same_as_plain += 1
            else:
                changed += 1
        report[f"nms{suffix}"] = {
            "same_as_plain": same_as_plain,
            "changed": changed,
            "num_examples": len(records),
        }
    nms0 = load_by_id(selection_dir / f"{split}_router_mlp_top2_nms0p0.jsonl")
    nms03 = load_by_id(selection_dir / f"{split}_router_mlp_top2_nms0p3.jsonl")
    report["nms0p0_same_as_nms0p3"] = sum(
        [
            [item["evidence_id"] for item in nms0[example_id]["selected_evidence"]]
            == [item["evidence_id"] for item in nms03[example_id]["selected_evidence"]]
            for example_id in nms0
        ]
    )
    return report


def load_main_qwen_records(root: Path, split: str) -> dict[str, dict[str, dict]]:
    qwen_root = root / "outputs/vlm_qwen25vl_3b_full/predictions"
    follow_root = root / "outputs/followup_strengthening/qwen/predictions"
    return {
        "linear_min2": load_by_id(qwen_root / f"{split}_learned_policy_min2.jsonl"),
        "mlp_top2": load_by_id(qwen_root / f"{split}_router_mlp_top2.jsonl"),
        "mlp_nms": load_by_id(follow_root / f"{split}_router_mlp_top2_nms0p0.jsonl"),
        "oracle_top2": load_by_id(follow_root / f"{split}_oracle_iop_top2.jsonl"),
        "oracle_top3": load_by_id(follow_root / f"{split}_oracle_iop_top3.jsonl"),
        "fixed_f3_s3": load_by_id(qwen_root / f"{split}_fixed_f3_s3.jsonl"),
    }


def row_overlap(record: dict, example: dict) -> dict[str, float]:
    selected = list(record.get("selected_evidence", []))
    spans = target_spans(example)
    return {
        "iop": best_metric(selected, spans, interval_iop),
        "iou": best_metric(selected, spans, interval_iou),
    }


def answer_letter(index: int) -> str:
    return "ABCDE"[index] if 0 <= index < 5 else "?"


def evidence_brief(selected: list[dict]) -> str:
    parts = []
    for item in selected:
        interval = evidence_interval(item)
        if interval is None:
            span = "?"
        else:
            span = f"{interval[0]:.1f}-{interval[1]:.1f}s"
        parts.append(f"{item.get('evidence_id', '').split(':')[-2]}:{span}")
    return ", ".join(parts)


def qualitative_cases(root: Path, examples: dict[str, dict]) -> list[dict]:
    records = load_main_qwen_records(root, "test")
    cases = []
    for example_id, example in examples.items():
        if example_id not in records["mlp_nms"]:
            continue
        mlp = records["mlp_top2"][example_id]
        nms = records["mlp_nms"][example_id]
        oracle = records["oracle_top2"][example_id]
        fixed = records["fixed_f3_s3"][example_id]
        mlp_overlap = row_overlap(mlp, example)
        nms_overlap = row_overlap(nms, example)
        oracle_overlap = row_overlap(oracle, example)
        fixed_overlap = row_overlap(fixed, example)
        cases.append(
            {
                "example_id": example_id,
                "question": example["question"],
                "options": example["options"],
                "gold": int(example["answer_index"]),
                "question_type": example.get("metadata", {}).get("question_type", ""),
                "mlp_correct": float(mlp.get("correct", 0.0)),
                "nms_correct": float(nms.get("correct", 0.0)),
                "oracle_correct": float(oracle.get("correct", 0.0)),
                "fixed_correct": float(fixed.get("correct", 0.0)),
                "mlp_iop": mlp_overlap["iop"],
                "nms_iop": nms_overlap["iop"],
                "oracle_iop": oracle_overlap["iop"],
                "fixed_iop": fixed_overlap["iop"],
                "mlp_selection": evidence_brief(mlp["selected_evidence"]),
                "nms_selection": evidence_brief(nms["selected_evidence"]),
                "oracle_selection": evidence_brief(oracle["selected_evidence"]),
                "fixed_selection": evidence_brief(fixed["selected_evidence"]),
                "mlp_pred": int(mlp.get("predicted_index", -1)),
                "nms_pred": int(nms.get("predicted_index", -1)),
                "oracle_pred": int(oracle.get("predicted_index", -1)),
                "fixed_pred": int(fixed.get("predicted_index", -1)),
            }
        )

    used_ids: set[str] = set()

    def pick(predicate, key):
        filtered = [
            case
            for case in cases
            if case["example_id"] not in used_ids and predicate(case)
        ]
        if not filtered:
            return None
        chosen = max(filtered, key=key)
        used_ids.add(chosen["example_id"])
        return chosen

    return [
        pick(
            lambda c: c["nms_correct"] and c["nms_iop"] >= 0.5 and c["mlp_iop"] < 0.5,
            lambda c: c["nms_iop"] - c["mlp_iop"],
        ),
        pick(
            lambda c: c["oracle_correct"] and c["oracle_iop"] >= 0.5 and c["nms_iop"] < 0.5,
            lambda c: (c["oracle_iop"] - c["nms_iop"]) + (0.5 if not c["nms_correct"] else 0.0),
        ),
        pick(
            lambda c: c["fixed_correct"] and c["fixed_iop"] >= 0.5 and not c["nms_correct"],
            lambda c: (c["fixed_iop"] - c["nms_iop"]) + (0.25 if not c["oracle_correct"] else 0.0),
        ),
    ]


def write_qualitative(output_dir: Path, cases: list[dict | None]) -> str:
    titles = [
        "NMS Improves Grounding",
        "Oracle Shows Router Gap",
        "High-Coverage Fixed Evidence Helps",
    ]
    sections = ["# Qualitative Follow-Up Cases\n"]
    for title, case in zip(titles, cases, strict=True):
        if case is None:
            continue
        sections.append(f"## {title}\n")
        sections.append(f"- Example: `{case['example_id']}` / type `{case['question_type']}`\n")
        sections.append(f"- Question: {case['question']}\n")
        options = ", ".join(f"{answer_letter(i)}. {text}" for i, text in enumerate(case["options"]))
        sections.append(f"- Options: {options}\n")
        sections.append(f"- Gold: {answer_letter(case['gold'])}\n")
        for key, label in [
            ("mlp", "MLP top-2"),
            ("nms", "MLP top-2 + NMS"),
            ("oracle", "Oracle IoP top-2"),
            ("fixed", "Fixed 3+3"),
        ]:
            sections.append(
                f"- {label}: pred {answer_letter(case[key + '_pred'])}, "
                f"correct {int(case[key + '_correct'])}, "
                f"IoP {case[key + '_iop']:.3f}, "
                f"evidence {case[key + '_selection']}\n"
            )
        sections.append("\n")
    text = "".join(sections)
    (output_dir / "qualitative_cases.md").write_text(text, encoding="utf-8")
    return text


def write_pareto_plot(output_dir: Path, test_metrics_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    payload = json.loads(test_metrics_path.read_text(encoding="utf-8"))
    labels = []
    costs = []
    acc = []
    gqa = []
    for label, data in payload.items():
        metrics = data["metrics"]
        labels.append(label.replace("qwen_", "").replace("_", " "))
        costs.append(metrics["selected_evidence_cost"])
        acc.append(metrics["acc_qa"])
        gqa.append(metrics["acc_gqa_iop_0_5"])

    plt.figure(figsize=(7.5, 4.8))
    plt.scatter(costs, gqa, s=80, label="Acc@GQA")
    plt.scatter(costs, acc, s=80, marker="s", label="Acc@QA")
    for x, y, label in zip(costs, gqa, labels, strict=True):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.xlabel("Evidence cost")
    plt.ylabel("Score")
    plt.title("NExT-GQA Test Cost-Accuracy-Grounding Tradeoff")
    plt.ylim(0.15, 0.78)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "test_pareto_tradeoff.png", dpi=200)
    plt.savefig(output_dir / "test_pareto_tradeoff.pdf")
    plt.close()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_paths = {
        "val": {
            "examples": root / "candidates/val.visual_features.jsonl",
            "pool": root / "outputs/research_controls/predictions/fixed_f3_s3.jsonl",
        },
        "test": {
            "examples": root / "candidates/test.visual_features.jsonl",
            "pool": root / "outputs/vlm_qwen25vl_3b_full/predictions/test_fixed_f3_s3.jsonl",
        },
    }

    analysis = {}
    for split, paths in split_paths.items():
        examples = load_by_id(paths["examples"])
        pool_records = load_jsonl(paths["pool"])
        recall = candidate_pool_recall(pool_records, examples)
        write_candidate_recall(output_dir, split, recall)

        method_paths = {
            "linear_min2": root / f"outputs/vlm_qwen25vl_3b_full/predictions/{split}_learned_policy_min2.jsonl",
            "mlp_top2": root / f"outputs/vlm_qwen25vl_3b_full/predictions/{split}_router_mlp_top2.jsonl",
            "mlp_nms0": root / f"outputs/followup_strengthening/qwen/predictions/{split}_router_mlp_top2_nms0p0.jsonl",
            "oracle_iop_top2": root / f"outputs/followup_strengthening/qwen/predictions/{split}_oracle_iop_top2.jsonl",
            "oracle_iop_top3": root / f"outputs/followup_strengthening/qwen/predictions/{split}_oracle_iop_top3.jsonl",
            "fixed_f3_s3": root / f"outputs/vlm_qwen25vl_3b_full/predictions/{split}_fixed_f3_s3.jsonl",
        }
        selection_summaries = {
            label: summarize_selection(load_jsonl(path), examples)
            for label, path in method_paths.items()
        }

        nms_selection_paths = {
            "mlp_plain_selection": root / f"outputs/router_mlp/predictions/{split}_router_mlp.jsonl",
            "mlp_nms0_selection": root / f"outputs/followup_strengthening/selections/{split}_router_mlp_top2_nms0p0.jsonl",
            "mlp_nms03_selection": root / f"outputs/followup_strengthening/selections/{split}_router_mlp_top2_nms0p3.jsonl",
            "mlp_nms05_selection": root / f"outputs/followup_strengthening/selections/{split}_router_mlp_top2_nms0p5.jsonl",
        }
        nms_summaries = {
            label: summarize_selection(load_jsonl(path), examples)
            for label, path in nms_selection_paths.items()
        }
        write_selection_summary(output_dir, split, selection_summaries)
        write_selection_summary(output_dir, f"{split}_nms_ablation", nms_summaries)

        analysis[split] = {
            "candidate_pool_recall": recall,
            "selection_summary": selection_summaries,
            "nms_ablation": nms_summaries,
            "nms_identity": nms_identity_report(root, split),
        }

    test_examples = load_by_id(split_paths["test"]["examples"])
    analysis["qualitative_cases"] = qualitative_cases(root, test_examples)
    write_qualitative(output_dir, analysis["qualitative_cases"])
    write_pareto_plot(
        output_dir,
        root / "outputs/followup_strengthening/qwen/metrics/test_followup_qwen_compare.json",
    )

    (output_dir / "followup_analysis.json").write_text(
        json.dumps(analysis, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote analysis artifacts to {output_dir}")


if __name__ == "__main__":
    main()
