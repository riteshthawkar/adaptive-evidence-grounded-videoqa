import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_METHOD_ORDER = [
    "qwen25vl_fixed_segment1",
    "qwen25vl_policy_min2",
    "qwen25vl_fixed_f3_s3",
]

METHOD_LABELS = {
    "qwen25vl_fixed_segment1": "Qwen one segment",
    "qwen25vl_policy_min2": "Qwen learned two-item",
    "qwen25vl_fixed_f3_s3": "Qwen fixed 3+3",
    "clip_fixed_segment1": "Frozen one segment",
    "clip_policy_min2": "Frozen learned two-item",
    "clip_fixed_f3_s3": "Frozen fixed 3+3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot accuracy-cost and grounded-accuracy-cost tradeoffs."
    )
    parser.add_argument("--summary-json", required=True, help="Metric summary JSON.")
    parser.add_argument("--output", required=True, help="Output figure path.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHOD_ORDER,
        help="Method labels to plot, in display order.",
    )
    parser.add_argument(
        "--title",
        default="NExT-GQA validation tradeoff",
        help="Figure title.",
    )
    return parser.parse_args()


def load_points(path: str | Path, methods: list[str]) -> list[tuple[str, dict]]:
    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    points = []
    for method in methods:
        if method not in summary:
            raise KeyError(f"Method {method!r} not found in {path}")
        points.append((method, summary[method]["metrics"]))
    return points


def annotate_points(axis, xs: list[float], ys: list[float], labels: list[str]) -> None:
    for x, y, label in zip(xs, ys, labels, strict=True):
        axis.annotate(
            label,
            (x, y),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=7,
        )


def main() -> None:
    args = parse_args()
    points = load_points(args.summary_json, args.methods)
    labels = [METHOD_LABELS.get(method, method) for method, _ in points]
    costs = [metrics["selected_evidence_cost"] for _, metrics in points]
    acc = [metrics["acc_qa"] for _, metrics in points]
    grounded_acc = [metrics["acc_gqa_iop_0_5"] for _, metrics in points]

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    fig.suptitle(args.title, y=1.03, fontsize=10)

    for axis, values, ylabel in [
        (axes[0], acc, "Answer accuracy"),
        (axes[1], grounded_acc, "Grounded answer accuracy"),
    ]:
        axis.plot(costs, values, color="#2f6f73", linewidth=1.5, marker="o", markersize=5)
        axis.set_xlabel("Average evidence cost")
        axis.set_ylabel(ylabel)
        axis.set_ylim(0.0, min(1.0, max(values) + 0.12))
        axis.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.9)
        annotate_points(axis, costs, values, labels)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
