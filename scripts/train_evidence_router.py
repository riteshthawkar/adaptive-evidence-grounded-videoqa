import argparse
import json
from pathlib import Path

from adaptive_evidence_vqa.data.base import load_jsonl
from adaptive_evidence_vqa.data.normalized import load_normalized_examples
from adaptive_evidence_vqa.models.evidence_router import (
    EvidenceRouterConfig,
    groups_from_oracle_trace_records,
    groups_from_temporal_examples,
    save_evidence_router,
    train_evidence_router,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight MLP router that ranks evidence candidates before VLM answering."
    )
    parser.add_argument("--train-traces-path", help="Oracle-trace JSONL used for supervised router training.")
    parser.add_argument("--validation-traces-path", help="Oracle-trace JSONL used for router validation.")
    parser.add_argument(
        "--train-candidates-path",
        help="Candidate-pool JSONL with temporal labels. Used when --label-source temporal.",
    )
    parser.add_argument(
        "--validation-candidates-path",
        help="Candidate-pool JSONL with temporal labels. Used when --label-source temporal.",
    )
    parser.add_argument(
        "--label-source",
        choices=("oracle", "temporal"),
        default="oracle",
        help="Supervision source: oracle subsets from trace files, or temporal IoP labels from candidate files.",
    )
    parser.add_argument("--model-dir", required=True, help="Directory to store the trained router.")
    parser.add_argument("--train-limit", type=int, default=None, help="Optional number of train records/examples.")
    parser.add_argument("--validation-limit", type=int, default=None, help="Optional validation records/examples.")
    parser.add_argument("--top-k", type=int, default=2, help="Evidence items selected by the router at inference.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="MLP hidden dimension.")
    parser.add_argument("--dropout", type=float, default=0.10, help="MLP dropout probability.")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=512, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    parser.add_argument("--device", default="auto", help="Training device: auto, cpu, cuda, etc.")
    return parser.parse_args()


def limited_jsonl(path: str, limit: int | None) -> list[dict]:
    records = load_jsonl(path)
    if limit is not None:
        return records[:limit]
    return records


def limited_examples(path: str, limit: int | None):
    examples = load_normalized_examples(path)
    if limit is not None:
        return examples[:limit]
    return examples


def main() -> None:
    args = parse_args()
    if args.label_source == "oracle":
        if not args.train_traces_path:
            raise ValueError("--train-traces-path is required when --label-source oracle.")
        train_groups = groups_from_oracle_trace_records(
            limited_jsonl(args.train_traces_path, args.train_limit)
        )
        validation_groups = (
            groups_from_oracle_trace_records(
                limited_jsonl(args.validation_traces_path, args.validation_limit)
            )
            if args.validation_traces_path
            else []
        )
    else:
        if not args.train_candidates_path:
            raise ValueError("--train-candidates-path is required when --label-source temporal.")
        train_groups = groups_from_temporal_examples(
            limited_examples(args.train_candidates_path, args.train_limit)
        )
        validation_groups = (
            groups_from_temporal_examples(
                limited_examples(args.validation_candidates_path, args.validation_limit)
            )
            if args.validation_candidates_path
            else []
        )

    config = EvidenceRouterConfig(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        top_k=args.top_k,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
    )
    model, feature_extractor, result = train_evidence_router(
        train_groups=train_groups,
        validation_groups=validation_groups,
        config=config,
    )
    save_evidence_router(
        model,
        feature_extractor,
        config,
        args.model_dir,
        history=result.history,
    )

    summary = {
        "model_dir": args.model_dir,
        "label_source": args.label_source,
        "train_traces_path": args.train_traces_path,
        "validation_traces_path": args.validation_traces_path,
        "train_candidates_path": args.train_candidates_path,
        "validation_candidates_path": args.validation_candidates_path,
        "num_train_groups": len(train_groups),
        "num_validation_groups": len(validation_groups),
        "config": {
            "top_k": args.top_k,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "seed": args.seed,
            "device": args.device,
        },
        "best_epoch": result.best_epoch,
        "validation_metrics": result.validation_metrics,
        "history": result.history,
    }
    output_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote trained evidence router to {output_dir}")
    print(json.dumps({key: summary[key] for key in ("num_train_groups", "num_validation_groups", "best_epoch", "validation_metrics")}, indent=2))


if __name__ == "__main__":
    main()
