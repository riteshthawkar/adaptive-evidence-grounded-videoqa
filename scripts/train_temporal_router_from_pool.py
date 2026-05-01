import argparse
import json
import random
from pathlib import Path

from adaptive_evidence_vqa.data.base import load_jsonl
from adaptive_evidence_vqa.data.evidence_records import parse_evidence_record
from adaptive_evidence_vqa.data.normalized import load_normalized_examples
from adaptive_evidence_vqa.eval.metrics import temporal_target_spans
from adaptive_evidence_vqa.models.evidence_router import (
    EvidenceRouterConfig,
    RouterGroup,
    _router_candidates_from_items,
    save_evidence_router,
    temporal_iop_target,
    train_evidence_router,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an MLP evidence router from temporal-overlap labels over an "
            "existing retrieved evidence pool such as fixed 3+3."
        )
    )
    parser.add_argument("--candidate-path", required=True)
    parser.add_argument("--pool-selection-path", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_pool_records(path: str | Path) -> dict[str, list]:
    pools = {}
    for record in load_jsonl(path):
        pools[record["example_id"]] = [
            parse_evidence_record(item) for item in record.get("selected_evidence", [])
        ]
    return pools


def groups_from_examples_and_pool(candidate_path: str, pool_selection_path: str) -> list[RouterGroup]:
    examples = load_normalized_examples(candidate_path)
    pools_by_id = load_pool_records(pool_selection_path)
    groups: list[RouterGroup] = []
    for example in examples:
        items = pools_by_id.get(example.example_id)
        if not items:
            continue
        target_spans = temporal_target_spans(example.temporal_grounding, example.metadata)
        if not target_spans:
            continue
        targets = {item.evidence_id: temporal_iop_target(item, target_spans) for item in items}
        if max(targets.values(), default=0.0) <= 0.0:
            continue
        ranked_items = sorted(
            items,
            key=lambda item: (-float(item.retrieval_score), item.modality.value, item.evidence_id),
        )
        groups.append(
            RouterGroup(
                example_id=example.example_id,
                candidates=_router_candidates_from_items(example, ranked_items, targets),
            )
        )
    return groups


def split_groups(groups: list[RouterGroup], *, validation_fraction: float, seed: int):
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    validation_size = int(round(len(shuffled) * validation_fraction))
    validation_groups = shuffled[:validation_size]
    train_groups = shuffled[validation_size:]
    return train_groups, validation_groups


def target_summary(groups: list[RouterGroup]) -> dict[str, float]:
    targets = [candidate.target for group in groups for candidate in group.candidates]
    positives = [value for value in targets if value > 0.0]
    return {
        "num_groups": len(groups),
        "num_candidates": len(targets),
        "positive_rate": len(positives) / len(targets) if targets else 0.0,
        "mean_target": sum(targets) / len(targets) if targets else 0.0,
        "mean_positive_target": sum(positives) / len(positives) if positives else 0.0,
    }


def main() -> None:
    args = parse_args()
    groups = groups_from_examples_and_pool(args.candidate_path, args.pool_selection_path)
    if not groups:
        raise ValueError("No temporal-supervision groups were created.")
    train_groups, validation_groups = split_groups(
        groups,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )

    config = EvidenceRouterConfig(
        top_k=args.top_k,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
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
        "label_source": "temporal_iop_from_retrieved_pool",
        "candidate_path": args.candidate_path,
        "pool_selection_path": args.pool_selection_path,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "train": target_summary(train_groups),
        "validation": target_summary(validation_groups),
        "config": {
            "top_k": args.top_k,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "device": args.device,
        },
        "best_epoch": result.best_epoch,
        "validation_metrics": result.validation_metrics,
        "history": result.history,
    }
    output_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
