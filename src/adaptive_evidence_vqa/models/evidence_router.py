import hashlib
import json
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from adaptive_evidence_vqa.data.evidence_records import parse_evidence_record
from adaptive_evidence_vqa.eval.metrics import temporal_target_spans
from adaptive_evidence_vqa.models.answerer import Answerer
from adaptive_evidence_vqa.schemas import (
    AcquisitionStep,
    AcquisitionTrace,
    AnswerOption,
    EvidenceItem,
    ModelPrediction,
    QuestionExample,
)
from adaptive_evidence_vqa.utils import normalize_text


@dataclass(slots=True)
class EvidenceRouterConfig:
    visual_feature_dim: int = 512
    query_feature_dim: int = 256
    item_text_feature_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.10
    top_k: int = 2
    epochs: int = 20
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 4
    seed: int = 13
    device: str = "auto"


@dataclass(slots=True)
class RouterCandidate:
    example: QuestionExample
    item: EvidenceItem
    target: float
    rank: int
    pool_size: int
    modality_rank: int
    modality_pool_size: int


@dataclass(slots=True)
class RouterGroup:
    example_id: str
    candidates: list[RouterCandidate]


@dataclass(slots=True)
class RouterTrainingResult:
    history: list[dict[str, float]]
    best_epoch: int
    validation_metrics: dict[str, float]


class EvidenceRouterFeatureExtractor:
    numeric_feature_names: tuple[str, ...] = (
        "retrieval_score",
        "acquisition_cost",
        "start_fraction",
        "end_fraction",
        "center_fraction",
        "duration_fraction",
        "global_rank_fraction",
        "modality_rank_fraction",
        "pool_size_fraction",
        "modality_pool_size_fraction",
        "question_length_fraction",
        "num_options_fraction",
        "subtitle_modality",
        "frame_modality",
        "segment_modality",
    )

    def __init__(
        self,
        visual_feature_dim: int = 512,
        query_feature_dim: int = 256,
        item_text_feature_dim: int = 128,
        feature_cache_size: int = 512,
    ) -> None:
        self.visual_feature_dim = visual_feature_dim
        self.query_feature_dim = query_feature_dim
        self.item_text_feature_dim = item_text_feature_dim
        self.feature_cache_size = feature_cache_size
        self._feature_cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    @property
    def total_dim(self) -> int:
        return (
            self.visual_feature_dim
            + self.query_feature_dim
            + self.item_text_feature_dim
            + len(self.numeric_feature_names)
        )

    def _hash_index(self, key: str, dim: int) -> int:
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little") % dim

    def _add_hashed_tokens(
        self,
        vector: np.ndarray,
        *,
        offset: int,
        dim: int,
        prefix: str,
        text: str,
    ) -> None:
        for token in set(normalize_text(text).split()):
            if token:
                vector[offset + self._hash_index(f"{prefix}:{token}", dim)] += 1.0

    def _load_visual_store(self, path: str) -> dict[str, np.ndarray] | None:
        if not path:
            return None
        cached = self._feature_cache.get(path)
        if cached is not None:
            self._feature_cache.move_to_end(path)
            return cached

        feature_path = Path(path)
        if not feature_path.exists():
            return None
        with np.load(feature_path) as loaded:
            store = {key: np.asarray(loaded[key], dtype=np.float32) for key in loaded.files}

        self._feature_cache[path] = store
        if len(self._feature_cache) > self.feature_cache_size:
            self._feature_cache.popitem(last=False)
        return store

    def _visual_feature(self, item: EvidenceItem) -> np.ndarray:
        vector = np.zeros(self.visual_feature_dim, dtype=np.float32)
        path = str(item.metadata.get("visual_feature_path", "") or "")
        feature_index = item.metadata.get("feature_index")
        if feature_index is None:
            return vector

        store = self._load_visual_store(path)
        if store is None:
            return vector

        key = "frame_embeddings" if item.modality.value == "frame" else "segment_embeddings"
        array = store.get(key)
        if array is None:
            return vector

        index = int(feature_index)
        if index < 0 or index >= len(array):
            return vector

        raw = np.asarray(array[index], dtype=np.float32).reshape(-1)
        length = min(len(raw), self.visual_feature_dim)
        vector[:length] = raw[:length]
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
        return vector

    def transform(
        self,
        example: QuestionExample,
        item: EvidenceItem,
        *,
        rank: int,
        pool_size: int,
        modality_rank: int,
        modality_pool_size: int,
    ) -> np.ndarray:
        vector = np.zeros(self.total_dim, dtype=np.float32)
        vector[: self.visual_feature_dim] = self._visual_feature(item)

        query_offset = self.visual_feature_dim
        query_text = " ".join([example.question] + [option.text for option in example.options])
        self._add_hashed_tokens(
            vector,
            offset=query_offset,
            dim=self.query_feature_dim,
            prefix="query",
            text=query_text,
        )

        item_offset = query_offset + self.query_feature_dim
        self._add_hashed_tokens(
            vector,
            offset=item_offset,
            dim=self.item_text_feature_dim,
            prefix=f"item:{item.modality.value}",
            text=item.text,
        )

        duration = float(example.metadata.get("video_duration", 0.0) or 0.0)
        if duration <= 0.0:
            clip_span = example.metadata.get("clip_span")
            if isinstance(clip_span, (list, tuple)) and len(clip_span) == 2:
                duration = max(float(clip_span[1]), float(clip_span[0]), 1.0)
            else:
                duration = 1.0

        start_time = item.start_time if item.start_time is not None else item.end_time
        end_time = item.end_time if item.end_time is not None else item.start_time
        if start_time is None:
            start_time = 0.0
        if end_time is None:
            end_time = start_time
        start, end = sorted((float(start_time), float(end_time)))
        center = 0.5 * (start + end)
        item_duration = max(0.0, end - start)
        modality = item.modality.value
        modality_one_hot = (
            1.0 if modality == "subtitle" else 0.0,
            1.0 if modality == "frame" else 0.0,
            1.0 if modality == "segment" else 0.0,
        )
        question_length = len(normalize_text(example.question).split())

        numeric = np.asarray(
            [
                float(item.retrieval_score),
                float(item.acquisition_cost) / 7.5,
                start / duration,
                end / duration,
                center / duration,
                item_duration / duration,
                float(rank) / max(float(pool_size - 1), 1.0),
                float(modality_rank) / max(float(modality_pool_size - 1), 1.0),
                float(pool_size) / 12.0,
                float(modality_pool_size) / 6.0,
                float(question_length) / 32.0,
                float(len(example.options)) / 5.0,
                *modality_one_hot,
            ],
            dtype=np.float32,
        )
        vector[item_offset + self.item_text_feature_dim :] = numeric

        hashed_start = self.visual_feature_dim
        hashed_end = self.visual_feature_dim + self.query_feature_dim + self.item_text_feature_dim
        hashed_norm = float(np.linalg.norm(vector[hashed_start:hashed_end]))
        if hashed_norm > 0.0:
            vector[hashed_start:hashed_end] /= hashed_norm
        return vector


def _item_interval(item: EvidenceItem) -> tuple[float, float] | None:
    if item.start_time is None and item.end_time is None:
        return None
    start = item.start_time if item.start_time is not None else item.end_time
    end = item.end_time if item.end_time is not None else item.start_time
    if start is None or end is None:
        return None
    return tuple(sorted((float(start), float(end))))


def temporal_iop_target(item: EvidenceItem, target_spans: list[tuple[float, float]]) -> float:
    interval = _item_interval(item)
    if interval is None or not target_spans:
        return 0.0
    start, end = interval
    predicted_length = max(end - start, 1e-6)
    if item.modality.value == "frame":
        return 1.0 if any(span_start <= start <= span_end for span_start, span_end in target_spans) else 0.0
    best = 0.0
    for span_start, span_end in target_spans:
        intersection = max(0.0, min(end, span_end) - max(start, span_start))
        best = max(best, intersection / predicted_length)
    return float(best)


def question_from_trace_record(record: dict) -> QuestionExample:
    temporal_grounding = record.get("temporal_grounding")
    return QuestionExample(
        example_id=record["example_id"],
        video_id=record["video_id"],
        question=record["question"],
        options=tuple(
            AnswerOption(index=index, text=text)
            for index, text in enumerate(record["options"])
        ),
        answer_index=record.get("answer_index"),
        temporal_grounding=tuple(temporal_grounding) if temporal_grounding is not None else None,
        metadata=dict(record.get("metadata", {})),
    )


def _router_candidates_from_items(
    example: QuestionExample,
    items: list[EvidenceItem],
    targets_by_id: dict[str, float],
) -> list[RouterCandidate]:
    modality_offsets = {"subtitle": 0, "frame": 0, "segment": 0}
    modality_sizes = {
        modality: sum(1 for item in items if item.modality.value == modality)
        for modality in modality_offsets
    }
    candidates = []
    for rank, item in enumerate(items):
        modality = item.modality.value
        modality_rank = modality_offsets[modality]
        modality_offsets[modality] += 1
        candidates.append(
            RouterCandidate(
                example=example,
                item=item,
                target=float(targets_by_id.get(item.evidence_id, 0.0)),
                rank=rank,
                pool_size=len(items),
                modality_rank=modality_rank,
                modality_pool_size=modality_sizes[modality],
            )
        )
    return candidates


def groups_from_oracle_trace_records(records: list[dict]) -> list[RouterGroup]:
    groups: list[RouterGroup] = []
    for record in records:
        example = question_from_trace_record(record)
        seed_evidence = [parse_evidence_record(item) for item in record.get("seed_evidence", [])]
        if not seed_evidence:
            continue

        oracle_ids = {item["evidence_id"] for item in record.get("oracle_subset", [])}
        if not oracle_ids:
            oracle_ids = {
                step["selected_evidence_id"]
                for step in record.get("trace", [])
                if step.get("selected_evidence_id")
            }
        if not oracle_ids:
            continue

        targets_by_id = {item.evidence_id: 1.0 for item in seed_evidence if item.evidence_id in oracle_ids}
        groups.append(
            RouterGroup(
                example_id=example.example_id,
                candidates=_router_candidates_from_items(example, seed_evidence, targets_by_id),
            )
        )
    return groups


def groups_from_temporal_examples(examples: list[QuestionExample]) -> list[RouterGroup]:
    groups: list[RouterGroup] = []
    for example in examples:
        target_spans = temporal_target_spans(example.temporal_grounding, example.metadata)
        if not target_spans:
            continue
        items = sorted(
            list(example.evidence_pool),
            key=lambda item: (-float(item.retrieval_score), item.modality.value, item.evidence_id),
        )
        targets = {item.evidence_id: temporal_iop_target(item, target_spans) for item in items}
        if max(targets.values(), default=0.0) <= 0.0:
            continue
        groups.append(
            RouterGroup(
                example_id=example.example_id,
                candidates=_router_candidates_from_items(example, items, targets),
            )
        )
    return groups


class EvidenceRouterNet:
    def __new__(cls, input_dim: int, hidden_dim: int = 256, dropout: float = 0.10):
        import torch.nn as nn

        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 32)),
            nn.GELU(),
            nn.Linear(max(hidden_dim // 2, 32), 1),
        )


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def featurize_groups(
    groups: list[RouterGroup],
    feature_extractor: EvidenceRouterFeatureExtractor,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    features: list[np.ndarray] = []
    targets: list[float] = []
    spans: list[tuple[int, int]] = []
    for group in groups:
        start = len(features)
        for candidate in group.candidates:
            features.append(
                feature_extractor.transform(
                    candidate.example,
                    candidate.item,
                    rank=candidate.rank,
                    pool_size=candidate.pool_size,
                    modality_rank=candidate.modality_rank,
                    modality_pool_size=candidate.modality_pool_size,
                )
            )
            targets.append(candidate.target)
        spans.append((start, len(features)))
    if not features:
        return (
            np.zeros((0, feature_extractor.total_dim), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            spans,
        )
    return np.stack(features).astype(np.float32), np.asarray(targets, dtype=np.float32), spans


def router_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    spans: list[tuple[int, int]],
    *,
    top_k: int,
) -> dict[str, float]:
    if len(targets) == 0:
        return {
            "loss": 0.0,
            "positive_rate": 0.0,
            "topk_hit": 0.0,
            "topk_mean_target": 0.0,
            "topk_best_target": 0.0,
        }

    probabilities = 1.0 / (1.0 + np.exp(-logits))
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    loss = -np.mean(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped))
    topk_hits = []
    topk_mean_targets = []
    topk_best_targets = []
    for start, end in spans:
        if end <= start:
            continue
        group_logits = logits[start:end]
        group_targets = targets[start:end]
        k = min(max(top_k, 1), len(group_logits))
        selected = np.argsort(-group_logits)[:k]
        selected_targets = group_targets[selected]
        topk_hits.append(1.0 if float(np.max(selected_targets)) > 0.0 else 0.0)
        topk_mean_targets.append(float(np.mean(selected_targets)))
        topk_best_targets.append(float(np.max(selected_targets)))

    return {
        "loss": float(loss),
        "positive_rate": float(np.mean(targets > 0.0)),
        "topk_hit": float(np.mean(topk_hits)) if topk_hits else 0.0,
        "topk_mean_target": float(np.mean(topk_mean_targets)) if topk_mean_targets else 0.0,
        "topk_best_target": float(np.mean(topk_best_targets)) if topk_best_targets else 0.0,
    }


def train_evidence_router(
    train_groups: list[RouterGroup],
    validation_groups: list[RouterGroup] | None = None,
    config: EvidenceRouterConfig | None = None,
) -> tuple[object, EvidenceRouterFeatureExtractor, RouterTrainingResult]:
    import torch
    import torch.nn.functional as F

    if not train_groups:
        raise ValueError("train_evidence_router requires at least one training group.")

    resolved = config or EvidenceRouterConfig()
    torch.manual_seed(resolved.seed)
    rng = np.random.default_rng(resolved.seed)
    device = resolve_device(resolved.device)

    feature_extractor = EvidenceRouterFeatureExtractor(
        visual_feature_dim=resolved.visual_feature_dim,
        query_feature_dim=resolved.query_feature_dim,
        item_text_feature_dim=resolved.item_text_feature_dim,
    )
    train_features, train_targets, _ = featurize_groups(train_groups, feature_extractor)
    val_features, val_targets, val_spans = featurize_groups(validation_groups or [], feature_extractor)

    model = EvidenceRouterNet(
        input_dim=feature_extractor.total_dim,
        hidden_dim=resolved.hidden_dim,
        dropout=resolved.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=resolved.learning_rate,
        weight_decay=resolved.weight_decay,
    )

    positive_count = float(np.sum(train_targets > 0.0))
    negative_count = float(len(train_targets) - positive_count)
    pos_weight_value = negative_count / max(positive_count, 1.0)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    x_train = torch.tensor(train_features, dtype=torch.float32)
    y_train = torch.tensor(train_targets, dtype=torch.float32).unsqueeze(1)
    x_val = torch.tensor(val_features, dtype=torch.float32, device=device) if len(val_features) else None

    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_epoch = 0
    best_score = float("-inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, resolved.epochs + 1):
        model.train()
        order = rng.permutation(len(train_features))
        batch_losses = []
        for batch_start in range(0, len(order), resolved.batch_size):
            batch_indices = order[batch_start : batch_start + resolved.batch_size]
            batch_x = x_train[batch_indices].to(device)
            batch_y = y_train[batch_indices].to(device)
            logits = model(batch_x)
            loss = F.binary_cross_entropy_with_logits(logits, batch_y, pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            if x_val is not None:
                val_logits = model(x_val).squeeze(1).detach().cpu().numpy()
                val_metrics = router_metrics(
                    val_logits,
                    val_targets,
                    val_spans,
                    top_k=resolved.top_k,
                )
            else:
                train_sample = torch.tensor(train_features[: min(len(train_features), 4096)], dtype=torch.float32, device=device)
                train_logits = model(train_sample).squeeze(1).detach().cpu().numpy()
                val_metrics = router_metrics(
                    train_logits,
                    train_targets[: len(train_logits)],
                    [(0, len(train_logits))],
                    top_k=resolved.top_k,
                )

        history_entry = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(batch_losses)) if batch_losses else 0.0,
            "validation_loss": val_metrics["loss"],
            "validation_topk_hit": val_metrics["topk_hit"],
            "validation_topk_mean_target": val_metrics["topk_mean_target"],
            "validation_topk_best_target": val_metrics["topk_best_target"],
        }
        history.append(history_entry)

        score = val_metrics["topk_best_target"] + val_metrics["topk_hit"] - val_metrics["loss"]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= resolved.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    final_metrics = {}
    if x_val is not None:
        with torch.no_grad():
            final_logits = model(x_val).squeeze(1).detach().cpu().numpy()
        final_metrics = router_metrics(final_logits, val_targets, val_spans, top_k=resolved.top_k)

    return model, feature_extractor, RouterTrainingResult(
        history=history,
        best_epoch=best_epoch,
        validation_metrics=final_metrics,
    )


def save_evidence_router(
    model: object,
    feature_extractor: EvidenceRouterFeatureExtractor,
    config: EvidenceRouterConfig,
    output_dir: str | Path,
    *,
    history: list[dict[str, float]] | None = None,
) -> None:
    import torch

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "router.pt")
    metadata = {
        "model_name": "mlp_evidence_router",
        "config": asdict(config),
        "feature_extractor": {
            "visual_feature_dim": feature_extractor.visual_feature_dim,
            "query_feature_dim": feature_extractor.query_feature_dim,
            "item_text_feature_dim": feature_extractor.item_text_feature_dim,
            "numeric_feature_names": list(feature_extractor.numeric_feature_names),
            "total_dim": feature_extractor.total_dim,
        },
        "history": history or [],
    }
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


class MLPEvidenceRouterPolicy:
    model_name = "mlp_evidence_router"

    def __init__(
        self,
        model: object,
        feature_extractor: EvidenceRouterFeatureExtractor,
        answerer: Answerer,
        config: EvidenceRouterConfig,
        device: str,
    ) -> None:
        self.model = model
        self.feature_extractor = feature_extractor
        self.answerer = answerer
        self.config = config
        self.device = device

    @classmethod
    def load(
        cls,
        model_dir: str | Path,
        answerer: Answerer,
    ) -> "MLPEvidenceRouterPolicy":
        import torch

        path = Path(model_dir)
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        config = EvidenceRouterConfig(**metadata["config"])
        device = resolve_device(config.device)
        feature_meta = metadata["feature_extractor"]
        feature_extractor = EvidenceRouterFeatureExtractor(
            visual_feature_dim=int(feature_meta["visual_feature_dim"]),
            query_feature_dim=int(feature_meta["query_feature_dim"]),
            item_text_feature_dim=int(feature_meta["item_text_feature_dim"]),
        )
        model = EvidenceRouterNet(
            input_dim=feature_extractor.total_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        ).to(device)
        model.load_state_dict(torch.load(path / "router.pt", map_location=device))
        model.eval()
        return cls(
            model=model,
            feature_extractor=feature_extractor,
            answerer=answerer,
            config=config,
            device=device,
        )

    def _candidate_features(
        self,
        example: QuestionExample,
        candidate_pool: dict[str, tuple[EvidenceItem, ...]],
    ) -> tuple[list[EvidenceItem], np.ndarray]:
        items: list[EvidenceItem] = []
        for modality in ("subtitle", "frame", "segment"):
            items.extend(candidate_pool.get(modality, ()))
        items = sorted(
            items,
            key=lambda item: (-float(item.retrieval_score), item.modality.value, item.evidence_id),
        )
        modality_offsets = {"subtitle": 0, "frame": 0, "segment": 0}
        modality_sizes = {
            modality: sum(1 for item in items if item.modality.value == modality)
            for modality in modality_offsets
        }
        features = []
        for rank, item in enumerate(items):
            modality = item.modality.value
            modality_rank = modality_offsets[modality]
            modality_offsets[modality] += 1
            features.append(
                self.feature_extractor.transform(
                    example,
                    item,
                    rank=rank,
                    pool_size=len(items),
                    modality_rank=modality_rank,
                    modality_pool_size=modality_sizes[modality],
                )
            )
        if not features:
            return items, np.zeros((0, self.feature_extractor.total_dim), dtype=np.float32)
        return items, np.stack(features).astype(np.float32)

    def score_items(
        self,
        example: QuestionExample,
        candidate_pool: dict[str, tuple[EvidenceItem, ...]],
    ) -> list[tuple[float, EvidenceItem]]:
        import torch

        items, features = self._candidate_features(example, candidate_pool)
        if len(items) == 0:
            return []
        with torch.no_grad():
            tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
            logits = self.model(tensor).squeeze(1).detach().cpu().numpy()
        return list(zip((float(value) for value in logits), items, strict=True))

    def run(
        self,
        example: QuestionExample,
        candidate_pool: dict[str, tuple[EvidenceItem, ...]],
        max_items: int = 6,
    ) -> AcquisitionTrace:
        scored = sorted(
            self.score_items(example, candidate_pool),
            key=lambda pair: pair[0],
            reverse=True,
        )
        selected = [item for _, item in scored[: min(self.config.top_k, max_items, len(scored))]]
        steps: list[AcquisitionStep] = []
        acquired: list[EvidenceItem] = []
        for step_index, item in enumerate(selected):
            acquired.append(item)
            score = next(score for score, candidate in scored if candidate.evidence_id == item.evidence_id)
            steps.append(
                AcquisitionStep(
                    step_index=step_index,
                    action=f"acquire_{item.modality.value}",
                    selected_item=item,
                    confidence_after_step=float(1.0 / (1.0 + np.exp(-score))),
                )
            )

        final_prediction: ModelPrediction = self.answerer.predict(example, tuple(acquired))
        steps.append(
            AcquisitionStep(
                step_index=len(steps),
                action="stop",
                selected_item=None,
                confidence_after_step=final_prediction.confidence,
            )
        )
        return AcquisitionTrace(steps=tuple(steps), final_prediction=final_prediction)
