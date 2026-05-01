import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from adaptive_evidence_vqa.retrieval.hybrid import ClipTextEncoder, TextEncoder
from adaptive_evidence_vqa.schemas import EvidenceItem, ModelPrediction, QuestionExample
from adaptive_evidence_vqa.utils import jaccard_overlap, normalize_text, softmax

ExampleWithEvidence = tuple[QuestionExample, tuple[EvidenceItem, ...]]


def _softmax_numpy(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=np.float32)
    shifted = values - np.max(values)
    exps = np.exp(shifted)
    return exps / np.clip(np.sum(exps), 1e-8, None)


@dataclass(slots=True)
class CalibratedMultimodalAnswererConfig:
    model_name: str = "openai/clip-vit-base-patch32"
    device: str | None = None
    epochs: int = 12
    batch_size: int = 64
    learning_rate: float = 0.2
    weight_decay: float = 1e-4
    patience: int = 3
    seed: int = 13


@dataclass(slots=True)
class CalibratedMultimodalFeatureExtractor:
    text_encoder: TextEncoder
    feature_names: tuple[str, ...] = field(
        default_factory=lambda: (
            "bias",
            "option_question_overlap",
            "option_token_count",
            "query_token_count",
            "frame_similarity_max",
            "frame_similarity_mean",
            "frame_similarity_top2_mean",
            "frame_retrieval_max",
            "frame_retrieval_mean",
            "segment_similarity_max",
            "segment_similarity_mean",
            "segment_similarity_top2_mean",
            "segment_retrieval_max",
            "segment_retrieval_mean",
            "visual_similarity_max",
            "visual_similarity_mean",
            "visual_similarity_margin_segment_frame",
            "evidence_count",
            "evidence_cost",
        )
    )
    _feature_cache: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)

    @property
    def total_dim(self) -> int:
        return len(self.feature_names)

    def _load_feature_file(self, feature_path: str) -> dict[str, np.ndarray]:
        if feature_path not in self._feature_cache:
            with np.load(feature_path) as data:
                self._feature_cache[feature_path] = {
                    key: data[key].astype(np.float32, copy=False)
                    for key in data.files
                }
        return self._feature_cache[feature_path]

    def _visual_feature(self, evidence: EvidenceItem) -> np.ndarray | None:
        feature_path = evidence.metadata.get("visual_feature_path")
        feature_index = evidence.metadata.get("feature_index")
        if not isinstance(feature_path, str) or not isinstance(feature_index, int):
            return None

        if evidence.modality.value == "frame":
            array_name = "frame_embeddings"
        elif evidence.modality.value == "segment":
            array_name = "segment_embeddings"
        else:
            return None

        feature_store = self._load_feature_file(feature_path)
        if array_name not in feature_store:
            return None
        matrix = feature_store[array_name]
        if feature_index < 0 or feature_index >= len(matrix):
            return None
        return matrix[feature_index]

    @staticmethod
    def _score_stats(scores: list[float]) -> tuple[float, float, float]:
        if not scores:
            return 0.0, 0.0, 0.0
        ordered = sorted(scores, reverse=True)
        top2 = ordered[:2]
        return (
            float(ordered[0]),
            float(sum(scores) / len(scores)),
            float(sum(top2) / len(top2)),
        )

    @staticmethod
    def _retrieval_stats(items: tuple[EvidenceItem, ...]) -> tuple[float, float]:
        if not items:
            return 0.0, 0.0
        scores = [float(item.retrieval_score) for item in items]
        return max(scores), sum(scores) / len(scores)

    def transform_options(
        self,
        example: QuestionExample,
        evidence: tuple[EvidenceItem, ...],
    ) -> np.ndarray:
        query_texts = [f"{example.question} {option.text}" for option in example.options]
        query_embeddings = self.text_encoder.encode(query_texts)

        frames = tuple(item for item in evidence if item.modality.value == "frame")
        segments = tuple(item for item in evidence if item.modality.value == "segment")
        frame_features = [self._visual_feature(item) for item in frames]
        segment_features = [self._visual_feature(item) for item in segments]
        frame_features = [feature for feature in frame_features if feature is not None]
        segment_features = [feature for feature in segment_features if feature is not None]
        frame_retrieval_max, frame_retrieval_mean = self._retrieval_stats(frames)
        segment_retrieval_max, segment_retrieval_mean = self._retrieval_stats(segments)
        evidence_cost = sum(item.acquisition_cost for item in evidence)

        rows = []
        for option, query_text, query_embedding in zip(
            example.options,
            query_texts,
            query_embeddings,
            strict=True,
        ):
            frame_scores = [float(np.dot(query_embedding, feature)) for feature in frame_features]
            segment_scores = [float(np.dot(query_embedding, feature)) for feature in segment_features]
            all_visual_scores = frame_scores + segment_scores

            frame_max, frame_mean, frame_top2 = self._score_stats(frame_scores)
            segment_max, segment_mean, segment_top2 = self._score_stats(segment_scores)
            visual_max, visual_mean, _ = self._score_stats(all_visual_scores)
            option_tokens = normalize_text(option.text).split()
            query_tokens = normalize_text(query_text).split()
            rows.append(
                [
                    1.0,
                    jaccard_overlap(example.question, option.text),
                    float(len(option_tokens)),
                    float(len(query_tokens)),
                    frame_max,
                    frame_mean,
                    frame_top2,
                    frame_retrieval_max,
                    frame_retrieval_mean,
                    segment_max,
                    segment_mean,
                    segment_top2,
                    segment_retrieval_max,
                    segment_retrieval_mean,
                    visual_max,
                    visual_mean,
                    segment_max - frame_max,
                    float(len(evidence)),
                    float(evidence_cost),
                ]
            )

        features = np.asarray(rows, dtype=np.float32)
        features[:, 2] /= 12.0
        features[:, 3] /= 36.0
        features[:, 17] /= 12.0
        features[:, 18] /= 12.0
        return features


class CalibratedMultimodalAnswerer:
    model_name = "calibrated_multimodal"

    def __init__(
        self,
        weights: np.ndarray,
        feature_extractor: CalibratedMultimodalFeatureExtractor,
        config: CalibratedMultimodalAnswererConfig,
        history: list[dict[str, float]] | None = None,
    ) -> None:
        self.weights = weights.astype(np.float32, copy=False)
        self.feature_extractor = feature_extractor
        self.config = config
        self.history = history or []

    def option_scores(
        self,
        example: QuestionExample,
        evidence: tuple[EvidenceItem, ...],
    ) -> np.ndarray:
        return self.feature_extractor.transform_options(example, evidence) @ self.weights

    def predict(
        self,
        example: QuestionExample,
        evidence: tuple[EvidenceItem, ...],
    ) -> ModelPrediction:
        scores = self.option_scores(example, evidence)
        probabilities = _softmax_numpy(scores)
        predicted_index = int(np.argmax(scores)) if scores.size else 0
        return ModelPrediction(
            predicted_index=predicted_index,
            option_scores=tuple(float(score) for score in scores),
            confidence=float(probabilities[predicted_index]) if probabilities.size else 0.0,
            supporting_evidence=evidence,
        )

    def save(self, model_dir: str | Path) -> None:
        output_dir = Path(model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(output_dir / "weights.npz", weights=self.weights)
        metadata = {
            "model_name": self.model_name,
            "config": asdict(self.config),
            "feature_names": list(self.feature_extractor.feature_names),
            "history": self.history,
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        model_dir: str | Path,
        *,
        device: str | None = None,
    ) -> "CalibratedMultimodalAnswerer":
        input_dir = Path(model_dir)
        metadata = json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))
        config = CalibratedMultimodalAnswererConfig(**metadata["config"])
        if device is not None:
            config.device = device
        text_encoder = ClipTextEncoder(model_name=config.model_name, device=config.device)
        feature_extractor = CalibratedMultimodalFeatureExtractor(
            text_encoder=text_encoder,
            feature_names=tuple(metadata["feature_names"]),
        )
        weights = np.load(input_dir / "weights.npz")["weights"]
        return cls(
            weights=weights,
            feature_extractor=feature_extractor,
            config=config,
            history=metadata.get("history", []),
        )

    @classmethod
    def fit(
        cls,
        train_examples: list[ExampleWithEvidence],
        validation_examples: list[ExampleWithEvidence] | None = None,
        config: CalibratedMultimodalAnswererConfig | None = None,
        text_encoder: TextEncoder | None = None,
    ) -> "CalibratedMultimodalAnswerer":
        if not train_examples:
            raise ValueError("CalibratedMultimodalAnswerer.fit requires at least one training example.")

        resolved_config = config or CalibratedMultimodalAnswererConfig()
        extractor = CalibratedMultimodalFeatureExtractor(
            text_encoder=text_encoder
            or ClipTextEncoder(model_name=resolved_config.model_name, device=resolved_config.device)
        )
        train_features, train_labels = cls._prepare_dataset(train_examples, extractor)
        validation_features, validation_labels = cls._prepare_dataset(validation_examples or [], extractor)

        rng = np.random.default_rng(resolved_config.seed)
        weights = np.zeros(extractor.total_dim, dtype=np.float32)
        best_weights = weights.copy()
        best_score = float("-inf")
        best_epoch = 0
        epochs_without_improvement = 0
        history: list[dict[str, float]] = []

        for epoch in range(1, resolved_config.epochs + 1):
            order = rng.permutation(len(train_features))
            for batch_start in range(0, len(order), resolved_config.batch_size):
                batch_indices = order[batch_start : batch_start + resolved_config.batch_size]
                gradient = np.zeros_like(weights)
                for index in batch_indices:
                    option_features = train_features[index]
                    label = train_labels[index]
                    scores = option_features @ weights
                    probabilities = _softmax_numpy(scores)
                    probabilities[label] -= 1.0
                    gradient += probabilities @ option_features

                gradient /= max(len(batch_indices), 1)
                gradient += resolved_config.weight_decay * weights
                weights -= resolved_config.learning_rate * gradient

            train_metrics = cls._dataset_metrics(train_features, train_labels, weights)
            validation_metrics = (
                cls._dataset_metrics(validation_features, validation_labels, weights)
                if validation_features
                else train_metrics
            )
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "validation_loss": validation_metrics["loss"],
                    "validation_accuracy": validation_metrics["accuracy"],
                }
            )

            score = validation_metrics["accuracy"] - validation_metrics["loss"]
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_weights = weights.copy()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= resolved_config.patience:
                break

        model = cls(
            weights=best_weights,
            feature_extractor=extractor,
            config=resolved_config,
            history=history,
        )
        model.history.append({"best_epoch": float(best_epoch)})
        return model

    @staticmethod
    def _prepare_dataset(
        examples: list[ExampleWithEvidence],
        extractor: CalibratedMultimodalFeatureExtractor,
    ) -> tuple[list[np.ndarray], list[int]]:
        features = []
        labels = []
        for example, evidence in examples:
            if example.answer_index is None:
                continue
            features.append(extractor.transform_options(example, evidence))
            labels.append(example.answer_index)
        return features, labels

    @staticmethod
    def _dataset_metrics(
        features: list[np.ndarray],
        labels: list[int],
        weights: np.ndarray,
    ) -> dict[str, float]:
        if not features:
            return {"loss": 0.0, "accuracy": 0.0}
        losses = []
        correct = 0
        for option_features, label in zip(features, labels, strict=True):
            scores = option_features @ weights
            probabilities = _softmax_numpy(scores)
            losses.append(-float(np.log(np.clip(probabilities[label], 1e-8, 1.0))))
            correct += int(int(np.argmax(scores)) == label)
        return {
            "loss": float(sum(losses) / len(losses)),
            "accuracy": float(correct / len(features)),
        }
