import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable

import cv2

from adaptive_evidence_vqa.data.base import load_jsonl


LETTERS = "ABCDE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a vision-language model over evidence selected by an existing "
            "grounded VideoQA method."
        )
    )
    parser.add_argument(
        "--candidate-path",
        required=True,
        help="Candidate-pool JSONL containing questions, answer options, and gold labels.",
    )
    parser.add_argument(
        "--selection-path",
        required=True,
        help="Prediction JSONL whose selected_evidence field should be re-answered by the VLM.",
    )
    parser.add_argument("--predictions-output", required=True, help="Path for VLM prediction JSONL.")
    parser.add_argument("--summary-output", help="Optional path for aggregate JSON summary.")
    parser.add_argument("--image-cache-dir", required=True, help="Directory for sampled segment frames.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of records to process.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append missing predictions and skip example_ids already present in --predictions-output.",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Hugging Face VLM model name.",
    )
    parser.add_argument("--device-map", default="auto", help="Device map passed to from_pretrained.")
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        help="Torch dtype passed to from_pretrained, e.g. auto, bfloat16, float16, float32.",
    )
    parser.add_argument("--attn-implementation", default=None, help="Optional attention implementation.")
    parser.add_argument("--max-new-tokens", type=int, default=8, help="Generation length cap.")
    parser.add_argument(
        "--segment-frames",
        type=int,
        default=1,
        help="Number of frames sampled from each selected segment evidence item.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=6,
        help="Maximum number of visual inputs per question after segment sampling.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=None,
        help="Optional Qwen processor min_pixels image budget.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=65536,
        help="Optional Qwen processor max_pixels image budget.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of examples to generate in one VLM forward pass. Falls back by splitting on CUDA OOM.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Env file containing HF_TOKEN. Falls back to .encv if this path is absent.",
    )
    parser.add_argument(
        "--method-label",
        default=None,
        help="Optional label for the evidence-selection method being re-scored.",
    )
    parser.add_argument("--progress-every", type=int, default=10, help="Progress print interval.")
    return parser.parse_args()


def load_env_file(path: str | Path) -> None:
    env_path = Path(path)
    if not env_path.exists() and env_path.name == ".env":
        fallback = env_path.with_name(".encv")
        if fallback.exists():
            env_path = fallback
    if not env_path.exists():
        return

    with env_path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", maxsplit=1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)


def sanitize_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def resolve_path(path: str | None, *, repo_root: Path) -> Path | None:
    if not path:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def evidence_start(item: dict) -> float:
    value = item.get("start_time")
    if value is None:
        value = item.get("end_time")
    return float(value) if value is not None else 0.0


def evidence_end(item: dict) -> float:
    value = item.get("end_time")
    if value is None:
        value = item.get("start_time")
    return float(value) if value is not None else evidence_start(item)


def selected_cost(selected_evidence: list[dict]) -> float:
    return float(sum(float(item.get("acquisition_cost", 1.0)) for item in selected_evidence))


def load_candidates(path: str | Path) -> dict[str, dict]:
    return {record["example_id"]: record for record in load_jsonl(path)}


def choose_records(records: list[dict], *, limit: int | None) -> list[dict]:
    if limit is None:
        return records
    return records[:limit]


def option_lines(options: list[str]) -> str:
    return "\n".join(f"{LETTERS[index]}. {option}" for index, option in enumerate(options))


def build_prompt(example: dict, selected_evidence: list[dict]) -> str:
    text_evidence = [
        item.get("text", "").strip()
        for item in selected_evidence
        if item.get("modality") == "subtitle" and item.get("text", "").strip()
    ]
    evidence_context = ""
    if text_evidence:
        evidence_context = "\nText evidence:\n" + "\n".join(f"- {text}" for text in text_evidence[:8])

    return (
        "You are answering a multiple-choice video question using only the provided visual evidence. "
        "The images are ordered by timestamp; segment evidence is represented by sampled frames. "
        "Choose the best answer. Reply with exactly one letter: A, B, C, D, or E.\n\n"
        f"Question: {example['question']}\n"
        f"Options:\n{option_lines(list(example['options']))}"
        f"{evidence_context}\n\n"
        "Answer:"
    )


def parse_answer_index(response: str, options: list[str] | None = None) -> int:
    text = response.strip()
    upper = text.upper()

    patterns = [
        r"^\s*\(?\s*([A-E])\s*\)?(?:[\s.:,-]|$)",
        r"\bANSWER\s*(?:IS|:)?\s*\(?\s*([A-E])\s*\)?\b",
        r"\bOPTION\s*\(?\s*([A-E])\s*\)?\b",
        r"\b([A-E])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            return LETTERS.index(match.group(1))

    if options:
        lowered = text.lower()
        for index, option in enumerate(options):
            option_text = option.strip().lower()
            if option_text and option_text in lowered:
                return index

    return -1


def frame_times_for_segment(start: float, end: float, count: int) -> list[float]:
    if count <= 0:
        return []
    start, end = sorted((start, end))
    if end <= start:
        return [start]
    return [start + (end - start) * ((index + 0.5) / count) for index in range(count)]


def read_frame(video_path: Path, time_seconds: float) -> object | None:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return None
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_seconds) * 1000.0)
        ok, frame = capture.read()
        if ok and frame is not None:
            return frame

        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0.0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(time_seconds * fps))))
            ok, frame = capture.read()
            if ok and frame is not None:
                return frame
        return None
    finally:
        capture.release()


def extract_frame(video_path: Path, output_path: Path, time_seconds: float) -> Path | None:
    if output_path.exists():
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = read_frame(video_path, time_seconds)
    if frame is None:
        return None
    if not cv2.imwrite(str(output_path), frame):
        return None
    return output_path


def sampled_segment_paths(
    item: dict,
    *,
    cache_dir: Path,
    repo_root: Path,
    count: int,
) -> list[Path]:
    metadata = item.get("metadata", {})
    video_path = resolve_path(
        str(metadata.get("video_source_path") or item.get("source_path") or ""),
        repo_root=repo_root,
    )
    if video_path is None or not video_path.exists():
        return []

    example_id = sanitize_id(str(item.get("evidence_id", "example")).split(":segment:")[0])
    evidence_id = sanitize_id(str(item.get("evidence_id", "segment")))
    paths = []
    for index, time_seconds in enumerate(
        frame_times_for_segment(evidence_start(item), evidence_end(item), count)
    ):
        output_path = (
            cache_dir
            / example_id
            / f"{evidence_id}_sample{index:02d}_{time_seconds:.3f}.jpg"
        )
        sampled = extract_frame(video_path, output_path, time_seconds)
        if sampled is not None:
            paths.append(sampled)
    return paths


def visual_input_paths(
    selected_evidence: list[dict],
    *,
    cache_dir: Path,
    repo_root: Path,
    segment_frames: int,
    max_images: int,
) -> tuple[list[Path], list[str]]:
    image_paths: list[Path] = []
    warnings: list[str] = []
    ordered_evidence = sorted(
        selected_evidence,
        key=lambda item: (evidence_start(item), evidence_end(item), str(item.get("evidence_id", ""))),
    )

    for item in ordered_evidence:
        modality = item.get("modality")
        if modality == "frame":
            path = resolve_path(item.get("source_path"), repo_root=repo_root)
            if path is not None and path.exists():
                image_paths.append(path)
            else:
                warnings.append(f"missing_frame:{item.get('evidence_id')}")
        elif modality == "segment":
            sampled_paths = sampled_segment_paths(
                item,
                cache_dir=cache_dir,
                repo_root=repo_root,
                count=segment_frames,
            )
            if sampled_paths:
                image_paths.extend(sampled_paths)
            else:
                warnings.append(f"missing_segment_frames:{item.get('evidence_id')}")

    if max_images > 0 and len(image_paths) > max_images:
        step = len(image_paths) / max_images
        selected_indices = [min(len(image_paths) - 1, int(index * step)) for index in range(max_images)]
        image_paths = [image_paths[index] for index in selected_indices]
        warnings.append(f"truncated_images:{len(ordered_evidence)}_evidence_items")

    return image_paths, warnings


def load_model_and_processor(args: argparse.Namespace):
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    model_kwargs = {
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
    }
    if token:
        model_kwargs["token"] = token
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    processor_kwargs = {}
    if token:
        processor_kwargs["token"] = token
    if args.min_pixels is not None:
        processor_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        processor_kwargs["max_pixels"] = args.max_pixels

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_name, **model_kwargs)
    processor = AutoProcessor.from_pretrained(args.model_name, **processor_kwargs)
    processor.tokenizer.padding_side = "left"
    model.eval()
    return model, processor, process_vision_info, torch


def generate_answer(
    *,
    model,
    processor,
    process_vision_info,
    torch_module,
    prompt: str,
    image_paths: list[Path],
    max_new_tokens: int,
) -> str:
    content = [{"type": "image", "image": str(path)} for path in image_paths]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = inputs.to(device)

    with torch_module.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def build_messages(prompt: str, image_paths: list[Path]) -> list[dict]:
    content = [{"type": "image", "image": str(path)} for path in image_paths]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def generate_answers_batch(
    *,
    model,
    processor,
    process_vision_info,
    torch_module,
    prompts: list[str],
    image_paths_by_prompt: list[list[Path]],
    max_new_tokens: int,
) -> list[str]:
    if not prompts:
        return []
    if len(prompts) == 1:
        return [
            generate_answer(
                model=model,
                processor=processor,
                process_vision_info=process_vision_info,
                torch_module=torch_module,
                prompt=prompts[0],
                image_paths=image_paths_by_prompt[0],
                max_new_tokens=max_new_tokens,
            )
        ]

    conversations = [
        build_messages(prompt, image_paths)
        for prompt, image_paths in zip(prompts, image_paths_by_prompt, strict=True)
    ]
    try:
        texts = [
            processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in conversations
        ]
        image_inputs, video_inputs = process_vision_info(conversations)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        device = next(model.parameters()).device
        inputs = inputs.to(device)

        with torch_module.inference_mode():
            generated_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        generated_ids_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids, strict=True)
        ]
        return [
            response.strip()
            for response in processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        ]
    except torch_module.OutOfMemoryError:
        torch_module.cuda.empty_cache()
        midpoint = len(prompts) // 2
        left = generate_answers_batch(
            model=model,
            processor=processor,
            process_vision_info=process_vision_info,
            torch_module=torch_module,
            prompts=prompts[:midpoint],
            image_paths_by_prompt=image_paths_by_prompt[:midpoint],
            max_new_tokens=max_new_tokens,
        )
        right = generate_answers_batch(
            model=model,
            processor=processor,
            process_vision_info=process_vision_info,
            torch_module=torch_module,
            prompts=prompts[midpoint:],
            image_paths_by_prompt=image_paths_by_prompt[midpoint:],
            max_new_tokens=max_new_tokens,
        )
        return left + right


def iter_existing_predictions(path: Path) -> Iterable[dict]:
    if not path.exists():
        return []
    return load_jsonl(path)


def summarize(records: list[dict], args: argparse.Namespace) -> dict:
    correct_rows = [record for record in records if "correct" in record]
    failed = [record for record in records if int(record.get("predicted_index", -1)) < 0]
    return {
        "model_name": args.model_name,
        "selection_path": args.selection_path,
        "num_examples": len(records),
        "num_with_gold": len(correct_rows),
        "accuracy": (
            sum(float(record["correct"]) for record in correct_rows) / len(correct_rows)
            if correct_rows
            else 0.0
        ),
        "parse_failure_rate": len(failed) / len(records) if records else 0.0,
        "selected_evidence_cost": (
            sum(float(record.get("selected_evidence_cost", 0.0)) for record in records) / len(records)
            if records
            else 0.0
        ),
        "selected_evidence_count": (
            sum(float(record.get("selected_evidence_count", 0.0)) for record in records) / len(records)
            if records
            else 0.0
        ),
        "vlm_image_count": (
            sum(float(record.get("vlm_image_count", 0.0)) for record in records) / len(records)
            if records
            else 0.0
        ),
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "segment_frames": args.segment_frames,
            "max_images": args.max_images,
            "max_pixels": args.max_pixels,
            "batch_size": args.batch_size,
        },
    }


def atomic_write_json(payload: dict, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(output_path.parent),
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        tmp_name = handle.name
    Path(tmp_name).replace(output_path)


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    load_env_file(repo_root / args.env_file)

    candidates = load_candidates(args.candidate_path)
    selection_records = choose_records(load_jsonl(args.selection_path), limit=args.limit)
    predictions_path = Path(args.predictions_output)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.image_cache_dir)

    existing_records = list(iter_existing_predictions(predictions_path)) if args.resume else []
    seen_example_ids = {record["example_id"] for record in existing_records}
    output_mode = "a" if args.resume and predictions_path.exists() else "w"

    selection_label = args.method_label
    if selection_label is None and selection_records:
        selection_label = selection_records[0].get("method_name")
    if selection_label is None:
        selection_label = Path(args.selection_path).stem

    pending_records = [
        record
        for record in selection_records
        if record["example_id"] not in seen_example_ids
    ]

    print(
        f"Loading {args.model_name}; pending={len(pending_records)} "
        f"existing={len(existing_records)}",
        flush=True,
    )
    model, processor, process_vision_info, torch_module = load_model_and_processor(args)

    written = 0
    batch_size = max(1, int(args.batch_size))
    with predictions_path.open(output_mode, encoding="utf-8") as handle:
        for batch_start in range(0, len(pending_records), batch_size):
            batch = pending_records[batch_start : batch_start + batch_size]
            prepared = []
            prompts = []
            image_paths_by_prompt = []
            for selection in batch:
                example = candidates[selection["example_id"]]
                selected_evidence = list(selection.get("selected_evidence", []))
                image_paths, warnings = visual_input_paths(
                    selected_evidence,
                    cache_dir=cache_dir,
                    repo_root=repo_root,
                    segment_frames=args.segment_frames,
                    max_images=args.max_images,
                )
                prompt = build_prompt(example, selected_evidence)
                prepared.append(
                    {
                        "selection": selection,
                        "example": example,
                        "selected_evidence": selected_evidence,
                        "image_paths": image_paths,
                        "warnings": warnings,
                    }
                )
                if image_paths:
                    prompts.append(prompt)
                    image_paths_by_prompt.append(image_paths)

            generated = []
            if prompts:
                generated = generate_answers_batch(
                    model=model,
                    processor=processor,
                    process_vision_info=process_vision_info,
                    torch_module=torch_module,
                    prompts=prompts,
                    image_paths_by_prompt=image_paths_by_prompt,
                    max_new_tokens=args.max_new_tokens,
                )

            generated_index = 0
            for prepared_item in prepared:
                selection = prepared_item["selection"]
                example = prepared_item["example"]
                selected_evidence = prepared_item["selected_evidence"]
                image_paths = prepared_item["image_paths"]
                warnings = prepared_item["warnings"]
                if image_paths:
                    raw_response = generated[generated_index]
                    generated_index += 1
                else:
                    raw_response = ""
                predicted_index = parse_answer_index(raw_response, list(example["options"]))
                gold_index = selection.get("gold_index", example.get("answer_index"))

                result = {
                    "method_name": f"vlm_{selection_label}",
                    "method_type": "vlm_rescore",
                    "selection_method_name": selection_label,
                    "vlm_model_name": args.model_name,
                    "example_id": selection["example_id"],
                    "video_id": selection.get("video_id", example.get("video_id")),
                    "predicted_index": predicted_index,
                    "prediction_confidence": 0.0,
                    "raw_response": raw_response,
                    "selected_evidence": selected_evidence,
                    "selected_evidence_count": int(
                        selection.get("selected_evidence_count", len(selected_evidence))
                    ),
                    "selected_evidence_cost": float(
                        selection.get("selected_evidence_cost", selected_cost(selected_evidence))
                    ),
                    "vlm_image_count": len(image_paths),
                    "vlm_image_warnings": warnings,
                    "prompt_version": "vlm_mcq_letter_v1",
                }
                if "trace" in selection:
                    result["trace"] = selection["trace"]
                if "selected_temporal_iou" in selection:
                    result["selected_temporal_iou"] = selection["selected_temporal_iou"]
                if gold_index is not None:
                    result["gold_index"] = int(gold_index)
                    result["correct"] = 1.0 if predicted_index == int(gold_index) else 0.0

                handle.write(json.dumps(result, ensure_ascii=True) + "\n")
                written += 1
            handle.flush()
            processed = min(batch_start + len(batch), len(pending_records))
            if args.progress_every > 0 and processed % args.progress_every == 0:
                print(
                    f"processed={processed}/{len(pending_records)} "
                    f"written={written} last_example={batch[-1]['example_id']}",
                    flush=True,
                )

    all_records = list(iter_existing_predictions(predictions_path))
    summary = summarize(all_records, args)
    if args.summary_output:
        atomic_write_json(summary, args.summary_output)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
