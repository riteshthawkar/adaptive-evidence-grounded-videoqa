#!/usr/bin/env python3
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from adaptive_evidence_vqa.data.base import load_jsonl, save_jsonl
from adaptive_evidence_vqa.data.visual import (
    build_video_index,
    materialize_visual_evidence,
    resolve_video_path,
)


_VIDEO_ROOT: str | None = None
_FRAMES_DIR: str | None = None
_SEGMENTS_DIR: str | None = None
_EXTRACT_SEGMENTS = False
_FFMPEG_BIN = "ffmpeg"
_OVERWRITE = False
_VIDEO_INDEX: dict[str, str] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize visual evidence with multiple worker processes."
    )
    parser.add_argument("--input-path", required=True, help="Path to candidate-pool JSONL.")
    parser.add_argument("--video-root", required=True, help="Root directory containing source videos.")
    parser.add_argument("--output-path", required=True, help="Path to enriched output JSONL.")
    parser.add_argument("--frames-dir", required=True, help="Directory for extracted frame images.")
    parser.add_argument("--segments-dir", help="Directory for extracted segment clips.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of records.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel materialization workers.")
    parser.add_argument("--chunksize", type=int, default=1, help="Process-pool chunksize.")
    parser.add_argument("--progress-interval", type=int, default=250, help="Records between progress logs.")
    parser.add_argument(
        "--extract-segments",
        action="store_true",
        help="If set, extract individual segment clips; otherwise keep segment source paths pointing to the full video.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing visual artifacts.")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg executable name or path.")
    return parser.parse_args()


def init_worker(
    video_root: str,
    frames_dir: str,
    segments_dir: str | None,
    extract_segments: bool,
    ffmpeg_bin: str,
    overwrite: bool,
) -> None:
    global _VIDEO_ROOT
    global _FRAMES_DIR
    global _SEGMENTS_DIR
    global _EXTRACT_SEGMENTS
    global _FFMPEG_BIN
    global _OVERWRITE
    global _VIDEO_INDEX

    _VIDEO_ROOT = video_root
    _FRAMES_DIR = frames_dir
    _SEGMENTS_DIR = segments_dir
    _EXTRACT_SEGMENTS = extract_segments
    _FFMPEG_BIN = ffmpeg_bin
    _OVERWRITE = overwrite
    _VIDEO_INDEX = build_video_index(video_root)


def materialize_record(record: dict) -> dict:
    if _VIDEO_ROOT is None or _FRAMES_DIR is None:
        raise RuntimeError("Worker was not initialized.")

    video_path = resolve_video_path(
        video_id=record["video_id"],
        video_root=_VIDEO_ROOT,
        video_index=_VIDEO_INDEX,
    )
    if video_path is None:
        metadata = dict(record.get("metadata", {}))
        metadata["visual_materialization_status"] = "video_missing"
        record["metadata"] = metadata
        return record

    return materialize_visual_evidence(
        record=record,
        video_path=video_path,
        frames_root=_FRAMES_DIR,
        segments_root=_SEGMENTS_DIR,
        extract_segments=_EXTRACT_SEGMENTS,
        ffmpeg_bin=_FFMPEG_BIN,
        overwrite=_OVERWRITE,
    )


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input_path)
    if args.limit is not None:
        records = records[: args.limit]

    workers = max(1, args.workers)
    chunksize = max(1, args.chunksize)
    Path(args.frames_dir).mkdir(parents=True, exist_ok=True)
    if args.segments_dir is not None:
        Path(args.segments_dir).mkdir(parents=True, exist_ok=True)

    enriched_records = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(
            args.video_root,
            args.frames_dir,
            args.segments_dir,
            args.extract_segments,
            args.ffmpeg_bin,
            args.overwrite,
        ),
    ) as executor:
        iterator = executor.map(materialize_record, records, chunksize=chunksize)
        for index, enriched in enumerate(iterator, start=1):
            enriched_records.append(enriched)
            if args.progress_interval > 0 and index % args.progress_interval == 0:
                print(
                    f"Materialized {index}/{len(records)} records with {workers} workers",
                    file=sys.stderr,
                    flush=True,
                )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(enriched_records, output_path)

    missing_videos = sorted(
        {
            record["video_id"]
            for record in enriched_records
            if record.get("metadata", {}).get("visual_materialization_status") == "video_missing"
        }
    )
    print(f"Wrote {len(enriched_records)} enriched records to {output_path}")
    if missing_videos:
        print(f"Missing videos for {len(missing_videos)} records")
        for video_id in missing_videos[:10]:
            print(f"  missing: {video_id}")


if __name__ == "__main__":
    main()
