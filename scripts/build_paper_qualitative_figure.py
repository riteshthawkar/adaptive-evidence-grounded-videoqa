import argparse
import json
import math
import subprocess
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


LETTERS = "ABCDE"
METHOD_LABELS = {
    "mlp_plain": "MLP top-2",
    "mlp_nms": "MLP+NMS",
    "oracle_top2": "Oracle top-2",
    "fixed_f3_s3": "Fixed 3+3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paper-ready qualitative visual panels from selected cases."
    )
    parser.add_argument(
        "--cases-json",
        default=(
            "runs/nextgqa_full_seed13/outputs/followup_strengthening/analysis/"
            "paper_qualitative/paper_qualitative_cases.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "runs/nextgqa_full_seed13/outputs/followup_strengthening/analysis/"
            "paper_qualitative/figures"
        ),
    )
    parser.add_argument("--max-cases", type=int, default=4)
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for name in names:
        path = Path(name)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def wrap_text(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width, break_long_words=False) or [""]


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
    return tuple(sorted((float(start), float(end))))


def span_label(interval: tuple[float, float] | None) -> str:
    if interval is None:
        return "?"
    if abs(interval[0] - interval[1]) < 1e-6:
        return f"{interval[0]:.1f}s"
    return f"{interval[0]:.1f}-{interval[1]:.1f}s"


def evidence_name(item: dict) -> str:
    modality = item.get("modality", "?")
    evidence_id = str(item.get("evidence_id", ""))
    index = evidence_id.split(":")[-1] if ":" in evidence_id else "?"
    prefix = "F" if modality == "frame" else "S" if modality == "segment" else modality[:1].upper()
    return f"{prefix}{index}@{span_label(evidence_interval(item))}"


def sanitize_stem(text: str) -> str:
    keep = []
    for char in text.lower():
        if char.isalnum():
            keep.append(char)
        elif keep and keep[-1] != "_":
            keep.append("_")
    return "".join(keep).strip("_")[:72]


def extract_frame(item: dict, cache_dir: Path, tag: str) -> Path | None:
    source = Path(str(item.get("source_path", "")))
    if source.suffix.lower() in {".jpg", ".jpeg", ".png"} and source.exists():
        return source

    video_path = source
    metadata_path = item.get("metadata", {}).get("video_source_path")
    if (not video_path.exists() or video_path.suffix.lower() != ".mp4") and metadata_path:
        video_path = Path(str(metadata_path))
    if not video_path.exists():
        return None

    interval = evidence_interval(item)
    if interval is None:
        return None
    timestamp = max(0.0, sum(interval) / 2.0)
    out_path = cache_dir / f"{tag}_{timestamp:.3f}.jpg"
    if out_path.exists():
        return out_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(out_path),
    ]
    subprocess.run(command, check=True)
    return out_path if out_path.exists() else None


def fit_image(path: Path | None, size: tuple[int, int]) -> Image.Image:
    width, height = size
    if path is None or not path.exists():
        image = Image.new("RGB", size, "#f1f5f9")
        draw = ImageDraw.Draw(image)
        font = load_font(24, bold=True)
        draw.text((width // 2, height // 2), "missing", anchor="mm", fill="#64748b", font=font)
        return image

    image = Image.open(path).convert("RGB")
    scale = max(width / image.width, height / image.height)
    resized = image.resize((math.ceil(image.width * scale), math.ceil(image.height * scale)))
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def draw_text_lines(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    lines: list[str],
    font: ImageFont.FreeTypeFont,
    fill: str,
    line_gap: int = 8,
) -> int:
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y += (bbox[3] - bbox[1]) + line_gap
    return y


def method_keys_for_case(case: dict) -> list[str]:
    if case["title"].startswith("Diversity"):
        return ["mlp_plain", "mlp_nms", "oracle_top2"]
    return ["mlp_nms", "oracle_top2", "fixed_f3_s3"]


def draw_case(case: dict, output_dir: Path, index: int) -> Image.Image:
    canvas_width = 1500
    margin = 32
    header_height = 158
    method_height = 220
    method_gap = 18
    thumb_size = (260, 146)
    label_width = 292
    canvas_height = (
        margin
        + header_height
        + len(method_keys_for_case(case)) * method_height
        + (len(method_keys_for_case(case)) - 1) * method_gap
        + margin
    )
    image = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(image)

    title_font = load_font(32, bold=True)
    body_font = load_font(22)
    small_font = load_font(18)
    small_bold = load_font(18, bold=True)
    method_font = load_font(24, bold=True)

    gold = f"{LETTERS[case['gold']]}. {case['options'][case['gold']]}"
    draw.text((margin, margin), case["title"], font=title_font, fill="#0f172a")
    question = f"Q: {case['question']}  |  Gold: {gold}"
    draw_text_lines(
        draw,
        (margin, margin + 48),
        wrap_text(question, 106),
        body_font,
        "#334155",
        line_gap=5,
    )
    draw_text_lines(
        draw,
        (margin, margin + 92),
        wrap_text(f"Takeaway: {case['takeaway']}", 126),
        small_font,
        "#475569",
        line_gap=4,
    )

    y = margin + header_height
    cache_dir = output_dir / "extracted_frames"
    for method_key in method_keys_for_case(case):
        metrics = case["metrics"][method_key]
        correct = bool(metrics["correct"])
        border = "#15803d" if correct else "#b91c1c"
        fill = "#f8fafc"
        draw.rounded_rectangle(
            (margin, y, canvas_width - margin, y + method_height),
            radius=8,
            outline=border,
            width=4,
            fill=fill,
        )
        pred = LETTERS[metrics["pred"]] if 0 <= metrics["pred"] < len(LETTERS) else "?"
        status = "correct" if correct else "wrong"
        method_text = [
            METHOD_LABELS[method_key],
            f"Pred {pred} ({status})",
            f"IoP {metrics['iop']:.2f}, IoU {metrics['iou']:.2f}",
        ]
        draw_text_lines(draw, (margin + 18, y + 32), method_text, method_font, "#0f172a", 11)

        x = margin + label_width
        last_thumb = None
        for item_index, item in enumerate(metrics["selected_evidence"][:4]):
            tag = f"case{index}_{method_key}_{item_index}_{sanitize_stem(case['example_id'])}"
            frame_path = extract_frame(item, cache_dir, tag)
            thumb = fit_image(frame_path, thumb_size)
            image.paste(thumb, (x, y + 28))
            last_thumb = (x, y + 28, x + thumb_size[0], y + 28 + thumb_size[1])
            draw.rectangle(last_thumb, outline="#0f172a", width=2)
            draw.rectangle(
                (x, y + 28 + thumb_size[1] - 33, x + thumb_size[0], y + 28 + thumb_size[1]),
                fill=(15, 23, 42),
            )
            draw.text((x + 9, y + 28 + thumb_size[1] - 25), evidence_name(item), font=small_bold, fill="white")
            x += thumb_size[0] + 18

        if len(metrics["selected_evidence"]) > 4:
            extra = len(metrics["selected_evidence"]) - 4
            if last_thumb is not None:
                _, ly0, lx1, _ = last_thumb
                badge = (lx1 - 70, ly0 + 8, lx1 - 8, ly0 + 40)
                draw.rounded_rectangle(badge, radius=6, fill="#0f172a")
                draw.text((badge[0] + 10, badge[1] + 6), f"+{extra}", font=small_bold, fill="white")

        y += method_height + method_gap

    return image


def draw_compact_method_row(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    case: dict,
    method_key: str,
    box: tuple[int, int, int, int],
    cache_dir: Path,
    tag_prefix: str,
) -> None:
    x0, y0, x1, y1 = box
    metrics = case["metrics"][method_key]
    correct = bool(metrics["correct"])
    border = "#15803d" if correct else "#b91c1c"
    draw.rounded_rectangle(box, radius=8, outline=border, width=3, fill="#f8fafc")

    label_font = load_font(22, bold=True)
    metric_font = load_font(19, bold=True)
    thumb_font = load_font(16, bold=True)
    status = "correct" if correct else "wrong"
    pred = LETTERS[metrics["pred"]] if 0 <= metrics["pred"] < len(LETTERS) else "?"
    draw.text((x0 + 14, y0 + 18), METHOD_LABELS[method_key], font=label_font, fill="#0f172a")
    draw.text((x0 + 14, y0 + 50), f"Pred {pred} ({status})", font=metric_font, fill="#0f172a")
    draw.text((x0 + 14, y0 + 78), f"IoP {metrics['iop']:.2f}", font=metric_font, fill="#0f172a")

    thumb_size = (205, 115)
    thumb_x = x0 + 198
    thumb_y = y0 + 14
    for item_index, item in enumerate(metrics["selected_evidence"][:2]):
        tag = f"{tag_prefix}_{method_key}_{item_index}_{sanitize_stem(case['example_id'])}"
        frame_path = extract_frame(item, cache_dir, tag)
        thumb = fit_image(frame_path, thumb_size)
        canvas.paste(thumb, (thumb_x, thumb_y))
        draw.rectangle(
            (thumb_x, thumb_y, thumb_x + thumb_size[0], thumb_y + thumb_size[1]),
            outline="#0f172a",
            width=2,
        )
        draw.rectangle(
            (thumb_x, thumb_y + thumb_size[1] - 28, thumb_x + thumb_size[0], thumb_y + thumb_size[1]),
            fill="#0f172a",
        )
        draw.text(
            (thumb_x + 8, thumb_y + thumb_size[1] - 22),
            evidence_name(item),
            font=thumb_font,
            fill="white",
        )
        thumb_x += thumb_size[0] + 12


def draw_compact_case_card(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    case: dict,
    box: tuple[int, int, int, int],
    method_keys: list[str],
    cache_dir: Path,
    tag_prefix: str,
) -> None:
    x0, y0, x1, y1 = box
    title_font = load_font(30, bold=True)
    body_font = load_font(20)
    note_font = load_font(17)
    draw.rounded_rectangle(box, radius=10, outline="#cbd5e1", width=2, fill="white")
    draw.text((x0 + 20, y0 + 18), case["title"], font=title_font, fill="#0f172a")

    gold = f"{LETTERS[case['gold']]}. {case['options'][case['gold']]}"
    question = f"Q: {case['question']} | Gold: {gold}"
    draw_text_lines(
        draw,
        (x0 + 20, y0 + 58),
        wrap_text(question, 64),
        body_font,
        "#334155",
        line_gap=3,
    )

    row_y = y0 + 126
    row_height = 146
    row_gap = 18
    for method_key in method_keys:
        draw_compact_method_row(
            canvas,
            draw,
            case,
            method_key,
            (x0 + 20, row_y, x1 - 20, row_y + row_height),
            cache_dir,
            tag_prefix,
        )
        row_y += row_height + row_gap

    draw_text_lines(
        draw,
        (x0 + 20, row_y + 2),
        wrap_text(case["takeaway"], 82),
        note_font,
        "#475569",
        line_gap=3,
    )


def draw_compact_paper_figure(cases: list[dict], output_dir: Path) -> Image.Image:
    selected = {case["title"]: case for case in cases}
    diversity = selected.get("Diversity Removes Redundant Evidence", cases[0])
    router_gap = selected.get("Oracle Exposes the Router Gap", cases[min(1, len(cases) - 1)])

    canvas_width = 1450
    canvas_height = 570
    margin = 20
    row_gap = 18
    row_header = 56
    cell_height = 198
    col_gap = 14
    cell_width = (canvas_width - 2 * margin - 2 * col_gap) // 3
    image = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(image)
    cache_dir = output_dir / "extracted_frames"

    title_font = load_font(22, bold=True)
    body_font = load_font(15)

    row_specs = [
        (
            "(A) Redundancy reduction",
            diversity,
            ["mlp_plain", "mlp_nms", "oracle_top2"],
        ),
        (
            "(B) Missed routed evidence",
            router_gap,
            ["mlp_nms", "oracle_top2", "fixed_f3_s3"],
        ),
    ]

    def draw_cell(case: dict, method_key: str, box: tuple[int, int, int, int], tag_prefix: str) -> None:
        x0, y0, x1, y1 = box
        metrics = case["metrics"][method_key]
        correct = bool(metrics["correct"])
        border = "#15803d" if correct else "#b91c1c"
        draw.rounded_rectangle(box, radius=8, outline=border, width=3, fill="#f8fafc")

        method_font = load_font(18, bold=True)
        metric_font = load_font(15, bold=True)
        thumb_font = load_font(13, bold=True)
        pred = LETTERS[metrics["pred"]] if 0 <= metrics["pred"] < len(LETTERS) else "?"
        status = "correct" if correct else "wrong"
        draw.text((x0 + 10, y0 + 10), METHOD_LABELS[method_key], font=method_font, fill="#0f172a")
        draw.text(
            (x0 + 10, y0 + 36),
            f"Pred {pred} ({status})   IoP {metrics['iop']:.2f}",
            font=metric_font,
            fill="#0f172a",
        )

        thumb_size = (198, 111)
        thumb_gap = 10
        thumb_x = x0 + 10
        thumb_y = y0 + 64
        last_thumb = None
        for item_index, item in enumerate(metrics["selected_evidence"][:2]):
            tag = f"{tag_prefix}_{method_key}_{item_index}_{sanitize_stem(case['example_id'])}"
            frame_path = extract_frame(item, cache_dir, tag)
            thumb = fit_image(frame_path, thumb_size)
            image.paste(thumb, (thumb_x, thumb_y))
            last_thumb = (thumb_x, thumb_y, thumb_x + thumb_size[0], thumb_y + thumb_size[1])
            draw.rectangle(last_thumb, outline="#0f172a", width=2)
            draw.rectangle(
                (thumb_x, thumb_y + thumb_size[1] - 24, thumb_x + thumb_size[0], thumb_y + thumb_size[1]),
                fill="#0f172a",
            )
            draw.text(
                (thumb_x + 7, thumb_y + thumb_size[1] - 20),
                evidence_name(item),
                font=thumb_font,
                fill="white",
            )
            thumb_x += thumb_size[0] + thumb_gap

        if len(metrics["selected_evidence"]) > 2 and last_thumb is not None:
            _, ly0, lx1, _ = last_thumb
            extra = len(metrics["selected_evidence"]) - 2
            badge = (lx1 - 46, ly0 + 7, lx1 - 7, ly0 + 31)
            draw.rounded_rectangle(badge, radius=5, fill="#0f172a")
            draw.text((badge[0] + 8, badge[1] + 4), f"+{extra}", font=thumb_font, fill="white")

    y = margin
    for row_index, (row_title, case, method_keys) in enumerate(row_specs, start=1):
        row_height = row_header + cell_height
        row_fill = "#f8fafc" if row_index == 1 else "#f9fafb"
        row_outline = "#cbd5e1"
        draw.rounded_rectangle(
            (margin - 8, y - 8, canvas_width - margin + 8, y + row_height + 8),
            radius=14,
            fill=row_fill,
            outline=row_outline,
            width=2,
        )
        gold = f"{LETTERS[case['gold']]}. {case['options'][case['gold']]}"
        question = f"Q: {case['question']} | Gold: {gold}"
        draw.text((margin, y), row_title, font=title_font, fill="#0f172a")
        draw_text_lines(draw, (margin, y + 28), wrap_text(question, 130), body_font, "#334155", line_gap=2)

        cell_y = y + row_header
        for col_index, method_key in enumerate(method_keys):
            cell_x = margin + col_index * (cell_width + col_gap)
            draw_cell(
                case,
                method_key,
                (cell_x, cell_y, cell_x + cell_width, cell_y + cell_height),
                f"compact_r{row_index}_c{col_index}",
            )
        y = cell_y + cell_height + row_gap
    return image


def draw_appendix_qualitative_figure(cases: list[dict], output_dir: Path) -> Image.Image:
    canvas_width = 1450
    canvas_height = 1860
    margin = 22
    row_gap = 18
    row_header = 78
    cell_height = 270
    col_gap = 14
    cell_width = (canvas_width - 2 * margin - 2 * col_gap) // 3
    image = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(image)
    cache_dir = output_dir / "extracted_frames"

    title_font = load_font(22, bold=True)
    body_font = load_font(15)
    method_font = load_font(18, bold=True)
    metric_font = load_font(15, bold=True)
    thumb_font = load_font(13, bold=True)

    row_titles = [
        "(C) High-coverage context",
        "(D) Compact oracle support",
        "(E) Oracle recovers support",
        "(F) Grounded learned success",
        "(G) Grounded answer error",
    ]

    def draw_cell(case: dict, method_key: str, box: tuple[int, int, int, int], tag_prefix: str) -> None:
        x0, y0, x1, y1 = box
        metrics = case["metrics"][method_key]
        correct = bool(metrics["correct"])
        border = "#15803d" if correct else "#b91c1c"
        draw.rounded_rectangle(box, radius=8, outline=border, width=3, fill="#f8fafc")

        pred = LETTERS[metrics["pred"]] if 0 <= metrics["pred"] < len(LETTERS) else "?"
        status = "correct" if correct else "wrong"
        draw.text((x0 + 10, y0 + 10), METHOD_LABELS[method_key], font=method_font, fill="#0f172a")
        draw.text(
            (x0 + 10, y0 + 36),
            f"Pred {pred} ({status})   IoP {metrics['iop']:.2f}",
            font=metric_font,
            fill="#0f172a",
        )

        thumb_size = (198, 111)
        thumb_gap = 10
        thumb_x = x0 + 10
        thumb_y = y0 + 64
        last_thumb = None
        for item_index, item in enumerate(metrics["selected_evidence"][:2]):
            tag = f"{tag_prefix}_{method_key}_{item_index}_{sanitize_stem(case['example_id'])}"
            frame_path = extract_frame(item, cache_dir, tag)
            thumb = fit_image(frame_path, thumb_size)
            image.paste(thumb, (thumb_x, thumb_y))
            last_thumb = (thumb_x, thumb_y, thumb_x + thumb_size[0], thumb_y + thumb_size[1])
            draw.rectangle(last_thumb, outline="#0f172a", width=2)
            draw.rectangle(
                (thumb_x, thumb_y + thumb_size[1] - 24, thumb_x + thumb_size[0], thumb_y + thumb_size[1]),
                fill="#0f172a",
            )
            draw.text(
                (thumb_x + 7, thumb_y + thumb_size[1] - 20),
                evidence_name(item),
                font=thumb_font,
                fill="white",
            )
            thumb_x += thumb_size[0] + thumb_gap

        if len(metrics["selected_evidence"]) > 2 and last_thumb is not None:
            _, ly0, lx1, _ = last_thumb
            extra = len(metrics["selected_evidence"]) - 2
            badge = (lx1 - 46, ly0 + 7, lx1 - 7, ly0 + 31)
            draw.rounded_rectangle(badge, radius=5, fill="#0f172a")
            draw.text((badge[0] + 8, badge[1] + 4), f"+{extra}", font=thumb_font, fill="white")

        takeaway_lines = wrap_text(case["takeaway"], 54)[:3]
        draw_text_lines(
            draw,
            (x0 + 10, y0 + 190),
            takeaway_lines,
            load_font(14),
            "#475569",
            line_gap=2,
        )

    y = margin
    for row_index, case in enumerate(cases[:5], start=1):
        row_height = row_header + cell_height
        row_fill = "#f8fafc" if row_index % 2 else "#f9fafb"
        draw.rounded_rectangle(
            (margin - 8, y - 8, canvas_width - margin + 8, y + row_height + 8),
            radius=14,
            fill=row_fill,
            outline="#cbd5e1",
            width=2,
        )

        row_title = row_titles[row_index - 1] if row_index <= len(row_titles) else f"({row_index}) Example"
        gold = f"{LETTERS[case['gold']]}. {case['options'][case['gold']]}"
        question = f"Q: {case['question']} | Gold: {gold}"
        draw.text((margin, y), row_title, font=title_font, fill="#0f172a")
        draw_text_lines(draw, (margin, y + 28), wrap_text(question, 128), body_font, "#334155", line_gap=2)

        cell_y = y + row_header
        for col_index, method_key in enumerate(["mlp_nms", "oracle_top2", "fixed_f3_s3"]):
            cell_x = margin + col_index * (cell_width + col_gap)
            draw_cell(
                case,
                method_key,
                (cell_x, cell_y, cell_x + cell_width, cell_y + cell_height),
                f"appendix_r{row_index}_c{col_index}",
            )
        y = cell_y + cell_height + row_gap
    return image


def save_panel(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    image.save(path.with_suffix(".pdf"), "PDF", resolution=300.0)


def main() -> None:
    args = parse_args()
    cases_path = Path(args.cases_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = payload["cases"][: args.max_cases]
    appendix_cases = payload.get("appendix_cases", [])
    panels = []
    manifest = []
    for index, case in enumerate(cases, start=1):
        panel = draw_case(case, output_dir, index)
        stem = f"case_{index}_{sanitize_stem(case['title'])}"
        panel_path = output_dir / f"{stem}.png"
        save_panel(panel, panel_path)
        panels.append(panel)
        manifest.append(
            {
                "title": case["title"],
                "example_id": case["example_id"],
                "image": str(panel_path),
                "pdf": str(panel_path.with_suffix(".pdf")),
            }
        )

    if panels:
        spacer = 26
        combined_width = max(panel.width for panel in panels)
        combined_height = sum(panel.height for panel in panels) + spacer * (len(panels) - 1)
        combined = Image.new("RGB", (combined_width, combined_height), "white")
        y = 0
        for panel in panels:
            combined.paste(panel, (0, y))
            y += panel.height + spacer
        save_panel(combined, output_dir / "paper_qualitative_long_figure.png")

    if cases:
        compact = draw_compact_paper_figure(cases, output_dir)
        save_panel(compact, output_dir / "paper_qualitative_figure.png")

    if appendix_cases:
        appendix = draw_appendix_qualitative_figure(appendix_cases, output_dir)
        save_panel(appendix, output_dir / "appendix_qualitative_cases.png")

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if manifest:
        snippet = r"""\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{paper_qualitative/figures/paper_qualitative_figure.pdf}
\caption{Qualitative examples from NExT-GQA test in a two-row, three-column layout. Row A shows that temporal NMS prevents the learned router from selecting redundant nearby segments and recovers support overlap without changing the correct answer. Row B shows that the oracle selector finds compact support already present in the retrieved pool, exposing the remaining router gap.}
\label{fig:qualitative-cases}
\end{figure}
"""
        (output_dir / "paper_qualitative_figure_snippet.tex").write_text(snippet, encoding="utf-8")
    print(f"Wrote {len(panels)} qualitative panels to {output_dir}")


if __name__ == "__main__":
    main()
