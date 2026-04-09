#!/usr/bin/env python3
"""Run OMR on a PDF and export a single MusicXML output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.cli import run_assemble, run_export
from src.manual_page_cropper import crop_pages_with_editor
from src.eval.evaluate_stage_b_checkpoint import _run_stage_b_inference_with_progress
from src.model_assets import (
    default_stage_a_weights,
    default_stage_b_checkpoint,
    ensure_default_stage_a_weights,
    ensure_default_stage_b_checkpoint,
)
from src.models.yolo_stage_a import YoloStageA, YoloStageAConfig
from src.pipeline.export_musicxml import _write_musicxml_safe, assembled_score_to_music21, load_assembled_score, validate_musicxml_roundtrip, write_musicxml_per_page, write_musicxml_per_system


def _write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _vertical_overlap_ratio(a: Dict[str, float], b: Dict[str, float]) -> float:
    overlap_top = max(float(a["y_min"]), float(b["y_min"]))
    overlap_bottom = min(float(a["y_max"]), float(b["y_max"]))
    overlap = max(0.0, overlap_bottom - overlap_top)
    a_h = max(1e-6, float(a["y_max"]) - float(a["y_min"]))
    b_h = max(1e-6, float(b["y_max"]) - float(b["y_min"]))
    return overlap / max(1e-6, min(a_h, b_h))


def _dedupe_page_crop_rows_keep_latest(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    # Keeps one crop for near-identical staff detections; keeps latest (last) in each duplicate cluster.
    grouped: Dict[int, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row.get("system_index", 0)), []).append(row)

    deduped: List[Dict[str, object]] = []
    for system_index in sorted(grouped):
        kept: List[Dict[str, object]] = []
        for row in grouped[system_index]:
            bbox = row.get("bbox", {})
            y_center = (float(bbox["y_min"]) + float(bbox["y_max"])) / 2.0
            matched_idx = None
            for idx, prev in enumerate(kept):
                prev_bbox = prev.get("bbox", {})
                prev_center = (float(prev_bbox["y_min"]) + float(prev_bbox["y_max"])) / 2.0
                min_height = max(
                    1e-6,
                    min(
                        float(bbox["y_max"]) - float(bbox["y_min"]),
                        float(prev_bbox["y_max"]) - float(prev_bbox["y_min"]),
                    ),
                )
                center_distance = abs(y_center - prev_center)
                overlap_ratio = _vertical_overlap_ratio(bbox, prev_bbox)
                if overlap_ratio >= 0.80 and center_distance <= (0.40 * min_height):
                    matched_idx = idx
                    break
            if matched_idx is None:
                kept.append(row)
            else:
                # Replace with current row so the latest crop is kept.
                kept[matched_idx] = row
        deduped.extend(kept)
    return deduped


def _render_pdf_with_pymupdf(pdf_path: Path, output_dir: Path, dpi: int) -> List[Path]:
    import fitz  # type: ignore

    output_dir.mkdir(parents=True, exist_ok=True)
    zoom = max(0.1, float(dpi) / 72.0)
    matrix = fitz.Matrix(zoom, zoom)
    pages: List[Path] = []
    with fitz.open(str(pdf_path)) as document:
        for page_index, page in enumerate(document):
            image_path = output_dir / f"page_{page_index + 1:04d}.png"
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            pixmap.save(str(image_path))
            pages.append(image_path)
    return pages


def _render_pdf_with_pdfium(pdf_path: Path, output_dir: Path, dpi: int) -> List[Path]:
    import pypdfium2 as pdfium  # type: ignore

    output_dir.mkdir(parents=True, exist_ok=True)
    scale = max(0.1, float(dpi) / 72.0)
    pages: List[Path] = []
    document = pdfium.PdfDocument(str(pdf_path))
    try:
        for page_index in range(len(document)):
            page = document[page_index]
            bitmap = page.render(scale=scale)
            image = bitmap.to_pil()
            image_path = output_dir / f"page_{page_index + 1:04d}.png"
            image.save(image_path)
            pages.append(image_path)
            if hasattr(page, "close"):
                page.close()
    finally:
        if hasattr(document, "close"):
            document.close()
    return pages


def _render_pdf_with_pillow(pdf_path: Path, output_dir: Path) -> List[Path]:
    from PIL import Image, ImageSequence

    output_dir.mkdir(parents=True, exist_ok=True)
    pages: List[Path] = []
    with Image.open(pdf_path) as document:
        for page_index, frame in enumerate(ImageSequence.Iterator(document)):
            image_path = output_dir / f"page_{page_index + 1:04d}.png"
            frame.convert("RGB").save(image_path)
            pages.append(image_path)
    return pages


def render_pdf_pages(pdf_path: Path, output_dir: Path, dpi: int) -> List[Path]:
    render_errors: List[str] = []

    for renderer_name, renderer in (
        ("PyMuPDF", lambda: _render_pdf_with_pymupdf(pdf_path, output_dir, dpi)),
        ("pypdfium2", lambda: _render_pdf_with_pdfium(pdf_path, output_dir, dpi)),
        ("Pillow", lambda: _render_pdf_with_pillow(pdf_path, output_dir)),
    ):
        try:
            pages = renderer()
            if pages:
                return pages
        except Exception as exc:  # pragma: no cover
            render_errors.append(f"{renderer_name}: {exc}")

    detail = "; ".join(render_errors) if render_errors else "no renderer available"
    raise RuntimeError(
        "Failed to render PDF pages. Install one renderer dependency: "
        "'pip install pymupdf' or 'pip install pypdfium2'. "
        f"Details: {detail}"
    )


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run Stage-A + Stage-B + assembly + MusicXML export from a PDF input.")
    parser.add_argument("--pdf", type=Path, required=True, help="Input PDF score path.")
    parser.add_argument("--output-musicxml", type=Path, required=True, help="Output MusicXML path.")
    parser.add_argument("--project-root", type=Path, default=project_root, help="Repository root path.")
    parser.add_argument("--work-dir", type=Path, default=project_root / "output" / "pdf_run", help="Working directory.")

    parser.add_argument(
        "--weights",
        type=Path,
        default=default_stage_a_weights(project_root),
        help="Stage-A YOLO weights path.",
    )
    parser.add_argument(
        "--stage-b-checkpoint",
        type=Path,
        default=default_stage_b_checkpoint(project_root),
        help="Stage-B checkpoint path (.safetensors).",
    )
    parser.add_argument("--confidence", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold.")
    parser.add_argument("--dedupe-iou-threshold", type=float, default=0.85, help="IoU threshold for duplicate staff-box removal.")
    parser.add_argument(
        "--enforce-full-width-crops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use each system's full detected x-span for all crops, with fixed extra padding.",
    )
    parser.add_argument(
        "--full-width-left-page-edge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When full-width mode is enabled, force crop x_min to page left edge.",
    )
    parser.add_argument(
        "--full-width-right-page-edge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When full-width mode is enabled, force crop x_max to page right edge.",
    )
    parser.add_argument("--pdf-dpi", type=int, default=300, help="PDF render DPI.")
    parser.add_argument(
        "--manual-page-crop",
        action="store_true",
        help="Open an interactive window to add one or more bar crops per rendered page before Stage-B inference.",
    )

    parser.add_argument("--beam-width", type=int, default=5, help="Stage-B constrained beam width.")
    parser.add_argument("--max-decode-steps", type=int, default=512, help="Stage-B max decode steps.")
    parser.add_argument("--length-penalty-alpha", type=float, default=0.6, help="Stage-B length normalization alpha.")
    parser.add_argument(
        "--stage-b-kv-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable KV cache in Stage-B decoding (default: enabled).",
    )
    parser.add_argument("--image-height", type=int, default=250, help="Stage-B input image height.")
    parser.add_argument("--image-max-width", type=int, default=2500, help="Stage-B input image max width (max 3000).")
    parser.add_argument("--stage-b-device", type=str, default=None, help="Stage-B device (e.g. cuda, cpu).")
    parser.add_argument("--progress-every-seconds", type=float, default=10.0, help="Stage-B progress log interval.")
    parser.add_argument("--quiet", action="store_true", help="Disable Stage-B progress logs.")
    parser.add_argument(
        "--strict-export",
        action="store_true",
        help="Fail if MusicXML does not pass XSD validation (default: write best-effort output).",
    )
    export_group = parser.add_mutually_exclusive_group()
    export_group.add_argument(
        "--export-pages",
        action="store_true",
        help="Export one MusicXML file per page instead of a single combined file.",
    )
    export_group.add_argument(
        "--export-systems",
        action="store_true",
        help="Export one MusicXML file per system instead of a single combined file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = args.project_root.resolve()
    pdf_path = (args.pdf if args.pdf.is_absolute() else (project_root / args.pdf)).resolve()
    output_musicxml = (
        args.output_musicxml if args.output_musicxml.is_absolute() else (project_root / args.output_musicxml)
    ).resolve()
    work_dir = (args.work_dir if args.work_dir.is_absolute() else (project_root / args.work_dir)).resolve()
    weights = (args.weights if args.weights.is_absolute() else (project_root / args.weights)).resolve()
    stage_b_checkpoint = (
        args.stage_b_checkpoint
        if args.stage_b_checkpoint.is_absolute()
        else (project_root / args.stage_b_checkpoint)
    ).resolve()

    if weights == default_stage_a_weights(project_root).resolve():
        weights = ensure_default_stage_a_weights(project_root).resolve()
    if stage_b_checkpoint == default_stage_b_checkpoint(project_root).resolve():
        stage_b_checkpoint = ensure_default_stage_b_checkpoint(project_root).resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not bool(args.manual_page_crop) and not weights.exists():
        raise FileNotFoundError(f"Stage-A weights not found: {weights}")
    if not stage_b_checkpoint.exists():
        raise FileNotFoundError(f"Stage-B checkpoint not found: {stage_b_checkpoint}")

    pages_dir = work_dir / "pages"
    rendered_page_images = render_pdf_pages(pdf_path, pages_dir, dpi=max(72, int(args.pdf_dpi)))
    if not rendered_page_images:
        raise RuntimeError(f"No pages rendered from PDF: {pdf_path}")
    manual_crop_metadata = None
    page_images = list(rendered_page_images)
    if bool(args.manual_page_crop):
        manually_cropped_pages_dir = work_dir / "manual_pages"
        manual_crop_metadata = work_dir / "manual_page_crops.json"
        page_images = crop_pages_with_editor(
            rendered_page_images,
            manually_cropped_pages_dir,
            metadata_path=manual_crop_metadata,
        )

    all_crop_rows: List[Dict[str, object]] = []
    if bool(args.manual_page_crop):
        manual_rows_raw = _read_json(Path(manual_crop_metadata))
        if not isinstance(manual_rows_raw, list):
            raise RuntimeError(f"Invalid manual crop metadata format: {manual_crop_metadata}")
        ordered_manual_rows = sorted(
            [row for row in manual_rows_raw if isinstance(row, dict)],
            key=lambda row: (
                int(row["page_index"]),
                int(row.get("bar_index") if row.get("bar_index") is not None else -1),
            ),
        )
        for row in ordered_manual_rows:
            page_index = int(row["page_index"])
            bar_index = row.get("bar_index")
            crop_path = Path(str(row["output_path"])).resolve()
            rect = row.get("bar_bbox")
            if isinstance(rect, dict):
                bbox = {
                    "x_min": float(rect["left"]),
                    "y_min": float(rect["top"]),
                    "x_max": float(rect["right"]),
                    "y_max": float(rect["bottom"]),
                }
            else:
                from PIL import Image

                with Image.open(crop_path) as image_obj:
                    bbox = {
                        "x_min": 0.0,
                        "y_min": 0.0,
                        "x_max": float(image_obj.width),
                        "y_max": float(image_obj.height),
                    }
            all_crop_rows.append(
                {
                    "sample_id": f"page_{page_index + 1:04d}:{crop_path.stem}",
                    "crop_path": str(crop_path),
                    "system_index": int(bar_index if bar_index is not None else 0),
                    "staff_index": 0,
                    "page_index": page_index,
                    "bbox": bbox,
                }
            )
    else:
        stage_a = YoloStageA(
            YoloStageAConfig(
                weights_path=weights,
                confidence_threshold=float(args.confidence),
                iou_threshold=float(args.iou),
                dedupe_iou_threshold=float(args.dedupe_iou_threshold),
                enforce_full_width_crops=bool(args.enforce_full_width_crops),
                full_width_left_page_edge=bool(args.full_width_left_page_edge),
                full_width_right_page_edge=bool(args.full_width_right_page_edge),
            )
        )
        for page_index, page_image in enumerate(page_images):
            detections = stage_a.detect_regions(page_image)
            crops_dir = work_dir / "crops" / f"page_{page_index + 1:04d}"
            crops = stage_a.crop_staff_regions(page_image, detections, crops_dir)
            page_crop_rows: List[Dict[str, object]] = []
            for crop in crops:
                sample_id = f"page_{page_index + 1:04d}:{Path(crop.crop_path).stem}"
                page_crop_rows.append(
                    {
                        "sample_id": sample_id,
                        "crop_path": str(Path(crop.crop_path).resolve()),
                        "system_index": int(crop.system_index),
                        "staff_index": int(crop.staff_index),
                        "page_index": int(page_index),
                        "bbox": {
                            "x_min": float(crop.bbox.x_min),
                            "y_min": float(crop.bbox.y_min),
                            "x_max": float(crop.bbox.x_max),
                            "y_max": float(crop.bbox.y_max),
                        },
                    }
                )
            all_crop_rows.extend(_dedupe_page_crop_rows_keep_latest(page_crop_rows))

    if not all_crop_rows:
        raise RuntimeError("No staff crops were detected across all PDF pages.")

    stage_a_manifest = work_dir / "stage_a_crops_all.jsonl"
    _write_jsonl(stage_a_manifest, all_crop_rows)

    # Run CV analysis on each crop (unless --cv-priors off)
    stage_b_predictions = work_dir / "stage_b_predictions.jsonl"
    stage_b_result = _run_stage_b_inference_with_progress(
        project_root=project_root,
        crops_manifest=stage_a_manifest,
        output_predictions=stage_b_predictions,
        checkpoint=stage_b_checkpoint,
        beam_width=max(1, int(args.beam_width)),
        max_decode_steps=max(8, int(args.max_decode_steps)),
        image_height=max(32, int(args.image_height)),
        image_max_width=min(3000, max(256, int(args.image_max_width))),
        device_name=args.stage_b_device,
        progress_every_seconds=max(0.2, float(args.progress_every_seconds)),
        quiet=bool(args.quiet),
        length_penalty_alpha=float(args.length_penalty_alpha),
        use_kv_cache=bool(args.stage_b_kv_cache),
    )

    assembly_manifest = work_dir / "assembled_score.json"
    assembly_result = run_assemble(
        argparse.Namespace(
            staff_predictions=stage_b_predictions,
            output_assembly=assembly_manifest,
        )
    )
    separate_pages = bool(getattr(args, "export_pages", False))
    separate_systems = bool(getattr(args, "export_systems", False))

    if separate_pages:
        assembled = load_assembled_score(assembly_manifest)
        page_results = write_musicxml_per_page(assembled, output_musicxml)
        export_result = {
            "separate_pages": True,
            "pages": page_results,
            "output_paths": [r.get("output_path") for r in page_results],
        }
    elif separate_systems:
        assembled = load_assembled_score(assembly_manifest)
        system_results = write_musicxml_per_system(assembled, output_musicxml)
        export_result = {
            "separate_systems": True,
            "systems": system_results,
            "output_paths": [r.get("output_path") for r in system_results],
        }
    else:
        try:
            export_result = run_export(
                argparse.Namespace(
                    assembly_manifest=assembly_manifest,
                    output_musicxml=output_musicxml,
                )
            )
        except (ValueError, KeyError) as exc:
            if bool(args.strict_export) and isinstance(exc, ValueError):
                raise
            # Best-effort fallback: write MusicXML even if schema validation or voice consistency fails.
            assembled = load_assembled_score(assembly_manifest)
            music_score = assembled_score_to_music21(assembled)
            output_musicxml.parent.mkdir(parents=True, exist_ok=True)
            _write_musicxml_safe(music_score, output_musicxml)
            try:
                validation = validate_musicxml_roundtrip(music_score)
            except (KeyError, Exception):
                validation = {"schema_valid": False, "best_effort_validation_skipped": True}
            export_result = {
                **validation,
                "output_path": str(output_musicxml),
                "best_effort": True,
                "warning": str(exc),
            }

    result = {
        "pdf": str(pdf_path),
        "rendered_pages": len(page_images),
        "stage_a_crops": len(all_crop_rows),
        "stage_b": stage_b_result,
        "assembly": assembly_result,
        "export": export_result,
        "outputs": {
            "work_dir": str(work_dir),
            "stage_a_manifest": str(stage_a_manifest),
            "stage_b_predictions": str(stage_b_predictions),
            "assembly_manifest": str(assembly_manifest),
            "musicxml": str(output_musicxml) if not (separate_pages or separate_systems) else None,
            "musicxml_files": export_result.get("output_paths") if (separate_pages or separate_systems) else None,
            "manual_page_crops": str(manual_crop_metadata) if manual_crop_metadata is not None else None,
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
