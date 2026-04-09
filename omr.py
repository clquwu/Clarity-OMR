#!/usr/bin/env python3
"""Clarity OMR - PDF to MusicXML.

Usage:
    python omr.py score.pdf                     # outputs score-omr.musicxml
    python omr.py score.pdf -o output.musicxml  # custom output path
    python omr.py score.pdf --fast              # faster (lower quality)

Models are downloaded automatically from HuggingFace on first run.
Stage-B runs from `info/model.safetensors` by default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.model_assets import (
    ensure_default_stage_a_weights,
    ensure_default_stage_b_checkpoint,
)


def _detect_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _default_output(pdf_path: Path) -> Path:
    return pdf_path.with_stem(pdf_path.stem + "-omr").with_suffix(".musicxml")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clarity OMR - convert a PDF score to MusicXML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python omr.py score.pdf\n"
               "  python omr.py score.pdf -o output.musicxml\n"
               "  python omr.py score.pdf --fast\n",
    )
    parser.add_argument("pdf", type=Path, help="Input PDF score.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output MusicXML path (default: <input>-omr.musicxml).")
    parser.add_argument("--fast", action="store_true", help="Faster CPU inference only (beam-width 2 instead of 5).")
    parser.add_argument("--beam-width", type=int, default=None, help="Override beam width (default: 5, --fast: 2).")
    parser.add_argument("--work-dir", type=Path, default=None, help="Working directory for intermediate files.")
    parser.add_argument("--device", type=str, default=None, help="Force device (cuda or cpu).")
    parser.add_argument("--pdf-dpi", type=int, default=300, help="PDF render DPI (default: 300).")
    export_group = parser.add_mutually_exclusive_group()
    export_group.add_argument("--export-pages", action="store_true", help="Export one MusicXML file per page.")
    export_group.add_argument("--export-systems", action="store_true", help="Export one MusicXML file per system.")
    args = parser.parse_args()

    # Resolve paths
    pdf_path = args.pdf.resolve()
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    output_path = (args.output or _default_output(pdf_path)).resolve()
    project_root = Path(__file__).resolve().parent
    work_dir = (args.work_dir or (project_root / "output" / "pdf_run")).resolve()

    # Download and resolve default models if needed
    stage_a_weights = ensure_default_stage_a_weights(project_root)
    stage_b_checkpoint = ensure_default_stage_b_checkpoint(project_root)

    # Detect device and set defaults
    device = args.device or _detect_device()
    if args.fast and device != "cpu":
        print("Error: --fast is only supported on CPU.", file=sys.stderr)
        sys.exit(1)
    beam_width = args.beam_width
    if beam_width is None:
        beam_width = 2 if args.fast else 5

    print(f"Input:  {pdf_path}", file=sys.stderr)
    print(f"Output: {output_path}", file=sys.stderr)
    print(f"Device: {device}", file=sys.stderr)
    print(f"Beam:   {beam_width}", file=sys.stderr)
    print(file=sys.stderr)

    # Run pipeline via parsed args
    sys.path.insert(0, str(project_root))
    from src.pdf_to_musicxml import build_parser, main as _pipeline_main

    pipeline_argv = [
        "--pdf", str(pdf_path),
        "--output-musicxml", str(output_path),
        "--project-root", str(project_root),
        "--work-dir", str(work_dir),
        "--weights", str(stage_a_weights),
        "--stage-b-checkpoint", str(stage_b_checkpoint),
        "--beam-width", str(beam_width),
        "--image-height", "250",
        "--image-max-width", "2500",
        "--length-penalty-alpha", "0.4",
        "--pdf-dpi", str(args.pdf_dpi),
        "--stage-b-device", device,
    ]
    if args.export_systems:
        pipeline_argv.append("--export-systems")
    elif args.export_pages:
        pipeline_argv.append("--export-pages")
    saved_argv = sys.argv
    sys.argv = ["omr.py"] + pipeline_argv
    try:
        _pipeline_main()
    finally:
        sys.argv = saved_argv


if __name__ == "__main__":
    main()
