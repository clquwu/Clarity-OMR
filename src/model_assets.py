"""Default model asset resolution and download helpers."""

from __future__ import annotations

import sys
from pathlib import Path

HF_REPO = "clquwu/Clarity-OMR"
DEFAULT_MODEL_DIRNAME = "info"
YOLO_FILENAME = "yolo.pt"
STAGE_B_SAFETENSORS_FILENAME = "model.safetensors"


def default_model_dir(project_root: Path) -> Path:
    return project_root / DEFAULT_MODEL_DIRNAME


def default_stage_a_weights(project_root: Path) -> Path:
    return default_model_dir(project_root) / YOLO_FILENAME


def default_stage_b_checkpoint(project_root: Path) -> Path:
    return default_model_dir(project_root) / STAGE_B_SAFETENSORS_FILENAME


def ensure_default_stage_a_weights(project_root: Path) -> Path:
    model_dir = default_model_dir(project_root)
    weights_path = default_stage_a_weights(project_root)
    if weights_path.exists():
        return weights_path
    _download_from_hf(YOLO_FILENAME, model_dir)
    return weights_path


def ensure_default_stage_b_checkpoint(project_root: Path) -> Path:
    model_dir = default_model_dir(project_root)
    safetensors_path = default_stage_b_checkpoint(project_root)
    if safetensors_path.exists():
        return safetensors_path

    model_dir.mkdir(parents=True, exist_ok=True)
    _download_from_hf(STAGE_B_SAFETENSORS_FILENAME, model_dir)
    return safetensors_path


def _download_from_hf(filename: str, model_dir: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download models. "
            "Install with: pip install huggingface_hub"
        ) from exc

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {filename} from {HF_REPO} ...", file=sys.stderr, flush=True)
    downloaded = hf_hub_download(
        repo_id=HF_REPO,
        filename=filename,
        local_dir=str(model_dir),
    )
    return Path(downloaded).resolve()
