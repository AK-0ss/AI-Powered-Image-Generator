from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

from PIL import Image

from .config import StorageConfig


def _timestamp_folder(base_dir: Path) -> Path:
    ts = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    folder = base_dir / ts
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def export_image(image: Image.Image, path: Path, fmt: str, quality: int = 92) -> None:
    params: dict[str, int] = {}
    if fmt.lower() in {"jpeg", "jpg"}:
        params["quality"] = quality
    image.save(path, format=fmt.upper(), **params)


def persist_generation(
    images: Iterable[Image.Image],
    metadata: dict,
    storage_cfg: StorageConfig,
) -> dict:
    """Store generated images and accompanying metadata."""

    output_dir = _timestamp_folder(storage_cfg.base_dir)
    saved_files: list[str] = []

    for idx, image in enumerate(images, start=1):
        if storage_cfg.save_png:
            png_path = output_dir / f"image_{idx:02d}.png"
            export_image(image, png_path, "PNG")
            saved_files.append(str(png_path))

        if storage_cfg.save_jpeg:
            jpg_path = output_dir / f"image_{idx:02d}.jpg"
            export_image(image, jpg_path, "JPEG", storage_cfg.jpeg_quality)
            saved_files.append(str(jpg_path))

    metadata_path = output_dir / storage_cfg.metadata_filename
    metadata_with_files = {**metadata, "files": saved_files}
    metadata_path.write_text(json.dumps(metadata_with_files, indent=2))

    return {
        "directory": str(output_dir),
        "files": saved_files,
        "metadata_path": str(metadata_path),
    }

