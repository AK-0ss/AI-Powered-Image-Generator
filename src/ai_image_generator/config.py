from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


load_dotenv()


@dataclass
class ModelConfig:
    """Configuration for the underlying text-to-image model."""

    model_id: str = os.getenv("MODEL_ID", "runwayml/stable-diffusion-v1-5")
    revision: Optional[str] = os.getenv("MODEL_REVISION", None)
    torch_dtype: str = os.getenv("TORCH_DTYPE", "auto")
    enable_cpu_offload: bool = os.getenv("ENABLE_CPU_OFFLOAD", "false").lower() == "true"
    use_safetensors: bool = os.getenv("USE_SAFETENSORS", "true").lower() != "false"
    local_cache_dir: Path = Path(os.getenv("MODEL_CACHE_DIR", "models"))


@dataclass
class PipelineConfig:
    """Default parameters for image generation."""

    height: int = int(os.getenv("IMAGE_HEIGHT", "768"))
    width: int = int(os.getenv("IMAGE_WIDTH", "768"))
    guidance_scale: float = float(os.getenv("GUIDANCE_SCALE", "7.5"))
    num_inference_steps: int = int(os.getenv("INFERENCE_STEPS", "40"))
    scheduler_name: str = os.getenv("SCHEDULER_NAME", "DPMSolverMultistepScheduler")
    enable_compiler: bool = os.getenv("ENABLE_TORCH_COMPILE", "false").lower() == "true"


@dataclass
class StorageConfig:
    """Where to persist generated assets."""

    base_dir: Path = Path(os.getenv("OUTPUT_DIR", "outputs"))
    metadata_filename: str = "metadata.json"
    save_png: bool = True
    save_jpeg: bool = True
    jpeg_quality: int = 92


@dataclass
class AppConfig:
    """Top-level application wiring."""

    safety_filter_enabled: bool = os.getenv("ENABLE_SAFETY_FILTER", "true").lower() != "false"
    watermark_text: str = os.getenv("WATERMARK_TEXT", "AI Generated • Demo")
    watermark_opacity: float = float(os.getenv("WATERMARK_OPACITY", "0.65"))
    max_images_per_prompt: int = int(os.getenv("MAX_IMAGES_PER_PROMPT", "4"))


def build_config() -> tuple[ModelConfig, PipelineConfig, StorageConfig, AppConfig]:
    """Helper that returns all configs, ensuring directories exist."""

    model_cfg = ModelConfig()
    pipe_cfg = PipelineConfig()
    storage_cfg = StorageConfig()
    app_cfg = AppConfig()

    storage_cfg.base_dir.mkdir(parents=True, exist_ok=True)
    model_cfg.local_cache_dir.mkdir(parents=True, exist_ok=True)

    return model_cfg, pipe_cfg, storage_cfg, app_cfg

