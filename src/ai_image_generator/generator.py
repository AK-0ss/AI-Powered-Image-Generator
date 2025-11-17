from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

import torch
from diffusers import (
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    StableDiffusionPipeline,
)

from .config import AppConfig, ModelConfig, PipelineConfig
from .prompts import PromptBundle


Callback = Callable[[int, int], None]


SCHEDULER_MAP = {
    "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
    "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
    "EulerDiscreteScheduler": EulerDiscreteScheduler,
}


@dataclass
class GenerationRequest:
    prompt_bundle: PromptBundle
    num_images: int
    num_inference_steps: int
    guidance_scale: float
    height: int
    width: int
    seed: Optional[int] = None
    negative_prompt: Optional[str] = None


class TextToImageGenerator:
    def __init__(self, model_cfg: ModelConfig, pipe_cfg: PipelineConfig, app_cfg: AppConfig):
        self.model_cfg = model_cfg
        self.pipe_cfg = pipe_cfg
        self.app_cfg = app_cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = self._resolve_dtype(model_cfg.torch_dtype)
        self.pipeline = self._load_pipeline()

    def _resolve_dtype(self, dtype_pref: str) -> torch.dtype:
        if dtype_pref == "auto":
            if self.device == "cuda":
                return torch.float16
            return torch.float32
        if dtype_pref == "float16":
            return torch.float16
        if dtype_pref == "bfloat16":
            return torch.bfloat16
        return torch.float32

    def _load_pipeline(self) -> StableDiffusionPipeline:
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_cfg.model_id,
            torch_dtype=self.torch_dtype,
            revision=self.model_cfg.revision,
            use_safetensors=self.model_cfg.use_safetensors,
            cache_dir=self.model_cfg.local_cache_dir,
        )
        SchedulerClass = SCHEDULER_MAP.get(self.pipe_cfg.scheduler_name, DPMSolverMultistepScheduler)
        pipe.scheduler = SchedulerClass.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)

        if self.device == "cuda":
            pipe.to("cuda")
            if self.model_cfg.enable_cpu_offload:
                pipe.enable_model_cpu_offload()
        else:
            pipe.to("cpu")
            pipe.enable_attention_slicing()

        if self.pipe_cfg.enable_compiler and hasattr(torch, "compile"):
            pipe.unet = torch.compile(pipe.unet)

        return pipe

    def estimate_duration(self, num_images: int, num_inference_steps: int) -> float:
        base = 0.6 if self.device == "cuda" else 2.2
        step_factor = num_inference_steps / 30
        return round(base * step_factor * max(1, num_images), 2)

    def generate(
        self,
        request: GenerationRequest,
        progress_callback: Optional[Callback] = None,
    ) -> List:
        """Run inference and return PIL images."""

        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(request.seed)

        total_steps = request.num_inference_steps

        def callback(step: int, timestep: int, latents) -> None:  # noqa: ANN001
            if progress_callback:
                progress_callback(step, total_steps)

        images = self.pipeline(
            prompt=[request.prompt_bundle.prompt] * request.num_images,
            negative_prompt=[request.prompt_bundle.negative_prompt or ""] * request.num_images,
            num_inference_steps=total_steps,
            guidance_scale=request.guidance_scale,
            height=request.height,
            width=request.width,
            generator=generator,
            callback=callback if progress_callback else None,
            callback_steps=1,
        ).images

        return images

