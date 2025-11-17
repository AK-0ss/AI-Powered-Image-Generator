from __future__ import annotations

import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from ai_image_generator.config import AppConfig, ModelConfig, PipelineConfig, StorageConfig, build_config
from ai_image_generator.filters import check_prompt
from ai_image_generator.generator import GenerationRequest, TextToImageGenerator
from ai_image_generator.prompts import STYLE_PRESETS, enhance_prompt
from ai_image_generator.storage import persist_generation
from ai_image_generator.watermark import apply_watermark


@st.cache_resource(show_spinner=False)
def load_components() -> tuple[TextToImageGenerator, ModelConfig, PipelineConfig, StorageConfig, AppConfig]:
    model_cfg, pipe_cfg, storage_cfg, app_cfg = build_config()
    generator = TextToImageGenerator(model_cfg, pipe_cfg, app_cfg)
    return generator, model_cfg, pipe_cfg, storage_cfg, app_cfg


def layout_sidebar(model_cfg: ModelConfig, generator: TextToImageGenerator) -> None:
    st.sidebar.header("System")
    st.sidebar.caption(
        f"Model: `{model_cfg.model_id}`\n\nDevice: `{generator.device}`\n\n"
        "Switch models by setting the `MODEL_ID` environment variable."
    )
    st.sidebar.markdown("### Responsible Use")
    st.sidebar.write(
        "- Avoid harmful or explicit prompts.\n"
        "- Generated images are watermarked.\n"
        "- Refer to `docs/ethics.md` for full guidelines."
    )


def _progress_callback_factory(progress_bar, status_text):
    def _callback(step: int, total: int) -> None:
        progress = int((step / max(total, 1)) * 100)
        progress_bar.progress(progress, text=f"Diffusion steps: {step}/{total}")
        status_text.info(f"Running step {step}/{total}")

    return _callback


def main() -> None:
    st.set_page_config(page_title="AI Image Generator", page_icon="🎨", layout="wide")
    generator, model_cfg, pipe_cfg, storage_cfg, app_cfg = load_components()
    layout_sidebar(model_cfg, generator)

    st.title("AI-Powered Image Generator")
    st.write(
        "Describe any scene and generate images locally using open-source diffusion models. "
        "This demo supports GPU acceleration with CPU fallback."
    )

    with st.expander("Hardware Tips", expanded=False):
        st.write(
            "- GPU (CUDA) drastically speeds up generation.\n"
            "- For CPU runs, lower image resolution or inference steps.\n"
            "- Use `MODEL_ID` env var to try lighter models such as `stabilityai/sdxl-turbo`."
        )

    styles = ["none"] + sorted(STYLE_PRESETS.keys())

    with st.form("generation_form"):
        prompt = st.text_area("Prompt", placeholder="e.g. a futuristic city at sunset", height=120)
        style_choice = st.selectbox("Style guidance", options=styles, index=0)
        negative_prompt = st.text_input("Negative prompt (optional)", placeholder="e.g. low quality, blurry")
        num_images = st.slider("Number of images", 1, app_cfg.max_images_per_prompt, 1)
        guidance_scale = st.slider("Guidance scale", 1.0, 15.0, pipe_cfg.guidance_scale, 0.5)
        num_inference_steps = st.slider("Diffusion steps", 10, 60, pipe_cfg.num_inference_steps, 1)
        height = st.number_input("Height", min_value=256, max_value=1024, value=pipe_cfg.height, step=64)
        width = st.number_input("Width", min_value=256, max_value=1024, value=pipe_cfg.width, step=64)
        seed = st.number_input("Seed (optional, -1 for random)", min_value=-1, value=-1, step=1)
        submitted = st.form_submit_button("Generate Images", type="primary")

    if submitted:
        if not prompt.strip():
            st.error("Prompt is required.")
            return

        filter_result = check_prompt(prompt)
        if not filter_result.allowed:
            st.error(f"Prompt blocked: {filter_result.reason}. Terms: {', '.join(filter_result.blocked_terms)}")
            return

        style = style_choice if style_choice != "none" else None
        prompt_bundle = enhance_prompt(
            prompt,
            style=style,
            extra_negative=[negative_prompt] if negative_prompt else None,
        )

        request = GenerationRequest(
            prompt_bundle=prompt_bundle,
            num_images=num_images,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            seed=seed if seed >= 0 else None,
        )

        est_time = generator.estimate_duration(num_images, num_inference_steps)
        st.info(f"Estimated completion time: ~{est_time}s on this hardware.")

        progress_bar = st.progress(0, text="Preparing pipeline…")
        status_text = st.empty()
        callback = _progress_callback_factory(progress_bar, status_text)

        with st.spinner("Generating..."):
            images = generator.generate(request, progress_callback=callback)

        progress_bar.progress(100, text="Done")
        status_text.success("Generation complete!")

        stamped_images = [
            apply_watermark(img, text=app_cfg.watermark_text, opacity=app_cfg.watermark_opacity) for img in images
        ]

        metadata = {
            "prompt": prompt,
            "enhanced_prompt": prompt_bundle.prompt,
            "negative_prompt": prompt_bundle.negative_prompt,
            "style": style,
            "num_images": num_images,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "height": height,
            "width": width,
            "seed": request.seed,
            "model_id": model_cfg.model_id,
            "device": generator.device,
            "timestamp": datetime.utcnow().isoformat(),
        }

        save_result = persist_generation(stamped_images, metadata, storage_cfg)
        st.success(f"Images saved to `{save_result['directory']}`")

        cols = st.columns(num_images)
        for idx, (img, col) in enumerate(zip(stamped_images, cols, strict=False), start=1):
            with col:
                st.image(img, caption=f"Image {idx}")
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                buffer.seek(0)
                st.download_button(
                    label=f"Download image {idx}",
                    data=buffer.getvalue(),
                    file_name=f"image_{idx:02d}.png",
                    mime="image/png",
                    key=f"download_{idx}",
                )


if __name__ == "__main__":
    main()

