## AI-Powered Image Generator

Local-first Streamlit app that turns natural-language prompts into high-fidelity visuals by wrapping the Hugging Face `diffusers` Stable Diffusion pipeline. The project targets rapid experimentation, prompt iteration, and responsible sharing with guardrails such as lexical filtering, watermarking, and detailed metadata.

---

### Project Overview & Architecture
- **UI layer (`app.py`)** – Streamlit form drives prompt collection, tuning controls, and live progress updates. Sidebar exposes hardware + model info and links to ethics guidance.
- **Inference core (`src/ai_image_generator/generator.py`)** – `TextToImageGenerator` builds a Stable Diffusion pipeline with selectable schedulers, torch compile/offload toggles, and seedable execution. Produces PIL images plus ETA estimates.
- **Prompt engineering (`prompts.py`)** – Style presets (photorealistic, digital_art, anime, cyberpunk, van_gogh) inject positive/negative tokens and reusable quality descriptors through `enhance_prompt`.
- **Safety + policy (`filters.py`, `docs/ethics.md`)** – Keyword filter blocks obvious misuse before inference; documentation provides context for acceptable usage.
- **Post-processing (`watermark.py`, `storage.py`)** – Each image receives an “AI Generated” watermark, then gets persisted as PNG/JPEG along with a `metadata.json` manifest that records prompts, parameters, seeds, and device info.
- **Configuration (`config.py`)** – Dataclasses centralize environment overrides for model selection, scheduler defaults, and storage paths; `build_config` prepares cache/output folders automatically.

```
Prompt -> filter -> enhance_prompt -> TextToImageGenerator
      -> watermark -> persist_generation -> outputs/<timestamp>/
```

---

### Setup & Installation
1. **Clone & Python**  
   - Requires Python 3.10+ (3.11 tested).  
   - Optional: `py -3.11 -m venv .venv`
2. **Activate environment**
   ```powershell
   .venv\Scripts\activate
   ```
3. **Install PyTorch that matches your CUDA drivers**  
   - CUDA 12.1 example:  
     ```powershell
     pip install torch --index-url https://download.pytorch.org/whl/cu121
     ```  
   - CPU-only or ROCm: install the default `torch` wheel.
4. **Install project deps**
   ```powershell
pip install -r requirements.txt
```
5. **Model download / caching**
   - The app automatically caches models under `models/` using `MODEL_ID`.  
   - To prefetch (useful offline or on CI), authenticate with Hugging Face and run:
     ```powershell
     huggingface-cli login          # optional if the model is gated
     huggingface-cli download runwayml/stable-diffusion-v1-5 --local-dir models/models--runwayml--stable-diffusion-v1-5
     ```
   - Swap to SDXL Turbo or custom checkpoints by editing `.env` (`MODEL_ID=stabilityai/sdxl-turbo`).
6. **Environment configuration (optional)**  
   Create `.env` in the repo root:
   ```
   MODEL_ID=runwayml/stable-diffusion-v1-5
   MODEL_CACHE_DIR=models
   ENABLE_CPU_OFFLOAD=false
   GUIDANCE_SCALE=7.5
   INFERENCE_STEPS=40
   WATERMARK_TEXT=AI Generated • Demo
   ```

---

### Hardware Requirements
| Scenario | Recommended hardware | Notes |
| --- | --- | --- |
| Optimal | NVIDIA GPU with CUDA 11.8+ and ≥8 GB VRAM | 512×512 @ 30–40 steps takes ~6–8 s/image. SDXL high-res prefers 12 GB+. |
| Mid-tier | Laptop GPU (4–6 GB) | Lower resolution (≤640px) or switch to `stabilityai/sdxl-turbo`. Consider enabling `ENABLE_CPU_OFFLOAD`. |
| CPU-only | 8c/16t CPU + 32 GB RAM | Expect 2–4 minutes per image at 512×512. Keep steps ≤25 and use guidance ≤7. |
| Storage | ~10 GB free | Model weights + generated outputs grow quickly; metadata per batch in `outputs/`. |

---

### Usage & Example Prompts
1. Activate your environment and launch Streamlit:
   ```powershell
streamlit run app.py
```
   Visit `http://localhost:8501`.
2. Fill out the form:
   - Prompt + optional negative prompt
   - Style preset
   - Image count, guidance scale, inference steps, resolution, seed
3. Monitor the progress bar + ETA, then review previews and download files or open the saved folder path.

**Example prompts**
- `a solar-powered treehouse built into a redwood forest canopy, cinematic lighting`
- `retro-futuristic skyline at golden hour, art deco reflections, fog`
- `macro shot of bioluminescent coral reef creatures, highly detailed`
- Pair with negatives like `low quality, oversaturated, text artifact` to steer outputs.

---

### Technology Stack & Model Details
- **Frontend**: Streamlit 1.x, leveraging forms, columns, expander panels, and cached resources.
- **Inference**: Hugging Face `diffusers` Stable Diffusion v1.5 by default, with alternate schedulers (`DPMSolverMultistep`, `Euler` variants) and `torch.compile` toggle.
- **ML Runtime**: PyTorch ≥2.1 (CUDA/CPU). Automatic dtype resolution (`float16` on GPU, `float32` on CPU) and optional offloading.
- **Image tooling**: Pillow for watermark overlays, conversion, and export (`PNG` + `JPEG`).
- **Data handling**: `dotenv` for env injection, dataclasses for strongly typed configs, JSON metadata writer, timestamped storage.

---

### Prompt Engineering Tips
- **Structure**: `subject`, `environment`, `mood`, `camera/medium`, `quality tokens`. Example: `ancient library hidden in ice caves, cinematic lighting, 8k, volumetric fog`.
- **Use presets**: Choose a style preset first, then add specific descriptors rather than repeating stylistic cues.
- **Quality tokens**: The app automatically adds `highly detailed`, `professional lighting`, etc.—avoid duplicates for clarity.
- **Negative prompts**: Always list artifacts you dislike (`blurry, bad hands, text, watermark`) to reduce reruns.
- **Seeds**: Fix a seed for iterative tweaks; set `-1` to randomize when exploring.
- **Step vs. guidance**: High guidance (>10) can oversaturate; start 6–8. Steps beyond 40 rarely help unless doing photorealistic scenes on a strong GPU.

---

### Limitations & Future Improvements
- **Generation time**: CPU inference is slow; GPU VRAM below 6 GB requires smaller resolutions. Consider enabling `ENABLE_CPU_OFFLOAD` or switching to lighter checkpoints.
- **Memory footprint**: Stable Diffusion pipelines consume 4–8 GB VRAM (or ~12 GB RAM on CPU). Long prompts plus multiple concurrent images can exhaust resources.
- **Safety**: Current filter is lexical only; sophisticated misuse may slip through. No NSFW classifier is integrated yet.
- **Fine-tuning**: The app does not expose DreamBooth/LoRA training—models are inference-only.
- **Metadata**: All prompts are stored locally; redact manually if sharing outputs externally.

**Future directions**
- LoRA or ControlNet adapters with per-style toggles.
- Fine-tuning workflow (DreamBooth) on user-provided datasets.
- Style-transfer / reference image conditioning hooks.
- Async job queue + REST API for batch rendering.
- Automatic hardware benchmarking + adaptive defaults (steps/resolution).
- Expanded moderation (open-source NSFW detectors, blocklists, or third-party APIs).

---

### Directory Reference
```
├── app.py
├── requirements.txt
├── src/ai_image_generator/
│   ├── config.py         # Env-driven configs
│   ├── generator.py      # Diffusers pipeline wrapper
│   ├── prompts.py        # Prompt helper + presets
│   ├── filters.py        # Lexical guardrails
│   ├── storage.py        # Timestamped exports + metadata
│   └── watermark.py      # Overlay utility
├── models/               # Hugging Face cache (gitignored)
├── outputs/              # Generated batches (PNG/JPEG + metadata)
└── docs/                 # Ethics + research notes
```

Review `docs/ethics.md` before distributing outputs and keep watermarks enabled unless you have explicit consent to remove them.
