## Generative Model Research Notes

### 1. GANs
- Invented by Goodfellow et al. (2014). Generator + discriminator trained adversarially.
- Strengths: fast inference, sharp images when training converges, good for domain-specific datasets.
- Limitations: mode collapse, unstable training, hard to scale to diverse prompts, limited controllability for text.

### 2. Diffusion Models
- Denoising diffusion probabilistic models (DDPM) add Gaussian noise then learn to reverse the process.
- Stable Diffusion = latent diffusion: applies diffusion in the VAE latent space for efficiency.
- Advantages over GANs: stable training, better diversity, natural conditioning (text via CLIP text encoder), scalable via guidance.
- Sampling speed improved with advanced schedulers (DDIM, DPM-Solver, Euler). Guidance scale trades fidelity vs. creativity.

### 3. Prompt Engineering
- Positive prompt enhancements: add quality descriptors (e.g., “4k, highly detailed, global illumination”).
- Style presets: curated token bundles that bias generation toward artistic movements or mediums.
- Negative prompts remove artifacts (“blurry, deformed hands, text artifact”).
- Empirical strategy: craft base prompt → add 2‑3 descriptive adjectives → append style preset → include camera/lighting terms.

### 4. Model Selection Criteria
- License compatibility (CreativeML Open RAIL for SD 1.5, SDXL).
- VRAM footprint: SD 1.5 ≈ 4.5 GB FP16, SDXL ≈ 7+ GB. Turbo models drastically faster but slightly lower fidelity.
- Hardware scaling: GPU preferred; CPU works with attention slicing, sequential offload, smaller resolutions (<768).

### 5. References
- “High-Resolution Image Synthesis with Latent Diffusion Models,” Rombach et al., 2022.
- “Elucidating the Design Space of Diffusion-Based Generative Models,” Karras et al., 2022.
- https://huggingface.co/docs/diffusers

