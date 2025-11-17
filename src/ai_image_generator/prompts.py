from __future__ import annotations

from dataclasses import dataclass


BASE_QUALITY_TOKENS = [
    "highly detailed",
    "8k resolution",
    "sharp focus",
    "professional lighting",
    "volumetric light",
]


STYLE_PRESETS = {
    "photorealistic": {
        "positive": [
            "photorealistic",
            "ultra realistic",
            "dslr",
            "cinematic composition",
        ],
        "negative": ["cartoon", "anime", "painting"],
    },
    "digital_art": {
        "positive": [
            "digital painting",
            "concept art",
            "illustration",
            "dynamic brush strokes",
        ],
        "negative": ["noise", "grainy"],
    },
    "anime": {
        "positive": [
            "anime key visual",
            "studio quality",
            "clean line art",
            "cel shading",
        ],
        "negative": ["blurry", "realistic skin"],
    },
    "cyberpunk": {
        "positive": [
            "cyberpunk aesthetic",
            "neon lighting",
            "futuristic interface",
            "rain soaked streets",
        ],
        "negative": ["low contrast", "monochrome"],
    },
    "van_gogh": {
        "positive": [
            "vincent van gogh style",
            "impasto oil painting",
            "post impressionism",
            "bold swirling brushwork",
        ],
        "negative": ["photorealistic", "digital"],
    },
}


@dataclass
class PromptBundle:
    prompt: str
    negative_prompt: str


def enhance_prompt(
    base_prompt: str,
    style: str | None = None,
    extra_positive: list[str] | None = None,
    extra_negative: list[str] | None = None,
    include_quality_tokens: bool = True,
) -> PromptBundle:
    """Apply lightweight prompt engineering heuristics."""

    positive_tokens: list[str] = [base_prompt.strip()]
    negative_tokens: list[str] = []

    if include_quality_tokens:
        positive_tokens.extend(BASE_QUALITY_TOKENS)

    if style and style in STYLE_PRESETS:
        preset = STYLE_PRESETS[style]
        positive_tokens.extend(preset["positive"])
        negative_tokens.extend(preset["negative"])

    if extra_positive:
        positive_tokens.extend(extra_positive)

    if extra_negative:
        negative_tokens.extend(extra_negative)

    positive = ", ".join(dict.fromkeys(filter(None, positive_tokens)))
    negative = ", ".join(dict.fromkeys(filter(None, negative_tokens)))

    return PromptBundle(prompt=positive, negative_prompt=negative)

