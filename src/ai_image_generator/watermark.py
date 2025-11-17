from __future__ import annotations

from pathlib import Path
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont


def _resolve_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidate_paths = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidate_paths:
        if path.exists():
            return ImageFont.truetype(str(path), size=32)
    return ImageFont.load_default()


def apply_watermark(
    image: Image.Image,
    text: str,
    opacity: float = 0.6,
    margin_ratio: float = 0.03,
) -> Image.Image:
    """Overlay a semi-transparent watermark to signal AI origin."""

    if not text:
        return image

    if image.mode != "RGBA":
        base = image.convert("RGBA")
    else:
        base = image.copy()

    txt_layer = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_layer)
    font = _resolve_font()

    width, height = base.size
    margin = int(min(width, height) * margin_ratio)

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    x = width - text_width - margin
    y = height - text_height - margin

    draw.rounded_rectangle(
        [(x - margin // 2, y - margin // 2), (x + text_width + margin // 2, y + text_height + margin // 2)],
        radius=8,
        fill=(0, 0, 0, int(255 * opacity * 0.6)),
    )

    draw.text(
        (x, y),
        text,
        font=font,
        fill=(255, 255, 255, int(255 * opacity)),
    )

    watermarked = Image.alpha_composite(base, txt_layer)
    return watermarked.convert("RGB")

