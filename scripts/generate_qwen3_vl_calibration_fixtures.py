#!/usr/bin/env python3
"""Generate deterministic bounded Qwen3-VL shape, object, and OCR fixtures."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _save(image: Image.Image, path: Path) -> None:
    image.save(path, format="PNG", optimize=False)
    path.chmod(0o600)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.expanduser().resolve(strict=False)
    if output.exists():
        raise ValueError("fixture output directory must be new")
    output.mkdir(mode=0o700)

    canvas = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(canvas)
    draw.ellipse((42, 42, 182, 182), fill="black")
    _save(canvas, output / "circle.png")

    canvas = Image.new("RGB", (224, 224), (190, 225, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((42, 105, 182, 198), fill=(245, 220, 170), outline="black", width=5)
    draw.polygon(((28, 110), (112, 35), (196, 110)), fill=(190, 55, 45), outline="black")
    draw.rectangle((94, 142, 130, 198), fill=(105, 65, 35), outline="black", width=4)
    draw.rectangle((55, 125, 85, 155), fill=(110, 190, 240), outline="black", width=4)
    draw.rectangle((139, 125, 169, 155), fill=(110, 190, 240), outline="black", width=4)
    _save(canvas, output / "house.png")

    canvas = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = Path("/System/Library/Fonts/Helvetica.ttc")
    font = ImageFont.truetype(str(font_path), 49)
    text = "APPLE"
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text(((224 - width) / 2, (224 - height) / 2 - bounds[1]), text, fill="black", font=font)
    _save(canvas, output / "apple.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
