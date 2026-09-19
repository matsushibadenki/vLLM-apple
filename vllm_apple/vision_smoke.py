"""Paired image-input smoke test; no model qualification is implied."""

import struct
import zlib
from dataclasses import replace

from .phase_probe import PhaseProbeConfig, measure_stream


def solid_png(rgb: tuple[int, int, int]) -> bytes:
    """A deterministic 32x32 RGB fixture, generated without external assets."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data)))

    row = b"\x00" + bytes(rgb) * 32
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 32, 32, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(row * 32))
            + chunk(b"IEND", b""))


def run_vision_smoke(config: PhaseProbeConfig, *, measure=measure_stream) -> dict:
    prompts = (
        ("english", "Name the color filling the image. Reply only red or blue.", ("red", "blue")),
        ("japanese", "画像全体の色を答えてください。赤または青だけを出力してください。", ("赤", "青")),
        ("simplified_chinese", "图片整体是什么颜色？只回答红色或蓝色。", ("红色", "蓝色")),
    )
    checks = {}
    for language, prompt, expected in prompts:
        for color, rgb, answer in zip(("red", "blue"), ((255, 0, 0), (0, 0, 255)), expected):
            result = measure(
                replace(config, prompt=prompt, samples=1, maximum_output_tokens=16),
                expected_text=answer, expected_match_mode="trimmed_exact",
                image_png=solid_png(rgb),
            )
            checks[f"{language}_{color}"] = result.expected_text_matched is True
    return {
        "schema_version": 1, "probe": "paired_solid_color_png_v1",
        "model": config.model, "backend": config.backend,
        "match_policy": "trimmed_exact", "checks": checks,
        "sample_count": len(checks), "stores_generated_text": False,
        "stores_images": False, "passed": all(checks.values()),
        "qualification": False,
    }
