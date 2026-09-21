#!/usr/bin/env python3
"""Compare Core ML Qwen3-VL embeddings with the Homebrew MLX baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
import stat
from pathlib import Path

_DESCRIPTION_PROMPTS = (
    ("en", "Describe the image briefly."),
    ("ja", "画像を簡潔に説明してください。"),
    ("zh-Hans", "请简要描述这张图片。"),
)
_COLOR_PROMPTS = (
    ("en", "What is the dominant color? Answer with one English color word."),
    ("ja", "最も目立つ色は何色ですか。日本語の色名を一語で答えてください。"),
    ("zh-Hans", "最显眼的颜色是什么？请只用一个中文颜色词回答。"),
)
_COLOR_LABELS = {
    "red": {"en": ("red",), "ja": ("赤",), "zh-Hans": ("红",)},
    "green": {"en": ("green",), "ja": ("緑",), "zh-Hans": ("绿",)},
    "blue": {"en": ("blue",), "ja": ("青", "ブルー"), "zh-Hans": ("蓝",)},
}


class _FixedVisionTower:
    def __init__(self, original, result):
        self.patch_embed = original.patch_embed
        self._result = result

    def __call__(self, *_args, **_kwargs):
        return self._result


def _array(path: Path, shape, np, mx):
    expected_bytes = 1
    for dimension in shape:
        expected_bytes *= dimension
    expected_bytes *= 2
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != expected_bytes
    ):
        raise ValueError(f"invalid bounded FP16 input: {path.name}")
    return mx.array(np.fromfile(path, dtype="<f2").reshape(shape))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pixel-values", type=Path, required=True)
    parser.add_argument("--transport", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--task-label", choices=tuple(_COLOR_LABELS))
    arguments = parser.parse_args()
    if not 1 <= arguments.max_tokens <= 64:
        raise ValueError("max tokens must be between 1 and 64")
    transport = arguments.transport.expanduser().resolve(strict=True)
    if stat.S_IMODE(transport.stat().st_mode) & 0o077:
        raise ValueError("transport directory must be private")

    import mlx.core as mx
    import numpy as np
    from mlx_vlm import apply_chat_template, generate
    from mlx_vlm.utils import load

    model, processor = load(str(arguments.model), lazy=True)
    original = model.vision_tower
    pixels = _array(
        arguments.pixel_values.expanduser().resolve(strict=True),
        (256, 1536),
        np,
        mx,
    ).astype(original.patch_embed.proj.weight.dtype)
    grid = mx.array([[1, 16, 16]], dtype=mx.int64)
    baseline = original(pixels, grid)
    candidate = (
        _array(transport / "final.fp16", (64, 2048), np, mx),
        [
            _array(transport / f"deepstack_{index}.fp16", (64, 2048), np, mx)
            for index in range(3)
        ],
    )
    mx.eval(baseline[0], *baseline[1], candidate[0], *candidate[1])

    image = arguments.image.expanduser().resolve(strict=True)
    results = []
    try:
        prompts = _COLOR_PROMPTS if arguments.task_label else _DESCRIPTION_PROMPTS
        for language, prompt_text in prompts:
            prompt = apply_chat_template(
                processor, model.config, prompt_text, num_images=1
            )
            outputs = {}
            for name, vision_result in (
                ("baseline", baseline),
                ("candidate", candidate),
                ("baseline_replay", baseline),
            ):
                model.vision_tower = _FixedVisionTower(original, vision_result)
                response = generate(
                    model,
                    processor,
                    prompt,
                    image=[str(image)],
                    max_tokens=arguments.max_tokens,
                    temperature=0,
                    verbose=False,
                )
                encoded = response.text.encode("utf-8")
                outputs[name] = {
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                    "utf8_bytes": len(encoded),
                    "generation_tokens": response.generation_tokens,
                }
                if arguments.task_label:
                    normalized = response.text.casefold().strip()
                    outputs[name]["task_correct"] = any(
                        label.casefold() in normalized
                        for label in _COLOR_LABELS[arguments.task_label][language]
                    )
            results.append(
                {
                    "language": language,
                    "baseline_stable": (
                        outputs["baseline"] == outputs["baseline_replay"]
                    ),
                    "exact_match": outputs["baseline"] == outputs["candidate"],
                    "baseline": outputs["baseline"],
                    "candidate": outputs["candidate"],
                    "baseline_replay": outputs["baseline_replay"],
                    "task_correct": (
                        outputs["baseline"].get("task_correct") is True
                        and outputs["candidate"].get("task_correct") is True
                    ) if arguments.task_label else None,
                }
            )
    finally:
        model.vision_tower = original

    report = {
        "schema_version": 1,
        "scope": "qwen3_vl_coreml_vs_homebrew_mlx_bf16_greedy_quality",
        "stores_generated_text": False,
        "max_tokens": arguments.max_tokens,
        "task": "dominant_color" if arguments.task_label else "description",
        "task_label": arguments.task_label,
        "cases": results,
        "passed": all(
            case["baseline_stable"]
            and (
                case["task_correct"]
                if arguments.task_label
                else case["exact_match"]
            )
            for case in results
        ),
    }
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
