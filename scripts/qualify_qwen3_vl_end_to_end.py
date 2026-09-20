#!/usr/bin/env python3
"""Run bounded processor -> Core ML -> MLX Qwen3-VL chat requests."""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


_PROMPTS = (
    ("en", "What is the dominant color? Answer with one English color word."),
    ("ja", "最も目立つ色は何色ですか。日本語の色名を一語で答えてください。"),
    ("zh-Hans", "最显眼的颜色是什么？请只用一个中文颜色词回答。"),
)
_LABELS = {
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


def _load_output(root: Path, np, mx):
    def read(name: str):
        path = root / f"{name}.fp16"
        if path.is_symlink() or path.stat().st_size != 64 * 2048 * 2:
            raise ValueError("Core ML chat smoke output is invalid")
        return mx.array(np.fromfile(path, dtype="<f2").reshape(64, 2048))

    return read("final"), [read(f"deepstack_{index}") for index in range(3)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--label", choices=tuple(_LABELS), action="append", required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--segment", type=Path, action="append", required=True)
    parser.add_argument("--max-tokens", type=int, default=12)
    parser.add_argument("--inject-failure-after-request", type=int)
    arguments = parser.parse_args()
    if (
        len(arguments.segment) != 4
        or not 1 <= arguments.max_tokens <= 32
        or len(arguments.image) != len(arguments.label)
        or not 1 <= len(arguments.image) <= 8
        or (
            arguments.inject_failure_after_request is not None
            and not 0
            <= arguments.inject_failure_after_request
            < len(arguments.image)
        )
    ):
        raise ValueError("chat smoke requires four segments and 1-32 tokens")

    import mlx.core as mx
    import numpy as np
    from PIL import Image
    from mlx_vlm import apply_chat_template, generate
    from mlx_vlm.utils import load

    from vllm_apple.qwen3_vl_pipeline_coreml import (
        qualify_qwen3_vl_segment_pipeline_coreml,
    )

    model, processor = load(str(arguments.model), lazy=True)
    original = model.vision_tower
    temporary = Path(tempfile.mkdtemp(prefix="vllm-apple-qwen3-vl-chat-"))
    temporary.chmod(0o700)
    output_root = temporary / "transport"
    output_root.mkdir(mode=0o700)
    requests = []
    for index, (image_argument, label) in enumerate(
        zip(arguments.image, arguments.label, strict=True)
    ):
        image_path = image_argument.expanduser().resolve(strict=True)
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            processor_started = time.perf_counter_ns()
            processed = processor.image_processor(images=[image])
            processor_latency = time.perf_counter_ns() - processor_started
        pixels = np.asarray(processed["pixel_values"]).astype(np.float16)
        grid = np.asarray(processed["image_grid_thw"]).reshape(-1).tolist()
        if pixels.size != 256 * 1536 or grid != [1, 16, 16]:
            raise ValueError("chat smoke processor output does not match fixed profile")
        pixel_path = temporary / f"pixel_values_{index}.fp16"
        pixels.reshape(256, 1536).tofile(pixel_path)
        pixel_path.chmod(0o600)
        requests.append(
            {
                "index": index,
                "image_path": image_path,
                "label": label,
                "pixels": pixels,
                "pixel_path": pixel_path,
                "grid": grid,
                "processor_latency_nanoseconds": processor_latency,
            }
        )
    report = None
    try:
        coreml = qualify_qwen3_vl_segment_pipeline_coreml(
            tuple(arguments.segment),
            patch_package_root=arguments.patch,
            pixel_values_files=tuple(request["pixel_path"] for request in requests),
            _transport_directory=output_root,
        )
        target_dtype = original.patch_embed.proj.weight.dtype
        mlx_grid = mx.array([[1, 16, 16]], dtype=mx.int64)
        request_reports = []
        previous_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        try:
            for request in requests:
                mlx_pixels = mx.array(request["pixels"].reshape(256, 1536)).astype(
                    target_dtype
                )
                baseline = original(mlx_pixels, mlx_grid)
                request_root = (
                    output_root
                    if len(requests) == 1
                    else output_root / f"request_{request['index']}"
                )
                candidate = _load_output(request_root, np, mx)
                mx.eval(baseline[0], *baseline[1], candidate[0], *candidate[1])
                cases = []
                for language, prompt_text in _PROMPTS:
                    prompt = apply_chat_template(
                        processor, model.config, prompt_text, num_images=1
                    )
                    outputs = {}
                    for name, result in (
                        ("baseline", baseline),
                        ("candidate", candidate),
                    ):
                        model.vision_tower = _FixedVisionTower(original, result)
                        started = time.perf_counter_ns()
                        response = generate(
                            model,
                            processor,
                            prompt,
                            image=[str(request["image_path"])],
                            max_tokens=arguments.max_tokens,
                            temperature=0,
                            verbose=False,
                        )
                        elapsed = time.perf_counter_ns() - started
                        encoded = response.text.encode("utf-8")
                        normalized = response.text.casefold().strip()
                        outputs[name] = {
                            "sha256": hashlib.sha256(encoded).hexdigest(),
                            "generation_tokens": response.generation_tokens,
                            "latency_nanoseconds": elapsed,
                            "task_correct": any(
                                expected.casefold() in normalized
                                for expected in _LABELS[request["label"]][language]
                            ),
                        }
                    cases.append(
                        {
                            "language": language,
                            "baseline": outputs["baseline"],
                            "candidate": outputs["candidate"],
                            "exact_match": outputs["baseline"]["sha256"]
                            == outputs["candidate"]["sha256"],
                        }
                    )
                current_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                request_reports.append(
                    {
                        "index": request["index"],
                        "label": request["label"],
                        "grid_thw": request["grid"],
                        "processor_latency_nanoseconds": request[
                            "processor_latency_nanoseconds"
                        ],
                        "peak_rss_bytes": current_peak_rss,
                        "peak_rss_increment_bytes": max(
                            0, current_peak_rss - previous_peak_rss
                        ),
                        "cases": cases,
                    }
                )
                previous_peak_rss = current_peak_rss
                if arguments.inject_failure_after_request == request["index"]:
                    raise RuntimeError("injected chat smoke cleanup failure")
        finally:
            model.vision_tower = original

        report = {
            "schema_version": 2,
            "scope": "qwen3_vl_processor_coreml_mlx_chat_end_to_end",
            "stores_generated_text": False,
            "request_count": len(requests),
            "coreml_model_load_count": coreml["model_load_count"],
            "coreml_artifacts_reused": coreml["model_load_count"] == 5,
            "isolated_transport_directory_count": len(requests),
            "coreml_segment_latency_samples_nanoseconds": coreml[
                "inference_latency_samples_nanoseconds"
            ],
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "requests": request_reports,
            "passed": all(
                case["baseline"]["task_correct"]
                and case["candidate"]["task_correct"]
                for request in request_reports
                for case in request["cases"]
            ),
        }
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    if report is None:
        raise RuntimeError("chat smoke did not produce a report")
    report["temporary_cleanup_verified"] = not temporary.exists()
    report["passed"] = report["passed"] and report["temporary_cleanup_verified"]
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
