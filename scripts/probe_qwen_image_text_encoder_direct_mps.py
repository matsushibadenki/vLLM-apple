#!/usr/bin/env python3
"""Probe isolated CPU or direct-MPS Qwen-Image 2.1 text encoding."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path

MAX_PROMPT_BYTES = 4096


def _telemetry(torch: object) -> dict[str, int]:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = int(usage if platform.system() == "Darwin" else usage * 1024)
    mps = getattr(torch, "mps")
    return {
        "peak_rss_bytes": rss,
        "mps_current_allocated_bytes": int(mps.current_allocated_memory()),
        "mps_driver_allocated_bytes": int(mps.driver_allocated_memory()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--prompt", default="A small friendly robot in a clean workshop")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--condition-resolution", type=int, default=256)
    parser.add_argument(
        "--device", choices=("cpu", "mps", "sequential-mps"), default="mps"
    )
    arguments = parser.parse_args()
    if not arguments.prompt.strip() or len(arguments.prompt.encode()) > MAX_PROMPT_BYTES:
        raise ValueError("probe prompt is invalid")
    if not 64 <= arguments.condition_resolution <= 1024:
        raise ValueError("condition resolution is invalid")
    root = arguments.model.expanduser().resolve(strict=True)
    component = root / "text_encoder"
    processor_root = root / "processor"
    if not component.is_dir() or not processor_root.is_dir():
        raise ValueError("Qwen-Image text encoder artifact is incomplete")

    import torch
    from diffusers import QwenImage21Pipeline
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    started = time.monotonic_ns()
    processor = Qwen3VLProcessor.from_pretrained(
        str(processor_root), local_files_only=True
    )
    load_device = "cpu" if arguments.device == "sequential-mps" else arguments.device
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        str(component),
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": load_device},
    )
    loaded = _telemetry(torch)
    pipeline = QwenImage21Pipeline(
        scheduler=None,
        vae=None,
        text_encoder=encoder,
        processor=processor,
        transformer=None,
    )
    condition_images = None
    if arguments.image is not None:
        from PIL import Image

        image_path = arguments.image.expanduser().resolve(strict=True)
        with Image.open(image_path) as source:
            source.load()
            condition = source.convert("RGB")
        ratio = condition.width / condition.height
        width = max(
            32,
            round(math.sqrt(arguments.condition_resolution**2 * ratio) / 32) * 32,
        )
        height = max(32, round(width / ratio / 32) * 32)
        condition_images = [
            pipeline.image_processor.resize(condition, width=width, height=height)
        ]
    execution_device = arguments.device
    if arguments.device == "sequential-mps":
        enable = getattr(pipeline, "enable_sequential_cpu_offload", None)
        if not callable(enable):
            raise RuntimeError("Qwen-Image pipeline lacks sequential CPU offload")
        enable(device="mps")
        execution_device = getattr(pipeline, "_execution_device", "mps")
    encode_prompt = getattr(pipeline, "encode_prompt", None)
    if not callable(encode_prompt):
        raise RuntimeError("Qwen-Image pipeline lacks encode_prompt")
    prompt_embeds, prompt_mask, _ = encode_prompt(
        prompt=arguments.prompt,
        image=condition_images,
        device=execution_device,
        num_images_per_prompt=1,
    )
    torch.mps.synchronize()
    encoded = _telemetry(torch)
    if not bool(torch.isfinite(prompt_embeds).all().item()) or (
        prompt_mask is not None
        and not bool(torch.isfinite(prompt_mask).all().item())
    ):
        raise RuntimeError("Qwen-Image prompt encoding returned non-finite values")
    embedding_shape = list(prompt_embeds.shape)
    mask_shape = list(prompt_mask.shape) if prompt_mask is not None else None
    del pipeline, encoder, prompt_embeds, prompt_mask
    gc.collect()
    torch.mps.empty_cache()
    torch.mps.synchronize()
    cleaned = _telemetry(torch)
    payload = {
        "schema_version": 1,
        "scope": "qwen_image_text_encoder_direct_mps_probe",
        "strategy": f"safetensors_isolated_{arguments.device}",
        "embedding_shape": embedding_shape,
        "mask_shape": mask_shape,
        "condition_image_shape": (
            [condition_images[0].height, condition_images[0].width]
            if condition_images is not None
            else None
        ),
        "load": loaded,
        "encode": encoded,
        "cleanup": cleaned,
        "wall_time_nanoseconds": time.monotonic_ns() - started,
        "stores_prompt": False,
        "stores_embedding": False,
        "passed": cleaned["mps_current_allocated_bytes"] <= 1024 * 1024,
    }
    encoded_json = json.dumps(payload, sort_keys=True)
    if len(encoded_json.encode()) > 64 * 1024:
        raise RuntimeError("probe output exceeds its bound")
    os.write(sys.stdout.fileno(), (encoded_json + "\n").encode())
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
