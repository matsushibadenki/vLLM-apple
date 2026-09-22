#!/usr/bin/env python3
"""Bounded synthetic 60-block streaming smoke for MFLUX Qwen Image."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.inspect_mflux_qwen_streaming import _deployable_tree_identity  # noqa: E402
from vllm_apple.hardware import detect_hardware  # noqa: E402
from vllm_apple.mflux_qwen_transformer_loader import QwenTransformerBlockLoader  # noqa: E402
from vllm_apple.mflux_qwen_transformer_plan import (  # noqa: E402
    inspect_mflux_qwen_transformer_staging,
)
from vllm_apple.qualification import save_qualification_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=60)
    parser.add_argument("--with-rope", action="store_true")
    arguments = parser.parse_args()
    if not 1 <= arguments.blocks <= 60:
        parser.error("--blocks must be in 1..60")
    root = arguments.model.expanduser().resolve(strict=True)
    plan = inspect_mflux_qwen_transformer_staging(root)
    before = detect_hardware()
    if before.memory.pressure.value != "normal" or (
        before.memory.available_bytes < plan.static_plus_maximum_block_bytes * 8
    ):
        raise RuntimeError("Qwen transformer streaming rejected before weight load")
    root_digest, file_count, artifact_bytes = _deployable_tree_identity(root)

    import mlx.core as mx
    import numpy as np
    from mflux.models.qwen.model.qwen_transformer.qwen_rope import QwenEmbedRopeMLX

    loader = QwenTransformerBlockLoader(root)
    image = mx.ones((1, 4, 3072), dtype=mx.bfloat16)
    text = mx.ones((1, 13, 3072), dtype=mx.bfloat16)
    mask = mx.ones((1, 13), dtype=mx.int32)
    timestep_embedding = mx.ones((1, 3072), dtype=mx.bfloat16)
    rotary_embeddings = None
    if arguments.with_rope:
        rotary_embeddings = QwenEmbedRopeMLX(
            theta=10000, axes_dim=[16, 56, 56], scale_rope=True
        )(video_fhw=[(1, 2, 2)], txt_seq_lens=[13])
        if (
            rotary_embeddings[0][0].shape != (4, 64)
            or rotary_embeddings[1][0].shape != (13, 64)
        ):
            raise ValueError("Qwen transformer RoPE shape is invalid")
    mx.eval(image, text, mask, timestep_embedding)
    initial_image_digest = hashlib.sha256(np.array(image.astype(mx.float32)).tobytes()).hexdigest()
    initial_text_digest = hashlib.sha256(np.array(text.astype(mx.float32)).tobytes()).hexdigest()
    samples: list[dict[str, object]] = []
    started = time.perf_counter_ns()
    failure: str | None = None
    for index in range(arguments.blocks):
        state = detect_hardware()
        if state.memory.pressure.value != "normal" or (
            state.memory.available_bytes < plan.static_plus_maximum_block_bytes * 4
        ):
            failure = f"block {index}: memory admission failed"
            break
        block_started = time.perf_counter_ns()
        try:
            block = loader.load(index)
            output_text, output_image = block(
                hidden_states=image,
                encoder_hidden_states=text,
                encoder_hidden_states_mask=mask,
                text_embeddings=timestep_embedding,
                image_rotary_emb=rotary_embeddings,
                block_idx=index,
            )
            mx.eval(output_text, output_image)
            finite = bool(
                mx.all(mx.isfinite(output_text)).item()
                and mx.all(mx.isfinite(output_image)).item()
            )
            shape_ok = output_text.shape == (1, 13, 3072) and output_image.shape == (1, 4, 3072)
            if not finite or not shape_ok:
                raise ValueError("non-finite output or unexpected shape")
            image, text = output_image, output_text
            del output_image, output_text, block
            gc.collect()
            mx.clear_cache()
            after = detect_hardware()
            sample = {
                "block_index": index,
                "elapsed_nanoseconds": max(1, time.perf_counter_ns() - block_started),
                "finite": finite,
                "image_shape": list(image.shape),
                "text_shape": list(text.shape),
                "memory_pressure": after.memory.pressure.value,
                "thermal_state": after.thermal_state.value,
                "peak_mlx_bytes": mx.get_peak_memory(),
                "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1024 if platform.system() == "Linux" else 1),
            }
            samples.append(sample)
            if after.memory.pressure.value != "normal" or after.thermal_state.value not in {
                "nominal", "fair",
            }:
                failure = f"block {index}: memory or thermal state degraded"
                break
        except Exception as error:
            failure = f"block {index}: {type(error).__name__}: {error}"
            break
    final_image_digest = hashlib.sha256(np.array(image.astype(mx.float32)).tobytes()).hexdigest()
    final_text_digest = hashlib.sha256(np.array(text.astype(mx.float32)).tobytes()).hexdigest()
    passed = failure is None and len(samples) == arguments.blocks
    report = {
        "schema_version": 1,
        "scope": "mflux_qwen_image_transformer_synthetic_sequential_block_smoke",
        "candidate_id": "qwen-image-2512",
        "artifact_root_sha256": root_digest,
        "artifact_file_count": file_count,
        "artifact_bytes": artifact_bytes,
        "requested_blocks": arguments.blocks,
        "completed_blocks": len(samples),
        "input_image_shape": [1, 4, 3072],
        "input_text_shape": [1, 13, 3072],
        "initial_image_sha256": initial_image_digest,
        "initial_text_sha256": initial_text_digest,
        "final_image_sha256": final_image_digest,
        "final_text_sha256": final_text_digest,
        "samples": samples,
        "failure": failure,
        "elapsed_nanoseconds": max(1, time.perf_counter_ns() - started),
        "peak_mlx_bytes": mx.get_peak_memory(),
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1024 if platform.system() == "Linux" else 1),
        "mlx_version": importlib.metadata.version("mlx"),
        "mflux_version": importlib.metadata.version("mlx-gen"),
        "uses_real_prompt": False,
        "uses_image_rotary_embedding": arguments.with_rope,
        "image_grid": [1, 2, 2] if arguments.with_rope else None,
        "full_transformer_qualified": False,
        "image_generation_qualified": False,
        "stores_prompt": False,
        "stores_output": False,
        "passed": passed,
    }
    save_qualification_report(report, arguments.report)
    print(json.dumps({
        "passed": passed,
        "completed_blocks": len(samples),
        "failure": failure,
        "peak_mlx_bytes": report["peak_mlx_bytes"],
        "peak_process_rss_bytes": report["peak_process_rss_bytes"],
    }, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
