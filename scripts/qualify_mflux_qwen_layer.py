#!/usr/bin/env python3
"""Qualify one selectively loaded Qwen Image text-encoder layer on Apple Silicon."""
from __future__ import annotations

import argparse
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
from vllm_apple.mflux_qwen_layer_loader import load_mflux_qwen_text_layer  # noqa: E402
from vllm_apple.mflux_qwen_streaming_plan import (  # noqa: E402
    inspect_mflux_qwen_text_encoder_staging,
)
from vllm_apple.qualification import save_qualification_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.model.expanduser().resolve(strict=True)
    plan = inspect_mflux_qwen_text_encoder_staging(root)
    before = detect_hardware()
    if before.memory.pressure.value != "normal" or (
        before.memory.available_bytes < plan.maximum_layer_payload_bytes * 4
    ):
        raise RuntimeError("Qwen text layer smoke rejected before weight load")
    root_digest, file_count, artifact_bytes = _deployable_tree_identity(root)

    import mlx.core as mx
    from mflux.models.qwen.model.qwen_text_encoder.qwen_rope import QwenRotaryEmbedding

    started = time.perf_counter_ns()
    layer = load_mflux_qwen_text_layer(root, 0)
    x = mx.ones((1, 4, 3584), dtype=mx.bfloat16)
    positions = mx.broadcast_to(mx.arange(4, dtype=mx.int32)[None, None, :], (3, 1, 4))
    rope = QwenRotaryEmbedding(128)(x, positions)
    output = layer(x, mx.zeros((1, 1, 4, 4), dtype=mx.float32), rope)
    mx.eval(output)
    finite = bool(mx.all(mx.isfinite(output)).item())
    source = mx.load(str(root / "text_encoder" / "0.safetensors"))
    exact_norm = bool(mx.all(
        layer.input_layernorm.weight
        == source["encoder.layers.0.input_layernorm.weight"]
    ).item())
    exact_projection = bool(mx.all(
        layer.self_attn.q_proj.weight
        == source["encoder.layers.0.self_attn.q_proj.weight"]
    ).item())
    elapsed = max(1, time.perf_counter_ns() - started)
    after = detect_hardware()
    passed = (
        finite and exact_norm and exact_projection
        and after.memory.pressure.value == "normal"
        and after.thermal_state.value in {"nominal", "fair"}
    )
    payload = {
        "schema_version": 1,
        "scope": "mflux_qwen_image_text_encoder_layer0_selective_load_smoke",
        "candidate_id": "qwen-image-2512",
        "artifact_root_sha256": root_digest,
        "artifact_file_count": file_count,
        "artifact_bytes": artifact_bytes,
        "layer_index": 0,
        "layer_payload_bytes": plan.layer_payload_bytes[0],
        "output_shape": list(output.shape),
        "output_finite": finite,
        "exact_norm_weight": exact_norm,
        "exact_q_projection_weight": exact_projection,
        "elapsed_nanoseconds": elapsed,
        "peak_mlx_bytes": mx.get_peak_memory(),
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1024 if platform.system() == "Linux" else 1),
        "initial_memory_pressure": before.memory.pressure.value,
        "final_memory_pressure": after.memory.pressure.value,
        "final_thermal_state": after.thermal_state.value,
        "mlx_version": importlib.metadata.version("mlx"),
        "mflux_version": importlib.metadata.version("mlx-gen"),
        "full_text_encoder_qualified": False,
        "image_generation_qualified": False,
        "stores_prompt": False,
        "stores_output": False,
        "passed": passed,
    }
    save_qualification_report(payload, arguments.report)
    print(json.dumps(payload, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
