#!/usr/bin/env python3
"""Bounded 28-layer MFLUX Qwen Image text-encoder synthetic smoke."""
from __future__ import annotations

import argparse
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
from vllm_apple.mflux_qwen_streaming_encoder import (  # noqa: E402
    build_streaming_qwen_text_encoder,
)
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
        before.memory.available_bytes < plan.static_plus_maximum_layer_bytes * 4
    ):
        raise RuntimeError("full Qwen text encoder rejected before weight load")
    root_digest, file_count, artifact_bytes = _deployable_tree_identity(root)

    import mlx.core as mx
    import numpy as np

    layers: list[dict[str, object]] = []

    def record_layer(index: int, peak_bytes: int) -> None:
        snapshot = detect_hardware()
        layers.append({
            "index": index,
            "peak_mlx_bytes": peak_bytes,
            "memory_pressure": snapshot.memory.pressure.value,
            "thermal_state": snapshot.thermal_state.value,
        })

    started = time.perf_counter_ns()
    encoder = build_streaming_qwen_text_encoder(root, on_layer=record_layer)
    input_ids = mx.ones((1, 40), dtype=mx.int32)
    attention_mask = mx.ones((1, 40), dtype=mx.int32)
    output, output_mask = encoder(input_ids, attention_mask)
    mx.eval(output, output_mask)
    elapsed = max(1, time.perf_counter_ns() - started)
    finite = bool(mx.all(mx.isfinite(output)).item())
    output_digest = hashlib.sha256(np.array(output.astype(mx.float32)).tobytes()).hexdigest()
    after = detect_hardware()
    complete = [layer["index"] for layer in layers] == list(range(28))
    all_normal = all(layer["memory_pressure"] == "normal" for layer in layers)
    thermal_safe = all(
        layer["thermal_state"] in {"nominal", "fair"} for layer in layers
    )
    passed = (
        complete and finite and all_normal and thermal_safe
        and after.memory.pressure.value == "normal"
        and after.thermal_state.value in {"nominal", "fair"}
        and output.shape == (1, 6, 3584)
        and output_mask.shape == (1, 6)
    )
    payload = {
        "schema_version": 1,
        "scope": "mflux_qwen_image_full_text_encoder_synthetic_smoke",
        "candidate_id": "qwen-image-2512",
        "artifact_root_sha256": root_digest,
        "artifact_file_count": file_count,
        "artifact_bytes": artifact_bytes,
        "layer_count": len(layers),
        "layers": layers,
        "input_shape": [1, 40],
        "output_shape": list(output.shape),
        "output_mask_shape": list(output_mask.shape),
        "output_finite": finite,
        "output_sha256": output_digest,
        "elapsed_nanoseconds": elapsed,
        "peak_mlx_bytes": mx.get_peak_memory(),
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1024 if platform.system() == "Linux" else 1),
        "initial_memory_pressure": before.memory.pressure.value,
        "final_memory_pressure": after.memory.pressure.value,
        "final_thermal_state": after.thermal_state.value,
        "mlx_version": importlib.metadata.version("mlx"),
        "mflux_version": importlib.metadata.version("mlx-gen"),
        "image_generation_qualified": False,
        "stores_prompt": False,
        "stores_output": False,
        "passed": passed,
    }
    save_qualification_report(payload, arguments.report)
    print(json.dumps({
        "passed": passed,
        "layer_count": len(layers),
        "peak_mlx_bytes": payload["peak_mlx_bytes"],
        "peak_process_rss_bytes": payload["peak_process_rss_bytes"],
        "output_sha256": output_digest,
    }, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
