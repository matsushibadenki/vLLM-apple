#!/usr/bin/env python3
"""Compare a streamed Qwen text layer with the standard MLX reader."""

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
from vllm_apple.mflux_qwen_layer_loader import load_mflux_qwen_text_layer  # noqa: E402
from vllm_apple.mflux_qwen_streaming_plan import (  # noqa: E402
    inspect_mflux_qwen_text_encoder_staging,
)
from vllm_apple.qualification import save_qualification_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--layer", type=int, choices=range(28), default=0)
    arguments = parser.parse_args()
    root = arguments.model.expanduser().resolve(strict=True)
    plan = inspect_mflux_qwen_text_encoder_staging(root)
    before = detect_hardware()
    if (
        before.memory.pressure.value != "normal"
        or before.memory.available_bytes < plan.maximum_layer_payload_bytes * 6
    ):
        raise RuntimeError("Qwen text layer parity rejected before weight load")
    root_digest, file_count, artifact_bytes = _deployable_tree_identity(root)

    import mlx.core as mx
    import numpy as np
    from mflux.models.qwen.model.qwen_text_encoder.qwen_encoder_layer import (
        QwenEncoderLayer,
    )
    from mflux.models.qwen.model.qwen_text_encoder.qwen_rope import QwenRotaryEmbedding
    from mlx.utils import tree_unflatten

    layer_index = arguments.layer
    prefix = f"encoder.layers.{layer_index}."
    inputs = mx.ones((1, 4, 3584), dtype=mx.bfloat16)
    attention_mask = mx.zeros((1, 1, 4, 4), dtype=mx.float32)
    positions = mx.broadcast_to(mx.arange(4, dtype=mx.int32)[None, None, :], (3, 1, 4))
    rope = QwenRotaryEmbedding(128)(inputs, positions)
    started = time.perf_counter_ns()

    streamed = load_mflux_qwen_text_layer(root, layer_index)
    streamed_output = streamed(inputs, attention_mask, rope)
    mx.eval(streamed_output)
    streamed_array = np.array(streamed_output.astype(mx.float32))
    del streamed, streamed_output
    gc.collect()
    mx.clear_cache()

    component = root / "text_encoder"
    weight_map = json.loads(
        (component / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    names = sorted(name for name in weight_map if name.startswith(prefix))
    reference_weights = []
    for shard_name in sorted({weight_map[name] for name in names}):
        shard = mx.load(str(component / shard_name))
        reference_weights.extend(
            (name.removeprefix(prefix), shard[name])
            for name in names
            if weight_map[name] == shard_name
        )
        del shard
    reference = QwenEncoderLayer()
    reference.update(tree_unflatten(reference_weights), strict=True)
    reference_output = reference(inputs, attention_mask, rope)
    mx.eval(reference_output)
    reference_array = np.array(reference_output.astype(mx.float32))

    difference = np.abs(streamed_array - reference_array)
    exact = bool(np.array_equal(streamed_array, reference_array))
    after = detect_hardware()
    passed = (
        exact
        and before.memory.pressure.value == "normal"
        and after.memory.pressure.value == "normal"
        and after.thermal_state.value in {"nominal", "fair"}
    )
    report = {
        "schema_version": 1,
        "scope": "mflux_qwen_image_streamed_text_layer_standard_reader_parity",
        "candidate_id": "qwen-image-2512",
        "artifact_root_sha256": root_digest,
        "artifact_file_count": file_count,
        "artifact_bytes": artifact_bytes,
        "layer_index": layer_index,
        "layer_payload_bytes": plan.layer_payload_bytes[layer_index],
        "comparison": "exact-offset selective reader versus mlx.core.load",
        "output_shape": list(streamed_array.shape),
        "output_exact": exact,
        "output_maximum_absolute_error": float(difference.max()),
        "output_sha256": hashlib.sha256(streamed_array.tobytes()).hexdigest(),
        "elapsed_nanoseconds": max(1, time.perf_counter_ns() - started),
        "peak_mlx_bytes": mx.get_peak_memory(),
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1024 if platform.system() == "Linux" else 1),
        "initial_memory_pressure": before.memory.pressure.value,
        "final_memory_pressure": after.memory.pressure.value,
        "final_thermal_state": after.thermal_state.value,
        "mlx_version": importlib.metadata.version("mlx"),
        "mflux_version": importlib.metadata.version("mlx-gen"),
        "stores_prompt": False,
        "stores_output": False,
        "semantic_quality_qualified": False,
        "passed": passed,
    }
    save_qualification_report(report, arguments.report)
    print(
        json.dumps(
            {
                "passed": passed,
                "layer_index": layer_index,
                "output_exact": exact,
                "output_maximum_absolute_error": report["output_maximum_absolute_error"],
                "peak_mlx_bytes": report["peak_mlx_bytes"],
                "peak_process_rss_bytes": report["peak_process_rss_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
