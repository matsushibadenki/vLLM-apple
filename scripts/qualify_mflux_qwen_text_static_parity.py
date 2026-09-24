#!/usr/bin/env python3
"""Compare streamed Qwen text static weights with the standard MLX reader."""

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
    if (
        before.memory.pressure.value != "normal"
        or before.memory.available_bytes < plan.static_payload_bytes * 3
    ):
        raise RuntimeError("Qwen text static parity rejected before weight load")
    root_digest, file_count, artifact_bytes = _deployable_tree_identity(root)

    import mlx.core as mx
    import numpy as np

    started = time.perf_counter_ns()
    encoder = build_streaming_qwen_text_encoder(root, layer_limit=1).encoder
    streamed = {
        "encoder.embed_tokens.weight": encoder.embed_tokens.weight,
        "encoder.norm.weight": encoder.norm.weight,
        "encoder.rotary_emb.inv_freq": encoder.rotary_emb.inv_freq,
    }
    component = root / "text_encoder"
    weight_map = json.loads(
        (component / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    comparisons = []
    for name, streamed_tensor in streamed.items():
        shard = mx.load(str(component / weight_map[name]))
        reference_tensor = shard[name]
        mx.eval(streamed_tensor, reference_tensor)
        # NumPy has no native bfloat16 buffer format.  Expanding both sides to
        # float32 preserves every BF16 value and makes the equality portable.
        streamed_array = np.array(streamed_tensor.astype(mx.float32))
        reference_array = np.array(reference_tensor.astype(mx.float32))
        exact = bool(np.array_equal(streamed_array, reference_array))
        comparisons.append(
            {
                "name": name,
                "shape": list(streamed_array.shape),
                "dtype": str(streamed_tensor.dtype),
                "exact": exact,
                "sha256": hashlib.sha256(streamed_array.tobytes()).hexdigest(),
            }
        )
        del shard, reference_tensor, streamed_array, reference_array
        gc.collect()
        mx.clear_cache()
    after = detect_hardware()
    passed = (
        all(item["exact"] for item in comparisons)
        and before.memory.pressure.value == "normal"
        and after.memory.pressure.value == "normal"
        and after.thermal_state.value in {"nominal", "fair"}
    )
    report = {
        "schema_version": 1,
        "scope": "mflux_qwen_image_streamed_text_static_standard_reader_parity",
        "candidate_id": "qwen-image-2512",
        "artifact_root_sha256": root_digest,
        "artifact_file_count": file_count,
        "artifact_bytes": artifact_bytes,
        "static_payload_bytes": plan.static_payload_bytes,
        "comparison": "exact-offset selective reader versus mlx.core.load",
        "tensors": comparisons,
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
                "tensor_count": len(comparisons),
                "all_exact": all(item["exact"] for item in comparisons),
                "peak_mlx_bytes": report["peak_mlx_bytes"],
                "peak_process_rss_bytes": report["peak_process_rss_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
