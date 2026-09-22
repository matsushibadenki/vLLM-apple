#!/usr/bin/env python3
"""One-block quantized Qwen Image transformer execution smoke."""
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
from vllm_apple.mflux_qwen_transformer_loader import (  # noqa: E402
    load_mflux_qwen_transformer_block,
)
from vllm_apple.mflux_qwen_transformer_plan import (  # noqa: E402
    inspect_mflux_qwen_transformer_staging,
)
from vllm_apple.qualification import save_qualification_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.model.expanduser().resolve(strict=True)
    plan = inspect_mflux_qwen_transformer_staging(root)
    before = detect_hardware()
    if before.memory.pressure.value != "normal" or (
        before.memory.available_bytes < plan.static_plus_maximum_block_bytes * 8
    ):
        raise RuntimeError("Qwen transformer block rejected before weight load")
    root_digest, file_count, artifact_bytes = _deployable_tree_identity(root)

    import mlx.core as mx
    import numpy as np

    started = time.perf_counter_ns()
    block = load_mflux_qwen_transformer_block(root, arguments.block)
    image = mx.ones((1, 4, 3072), dtype=mx.bfloat16)
    text = mx.ones((1, 13, 3072), dtype=mx.bfloat16)
    mask = mx.ones((1, 13), dtype=mx.int32)
    timestep_embedding = mx.ones((1, 3072), dtype=mx.bfloat16)
    output_text, output_image = block(
        hidden_states=image,
        encoder_hidden_states=text,
        encoder_hidden_states_mask=mask,
        text_embeddings=timestep_embedding,
        image_rotary_emb=None,
        block_idx=arguments.block,
    )
    mx.eval(output_text, output_image)
    after = detect_hardware()
    finite = bool(
        mx.all(mx.isfinite(output_text)).item()
        and mx.all(mx.isfinite(output_image)).item()
    )
    passed = (
        finite
        and output_text.shape == (1, 13, 3072)
        and output_image.shape == (1, 4, 3072)
        and after.memory.pressure.value == "normal"
        and after.thermal_state.value in {"nominal", "fair"}
    )
    report = {
        "schema_version": 1,
        "scope": "mflux_qwen_image_transformer_single_block_synthetic_smoke",
        "candidate_id": "qwen-image-2512",
        "artifact_root_sha256": root_digest,
        "artifact_file_count": file_count,
        "artifact_bytes": artifact_bytes,
        "block_index": arguments.block,
        "input_image_shape": [1, 4, 3072],
        "input_text_shape": [1, 13, 3072],
        "output_image_shape": list(output_image.shape),
        "output_text_shape": list(output_text.shape),
        "output_finite": finite,
        "output_image_sha256": hashlib.sha256(
            np.array(output_image.astype(mx.float32)).tobytes()
        ).hexdigest(),
        "output_text_sha256": hashlib.sha256(
            np.array(output_text.astype(mx.float32)).tobytes()
        ).hexdigest(),
        "elapsed_nanoseconds": max(1, time.perf_counter_ns() - started),
        "peak_mlx_bytes": mx.get_peak_memory(),
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1024 if platform.system() == "Linux" else 1),
        "initial_memory_pressure": before.memory.pressure.value,
        "final_memory_pressure": after.memory.pressure.value,
        "final_thermal_state": after.thermal_state.value,
        "mlx_version": importlib.metadata.version("mlx"),
        "mflux_version": importlib.metadata.version("mlx-gen"),
        "full_transformer_qualified": False,
        "image_generation_qualified": False,
        "stores_prompt": False,
        "stores_output": False,
        "passed": passed,
    }
    save_qualification_report(report, arguments.report)
    print(json.dumps({
        "passed": passed,
        "output_finite": finite,
        "peak_mlx_bytes": report["peak_mlx_bytes"],
        "peak_process_rss_bytes": report["peak_process_rss_bytes"],
    }, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
