#!/usr/bin/env python3
"""Bounded Qwen-Image-2512 VAE decoder-only synthetic latent smoke."""
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
from vllm_apple.mflux_qwen_vae_loader import (  # noqa: E402
    inspect_mflux_qwen_vae_decoder,
    load_mflux_qwen_vae_decoder,
)
from vllm_apple.qualification import save_qualification_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.model.expanduser().resolve(strict=True)
    inventory = inspect_mflux_qwen_vae_decoder(root)
    before = detect_hardware()
    if before.memory.pressure.value != "normal" or (
        before.memory.available_bytes < inventory["decoder_payload_bytes"] * 8
    ):
        raise RuntimeError("Qwen VAE decoder rejected before weight load")
    root_digest, file_count, artifact_bytes = _deployable_tree_identity(root)

    import mlx.core as mx
    import numpy as np
    from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator

    started = time.perf_counter_ns()
    vae = load_mflux_qwen_vae_decoder(root)
    packed = mx.zeros((1, 4, 64), dtype=mx.bfloat16)
    unpacked = QwenLatentCreator.unpack_latents(packed, height=32, width=32)
    decoded = vae.decode(unpacked)
    mx.eval(decoded)
    finite = bool(mx.all(mx.isfinite(decoded)).item())
    after = detect_hardware()
    passed = (
        finite
        and unpacked.shape == (1, 16, 4, 4)
        and decoded.shape == (1, 3, 1, 32, 32)
        and after.memory.pressure.value == "normal"
        and after.thermal_state.value in {"nominal", "fair"}
    )
    report = {
        "schema_version": 1,
        "scope": "mflux_qwen_image_vae_decoder_only_synthetic_latent_smoke",
        "candidate_id": "qwen-image-2512",
        "artifact_root_sha256": root_digest,
        "artifact_file_count": file_count,
        "artifact_bytes": artifact_bytes,
        "inventory": inventory,
        "packed_shape": list(packed.shape),
        "unpacked_shape": list(unpacked.shape),
        "decoded_shape": list(decoded.shape),
        "decoded_finite": finite,
        "decoded_sha256": hashlib.sha256(np.array(decoded.astype(mx.float32)).tobytes()).hexdigest(),
        "elapsed_nanoseconds": max(1, time.perf_counter_ns() - started),
        "peak_mlx_bytes": mx.get_peak_memory(),
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1024 if platform.system() == "Linux" else 1),
        "initial_memory_pressure": before.memory.pressure.value,
        "final_memory_pressure": after.memory.pressure.value,
        "final_thermal_state": after.thermal_state.value,
        "mlx_version": importlib.metadata.version("mlx"),
        "mflux_version": importlib.metadata.version("mlx-gen"),
        "uses_real_latent": False,
        "image_generation_qualified": False,
        "stores_prompt": False,
        "stores_output": False,
        "passed": passed,
    }
    save_qualification_report(report, arguments.report)
    print(json.dumps({
        "passed": passed,
        "decoder_tensor_count": inventory["decoder_tensor_count"],
        "decoded_shape": list(decoded.shape),
        "peak_mlx_bytes": report["peak_mlx_bytes"],
        "peak_process_rss_bytes": report["peak_process_rss_bytes"],
    }, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
