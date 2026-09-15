#!/usr/bin/env python3
"""Qualify representative CPU, MLX GPU, and Core ML ANE overlap on this Mac."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from vllm_apple.ane_probe import CoreMLANEModelProbeConfig
from vllm_apple.coreml_worker import CoreMLPersistentWorker
from vllm_apple.device_contention import (
    ContentionBenchmarkConfig,
    ContentionProfile,
    run_contention_benchmark,
    save_contention_profile,
)
from vllm_apple.device_resources import contention_profile_id
from vllm_apple.execution import ExecutionBackend
from vllm_apple.hardware import detect_hardware


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coreml-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=5)
    arguments = parser.parse_args()
    hardware = detect_hardware()
    if not hardware.is_apple_silicon:
        raise SystemExit("Apple Silicon is required")
    profile_id = contention_profile_id(
        hardware.soc, hardware.os_version, hardware.architecture
    )
    import mlx.core as mx

    cpu_input = bytes(range(256)) * 32_768  # 8 MiB deterministic bandwidth work.
    left = mx.ones((1536, 1536), dtype=mx.float32)
    right = mx.full((1536, 1536), 0.5, dtype=mx.float32)
    mx.eval(left, right)

    def cpu_operation() -> str:
        return hashlib.sha256(cpu_input).hexdigest()

    def mlx_operation() -> str:
        output = left @ right
        mx.eval(output)
        evidence = (float(output[0, 0].item()), float(output[-1, -1].item()))
        return hashlib.sha256(json.dumps(evidence, separators=(",", ":")).encode()).hexdigest()

    config = CoreMLANEModelProbeConfig(
        arguments.coreml_model, arguments.coreml_model,
        "0" * 64, "input", "output",
        (1.0, -2.0, 0.5, 4.0), (2.0, -4.0, 1.0, 8.0), 1_000_000_000,
    )
    worker = CoreMLPersistentWorker(config)

    def ane_operation() -> str:
        values = worker.predict(config.input_values).values
        return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()

    operations = {
        ExecutionBackend.CPU: cpu_operation,
        ExecutionBackend.NATIVE_MLX: mlx_operation,
        ExecutionBackend.COREML_DRAFT: ane_operation,
    }
    pairs = (
        (ExecutionBackend.CPU, ExecutionBackend.NATIVE_MLX),
        (ExecutionBackend.CPU, ExecutionBackend.COREML_DRAFT),
        (ExecutionBackend.NATIVE_MLX, ExecutionBackend.COREML_DRAFT),
    )
    try:
        evidence = tuple(run_contention_benchmark(
            ContentionBenchmarkConfig(profile_id, first, second, arguments.samples),
            operations[first], operations[second],
        ) for first, second in pairs)
    finally:
        worker.close()
    qualified = tuple(item for item in evidence if item.qualified)
    output_path = None
    if qualified:
        output_path = save_contention_profile(
            ContentionProfile(profile_id, qualified), arguments.output
        )
    print(json.dumps({
        "profile_id": profile_id,
        "qualified_pairs": len(qualified),
        "profile_path": None if output_path is None else str(output_path),
        "evidence": [{
            "first_backend": item.first_backend.value,
            "second_backend": item.second_backend.value,
            "sequential_latency_nanoseconds": item.sequential_latency_nanoseconds,
            "parallel_latency_nanoseconds": item.parallel_latency_nanoseconds,
            "improvement_ratio": 1 - (
                item.parallel_latency_nanoseconds / item.sequential_latency_nanoseconds
            ),
            "sample_count": item.sample_count,
            "outputs_match": item.outputs_match,
            "qualified": item.qualified,
        } for item in evidence],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
