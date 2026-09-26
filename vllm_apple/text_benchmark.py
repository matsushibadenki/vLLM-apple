"""Bounded, closed-loop HTTP benchmark; no model loading or backend promotion."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .phase_probe import PhaseProbeConfig, PhaseProbeError, measure_stream
from .phase_profile import ExecutionPhaseProfiler, _BoundedLatency
from .soak import _read_private_token

# Identical tasks across languages, with an explicit quality gate for goodput.
CASES = (
    ("en", "What is 1+1? Reply with only the digit.", "2"),
    ("ja", "1+1は？数字だけで答えてください。", "2"),
    ("zh", "1+1等于几？只回答数字。", "2"),
)


def run_text_benchmark(
    config: PhaseProbeConfig, *, requests: int = 30, concurrency: int = 1,
    ttft_slo_ms: float = 1000, e2e_slo_ms: float = 5000,
) -> dict[str, Any]:
    """Run exactly requests attempts, retaining failures in the denominator.

    At most concurrency futures exist. Throughput uses wall time, never the sum
    of overlapping request durations. This arithmetic smoke is not a general
    chat/coding quality certification or an open-loop saturation benchmark.
    """
    if type(requests) is not int or not 1 <= requests <= 100_000:
        raise ValueError("requests must be an integer between 1 and 100000")
    if type(concurrency) is not int or not 1 <= concurrency <= min(requests, 32):
        raise ValueError("concurrency must be between 1 and min(requests, 32)")
    for value in (ttft_slo_ms, e2e_slo_ms):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("SLO limits must be finite and positive")
    if not math.isfinite(config.timeout_seconds) or config.timeout_seconds <= 0:
        raise ValueError("timeout must be finite and positive")
    profiler = ExecutionPhaseProfiler(config.hardware_fingerprint, config.model, config.backend)
    lock = threading.Lock()
    next_index = 0
    errors: dict[str, int] = {}
    successes = quality_passes = slo_passes = output_tokens = good_tokens = 0
    slices = {language: {"attempted": 0, "completed": 0, "quality_passed": 0,
                         "slo_passed": 0} for language, _, _ in CASES}
    latencies = _BoundedLatency()
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic_ns()

    def worker() -> None:
        nonlocal next_index, successes, quality_passes, slo_passes, output_tokens, good_tokens
        while True:
            with lock:
                if next_index >= requests:
                    return
                index = next_index
                next_index += 1
                language, prompt, expected = CASES[index % len(CASES)]
                slices[language]["attempted"] += 1
            try:
                result = measure_stream(
                    replace(config, prompt=prompt), expected_text=expected,
                    expected_match_mode="trimmed_exact",
                )
            except PhaseProbeError as error:
                with lock:
                    errors[error.code] = errors.get(error.code, 0) + 1
                continue
            measurement = result.measurement
            with lock:
                profiler.record(measurement)
                successes += 1
                output_tokens += measurement.output_tokens
                slices[language]["completed"] += 1
                quality = result.expected_text_matched is True
                quality_passes += int(quality)
                slices[language]["quality_passed"] += int(quality)
                if measurement.stream_done_ns is not None:
                    e2e_ns = measurement.stream_done_ns - measurement.started_ns
                    latencies.record(e2e_ns)
                    meets_slo = (measurement.ttft_ns <= ttft_slo_ms * 1_000_000
                                 and e2e_ns <= e2e_slo_ms * 1_000_000)
                    if quality and meets_slo:
                        slo_passes += 1
                        good_tokens += measurement.output_tokens
                        slices[language]["slo_passed"] += 1

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker) for _ in range(concurrency)]
        for future in futures:
            future.result()
    elapsed = (time.monotonic_ns() - started) / 1_000_000_000
    workload = {"cases": CASES, "requests": requests, "concurrency": concurrency,
                "maximum_output_tokens": config.maximum_output_tokens, "temperature": 0,
                "timeout_seconds": config.timeout_seconds,
                "ttft_slo_ms": ttft_slo_ms, "e2e_slo_ms": e2e_slo_ms}
    return {
        "schema_version": 1, "report_kind": "text_http_benchmark",
        "started_at": started_at,
        "cache_policy": "backend_managed_uncontrolled",
        "artifact_identity_verified": False,
        "workload_sha256": hashlib.sha256(json.dumps(
            workload, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "load_policy": "closed_loop", "quality_policy": "arithmetic_trimmed_exact",
        "requests": requests, "concurrency": concurrency,
        "maximum_output_tokens": config.maximum_output_tokens,
        "timeout_seconds": config.timeout_seconds,
        "slo": {"ttft_ms": ttft_slo_ms, "e2e_ms": e2e_slo_ms},
        "completed": successes, "failed": requests - successes, "errors": errors,
        "quality_passed": quality_passes, "slo_quality_passed": slo_passes,
        "output_tokens": output_tokens, "slo_quality_output_tokens": good_tokens,
        "elapsed_seconds": elapsed,
        "output_tokens_per_second": output_tokens / elapsed if elapsed > 0 else None,
        "goodput_tokens_per_second": good_tokens / elapsed if elapsed > 0 else None,
        "languages": slices,
        "e2e_p99_upper_bound_ms": latencies.percentile_ms(.99) if latencies.count else None,
        "e2e_p99_reference_only": latencies.count < 1000,
        "phase_profile": profiler.snapshot(),
        "unavailable": ["backend_token_timestamps", "queue_time", "tokenize_time",
                        "model_load_time", "energy_per_token", "gpu_utilization",
                        "mlx_allocator", "swap_delta", "memory_pressure", "thermal"]
                       + (["rss"] if config.target_pid is None else []),
        "qualification": False, "stores_prompt": False, "stores_generated_text": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--hardware-fingerprint", required=True)
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--ttft-slo-ms", type=float, default=1000)
    parser.add_argument("--e2e-slo-ms", type=float, default=5000)
    parser.add_argument("--target-pid", type=int)
    parser.add_argument("--session-token-file", type=Path)
    args = parser.parse_args()
    try:
        config = PhaseProbeConfig(
            base_url=args.base_url, model=args.model, backend=args.backend,
            hardware_fingerprint=args.hardware_fingerprint,
            maximum_output_tokens=args.max_tokens, timeout_seconds=args.timeout,
            target_pid=args.target_pid,
            session_token=_read_private_token(args.session_token_file)
            if args.session_token_file else None,
        )
        result = run_text_benchmark(config, requests=args.requests, concurrency=args.concurrency,
                                    ttft_slo_ms=args.ttft_slo_ms, e2e_slo_ms=args.e2e_slo_ms)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["quality_passed"] == result["requests"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
