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
from typing import Any, Sequence

from .benchmark_backend_memory import observe_backend_memory
from .benchmark_context import observe_benchmark_context
from .phase_probe import PhaseProbeConfig, PhaseProbeError, measure_stream
from .phase_profile import LATENCY_BUCKETS_NS, ExecutionPhaseProfiler, _BoundedLatency
from .soak import _read_private_token

# Identical tasks across languages, with an explicit quality gate for goodput.
CASES = (
    ("en", "What is 1+1? Reply with only the digit.", "2"),
    ("ja", "1+1は？数字だけで答えてください。", "2"),
    ("zh", "1+1等于几？只回答数字。", "2"),
)
BenchmarkCase = tuple[str, str, str]


def _latency_distribution(histogram: _BoundedLatency, attempts: int) -> dict[str, Any]:
    """Disjoint, inclusive-upper-bound buckets; the last bucket is overflow."""
    return {
        "sample_count": histogram.count,
        "unavailable_attempt_count": attempts - histogram.count,
        "bucket_upper_bounds_ms": [value / 1_000_000 for value in LATENCY_BUCKETS_NS] + [None],
        "bucket_counts": list(histogram.counts),
        "bucket_policy": "disjoint_inclusive_upper_bound_last_is_overflow",
        "mean_ms": histogram.snapshot()["mean_ms"] if histogram.count else None,
        "max_ms": histogram.snapshot()["max_ms"] if histogram.count else None,
    }


def _validated_cases(cases: Sequence[BenchmarkCase]) -> tuple[BenchmarkCase, ...]:
    if not 1 <= len(cases) <= 64:
        raise ValueError("benchmark requires between 1 and 64 cases")
    validated: list[BenchmarkCase] = []
    labels: set[str] = set()
    for case in cases:
        if not isinstance(case, (tuple, list)) or len(case) != 3:
            raise ValueError("each benchmark case must contain label, prompt and expected text")
        label, prompt, expected = case
        if not all(isinstance(value, str) and value for value in case):
            raise ValueError("benchmark case values must be non-empty strings")
        if len(label.encode("utf-8")) > 64 or label in labels:
            raise ValueError("benchmark case labels must be unique and no larger than 64 bytes")
        if len(prompt.encode("utf-8")) > 8 * 1024 * 1024:
            raise ValueError("benchmark prompts must not exceed 8 MiB")
        if len(expected.encode("utf-8")) > 1024:
            raise ValueError("expected text must not exceed 1 KiB")
        labels.add(label)
        validated.append((label, prompt, expected))
    return tuple(validated)


def run_text_benchmark(
    config: PhaseProbeConfig, *, requests: int = 30, concurrency: int = 1,
    ttft_slo_ms: float = 1000, e2e_slo_ms: float = 5000,
    cases: Sequence[BenchmarkCase] = CASES,
    artifact_identity_sha256: str | None = None,
    backend_build_sha256: str | None = None,
    warmup_requests: int = 0,
    collect_operating_context: bool = False,
    collect_backend_memory: bool = False,
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
    if type(warmup_requests) is not int or not 0 <= warmup_requests <= 100:
        raise ValueError("warmup requests must be an integer between 0 and 100")
    for value in (ttft_slo_ms, e2e_slo_ms):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("SLO limits must be finite and positive")
    if not math.isfinite(config.timeout_seconds) or config.timeout_seconds <= 0:
        raise ValueError("timeout must be finite and positive")
    for name, digest in (
        ("artifact identity", artifact_identity_sha256),
        ("backend build", backend_build_sha256),
    ):
        if digest is not None and (
            len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    if (artifact_identity_sha256 is None) != (backend_build_sha256 is None):
        raise ValueError("artifact identity and backend build must be provided together")
    selected_cases = _validated_cases(cases)
    context = {"before_warmup": observe_benchmark_context(config.target_pid)} \
        if collect_operating_context else None
    warmup = {"attempted": warmup_requests, "completed": 0, "quality_passed": 0,
              "errors": {}}
    for index in range(warmup_requests):
        _, prompt, expected = selected_cases[index % len(selected_cases)]
        try:
            result = measure_stream(
                replace(config, prompt=prompt), expected_text=expected,
                expected_match_mode="trimmed_exact",
            )
        except PhaseProbeError as error:
            errors = warmup["errors"]
            assert isinstance(errors, dict)
            errors[error.code] = errors.get(error.code, 0) + 1
            continue
        warmup["completed"] += 1
        warmup["quality_passed"] += int(result.expected_text_matched is True)
    if context is not None:
        context["before_measurement"] = observe_benchmark_context(config.target_pid)
    backend_memory = {"before_measurement": observe_backend_memory(config)} \
        if collect_backend_memory else None
    profiler = ExecutionPhaseProfiler(config.hardware_fingerprint, config.model, config.backend)
    lock = threading.Lock()
    next_index = 0
    errors: dict[str, int] = {}
    successes = quality_passes = slo_passes = output_tokens = good_tokens = 0
    slices = {language: {"attempted": 0, "completed": 0, "quality_passed": 0,
                         "slo_passed": 0} for language, _, _ in selected_cases}
    latencies = _BoundedLatency()
    ttft_latencies = _BoundedLatency()
    cache_samples = cached_tokens = cache_prompt_tokens = cache_hit_requests = 0
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic_ns()

    def worker() -> None:
        nonlocal next_index, successes, quality_passes, slo_passes, output_tokens, good_tokens
        nonlocal cache_samples, cached_tokens, cache_prompt_tokens, cache_hit_requests
        while True:
            with lock:
                if next_index >= requests:
                    return
                index = next_index
                next_index += 1
                language, prompt, expected = selected_cases[index % len(selected_cases)]
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
                ttft_latencies.record(measurement.ttft_ns)
                if result.cached_prompt_tokens is not None:
                    cache_samples += 1
                    cached_tokens += result.cached_prompt_tokens
                    cache_prompt_tokens += measurement.prompt_tokens
                    cache_hit_requests += int(result.cached_prompt_tokens > 0)
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
    if backend_memory is not None:
        backend_memory["after_measurement"] = observe_backend_memory(config)
    if context is not None:
        context["after_measurement"] = observe_benchmark_context(config.target_pid)
    workload = {"cases": selected_cases, "requests": requests, "concurrency": concurrency,
                "warmup_requests": warmup_requests,
                "maximum_output_tokens": config.maximum_output_tokens, "temperature": 0,
                "timeout_seconds": config.timeout_seconds,
                "ttft_slo_ms": ttft_slo_ms, "e2e_slo_ms": e2e_slo_ms}
    return {
        "schema_version": 1, "report_kind": "text_http_benchmark",
        "route": config.backend,
        "started_at": started_at,
        "cache_policy": (
            "backend_managed_conditioned" if warmup_requests
            else "backend_managed_uncontrolled"
        ),
        "warmup_requests": warmup_requests,
        "warmup": warmup,
        "operating_context": context,
        "backend_memory_observations": backend_memory,
        "artifact_identity_verified": artifact_identity_sha256 is not None,
        "artifact_identity_sha256": artifact_identity_sha256,
        "backend_build_sha256": backend_build_sha256,
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
        "prompt_cache_usage": {
            "source": "response_usage_prompt_tokens_details_cached_tokens",
            "observed_requests": cache_samples,
            "unavailable_attempts": requests - cache_samples,
            "cached_prompt_tokens": cached_tokens if cache_samples else None,
            "observed_prompt_tokens": cache_prompt_tokens if cache_samples else None,
            "requests_with_reuse": cache_hit_requests if cache_samples else None,
            "observed_token_reuse_ratio": cached_tokens / cache_prompt_tokens
            if cache_prompt_tokens else None,
            "evictions": None, "includes_warmup": False,
        },
        "e2e_p99_upper_bound_ms": latencies.percentile_ms(.99) if latencies.count else None,
        "e2e_p99_reference_only": latencies.count < 1000,
        "latency_distributions": {
            "ttft": _latency_distribution(ttft_latencies, requests),
            "e2e": _latency_distribution(latencies, requests),
            "population": "completed_responses_including_quality_and_slo_failures",
            "includes_warmup": False,
        },
        "phase_profile": profiler.snapshot(),
        "unavailable": ["backend_token_timestamps", "queue_time", "tokenize_time",
                        "model_load_time", "energy_per_token", "gpu_utilization",
                        "mlx_allocator", "swap_delta", "memory_pressure", "cache_hit_rate",
                        "continuous_thermal_observation"]
                       + (["operating_context"] if context is None else [])
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
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--collect-operating-context", action="store_true")
    parser.add_argument("--collect-backend-memory", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--ttft-slo-ms", type=float, default=1000)
    parser.add_argument("--e2e-slo-ms", type=float, default=5000)
    parser.add_argument("--target-pid", type=int)
    parser.add_argument("--session-token-file", type=Path)
    parser.add_argument("--artifact-identity-sha256")
    parser.add_argument("--backend-build-sha256")
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
                                    ttft_slo_ms=args.ttft_slo_ms, e2e_slo_ms=args.e2e_slo_ms,
                                    artifact_identity_sha256=args.artifact_identity_sha256,
                                    backend_build_sha256=args.backend_build_sha256,
                                    warmup_requests=args.warmup_requests,
                                    collect_operating_context=args.collect_operating_context,
                                    collect_backend_memory=args.collect_backend_memory)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["quality_passed"] == result["requests"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
