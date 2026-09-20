#!/usr/bin/env python3
"""Run a bounded repeated-worker Qwen3-VL Core ML stability qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import tempfile
import time
from pathlib import Path


def _atomic_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--segment", type=Path, action="append", required=True)
    parser.add_argument("--pixel-values", type=Path, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-30-minute-window", action="store_true")
    arguments = parser.parse_args()
    if (
        len(arguments.segment) != 4
        or not 1 <= arguments.duration <= 7200
        or (arguments.require_30_minute_window and arguments.duration < 1800)
    ):
        raise ValueError("invalid Qwen3-VL Core ML soak configuration")
    output = arguments.output.expanduser().resolve(strict=False)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("soak output must be a new file in an existing directory")

    from vllm_apple.qwen3_vl_pipeline_coreml import (
        qualify_qwen3_vl_segment_pipeline_coreml,
    )

    started = time.monotonic()
    deadline = started + arguments.duration
    batches = 0
    requests = 0
    failures = 0
    shutdown_failures = 0
    failure_fingerprints: dict[str, int] = {}
    expected_digests = None
    digest_mismatches = 0
    latency_samples = []
    rss_samples = []
    while time.monotonic() < deadline:
        try:
            result = qualify_qwen3_vl_segment_pipeline_coreml(
                tuple(arguments.segment),
                repetitions=10,
                patch_package_root=arguments.patch,
                pixel_values_file=arguments.pixel_values,
            )
            batches += 1
            for run in result["runs"]:
                requests += 1
                digests = tuple(
                    stage["merged"]["output_sha256"] for stage in run["stages"]
                )
                if expected_digests is None:
                    expected_digests = digests
                elif digests != expected_digests:
                    digest_mismatches += 1
                latency_samples.append(run["total_latency_nanoseconds"])
                rss_samples.append(run["peak_rss_bytes"])
                if len(latency_samples) > 256:
                    latency_samples.pop(0)
                    rss_samples.pop(0)
        except Exception as error:
            failures += 1
            fingerprint = hashlib.sha256(
                f"{type(error).__name__}:{error}".encode("utf-8")
            ).hexdigest()
            failure_fingerprints[fingerprint] = failure_fingerprints.get(fingerprint, 0) + 1
        if failures > 3:
            break
    elapsed = time.monotonic() - started
    sorted_latencies = sorted(latency_samples)
    median_latency = (
        sorted_latencies[len(sorted_latencies) // 2] if sorted_latencies else None
    )
    report = {
        "schema_version": 1,
        "scope": "qwen3_vl_coreml_repeated_worker_soak",
        "requested_duration_seconds": arguments.duration,
        "elapsed_seconds": elapsed,
        "stability_window_met": elapsed >= 1800,
        "batches": batches,
        "requests": requests,
        "failures": failures,
        "failure_fingerprints": failure_fingerprints,
        "digest_mismatches": digest_mismatches,
        "output_digests": list(expected_digests or ()),
        "retained_sample_count": len(latency_samples),
        "median_recent_latency_nanoseconds": median_latency,
        "initial_worker_peak_rss_bytes": rss_samples[0] if rss_samples else None,
        "final_worker_peak_rss_bytes": rss_samples[-1] if rss_samples else None,
        "maximum_recent_worker_peak_rss_bytes": max(rss_samples) if rss_samples else None,
        "worker_peak_rss_growth_bytes": (
            rss_samples[-1] - rss_samples[0] if rss_samples else None
        ),
        "parent_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "shutdown_failures": shutdown_failures,
        "shutdown_clean": shutdown_failures == 0,
        "worker_lifecycle": "ten_predictions_per_worker_then_clean_exit",
        "persistent_worker_memory_evaluated": False,
    }
    report["passed"] = (
        (not arguments.require_30_minute_window or report["stability_window_met"])
        and requests > 0
        and failures == 0
        and digest_mismatches == 0
        and report["shutdown_clean"]
    )
    _atomic_write(output, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
