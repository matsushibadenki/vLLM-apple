#!/usr/bin/env python3
"""Qualify one persistent Qwen3-VL Core ML worker over a bounded window."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import tempfile
import time
from pathlib import Path


def _atomic_write(path: Path, payload: dict[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
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
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--pixel-values", type=Path, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-30-minute-window", action="store_true")
    arguments = parser.parse_args()
    if (
        len(arguments.model) != 5
        or not 1 <= arguments.duration <= 7200
        or not 0.1 <= arguments.interval <= 10
        or (arguments.require_30_minute_window and arguments.duration < 1800)
    ):
        raise ValueError("invalid persistent Qwen3-VL soak configuration")
    output = arguments.output.expanduser().resolve(strict=False)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("persistent soak output must be new")

    from vllm_apple.qwen3_vl_persistent_worker import Qwen3VLPersistentWorker

    started = time.monotonic()
    deadline = started + arguments.duration
    expected_digests = None
    requests = 0
    failures = 0
    digest_mismatches = 0
    failure_fingerprints: dict[str, int] = {}
    latency_samples: list[int] = []
    rss_samples: list[int] = []
    shutdown_clean = False
    workspace = Path(tempfile.mkdtemp(prefix="qwen3-vl-persistent-soak-"))
    workspace.chmod(0o700)
    worker = Qwen3VLPersistentWorker(tuple(arguments.model))
    try:
        while time.monotonic() < deadline and failures <= 3:
            request_started = time.monotonic()
            request_root = workspace / f"request-{requests:08d}"
            request_root.mkdir(mode=0o700)
            try:
                result = worker.predict(arguments.pixel_values, request_root)
                requests += 1
                digests = tuple(result["output_digests"])
                if expected_digests is None:
                    expected_digests = digests
                elif digests != expected_digests:
                    digest_mismatches += 1
                latency_samples.append(result["latency_nanoseconds"])
                rss_samples.append(result["peak_rss_bytes"])
                if len(latency_samples) > 256:
                    latency_samples.pop(0)
                    rss_samples.pop(0)
            except Exception as error:
                failures += 1
                fingerprint = hashlib.sha256(
                    f"{type(error).__name__}:{error}".encode("utf-8")
                ).hexdigest()
                failure_fingerprints[fingerprint] = (
                    failure_fingerprints.get(fingerprint, 0) + 1
                )
            finally:
                shutil.rmtree(request_root, ignore_errors=True)
            remaining = arguments.interval - (time.monotonic() - request_started)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        shutdown_clean = worker.close()
        shutil.rmtree(workspace, ignore_errors=True)
    elapsed = time.monotonic() - started
    sorted_latencies = sorted(latency_samples)
    report = {
        "schema_version": 1,
        "scope": "qwen3_vl_coreml_single_persistent_worker_soak",
        "requested_duration_seconds": arguments.duration,
        "elapsed_seconds": elapsed,
        "interval_seconds": arguments.interval,
        "stability_window_met": elapsed >= 1800,
        "requests": requests,
        "failures": failures,
        "failure_fingerprints": failure_fingerprints,
        "digest_mismatches": digest_mismatches,
        "output_digests": list(expected_digests or ()),
        "retained_sample_count": len(latency_samples),
        "median_recent_latency_nanoseconds": (
            sorted_latencies[len(sorted_latencies) // 2]
            if sorted_latencies
            else None
        ),
        "initial_worker_peak_rss_bytes": rss_samples[0] if rss_samples else None,
        "final_worker_peak_rss_bytes": rss_samples[-1] if rss_samples else None,
        "maximum_recent_worker_peak_rss_bytes": max(rss_samples) if rss_samples else None,
        "worker_peak_rss_growth_bytes": (
            rss_samples[-1] - rss_samples[0] if rss_samples else None
        ),
        "restart_count": worker.restart_count,
        "parent_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "shutdown_clean": shutdown_clean,
        "temporary_cleanup_verified": not workspace.exists(),
    }
    report["passed"] = (
        (not arguments.require_30_minute_window or report["stability_window_met"])
        and requests > 0
        and failures == 0
        and digest_mismatches == 0
        and worker.restart_count == 0
        and shutdown_clean
        and report["temporary_cleanup_verified"]
    )
    _atomic_write(output, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
