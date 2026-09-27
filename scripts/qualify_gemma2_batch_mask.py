#!/usr/bin/env python3
"""Run bounded M4 qualification for the opt-in Gemma 2 batch-mask fix."""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import signal
import socket
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any

from vllm_apple.phase_probe import PhaseProbeConfig, _resident_bytes, measure_stream
from vllm_apple.text_benchmark import run_text_benchmark


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _wait_ready(base_url: str, process: subprocess.Popen[bytes], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("backend exited during startup")
        try:
            with urllib.request.urlopen(base_url + "/v1/models", timeout=1):
                return
        except OSError:
            time.sleep(0.25)
    raise RuntimeError("backend startup deadline exceeded")


def _stream_request(port: int, model: str, *, request_id: str | None = None,
                    max_tokens: int = 512) -> tuple[http.client.HTTPConnection, Any]:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content":
                      "Write a detailed 400-word explanation of addition."}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }, separators=(",", ":")).encode()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {"Content-Type": "application/json"}
    if request_id is not None:
        headers["X-VLLM-Apple-Request-ID"] = request_id
    connection.request("POST", "/v1/chat/completions", body=body, headers=headers)
    response = connection.getresponse()
    return connection, response


def _disconnect_stream(port: int, model: str) -> dict[str, object]:
    connection, response = _stream_request(port, model)
    status = response.status
    first = response.readline(1024)
    connection.close()
    return {"status": status, "received_stream_data": bool(first),
            "closed_by_client": True, "cancel_acknowledged": False}


def _cancel_stream(port: int, model: str) -> dict[str, object]:
    request_id = "qualification-active-cancel"
    connection, response = _stream_request(port, model, request_id=request_id)
    first = response.readline(1024)
    started = time.monotonic()
    cancel = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    cancel.request("DELETE", f"/vllm-apple/requests/{request_id}")
    cancel_response = cancel.getresponse()
    payload = json.loads(cancel_response.read(4096))
    cancel_status = cancel_response.status
    cancel.close()
    remaining = 0
    line = b""
    while True:
        line = response.readline(1024 * 1024 + 1)
        if not line:
            break
        remaining += len(line)
        if remaining > 4 * 1024 * 1024 or line.strip() == b"data: [DONE]":
            break
    connection.close()
    return {
        "request_id": request_id, "first_stream_data": bool(first),
        "cancel_http_status": cancel_status,
        "cancel_requested": payload.get("cancel_requested") is True,
        "completion_observed": line.strip() == b"data: [DONE]",
        "completion_after_cancel_ms": round((time.monotonic() - started) * 1000, 3),
        "remaining_response_bytes": remaining,
    }


def _slow_consumer(port: int, model: str) -> dict[str, object]:
    connection, response = _stream_request(port, model, max_tokens=128)
    if connection.sock is not None:
        connection.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
    delayed_at = time.monotonic()
    time.sleep(1)
    total = 0
    done = False
    while True:
        line = response.readline(1024 * 1024 + 1)
        if not line:
            break
        total += len(line)
        if total > 4 * 1024 * 1024:
            raise RuntimeError("slow consumer response exceeded 4 MiB")
        if line.strip() == b"data: [DONE]":
            done = True
            break
    connection.close()
    return {"requested_initial_read_delay_ms": 1000,
            "elapsed_to_completion_ms": round((time.monotonic() - delayed_at) * 1000, 3),
            "receive_buffer_bytes": 1024, "completed": done, "response_bytes": total}


def _long_prefix_cases() -> tuple[tuple[str, str, str], ...]:
    prefix = " ".join(f"Reference marker {index:04d}." for index in range(256))
    return (
        ("prefix-edit-a", prefix + " Ignore the markers. What is 1+1? Reply only 2.", "2"),
        ("prefix-edit-b", prefix + " Ignore the markers. What is 2+2? Reply only 4.", "4"),
        ("prefix-edit-c", prefix + " Ignore the markers. What is 3+3? Reply only 6.", "6"),
    )


def _validate_duration(duration_seconds: float, require_30_minute_window: bool) -> None:
    if not math.isfinite(duration_seconds) or not 0 <= duration_seconds <= 28_800:
        raise ValueError("--duration-seconds must be finite and between 0 and 28800")
    if require_30_minute_window and duration_seconds < 1_800:
        raise ValueError("--require-30-minute-window requires at least 1800 seconds")


def _new_stability_summary(duration_seconds: float, rss_bytes: int) -> dict[str, object]:
    return {
        "requested_duration_seconds": duration_seconds,
        "elapsed_seconds": 0.0,
        "cycles": 0,
        "benchmark_windows": 0,
        "requests": 0,
        "completed": 0,
        "failed": 0,
        "quality_passed": 0,
        "slo_quality_passed": 0,
        "errors": {},
        "cancel_attempts": 0,
        "cancel_passed": 0,
        "cancel_completion_max_ms": 0.0,
        "slow_consumer_attempts": 0,
        "slow_consumer_passed": 0,
        "slow_consumer_elapsed_max_ms": 0.0,
        "rss_start_bytes": rss_bytes,
        "rss_end_bytes": rss_bytes,
        "rss_peak_bytes": rss_bytes,
        "first_window": None,
        "last_window": None,
    }


def _accumulate_window(summary: dict[str, object], window: dict[str, object]) -> None:
    summary["benchmark_windows"] = int(summary["benchmark_windows"]) + 1
    for key in ("requests", "completed", "failed", "quality_passed",
                "slo_quality_passed"):
        summary[key] = int(summary[key]) + int(window.get(key, 0))
    errors = summary["errors"]
    assert isinstance(errors, dict)
    for item in window.get("errors", []):
        if isinstance(item, dict):
            code = str(item.get("code", item.get("error", "unknown")))
        else:
            code = type(item).__name__
        errors[code] = int(errors.get(code, 0)) + 1
    if summary["first_window"] is None:
        summary["first_window"] = window
    summary["last_window"] = window


def _stability_passed(summary: dict[str, object], *, require_fault_checks: bool) -> bool:
    requests = int(summary["requests"])
    passed = (
        requests > 0
        and int(summary["completed"]) == requests
        and int(summary["quality_passed"]) == requests
        and int(summary["slo_quality_passed"]) == requests
        and int(summary["failed"]) == 0
    )
    if require_fault_checks:
        passed = passed and (
            int(summary["cancel_attempts"]) > 0
            and summary["cancel_attempts"] == summary["cancel_passed"]
            and int(summary["slow_consumer_attempts"]) > 0
            and summary["slow_consumer_attempts"] == summary["slow_consumer_passed"]
        )
    return bool(passed)


def _run_stability_window(config: PhaseProbeConfig, *, port: int, model: str,
                          process_pid: int, duration_seconds: float) -> dict[str, object]:
    started = time.monotonic()
    deadline = started + duration_seconds
    summary = _new_stability_summary(duration_seconds, _resident_bytes(process_pid))
    cycle = 0
    while time.monotonic() < deadline:
        cycle += 1
        # Long-prefix edits regularly exercise prompt batching; short windows keep
        # cancellation and recovery checks frequent during a bounded soak.
        if cycle % 5 == 0:
            window = run_text_benchmark(
                config, requests=3, concurrency=2, cases=_long_prefix_cases(),
                ttft_slo_ms=10_000, e2e_slo_ms=20_000)
        else:
            window = run_text_benchmark(
                config, requests=12, concurrency=2,
                ttft_slo_ms=5_000, e2e_slo_ms=10_000)
        _accumulate_window(summary, window)

        cancel = _cancel_stream(port, model)
        summary["cancel_attempts"] = int(summary["cancel_attempts"]) + 1
        cancel_ok = (cancel.get("cancel_http_status") == 202
                     and cancel.get("cancel_requested") is True
                     and cancel.get("completion_observed") is True)
        summary["cancel_passed"] = int(summary["cancel_passed"]) + int(cancel_ok)
        summary["cancel_completion_max_ms"] = max(
            float(summary["cancel_completion_max_ms"]),
            float(cancel.get("completion_after_cancel_ms", 0)))

        if cycle % 10 == 0 or cycle == 1:
            slow = _slow_consumer(port, model)
            summary["slow_consumer_attempts"] = int(summary["slow_consumer_attempts"]) + 1
            slow_ok = slow.get("completed") is True
            summary["slow_consumer_passed"] = (
                int(summary["slow_consumer_passed"]) + int(slow_ok))
            summary["slow_consumer_elapsed_max_ms"] = max(
                float(summary["slow_consumer_elapsed_max_ms"]),
                float(slow.get("elapsed_to_completion_ms", 0)))

        rss = _resident_bytes(process_pid)
        summary["rss_end_bytes"] = rss
        summary["rss_peak_bytes"] = max(int(summary["rss_peak_bytes"]), rss)
        summary["cycles"] = cycle

    summary["elapsed_seconds"] = time.monotonic() - started
    summary["rss_growth_bytes"] = (
        int(summary["rss_end_bytes"]) - int(summary["rss_start_bytes"]))
    summary["passed"] = _stability_passed(summary, require_fault_checks=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=19096)
    parser.add_argument("--sustained-requests", type=int, default=100)
    parser.add_argument("--long-requests", type=int, default=12)
    parser.add_argument("--duration-seconds", type=float, default=0)
    parser.add_argument("--require-30-minute-window", action="store_true")
    args = parser.parse_args()
    if not args.python.is_file() or not os.access(args.python, os.X_OK):
        parser.error("--python must be an executable regular file")
    if not args.model.is_dir():
        parser.error("--model must be an existing local directory")
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    if not 1 <= args.sustained_requests <= 10_000 or not 1 <= args.long_requests <= 1_000:
        parser.error("request counts are outside bounded limits")
    try:
        _validate_duration(args.duration_seconds, args.require_30_minute_window)
    except ValueError as error:
        parser.error(str(error))

    repository = Path(__file__).resolve().parents[1]
    model = args.model.resolve()
    base_url = f"http://127.0.0.1:{args.port}"
    # Keep a virtual-environment launcher path intact. Resolving its symlink to
    # the base interpreter changes sys.prefix and can hide the MLX packages.
    python_executable = str(args.python.absolute())
    command = [python_executable, "-m", "vllm_apple.mlx_gemma2_compat",
               "--model", str(model), "--host", "127.0.0.1", "--port", str(args.port),
               "--decode-concurrency", "2", "--prompt-concurrency", "2",
               "--prompt-cache-size", "4", "--log-level", "ERROR"]
    environment = dict(os.environ, PYTHONPATH=str(repository), HF_HUB_OFFLINE="1")
    started_at = time.time()
    report: dict[str, object] = {
        "schema_version": 1, "report_kind": "gemma2_batch_mask_qualification",
        "started_at_unix": started_at, "hardware": "Apple M4 / 32 GiB",
        "command": command, "model_config_sha256": hashlib.sha256(
            (model / "config.json").read_bytes()).hexdigest(),
        "stores_prompt": False, "stores_generated_text": False,
        "cancel_acknowledgement_available": True,
    }
    log = tempfile.TemporaryFile()
    process = subprocess.Popen(command, cwd=repository, env=environment, stdout=log, stderr=log)
    clean_shutdown = False
    try:
        _wait_ready(base_url, process, 90)
        config = PhaseProbeConfig(base_url, str(model), "Apple-M4-32GiB",
                                  backend="mlx_lm_gemma2_mask_fix",
                                  maximum_output_tokens=16, timeout_seconds=30,
                                  target_pid=process.pid)
        report["warmup"] = run_text_benchmark(config, requests=3)
        report["long_prefix_edit"] = run_text_benchmark(
            config, requests=args.long_requests, concurrency=2, cases=_long_prefix_cases(),
            ttft_slo_ms=10_000, e2e_slo_ms=20_000)
        report["sustained"] = run_text_benchmark(
            config, requests=args.sustained_requests, concurrency=2,
            ttft_slo_ms=5_000, e2e_slo_ms=10_000)
        report["explicit_cancel"] = _cancel_stream(args.port, str(model))
        report["slow_consumer"] = _slow_consumer(args.port, str(model))
        if args.duration_seconds:
            report["stability_window"] = _run_stability_window(
                config, port=args.port, model=str(model), process_pid=process.pid,
                duration_seconds=args.duration_seconds)
        report["disconnect"] = _disconnect_stream(args.port, str(model))
        time.sleep(1)
        recovery = measure_stream(
            replace(config, prompt="What is 1+1? Reply with only the digit."),
            expected_text="2", expected_match_mode="trimmed_exact")
        report["post_disconnect_recovery"] = {
            "completed": recovery.measurement.stream_done_ns is not None,
            "quality_passed": recovery.expected_text_matched,
        }
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        clean_shutdown = process.returncode == 0
        log.seek(0)
        log_bytes = log.read(64 * 1024)
        report["backend_log_sha256"] = hashlib.sha256(log_bytes).hexdigest()
        report["backend_log_truncated"] = len(log_bytes) == 64 * 1024
        report["backend_exit_code"] = process.returncode
        report["shutdown_clean"] = clean_shutdown
        report["elapsed_seconds"] = time.time() - started_at
        log.close()

    long_report = report.get("long_prefix_edit", {})
    sustained = report.get("sustained", {})
    recovery = report.get("post_disconnect_recovery", {})
    explicit_cancel = report.get("explicit_cancel", {})
    slow_consumer = report.get("slow_consumer", {})
    stability = report.get("stability_window")
    duration_passed = (
        args.duration_seconds == 0
        or (isinstance(stability, dict)
            and stability.get("passed") is True
            and float(stability.get("elapsed_seconds", 0)) >= args.duration_seconds)
    )
    report["passed"] = bool(
        isinstance(long_report, dict)
        and long_report.get("completed") == args.long_requests
        and long_report.get("quality_passed") == args.long_requests
        and isinstance(sustained, dict)
        and sustained.get("completed") == args.sustained_requests
        and sustained.get("quality_passed") == args.sustained_requests
        and isinstance(recovery, dict)
        and recovery.get("completed") is True
        and recovery.get("quality_passed") is True
        and isinstance(explicit_cancel, dict)
        and explicit_cancel.get("cancel_http_status") == 202
        and explicit_cancel.get("cancel_requested") is True
        and isinstance(slow_consumer, dict)
        and slow_consumer.get("completed") is True
        and duration_passed
        and clean_shutdown
    )
    report["qualification_scope"] = (
        "30-minute M4 mixed-load qualification"
        if args.require_30_minute_window
        else "bounded M4 regression; not 30-minute certification"
    )
    _atomic_json(args.output.resolve(), report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
