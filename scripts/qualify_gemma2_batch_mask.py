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
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from vllm_apple.backend import (
    BackendConfig,
    BackendHTTPError,
    BackendProcess,
    BackendSupervisor,
    OpenAIProxyEngine,
)
from vllm_apple.hardware import detect_thermal_state
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


def _stream_request(port: int, model: str, *, request_id: str | None = None,
                    max_tokens: int = 512,
                    seed: int | None = None,
                    timeout_ms: int | None = None) -> tuple[http.client.HTTPConnection, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content":
                      "Write a detailed 400-word explanation of addition."}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if seed is not None:
        payload["seed"] = seed
    body = json.dumps(payload, separators=(",", ":")).encode()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {"Content-Type": "application/json"}
    if request_id is not None:
        headers["X-VLLM-Apple-Request-ID"] = request_id
    if timeout_ms is not None:
        headers["X-VLLM-Apple-Timeout-Ms"] = str(timeout_ms)
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


def _delete_request(port: int, request_id: str) -> tuple[int, dict[str, object]]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request("DELETE", f"/vllm-apple/requests/{request_id}")
    response = connection.getresponse()
    status = response.status
    payload = json.loads(response.read(4096))
    connection.close()
    return status, payload


def _queued_cancel(port: int, model: str) -> dict[str, object]:
    blocker_id = "qualification-queue-blocker"
    queued_id = "qualification-queued-cancel"
    blocker, blocker_response = _stream_request(
        port, model, request_id=blocker_id, max_tokens=512, seed=7)
    blocker_first = blocker_response.readline(1024)
    submitted = threading.Event()
    queued_result: dict[str, object] = {}

    def submit_queued() -> None:
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
            body = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": "Reply only 2 to 1+1."}],
                "max_tokens": 16,
                "temperature": 0,
                "stream": True,
            }, separators=(",", ":")).encode()
            connection.request("POST", "/v1/chat/completions", body=body, headers={
                "Content-Type": "application/json",
                "X-VLLM-Apple-Request-ID": queued_id,
            })
            submitted.set()
            response = connection.getresponse()
            queued_result["http_status"] = response.status
            queued_result["body_bytes"] = len(response.read(64 * 1024))
            connection.close()
        except Exception as error:
            queued_result["error"] = type(error).__name__
            submitted.set()

    thread = threading.Thread(target=submit_queued, daemon=True)
    thread.start()
    if not submitted.wait(timeout=5):
        raise RuntimeError("queued cancellation request was not submitted")
    time.sleep(0.1)
    queued_status, queued_payload = _delete_request(port, queued_id)
    blocker_status, blocker_payload = _delete_request(port, blocker_id)
    blocker.close()
    thread.join(timeout=10)
    return {
        "blocker_first_stream_data": bool(blocker_first),
        "queued_cancel_http_status": queued_status,
        "queued_cancel_requested": queued_payload.get("cancel_requested") is True,
        "queued_request_state": queued_payload.get("request_state"),
        "blocker_cancel_http_status": blocker_status,
        "blocker_request_state": blocker_payload.get("request_state"),
        "queued_handler_finished": not thread.is_alive(),
        "queued_handler_http_status": queued_result.get("http_status"),
        "queued_handler_body_bytes": queued_result.get("body_bytes"),
    }


def _timeout_stream(port: int, model: str) -> dict[str, object]:
    started = time.monotonic()
    connection, response = _stream_request(
        port, model, request_id="qualification-timeout", max_tokens=512,
        seed=11, timeout_ms=100)
    total = 0
    done = False
    while True:
        line = response.readline(1024 * 1024 + 1)
        if not line:
            break
        total += len(line)
        if line.strip() == b"data: [DONE]":
            done = True
            break
        if total > 4 * 1024 * 1024:
            raise RuntimeError("timeout response exceeded 4 MiB")
    connection.close()
    return {
        "timeout_ms": 100,
        "http_status": response.status,
        "completion_observed": done,
        "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
        "response_bytes": total,
    }


def _long_prefix_cases() -> tuple[tuple[str, str, str], ...]:
    prefix = " ".join(f"Reference marker {index:04d}." for index in range(256))
    return (
        ("prefix-edit-a", prefix + " Ignore the markers. What is 1+1? Reply only 2.", "2"),
        ("prefix-edit-b", prefix + " Ignore the markers. What is 2+2? Reply only 4.", "4"),
        ("prefix-edit-c", prefix + " Ignore the markers. What is 3+3? Reply only 6.", "6"),
    )


def _validate_duration(
    duration_seconds: float,
    require_30_minute_window: bool,
    require_8_hour_window: bool = False,
) -> None:
    if not math.isfinite(duration_seconds) or not 0 <= duration_seconds <= 28_800:
        raise ValueError("--duration-seconds must be finite and between 0 and 28800")
    if require_30_minute_window and duration_seconds < 1_800:
        raise ValueError("--require-30-minute-window requires at least 1800 seconds")
    if require_8_hour_window and duration_seconds < 28_800:
        raise ValueError("--require-8-hour-window requires 28800 seconds")


def _rss_trend(samples: list[dict[str, float | int]]) -> dict[str, object]:
    """Summarize recent RSS direction without treating a peak as a leak."""
    latest_pid = samples[-1].get("pid") if samples else None
    current_epoch = [item for item in samples if item.get("pid") == latest_pid]
    if len(current_epoch) < 4:
        return {
            "sample_count": len(samples), "slope_bytes_per_hour": None,
            "current_pid": latest_pid,
            "current_pid_sample_count": len(current_epoch),
            "recent_growth_bytes": None, "plateau_observed": False,
        }
    recent = current_epoch[len(current_epoch) // 2:]
    origin = float(recent[0]["elapsed_seconds"])
    points = [
        (float(item["elapsed_seconds"]) - origin, float(item["rss_bytes"]))
        for item in recent
    ]
    count = len(points)
    x_mean = sum(x for x, _ in points) / count
    y_mean = sum(y for _, y in points) / count
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    slope = (
        sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator
        if denominator > 0 else 0.0
    )
    recent_growth = int(points[-1][1] - points[0][1])
    slope_per_hour = slope * 3600
    return {
        "sample_count": len(samples),
        "current_pid": latest_pid,
        "current_pid_sample_count": len(current_epoch),
        "recent_sample_count": count,
        "slope_bytes_per_hour": round(slope_per_hour, 3),
        "recent_growth_bytes": recent_growth,
        "plateau_observed": (
            slope_per_hour <= 16 * 1024 * 1024
            and recent_growth <= 64 * 1024 * 1024
        ),
    }


def _new_stability_summary(
    duration_seconds: float, rss_bytes: int, process_pid: int | None = None
) -> dict[str, object]:
    thermal = detect_thermal_state().value
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
        "queued_cancel_attempts": 0,
        "queued_cancel_passed": 0,
        "timeout_attempts": 0,
        "timeout_passed": 0,
        "rss_start_bytes": rss_bytes,
        "rss_end_bytes": rss_bytes,
        "rss_peak_bytes": rss_bytes,
        "thermal_start": thermal,
        "thermal_end": thermal,
        "thermal_samples": {thermal: 1},
        "first_window": None,
        "last_window": None,
        "rss_samples": [{
            "elapsed_seconds": 0.0, "rss_bytes": rss_bytes, "pid": process_pid,
        }],
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


def _stability_passed(
    summary: dict[str, object], *, require_fault_checks: bool,
    require_long_window_checks: bool = False,
) -> bool:
    requests = int(summary["requests"])
    passed = (
        requests > 0
        and int(summary["completed"]) == requests
        and int(summary["quality_passed"]) == requests
        and int(summary["slo_quality_passed"]) == requests
        and int(summary["failed"]) == 0
        and int(summary.get("unsafe_thermal_samples", 0)) == 0
    )
    if require_fault_checks:
        passed = passed and (
            int(summary["cancel_attempts"]) > 0
            and summary["cancel_attempts"] == summary["cancel_passed"]
            and int(summary["slow_consumer_attempts"]) > 0
            and summary["slow_consumer_attempts"] == summary["slow_consumer_passed"]
        )
    if require_long_window_checks:
        rss_trend = summary.get("rss_trend")
        passed = passed and (
            int(summary["queued_cancel_attempts"]) > 0
            and summary["queued_cancel_attempts"] == summary["queued_cancel_passed"]
            and int(summary["timeout_attempts"]) > 0
            and summary["timeout_attempts"] == summary["timeout_passed"]
            and isinstance(rss_trend, dict)
            and rss_trend.get("plateau_observed") is True
            and int(summary.get("worker_crash_attempts", 0)) > 0
            and summary["worker_crash_attempts"] == summary["worker_crash_passed"]
        )
    return bool(passed)


def _run_stability_window(config: PhaseProbeConfig, *, port: int, model: str,
                          process: BackendProcess, supervisor: BackendSupervisor,
                          duration_seconds: float,
                          long_concurrency: int = 2,
                          require_long_window_checks: bool = False,
                          fault_check_interval_cycles: int = 10,
                          checkpoint: Callable[[dict[str, object]], None] | None = None,
                          require_sleep_wake: bool = False,
                          worker_crash_interval_seconds: float = 0) -> dict[str, object]:
    started = time.monotonic()
    deadline = started + duration_seconds
    process_pid = process.pid
    if process_pid is None:
        raise RuntimeError("managed backend PID unavailable")
    summary = _new_stability_summary(
        duration_seconds, _resident_bytes(process_pid), process_pid
    )
    cycle = 0
    previous_wall = time.time()
    previous_monotonic = time.monotonic()
    summary["sleep_wake_observations"] = 0
    summary["maximum_suspend_gap_seconds"] = 0.0
    summary["worker_crash_attempts"] = 0
    summary["worker_crash_passed"] = 0
    summary["worker_crashes"] = []
    next_worker_crash = worker_crash_interval_seconds
    while time.monotonic() < deadline:
        cycle += 1
        current_wall = time.time()
        current_monotonic = time.monotonic()
        suspend_gap = max(
            0.0,
            (current_wall - previous_wall) - (current_monotonic - previous_monotonic),
        )
        if suspend_gap >= 5:
            summary["sleep_wake_observations"] = (
                int(summary["sleep_wake_observations"]) + 1
            )
        summary["maximum_suspend_gap_seconds"] = max(
            float(summary["maximum_suspend_gap_seconds"]), suspend_gap
        )
        previous_wall = current_wall
        previous_monotonic = current_monotonic
        # Long-prefix edits regularly exercise prompt batching; short windows keep
        # cancellation and recovery checks frequent during a bounded soak.
        current_pid = process.pid
        if current_pid is None:
            raise RuntimeError("managed backend PID unavailable during soak")
        cycle_config = replace(config, target_pid=current_pid)
        if cycle % 5 == 0:
            window = run_text_benchmark(
                cycle_config, requests=3, concurrency=long_concurrency,
                cases=_long_prefix_cases(),
                ttft_slo_ms=10_000, e2e_slo_ms=20_000)
        else:
            window = run_text_benchmark(
                cycle_config, requests=12, concurrency=2,
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

        if cycle % fault_check_interval_cycles == 0 or cycle == 1:
            slow = _slow_consumer(port, model)
            summary["slow_consumer_attempts"] = int(summary["slow_consumer_attempts"]) + 1
            slow_ok = slow.get("completed") is True
            summary["slow_consumer_passed"] = (
                int(summary["slow_consumer_passed"]) + int(slow_ok))
            summary["slow_consumer_elapsed_max_ms"] = max(
                float(summary["slow_consumer_elapsed_max_ms"]),
                float(slow.get("elapsed_to_completion_ms", 0)))

        if cycle % fault_check_interval_cycles == 0:
            queued = _queued_cancel(port, model)
            summary["queued_cancel_attempts"] = int(summary["queued_cancel_attempts"]) + 1
            queued_ok = (
                queued.get("queued_cancel_http_status") == 202
                and queued.get("queued_cancel_requested") is True
                and queued.get("queued_request_state") == "queued"
                and queued.get("blocker_cancel_http_status") == 202
                and queued.get("queued_handler_finished") is True
                and queued.get("queued_handler_http_status") == 404
            )
            summary["queued_cancel_passed"] = (
                int(summary["queued_cancel_passed"]) + int(queued_ok)
            )
            timeout = _timeout_stream(port, model)
            summary["timeout_attempts"] = int(summary["timeout_attempts"]) + 1
            timeout_ok = (
                timeout.get("http_status") == 200
                and timeout.get("completion_observed") is True
                and float(timeout.get("elapsed_ms", 60_000)) < 5_000
            )
            summary["timeout_passed"] = int(summary["timeout_passed"]) + int(timeout_ok)

        elapsed = time.monotonic() - started
        if worker_crash_interval_seconds and elapsed >= next_worker_crash:
            old_pid = process.pid
            if old_pid is None:
                raise RuntimeError("managed backend PID unavailable before crash")
            previous_restart_count = int(supervisor.snapshot()["restart_count"])
            crash_started = time.monotonic()
            os.kill(old_pid, signal.SIGKILL)
            unavailable: dict[str, object] = {
                "observed": False, "status": None, "code": None,
            }
            not_ready_deadline = time.monotonic() + 2
            while process.ready and time.monotonic() < not_ready_deadline:
                time.sleep(0.005)
            try:
                OpenAIProxyEngine(process.base_url, process).open_chat_stream({
                    "model": model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": True,
                })
            except BackendHTTPError as error:
                unavailable = {
                    "observed": True, "status": error.status, "code": error.code,
                }
            restarted = supervisor.wait_for_restart(previous_restart_count, timeout=30)
            new_pid = process.pid
            recovery_passed = False
            if restarted and new_pid is not None:
                recovery = measure_stream(
                    replace(
                        config, target_pid=new_pid,
                        prompt="What is 1+1? Reply with only the digit.",
                    ),
                    expected_text="2", expected_match_mode="trimmed_exact",
                )
                recovery_passed = bool(
                    recovery.measurement.stream_done_ns is not None
                    and recovery.expected_text_matched
                )
            crash_passed = bool(
                restarted and new_pid is not None and new_pid != old_pid
                and recovery_passed
                and unavailable == {
                    "observed": True, "status": 503, "code": "backend_unavailable"
                }
            )
            summary["worker_crash_attempts"] = int(summary["worker_crash_attempts"]) + 1
            summary["worker_crash_passed"] = (
                int(summary["worker_crash_passed"]) + int(crash_passed)
            )
            crashes = summary["worker_crashes"]
            assert isinstance(crashes, list)
            crashes.append({
                "old_pid": old_pid,
                "new_pid": new_pid,
                "restart_seconds": round(time.monotonic() - crash_started, 3),
                "client_during_restart": unavailable,
                "recovery_quality_passed": recovery_passed,
                "passed": crash_passed,
            })
            next_worker_crash += worker_crash_interval_seconds

        current_pid = process.pid
        if current_pid is None:
            raise RuntimeError("managed backend PID unavailable after fault checks")
        rss = _resident_bytes(current_pid)
        summary["rss_end_bytes"] = rss
        summary["rss_peak_bytes"] = max(int(summary["rss_peak_bytes"]), rss)
        rss_samples = summary["rss_samples"]
        assert isinstance(rss_samples, list)
        rss_samples.append({
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "rss_bytes": rss,
            "pid": current_pid,
        })
        if len(rss_samples) > 512:
            del rss_samples[1::2]
        thermal = detect_thermal_state().value
        thermal_samples = summary["thermal_samples"]
        assert isinstance(thermal_samples, dict)
        thermal_samples[thermal] = int(thermal_samples.get(thermal, 0)) + 1
        summary["thermal_end"] = thermal
        summary["cycles"] = cycle
        if checkpoint is not None:
            checkpoint(summary)

    summary["elapsed_seconds"] = time.monotonic() - started
    summary["rss_growth_bytes"] = (
        int(summary["rss_end_bytes"]) - int(summary["rss_start_bytes"]))
    thermal_samples = summary["thermal_samples"]
    assert isinstance(thermal_samples, dict)
    summary["unsafe_thermal_samples"] = sum(
        int(thermal_samples.get(state, 0)) for state in ("serious", "critical"))
    rss_samples = summary["rss_samples"]
    assert isinstance(rss_samples, list)
    summary["rss_trend"] = _rss_trend(rss_samples)
    summary["sleep_wake_passed"] = (
        not require_sleep_wake or int(summary["sleep_wake_observations"]) > 0
    )
    summary["passed"] = _stability_passed(
        summary, require_fault_checks=True,
        require_long_window_checks=require_long_window_checks,
    ) and bool(summary["sleep_wake_passed"])
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
    parser.add_argument("--require-8-hour-window", action="store_true")
    parser.add_argument("--require-sleep-wake", action="store_true")
    parser.add_argument("--fault-check-interval-cycles", type=int, default=10)
    parser.add_argument("--worker-crash-interval-seconds", type=float, default=0)
    parser.add_argument("--decode-concurrency", type=int, default=2)
    parser.add_argument("--prompt-concurrency", type=int, default=2)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--long-concurrency", type=int, default=2)
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
        _validate_duration(
            args.duration_seconds,
            args.require_30_minute_window,
            args.require_8_hour_window,
        )
    except ValueError as error:
        parser.error(str(error))
    if not 1 <= args.decode_concurrency <= 32 or not 1 <= args.prompt_concurrency <= 8:
        parser.error("backend concurrency is outside bounded limits")
    if not 1 <= args.long_concurrency <= 2:
        parser.error("--long-concurrency must be 1 or 2")
    if not 1 <= args.fault_check_interval_cycles <= 100:
        parser.error("--fault-check-interval-cycles must be between 1 and 100")
    if args.require_sleep_wake and not args.require_8_hour_window:
        parser.error("--require-sleep-wake requires --require-8-hour-window")
    if (
        not math.isfinite(args.worker_crash_interval_seconds)
        or not 0 <= args.worker_crash_interval_seconds <= 7_200
    ):
        parser.error("--worker-crash-interval-seconds must be between 0 and 7200")
    if args.require_8_hour_window and args.worker_crash_interval_seconds < 300:
        parser.error("8-hour qualification requires worker crashes every 300-7200 seconds")
    if (
        args.worker_crash_interval_seconds
        and args.duration_seconds / args.worker_crash_interval_seconds > 100
    ):
        parser.error("worker crash schedule exceeds the bounded 100-restart limit")
    if not 128 <= args.prefill_step_size <= 4096:
        parser.error("--prefill-step-size must be between 128 and 4096")

    repository = Path(__file__).resolve().parents[1]
    model = args.model.resolve()
    base_url = f"http://127.0.0.1:{args.port}"
    # Keep a virtual-environment launcher path intact. Resolving its symlink to
    # the base interpreter changes sys.prefix and can hide the MLX packages.
    python_executable = str(args.python.absolute())
    backend_config = BackendConfig(
        model=str(model), executable=Path(python_executable), host="127.0.0.1",
        port=args.port, startup_timeout=90, backend_kind="mlx_lm",
        python_module="vllm_apple.mlx_gemma2_compat",
        extra_arguments=(
            "--decode-concurrency", str(args.decode_concurrency),
            "--prompt-concurrency", str(args.prompt_concurrency),
            "--prefill-step-size", str(args.prefill_step_size),
            "--prompt-cache-size", "4",
        ),
    )
    process = BackendProcess(backend_config)
    supervisor = BackendSupervisor(
        process, poll_interval=0.05, initial_backoff=0.25,
        maximum_backoff=2, maximum_restarts=100,
    )
    command = backend_config.command()
    started_at = time.time()
    report: dict[str, object] = {
        "schema_version": 1, "report_kind": "gemma2_batch_mask_qualification",
        "started_at_unix": started_at, "hardware": "Apple M4 / 32 GiB",
        "command": command, "model_config_sha256": hashlib.sha256(
            (model / "config.json").read_bytes()).hexdigest(),
        "stores_prompt": False, "stores_generated_text": False,
        "cancel_acknowledgement_available": True,
    }
    clean_shutdown = False
    original_pythonpath = os.environ.get("PYTHONPATH")
    original_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["PYTHONPATH"] = str(repository)
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        supervisor.start()
        if process.pid is None:
            raise RuntimeError("managed backend PID unavailable after startup")
        config = PhaseProbeConfig(base_url, str(model), "Apple-M4-32GiB",
                                  backend="mlx_lm_gemma2_mask_fix",
                                  maximum_output_tokens=16, timeout_seconds=30,
                                  target_pid=process.pid)
        report["warmup"] = run_text_benchmark(config, requests=3)
        report["long_prefix_edit"] = run_text_benchmark(
            config, requests=args.long_requests, concurrency=args.long_concurrency,
            cases=_long_prefix_cases(),
            ttft_slo_ms=10_000, e2e_slo_ms=20_000)
        report["sustained"] = run_text_benchmark(
            config, requests=args.sustained_requests, concurrency=2,
            ttft_slo_ms=5_000, e2e_slo_ms=10_000)
        report["explicit_cancel"] = _cancel_stream(args.port, str(model))
        report["slow_consumer"] = _slow_consumer(args.port, str(model))
        report["queued_cancel"] = _queued_cancel(args.port, str(model))
        report["request_timeout"] = _timeout_stream(args.port, str(model))
        if args.duration_seconds:
            def checkpoint(summary: dict[str, object]) -> None:
                _atomic_json(args.output.resolve(), {
                    **report,
                    "status": "running",
                    "stability_window": summary,
                    "passed": False,
                })

            report["stability_window"] = _run_stability_window(
                config, port=args.port, model=str(model), process=process,
                supervisor=supervisor,
                duration_seconds=args.duration_seconds,
                long_concurrency=args.long_concurrency,
                require_long_window_checks=args.require_8_hour_window,
                fault_check_interval_cycles=args.fault_check_interval_cycles,
                checkpoint=checkpoint,
                require_sleep_wake=args.require_sleep_wake,
                worker_crash_interval_seconds=args.worker_crash_interval_seconds)
        report["disconnect"] = _disconnect_stream(args.port, str(model))
        time.sleep(1)
        recovery_pid = process.pid
        if recovery_pid is None:
            raise RuntimeError("managed backend PID unavailable for final recovery")
        recovery = measure_stream(
            replace(
                config, target_pid=recovery_pid,
                prompt="What is 1+1? Reply with only the digit.",
            ),
            expected_text="2", expected_match_mode="trimmed_exact")
        report["post_disconnect_recovery"] = {
            "completed": recovery.measurement.stream_done_ns is not None,
            "quality_passed": recovery.expected_text_matched,
        }
    finally:
        supervisor.stop()
        clean_shutdown = not process.running
        log_bytes = "\n".join(process.recent_logs()).encode()[:64 * 1024]
        report["backend_log_sha256"] = hashlib.sha256(log_bytes).hexdigest()
        report["backend_log_truncated"] = len(log_bytes) == 64 * 1024
        report["supervisor"] = supervisor.snapshot()
        report["shutdown_clean"] = clean_shutdown
        report["elapsed_seconds"] = time.time() - started_at
        if original_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = original_pythonpath
        if original_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = original_offline

    long_report = report.get("long_prefix_edit", {})
    sustained = report.get("sustained", {})
    recovery = report.get("post_disconnect_recovery", {})
    explicit_cancel = report.get("explicit_cancel", {})
    slow_consumer = report.get("slow_consumer", {})
    queued_cancel = report.get("queued_cancel", {})
    request_timeout = report.get("request_timeout", {})
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
        and isinstance(queued_cancel, dict)
        and queued_cancel.get("queued_cancel_http_status") == 202
        and queued_cancel.get("queued_cancel_requested") is True
        and queued_cancel.get("queued_request_state") == "queued"
        and queued_cancel.get("blocker_cancel_http_status") == 202
        and queued_cancel.get("queued_handler_finished") is True
        and queued_cancel.get("queued_handler_http_status") == 404
        and isinstance(request_timeout, dict)
        and request_timeout.get("http_status") == 200
        and request_timeout.get("completion_observed") is True
        and float(request_timeout.get("elapsed_ms", 60_000)) < 5_000
        and duration_passed
        and clean_shutdown
    )
    report["qualification_scope"] = (
        "8-hour M4 mixed-load qualification"
        if args.require_8_hour_window
        else "30-minute M4 mixed-load qualification"
        if args.require_30_minute_window
        else "bounded M4 regression; not 30-minute certification"
    )
    report["status"] = "complete"
    _atomic_json(args.output.resolve(), report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
