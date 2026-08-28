from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
import stat
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Sequence


MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MIN_STABILITY_SECONDS = 30 * 60
LATENCY_BUCKETS_MS = (5, 10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000)


@dataclass(frozen=True, slots=True)
class SoakConfig:
    base_url: str = "http://127.0.0.1:8000"
    duration_seconds: float = 300
    warmup_seconds: float = 5
    concurrency: int = 8
    request_timeout_seconds: float = 30
    mode: str = "health"
    model: str | None = None
    session_token: str | None = None
    target_pid: int | None = None
    max_rss_growth_bytes: int | None = None
    allow_remote: bool = False
    require_30_minute_window: bool = False

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.duration_seconds)
            or not math.isfinite(self.warmup_seconds)
            or self.duration_seconds <= 0
            or self.warmup_seconds < 0
        ):
            raise ValueError("duration must be positive and warmup cannot be negative")
        if self.concurrency <= 0 or self.concurrency > 256:
            raise ValueError("concurrency must be between 1 and 256")
        if (
            not math.isfinite(self.request_timeout_seconds)
            or self.request_timeout_seconds <= 0
        ):
            raise ValueError("request timeout must be positive")
        if self.mode not in {"health", "chat", "chat-stream", "chat-mixed"}:
            raise ValueError("unsupported soak mode")
        if self.mode != "health" and not self.model:
            raise ValueError("chat modes require a model")
        if self.target_pid is not None and self.target_pid <= 0:
            raise ValueError("target pid must be positive")
        if self.max_rss_growth_bytes is not None and self.max_rss_growth_bytes < 0:
            raise ValueError("max RSS growth cannot be negative")
        if self.require_30_minute_window:
            if self.duration_seconds < MIN_STABILITY_SECONDS:
                raise ValueError("30-minute certification requires at least 1800 seconds")
            if self.target_pid is None or self.max_rss_growth_bytes is None:
                raise ValueError("30-minute certification requires PID and RSS growth limit")
        _validated_base_url(self.base_url, allow_remote=self.allow_remote)


class BoundedMetrics:
    """Thread-safe counters with fixed-size latency and error storage."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests = 0
        self._successes = 0
        self._failures = 0
        self._latency_counts = [0] * (len(LATENCY_BUCKETS_MS) + 1)
        self._latency_sum_ms = 0.0
        self._latency_max_ms = 0.0
        self._errors: dict[str, int] = {}

    def record(self, latency_ms: float, error_code: str | None = None) -> None:
        with self._lock:
            self._requests += 1
            if error_code is None:
                self._successes += 1
            else:
                self._failures += 1
                key = error_code if error_code in self._errors or len(self._errors) < 16 else "other"
                self._errors[key] = self._errors.get(key, 0) + 1
            self._latency_sum_ms += latency_ms
            self._latency_max_ms = max(self._latency_max_ms, latency_ms)
            bucket = len(LATENCY_BUCKETS_MS)
            for index, upper_bound in enumerate(LATENCY_BUCKETS_MS):
                if latency_ms <= upper_bound:
                    bucket = index
                    break
            self._latency_counts[bucket] += 1

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            requests = self._requests
            return {
                "requests": requests,
                "successes": self._successes,
                "failures": self._failures,
                "latency_ms": {
                    "mean": round(self._latency_sum_ms / requests, 3) if requests else 0.0,
                    "p50_upper_bound": self._percentile_upper_bound(0.50),
                    "p95_upper_bound": self._percentile_upper_bound(0.95),
                    "max": round(self._latency_max_ms, 3),
                },
                "errors": dict(self._errors),
                "storage": {
                    "latency_buckets": len(self._latency_counts),
                    "error_keys": len(self._errors),
                },
            }

    def _percentile_upper_bound(self, percentile: float) -> int | str:
        if self._requests == 0:
            return 0
        target = max(1, int(self._requests * percentile + 0.999999))
        cumulative = 0
        for index, count in enumerate(self._latency_counts):
            cumulative += count
            if cumulative >= target:
                if index < len(LATENCY_BUCKETS_MS):
                    return LATENCY_BUCKETS_MS[index]
                return f">{LATENCY_BUCKETS_MS[-1]}"
        return f">{LATENCY_BUCKETS_MS[-1]}"


def run_soak(config: SoakConfig) -> dict[str, object]:
    if config.warmup_seconds:
        _run_phase(config, config.warmup_seconds, metrics=None)

    rss_baseline = _resident_bytes(config.target_pid) if config.target_pid else None
    rss_peak = [rss_baseline or 0]
    monitor_stop = threading.Event()
    process_lost = threading.Event()
    monitor = None
    if config.target_pid:
        monitor = threading.Thread(
            target=_monitor_rss,
            args=(config.target_pid, monitor_stop, rss_peak, process_lost),
            daemon=True,
        )
        monitor.start()

    metrics = BoundedMetrics()
    started = time.monotonic()
    try:
        _run_phase(config, config.duration_seconds, metrics=metrics)
    finally:
        monitor_stop.set()
        if monitor:
            monitor.join(timeout=2)
    elapsed = time.monotonic() - started
    try:
        rss_final = _resident_bytes(config.target_pid) if config.target_pid else None
    except (OSError, ValueError, subprocess.SubprocessError):
        process_lost.set()
        rss_final = None
    rss_growth = None
    if rss_baseline is not None and rss_final is not None:
        rss_growth = max(0, rss_final - rss_baseline)

    result = metrics.snapshot()
    failures = int(result["failures"])
    rss_within_limit = (
        config.max_rss_growth_bytes is None
        or rss_growth is None
        or rss_growth <= config.max_rss_growth_bytes
    )
    result.update(
        {
            "elapsed_seconds": round(elapsed, 3),
            "requests_per_second": round(int(result["requests"]) / elapsed, 3),
            "rss": {
                "baseline_bytes": rss_baseline,
                "peak_bytes": rss_peak[0] if config.target_pid else None,
                "final_bytes": rss_final,
                "growth_bytes": rss_growth,
                "limit_bytes": config.max_rss_growth_bytes,
            },
            "process_alive": not process_lost.is_set(),
            "stability_window_met": elapsed >= MIN_STABILITY_SECONDS,
            "certification_required": config.require_30_minute_window,
            "passed": failures == 0
            and rss_within_limit
            and not process_lost.is_set()
            and (
                not config.require_30_minute_window
                or elapsed >= MIN_STABILITY_SECONDS
            ),
        }
    )
    return result


def _run_phase(
    config: SoakConfig,
    duration_seconds: float,
    metrics: BoundedMetrics | None,
) -> None:
    deadline = time.monotonic() + duration_seconds

    def worker(worker_index: int) -> None:
        request_index = worker_index
        while time.monotonic() < deadline:
            started = time.monotonic()
            error_code = None
            try:
                stream = config.mode == "chat-stream" or (
                    config.mode == "chat-mixed" and request_index % 2 == 1
                )
                request = _build_request(config, stream=stream)
                with urllib.request.urlopen(
                    request,
                    timeout=config.request_timeout_seconds,
                ) as response:
                    if not 200 <= response.status < 300:
                        error_code = f"http_{response.status}"
                    else:
                        error_code = _validate_response(
                            response, stream=stream, health=config.mode == "health"
                        )
            except urllib.error.HTTPError as error:
                error.read(MAX_RESPONSE_BYTES + 1)
                error_code = f"http_{error.code}"
            except (OSError, TimeoutError, urllib.error.URLError) as error:
                error_code = type(error).__name__
            if metrics is not None:
                metrics.record((time.monotonic() - started) * 1000, error_code)
            request_index += config.concurrency

    workers = [
        threading.Thread(target=worker, args=(index,))
        for index in range(config.concurrency)
    ]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join(timeout=duration_seconds + config.request_timeout_seconds + 1)
    if any(thread.is_alive() for thread in workers):
        raise RuntimeError("soak workers did not stop within the request timeout")


def _build_request(config: SoakConfig, *, stream: bool = False) -> urllib.request.Request:
    base_url = _validated_base_url(config.base_url, allow_remote=config.allow_remote)
    headers = {"Accept": "application/json"}
    if config.session_token:
        headers["Authorization"] = f"Bearer {config.session_token}"
    if config.mode == "health":
        return urllib.request.Request(base_url + "/health", headers=headers)
    body = json.dumps(
        {
            "model": config.model,
            "messages": [{"role": "user", "content": "Reply with one token."}],
            "max_tokens": 8,
            "temperature": 0,
            "stream": stream,
            **({"stream_options": {"include_usage": True}} if stream else {}),
        },
        separators=(",", ":"),
    ).encode("utf-8")
    headers["Content-Type"] = "application/json"
    return urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=body,
        headers=headers,
        method="POST",
    )


def _validate_response(
    response: BinaryIO, *, stream: bool, health: bool = False
) -> str | None:
    body = response.read(MAX_RESPONSE_BYTES + 1)
    if len(body) > MAX_RESPONSE_BYTES:
        return "response_too_large"
    if health:
        return None
    if stream:
        saw_json = False
        saw_generation = False
        saw_done = False
        try:
            for line in body.decode("utf-8").splitlines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    saw_done = True
                    continue
                payload = json.loads(data)
                if not isinstance(payload, dict) or not isinstance(payload.get("choices"), list):
                    return "invalid_stream_chunk"
                saw_json = True
                for choice in payload["choices"]:
                    delta = choice.get("delta") if isinstance(choice, dict) else None
                    if isinstance(delta, dict) and any(
                        isinstance(delta.get(field), str) and bool(delta[field])
                        for field in ("content", "reasoning_content", "reasoning")
                    ):
                        saw_generation = True
        except (UnicodeDecodeError, json.JSONDecodeError):
            return "invalid_stream_json"
        return None if saw_json and saw_generation and saw_done else "incomplete_stream"
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "invalid_chat_json"
    if not isinstance(payload, dict) or not isinstance(payload.get("choices"), list):
        return "invalid_chat_response"
    if not payload["choices"]:
        return "empty_chat_response"
    choice = payload["choices"][0]
    message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(message, dict) or not any(
        isinstance(message.get(field), str) and bool(message[field])
        for field in ("content", "reasoning_content", "reasoning")
    ):
        return "empty_chat_response"
    return None


def _validated_base_url(value: str, *, allow_remote: bool) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("base URL must use http or https and include a host")
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError("base URL must not include credentials, query, or fragment")
    try:
        parsed.port
    except ValueError as error:
        raise ValueError("base URL contains an invalid port") from error
    if not allow_remote:
        try:
            loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = parsed.hostname.lower() == "localhost"
        if not loopback:
            raise ValueError("remote targets require --allow-remote")
    return value.rstrip("/")


def _read_private_token(path: Path) -> str:
    attributes = path.lstat()
    if not stat.S_ISREG(attributes.st_mode) or attributes.st_uid != os.getuid():
        raise ValueError("session token must be a regular file owned by the current user")
    if stat.S_IMODE(attributes.st_mode) & 0o077:
        raise ValueError("session token must not be accessible by group or others")
    token = path.read_text(encoding="utf-8").strip()
    if len(token) < 32:
        raise ValueError("session token is invalid")
    return token


def _resident_bytes(pid: int) -> int:
    result = subprocess.run(
        ["/bin/ps", "-o", "rss=", "-p", str(pid)],
        capture_output=True,
        check=True,
        text=True,
        timeout=2,
    )
    return int(result.stdout.strip()) * 1024


def _monitor_rss(
    pid: int, stop: threading.Event, peak: list[int], process_lost: threading.Event
) -> None:
    while not stop.wait(1):
        try:
            peak[0] = max(peak[0], _resident_bytes(pid))
        except (OSError, ValueError, subprocess.SubprocessError):
            process_lost.set()
            return


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vllm-apple-soak",
        description="Bounded concurrent load and RSS stability runner",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--duration", type=float, default=300)
    parser.add_argument("--warmup", type=float, default=5)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument(
        "--mode",
        choices=("health", "chat", "chat-stream", "chat-mixed"),
        default="health",
    )
    parser.add_argument("--model")
    parser.add_argument("--session-token-file", type=Path)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--max-rss-growth-mib", type=float)
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--require-30-minute-window", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        token = (
            _read_private_token(arguments.session_token_file)
            if arguments.session_token_file
            else None
        )
        if arguments.max_rss_growth_mib is not None and not math.isfinite(
            arguments.max_rss_growth_mib
        ):
            raise ValueError("max RSS growth must be finite")
        growth_limit = (
            int(arguments.max_rss_growth_mib * 1024 * 1024)
            if arguments.max_rss_growth_mib is not None
            else None
        )
        config = SoakConfig(
            base_url=arguments.url,
            duration_seconds=arguments.duration,
            warmup_seconds=arguments.warmup,
            concurrency=arguments.concurrency,
            request_timeout_seconds=arguments.timeout,
            mode=arguments.mode,
            model=arguments.model,
            session_token=token,
            target_pid=arguments.pid,
            max_rss_growth_bytes=growth_limit,
            allow_remote=arguments.allow_remote,
            require_30_minute_window=arguments.require_30_minute_window,
        )
        result = run_soak(config)
    except (
        OSError,
        OverflowError,
        ValueError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as error:
        print(json.dumps({"passed": False, "error": str(error)}, separators=(",", ":")))
        return 2
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
