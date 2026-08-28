from __future__ import annotations

import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .phase_profile import ExecutionPhaseProfiler, PhaseMeasurement
from .soak import _validated_base_url


MAX_SSE_LINE_BYTES = 1024 * 1024
MAX_PROMPT_BYTES = 8 * 1024 * 1024


class PhaseProbeError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


@dataclass(frozen=True, slots=True)
class PhaseProbeConfig:
    base_url: str
    model: str
    hardware_fingerprint: str
    backend: str = "vllm_metal"
    samples: int = 3
    maximum_output_tokens: int = 32
    timeout_seconds: float = 300
    session_token: str | None = None
    target_pid: int | None = None
    allow_remote: bool = False
    prompt: str = "Reply with a short factual sentence."

    def __post_init__(self) -> None:
        _validated_base_url(self.base_url, allow_remote=self.allow_remote)
        if not self.model or not self.hardware_fingerprint or not self.backend:
            raise ValueError("model, hardware fingerprint, and backend are required")
        if not 1 <= self.samples <= 1_000:
            raise ValueError("samples must be between 1 and 1000")
        if not 1 <= self.maximum_output_tokens <= 4096:
            raise ValueError("maximum_output_tokens must be between 1 and 4096")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.target_pid is not None and self.target_pid <= 0:
            raise ValueError("target_pid must be positive")
        if not self.prompt or len(self.prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
            raise ValueError("prompt must be between 1 byte and 8 MiB")


@dataclass(frozen=True, slots=True)
class StreamProbeResult:
    measurement: PhaseMeasurement
    expected_text_matched: bool | None
    steady_memory_bytes: int


def run_phase_probe(config: PhaseProbeConfig) -> dict[str, Any]:
    profiler = ExecutionPhaseProfiler(
        config.hardware_fingerprint, config.model, config.backend
    )
    for _ in range(config.samples):
        profiler.record(measure_stream(config).measurement)
    return profiler.snapshot()


def measure_stream(
    config: PhaseProbeConfig, *, expected_text: str | None = None
) -> StreamProbeResult:
    if expected_text is not None and (
        not expected_text or len(expected_text.encode("utf-8")) > 1024
    ):
        raise ValueError("expected_text must be between 1 byte and 1 KiB")
    base_url = _validated_base_url(config.base_url, allow_remote=config.allow_remote)
    body = json.dumps(
        {
            "model": config.model,
            "messages": [{"role": "user", "content": config.prompt}],
            "max_tokens": config.maximum_output_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if config.session_token:
        headers["Authorization"] = f"Bearer {config.session_token}"
    request = urllib.request.Request(
        base_url + "/v1/chat/completions", data=body, headers=headers, method="POST"
    )
    peak = [_resident_bytes(config.target_pid) if config.target_pid else 0]
    stop = threading.Event()
    monitor = None
    if config.target_pid:
        monitor = threading.Thread(
            target=_monitor_resident_bytes,
            args=(config.target_pid, stop, peak),
            daemon=True,
        )
        monitor.start()

    started_ns = time.monotonic_ns()
    first_token_ns: int | None = None
    completed_ns = started_ns
    usage: dict[str, Any] | None = None
    expected_matched = False
    match_tail = ""
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            while True:
                line = response.readline(MAX_SSE_LINE_BYTES + 1)
                if not line:
                    break
                if len(line) > MAX_SSE_LINE_BYTES:
                    raise PhaseProbeError("sse_line_too_large", "backend SSE line exceeded 1 MiB")
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    completed_ns = time.monotonic_ns()
                    break
                try:
                    event = json.loads(data)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise PhaseProbeError("invalid_sse_json", "backend returned invalid SSE JSON") from error
                if not isinstance(event, dict):
                    continue
                generated = _generated_text(event)
                if generated and first_token_ns is None:
                    first_token_ns = time.monotonic_ns()
                answer = _answer_text(event)
                if expected_text is not None and answer and not expected_matched:
                    candidate_text = match_tail + answer
                    expected_matched = expected_text in candidate_text
                    keep = max(0, len(expected_text) - 1)
                    match_tail = candidate_text[-keep:] if keep else ""
                candidate = event.get("usage")
                if isinstance(candidate, dict):
                    usage = candidate
            completed_ns = max(completed_ns, time.monotonic_ns())
    except urllib.error.HTTPError as error:
        error.read(64 * 1024)
        raise PhaseProbeError("backend_http_error", f"backend returned HTTP {error.code}") from error
    except (OSError, TimeoutError, urllib.error.URLError) as error:
        raise PhaseProbeError("backend_unavailable", str(error)) from error
    finally:
        stop.set()
        if monitor:
            monitor.join(timeout=1)

    if first_token_ns is None:
        raise PhaseProbeError("first_token_missing", "stream contained no generated token")
    prompt_tokens = _usage_integer(usage, "prompt_tokens")
    output_tokens = _usage_integer(usage, "completion_tokens")
    if prompt_tokens is None or output_tokens is None or output_tokens <= 0:
        raise PhaseProbeError(
            "usage_missing",
            "backend must return prompt_tokens and completion_tokens with stream usage",
        )
    return StreamProbeResult(
        measurement=PhaseMeasurement(
            started_ns=started_ns,
            first_token_ns=first_token_ns,
            completed_ns=completed_ns,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            peak_memory_bytes=peak[0],
        ),
        expected_text_matched=expected_matched if expected_text is not None else None,
        steady_memory_bytes=_resident_bytes(config.target_pid),
    )


def _generated_text(event: dict[str, Any]) -> str:
    choices = event.get("choices")
    if not isinstance(choices, list):
        return ""
    for choice in choices:
        delta = choice.get("delta") if isinstance(choice, dict) else None
        if isinstance(delta, dict):
            for name in ("reasoning_content", "reasoning", "content"):
                value = delta.get(name)
                if isinstance(value, str) and value:
                    return value
    return ""


def _answer_text(event: dict[str, Any]) -> str:
    choices = event.get("choices")
    if not isinstance(choices, list):
        return ""
    for choice in choices:
        delta = choice.get("delta") if isinstance(choice, dict) else None
        value = delta.get("content") if isinstance(delta, dict) else None
        if isinstance(value, str) and value:
            return value
    return ""


def _usage_integer(usage: dict[str, Any] | None, name: str) -> int | None:
    value = usage.get(name) if usage else None
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _resident_bytes(pid: int | None) -> int:
    if pid is None:
        return 0
    try:
        result = subprocess.run(
            ["/bin/ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            check=True,
            text=True,
            timeout=2,
        )
        return int(result.stdout.strip()) * 1024
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise PhaseProbeError("rss_unavailable", f"unable to inspect pid {pid}") from error


def _monitor_resident_bytes(pid: int, stop: threading.Event, peak: list[int]) -> None:
    while not stop.wait(0.05):
        try:
            peak[0] = max(peak[0], _resident_bytes(pid))
        except PhaseProbeError:
            return
