from __future__ import annotations

import hashlib
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

PROMOTION_PROBE_SCHEMA_VERSION = 1
MAX_PROMOTION_RESPONSE_BYTES = 1024 * 1024
MAX_PROMOTION_TEXT_BYTES = 64 * 1024


class PromotionProbeError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


@dataclass(frozen=True, slots=True)
class PromotionProbeConfig:
    base_url: str
    model: str
    timeout_seconds: float = 120.0
    seed: int = 1729
    maximum_output_tokens: int = 16

    def __post_init__(self) -> None:
        parsed = urllib.parse.urlparse(self.base_url)
        if parsed.scheme != "http" or parsed.hostname not in {
            "127.0.0.1",
            "::1",
            "localhost",
        }:
            raise ValueError("promotion probe backend must be loopback HTTP")
        if not self.model.strip() or len(self.model) > 512:
            raise ValueError("promotion probe model is invalid")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("promotion probe timeout must be positive and finite")
        if isinstance(self.seed, bool) or not 0 <= self.seed <= 2**31 - 1:
            raise ValueError("promotion probe seed is invalid")
        if not 1 <= self.maximum_output_tokens <= 256:
            raise ValueError("promotion probe output token limit must be 1 to 256")


@dataclass(frozen=True, slots=True)
class PromotionResponse:
    text: str
    stream_completed: bool

    def __post_init__(self) -> None:
        size = len(self.text.encode("utf-8"))
        if not 1 <= size <= MAX_PROMOTION_TEXT_BYTES:
            raise ValueError("promotion response text is empty or oversized")

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


PromotionTransport = Callable[[dict[str, object], bool], PromotionResponse]


def run_serving_promotion_probe(
    config: PromotionProbeConfig,
    *,
    transport: PromotionTransport | None = None,
) -> dict[str, object]:
    send = transport or _http_transport(config)
    common: dict[str, object] = {
        "model": config.model,
        "messages": [
            {
                "role": "user",
                "content": "Reply with exactly one short lowercase English word.",
            }
        ],
        "max_tokens": config.maximum_output_tokens,
        "seed": config.seed,
    }
    greedy_a = send({**common, "temperature": 0}, False)
    greedy_b = send({**common, "temperature": 0}, False)
    sampled_request = {**common, "temperature": 0.7, "top_p": 0.9}
    sampled_a = send(sampled_request, False)
    sampled_b = send(sampled_request, False)
    sampled_stream = send(sampled_request, True)
    checks = {
        "greedy_repeat_equal": greedy_a.digest == greedy_b.digest,
        "sampled_repeat_equal": sampled_a.digest == sampled_b.digest,
        "sampled_stream_equal": sampled_a.digest == sampled_stream.digest,
        "stream_completed": sampled_stream.stream_completed,
    }
    return {
        "schema_version": PROMOTION_PROBE_SCHEMA_VERSION,
        "model": config.model,
        "seed": config.seed,
        "temperature": 0.7,
        "digests": {
            "greedy": greedy_a.digest,
            "sampled": sampled_a.digest,
            "sampled_stream": sampled_stream.digest,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _http_transport(config: PromotionProbeConfig) -> PromotionTransport:
    endpoint = config.base_url.rstrip("/") + "/v1/chat/completions"

    def send(payload: dict[str, object], stream: bool) -> PromotionResponse:
        request_payload = {**payload, "stream": stream}
        if stream:
            request_payload["stream_options"] = {"include_usage": True}
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(request_payload, separators=(",", ":")).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if stream else "application/json",
            },
            method="POST",
        )
        try:
            response = urllib.request.urlopen(request, timeout=config.timeout_seconds)
        except (OSError, urllib.error.URLError) as error:
            raise PromotionProbeError("backend_request_failed", str(error)) from error
        with response:
            return _read_stream(response) if stream else _read_non_stream(response)

    return send


def _read_non_stream(response: object) -> PromotionResponse:
    raw = response.read(MAX_PROMOTION_RESPONSE_BYTES + 1)
    if len(raw) > MAX_PROMOTION_RESPONSE_BYTES:
        raise PromotionProbeError("response_oversized", "non-streaming response exceeded 1 MiB")
    try:
        payload = json.loads(raw)
        text = payload["choices"][0]["message"]["content"]
        if not isinstance(text, str):
            raise TypeError
        return PromotionResponse(text, False)
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise PromotionProbeError("invalid_non_stream_response", "invalid chat response") from error


def _read_stream(response: object) -> PromotionResponse:
    total = 0
    text_bytes = 0
    pieces: list[str] = []
    completed = False
    while True:
        line = response.readline(MAX_PROMOTION_RESPONSE_BYTES + 1)
        if not line:
            break
        total += len(line)
        if total > MAX_PROMOTION_RESPONSE_BYTES or len(line) > MAX_PROMOTION_RESPONSE_BYTES:
            raise PromotionProbeError("response_oversized", "streaming response exceeded 1 MiB")
        stripped = line.strip()
        if not stripped.startswith(b"data:"):
            continue
        data = stripped[5:].strip()
        if data == b"[DONE]":
            completed = True
            break
        try:
            payload = json.loads(data)
            content = payload["choices"][0].get("delta", {}).get("content")
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise PromotionProbeError("invalid_stream_response", "invalid SSE chunk") from error
        if content is not None:
            if not isinstance(content, str):
                raise PromotionProbeError("invalid_stream_response", "invalid SSE content")
            if not content:
                continue
            pieces.append(content)
            text_bytes += len(content.encode("utf-8"))
            if text_bytes > MAX_PROMOTION_TEXT_BYTES:
                raise PromotionProbeError("response_oversized", "generated stream text is oversized")
    try:
        return PromotionResponse("".join(pieces), completed)
    except ValueError as error:
        raise PromotionProbeError("incomplete_stream", str(error)) from error
