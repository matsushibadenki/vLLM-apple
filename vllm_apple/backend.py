from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable

from .compat import resolve_vllm_executable
from .kernel_context import KERNEL_TUNING_ACCEPTED_HEADER, InferenceKernelContext
from .observability import REQUEST_ID_HEADER, current_request_id
from .token_count_cache import TokenCountCache, TokenCountSingleFlight

MAX_UPSTREAM_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_BACKEND_HELP_BYTES = 1024 * 1024
MAX_TOKENIZE_SCAN_BYTES = 64 * 1024
VLLM_APPLE_TUNING_MIDDLEWARE = "vllm_apple.vllm_middleware.VLLMAppleKernelTuningMiddleware"


class BackendConfigurationError(ValueError):
    pass


class BackendStartupError(RuntimeError):
    def __init__(self, message: str, *, code: str = "backend_startup_failed") -> None:
        super().__init__(message)
        self.code = code


def supports_kernel_tuning_middleware(executable: Path, timeout: float = 10.0) -> bool:
    """Bounded capability probe for the two CLI flags required by the bridge."""
    if timeout <= 0:
        raise ValueError("middleware capability timeout must be positive")
    try:
        with tempfile.TemporaryFile() as output:
            completed = subprocess.run(
                [str(executable), "serve", "--help"],
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            output.seek(0)
            help_text = output.read(MAX_BACKEND_HELP_BYTES + 1)
    except (OSError, subprocess.SubprocessError):
        return False
    if completed.returncode != 0 or len(help_text) > MAX_BACKEND_HELP_BYTES:
        return False
    return all(
        option in help_text for option in (b"--middleware", b"--disable-frontend-multiprocessing")
    )


class BackendHTTPError(RuntimeError):
    def __init__(self, status: int, code: str | None, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass(frozen=True, slots=True)
class BackendConfig:
    model: str
    executable: Path
    host: str = "127.0.0.1"
    port: int = 8001
    max_model_len: int | None = None
    startup_timeout: float = 600.0
    extra_arguments: tuple[str, ...] = ()
    enable_kernel_tuning_middleware: bool = False
    backend_kind: str = "vllm_metal"

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise BackendConfigurationError("model cannot be empty")
        if self.host not in {"127.0.0.1", "::1", "localhost"}:
            raise BackendConfigurationError("managed inference backend must be loopback-only")
        if not 1 <= self.port <= 65535:
            raise BackendConfigurationError("invalid backend port")
        if self.max_model_len is not None and self.max_model_len <= 0:
            raise BackendConfigurationError("max_model_len must be positive")
        if self.startup_timeout <= 0:
            raise BackendConfigurationError("startup_timeout must be positive")
        if self.backend_kind not in {"vllm_metal", "mlx_lm"}:
            raise BackendConfigurationError("unsupported managed backend kind")
        reserved = {
            "--host",
            "--port",
            "--max-model-len",
            "--middleware",
            "--disable-frontend-multiprocessing",
        }
        if any(argument in reserved for argument in self.extra_arguments):
            raise BackendConfigurationError("extra_arguments cannot override managed options")

    def command(self) -> list[str]:
        if self.backend_kind == "mlx_lm":
            return [
                str(self.executable),
                "--model",
                self.model,
                "--host",
                self.host,
                "--port",
                str(self.port),
                "--log-level",
                "WARNING",
                *self.extra_arguments,
            ]
        command = [
            str(self.executable),
            "serve",
            self.model,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        if self.max_model_len is not None:
            command.extend(("--max-model-len", str(self.max_model_len)))
        if self.enable_kernel_tuning_middleware:
            command.extend(
                (
                    "--disable-frontend-multiprocessing",
                    "--middleware",
                    VLLM_APPLE_TUNING_MIDDLEWARE,
                )
            )
        command.extend(self.extra_arguments)
        return command


class BackendProcess:
    """Owns a vLLM-Metal server without importing it into the control process."""

    def __init__(self, config: BackendConfig, log_line_limit: int = 400) -> None:
        if log_line_limit <= 0:
            raise ValueError("log_line_limit must be positive")
        self.config = config
        self._process: subprocess.Popen[str] | None = None
        self._ready = False
        self._lock = threading.RLock()
        self._logs: deque[str] = deque(maxlen=log_line_limit)
        self._drainers: list[threading.Thread] = []

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._ready and self._process is not None and self._process.poll() is None

    @property
    def running(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    @property
    def pid(self) -> int | None:
        with self._lock:
            return self._process.pid if self._process is not None else None

    @property
    def base_url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}"

    def recent_logs(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._logs)

    def _drain(self, stream: Iterable[str], label: str) -> None:
        try:
            for line in stream:
                with self._lock:
                    self._logs.append(f"{label}: {line.rstrip()}")
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()

    def start(self) -> None:
        with self._lock:
            if self._process is not None:
                raise BackendStartupError("backend process was already started")
            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            try:
                process = subprocess.Popen(
                    self.config.command(),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    env=environment,
                )
            except OSError as error:
                raise BackendStartupError(f"unable to start backend: {error}") from error
            self._process = process
            for stream, label in ((process.stdout, "stdout"), (process.stderr, "stderr")):
                if stream is not None:
                    thread = threading.Thread(
                        target=self._drain,
                        args=(stream, label),
                        daemon=True,
                        name=f"vllm-apple-{label}-drain",
                    )
                    thread.start()
                    self._drainers.append(thread)

        deadline = time.monotonic() + self.config.startup_timeout
        while time.monotonic() < deadline:
            with self._lock:
                process = self._process
                exit_code = process.poll() if process else -1
            if exit_code is not None:
                logs = "\n".join(self.recent_logs()[-20:])
                raise BackendStartupError(
                    f"backend exited with {exit_code}\n{logs}",
                    code="backend_exited",
                )
            if self._probe_ready():
                with self._lock:
                    self._ready = True
                return
            time.sleep(0.1)
        self.stop()
        raise BackendStartupError(
            "backend readiness timed out",
            code="backend_readiness_timeout",
        )

    def _probe_ready(self) -> bool:
        for path in ("/health", "/v1/models"):
            try:
                with urllib.request.urlopen(self.base_url + path, timeout=1.0) as response:
                    if 200 <= response.status < 300:
                        return True
            except (OSError, urllib.error.URLError):
                continue
        return False

    def stop(self, timeout: float = 10.0) -> None:
        with self._lock:
            process = self._process
            self._ready = False
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        for thread in self._drainers:
            thread.join(timeout=1.0)
        with self._lock:
            if self._process is process:
                self._process = None
                self._drainers = []

    def restart(self) -> None:
        """Recycle the managed backend using the immutable validated config."""
        self.stop()
        self.start()


class OpenAIProxyEngine:
    """Low-buffer OpenAI API proxy for a local vLLM-Metal process."""

    def __init__(
        self,
        base_url: str,
        process: BackendProcess | None = None,
        token_count_cache: TokenCountCache | None = None,
        token_count_single_flight: TokenCountSingleFlight | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.process = process
        self._tuning_ack_lock = threading.Lock()
        self._tuning_acknowledged = 0
        self._tuning_ack_missing = 0
        self._tuning_ack_mismatch = 0
        self._tokenizer_lock = threading.Lock()
        self._tokenizer_measured = 0
        self._tokenizer_failures = 0
        self._token_count_cache = token_count_cache or TokenCountCache()
        self._token_count_single_flight = token_count_single_flight or TokenCountSingleFlight()

    def tuning_ack_snapshot(self) -> dict[str, int]:
        with self._tuning_ack_lock:
            return {
                "acknowledged": self._tuning_acknowledged,
                "missing": self._tuning_ack_missing,
                "mismatch": self._tuning_ack_mismatch,
            }

    def tokenizer_snapshot(self) -> dict[str, int]:
        cache = self._token_count_cache.snapshot()
        single_flight = self._token_count_single_flight.snapshot()
        with self._tokenizer_lock:
            return {
                "measured": self._tokenizer_measured,
                "failures": self._tokenizer_failures,
                "cache_capacity": cache.capacity,
                "cache_entries": cache.entries,
                "cache_hits": cache.hits,
                "cache_misses": cache.misses,
                "cache_evictions": cache.evictions,
                "cache_expirations": cache.expirations,
                "single_flight_capacity": single_flight.capacity,
                "single_flight_active": single_flight.active,
                "single_flight_leaders": single_flight.leaders,
                "single_flight_followers": single_flight.followers,
                "single_flight_bypasses": single_flight.bypasses,
                "single_flight_timeouts": single_flight.timeouts,
            }

    def estimate_prompt_tokens(self, request: dict[str, Any]) -> int | None:
        allowed = (
            "model",
            "messages",
            "add_generation_prompt",
            "continue_final_message",
            "add_special_tokens",
            "tools",
            "chat_template",
            "chat_template_kwargs",
        )
        payload = {key: request[key] for key in allowed if key in request}
        if "messages" not in payload:
            return None
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        fingerprint = hashlib.sha256(body).hexdigest()
        cached = self._token_count_cache.get(fingerprint)
        if cached is not None:
            return cached
        flight = self._token_count_single_flight.join(fingerprint)
        if not flight.leader:
            return self._token_count_single_flight.wait(flight, timeout=5.5)
        count: int | None = None
        try:
            with urllib.request.urlopen(self._request("/tokenize", body), timeout=5.0) as response:
                scanned = bytearray()
                while len(scanned) <= MAX_TOKENIZE_SCAN_BYTES:
                    chunk = response.read(min(4096, MAX_TOKENIZE_SCAN_BYTES + 1 - len(scanned)))
                    if not chunk:
                        break
                    scanned.extend(chunk)
                    match = re.search(rb'"count"\s*:\s*(\d+)', scanned)
                    if match is not None:
                        count = int(match.group(1))
                        with self._tokenizer_lock:
                            self._tokenizer_measured += 1
                        self._token_count_cache.put(fingerprint, count)
                        return count
        except (OSError, urllib.error.URLError, ValueError):
            pass
        finally:
            self._token_count_single_flight.complete(flight, count)
        with self._tokenizer_lock:
            self._tokenizer_failures += 1
        return None

    def _record_tuning_ack(
        self, context: InferenceKernelContext | None, response: BinaryIO
    ) -> None:
        if context is None:
            return
        headers = getattr(response, "headers", None)
        received = headers.get(KERNEL_TUNING_ACCEPTED_HEADER) if headers is not None else None
        with self._tuning_ack_lock:
            if received == context.tuning_id:
                self._tuning_acknowledged += 1
            elif received is None:
                self._tuning_ack_missing += 1
            else:
                self._tuning_ack_mismatch += 1

    @property
    def ready(self) -> bool:
        if self.process is not None:
            return self.process.ready
        try:
            with urllib.request.urlopen(self.base_url + "/v1/models", timeout=0.5) as response:
                return 200 <= response.status < 300
        except (OSError, urllib.error.URLError):
            return False

    def _request(
        self,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> urllib.request.Request:
        request_headers = dict(headers or {})
        request_id = current_request_id()
        if request_id is not None:
            request_headers.setdefault(REQUEST_ID_HEADER, request_id)
        if body is not None:
            request_headers["Content-Type"] = "application/json"
        return urllib.request.Request(
            self.base_url + path,
            data=body,
            headers=request_headers,
            method="POST" if body is not None else "GET",
        )

    def _read_json(self, response: BinaryIO) -> dict[str, Any]:
        data = response.read(MAX_UPSTREAM_RESPONSE_BYTES + 1)
        if len(data) > MAX_UPSTREAM_RESPONSE_BYTES:
            raise BackendHTTPError(
                502, "backend_response_too_large", "backend response is too large"
            )
        try:
            payload = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BackendHTTPError(
                502, "invalid_backend_response", "backend returned invalid JSON"
            ) from error
        if not isinstance(payload, dict):
            raise BackendHTTPError(
                502, "invalid_backend_response", "backend response must be an object"
            )
        return payload

    def _translate_http_error(self, error: urllib.error.HTTPError) -> BackendHTTPError:
        data = error.read(MAX_UPSTREAM_RESPONSE_BYTES + 1)
        code: str | None = None
        message = str(error.reason)
        try:
            payload = json.loads(data)
            body = payload.get("error", {}) if isinstance(payload, dict) else {}
            if isinstance(body, dict):
                code = body.get("code")
                message = str(body.get("message") or message)
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        return BackendHTTPError(error.code, code, message)

    def models(self) -> list[dict[str, Any]]:
        try:
            with urllib.request.urlopen(self._request("/v1/models"), timeout=5.0) as response:
                payload = self._read_json(response)
        except urllib.error.HTTPError as error:
            raise self._translate_http_error(error) from error
        except (OSError, urllib.error.URLError) as error:
            raise BackendHTTPError(503, "backend_unavailable", str(error)) from error
        data = payload.get("data")
        if not isinstance(data, list):
            raise BackendHTTPError(502, "invalid_backend_response", "models data must be a list")
        return [item for item in data if isinstance(item, dict)]

    def chat_completions(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.chat_completions_with_context(request, None)

    def chat_completions_with_context(
        self,
        request: dict[str, Any],
        context: InferenceKernelContext | None,
    ) -> dict[str, Any]:
        body = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        headers = context.to_http_headers() if context is not None else None
        try:
            with urllib.request.urlopen(
                self._request("/v1/chat/completions", body, headers), timeout=300.0
            ) as response:
                self._record_tuning_ack(context, response)
                return self._read_json(response)
        except urllib.error.HTTPError as error:
            raise self._translate_http_error(error) from error
        except (OSError, urllib.error.URLError) as error:
            raise BackendHTTPError(503, "backend_unavailable", str(error)) from error

    def open_chat_stream(self, request: dict[str, Any]) -> AbstractContextManager[BinaryIO]:
        return self.open_chat_stream_with_context(request, None)

    def open_chat_stream_with_context(
        self,
        request: dict[str, Any],
        context: InferenceKernelContext | None,
    ) -> AbstractContextManager[BinaryIO]:
        streaming_request = dict(request)
        streaming_request["stream"] = True
        body = json.dumps(streaming_request, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        headers = context.to_http_headers() if context is not None else None
        try:
            response = urllib.request.urlopen(
                self._request("/v1/chat/completions", body, headers), timeout=300.0
            )
            self._record_tuning_ack(context, response)
            return response
        except urllib.error.HTTPError as error:
            raise self._translate_http_error(error) from error
        except (OSError, urllib.error.URLError) as error:
            raise BackendHTTPError(503, "backend_unavailable", str(error)) from error


def make_backend_config(
    model: str,
    executable: str | None,
    port: int,
    max_model_len: int | None,
    startup_timeout: float,
    enable_kernel_tuning_middleware: bool = False,
) -> BackendConfig:
    resolved = resolve_vllm_executable(executable)
    if resolved is None:
        raise BackendConfigurationError(
            "vllm executable not found; pass --backend-executable or set VLLM_APPLE_VLLM_EXECUTABLE"
        )
    return BackendConfig(
        model=model,
        executable=resolved,
        port=port,
        max_model_len=max_model_len,
        startup_timeout=startup_timeout,
        enable_kernel_tuning_middleware=enable_kernel_tuning_middleware,
    )
