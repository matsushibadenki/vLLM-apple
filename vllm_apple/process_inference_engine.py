"""Bounded inference transport to a model owned by a subprocess main thread."""
from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
import threading
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .inference_request import (
    InferenceEngineBusy,
    InferenceRequestCancelled,
    InferenceRequestContext,
)

MAX_PROCESS_MESSAGE_BYTES = 4 * 1024 * 1024
MAX_PROCESS_RESPONSE_BYTES = 16 * 1024 * 1024
_FACTORY = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*\Z")
_TERMINAL_ERRORS = frozenset({
    "engine_busy", "invalid_request", "request_cancelled", "request_timed_out",
    "delegate_failed", "diagnostics_failed",
})


@dataclass(slots=True)
class _Pending:
    done: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None
    error_code: str | None = None
    completion: Future[None] = field(default_factory=Future)

    def finish(self) -> None:
        self.done.set()
        self.completion.set_result(None)


class MainThreadSubprocessInferenceEngine:
    """Keep native model initialization, execution, and cleanup on child main."""

    def __init__(
        self,
        factory: str,
        *,
        python_executable: Path,
        config_path: Path | None = None,
        maximum_pending_requests: int = 8,
        startup_timeout_seconds: float = 600.0,
        shutdown_timeout_seconds: float = 15.0,
    ) -> None:
        if (not isinstance(factory, str) or _FACTORY.fullmatch(factory) is None
                or type(maximum_pending_requests) is not int
                or not 1 <= maximum_pending_requests <= 64
                or not 0 < startup_timeout_seconds <= 1800
                or not 0 < shutdown_timeout_seconds <= 60):
            raise ValueError("invalid subprocess inference engine configuration")
        executable = _validated_python_executable(python_executable)
        command = [
            str(executable), "-m", "vllm_apple.main_thread_process_worker",
            "--factory", factory,
            "--maximum-pending-requests", str(maximum_pending_requests),
        ]
        if config_path is not None:
            command.extend(("--config", str(config_path.expanduser().absolute())))
        self._maximum_pending = maximum_pending_requests
        self._shutdown_timeout = shutdown_timeout_seconds
        self._write_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending: dict[str, _Pending] = {}
        self._startup = threading.Event()
        self._closed = threading.Event()
        self._ready = False
        self._models: list[dict[str, Any]] = []
        self._startup_error: str | None = None
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self._reader = threading.Thread(
            target=self._read_responses,
            name="vllm-apple-process-engine-reader",
            daemon=True,
        )
        self._reader.start()
        if not self._startup.wait(startup_timeout_seconds):
            self.close()
            raise TimeoutError("subprocess inference engine startup timed out")
        if self._startup_error is not None:
            self.close()
            raise RuntimeError(
                f"subprocess inference engine startup failed: {self._startup_error}"
            )

    @property
    def ready(self) -> bool:
        return self._ready and not self._closed.is_set() and self._process.poll() is None

    def models(self) -> list[dict[str, Any]]:
        return [dict(model) for model in self._models]

    def chat_completions_with_request_context(
        self,
        request: dict[str, Any],
        _kernel_context: object | None,
        request_context: InferenceRequestContext,
    ) -> dict[str, Any]:
        request_context.raise_if_cancelled()
        if not self.ready:
            raise RuntimeError("subprocess inference engine is unavailable")
        identifier = uuid.uuid4().hex
        pending = _Pending()
        with self._pending_lock:
            if len(self._pending) >= self._maximum_pending:
                raise InferenceEngineBusy("inference process queue is full")
            self._pending[identifier] = pending
        try:
            self._write({
                "type": "request",
                "id": identifier,
                "request": request,
                "timeout_seconds": request_context.remaining_seconds,
            })
            while not pending.done.wait(min(0.05, request_context.remaining_seconds)):
                try:
                    request_context.raise_if_cancelled()
                except InferenceRequestCancelled as error:
                    raise InferenceRequestCancelled(
                        str(error), completion=pending.completion
                    ) from error
            if pending.error_code is not None:
                if pending.error_code == "engine_busy":
                    raise InferenceEngineBusy("inference process queue is full")
                if pending.error_code in {"request_cancelled", "request_timed_out"}:
                    raise InferenceRequestCancelled(pending.error_code)
                raise RuntimeError(f"subprocess inference failed: {pending.error_code}")
            if pending.result is None:
                raise RuntimeError("subprocess inference returned no result")
            return pending.result
        except InferenceRequestCancelled:
            # Sending cancel is not an acknowledgment of native completion.
            self._cancel(identifier)
            raise
        except (ValueError, TypeError):
            # Serialization/size rejection happens before anything is written.
            with self._pending_lock:
                removed = self._pending.pop(identifier, None)
            if removed is not None:
                removed.finish()
            raise

    def chat_completions(self, request: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("subprocess requests require an inference request context")

    def open_chat_stream(self, request: dict[str, Any]):
        raise RuntimeError("subprocess streaming is not enabled")

    def diagnostics(self, timeout_seconds: float = 5.0) -> dict[str, Any]:
        if not 0 < timeout_seconds <= 30 or not self.ready:
            raise ValueError("invalid subprocess diagnostics request")
        identifier = uuid.uuid4().hex
        pending = _Pending()
        with self._pending_lock:
            if len(self._pending) >= self._maximum_pending:
                raise InferenceEngineBusy("inference process queue is full")
            self._pending[identifier] = pending
        # Keep timed-out diagnostics bounded until the child replies.
        self._write({"type": "diagnostics", "id": identifier})
        if not pending.done.wait(timeout_seconds):
            self._cancel(identifier)
            raise TimeoutError("subprocess diagnostics timed out")
        if pending.error_code is not None or pending.result is None:
            raise RuntimeError("subprocess diagnostics failed")
        return pending.result

    def close(self) -> bool:
        if self._closed.is_set():
            return self._process.poll() is not None
        self._closed.set()
        self._ready = False
        try:
            self._write({"type": "shutdown"})
        except (BrokenPipeError, OSError, RuntimeError):
            pass
        try:
            self._process.wait(timeout=self._shutdown_timeout)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2)
        self._fail_pending("engine_closed")
        if threading.current_thread() is not self._reader:
            self._reader.join(timeout=2)
        if self._process.stdin is not None:
            self._process.stdin.close()
        if not self._reader.is_alive() and self._process.stdout is not None:
            self._process.stdout.close()
        return self._process.poll() is not None

    def _cancel(self, identifier: str) -> None:
        try:
            self._write({"type": "cancel", "id": identifier})
        except (BrokenPipeError, OSError, RuntimeError):
            pass

    def _write(self, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode() + b"\n"
        if len(encoded) > MAX_PROCESS_MESSAGE_BYTES:
            raise ValueError("subprocess inference request is too large")
        stream = self._process.stdin
        if stream is None:
            raise RuntimeError("subprocess inference stdin is unavailable")
        with self._write_lock:
            stream.write(encoded)
            stream.flush()

    def _read_responses(self) -> None:
        stream = self._process.stdout
        if stream is None:
            self._startup_error = "stdout_unavailable"
            self._startup.set()
            return
        try:
            while True:
                line = stream.readline(MAX_PROCESS_RESPONSE_BYTES + 1)
                if not line:
                    break
                if len(line) > MAX_PROCESS_RESPONSE_BYTES or not line.endswith(b"\n"):
                    self._startup_error = "response_too_large"
                    break
                try:
                    payload = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._startup_error = "invalid_response"
                    break
                if not isinstance(payload, dict):
                    self._startup_error = "invalid_response"
                    break
                if payload.get("type") == "ready":
                    models = payload.get("models")
                    if not isinstance(models, list) or len(models) > 64:
                        self._startup_error = "invalid_ready_response"
                    else:
                        self._models = [item for item in models if isinstance(item, dict)]
                        self._ready = True
                    self._startup.set()
                    continue
                identifier = payload.get("id")
                if not isinstance(identifier, str):
                    self._startup_error = "invalid_response"
                    break
                terminal_result = payload.get("type") == "result" and isinstance(payload.get("result"), dict)
                code = payload.get("code")
                terminal_error = (payload.get("type") == "error"
                                  and isinstance(code, str) and code in _TERMINAL_ERRORS)
                if not (terminal_result or terminal_error):
                    self._startup_error = "invalid_response"
                    break
                with self._pending_lock:
                    pending = self._pending.pop(identifier, None)
                if pending is None:
                    continue
                if terminal_result:
                    pending.result = payload["result"]
                else:
                    code = payload.get("code")
                    pending.error_code = code if isinstance(code, str) else "invalid_response"
                pending.finish()
                if self._closed.is_set():
                    break
        finally:
            if not self._startup.is_set():
                self._startup_error = self._startup_error or "worker_exited"
                self._startup.set()
            self._ready = False
            # A broken protocol/EOF cannot acknowledge native completion. Stop
            # and reap only this engine's owned process before releasing leases.
            try:
                self._process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                try:
                    self._process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    try:
                        self._process.kill()
                    except ProcessLookupError:
                        pass
                    self._process.wait()
            self._fail_pending("worker_exited")
            stream.close()

    def _fail_pending(self, code: str) -> None:
        with self._pending_lock:
            pending = tuple(self._pending.values())
            self._pending.clear()
        for item in pending:
            item.error_code = code
            item.finish()


def _validated_python_executable(path: Path) -> Path:
    """Validate an interpreter without resolving away its venv entry point."""
    candidate = path.expanduser().absolute()
    try:
        link_info = candidate.lstat()
        resolved = candidate.resolve(strict=True)
        target_info = resolved.stat()
    except OSError as error:
        raise ValueError("subprocess inference Python is unavailable") from error
    current_python = resolved == Path(sys.executable).resolve(strict=True)
    if (
        not (stat.S_ISREG(link_info.st_mode) or stat.S_ISLNK(link_info.st_mode))
        or (
            link_info.st_uid not in (os.getuid(), 0)
            and not (current_python and candidate == Path(sys.executable).absolute())
        )
        or not stat.S_ISREG(target_info.st_mode)
        or (not current_python and target_info.st_uid not in (os.getuid(), 0))
        or (not current_python and stat.S_IMODE(target_info.st_mode) & 0o022)
        or not os.access(resolved, os.X_OK)
    ):
        raise ValueError("subprocess inference Python is unsafe")
    return candidate
