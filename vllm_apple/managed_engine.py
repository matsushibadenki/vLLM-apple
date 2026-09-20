"""Single-owner-thread lifecycle for thread-affine native inference engines."""
from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .inference_request import (
    InferenceEngineBusy,
    InferenceRequestCancelled,
    InferenceRequestContext,
)


@dataclass(slots=True)
class _Command:
    request: dict[str, Any]
    kernel_context: object | None
    request_context: InferenceRequestContext
    done: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None
    error: BaseException | None = None


class ThreadAffineInferenceEngine:
    """Create, invoke, and close a native engine on one bounded owner thread."""

    def __init__(
        self,
        factory: Callable[[], object],
        *,
        maximum_pending_requests: int = 8,
        startup_timeout_seconds: float = 600.0,
    ) -> None:
        if (
            not callable(factory)
            or type(maximum_pending_requests) is not int
            or not 1 <= maximum_pending_requests <= 64
            or not 0 < startup_timeout_seconds <= 1800
        ):
            raise ValueError("invalid thread-affine engine configuration")
        self._factory = factory
        self._commands: queue.Queue[_Command | None] = queue.Queue(
            maximum_pending_requests
        )
        self._startup = threading.Event()
        self._closed = threading.Event()
        self._startup_error: BaseException | None = None
        self._delegate: object | None = None
        self._ready = False
        self._models: list[dict[str, Any]] = []
        self._owner_ident: int | None = None
        self._thread = threading.Thread(
            target=self._run, name="vllm-apple-model-owner", daemon=True
        )
        self._thread.start()
        if not self._startup.wait(startup_timeout_seconds):
            raise TimeoutError("thread-affine engine startup timed out")
        if self._startup_error is not None:
            raise RuntimeError("thread-affine engine startup failed") from self._startup_error

    @property
    def ready(self) -> bool:
        return self._ready and not self._closed.is_set()

    @property
    def owner_thread_ident(self) -> int | None:
        return self._owner_ident

    def models(self) -> list[dict[str, Any]]:
        return [dict(model) for model in self._models]

    def chat_completions_with_request_context(
        self,
        request: dict[str, Any],
        kernel_context: object | None,
        request_context: InferenceRequestContext,
    ) -> dict[str, Any]:
        request_context.raise_if_cancelled()
        if self._closed.is_set():
            raise RuntimeError("thread-affine engine is closed")
        command = _Command(request, kernel_context, request_context)
        try:
            self._commands.put_nowait(command)
        except queue.Full as error:
            raise InferenceEngineBusy("inference engine queue is full") from error
        while not command.done.wait(min(0.05, request_context.remaining_seconds)):
            try:
                request_context.raise_if_cancelled()
            except InferenceRequestCancelled:
                command.cancelled.set()
                raise
        if command.error is not None:
            raise command.error
        assert command.result is not None
        return command.result

    def chat_completions(self, request: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("managed requests require an inference request context")

    def open_chat_stream(self, request: dict[str, Any]):
        raise RuntimeError("thread-affine streaming is not enabled")

    def close(self) -> bool:
        if self._closed.is_set():
            return True
        self._closed.set()
        self._commands.put(None)
        self._thread.join(timeout=10)
        return not self._thread.is_alive()

    def _run(self) -> None:
        self._owner_ident = threading.get_ident()
        try:
            self._delegate = self._factory()
            self._ready = bool(getattr(self._delegate, "ready", True))
            models = getattr(self._delegate, "models", None)
            self._models = list(models()) if callable(models) else []
        except BaseException as error:
            self._startup_error = error
            self._startup.set()
            return
        self._startup.set()
        try:
            while True:
                command = self._commands.get()
                if command is None:
                    return
                if command.cancelled.is_set():
                    command.error = InferenceRequestCancelled(
                        "inference request cancelled while queued"
                    )
                    command.done.set()
                    continue
                try:
                    method = getattr(
                        self._delegate, "chat_completions_with_request_context"
                    )
                    command.result = method(
                        command.request,
                        command.kernel_context,
                        command.request_context,
                    )
                except BaseException as error:
                    command.error = error
                finally:
                    command.done.set()
        finally:
            close = getattr(self._delegate, "close", None)
            if callable(close):
                close()
