"""Single-owner-thread lifecycle for thread-affine native inference engines."""
from __future__ import annotations

import math
import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
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
    completion: Future[None] = field(default_factory=Future)

    def finish(self) -> None:
        self.done.set()
        self.completion.set_result(None)


class _CommandQueue(queue.Queue):
    """Reclaim cancelled waiting commands, preserving FIFO and active ownership."""

    def discard_cancelled(self) -> None:
        with self.mutex:
            removed = []
            for _ in range(len(self.queue)):
                command = self.queue.popleft()
                if command is None:
                    self.queue.append(command)
                    continue
                error = None
                if command.cancelled.is_set():
                    error = InferenceRequestCancelled("inference request cancelled while queued")
                else:
                    try:
                        command.request_context.raise_if_cancelled()
                    except InferenceRequestCancelled as cancelled:
                        # Do not retain this queue frame through the exception
                        # traceback after removing the abandoned request.
                        error = InferenceRequestCancelled(str(cancelled))
                if error is None:
                    self.queue.append(command)
                    continue
                command.cancelled.set()
                command.error = error
                removed.append(command)
            if removed:
                self.unfinished_tasks -= len(removed)
                self.not_full.notify_all()
                if not self.unfinished_tasks:
                    self.all_tasks_done.notify_all()
        for command in removed:
            command.finish()


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
        self._commands: _CommandQueue = _CommandQueue(
            maximum_pending_requests
        )
        self._startup = threading.Event()
        self._closed = threading.Event()
        self._state_lock = threading.Lock()
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
            self._request_close()
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
        command = _Command(request, kernel_context, request_context)
        self._commands.discard_cancelled()
        with self._state_lock:
            if self._closed.is_set():
                raise RuntimeError("thread-affine engine is closed")
            try:
                self._commands.put_nowait(command)
            except queue.Full as error:
                raise InferenceEngineBusy("inference engine queue is full") from error
        while not command.done.wait(min(0.05, request_context.remaining_seconds)):
            try:
                request_context.raise_if_cancelled()
            except InferenceRequestCancelled as error:
                command.cancelled.set()
                self._commands.discard_cancelled()
                raise InferenceRequestCancelled(
                    str(error), completion=command.completion
                ) from error
        if command.error is not None:
            raise command.error
        assert command.result is not None
        return command.result

    def chat_completions(self, request: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("managed requests require an inference request context")

    def open_chat_stream(self, request: dict[str, Any]):
        raise RuntimeError("thread-affine streaming is not enabled")

    def _request_close(self) -> None:
        pending = []
        with self._state_lock:
            if self._closed.is_set():
                return
            self._closed.set()
            # Reject queued work before signalling the owner. Never wait for a
            # free queue slot while the delegate may still be executing.
            while True:
                try:
                    command = self._commands.get_nowait()
                except queue.Empty:
                    break
                if command is not None:
                    command.error = InferenceRequestCancelled("inference engine is closing")
                    pending.append(command)
            self._commands.put_nowait(None)
        for command in pending:
            command.finish()

    def close(self, *, timeout_seconds: float = 10.0) -> bool:
        """Request shutdown once; report success only after the owner exits.

        An active delegate retains ownership until its call returns. A timed-out
        close can be retried, and an owner callback can request close without
        attempting to join itself.
        """
        if not math.isfinite(timeout_seconds) or not 0 <= timeout_seconds <= 1800:
            raise ValueError("close timeout must be finite and between 0 and 1800 seconds")
        self._request_close()
        if threading.get_ident() == self._owner_ident:
            return False
        self._thread.join(timeout=timeout_seconds)
        return not self._thread.is_alive()

    def _run(self) -> None:
        self._owner_ident = threading.get_ident()
        try:
            try:
                self._delegate = self._factory()
                if self._closed.is_set():
                    return
                self._ready = bool(getattr(self._delegate, "ready", True))
                models = getattr(self._delegate, "models", None)
                self._models = list(models()) if callable(models) else []
            except BaseException as error:
                self._startup_error = error
                return
            finally:
                self._startup.set()
            while True:
                command = self._commands.get()
                if command is None:
                    return
                if command.cancelled.is_set() or self._closed.is_set():
                    command.error = InferenceRequestCancelled(
                        "inference request cancelled while queued"
                    )
                    command.finish()
                    continue
                try:
                    command.request_context.raise_if_cancelled()
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
                    command.finish()
        finally:
            self._ready = False
            close = getattr(self._delegate, "close", None)
            if callable(close):
                close()
