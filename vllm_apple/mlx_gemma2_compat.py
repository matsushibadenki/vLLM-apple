"""Opt-in compatibility shim for a verified MLX-LM Gemma 2 implementation.

No upstream files are modified. The original attention computation is retained;
only the batch-aware shared-head mask gains the grouped-query axis.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import threading
from functools import wraps
from pathlib import Path
from typing import Any

GEMMA2_SOURCE_SHA256 = "64b0935b06fe2c4d5d4ed23a9cf62deb6218c55a88b9403a657afe9e2be8f251"
SERVER_SOURCE_SHA256 = "8514178d18ee7e5edd1db8b8077ab97079ec72d1db666b85c110fb33cdc55147"
REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")
_ACTIVE: dict[str, tuple[Any, Any]] = {}
_PENDING: dict[str, Any] = {}
_ACTIVE_LOCK = threading.Lock()
_REQUEST = threading.local()
MAX_TIMEOUT_MS = 600_000


class _RequestQueue:
    """Queue marker used only by the reviewed compatibility launcher."""

    def __init__(self, queue_type: type[Any]) -> None:
        self._queue = queue_type()
        self.cancelled = False

    def put(self, value: Any) -> None:
        self._queue.put(value)

    def get(self, *args: Any, **kwargs: Any) -> Any:
        return self._queue.get(*args, **kwargs)


def grouped_query_mask(mask: Any, repeats: int) -> Any:
    if repeats > 1 and mask is not None and getattr(mask, "ndim", None) == 4:
        batch, heads, query, key = mask.shape
        if heads != 1:
            raise ValueError("Gemma 2 compatibility requires a shared-head attention mask")
        return mask.reshape(batch, 1, 1, query, key)
    return mask


def install_gemma2_batch_mask_fix() -> bool:
    """Apply once before model loading; reject unreviewed versions/source files."""
    if importlib.metadata.version("mlx-lm") != "0.32.0":
        raise ValueError("Gemma 2 batch mask fix requires reviewed MLX-LM 0.32.0")
    from mlx_lm.models import gemma2

    if hashlib.sha256(Path(gemma2.__file__).read_bytes()).hexdigest() != GEMMA2_SOURCE_SHA256:
        raise ValueError("Gemma 2 source changed; review the compatibility fix before use")
    original = gemma2.Attention.__call__
    if getattr(original, "_vllm_apple_gemma2_mask_fix", False):
        return False

    @wraps(original)
    def corrected(self: Any, x: Any, mask: Any = None, cache: Any = None) -> Any:
        return original(self, x, mask=grouped_query_mask(mask, self.repeats), cache=cache)

    corrected._vllm_apple_gemma2_mask_fix = True
    gemma2.Attention.__call__ = corrected
    return True


def _register_context(request_id: str, context: Any, response_queue: Any) -> None:
    with _ACTIVE_LOCK:
        if request_id in _ACTIVE:
            context.stop()
            response_queue.put(None)
            raise ValueError("request ID is already active")
        _ACTIVE[request_id] = (context, response_queue)


def _register_pending(request_id: str, response_queue: Any) -> None:
    with _ACTIVE_LOCK:
        if request_id in _PENDING or request_id in _ACTIVE:
            raise ValueError("request ID is already active")
        _PENDING[request_id] = response_queue


def _activate_pending(request_id: str, context: Any, response_queue: Any) -> bool:
    with _ACTIVE_LOCK:
        pending = _PENDING.get(request_id)
        if pending is not response_queue:
            context.stop()
            response_queue.put(None)
            return False
        _PENDING.pop(request_id, None)
        if response_queue.cancelled:
            context.stop()
            response_queue.put(None)
            return False
        _ACTIVE[request_id] = (context, response_queue)
        return True


def _release_pending(request_id: str, response_queue: Any) -> None:
    with _ACTIVE_LOCK:
        if _PENDING.get(request_id) is response_queue:
            _PENDING.pop(request_id, None)


def _release_context(request_id: str, context: Any) -> None:
    with _ACTIVE_LOCK:
        active = _ACTIVE.get(request_id)
        if active is not None and active[0] is context:
            _ACTIVE.pop(request_id, None)


def cancel_request_state(request_id: str) -> str | None:
    if REQUEST_ID.fullmatch(request_id) is None:
        return None
    with _ACTIVE_LOCK:
        active = _ACTIVE.get(request_id)
        if active is not None:
            context, response_queue = active
            context.stop()
            response_queue.put(None)
            return "active"
        pending = _PENDING.get(request_id)
        if pending is not None:
            pending.cancelled = True
            return "queued"
    return None


def cancel_request(request_id: str) -> bool:
    return cancel_request_state(request_id) is not None


def install_cancel_api() -> type[Any]:
    """Install an opt-in cancellation bridge for the reviewed upstream server."""
    from mlx_lm import server

    if hashlib.sha256(Path(server.__file__).read_bytes()).hexdigest() != SERVER_SOURCE_SHA256:
        raise ValueError("MLX-LM server source changed; review the cancellation bridge")
    if getattr(server.ResponseGenerator.generate, "_vllm_apple_cancel_bridge", False):
        return server.APIHandler

    original_next_request = server.ResponseGenerator._next_request

    @wraps(original_next_request)
    def next_request(self: Any, timeout: Any = None) -> Any:
        while True:
            item = original_next_request(self, timeout)
            if item is None:
                return None
            response_queue = item[0]
            if not isinstance(response_queue, _RequestQueue) or not response_queue.cancelled:
                return item
            response_queue.put(RuntimeError("request cancelled before execution"))
            # Only the first read may block. Drain already queued cancelled
            # requests without extending the upstream scheduling timeout.
            timeout = None

    next_request._vllm_apple_cancel_bridge = True
    server.ResponseGenerator._next_request = next_request

    def generate(self: Any, request: Any, generation_args: Any,
                 progress_callback: Any = None) -> tuple[Any, Any]:
        response_queue = _RequestQueue(server.Queue)
        request_id = getattr(_REQUEST, "request_id", None)
        timeout_seconds = getattr(_REQUEST, "timeout_seconds", None)
        timer = None
        if request_id is not None:
            _register_pending(request_id, response_queue)
            if timeout_seconds is not None:
                timer = threading.Timer(
                    timeout_seconds, cancel_request_state, args=(request_id,))
                timer.daemon = True
                timer.start()
        self.requests.put((response_queue, request, generation_args))

        def responses() -> Any:
            while True:
                response = response_queue.get()
                if response is None:
                    break
                if isinstance(response, Exception):
                    raise response
                if isinstance(response, tuple):
                    if progress_callback is not None:
                        progress_callback(*response)
                    continue
                yield response

        try:
            context = response_queue.get()
            if isinstance(context, Exception):
                raise context
            if request_id is None:
                return context, responses()
            _activate_pending(request_id, context, response_queue)
        except BaseException:
            if timer is not None:
                timer.cancel()
            if request_id is not None:
                _release_pending(request_id, response_queue)
            raise

        def tracked_responses() -> Any:
            try:
                yield from responses()
            finally:
                if timer is not None:
                    timer.cancel()
                _release_context(request_id, context)

        return context, tracked_responses()

    generate._vllm_apple_cancel_bridge = True
    server.ResponseGenerator.generate = generate

    class CompatHandler(server.APIHandler):
        def handle_completion(self, request: Any, stop_words: list[str]) -> None:
            supplied = self.headers.get("X-VLLM-Apple-Request-ID")
            if supplied is not None and REQUEST_ID.fullmatch(supplied) is None:
                self._set_completion_headers(400)
                self.end_headers()
                self.wfile.write(b'{"error":"invalid request ID"}')
                return
            timeout_header = self.headers.get("X-VLLM-Apple-Timeout-Ms")
            timeout_seconds = None
            if timeout_header is not None:
                try:
                    timeout_ms = int(timeout_header)
                except ValueError:
                    timeout_ms = 0
                if supplied is None or not 1 <= timeout_ms <= MAX_TIMEOUT_MS:
                    self._set_completion_headers(400)
                    self.end_headers()
                    self.wfile.write(b'{"error":"invalid request timeout"}')
                    return
                timeout_seconds = timeout_ms / 1000
            _REQUEST.request_id = supplied
            _REQUEST.timeout_seconds = timeout_seconds
            try:
                super().handle_completion(request, stop_words)
            finally:
                _REQUEST.request_id = None
                _REQUEST.timeout_seconds = None

        def do_DELETE(self) -> None:
            prefix = "/vllm-apple/requests/"
            request_id = self.path.removeprefix(prefix) if self.path.startswith(prefix) else ""
            cancel_state = cancel_request_state(request_id)
            status = 202 if cancel_state is not None else 404
            payload = json.dumps({"request_id": request_id,
                                  "cancel_requested": cancel_state is not None,
                                  "request_state": cancel_state},
                                 separators=(",", ":")).encode()
            self._set_completion_headers(status)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()

    return CompatHandler


if __name__ == "__main__":
    # Explicit experimental entry point; use the upstream parser and lifecycle.
    # Never alter the installed package or implicitly promote managed serving.
    install_gemma2_batch_mask_fix()
    from mlx_lm import server

    handler = install_cancel_api()
    defaults = server._run_http_server.__defaults__
    if defaults is None or len(defaults) != 2:
        raise RuntimeError("reviewed MLX-LM HTTP server signature changed")
    # server.run currently does not forward its handler_class argument. Bind
    # the reviewed private HTTP helper directly instead.
    server._run_http_server.__defaults__ = (defaults[0], handler)
    server.main()
