from __future__ import annotations

import contextvars
import json
import threading
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from .kernel_context import (
    KERNEL_TUNING_ACCEPTED_HEADER,
    KERNEL_TUNING_CONTEXT_HEADER,
    KERNEL_TUNING_CONTEXT_VERSION,
    KERNEL_TUNING_ID_HEADER,
    MAX_KERNEL_TUNING_HEADER_BYTES,
    InferenceKernelContext,
    PagedAttentionKernelSelection,
)
from .kernel_profile import PagedAttentionShape
from .metal_probe import MetalThreadConfiguration

SHAPE_FIELDS = (
    "context_tokens",
    "batch_size",
    "query_heads",
    "kv_heads",
    "head_dimension",
    "block_tokens",
    "blocks_per_sequence",
    "kv_working_set_bytes",
)
THREAD_FIELDS = ("score_width", "softmax_width", "output_width")
_Result = TypeVar("_Result")


class PagedAttentionKernelInvoker(Protocol[_Result]):
    def __call__(
        self,
        shape: PagedAttentionShape,
        configuration: MetalThreadConfiguration | None,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> _Result: ...


def parse_kernel_tuning_headers(
    headers: Mapping[str, str],
) -> InferenceKernelContext | None:
    normalized = {key.lower(): value for key, value in headers.items()}
    tuning_id = normalized.get(KERNEL_TUNING_ID_HEADER.lower())
    encoded = normalized.get(KERNEL_TUNING_CONTEXT_HEADER.lower())
    if tuning_id is None and encoded is None:
        return None
    if tuning_id is None or encoded is None:
        raise ValueError("incomplete kernel tuning headers")
    try:
        encoded_bytes = encoded.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError("kernel tuning context must be ASCII") from error
    if len(encoded_bytes) > MAX_KERNEL_TUNING_HEADER_BYTES:
        raise ValueError("kernel tuning context exceeds HTTP header limit")
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ValueError("invalid kernel tuning context JSON") from error
    expected = {
        "version",
        "tuning_id",
        "profile_id",
        "shape_fields",
        "thread_fields",
        "paged_attention",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("invalid kernel tuning context fields")
    if payload["version"] != KERNEL_TUNING_CONTEXT_VERSION:
        raise ValueError("unsupported kernel tuning context version")
    if (
        not isinstance(payload["shape_fields"], list)
        or tuple(payload["shape_fields"]) != SHAPE_FIELDS
    ):
        raise ValueError("unsupported kernel shape field order")
    if (
        not isinstance(payload["thread_fields"], list)
        or tuple(payload["thread_fields"]) != THREAD_FIELDS
    ):
        raise ValueError("unsupported Metal thread field order")
    selections = payload["paged_attention"]
    if not isinstance(selections, list) or not 1 <= len(selections) <= 16:
        raise ValueError("invalid paged attention selection count")

    parsed: list[PagedAttentionKernelSelection] = []
    seen_shapes: set[PagedAttentionShape] = set()
    try:
        for selection in selections:
            if not isinstance(selection, dict) or set(selection) != {"shape", "threads"}:
                raise ValueError("invalid paged attention selection fields")
            shape_values = selection["shape"]
            thread_values = selection["threads"]
            if not isinstance(shape_values, list) or len(shape_values) != len(SHAPE_FIELDS):
                raise ValueError("invalid paged attention shape")
            if not isinstance(thread_values, list) or len(thread_values) != len(THREAD_FIELDS):
                raise ValueError("invalid Metal thread configuration")
            shape = PagedAttentionShape(*shape_values)
            if shape in seen_shapes:
                raise ValueError("duplicate paged attention shape")
            seen_shapes.add(shape)
            parsed.append(
                PagedAttentionKernelSelection(shape, MetalThreadConfiguration(*thread_values))
            )
        context = InferenceKernelContext(
            tuning_id=payload["tuning_id"],
            profile_id=payload["profile_id"],
            paged_attention=tuple(parsed),
            version=payload["version"],
        )
    except (TypeError, ValueError) as error:
        raise ValueError("invalid kernel tuning context values") from error
    if tuning_id != context.tuning_id:
        raise ValueError("kernel tuning ID header mismatch")
    return context


@dataclass(frozen=True, slots=True)
class BackendTuningSnapshot:
    accepted_contexts: int
    rejected_contexts: int
    shape_hits: int
    shape_misses: int

    def to_dict(self) -> dict[str, int]:
        return {
            "accepted_contexts": self.accepted_contexts,
            "rejected_contexts": self.rejected_contexts,
            "shape_hits": self.shape_hits,
            "shape_misses": self.shape_misses,
        }


class BackendKernelTuningAdapter:
    """Request-local bridge from HTTP tuning metadata to a kernel invocation."""

    def __init__(self) -> None:
        self._context: contextvars.ContextVar[InferenceKernelContext | None] = (
            contextvars.ContextVar(f"vllm_apple_kernel_tuning_{id(self)}", default=None)
        )
        self._applied_tuning_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            f"vllm_apple_applied_tuning_{id(self)}", default=None
        )
        self._lock = threading.Lock()
        self._accepted_contexts = 0
        self._rejected_contexts = 0
        self._shape_hits = 0
        self._shape_misses = 0

    @contextmanager
    def bind_headers(self, headers: Mapping[str, str]) -> Iterable[InferenceKernelContext | None]:
        try:
            context = parse_kernel_tuning_headers(headers)
        except ValueError:
            context = None
            with self._lock:
                self._rejected_contexts += 1
        else:
            if context is not None:
                with self._lock:
                    self._accepted_contexts += 1
        token = self._context.set(context)
        applied_token = self._applied_tuning_id.set(None)
        try:
            yield context
        finally:
            self._applied_tuning_id.reset(applied_token)
            self._context.reset(token)

    def configuration_for(self, shape: PagedAttentionShape) -> MetalThreadConfiguration | None:
        context = self._context.get()
        if context is not None:
            for selection in context.paged_attention:
                if selection.shape == shape:
                    with self._lock:
                        self._shape_hits += 1
                    self._applied_tuning_id.set(context.tuning_id)
                    return selection.configuration
        with self._lock:
            self._shape_misses += 1
        return None

    def applied_tuning_id(self) -> str | None:
        return self._applied_tuning_id.get()

    def invoke_paged_attention(
        self,
        invoker: PagedAttentionKernelInvoker[_Result],
        shape: PagedAttentionShape,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> _Result:
        return invoker(shape, self.configuration_for(shape), *args, **kwargs)

    def snapshot(self) -> BackendTuningSnapshot:
        with self._lock:
            return BackendTuningSnapshot(
                self._accepted_contexts,
                self._rejected_contexts,
                self._shape_hits,
                self._shape_misses,
            )


class KernelTuningASGIMiddleware:
    """Dependency-free ASGI middleware suitable for a vLLM FastAPI application."""

    def __init__(self, app: Callable[..., Any], adapter: BackendKernelTuningAdapter) -> None:
        self.app = app
        self.adapter = adapter

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[..., Any],
        send: Callable[..., Any],
    ) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = _parse_asgi_headers(scope.get("headers", ()))
        with self.adapter.bind_headers(headers) as context:
            if context is None:
                await self.app(scope, receive, send)
                return

            async def send_with_ack(message: dict[str, Any]) -> None:
                if (
                    message.get("type") == "http.response.start"
                    and self.adapter.applied_tuning_id() == context.tuning_id
                ):
                    response_headers = list(message.get("headers", ()))
                    response_headers.append(
                        (
                            KERNEL_TUNING_ACCEPTED_HEADER.lower().encode("ascii"),
                            context.tuning_id.encode("ascii"),
                        )
                    )
                    message = {**message, "headers": response_headers}
                await send(message)

            await self.app(scope, receive, send_with_ack)


def _parse_asgi_headers(raw_headers: Iterable[tuple[bytes, bytes]]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_name, raw_value in raw_headers:
        try:
            name = raw_name.decode("ascii").lower()
            value = raw_value.decode("ascii")
        except UnicodeDecodeError:
            continue
        if name in {
            KERNEL_TUNING_ID_HEADER.lower(),
            KERNEL_TUNING_CONTEXT_HEADER.lower(),
        }:
            if name in parsed:
                # Multiple security-sensitive headers are always invalid.
                return {KERNEL_TUNING_ID_HEADER: "duplicate"}
            parsed[name] = value
    return parsed
