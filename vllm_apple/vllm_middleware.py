from __future__ import annotations

import json
from typing import Any

from .backend_tuning import (
    BackendKernelTuningAdapter,
    KernelTuningASGIMiddleware,
    PagedAttentionKernelInvoker,
)
from .kernel_profile import PagedAttentionShape

# One process-wide adapter intentionally joins the ASGI request path and the
# in-process vLLM-Metal Paged Attention call site.
backend_kernel_tuning = BackendKernelTuningAdapter()


class VLLMAppleKernelTuningMiddleware(KernelTuningASGIMiddleware):
    def __init__(self, app: Any) -> None:
        super().__init__(app, backend_kernel_tuning)

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") == "http" and scope.get("path") == "/v1/vllm-apple/memory":
            try:
                payload = json.dumps(mlx_memory_metrics(), separators=(",", ":")).encode()
            except (ImportError, AttributeError, RuntimeError):
                payload = b'{"error":"mlx_memory_metrics_unavailable"}'
                status = 503
            else:
                status = 200
            await send(
                {
                    "type": "http.response.start",
                    "status": status,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(payload)).encode("ascii")),
                        (b"cache-control", b"no-store"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": payload})
            return
        await super().__call__(scope, receive, send)


def mlx_memory_metrics() -> dict[str, int]:
    import mlx.core as mx

    provider = mx
    if not hasattr(provider, "get_active_memory") and hasattr(mx, "metal"):
        provider = mx.metal
    values = {
        "active_bytes": provider.get_active_memory(),
        "cache_bytes": provider.get_cache_memory(),
        "peak_bytes": provider.get_peak_memory(),
    }
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values.values()):
        raise RuntimeError("MLX returned invalid memory metrics")
    return values


def invoke_paged_attention_kernel(
    invoker: PagedAttentionKernelInvoker[Any],
    shape: PagedAttentionShape,
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Stable vLLM-Metal call-site hook for request-local tuned dispatch."""
    return backend_kernel_tuning.invoke_paged_attention(invoker, shape, *args, **kwargs)


def backend_tuning_metrics() -> dict[str, int]:
    return backend_kernel_tuning.snapshot().to_dict()
