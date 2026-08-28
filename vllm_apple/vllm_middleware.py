from __future__ import annotations

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
