from __future__ import annotations

import json
from dataclasses import astuple, dataclass

from .kernel_profile import PagedAttentionShape
from .metal_probe import MetalThreadConfiguration

KERNEL_TUNING_CONTEXT_VERSION = 1
MAX_KERNEL_TUNING_HEADER_BYTES = 4096
KERNEL_TUNING_ID_HEADER = "X-VLLM-Apple-Tuning-ID"
KERNEL_TUNING_CONTEXT_HEADER = "X-VLLM-Apple-Metal-Tuning"
KERNEL_TUNING_ACCEPTED_HEADER = "X-VLLM-Apple-Tuning-Accepted"


@dataclass(frozen=True, slots=True)
class PagedAttentionKernelSelection:
    shape: PagedAttentionShape
    configuration: MetalThreadConfiguration

    def to_dict(self) -> dict[str, object]:
        return {
            "shape": list(astuple(self.shape)),
            "threads": list(astuple(self.configuration)),
        }


@dataclass(frozen=True, slots=True)
class InferenceKernelContext:
    """Immutable tuning snapshot bound to one admitted inference request."""

    tuning_id: str
    profile_id: str
    paged_attention: tuple[PagedAttentionKernelSelection, ...]
    version: int = KERNEL_TUNING_CONTEXT_VERSION

    def __post_init__(self) -> None:
        if self.version != KERNEL_TUNING_CONTEXT_VERSION:
            raise ValueError("unsupported kernel tuning context version")
        identities = (self.tuning_id, self.profile_id)
        if any(
            len(value) != 24 or any(character not in "0123456789abcdef" for character in value)
            for value in identities
        ):
            raise ValueError("invalid kernel tuning context identity")
        if not 1 <= len(self.paged_attention) <= 16:
            raise ValueError("kernel tuning context requires 1 to 16 selections")

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "tuning_id": self.tuning_id,
            "profile_id": self.profile_id,
            "shape_fields": [
                "context_tokens",
                "batch_size",
                "query_heads",
                "kv_heads",
                "head_dimension",
                "block_tokens",
                "blocks_per_sequence",
                "kv_working_set_bytes",
            ],
            "thread_fields": ["score_width", "softmax_width", "output_width"],
            "paged_attention": [item.to_dict() for item in self.paged_attention],
        }

    def to_http_headers(self) -> dict[str, str]:
        encoded = json.dumps(
            self.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        if len(encoded.encode("ascii")) > MAX_KERNEL_TUNING_HEADER_BYTES:
            raise ValueError("kernel tuning context exceeds HTTP header limit")
        return {
            KERNEL_TUNING_ID_HEADER: self.tuning_id,
            KERNEL_TUNING_CONTEXT_HEADER: encoded,
        }
