from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from .model import InspectedModel, ModelInspectionError

KERNEL_SHAPE_PROFILE_SCHEMA_VERSION = 1
MAX_KERNEL_SHAPES = 16
DEFAULT_CONTEXT_TIERS = (128, 1024, 4096, 16384)


@dataclass(frozen=True, slots=True)
class PagedAttentionShape:
    context_tokens: int
    batch_size: int
    query_heads: int
    kv_heads: int
    head_dimension: int
    block_tokens: int
    blocks_per_sequence: int
    kv_working_set_bytes: int

    def __post_init__(self) -> None:
        values = (
            self.context_tokens,
            self.batch_size,
            self.query_heads,
            self.kv_heads,
            self.head_dimension,
            self.block_tokens,
            self.blocks_per_sequence,
            self.kv_working_set_bytes,
        )
        if any(isinstance(value, bool) or value <= 0 for value in values):
            raise ValueError("kernel shape values must be positive integers")
        if self.query_heads % self.kv_heads:
            raise ValueError("query heads must be divisible by KV heads")


@dataclass(frozen=True, slots=True)
class ModelKernelShapeProfile:
    schema_version: int
    profile_id: str
    model_id: str
    architecture: str
    layers: int
    kv_dtype_bytes: int
    shapes: tuple[PagedAttentionShape, ...]

    def __post_init__(self) -> None:
        if self.schema_version != KERNEL_SHAPE_PROFILE_SCHEMA_VERSION:
            raise ValueError("unsupported kernel shape profile schema")
        if len(self.profile_id) != 24:
            raise ValueError("profile ID must contain 24 hexadecimal characters")
        if any(character not in "0123456789abcdef" for character in self.profile_id):
            raise ValueError("profile ID must be lowercase hexadecimal")
        if not 1 <= len(self.model_id) <= 512 or not 1 <= len(self.architecture) <= 128:
            raise ValueError("model identity is outside its bound")
        if self.layers <= 0 or self.kv_dtype_bytes not in (1, 2, 4):
            raise ValueError("invalid model kernel metadata")
        if not 1 <= len(self.shapes) <= MAX_KERNEL_SHAPES:
            raise ValueError("shape count must be between 1 and 16")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "model_id": self.model_id,
            "architecture": self.architecture,
            "layers": self.layers,
            "kv_dtype_bytes": self.kv_dtype_bytes,
            "shapes": [asdict(shape) for shape in self.shapes],
        }


def build_model_kernel_shape_profile(
    model: InspectedModel,
    *,
    context_tiers: Iterable[int] = DEFAULT_CONTEXT_TIERS,
    block_tokens: int = 16,
) -> ModelKernelShapeProfile:
    if isinstance(block_tokens, bool) or not 1 <= block_tokens <= 1024:
        raise ValueError("block_tokens must be between 1 and 1024")
    config = model.config
    layers = _positive_int(config, "num_hidden_layers", "n_layer")
    query_heads = _positive_int(config, "num_attention_heads", "n_head")
    kv_heads = config.get("num_key_value_heads", query_heads)
    if isinstance(kv_heads, bool) or not isinstance(kv_heads, int) or kv_heads <= 0:
        raise ModelInspectionError("num_key_value_heads must be positive")
    if query_heads % kv_heads:
        raise ModelInspectionError("num_attention_heads must be divisible by KV heads")
    head_dimension = config.get("head_dim")
    if isinstance(head_dimension, bool) or not isinstance(head_dimension, int) or head_dimension <= 0:
        hidden_size = _positive_int(config, "hidden_size", "n_embd", "d_model")
        if hidden_size % query_heads:
            raise ModelInspectionError("hidden_size is not divisible by num_attention_heads")
        head_dimension = hidden_size // query_heads

    contexts = _bounded_contexts(context_tiers, model.memory_spec.model_max_context)
    shapes = tuple(
        PagedAttentionShape(
            context_tokens=context,
            batch_size=1,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dimension=head_dimension,
            block_tokens=block_tokens,
            blocks_per_sequence=(context + block_tokens - 1) // block_tokens,
            kv_working_set_bytes=context
            * 2
            * kv_heads
            * head_dimension
            * model.kv_dtype_bytes,
        )
        for context in contexts
    )
    architecture = _architecture(config)
    identity = {
        "schema_version": KERNEL_SHAPE_PROFILE_SCHEMA_VERSION,
        "model_id": model.model_id,
        "architecture": architecture,
        "layers": layers,
        "kv_dtype_bytes": model.kv_dtype_bytes,
        "shapes": [asdict(shape) for shape in shapes],
    }
    profile_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    return ModelKernelShapeProfile(
        schema_version=KERNEL_SHAPE_PROFILE_SCHEMA_VERSION,
        profile_id=profile_id,
        model_id=model.model_id,
        architecture=architecture,
        layers=layers,
        kv_dtype_bytes=model.kv_dtype_bytes,
        shapes=shapes,
    )


def _positive_int(config: dict[str, object], *keys: str) -> int:
    for key in keys:
        value = config.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    raise ModelInspectionError(f"missing positive model config value: {' or '.join(keys)}")


def _bounded_contexts(tiers: Iterable[int], model_limit: int | None) -> tuple[int, ...]:
    values: set[int] = set()
    for tier in tiers:
        if isinstance(tier, bool) or not isinstance(tier, int) or tier <= 0:
            raise ValueError("context tiers must be positive integers")
        values.add(min(tier, model_limit) if model_limit is not None else tier)
        if len(values) > MAX_KERNEL_SHAPES:
            raise ValueError("context tier count exceeds 16")
    if not values:
        raise ValueError("at least one context tier is required")
    return tuple(sorted(values))


def _architecture(config: dict[str, object]) -> str:
    architectures = config.get("architectures")
    if isinstance(architectures, list) and architectures:
        value = architectures[0]
        if isinstance(value, str) and 1 <= len(value) <= 128:
            return value
    model_type = config.get("model_type")
    if isinstance(model_type, str) and 1 <= len(model_type) <= 128:
        return model_type
    return "unknown"
