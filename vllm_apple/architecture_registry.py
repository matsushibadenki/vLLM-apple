"""Metadata-only architecture inventory; never certifies backend execution."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .model import ModelInspectionError, _bounded_config

REGISTRY_VERSION = 1
MAX_LAYERS = 512


@dataclass(frozen=True, slots=True)
class ArchitectureRecipe:
    family: str
    features: tuple[str, ...]
    source: str
    moe: bool = False
    alternating_window: bool = False


# Exact model_type matches only. These are structural recipes, not backend support.
ARCHITECTURE_RECIPES = MappingProxyType({
    "llama": ArchitectureRecipe(
        "dense_transformer", ("rope", "rms_norm", "swiglu"),
        "https://huggingface.co/docs/transformers/model_doc/llama",
    ),
    "qwen2": ArchitectureRecipe(
        "dense_transformer", ("rope", "rms_norm", "swiglu", "qkv_bias"),
        "https://huggingface.co/docs/transformers/model_doc/qwen2",
    ),
    "qwen3": ArchitectureRecipe(
        "dense_transformer", ("rope", "rms_norm", "swiglu", "qk_norm"),
        "https://huggingface.co/docs/transformers/model_doc/qwen3",
    ),
    "mixtral": ArchitectureRecipe(
        "moe_transformer", ("rope", "rms_norm", "swiglu", "routed_experts"),
        "https://huggingface.co/docs/transformers/model_doc/mixtral", moe=True,
    ),
    "gemma2": ArchitectureRecipe(
        "windowed_transformer", ("rope", "rms_norm", "geglu", "sandwich_norm",
                                 "embedding_scale", "attention_softcap", "logit_softcap"),
        "https://huggingface.co/docs/transformers/model_doc/gemma2",
        alternating_window=True,
    ),
})


def _positive(config: dict[str, Any], key: str, maximum: int = 16_777_216) -> int:
    value = config.get(key)
    if type(value) is not int or not 1 <= value <= maximum:
        raise ModelInspectionError(f"architecture_invalid_field:{key}")
    return value


def _identifier(value: object) -> str:
    if value is None:
        return "unknown"
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ModelInspectionError("architecture_invalid_field:model_type")
    return value


def _layer_types(config: dict[str, Any], recipe: ArchitectureRecipe, count: int) -> list[str]:
    explicit = config.get("layer_types")
    window = config.get("sliding_window")
    if window is not None:
        _positive(config, "sliding_window")
    enabled = config.get("use_sliding_window", False)
    if type(enabled) is not bool:
        raise ModelInspectionError("architecture_invalid_field:use_sliding_window")
    if explicit is not None:
        if (
            not isinstance(explicit, list) or len(explicit) != count
            or any(item not in ("full_attention", "sliding_attention") for item in explicit)
        ):
            raise ModelInspectionError("architecture_invalid_field:layer_types")
        result = explicit.copy()
    elif recipe.alternating_window:
        result = ["sliding_attention" if i % 2 == 0 else "full_attention" for i in range(count)]
    elif recipe.moe and window is not None:
        result = ["sliding_attention"] * count
    elif enabled:
        # Qwen's max_window_layers is the exclusive upper bound of windowed layers.
        boundary = config.get("max_window_layers")
        if type(boundary) is not int or not 0 <= boundary <= count:
            raise ModelInspectionError("architecture_invalid_field:max_window_layers")
        result = ["sliding_attention" if i < boundary else "full_attention" for i in range(count)]
    elif window is not None:
        if "use_sliding_window" not in config:
            raise ModelInspectionError("architecture_ambiguous_window_policy")
        result = ["full_attention"] * count
    else:
        result = ["full_attention"] * count
    if "sliding_attention" in result and window is None:
        raise ModelInspectionError("architecture_invalid_field:sliding_window")
    return result


def inspect_architecture(path: str | Path) -> dict[str, object]:
    """Read bounded local config only; never import a model or inspect weights.

    A recognized recipe describes metadata requirements. All execution stages stay
    unverified, even when its structure is valid. Unknown wrappers are not silently
    treated as supported text-only models.
    """
    root = Path(path).expanduser()
    config = _bounded_config(root / "config.json" if root.is_dir() else root)
    return describe_architecture(config)


def describe_architecture(config: dict[str, Any]) -> dict[str, object]:
    """Describe already inspected metadata without reading the model a second time."""
    try:
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ModelInspectionError("architecture_invalid_json_values") from error
    model_type = _identifier(config.get("model_type"))
    recipe = ARCHITECTURE_RECIPES.get(model_type)
    report: dict[str, object] = {
        "schema_version": 1,
        "registry_version": REGISTRY_VERSION,
        "model_type": model_type,
        "config_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "recognition": "recognized" if recipe else "unknown",
        "structure_status": "unverified",
        "family": recipe.family if recipe else None,
        "required_features": [],
        "layers": [],
        "state_layout": [],
        "backend": {"name": None, "build": None, "compatibility": "unverified"},
        "qualification": dict.fromkeys(
            ("loadable", "correct", "service", "performance"), "unverified"
        ),
        "artifact_status": "unverified",
        "issues": [],
        "source": recipe.source if recipe else None,
    }
    if recipe is None:
        report["issues"] = ["architecture_unknown"]
        return report
    try:
        # A wrapper requires its own recipe including modality and projection.
        if "text_config" in config:
            raise ModelInspectionError("architecture_wrapper_unverified")
        count = _positive(config, "num_hidden_layers", MAX_LAYERS)
        heads = _positive(config, "num_attention_heads", 65_536)
        kv_heads = config.get("num_key_value_heads", heads)
        if type(kv_heads) is not int or not 1 <= kv_heads <= heads or heads % kv_heads:
            raise ModelInspectionError("architecture_invalid_field:num_key_value_heads")
        hidden = _positive(config, "hidden_size")
        if "head_dim" in config:
            head_dim = _positive(config, "head_dim", 65_536)
        else:
            if hidden % heads:
                raise ModelInspectionError("architecture_invalid_field:hidden_size")
            head_dim = hidden // heads
        kind = "mha" if kv_heads == heads else "mqa" if kv_heads == 1 else "gqa"
        layer_types = _layer_types(config, recipe, count)
        features = set(recipe.features) | {kind}
        rope = config.get("rope_parameters", config.get("rope_scaling"))
        if rope is not None:
            if not isinstance(rope, dict):
                raise ModelInspectionError("architecture_invalid_field:rope_scaling")
            rope_type = rope.get("rope_type", rope.get("type"))
            if not isinstance(rope_type, str) or rope_type not in {
                "default", "linear", "dynamic", "yarn", "longrope", "llama3"
            }:
                raise ModelInspectionError("architecture_rope_variant_unverified")
            features.add(f"rope_{rope_type}")
        experts = active_experts = None
        if recipe.moe:
            experts = _positive(config, "num_local_experts", 65_536)
            active_experts = _positive(config, "num_experts_per_tok", experts)
        layers = []
        states = []
        for index, layer_type in enumerate(layer_types):
            window = config["sliding_window"] if layer_type == "sliding_attention" else None
            features.add(layer_type)
            layers.append({
                "index": index, "attention": kind, "pattern": layer_type,
                "query_heads": heads, "kv_heads": kv_heads, "head_dim": head_dim,
                "window_tokens": window, "ffn": "moe" if recipe.moe else "dense",
                "experts": experts, "active_experts": active_experts,
            })
            states.append({
                "owner_layer": index, "kind": "window_kv" if window else "append_kv",
                "window_tokens": window, "dtype": "unverified",
                "allocation_bytes": None, "retention_verified": False,
            })
        report.update({
            "structure_status": "described",
            "required_features": sorted(features), "layers": layers, "state_layout": states,
        })
    except ModelInspectionError as error:
        report["issues"] = [str(error)]
    return report
