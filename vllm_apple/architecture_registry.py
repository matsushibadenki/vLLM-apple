"""Metadata-only architecture inventory; never certifies backend execution."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .model import ModelInspectionError, _bounded_config

REGISTRY_VERSION = 2
MAX_LAYERS = 512
MAX_ARTIFACT_FILES = 4096
MAX_TOKENIZER_BYTES = 256 * 1024 * 1024
_WEIGHT_FORMATS = MappingProxyType({
    ".safetensors": "safetensors",
    ".gguf": "gguf",
    ".bin": "pytorch_bin",
})
_TOKENIZER_FILES = frozenset({
    "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
    "special_tokens_map.json", "vocab.json", "merges.txt",
})


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
    report = describe_architecture(config)
    report["artifact"] = _inspect_artifact_inventory(root) if root.is_dir() else _empty_artifact()
    return report


def _empty_artifact() -> dict[str, object]:
    return {
        "inventory_status": "unavailable",
        "weight_formats": [],
        "weight_file_count": 0,
        "weight_bytes": 0,
        "metadata_manifest_sha256": None,
        "tokenizer_files": [],
        "tokenizer_bytes": 0,
        "tokenizer_manifest_sha256": None,
    }


def _hash_regular_file(path: Path, *, maximum_bytes: int) -> tuple[int, str]:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise ModelInspectionError("architecture_tokenizer_inventory_unreadable") from error
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise ModelInspectionError("architecture_tokenizer_inventory_too_large")
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ModelInspectionError("architecture_tokenizer_inventory_changed")
            digest.update(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns
        ):
            raise ModelInspectionError("architecture_tokenizer_inventory_changed")
        return before.st_size, digest.hexdigest()
    finally:
        os.close(descriptor)


def _inspect_artifact_inventory(root: Path) -> dict[str, object]:
    """Inventory bounded, immediate metadata without claiming artifact integrity."""
    try:
        entries = sorted(root.iterdir(), key=lambda item: item.name)
    except OSError as error:
        raise ModelInspectionError("architecture_artifact_inventory_unreadable") from error
    if len(entries) > MAX_ARTIFACT_FILES:
        raise ModelInspectionError("architecture_artifact_inventory_too_large")
    weights: list[tuple[str, int, str]] = []
    tokenizer_files: list[str] = []
    tokenizer_manifest: list[dict[str, object]] = []
    tokenizer_bytes = 0
    for entry in entries:
        try:
            if entry.is_symlink() or not entry.is_file():
                continue
            size = entry.stat().st_size
        except OSError as error:
            raise ModelInspectionError("architecture_artifact_inventory_unreadable") from error
        weight_format = _WEIGHT_FORMATS.get(entry.suffix.lower())
        if weight_format is not None:
            weights.append((entry.name, size, weight_format))
        if entry.name in _TOKENIZER_FILES:
            tokenizer_files.append(entry.name)
            remaining = MAX_TOKENIZER_BYTES - tokenizer_bytes
            size, content_sha256 = _hash_regular_file(entry, maximum_bytes=remaining)
            tokenizer_bytes += size
            tokenizer_manifest.append({
                "name": entry.name, "bytes": size, "sha256": content_sha256,
            })
    manifest = [
        {"name": name, "bytes": size, "format": weight_format}
        for name, size, weight_format in weights
    ]
    digest = None
    if manifest:
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(encoded).hexdigest()
    tokenizer_digest = None
    if tokenizer_manifest:
        encoded = json.dumps(
            tokenizer_manifest, sort_keys=True, separators=(",", ":")
        ).encode()
        tokenizer_digest = hashlib.sha256(encoded).hexdigest()
    return {
        "inventory_status": "described" if weights else "unavailable",
        "weight_formats": sorted({item[2] for item in weights}),
        "weight_file_count": len(weights),
        "weight_bytes": sum(item[1] for item in weights),
        "metadata_manifest_sha256": digest,
        "tokenizer_files": tokenizer_files,
        "tokenizer_bytes": tokenizer_bytes,
        "tokenizer_manifest_sha256": tokenizer_digest,
    }


def attach_backend_capabilities(
    report: dict[str, object], *, name: str, build_sha256: str,
    versions: dict[str, str | None], declared_features: tuple[str, ...],
    probe_compatible: bool, issues: tuple[str, ...],
) -> dict[str, object]:
    """Attach probed build metadata without promoting execution qualification."""
    if name not in {"mlx_lm", "vllm_metal"}:
        raise ValueError("unsupported architecture backend")
    if len(build_sha256) != 64 or any(c not in "0123456789abcdef" for c in build_sha256):
        raise ValueError("backend build digest is invalid")
    if len(declared_features) > 64 or any(
        not feature or len(feature) > 128 for feature in declared_features
    ):
        raise ValueError("backend feature declarations are invalid")
    required = report.get("required_features")
    if not isinstance(required, list) or any(not isinstance(item, str) for item in required):
        raise ValueError("architecture report required features are invalid")
    missing = sorted(set(required) - set(declared_features))
    compatibility = (
        "declared_complete" if probe_compatible and not missing
        else "declared_incomplete"
    )
    report["backend"] = {
        "name": name,
        "build": build_sha256,
        "versions": versions,
        "declared_features": sorted(set(declared_features)),
        "missing_features": missing,
        "issues": list(issues),
        "compatibility": compatibility,
    }
    return report


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
        "backend": {
            "name": None, "build": None, "versions": {},
            "declared_features": [], "missing_features": [], "issues": [],
            "compatibility": "unverified",
        },
        "qualification": dict.fromkeys(
            ("loadable", "correct", "service", "performance"), "unverified"
        ),
        "artifact": _empty_artifact(),
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
