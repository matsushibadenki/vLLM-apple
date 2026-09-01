from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .qwen4_weight_map import _bounded_index, inspect_qwen4_weight_map


_COMPONENT_POLICIES = {
    "output_head": "resident",
    "token_embedding": "resident",
    "gated_residual": "resident",
    "gated_deltanet": "resident",
    "qwen_sparse_attention": "resident",
    "mixture_of_experts": "on_demand_expert",
    "per_layer_embedding": "partitioned_lookup",
    "vision_encoder": "optional_mode",
    "multi_token_prediction": "optional_mode",
}


def _component(name: str) -> str | None:
    if name == "lm_head.weight":
        return "output_head"
    if name == "model.language_model.embed_tokens.weight":
        return "token_embedding"
    if name.startswith("model.visual."):
        return "vision_encoder"
    if name.startswith("mtp."):
        return "multi_token_prediction"
    if ".attn_hyper_connection." in name or ".mlp_hyper_connection." in name:
        return "gated_residual"
    if name.startswith("model.language_model.hyper_connection_mixer."):
        return "gated_residual"
    if ".linear_attn." in name:
        return "gated_deltanet"
    if ".self_attn." in name:
        return "qwen_sparse_attention"
    if ".mlp." in name:
        return "mixture_of_experts"
    if ".ple." in name:
        return "per_layer_embedding"
    return None


def build_qwen4_conversion_plan(
    model_metadata: str | Path,
    index_path: str | Path,
    *,
    requested_modes: tuple[str, ...] = ("text",),
) -> dict[str, object]:
    mapping = inspect_qwen4_weight_map(
        model_metadata, index_path, requested_modes=requested_modes
    )
    if not mapping["compatible"]:
        raise ValueError("Qwen4 conversion requires a complete requested-mode mapping")
    weight_map, _, index_sha256 = _bounded_index(Path(index_path).expanduser())
    counts: Counter[str] = Counter()
    shard_entries: Counter[str] = Counter()
    unclassified = 0
    for name, shard in weight_map.items():
        component = _component(name)
        if component is None:
            unclassified += 1
        else:
            counts[component] += 1
        shard_entries[shard] += 1
    if unclassified:
        raise ValueError("Qwen4 conversion index contains unclassified tensors")
    enabled = {
        "output_head",
        "token_embedding",
        "gated_residual",
        "gated_deltanet",
        "qwen_sparse_attention",
        "mixture_of_experts",
        "per_layer_embedding",
    }
    if "vision" in requested_modes:
        enabled.add("vision_encoder")
    if "mtp" in requested_modes:
        enabled.add("multi_token_prediction")
    body = {
        "schema_version": 1,
        "architecture": "qwen4_exp",
        "config_fingerprint": mapping["config_fingerprint"],
        "index_sha256": index_sha256,
        "requested_modes": list(requested_modes),
        "source_entries": len(weight_map),
        "source_shards": len(shard_entries),
        "max_entries_per_shard": max(shard_entries.values()),
        "component_entries": dict(sorted(counts.items())),
        "component_policies": {
            name: _COMPONENT_POLICIES[name] for name in sorted(_COMPONENT_POLICIES)
        },
        "enabled_components": sorted(enabled),
        "disabled_optional_components": sorted(set(_COMPONENT_POLICIES) - enabled),
        "peak_open_source_shards": 1,
        "peak_open_destination_shards": 1,
        "requires_full_artifact_in_memory": False,
        "loads_tensor_data": False,
        "preserves_source_tensor_names": True,
        "requires_requantization": False,
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return {**body, "plan_id": hashlib.sha256(canonical).hexdigest()}
