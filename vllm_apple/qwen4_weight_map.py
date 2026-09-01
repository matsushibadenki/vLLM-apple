from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from .model import inspect_model_metadata


MAX_INDEX_BYTES = 4 * 1024 * 1024
MAX_WEIGHT_ENTRIES = 16_384
MAX_WEIGHT_NAME_BYTES = 1024
MAX_REPORTED_MISSING = 32
_SHARD_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}\.safetensors$")

_HYPER_SUFFIXES = (
    "block_inject_weight.weight",
    "hc_norm.weight",
    "input_mix_weight_down.weight",
    "input_mix_weight_up.weight",
)
_MLP_SUFFIXES = (
    "experts.down_proj",
    "experts.gate_up_proj",
    "gate.weight",
    "shared_expert.down_proj.weight",
    "shared_expert.gate_proj.weight",
    "shared_expert.up_proj.weight",
    "shared_expert_gate.weight",
)
_LINEAR_SUFFIXES = (
    "A_log",
    "conv1d.weight",
    "dt_bias",
    "in_proj_a.weight",
    "in_proj_b.weight",
    "in_proj_qkv.weight",
    "in_proj_z.weight",
    "norm.weight",
    "out_proj.weight",
)
_QSA_SUFFIXES = (
    "indexer.index_qk_proj.weight",
    "indexer.k_layernorm.weight",
    "indexer.q_layernorm.weight",
    "k_norm.weight",
    "k_proj.weight",
    "o_proj.weight",
    "q_norm.weight",
    "q_proj.weight",
    "v_proj.weight",
)


def _bounded_index(path: Path) -> tuple[dict[str, str], int, str]:
    try:
        attributes = path.stat()
        if (
            not path.is_file()
            or path.is_symlink()
            or attributes.st_uid != os.getuid()
            or not 1 <= attributes.st_size <= MAX_INDEX_BYTES
        ):
            raise ValueError("Qwen4 weight index is outside the bounded file policy")
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Qwen4 weight index is not readable JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"metadata", "weight_map"}:
        raise ValueError("Qwen4 weight index schema is invalid")
    metadata, weight_map = payload["metadata"], payload["weight_map"]
    if not isinstance(metadata, dict) or set(metadata) != {"total_size"}:
        raise ValueError("Qwen4 weight index metadata is invalid")
    total_size = metadata["total_size"]
    if (
        not isinstance(total_size, (int, float))
        or isinstance(total_size, bool)
        or not float(total_size).is_integer()
        or not 1 <= int(total_size) <= 16 * 1024**4
    ):
        raise ValueError("Qwen4 indexed artifact size is invalid")
    if not isinstance(weight_map, dict) or not 1 <= len(weight_map) <= MAX_WEIGHT_ENTRIES:
        raise ValueError("Qwen4 weight map entry count is invalid")
    decoded: dict[str, str] = {}
    for name, shard in weight_map.items():
        if (
            not isinstance(name, str)
            or not 1 <= len(name.encode("utf-8")) <= MAX_WEIGHT_NAME_BYTES
            or any(ord(character) < 0x20 for character in name)
        ):
            raise ValueError("Qwen4 weight name is invalid")
        if not isinstance(shard, str) or not _SHARD_PATTERN.fullmatch(shard):
            raise ValueError("Qwen4 weight shard name is invalid")
        decoded[name] = shard
    return decoded, int(total_size), hashlib.sha256(raw).hexdigest()


def _layer_keys(prefix: str, layer_type: str) -> set[str]:
    result = {
        *(f"{prefix}.attn_hyper_connection.{suffix}" for suffix in _HYPER_SUFFIXES),
        *(f"{prefix}.mlp_hyper_connection.{suffix}" for suffix in _HYPER_SUFFIXES),
        *(f"{prefix}.mlp.{suffix}" for suffix in _MLP_SUFFIXES),
    }
    attention = "linear_attn" if layer_type == "linear_attention" else "self_attn"
    suffixes = _LINEAR_SUFFIXES if layer_type == "linear_attention" else _QSA_SUFFIXES
    result.update(f"{prefix}.{attention}.{suffix}" for suffix in suffixes)
    return result


def _text_keys(config: dict[str, Any]) -> set[str]:
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        raise ValueError("Qwen4 text config is invalid")
    layer_types = text.get("layer_types")
    layers = text.get("num_hidden_layers")
    if not isinstance(layer_types, list) or not isinstance(layers, int) or len(layer_types) != layers:
        raise ValueError("Qwen4 layer map is invalid")
    required = {
        "lm_head.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.hyper_connection_mixer.hc_norm.weight",
        "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
        "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
    }
    for index, layer_type in enumerate(layer_types):
        if layer_type not in {"linear_attention", "full_attention"}:
            raise ValueError("Qwen4 layer type is unsupported")
        required.update(_layer_keys(f"model.language_model.layers.{index}", layer_type))
    ple_layers = text.get("ple_layer_ids", [])
    shard_count = text.get("split_ngram_parts", 0)
    if not isinstance(ple_layers, list) or not isinstance(shard_count, int):
        raise ValueError("Qwen4 PLE mapping metadata is invalid")
    for one_based_index in ple_layers:
        if not isinstance(one_based_index, int) or not 1 <= one_based_index <= layers:
            raise ValueError("Qwen4 PLE layer index is invalid")
        prefix = f"model.language_model.layers.{one_based_index - 1}.ple"
        required.update(
            f"{prefix}.{suffix}"
            for suffix in (
                "conv1d.weight",
                "key_proj.weight",
                "norm_conv.weight",
                "norm_key.weight",
                "norm_query.weight",
                "ple_embedding.layer_multipliers",
                "ple_embedding.ngram_heads_offsets",
                "ple_embedding.ngram_heads_vocab_sizes",
                "value_proj.weight",
            )
        )
        required.update(
            f"{prefix}.ple_embedding.ngram_embedding.shard_{index}.weight"
            for index in range(shard_count)
        )
    return required


def _mtp_keys(config: dict[str, Any]) -> set[str]:
    text = config["text_config"]
    mtp = text.get("mtp")
    if not isinstance(mtp, dict) or mtp.get("num_hidden_layers") != 1:
        raise ValueError("Qwen4 MTP mapping metadata is unsupported")
    required = {
        "mtp.fc_embedding.weight",
        "mtp.fc_hidden.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.hyper_connection_mixer.hc_norm.weight",
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
    }
    required.update(_layer_keys("mtp.layers.0", "full_attention"))
    return required


def _vision_keys(config: dict[str, Any]) -> set[str]:
    vision = config.get("vision_config")
    if not isinstance(vision, dict) or not isinstance(vision.get("depth"), int):
        raise ValueError("Qwen4 vision mapping metadata is invalid")
    required = {
        "model.visual.patch_embed.proj.bias",
        "model.visual.patch_embed.proj.weight",
        "model.visual.pos_embed.weight",
        "model.visual.merger.linear_fc1.bias",
        "model.visual.merger.linear_fc1.weight",
        "model.visual.merger.linear_fc2.bias",
        "model.visual.merger.linear_fc2.weight",
        "model.visual.merger.norm.bias",
        "model.visual.merger.norm.weight",
    }
    for index in range(vision["depth"]):
        prefix = f"model.visual.blocks.{index}"
        required.update(
            f"{prefix}.{suffix}"
            for suffix in (
                "attn.proj.bias",
                "attn.proj.weight",
                "attn.qkv.bias",
                "attn.qkv.weight",
                "mlp.linear_fc1.bias",
                "mlp.linear_fc1.weight",
                "mlp.linear_fc2.bias",
                "mlp.linear_fc2.weight",
                "norm1.bias",
                "norm1.weight",
                "norm2.bias",
                "norm2.weight",
            )
        )
    return required


def inspect_qwen4_weight_map(
    model_metadata: str | Path,
    index_path: str | Path,
    *,
    requested_modes: tuple[str, ...] = ("text",),
) -> dict[str, object]:
    config, capability = inspect_model_metadata(model_metadata)
    if capability.architecture != "qwen4_exp":
        raise ValueError("weight map is not Qwen4-Exp")
    if not requested_modes or len(set(requested_modes)) != len(requested_modes):
        raise ValueError("Qwen4 requested modes are invalid")
    if any(mode not in {"text", "mtp", "vision"} for mode in requested_modes):
        raise ValueError("Qwen4 requested modes are invalid")
    weight_map, total_size, index_sha256 = _bounded_index(Path(index_path).expanduser())
    required = _text_keys(config)
    if "mtp" in requested_modes:
        required.update(_mtp_keys(config))
    if "vision" in requested_modes:
        required.update(_vision_keys(config))
    missing = sorted(required - weight_map.keys())
    return {
        "schema_version": 1,
        "compatible": not missing,
        "architecture": capability.architecture,
        "config_fingerprint": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest(),
        "requested_modes": list(requested_modes),
        "index_entries": len(weight_map),
        "required_entries": len(required),
        "missing_count": len(missing),
        "missing": missing[:MAX_REPORTED_MISSING],
        "missing_truncated": len(missing) > MAX_REPORTED_MISSING,
        "artifact_bytes": total_size,
        "index_sha256": index_sha256,
        "shard_count": len(set(weight_map.values())),
        "loads_weights": False,
    }
