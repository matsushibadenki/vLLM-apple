from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import ModelMemorySpec


DEFAULT_UNINSPECTED_CONTEXT = 4096
_WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".bin")


class ModelInspectionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class InspectedModel:
    model_id: str
    path: Path
    config: dict[str, Any]
    memory_spec: ModelMemorySpec
    kv_dtype_bytes: int


def _hugging_face_cache_path(model_id: str, cache_root: Path | None = None) -> Path | None:
    if "/" not in model_id or model_id.startswith("/"):
        return None
    root = cache_root or (Path.home() / ".cache" / "huggingface" / "hub")
    repository = root / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if not repository.is_dir():
        return None
    candidates = [path for path in repository.iterdir() if (path / "config.json").is_file()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def resolve_model_path(model_id: str, cache_root: Path | None = None) -> Path | None:
    direct = Path(model_id).expanduser()
    if direct.is_dir():
        return direct.resolve()
    return _hugging_face_cache_path(model_id, cache_root)


def _positive_config_int(config: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = config.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    raise ModelInspectionError(f"missing positive model config value: {' or '.join(keys)}")


def _kv_dtype_bytes(config: dict[str, Any]) -> int:
    dtype = str(config.get("torch_dtype") or config.get("dtype") or "float16").lower()
    if dtype in {"float32", "fp32"}:
        return 4
    if dtype in {"float16", "fp16", "half", "bfloat16", "bf16"}:
        return 2
    if dtype in {"float8", "fp8", "uint8", "int8"}:
        return 1
    # Unknown storage quantization must not be applied to KV. Most MLX language
    # models still use 16-bit runtime KV even when weights are packed to 4/8 bit.
    return 2


def _weight_bytes(path: Path) -> int:
    total = 0
    seen_files: set[tuple[int, int]] = set()
    for candidate in path.rglob("*"):
        if not candidate.is_file() or not candidate.name.endswith(_WEIGHT_SUFFIXES):
            continue
        stat = candidate.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity not in seen_files:
            seen_files.add(identity)
            total += stat.st_size
    if total <= 0:
        raise ModelInspectionError("no supported model weight files found")
    return total


def inspect_model(model_id: str, cache_root: Path | None = None) -> InspectedModel:
    path = resolve_model_path(model_id, cache_root)
    if path is None:
        raise ModelInspectionError("model is neither a local directory nor present in HF cache")
    config_path = path / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ModelInspectionError("config.json was not found") from error
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelInspectionError("config.json is not readable JSON") from error
    if not isinstance(config, dict):
        raise ModelInspectionError("model config must be an object")

    layers = _positive_config_int(config, "num_hidden_layers", "n_layer")
    attention_heads = _positive_config_int(config, "num_attention_heads", "n_head")
    kv_heads_value = config.get("num_key_value_heads", attention_heads)
    if not isinstance(kv_heads_value, int) or isinstance(kv_heads_value, bool) or kv_heads_value <= 0:
        raise ModelInspectionError("num_key_value_heads must be positive")
    head_dim_value = config.get("head_dim")
    if isinstance(head_dim_value, int) and not isinstance(head_dim_value, bool) and head_dim_value > 0:
        head_dim = head_dim_value
    else:
        hidden_size = _positive_config_int(config, "hidden_size", "n_embd", "d_model")
        if hidden_size % attention_heads:
            raise ModelInspectionError("hidden_size is not divisible by num_attention_heads")
        head_dim = hidden_size // attention_heads

    dtype_bytes = _kv_dtype_bytes(config)
    kv_bytes_per_token = 2 * layers * kv_heads_value * head_dim * dtype_bytes
    max_context = None
    for key in ("max_position_embeddings", "model_max_length", "n_positions", "seq_length"):
        value = config.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            max_context = value
            break

    spec = ModelMemorySpec(
        model_id=model_id,
        weights_bytes=_weight_bytes(path),
        kv_bytes_per_token=kv_bytes_per_token,
        model_max_context=max_context,
    )
    return InspectedModel(
        model_id=model_id,
        path=path,
        config=config,
        memory_spec=spec,
        kv_dtype_bytes=dtype_bytes,
    )

