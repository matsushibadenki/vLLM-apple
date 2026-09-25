"""Selective decoder-only loader for the deployed MFLUX Qwen Image VAE."""
from __future__ import annotations

import json
import math
import stat
from pathlib import Path

from .mflux_qwen_layer_loader import _read_selected_tensors

_MAX_INDEX_BYTES = 256 * 1024
_MAX_DECODER_BYTES = 192 * 1024 * 1024


def inspect_mflux_qwen_vae_decoder(model_root: Path) -> dict[str, int]:
    """Verify decoder descriptors and index binding without loading tensor payloads."""
    component = model_root / "vae"
    index = component / "model.safetensors.index.json"
    index_info = index.lstat()
    if not stat.S_ISREG(index_info.st_mode) or not 1 <= index_info.st_size <= _MAX_INDEX_BYTES:
        raise ValueError("Qwen VAE index must be a bounded regular file")
    payload = json.loads(index.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"metadata", "weight_map"}:
        raise ValueError("Qwen VAE index schema is invalid")
    metadata, weight_map = payload["metadata"], payload["weight_map"]
    if not isinstance(metadata, dict) or metadata.get("quantization_level") != "4":
        raise ValueError("Qwen VAE package metadata is invalid")
    if not isinstance(weight_map, dict) or not 1 <= len(weight_map) <= 512:
        raise ValueError("Qwen VAE weight map is invalid")
    if any(not isinstance(name, str) or shard != "0.safetensors" for name, shard in weight_map.items()):
        raise ValueError("Qwen VAE shard binding is invalid")
    selected = {
        name for name in weight_map
        if name.startswith("decoder.") or name.startswith("post_quant_conv.")
    }
    if not selected or not any(name.startswith("decoder.") for name in selected):
        raise ValueError("Qwen VAE decoder coverage is incomplete")
    shard = component / "0.safetensors"
    before = shard.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size < 16:
        raise ValueError("Qwen VAE shard must be a regular file")
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("safetensors is required for Qwen VAE inspection") from error
    decoder_bytes = 0
    with safe_open(str(shard), framework="numpy") as handle:
        if set(handle.keys()) != set(weight_map):
            raise ValueError("Qwen VAE index and shard headers disagree")
        for name in selected:
            descriptor = handle.get_slice(name)
            shape = descriptor.get_shape()
            if descriptor.get_dtype() != "BF16" or not 1 <= len(shape) <= 5 or any(
                type(dimension) is not int or dimension <= 0 for dimension in shape
            ):
                raise ValueError("Qwen VAE decoder tensor descriptor is invalid")
            decoder_bytes += math.prod(shape) * 2
    after = shard.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise ValueError("Qwen VAE shard changed during inspection")
    if decoder_bytes > _MAX_DECODER_BYTES:
        raise ValueError("Qwen VAE decoder exceeds the byte budget")
    return {
        "shard_count": 1,
        "tensor_count": len(weight_map),
        "decoder_tensor_count": len(selected),
        "decoder_payload_bytes": decoder_bytes,
    }


def load_mflux_qwen_vae_decoder(model_root: Path):
    """Load decoder and post-quant convolution without materializing encoder weights."""
    from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
    from mlx.utils import tree_flatten, tree_unflatten

    inspect_mflux_qwen_vae_decoder(model_root)
    vae = QwenVAE()
    vae.encoder = None
    vae.quant_conv = None
    selected = {
        name for name, _ in tree_flatten(vae.parameters())
        if name.startswith("decoder.") or name.startswith("post_quant_conv.")
    }
    component = model_root / "vae"
    payload = json.loads((component / "model.safetensors.index.json").read_text(encoding="utf-8"))
    indexed = {
        name for name in payload["weight_map"]
        if name.startswith("decoder.") or name.startswith("post_quant_conv.")
    }
    if selected != indexed:
        raise ValueError("Qwen VAE decoder module and artifact parameters disagree")
    tensors = _read_selected_tensors(
        component / "0.safetensors", tuple(sorted(selected)), maximum_bytes=_MAX_DECODER_BYTES
    )
    vae.update(tree_unflatten(list(tensors.items())), strict=False)
    return vae
