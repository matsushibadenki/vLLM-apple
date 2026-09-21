"""Sequentially materialize MFLUX Qwen Image BF16 text-encoder layers."""
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Callable

from .hardware import detect_hardware
from .mflux_qwen_layer_loader import _read_selected_tensors, load_mflux_qwen_text_layer
from .mflux_qwen_streaming_plan import inspect_mflux_qwen_text_encoder_staging


class _StreamingLayer:
    def __init__(
        self, root: Path, index: int, payload_bytes: int,
        on_layer: Callable[[int, int], None] | None,
    ) -> None:
        self.root = root
        self.index = index
        self.payload_bytes = payload_bytes
        self.on_layer = on_layer

    def __call__(self, hidden_states, attention_mask, position_embeddings):
        import mlx.core as mx

        hardware = detect_hardware()
        if hardware.memory.pressure.value != "normal" or (
            hardware.memory.available_bytes < self.payload_bytes * 4
        ):
            raise MemoryError("Qwen text layer stopped before unsafe materialization")
        layer = load_mflux_qwen_text_layer(self.root, self.index)
        output = layer(hidden_states, attention_mask, position_embeddings)
        mx.eval(output)
        peak = mx.get_peak_memory()
        del layer
        gc.collect()
        mx.clear_cache()
        if self.on_layer is not None:
            self.on_layer(self.index, peak)
        return output


def build_streaming_qwen_text_encoder(
    model_root: Path, *, layer_limit: int = 28,
    on_layer: Callable[[int, int], None] | None = None,
):
    """Build a text encoder with static weights and bounded one-layer proxies."""
    if type(layer_limit) is not int or not 1 <= layer_limit <= 28:
        raise ValueError("Qwen text encoder diagnostic layer limit is invalid")
    plan = inspect_mflux_qwen_text_encoder_staging(model_root)
    if plan.quantized_weight_tensor_count or set(plan.tensor_dtype_counts) != {"BF16", "F32"}:
        raise ValueError("Qwen text encoder requires the verified BF16/F32 package")
    component = model_root / "text_encoder"
    weight_map = json.loads(
        (component / "model.safetensors.index.json").read_text()
    )["weight_map"]
    static_names = {
        "encoder.embed_tokens.weight",
        "encoder.norm.weight",
        "encoder.rotary_emb.inv_freq",
    }
    if {name for name in weight_map if not name.startswith("encoder.layers.")} != static_names:
        raise ValueError("Qwen text encoder static weight contract changed")
    grouped: dict[str, list[str]] = {}
    for name in sorted(static_names):
        grouped.setdefault(weight_map[name], []).append(name)

    from mflux.models.qwen.model.qwen_text_encoder.qwen_text_encoder import QwenTextEncoder
    from mlx.utils import tree_unflatten

    encoder = QwenTextEncoder()
    encoder.encoder.layers = tuple(
        _StreamingLayer(model_root, index, plan.layer_payload_bytes[index], on_layer)
        for index in range(layer_limit)
    )
    flattened = []
    for shard_name, names in sorted(grouped.items()):
        for name, tensor in _read_selected_tensors(
            component / shard_name, tuple(names), maximum_bytes=2 * 1024**3
        ).items():
            flattened.append((name.removeprefix("encoder."), tensor))
    encoder.encoder.update(tree_unflatten(flattened), strict=False)
    return encoder
