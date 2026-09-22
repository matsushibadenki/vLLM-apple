"""Selective one-block loader for the MFLUX Qwen Image quantized transformer."""
from __future__ import annotations

import json
from pathlib import Path

from .mflux_qwen_layer_loader import _read_selected_tensors
from .mflux_qwen_transformer_plan import inspect_mflux_qwen_transformer_staging


class QwenTransformerBlockLoader:
    """Bind one validated artifact inventory, then load individual blocks on demand."""

    def __init__(self, model_root: Path) -> None:
        self.component = model_root / "transformer"
        plan = inspect_mflux_qwen_transformer_staging(model_root)
        if set(plan.tensor_dtype_counts) != {"BF16", "U32"}:
            raise ValueError("Qwen transformer requires the verified BF16/U32 package")
        self.weight_map = json.loads(
            (self.component / "model.safetensors.index.json").read_text(encoding="utf-8")
        )["weight_map"]

    def load(self, block_index: int):
        """Materialize only one verified quantized transformer block."""
        if type(block_index) is not int or not 0 <= block_index < 60:
            raise ValueError("Qwen transformer block index is outside the fixed profile")
        prefix = f"transformer_blocks.{block_index}."
        grouped: dict[str, list[str]] = {}
        for name, shard_name in self.weight_map.items():
            if name.startswith(prefix):
                grouped.setdefault(shard_name, []).append(name)
        if not grouped:
            raise ValueError("Qwen transformer block has no indexed weights")

        from mflux.models.qwen.model.qwen_transformer.qwen_transformer_block import (
            QwenTransformerBlock,
        )
        from mlx import nn
        from mlx.utils import tree_unflatten

        block = QwenTransformerBlock(dim=3072, num_heads=24, head_dim=128)
        nn.quantize(
            block,
            class_predicate=lambda _path, module: hasattr(module, "to_quantized"),
            bits=4,
        )
        flattened = []
        for shard_name, names in sorted(grouped.items()):
            for name, tensor in _read_selected_tensors(
                self.component / shard_name, tuple(sorted(names)), maximum_bytes=512 * 1024**2
            ).items():
                flattened.append((name.removeprefix(prefix), tensor))
        block.update(tree_unflatten(flattened), strict=True)
        return block

    def load_static(self):
        """Materialize the validated non-block transformer layers only."""
        from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
        from mlx import nn
        from mlx.utils import tree_unflatten

        transformer = QwenTransformer(num_layers=0)
        transformer.norm_out.linear = nn.Linear(3072, 6144, bias=True)
        nn.quantize(
            transformer,
            class_predicate=lambda _path, module: hasattr(module, "to_quantized"),
            bits=4,
        )
        grouped: dict[str, list[str]] = {}
        for name, shard_name in self.weight_map.items():
            if not name.startswith("transformer_blocks."):
                grouped.setdefault(shard_name, []).append(name)
        if not grouped:
            raise ValueError("Qwen transformer has no indexed static weights")
        flattened = []
        for shard_name, names in sorted(grouped.items()):
            for name, tensor in _read_selected_tensors(
                self.component / shard_name, tuple(sorted(names)), maximum_bytes=128 * 1024**2
            ).items():
                flattened.append((name, tensor))
        transformer.update(tree_unflatten(flattened), strict=True)
        return transformer


def load_mflux_qwen_transformer_block(model_root: Path, block_index: int):
    """Materialize one block using a fresh artifact validation."""
    if type(block_index) is not int or not 0 <= block_index < 60:
        raise ValueError("Qwen transformer block index is outside the fixed profile")
    return QwenTransformerBlockLoader(model_root).load(block_index)
