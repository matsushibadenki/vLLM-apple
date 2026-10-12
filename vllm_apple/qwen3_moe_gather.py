"""All-resident Qwen3 gather candidate with no host router materialization."""
from __future__ import annotations

import math

from .expert_residency import ExpertKey
from .expert_timing import measure


def install_qwen3_moe_gather(block, backend, *, layer: int,
                            resident_budget_bytes: int, startup_budget_bytes: int):
    """Install at an idle point; all layer experts must fit, otherwise reject.

    Builds an all-resident bank, not a dynamic/offloaded cache. Startup budget
    includes original source tensors and conservative source-file/stack buffers,
    but cannot bound MLX allocator overhead. Callers must drop other source refs.
    Outputs remain lazy; caller must eval them before closing/reloading the hook.
    Router indices are trusted outputs of the original Qwen3 gate/top-k path.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.qwen3_moe import Qwen3MoeSparseMoeBlock
    from mlx_lm.models.switch_layers import SwiGLU, SwitchGLU

    if (type(block) is not Qwen3MoeSparseMoeBlock or backend._manifest is None
            or type(resident_budget_bytes) is not int or resident_budget_bytes <= 0
            or type(startup_budget_bytes) is not int or startup_budget_bytes <= 0):
        raise ValueError("invalid all-resident gather configuration")
    source = block.switch_mlp
    if type(source) is not SwitchGLU or type(source.activation) is not SwiGLU:
        raise ValueError("expected original standard SwitchGLU")
    count = source.gate_proj.num_experts
    file_bytes = 0
    source_bytes = 0
    for projection in (source.gate_proj, source.up_proj, source.down_proj):
        if ("bias" in projection or getattr(projection, "bits", None) != backend.quantization_bits
                or (backend.quantization_bits is not None and (
                    projection.group_size != backend.group_size or projection.mode != "affine"))):
            raise ValueError("source expert format differs from backend")
        for part in (("weight",) if backend.quantization_bits is None
                     else ("weight", "scales", "biases")):
            source_bytes += projection[part].nbytes
    for expert in range(count):
        entry = backend._manifest.entries.get(ExpertKey(layer, expert))
        if entry is None:
            raise ValueError("manifest is missing a layer expert")
        file_bytes += entry[0]
    if (file_bytes > resident_budget_bytes
            or source_bytes + 2 * file_bytes > startup_budget_bytes):
        raise ValueError("all-resident gather exceeds memory budget")
    tensors = {}
    for expert in range(count):
        resource = backend.load_expert(ExpertKey(layer, expert))
        try:
            for name, value in resource.handle.tensors.items():
                tensors.setdefault(name, []).append(value)
        finally:
            backend.release_expert(resource)
    bank = {name: mx.stack(values) for name, values in tensors.items()}
    mx.eval(*bank.values())
    tensors.clear()
    resident_bytes = sum(t.nbytes for t in bank.values())
    if resident_bytes > resident_budget_bytes:
        raise ValueError("gather bank exceeds resident budget")

    class GatherSwitch(nn.Module):
        def __init__(self):
            super().__init__()
            self.bank = bank
            self.resident_bytes = resident_bytes

        def close(self):
            self.bank.clear()
            self.resident_bytes = 0

        def __call__(self, x, indices):
            if (not self.bank or len(x.shape) not in (2, 3)
                    or indices.shape[:-1] != x.shape[:-1]
                    or not 1 <= indices.shape[-1] <= 64
                    or not 1 <= math.prod(x.shape[:-1]) <= 4096
                    or x.nbytes * indices.shape[-1] > 64 * 1024 * 1024):
                raise ValueError("invalid gather input or closed bank")
            def linear(value, name):
                if backend.quantization_bits is None:
                    return mx.gather_mm(value, self.bank[f"{name}.weight"].swapaxes(-1, -2),
                                        rhs_indices=indices)
                return mx.gather_qmm(
                    value, self.bank[f"{name}.weight"], self.bank[f"{name}.scales"],
                    self.bank[f"{name}.biases"], rhs_indices=indices, transpose=True,
                    group_size=backend.group_size, bits=backend.quantization_bits, mode="affine",
                )
            with measure(backend.timings, "expert_build"):
                expanded = mx.expand_dims(x, (-2, -3))
                up = linear(expanded, "up_proj")
                gate = linear(expanded, "gate_proj")
                result = linear((gate * mx.sigmoid(gate)) * up, "down_proj")
                return result.squeeze(-2)

    hook = GatherSwitch()
    block.switch_mlp = hook
    return hook
