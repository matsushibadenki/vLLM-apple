"""Explicit Qwen3 MoE reference hook; install only before serving requests."""
from __future__ import annotations

import math

from .expert_execution import ResidentExpertExecutor
from .expert_residency import ExpertKey
from .expert_timing import measure
from .mlx_expert_backend import MLXFileExpertBackend


def install_qwen3_moe_residency(
    block, backend: MLXFileExpertBackend, executor: ResidentExpertExecutor, *,
    layer: int, phase: str, grouped: bool = False,
):
    """Replace switch_mlp only, leaving original gate/top-k/aggregation untouched.

    Requires a manifest-bound backend. The returned hook owns no original switch
    bank. External source references must also be dropped by the model loader.
    This synchronous reference implementation is intentionally opt-in and is not
    safe to install/change phase concurrently with inference. Reload the original
    model to restore baseline; the hook does not retain rollback weight copies.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.qwen3_moe import Qwen3MoeSparseMoeBlock
    from mlx_lm.models.switch_layers import SwiGLU, SwitchGLU

    if (type(block) is not Qwen3MoeSparseMoeBlock or phase not in {"prefill", "decode"}
            or type(grouped) is not bool
            or not executor.manager.uses_backend(backend) or backend._manifest is None):
        raise ValueError("unsupported or unbound Qwen3 MoE hook")
    source = block.switch_mlp
    if type(source) is not SwitchGLU or type(source.activation) is not SwiGLU:
        raise ValueError("expected original standard SwitchGLU")
    num_experts = source.gate_proj.num_experts
    for projection in (source.gate_proj, source.up_proj, source.down_proj):
        if ("bias" in projection
                or getattr(projection, "bits", None) != backend.quantization_bits
                or (backend.quantization_bits is not None
                    and (projection.group_size != backend.group_size or projection.mode != "affine"))):
            raise ValueError("source expert format differs from backend")
    for expert in range(num_experts):
        if ExpertKey(layer, expert) not in backend._manifest.entries:
            raise ValueError("manifest is missing a layer expert")

    class ResidentSwitch(nn.Module):
        def __init__(self):
            super().__init__()
            self.phase = phase

        def set_phase(self, value):
            if value not in {"prefill", "decode"}:
                raise ValueError("invalid expert phase")
            self.phase = value

        def __call__(self, x, indices):
            if (len(x.shape) not in (2, 3) or len(indices.shape) != len(x.shape)
                    or indices.shape[:-1] != x.shape[:-1]
                    or not 1 <= indices.shape[-1] <= 64
                    or indices.dtype not in (mx.int32, mx.int64, mx.uint32, mx.uint64)):
                raise ValueError("invalid routed SwitchGLU input")
            rows = math.prod(x.shape[:-1])
            output_bytes = rows * indices.shape[-1] * x.shape[-1] * x.itemsize
            if not 1 <= rows <= 4096 or output_bytes > 64 * 1024 * 1024:
                raise ValueError("routed SwitchGLU output exceeds reference limit")
            with measure(backend.timings, "router_eval"):
                mx.eval(indices)
            with measure(backend.timings, "router_to_host"):
                selections = indices.reshape(rows, -1).tolist()
            if any(len(set(row)) != len(row)
                   or any(not 0 <= expert < num_experts for expert in row)
                   for row in selections):
                raise ValueError("invalid authoritative router selection")
            tokens = x.reshape(rows, -1)
            results = []
            for row, selected in enumerate(selections):
                fits = False
                if grouped:
                    budget = executor.manager.snapshot()
                    # File bytes include safetensors headers as well as tensors.
                    fits = (len(selected) <= budget["maximum_entries"] and sum(
                        backend._manifest.entries[ExpertKey(layer, expert)][0]
                        for expert in selected
                    ) <= budget["maximum_bytes"])
                if grouped and fits:
                    results.append(backend.execute_unaggregated(
                        executor, tokens[row:row + 1], phase=self.phase, layer=layer,
                        selected_experts=tuple(selected),
                    ))
                    continue
                outputs = [backend.execute_selected(
                    executor, tokens[row:row + 1], phase=self.phase, layer=layer,
                    selected_experts=(expert,), routing_weights=(1.0,),
                ) for expert in selected]
                with measure(backend.timings, "output_assembly"):
                    results.append(mx.stack(outputs, axis=-2))
            with measure(backend.timings, "output_assembly"):
                result = mx.concatenate(results, axis=0).reshape(
                    *x.shape[:-1], indices.shape[-1], x.shape[-1],
                )
            with measure(backend.timings, "output_eval"):
                mx.eval(result)
            return result

    hook = ResidentSwitch()
    block.switch_mlp = hook
    return hook
