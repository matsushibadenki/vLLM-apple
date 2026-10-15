"""Explicit synchronous multi-layer Strata session for standard Qwen3 MoE.

Install before inference, using offline exported immutable expert files. The
caller supplies layer identities; source banks are retained only during install
rollback. Concurrent model calls or lifecycle changes are unsupported.
"""
from __future__ import annotations

from .expert_execution import ResidentExpertExecutor
from .expert_residency import ExpertResidencyManager
from .mlx_expert_backend import MLXFileExpertBackend
from .qwen3_moe_residency import install_qwen3_moe_residency


class StrataRuntime:
    """Own one shared residency budget and the installed layer hooks."""

    def __init__(self, blocks, backend: MLXFileExpertBackend, *, maximum_entries: int,
                 maximum_bytes: int, mode: str = "sequential", eviction_policy: str = "lru",
                 maximum_group_rows: int = 8, maximum_samples: int = 4096,
                 process_budget=None, memory_observer=None):
        if (mode not in {"sequential", "grouped", "row_batch", "selection_batch"} or not blocks
                or any(type(layer) is not int or layer < 0 for layer in blocks)
                or len({id(block) for block in blocks.values()}) != len(blocks)):
            raise ValueError("invalid Strata layer mapping or mode")
        self.manager = ExpertResidencyManager(
            backend, maximum_entries=maximum_entries, maximum_bytes=maximum_bytes,
            eviction_policy=eviction_policy)
        self.executor = ResidentExpertExecutor(
            self.manager, maximum_samples=maximum_samples, timings=backend.timings)
        self.process_budget = process_budget
        self.memory_observer = memory_observer
        self._requested_bytes = maximum_bytes
        self._last_memory = None
        self.mode = mode
        self._closed = False
        self._hooks = {}
        originals = {}
        try:
            for layer, block in blocks.items():
                originals[layer] = block.switch_mlp
                self._hooks[layer] = install_qwen3_moe_residency(
                    block, backend, self.executor, layer=layer, phase="prefill",
                    grouped=mode != "sequential",
                    maximum_group_rows=maximum_group_rows if mode in {"row_batch", "selection_batch"} else 1,
                    selection_batch=mode == "selection_batch")
        except Exception:
            for layer, original in originals.items():
                blocks[layer].switch_mlp = original
            self.manager.close()
            raise
        # Never retain the original bank after successful installation.

    @classmethod
    def from_model(cls, model, *, expert_root, expected_model_sha256: str,
                   maximum_file_bytes: int, maximum_entries: int, maximum_bytes: int,
                   mode: str = "sequential", eviction_policy: str = "lru",
                   maximum_group_rows: int = 8, timings=None, process_budget=None, memory_observer=None):
        """Install all standard Qwen3 MoE layers, rejecting mismatched exports.

        Call after loading the model and before starting requests. This loads no
        model weights itself; callers must budget the original startup bank.
        """
        from mlx_lm.models.qwen3_moe import Model, Qwen3MoeSparseMoeBlock

        from .expert_residency import ExpertKey

        if type(model) is not Model:
            raise ValueError("Strata supports only the standard Qwen3 MoE model")
        blocks = {i: layer.mlp for i, layer in enumerate(model.layers)
                  if type(layer.mlp) is Qwen3MoeSparseMoeBlock}
        if not blocks:
            raise ValueError("model has no supported MoE layers")
        backend = MLXFileExpertBackend.from_manifest(
            expert_root, expected_model_sha256=expected_model_sha256,
            maximum_file_bytes=maximum_file_bytes, timings=timings)
        expected = {ExpertKey(layer, expert) for layer, block in blocks.items()
                    for expert in range(block.switch_mlp.gate_proj.num_experts)}
        if set(backend._manifest.entries) != expected:
            raise ValueError("export layer/expert coverage differs from loaded model")
        return cls(blocks, backend, maximum_entries=maximum_entries,
                   maximum_bytes=maximum_bytes, mode=mode, eviction_policy=eviction_policy,
                   maximum_group_rows=maximum_group_rows, process_budget=process_budget,
                   memory_observer=memory_observer)

    def set_phase(self, phase: str):
        self._require_open()
        if phase not in {"prefill", "decode"}:
            raise ValueError("invalid Strata phase")
        for hook in self._hooks.values():
            hook.set_phase(phase)

    def resize(self, *, maximum_entries: int, maximum_bytes: int):
        self._require_open()
        result = self.manager.resize(maximum_entries=maximum_entries, maximum_bytes=maximum_bytes)
        self._requested_bytes = maximum_bytes
        return result

    def admit_forward(self):
        self._require_open()
        if self.process_budget is None:
            return
        from .strata_memory import observe_process_memory
        observation = (self.memory_observer or observe_process_memory)()
        state = self.manager.snapshot()
        limit = self.process_budget.resident_limit(
            **observation, resident_bytes=state["resident_bytes"],
            requested_bytes=self._requested_bytes)
        if limit <= 0:
            raise MemoryError("Strata has no expert memory headroom")
        self.manager.resize(maximum_entries=state["maximum_entries"], maximum_bytes=limit)
        self._last_memory = dict(observation, admitted_resident_bytes=limit)

    def snapshot(self):
        return dict(mode=self.mode, closed=self._closed, layers=sorted(self._hooks),
                    residency=self.manager.snapshot(), memory_admission=self._last_memory)

    def close(self):
        if not self._closed:
            self.manager.close()  # Reject active leases before changing hooks.
            for hook in self._hooks.values():
                hook.close()
            self._closed = True

    def _require_open(self):
        if self._closed:
            raise RuntimeError("Strata runtime is closed; reload model to restore baseline")

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_exc):
        self.close()
