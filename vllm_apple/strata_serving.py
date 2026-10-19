"""Single-model, serialized serving adapter for the explicit Strata runtime."""
from __future__ import annotations

from .strata_runtime import StrataRuntime


class StrataServingModel:
    """Delegate model APIs while assigning telemetry phases by call width.

    A single-token forward is labeled decode, including the final singleton
    prompt chunk. Labels describe forward shape, not exact HTTP prompt accounting.
    """

    def __init__(self, model, runtime):
        self._model = model
        self.runtime = runtime

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __call__(self, inputs, *args, **kwargs):
        if len(inputs.shape) != 2 or inputs.shape[0] != 1:
            raise ValueError("Strata serving requires a single sequence")
        self.runtime.admit_forward()
        self.runtime.set_phase("prefill" if inputs.shape[1] > 1 else "decode")
        return self._model(inputs, *args, **kwargs)


class StrataModelProvider:
    """Preload once; reject model swaps, adapters and drafts before serving."""

    def __init__(self, provider, **runtime_options):
        if provider.is_distributed:
            raise ValueError("Strata serving supports one local worker")
        # Reject known exhausted admission before loading the original source bank.
        # Loading itself can still exceed the observed budget; this is not a hard cap.
        budget = runtime_options.get("process_budget")
        if budget is not None:
            from .strata_memory import ProcessMemoryBudget, observe_process_memory
            observer = runtime_options.get("memory_observer") or observe_process_memory
            if not isinstance(budget, ProcessMemoryBudget) or not callable(observer):
                raise ValueError("invalid process memory admission configuration")
            if budget.resident_limit(**observer(), resident_bytes=0, requested_bytes=1) <= 0:
                raise MemoryError("Strata startup has no memory headroom")
        provider.load_default()
        if provider.model is None or provider.draft_model is not None:
            raise ValueError("Strata serving requires a preloaded model without draft")
        self.runtime = StrataRuntime.from_model(provider.model, **runtime_options)
        self.model = StrataServingModel(provider.model, self.runtime)
        self._provider = provider
        self.is_batchable = False

    def __getattr__(self, name):
        return getattr(self._provider, name)

    def load_default(self):
        return self.load("default_model")

    def load(self, model_path, adapter_path=None, draft_model_path=None):
        if (model_path not in {"default_model", self.cli_args.model}
                or adapter_path is not None
                or draft_model_path not in {None, "default_model"}):
            raise ValueError("Strata serving rejects model changes, adapters and drafts")
        if self.runtime.snapshot()["closed"]:
            raise RuntimeError("Strata serving is closed")
        return self.model, self.tokenizer

    def close(self):
        import mlx.core as mx
        mx.synchronize()
        self.runtime.close()
