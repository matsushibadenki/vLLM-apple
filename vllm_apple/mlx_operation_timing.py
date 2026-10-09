"""Profile existing generate-module MLX calls without adding GPU work."""
from functools import wraps
from threading import local

from .step_diagnostics import StepDiagnostics


class PrefillOperationTiming:
    def __init__(self, mlx):
        self._mlx = mlx
        self._context = local()
        self.diagnostics = {name: StepDiagnostics(sample_limit=16,
            scope=f'prefill {name} host call; nested in prefill; not GPU-only')
            for name in ('eval', 'clear_cache')}
        self._timed = {name: diagnostic.wrap(getattr(mlx, name))
                       for name, diagnostic in self.diagnostics.items()}

    def __getattr__(self, name):
        return getattr(self._mlx, name)

    def eval(self, *args, **kwargs):
        method = self._timed['eval'] if getattr(self._context, 'active', False) else self._mlx.eval
        return method(*args, **kwargs)

    def clear_cache(self, *args, **kwargs):
        method = self._timed['clear_cache'] if getattr(self._context, 'active', False) else self._mlx.clear_cache
        return method(*args, **kwargs)

    def wrap_prompt(self, method):
        @wraps(method)
        def prompt(*args, **kwargs):
            previous = getattr(self._context, 'active', False)
            self._context.active = True
            try:
                return method(*args, **kwargs)
            finally:
                self._context.active = previous
        return prompt

    def snapshot(self):
        return {name: diagnostic.snapshot() for name, diagnostic in self.diagnostics.items()}


def install_prefill_operation_timing(generation):
    """Use only after generate source verification, on the isolated P1 worker."""
    if isinstance(generation.mx, PrefillOperationTiming):
        return generation.mx
    proxy = PrefillOperationTiming(generation.mx)
    generation.PromptProcessingBatch.prompt = proxy.wrap_prompt(generation.PromptProcessingBatch.prompt)
    generation.mx = proxy
    return proxy
