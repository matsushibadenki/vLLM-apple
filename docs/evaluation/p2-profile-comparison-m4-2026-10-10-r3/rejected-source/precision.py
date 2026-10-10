"""Compile only immutable Gemma 2 MLPs; cache mutation remains outside graphs."""
from __future__ import annotations

import hashlib
from functools import partial, wraps
from pathlib import Path

from vllm_apple.mlx_gemma2_compat import GEMMA2_SOURCE_SHA256


def install_mlp_compilation(module, mx):
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != GEMMA2_SOURCE_SHA256:
        raise ValueError('MLP compilation requires the reviewed Gemma 2 source')
    original = module.MLP.__call__
    if getattr(original, '_vllm_p2_compiled_mlp', False):
        return False

    @wraps(original)
    def call(self, x):
        compiled = vars(self).get('_vllm_compiled_mlp')
        if compiled is None:
            compiled = mx.compile(partial(original, self), shapeless=True)
            self._vllm_compiled_mlp = compiled
        return compiled(x)

    call._vllm_p2_compiled_mlp = True
    module.MLP.__call__ = call
    return True
