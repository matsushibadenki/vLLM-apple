"""Share immutable SPM vocabulary data, never a request's streaming state.

SPM field initialization follows MLX-LM (MIT), copyright Apple Inc.
Only used with the source-pinned P1 worker and an immutable tokenizer.
"""
from __future__ import annotations

import hashlib
import threading
from functools import wraps
from pathlib import Path

TOKENIZER_SOURCE_SHA256 = 'c9eea380fd7e1a624f8873a2d2298fd75283285750f74a9348b655659017ee09'
_TABLE_ATTRIBUTE = '_vllm_apple_immutable_spm_tokenmap'


class SPMTokenmapReuse:
    def __init__(self, original):
        self._original = original
        self._lock = threading.Lock()
        self._builds = 0
        self._instances = 0
        self._entries = 0

    def initialize(self, instance, tokenizer, trim_space=True):
        table = vars(tokenizer).get(_TABLE_ATTRIBUTE)
        if table is None:
            with self._lock:
                table = vars(tokenizer).get(_TABLE_ATTRIBUTE)
                if table is None:
                    self._original(instance, tokenizer, trim_space=trim_space)
                    table = tuple(instance.tokenmap)
                    setattr(tokenizer, _TABLE_ATTRIBUTE, table)
                    self._builds += 1
                    self._entries += len(table)
                    instance.tokenmap = table
                    self._instances += 1
                    return
        instance.trim_space = trim_space
        instance._sep = b'\xe2\x96\x81'
        instance.tokenmap = table
        instance.reset()
        self._instances += 1

    def snapshot(self):
        return dict(tables_built=self._builds, instances=self._instances,
                    table_entries=self._entries, shared_instances=self._instances-self._builds,
                    scope='immutable table per tokenizer; independent streaming state; non-atomic counters')


def install_spm_tokenmap_reuse(module):
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != TOKENIZER_SOURCE_SHA256:
        raise ValueError('SPM reuse requires the reviewed MLX-LM tokenizer source')
    original = module.SPMStreamingDetokenizer.__init__
    existing = getattr(original, '_vllm_spm_reuse', None)
    if existing is not None:
        return existing
    reuse = SPMTokenmapReuse(original)

    @wraps(original)
    def initialize(instance, tokenizer, trim_space=True):
        reuse.initialize(instance, tokenizer, trim_space=trim_space)

    initialize._vllm_spm_reuse = reuse
    module.SPMStreamingDetokenizer.__init__ = initialize
    return reuse
