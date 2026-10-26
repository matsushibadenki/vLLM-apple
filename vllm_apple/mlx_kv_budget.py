"""Explicit, source-guarded retained-KV budget. No installed files are edited.

Both sequential and batch paths use the original LRU insertion/eviction logic.
Only its maximum retained bytes and synchronization for scalar snapshots change.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import threading

SERVER_SHA256 = '8514178d18ee7e5edd1db8b8077ab97079ec72d1db666b85c110fb33cdc55147'
CACHE_SHA256 = '440709018cc528ee1e4e42e61ff8713ed2e0079566d9e8fa58eed3a92d334404'


def install_budget(cache_class, budget):
    if type(budget) is not int or not 64 * 1024**2 <= budget <= 4096 * 1024**2:
        raise ValueError('invalid retained KV budget')
    previous = getattr(cache_class, '_soliton_budget', None)
    if previous is not None:
        if previous != budget:
            raise ValueError('cannot change a live retained KV budget')
        return
    original_init = cache_class.__init__
    originals = {name: getattr(cache_class, name) for name in ('insert_cache', 'trim_to', 'fetch_nearest_cache')}

    def init(self, *args, **kwargs):
        self._soliton_kv_lock = threading.RLock()
        original_init(self, *args, **kwargs)
        self.max_bytes = min(self.max_bytes, budget)
        self.trim_to(n_bytes=self.max_bytes)

    def synchronized(fn):
        def call(self, *args, **kwargs):
            with self._soliton_kv_lock:
                return fn(self, *args, **kwargs)
        return call

    cache_class.__init__ = init
    for name, method in originals.items():
        setattr(cache_class, name, synchronized(method))
    cache_class._soliton_budget = budget


def configure_allocator(mx, raw):
    """Limit reusable free allocations once, preserving any stricter existing cap.

    This is not an active-memory limit. MLX reclaims excess on allocation; no
    per-request flush or mutation of live KV/weights is introduced.
    """
    if raw is None:
        return dict(configured_limit_bytes=None, effective_limit_bytes=None)
    if (not 1 <= len(raw) <= 4 or not raw.isascii() or not raw.isdecimal()
            or not 64 <= int(raw) <= 4096):
        raise ValueError('allocator cache MiB must be between 64 and 4096')
    configured = int(raw) * 1024**2
    previous = mx.set_cache_limit(configured)
    effective = min(previous, configured)
    if effective != configured:
        mx.set_cache_limit(effective)
    return dict(configured_limit_bytes=configured, effective_limit_bytes=effective)


def main():
    raw = os.environ.get('VLLM_APPLE_KV_CACHE_MIB', '')
    if (not 1 <= len(raw) <= 4 or not raw.isascii() or not raw.isdecimal()
            or not 64 <= int(raw) <= 4096):
        raise ValueError('explicit retained KV MiB budget required')
    if importlib.metadata.version('mlx-lm') != '0.32.0':
        raise ValueError('retained KV wrapper requires reviewed MLX-LM 0.32.0')
    from mlx_lm import server
    from mlx_lm.models import cache
    if (hashlib.sha256(Path(server.__file__).read_bytes()).hexdigest() != SERVER_SHA256
            or hashlib.sha256(Path(cache.__file__).read_bytes()).hexdigest() != CACHE_SHA256):
        raise ValueError('review retained KV wrapper after server/cache source changes')
    import mlx.core as mx
    allocator_policy = configure_allocator(mx, os.environ.get('VLLM_APPLE_ALLOCATOR_CACHE_MIB'))
    install_budget(cache.LRUPromptCache, int(raw) * 1024**2)
    original_get = server.APIHandler.do_GET

    def get(self):
        if self.path != '/vllm-apple/kv-cache':
            return original_get(self)
        import mlx.core as mx
        lru = self.response_generator.prompt_cache
        with lru._soliton_kv_lock:
            snapshot = dict(retained_bytes=lru.nbytes, max_bytes=lru.max_bytes,
                            entries=len(lru), accounting='LRU tensor bytes, not unique physical memory')
        snapshot['allocator'] = dict(active_bytes=mx.get_active_memory(),
            cache_bytes=mx.get_cache_memory(), peak_bytes=mx.get_peak_memory(), consistency='non_atomic', **allocator_policy)
        encoded = json.dumps(snapshot).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    server.APIHandler.do_GET = get
    server.main()

if __name__ == '__main__':
    main()
