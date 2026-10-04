"""Backend-owned, identity-bound wrapper around actual MLX-LM prompt caches."""
from __future__ import annotations

import threading
from typing import Any


class IdentityPromptCache:
    def __init__(self, inner: Any, namespace: str, *, enabled: bool = True) -> None:
        self.inner = inner
        self.namespace = namespace
        self.enabled = enabled
        self._lock = threading.Lock()
        self._metrics = dict(fetches=0, hits=0, misses=0, prompt_tokens=0,
                             reused_tokens=0, submitted_prefill_tokens=0, removed_entries=0)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def __len__(self) -> int:
        return len(self.inner)

    def _keys(self) -> set:
        return {(model, tuple(tokens)) for entries in self.inner._lru._lrus.values()
                for model, tokens in entries}

    def fetch_nearest_cache(self, model: Any, tokens: list[int]) -> Any:
        cache, remaining = (self.inner.fetch_nearest_cache((self.namespace, model), tokens)
                            if self.enabled else (None, tokens))
        reused = len(tokens)-len(remaining) if cache is not None else 0
        if not 0 <= reused <= len(tokens) or remaining != tokens[reused:]:
            raise ValueError('backend cache did not return an exact token suffix')
        with self._lock:
            self._metrics['fetches'] += 1
            self._metrics['hits' if reused else 'misses'] += 1
            self._metrics['prompt_tokens'] += len(tokens)
            self._metrics['reused_tokens'] += reused
            self._metrics['submitted_prefill_tokens'] += len(remaining)
        return cache, remaining

    def insert_cache(self, model: Any, tokens: list[int], cache: Any, **kwargs: Any) -> None:
        if not self.enabled:
            return
        before = self._keys()
        self.inner.insert_cache((self.namespace, model), tokens, cache, **kwargs)
        with self._lock:
            self._metrics['removed_entries'] += len(before-self._keys())

    def trim_to(self, **kwargs: Any) -> None:
        before = self._keys()
        self.inner.trim_to(**kwargs)
        with self._lock:
            self._metrics['removed_entries'] += len(before-self._keys())

    def snapshot(self) -> dict:
        with self._lock:
            return dict(**self._metrics, namespace=self.namespace, enabled=self.enabled,
                        cache_bytes=self.inner.nbytes, cache_entries=len(self.inner),
                        snapshot_consistency='non_atomic', qualification=False,
                        cache_bytes_source='backend_lru_accounting',
                        removal_reason='prefix pruning or capacity eviction; not separated')
