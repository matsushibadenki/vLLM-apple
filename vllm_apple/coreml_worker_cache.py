"""Bounded identity cache for isolated persistent Core ML model workers."""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from typing import Callable

from .ane_probe import CoreMLANEModelProbeConfig, CoreMLPrediction
from .coreml_worker import CoreMLPersistentWorker


@dataclass(frozen=True, slots=True)
class CoreMLWorkerCacheKey:
    hardware_fingerprint: str
    os_version: str
    model_root_sha256: str
    input_name: str
    output_name: str
    input_count: int

    def __post_init__(self) -> None:
        if (not 1 <= len(self.hardware_fingerprint) <= 128
                or not 1 <= len(self.os_version) <= 128
                or len(self.model_root_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.model_root_sha256)
                or not 1 <= len(self.input_name) <= 128
                or not 1 <= len(self.output_name) <= 128
                or type(self.input_count) is not int
                or not 1 <= self.input_count <= 4096):
            raise ValueError("invalid Core ML worker cache key")

    @property
    def cache_id(self) -> str:
        return hashlib.sha256(json.dumps({
            "hardware_fingerprint": self.hardware_fingerprint,
            "os_version": self.os_version,
            "model_root_sha256": self.model_root_sha256,
            "input_name": self.input_name,
            "output_name": self.output_name,
            "input_count": self.input_count,
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]

    @classmethod
    def from_config(
        cls, config: CoreMLANEModelProbeConfig, *,
        hardware_fingerprint: str, os_version: str,
    ) -> "CoreMLWorkerCacheKey":
        return cls(
            hardware_fingerprint, os_version, config.model_root_sha256,
            config.input_name, config.output_name, len(config.input_values),
        )


@dataclass(slots=True)
class _Entry:
    worker: CoreMLPersistentWorker
    leases: int
    last_used: int


class CoreMLWorkerLease:
    def __init__(self, cache: "CoreMLWorkerCache", key: CoreMLWorkerCacheKey,
                 worker: CoreMLPersistentWorker) -> None:
        self._cache = cache
        self.key = key
        self.worker = worker
        self._released = False

    def predict(self, values: tuple[float, ...]) -> CoreMLPrediction:
        if self._released:
            raise RuntimeError("Core ML worker cache lease was released")
        return self.worker.predict(values)

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._cache._release(self.key)

    def __enter__(self) -> "CoreMLWorkerLease":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.release()


class CoreMLWorkerCache:
    """LRU of subprocess-isolated loaded models; active leases are never evicted."""

    def __init__(
        self, maximum_workers: int = 4, *,
        worker_factory: Callable[[CoreMLANEModelProbeConfig], CoreMLPersistentWorker] = (
            CoreMLPersistentWorker
        ),
    ) -> None:
        if (type(maximum_workers) is not int
                or not 1 <= maximum_workers <= 16
                or not callable(worker_factory)):
            raise ValueError("invalid Core ML worker cache configuration")
        self._maximum = maximum_workers
        self._factory = worker_factory
        self._entries: dict[CoreMLWorkerCacheKey, _Entry] = {}
        self._clock = 0
        self._lock = threading.RLock()
        self._closed = False

    def acquire(
        self, key: CoreMLWorkerCacheKey, config: CoreMLANEModelProbeConfig
    ) -> CoreMLWorkerLease:
        expected = CoreMLWorkerCacheKey.from_config(
            config, hardware_fingerprint=key.hardware_fingerprint,
            os_version=key.os_version,
        )
        if key != expected:
            raise ValueError("Core ML worker cache key does not match model configuration")
        with self._lock:
            if self._closed:
                raise RuntimeError("Core ML worker cache is closed")
            self._clock += 1
            entry = self._entries.get(key)
            if entry is None:
                if len(self._entries) >= self._maximum:
                    idle = [(value.last_used, item_key) for item_key, value in self._entries.items()
                            if value.leases == 0]
                    if not idle:
                        raise RuntimeError("Core ML worker cache capacity is fully leased")
                    _, victim = min(idle, key=lambda item: item[0])
                    self._entries.pop(victim).worker.close()
                worker = self._factory(config)
                entry = _Entry(worker, 0, self._clock)
                self._entries[key] = entry
            entry.leases += 1
            entry.last_used = self._clock
            return CoreMLWorkerLease(self, key, entry.worker)

    def _release(self, key: CoreMLWorkerCacheKey) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.leases <= 0:
                raise RuntimeError("Core ML worker cache lease accounting failed")
            self._clock += 1
            entry.leases -= 1
            entry.last_used = self._clock

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "maximum_workers": self._maximum,
                "resident_workers": len(self._entries),
                "active_leases": sum(entry.leases for entry in self._entries.values()),
                "cache_ids": sorted(key.cache_id for key in self._entries),
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if any(entry.leases for entry in self._entries.values()):
                raise RuntimeError("cannot close Core ML worker cache with active leases")
            self._closed = True
            entries, self._entries = tuple(self._entries.values()), {}
        failures = []
        for entry in entries:
            try:
                entry.worker.close()
            except Exception as error:
                failures.append(type(error).__name__)
        if failures:
            raise RuntimeError(f"Core ML worker cache shutdown failures: {failures}")
