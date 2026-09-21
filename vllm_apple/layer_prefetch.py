"""Two-slot layer prefetch with bounded on-demand fallback and cleanup."""
from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

MAX_PREFETCH_LAYERS = 4096
_Resource = TypeVar("_Resource")
_Result = TypeVar("_Result")


class LayerResidencyBackend(Protocol[_Resource]):
    def load_layer(self, layer: int) -> _Resource: ...
    def release_layer(self, layer: int, resource: _Resource) -> None: ...


@dataclass(frozen=True, slots=True)
class LayerPrefetchReport(Generic[_Result]):
    results: tuple[_Result, ...]
    layer_count: int
    prefetched_layers: int
    prefetch_failures: int
    on_demand_fallbacks: int
    released_layers: int


class LayerPrefetchCoordinator(Generic[_Resource]):
    def __init__(self, backend: LayerResidencyBackend[_Resource]) -> None:
        self._backend = backend

    def execute(
        self,
        layers: tuple[int, ...],
        consume: Callable[[int, _Resource], _Result],
        *,
        cancellation: threading.Event | None = None,
    ) -> LayerPrefetchReport[_Result]:
        if (not layers or len(layers) > MAX_PREFETCH_LAYERS
                or len(set(layers)) != len(layers)
                or any(type(layer) is not int or layer < 0 for layer in layers)
                or not callable(consume)
                or (cancellation is not None and not isinstance(cancellation, threading.Event))):
            raise ValueError("invalid layer prefetch request")
        cancel = cancellation or threading.Event()
        results = []
        prefetched = failures = fallbacks = released = 0
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="layer-prefetch")
        future: Future[_Resource] | None = executor.submit(self._backend.load_layer, layers[0])
        future_layer: int | None = layers[0]
        try:
            for index, layer in enumerate(layers):
                if cancel.is_set():
                    raise RuntimeError("layer prefetch cancelled")
                assert future is not None
                try:
                    resource = future.result()
                    prefetched += 1
                except Exception:
                    failures += 1
                    if cancel.is_set():
                        raise RuntimeError("layer prefetch cancelled")
                    resource = self._backend.load_layer(layer)
                    fallbacks += 1
                next_future = (
                    executor.submit(self._backend.load_layer, layers[index + 1])
                    if index + 1 < len(layers) else None
                )
                future = next_future
                future_layer = layers[index + 1] if next_future is not None else None
                try:
                    results.append(consume(layer, resource))
                finally:
                    self._backend.release_layer(layer, resource)
                    released += 1
        finally:
            if future is not None:
                if future.cancel():
                    future = None
                else:
                    try:
                        orphan = future.result()
                    except Exception:
                        pass
                    else:
                        assert future_layer is not None
                        self._backend.release_layer(future_layer, orphan)
                        released += 1
            executor.shutdown(wait=True, cancel_futures=True)
        return LayerPrefetchReport(
            tuple(results), len(layers), prefetched, failures, fallbacks, released
        )
