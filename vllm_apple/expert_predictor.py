"""Bounded expert prefetch predictor that cannot alter router decisions."""
from __future__ import annotations

import threading
from collections import Counter, deque
from dataclasses import dataclass

from .expert_residency import ExpertKey

MAX_PREDICTOR_CONTEXTS = 65_536
MAX_PREDICTOR_HISTORY = 65_536


@dataclass(frozen=True, slots=True)
class ExpertPrefetchHint:
    previous: ExpertKey
    candidates: tuple[ExpertKey, ...]
    observations: int


class CorrectnessNeutralExpertPredictor:
    """Learns bounded first-order transitions and emits non-authoritative hints."""

    def __init__(self, *, maximum_contexts: int = 4096, maximum_history: int = 4096) -> None:
        if (not 1 <= maximum_contexts <= MAX_PREDICTOR_CONTEXTS
                or not 1 <= maximum_history <= MAX_PREDICTOR_HISTORY):
            raise ValueError("invalid expert predictor bounds")
        self._maximum_contexts = maximum_contexts
        self._transitions: dict[ExpertKey, Counter[ExpertKey]] = {}
        self._order: deque[ExpertKey] = deque()
        self._history: deque[ExpertKey] = deque(maxlen=maximum_history)
        self._observations = self._evictions = 0
        self._lock = threading.Lock()

    def observe(self, selected: ExpertKey) -> None:
        if not isinstance(selected, ExpertKey):
            raise ValueError("invalid selected expert")
        with self._lock:
            if self._history:
                previous = self._history[-1]
                counter = self._transitions.get(previous)
                if counter is None:
                    if len(self._transitions) == self._maximum_contexts:
                        victim = self._order.popleft()
                        del self._transitions[victim]
                        self._evictions += 1
                    counter = self._transitions[previous] = Counter()
                    self._order.append(previous)
                counter[selected] += 1
                self._observations += 1
            self._history.append(selected)

    def predict(self, previous: ExpertKey, *, maximum_candidates: int = 2) -> ExpertPrefetchHint:
        if (not isinstance(previous, ExpertKey)
                or type(maximum_candidates) is not int
                or not 1 <= maximum_candidates <= 64):
            raise ValueError("invalid expert prediction request")
        with self._lock:
            counter = self._transitions.get(previous, Counter()).copy()
        ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        return ExpertPrefetchHint(
            previous=previous,
            candidates=tuple(key for key, _ in ranked[:maximum_candidates]),
            observations=sum(counter.values()),
        )

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "contexts": len(self._transitions),
                "history": len(self._history),
                "observations": self._observations,
                "evictions": self._evictions,
            }

    @staticmethod
    def resolve(router_selected: tuple[ExpertKey, ...], _hint: ExpertPrefetchHint) -> tuple[ExpertKey, ...]:
        if (not router_selected
                or len(router_selected) > 64
                or any(not isinstance(key, ExpertKey) for key in router_selected)):
            raise ValueError("invalid router selection")
        return router_selected
