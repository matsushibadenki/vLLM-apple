"""Bounded importance reports derived from aggregate measurements."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum

MAX_IMPORTANCE_COMPONENTS = 65_536
MAX_IMPORTANCE_SAMPLES = (1 << 63) - 1


class ImportanceKind(str, Enum):
    LAYER = "layer"
    ATTENTION_HEAD = "attention_head"
    NEURON = "neuron"


@dataclass(frozen=True, order=True, slots=True)
class ImportanceKey:
    kind: ImportanceKind
    layer: int
    index: int | None = None

    def __post_init__(self) -> None:
        if (not isinstance(self.kind, ImportanceKind)
                or type(self.layer) is not int or not 0 <= self.layer < 1_000_000
                or (self.kind is ImportanceKind.LAYER) != (self.index is None)
                or (self.index is not None
                    and (type(self.index) is not int or not 0 <= self.index < 1_000_000))):
            raise ValueError("invalid importance key")


@dataclass(frozen=True, slots=True)
class ImportanceObservation:
    key: ImportanceKey
    samples: int
    mean_absolute_activation: float
    output_sensitivity: float

    def __post_init__(self) -> None:
        if (not isinstance(self.key, ImportanceKey)
                or type(self.samples) is not int or not 1 <= self.samples <= MAX_IMPORTANCE_SAMPLES
                or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                       or not math.isfinite(value) or value < 0
                       for value in (self.mean_absolute_activation, self.output_sensitivity))):
            raise ValueError("invalid importance observation")


@dataclass(slots=True)
class _Aggregate:
    samples: int
    weighted_activation: float
    weighted_sensitivity: float


class ImportanceAnalyzer:
    def __init__(self, maximum_components: int = 4096) -> None:
        if not 1 <= maximum_components <= MAX_IMPORTANCE_COMPONENTS:
            raise ValueError("invalid importance component bound")
        self._maximum_components = maximum_components
        self._aggregates: dict[ImportanceKey, _Aggregate] = {}

    def observe(self, observation: ImportanceObservation) -> None:
        if not isinstance(observation, ImportanceObservation):
            raise ValueError("invalid importance observation")
        aggregate = self._aggregates.get(observation.key)
        if aggregate is None:
            if len(self._aggregates) == self._maximum_components:
                raise ValueError("importance component bound exceeded")
            aggregate = self._aggregates[observation.key] = _Aggregate(0, 0.0, 0.0)
        if aggregate.samples + observation.samples > MAX_IMPORTANCE_SAMPLES:
            raise ValueError("importance sample count overflow")
        aggregate.samples += observation.samples
        aggregate.weighted_activation += observation.mean_absolute_activation * observation.samples
        aggregate.weighted_sensitivity += observation.output_sensitivity * observation.samples

    def report(self, *, maximum_results: int = 256) -> dict[str, object]:
        if type(maximum_results) is not int or not 1 <= maximum_results <= 4096:
            raise ValueError("invalid importance result bound")
        scored = []
        for key, aggregate in self._aggregates.items():
            activation = aggregate.weighted_activation / aggregate.samples
            sensitivity = aggregate.weighted_sensitivity / aggregate.samples
            score = activation * sensitivity
            scored.append((score, key, aggregate.samples, activation, sensitivity))
        scored.sort(key=lambda item: (-item[0], item[1]))
        score_total = sum(item[0] for item in scored)
        components = tuple({
            "kind": key.kind.value,
            "layer": key.layer,
            "index": key.index,
            "samples": samples,
            "mean_absolute_activation": activation,
            "output_sensitivity": sensitivity,
            "score": score,
            "normalized_score": score / score_total if score_total else 0.0,
        } for score, key, samples, activation, sensitivity in scored[:maximum_results])
        canonical = json.dumps(components, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return {
            "schema_version": 1,
            "component_count": len(scored),
            "returned_count": len(components),
            "truncated": len(scored) > len(components),
            "components": components,
            "report_id": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        }
