"""Bounded reference pruning and functional-similarity experiments."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

MAX_STRUCTURAL_VALUES = 65_536


class StructuralComponentKind(str, Enum):
    LAYER = "layer"
    ATTENTION_HEAD = "attention_head"
    MLP = "mlp"


@dataclass(frozen=True, slots=True)
class PruningResult:
    structured: bool
    axis: int | None
    mask: tuple[bool, ...]
    retained_values: tuple[float, ...]
    pruned_count: int
    squared_error: float


@dataclass(frozen=True, slots=True)
class FunctionalSimilarity:
    kind: StructuralComponentKind
    cosine_similarity: float
    mean_squared_error: float
    maximum_absolute_error: float
    samples: int
    comparison_id: str


def prune_unstructured(
    values: Sequence[int | float], *, fraction: float
) -> PruningResult:
    numbers = _finite_vector(values)
    count = _pruned_count(len(numbers), fraction)
    removed = set(sorted(range(len(numbers)), key=lambda index: (abs(numbers[index]), index))[:count])
    mask = tuple(index not in removed for index in range(len(numbers)))
    retained = tuple(value if mask[index] else 0.0 for index, value in enumerate(numbers))
    return PruningResult(
        False, None, mask, retained, count,
        sum(numbers[index] ** 2 for index in removed),
    )


def prune_structured(
    matrix: Sequence[Sequence[int | float]], *, axis: int, fraction: float
) -> PruningResult:
    rows = tuple(_finite_vector(row) for row in matrix)
    if (not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows)
            or len(rows) * len(rows[0]) > MAX_STRUCTURAL_VALUES
            or axis not in (0, 1)):
        raise ValueError("invalid structured pruning matrix")
    group_count = len(rows) if axis == 0 else len(rows[0])
    count = _pruned_count(group_count, fraction)
    norms = []
    for group in range(group_count):
        values = rows[group] if axis == 0 else tuple(row[group] for row in rows)
        norms.append(sum(value * value for value in values))
    removed = set(sorted(range(group_count), key=lambda index: (norms[index], index))[:count])
    group_mask = tuple(index not in removed for index in range(group_count))
    retained = tuple(
        value if (row not in removed if axis == 0 else column not in removed) else 0.0
        for row, values in enumerate(rows) for column, value in enumerate(values)
    )
    return PruningResult(True, axis, group_mask, retained, count, sum(norms[index] for index in removed))


def analyze_functional_similarity(
    kind: StructuralComponentKind,
    left: Sequence[int | float],
    right: Sequence[int | float],
) -> FunctionalSimilarity:
    if not isinstance(kind, StructuralComponentKind):
        raise ValueError("invalid structural component kind")
    first, second = _finite_vector(left), _finite_vector(right)
    if len(first) != len(second):
        raise ValueError("functional outputs must have equal lengths")
    dot = sum(a * b for a, b in zip(first, second, strict=True))
    left_norm = math.sqrt(sum(value * value for value in first))
    right_norm = math.sqrt(sum(value * value for value in second))
    if first == second:
        cosine = 1.0
    elif left_norm == 0 or right_norm == 0:
        cosine = 0.0
    else:
        cosine = max(-1.0, min(1.0, dot / (left_norm * right_norm)))
    errors = tuple(a - b for a, b in zip(first, second, strict=True))
    identity = json.dumps({
        "kind": kind.value,
        "left": hashlib.sha256(_canonical(first)).hexdigest(),
        "right": hashlib.sha256(_canonical(second)).hexdigest(),
        "samples": len(first),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return FunctionalSimilarity(
        kind, cosine, sum(error * error for error in errors) / len(errors),
        max(abs(error) for error in errors), len(errors), hashlib.sha256(identity).hexdigest(),
    )


def _pruned_count(size: int, fraction: float) -> int:
    if (not isinstance(fraction, (int, float)) or isinstance(fraction, bool)
            or not math.isfinite(fraction) or not 0 <= fraction < 1):
        raise ValueError("invalid pruning fraction")
    return min(size - 1, math.floor(size * fraction))


def _finite_vector(values: Sequence[int | float]) -> tuple[float, ...]:
    if not 1 <= len(values) <= MAX_STRUCTURAL_VALUES:
        raise ValueError("structural vector exceeds bounds")
    result = []
    for value in values:
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value)):
            raise ValueError("structural values must be finite numbers")
        result.append(float(value))
    return tuple(result)


def _canonical(values: tuple[float, ...]) -> bytes:
    return json.dumps(values, separators=(",", ":"), allow_nan=False).encode("utf-8")
