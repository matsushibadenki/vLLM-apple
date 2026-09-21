"""Bounded deterministic reference algorithms for weight optimization."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

MAX_OPTIMIZATION_VALUES = 65_536
MAX_CLUSTER_COUNT = 256
MAX_LOW_RANK = 64


@dataclass(frozen=True, slots=True)
class WeightClusteringResult:
    centroids: tuple[float, ...]
    assignments: tuple[int, ...]
    mean_squared_error: float
    maximum_absolute_error: float


@dataclass(frozen=True, slots=True)
class LowRankApproximation:
    rows: int
    columns: int
    rank: int
    left: tuple[tuple[float, ...], ...]
    right: tuple[tuple[float, ...], ...]
    frobenius_error: float


def cluster_weights(
    values: Sequence[int | float], *, clusters: int, iterations: int = 32
) -> WeightClusteringResult:
    numbers = _finite_vector(values)
    if (type(clusters) is not int or not 1 <= clusters <= min(MAX_CLUSTER_COUNT, len(numbers))
            or type(iterations) is not int or not 1 <= iterations <= 256):
        raise ValueError("invalid weight clustering configuration")
    ordered = sorted(numbers)
    centroids = [ordered[round(index * (len(ordered) - 1) / max(1, clusters - 1))]
                 for index in range(clusters)]
    assignments = [0] * len(numbers)
    for _ in range(iterations):
        updated = [min(range(clusters), key=lambda index: (abs(value - centroids[index]), index))
                   for value in numbers]
        sums = [0.0] * clusters
        counts = [0] * clusters
        for value, assignment in zip(numbers, updated, strict=True):
            sums[assignment] += value
            counts[assignment] += 1
        next_centroids = [sums[index] / counts[index] if counts[index] else centroids[index]
                          for index in range(clusters)]
        assignments = updated
        if next_centroids == centroids:
            break
        centroids = next_centroids
    errors = [value - centroids[assignment]
              for value, assignment in zip(numbers, assignments, strict=True)]
    return WeightClusteringResult(
        tuple(centroids), tuple(assignments),
        sum(error * error for error in errors) / len(errors),
        max(abs(error) for error in errors),
    )


def approximate_low_rank(
    matrix: Sequence[Sequence[int | float]], *, rank: int, iterations: int = 32
) -> LowRankApproximation:
    rows = tuple(_finite_vector(row) for row in matrix)
    if (not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows)
            or len(rows) * len(rows[0]) > MAX_OPTIMIZATION_VALUES
            or type(rank) is not int or not 1 <= rank <= min(MAX_LOW_RANK, len(rows), len(rows[0]))
            or type(iterations) is not int or not 1 <= iterations <= 256):
        raise ValueError("invalid low-rank approximation configuration")
    row_count, column_count = len(rows), len(rows[0])
    residual = [list(row) for row in rows]
    left_columns: list[list[float]] = []
    right_rows: list[list[float]] = []
    for component in range(rank):
        vector = [1.0 + ((index + component) % 7) / 7.0 for index in range(column_count)]
        vector = _normalize(vector)
        for _ in range(iterations):
            projected = [sum(residual[row][column] * vector[column]
                             for column in range(column_count)) for row in range(row_count)]
            next_vector = [sum(residual[row][column] * projected[row]
                               for row in range(row_count)) for column in range(column_count)]
            normalized = _normalize(next_vector)
            if normalized is None:
                vector = [0.0] * column_count
                break
            vector = normalized
        left_column = [sum(residual[row][column] * vector[column]
                           for column in range(column_count)) for row in range(row_count)]
        left_columns.append(left_column)
        right_rows.append(vector)
        for row in range(row_count):
            for column in range(column_count):
                residual[row][column] -= left_column[row] * vector[column]
    left = tuple(tuple(left_columns[column][row] for column in range(rank))
                 for row in range(row_count))
    error = math.sqrt(sum(value * value for row in residual for value in row))
    return LowRankApproximation(
        row_count, column_count, rank, left, tuple(tuple(row) for row in right_rows), error
    )


def _finite_vector(values: Sequence[int | float]) -> tuple[float, ...]:
    if not 1 <= len(values) <= MAX_OPTIMIZATION_VALUES:
        raise ValueError("weight vector exceeds bounds")
    result = []
    for value in values:
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value)):
            raise ValueError("weights must be finite numbers")
        result.append(float(value))
    return tuple(result)


def _normalize(values: Sequence[float]) -> list[float] | None:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return None
    return [value / norm for value in values]
