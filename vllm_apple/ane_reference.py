"""Deterministic bounded references for representative ANE fixed graphs."""
from __future__ import annotations

import math

MAX_ENCODER_WIDTH = 1024
MAX_ENCODER_LAYERS = 16


def encoder_weight(layer: int, row: int, column: int) -> float:
    return float(((layer + 1) * 7 + row * 17 + column * 13) % 29 - 14) / 16384.0


def encoder_bias(layer: int, row: int) -> float:
    return float(((layer + 1) * 5 + row * 3) % 11 - 5) / 1024.0


def representative_encoder_input(width: int) -> tuple[float, ...]:
    _validate_encoder_shape(width, 1)
    return tuple(float((index * 11) % 31 - 15) / 16.0 for index in range(width))


def run_encoder_reference(
    values: tuple[float, ...], *, layers: int
) -> tuple[float, ...]:
    width = len(values)
    _validate_encoder_shape(width, layers)
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError("encoder input must be finite")
    state = tuple(float(value) for value in values)
    for layer in range(layers):
        state = tuple(
            max(0.0, sum(
                state[column] * encoder_weight(layer, row, column)
                for column in range(width)
            ) + encoder_bias(layer, row))
            for row in range(width)
        )
    return state


def _validate_encoder_shape(width: int, layers: int) -> None:
    if (
        type(width) is not int or not 1 <= width <= MAX_ENCODER_WIDTH
        or type(layers) is not int or not 1 <= layers <= MAX_ENCODER_LAYERS
    ):
        raise ValueError("encoder shape exceeds its bound")
