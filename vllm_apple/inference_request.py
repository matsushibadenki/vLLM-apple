"""Bounded request context shared by HTTP admission and inference engines."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Protocol


class CancellationSignal(Protocol):
    def is_set(self) -> bool: ...


class InferenceRequestCancelled(RuntimeError):
    """Raised at a safe point after timeout or client disconnection."""


class InferenceEngineBusy(RuntimeError):
    """Raised when the bounded model-owner queue cannot accept more work."""


@dataclass(frozen=True, slots=True)
class InferenceRequestContext:
    request_id: str
    deadline: float
    cancellation: CancellationSignal

    def __post_init__(self) -> None:
        if (
            not isinstance(self.request_id, str)
            or not 1 <= len(self.request_id) <= 128
            or not math.isfinite(self.deadline)
            or not callable(getattr(self.cancellation, "is_set", None))
        ):
            raise ValueError("invalid inference request context")

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def raise_if_cancelled(self) -> None:
        if self.cancellation.is_set():
            raise InferenceRequestCancelled("inference request client disconnected")
        if self.remaining_seconds <= 0:
            raise InferenceRequestCancelled("inference request timed out")
