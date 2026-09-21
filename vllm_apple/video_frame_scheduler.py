"""Bounded presentation-time scheduler for decoded video frames."""
from __future__ import annotations

import heapq
import math
import threading
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ScheduledVideoFrame(Generic[T]):
    frame_id: str
    presentation_seconds: float
    duration_seconds: float
    payload: T
    keyframe: bool = False

    def __post_init__(self) -> None:
        if (
            not self.frame_id
            or len(self.frame_id) > 128
            or not math.isfinite(self.presentation_seconds)
            or self.presentation_seconds < 0
            or not math.isfinite(self.duration_seconds)
            or self.duration_seconds <= 0
        ):
            raise ValueError("invalid scheduled video frame")


@dataclass(frozen=True, slots=True)
class VideoFrameDecision(Generic[T]):
    frame: ScheduledVideoFrame[T]
    action: str
    lateness_seconds: float


@dataclass(frozen=True, slots=True)
class VideoFrameSchedulerSnapshot:
    queue_depth: int
    submitted: int
    presented: int
    late_drops: int
    rejected: int
    cancelled: int
    maximum_lateness_seconds: float


class VideoFrameScheduler(Generic[T]):
    """PTS-ordered scheduler that drops stale live frames before expensive inference."""

    def __init__(
        self,
        *,
        maximum_frames: int = 256,
        maximum_lateness_seconds: float = 0.1,
        maximum_reorder_seconds: float = 1.0,
    ) -> None:
        if (
            type(maximum_frames) is not int
            or not 1 <= maximum_frames <= 65_536
            or not math.isfinite(maximum_lateness_seconds)
            or maximum_lateness_seconds < 0
            or not math.isfinite(maximum_reorder_seconds)
            or maximum_reorder_seconds < 0
        ):
            raise ValueError("invalid video frame scheduler configuration")
        self._maximum_frames = maximum_frames
        self._maximum_lateness_seconds = maximum_lateness_seconds
        self._maximum_reorder_seconds = maximum_reorder_seconds
        self._heap: list[tuple[float, int, ScheduledVideoFrame[T]]] = []
        self._identifiers: set[str] = set()
        self._cancelled_ids: set[str] = set()
        self._sequence = 0
        self._latest_submitted_pts = 0.0
        self._submitted = 0
        self._presented = 0
        self._late_drops = 0
        self._rejected = 0
        self._cancelled = 0
        self._maximum_lateness = 0.0
        self._lock = threading.RLock()

    def submit(self, frame: ScheduledVideoFrame[T]) -> bool:
        with self._lock:
            if (
                frame.frame_id in self._identifiers
                or len(self._identifiers) >= self._maximum_frames
                or frame.presentation_seconds
                < self._latest_submitted_pts - self._maximum_reorder_seconds
            ):
                self._rejected += 1
                return False
            heapq.heappush(
                self._heap,
                (frame.presentation_seconds, self._sequence, frame),
            )
            self._sequence += 1
            self._identifiers.add(frame.frame_id)
            self._latest_submitted_pts = max(
                self._latest_submitted_pts, frame.presentation_seconds
            )
            self._submitted += 1
            return True

    def cancel(self, frame_id: str) -> bool:
        with self._lock:
            if frame_id not in self._identifiers or frame_id in self._cancelled_ids:
                return False
            self._cancelled_ids.add(frame_id)
            self._cancelled += 1
            return True

    def pop_ready(self, media_time_seconds: float) -> VideoFrameDecision[T] | None:
        if not math.isfinite(media_time_seconds) or media_time_seconds < 0:
            raise ValueError("invalid video media time")
        with self._lock:
            while self._heap:
                presentation, _, frame = self._heap[0]
                if presentation > media_time_seconds:
                    return None
                heapq.heappop(self._heap)
                self._identifiers.discard(frame.frame_id)
                if frame.frame_id in self._cancelled_ids:
                    self._cancelled_ids.discard(frame.frame_id)
                    continue
                lateness = max(0.0, media_time_seconds - presentation)
                self._maximum_lateness = max(self._maximum_lateness, lateness)
                if lateness > self._maximum_lateness_seconds:
                    self._late_drops += 1
                    return VideoFrameDecision(frame, "dropped_late", lateness)
                self._presented += 1
                return VideoFrameDecision(frame, "present", lateness)
            return None

    def drain_late(self, media_time_seconds: float) -> tuple[VideoFrameDecision[T], ...]:
        decisions = []
        while True:
            decision = self.pop_ready(media_time_seconds)
            if decision is None:
                break
            decisions.append(decision)
            if decision.action == "present":
                break
        return tuple(decisions)

    @property
    def snapshot(self) -> VideoFrameSchedulerSnapshot:
        with self._lock:
            return VideoFrameSchedulerSnapshot(
                queue_depth=len(self._identifiers) - len(self._cancelled_ids),
                submitted=self._submitted,
                presented=self._presented,
                late_drops=self._late_drops,
                rejected=self._rejected,
                cancelled=self._cancelled,
                maximum_lateness_seconds=self._maximum_lateness,
            )
