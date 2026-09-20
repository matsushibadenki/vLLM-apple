"""Bounded priority/deadline scheduler for audio pipeline work."""
from __future__ import annotations

import heapq
import math
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Generic, TypeVar


T = TypeVar("T")
R = TypeVar("R")


class AudioSchedulingPriority(IntEnum):
    REALTIME = 0
    INTERACTIVE = 1
    BACKGROUND = 2


@dataclass(frozen=True, slots=True)
class AudioScheduledTask(Generic[T]):
    task_id: str
    priority: AudioSchedulingPriority
    deadline: float
    estimated_duration_seconds: float
    payload: T

    def __post_init__(self) -> None:
        if (
            not self.task_id
            or len(self.task_id) > 128
            or not isinstance(self.priority, AudioSchedulingPriority)
            or not math.isfinite(self.deadline)
            or not math.isfinite(self.estimated_duration_seconds)
            or self.estimated_duration_seconds <= 0
        ):
            raise ValueError("invalid audio scheduled task")


@dataclass(frozen=True, slots=True)
class AudioTaskOutcome(Generic[R]):
    task_id: str
    status: str
    priority: AudioSchedulingPriority
    deadline: float
    started_at: float | None
    finished_at: float | None
    lateness_seconds: float
    result: R | None = None


@dataclass(frozen=True, slots=True)
class AudioSchedulerSnapshot:
    queue_depth: int
    submitted: int
    completed: int
    rejected: int
    cancelled: int
    deadline_misses: int
    maximum_lateness_seconds: float


class AudioDeadlineScheduler(Generic[T]):
    """Admission-controlled EDF queue within strict priority classes."""

    def __init__(
        self,
        *,
        maximum_tasks: int = 1024,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if type(maximum_tasks) is not int or not 1 <= maximum_tasks <= 65_536:
            raise ValueError("invalid audio scheduler capacity")
        self._maximum_tasks = maximum_tasks
        self._clock = clock
        self._heap: list[tuple[int, float, int, AudioScheduledTask[T]]] = []
        self._identifiers: set[str] = set()
        self._cancelled_ids: set[str] = set()
        self._sequence = 0
        self._submitted = 0
        self._completed = 0
        self._rejected = 0
        self._cancelled = 0
        self._deadline_misses = 0
        self._maximum_lateness_seconds = 0.0
        self._lock = threading.RLock()

    def submit(self, task: AudioScheduledTask[T]) -> bool:
        with self._lock:
            now = self._clock()
            if task.task_id in self._identifiers or len(self._identifiers) >= self._maximum_tasks:
                self._rejected += 1
                return False
            backlog = sum(
                queued.estimated_duration_seconds
                for _, _, _, queued in self._heap
                if queued.task_id not in self._cancelled_ids
                and queued.priority <= task.priority
                and queued.deadline <= task.deadline
            )
            if now + backlog + task.estimated_duration_seconds > task.deadline:
                self._rejected += 1
                return False
            heapq.heappush(
                self._heap,
                (int(task.priority), task.deadline, self._sequence, task),
            )
            self._sequence += 1
            self._identifiers.add(task.task_id)
            self._submitted += 1
            return True

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            if task_id not in self._identifiers or task_id in self._cancelled_ids:
                return False
            self._cancelled_ids.add(task_id)
            self._cancelled += 1
            return True

    def run_next(self, execute: Callable[[T], R]) -> AudioTaskOutcome[R] | None:
        with self._lock:
            task = self._pop_active_locked()
        if task is None:
            return None
        started = self._clock()
        if started > task.deadline:
            lateness = started - task.deadline
            with self._lock:
                self._deadline_misses += 1
                self._maximum_lateness_seconds = max(
                    self._maximum_lateness_seconds, lateness
                )
            return AudioTaskOutcome(
                task.task_id,
                "deadline_missed",
                task.priority,
                task.deadline,
                None,
                None,
                lateness,
            )
        result = execute(task.payload)
        finished = self._clock()
        lateness = max(0.0, finished - task.deadline)
        with self._lock:
            self._completed += 1
            if lateness:
                self._deadline_misses += 1
                self._maximum_lateness_seconds = max(
                    self._maximum_lateness_seconds, lateness
                )
        return AudioTaskOutcome(
            task.task_id,
            "completed" if not lateness else "completed_late",
            task.priority,
            task.deadline,
            started,
            finished,
            lateness,
            result,
        )

    def _pop_active_locked(self) -> AudioScheduledTask[T] | None:
        while self._heap:
            _, _, _, task = heapq.heappop(self._heap)
            self._identifiers.discard(task.task_id)
            if task.task_id in self._cancelled_ids:
                self._cancelled_ids.discard(task.task_id)
                continue
            return task
        return None

    @property
    def snapshot(self) -> AudioSchedulerSnapshot:
        with self._lock:
            return AudioSchedulerSnapshot(
                queue_depth=len(self._identifiers) - len(self._cancelled_ids),
                submitted=self._submitted,
                completed=self._completed,
                rejected=self._rejected,
                cancelled=self._cancelled,
                deadline_misses=self._deadline_misses,
                maximum_lateness_seconds=self._maximum_lateness_seconds,
            )
