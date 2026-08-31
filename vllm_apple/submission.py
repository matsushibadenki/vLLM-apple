from __future__ import annotations

import threading
from collections.abc import Callable
from time import monotonic
from typing import Generic, TypeVar, cast

from .scheduler import BasicScheduler, QueuedAdmissionError, Reservation, ScheduleRequest

_Result = TypeVar("_Result")


class SubmissionCancelledError(RuntimeError):
    pass


class SubmissionTimeoutError(TimeoutError):
    pass


class SubmissionHandle(Generic[_Result]):
    def __init__(self, token: str, cancel_pending: Callable[[str], bool]) -> None:
        self.token = token
        self._cancel_pending = cancel_pending
        self._done = threading.Event()
        self._cancel_requested = threading.Event()
        self._lock = threading.Lock()
        self._state = "pending"
        self._result: _Result | None = None
        self._error: BaseException | None = None

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def cancellation_requested(self) -> threading.Event:
        return self._cancel_requested

    def cancel(self) -> bool:
        with self._lock:
            if self._state != "pending":
                return False
        return self._cancel_pending(self.token)

    def request_cancellation(self) -> bool:
        with self._lock:
            if self._state not in {"pending", "running"}:
                return False
            self._cancel_requested.set()
            return True

    def result(self, timeout: float | None = None) -> _Result:
        if timeout is not None and timeout < 0:
            raise ValueError("submission timeout must not be negative")
        if not self._done.wait(timeout):
            raise SubmissionTimeoutError("submission did not complete before timeout")
        with self._lock:
            if self._state == "cancelled":
                raise SubmissionCancelledError("submission was cancelled")
            if self._error is not None:
                raise self._error
            return cast(_Result, self._result)

    def _begin(self) -> bool:
        with self._lock:
            if self._state != "pending":
                return False
            self._state = "running"
            return True

    def _cancelled(self) -> None:
        with self._lock:
            if self._state != "pending":
                return
            self._state = "cancelled"
            self._done.set()

    def _finish(self, result: _Result | None = None, error: BaseException | None = None) -> None:
        with self._lock:
            if self._state != "running":
                return
            self._result = result
            self._error = error
            self._state = "failed" if error is not None else "succeeded"
            self._done.set()


class GlobalSubmissionScheduler:
    """Bounded priority command execution isolated from application threads."""

    def __init__(self, scheduler: BasicScheduler, *, worker_count: int = 1) -> None:
        if worker_count <= 0 or worker_count > 8:
            raise ValueError("worker_count must be between 1 and 8")
        self.scheduler = scheduler
        self._worker_count = worker_count
        self._commands: dict[
            str, tuple[SubmissionHandle[object], Callable[[Reservation, threading.Event], object]]
        ] = {}
        self._condition = threading.Condition()
        self._workers: list[threading.Thread] = []
        self._stopping = False

    def start(self) -> None:
        with self._condition:
            if self._workers:
                return
            if self._stopping:
                raise RuntimeError("submission scheduler cannot restart after shutdown")
            for index in range(self._worker_count):
                worker = threading.Thread(
                    target=self._run_worker,
                    name=f"vllm-apple-submission-{index}",
                    daemon=True,
                )
                self._workers.append(worker)
                worker.start()

    def submit(
        self,
        request: ScheduleRequest,
        operation: Callable[[Reservation, threading.Event], _Result],
    ) -> SubmissionHandle[_Result]:
        with self._condition:
            if self._stopping:
                raise RuntimeError("submission scheduler is shutting down")
            token = self.scheduler.submit(request)
            handle: SubmissionHandle[_Result] = SubmissionHandle(token, self._cancel_pending)
            self._commands[token] = (handle, operation)  # type: ignore[assignment]
            self._condition.notify()
            return handle

    def shutdown(self, *, timeout: float = 5.0) -> bool:
        if timeout < 0:
            raise ValueError("shutdown timeout must not be negative")
        with self._condition:
            self._stopping = True
            pending = tuple(self._commands)
            self._condition.notify_all()
        for token in pending:
            self._cancel_pending(token)
        deadline = monotonic() + timeout
        for worker in tuple(self._workers):
            worker.join(max(0.0, deadline - monotonic()))
        return all(not worker.is_alive() for worker in self._workers)

    def snapshot(self) -> dict[str, int | bool]:
        with self._condition:
            return {
                "running": bool(self._workers) and not self._stopping,
                "workers": len(self._workers),
                "commands": len(self._commands),
            }

    def _cancel_pending(self, token: str) -> bool:
        with self._condition:
            command = self._commands.get(token)
            if command is None or command[0].state != "pending":
                return False
            if not self.scheduler.cancel(token):
                return False
            self._commands.pop(token, None)
            command[0]._cancelled()
            return True

    def _run_worker(self) -> None:
        while True:
            with self._condition:
                while not self._commands and not self._stopping:
                    self._condition.wait()
                if self._stopping and not self._commands:
                    return
            try:
                admitted = self.scheduler.admit_next(timeout=0)
            except QueuedAdmissionError as failure:
                with self._condition:
                    command = self._commands.pop(failure.token, None)
                if command is not None and command[0]._begin():
                    command[0]._finish(error=failure.error)
                continue
            if admitted is None:
                continue
            with self._condition:
                command = self._commands.get(admitted.token)
            if command is None:
                self.scheduler.complete_queued(admitted.token)
                continue
            handle, operation = command
            if not handle._begin():
                self.scheduler.complete_queued(admitted.token)
                with self._condition:
                    self._commands.pop(admitted.token, None)
                continue
            try:
                result = operation(admitted.reservation, handle.cancellation_requested)
            except BaseException as error:
                handle._finish(error=error)
            else:
                handle._finish(result=result)
            finally:
                self.scheduler.complete_queued(admitted.token)
                with self._condition:
                    self._commands.pop(admitted.token, None)
                    self._condition.notify_all()
