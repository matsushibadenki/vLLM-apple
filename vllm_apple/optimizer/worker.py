from __future__ import annotations

import os
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .errors import OptimizerErrorCode
from .events import OptimizerEventBus, OptimizerState
from .safety import OptimizationPathError, validate_immutable_output_path
from .types import OPTIMIZER_SCHEMA_VERSION


DEFAULT_MAX_ARTIFACT_FILES = 100_000
DEFAULT_LOG_TAIL_BYTES = 64 * 1024


class ArtifactValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class WorkerResult:
    plan_id: str
    state: OptimizerState
    exit_code: int | None
    output_path: str | None
    output_bytes: int
    file_count: int
    stdout_tail: str
    stderr_tail: str
    error_code: OptimizerErrorCode | None = None

    def __post_init__(self) -> None:
        if not self.plan_id or self.output_bytes < 0 or self.file_count < 0:
            raise ValueError("invalid worker result")
        if self.state == OptimizerState.COMPLETED:
            if not self.output_path or self.file_count <= 0 or self.error_code is not None:
                raise ValueError("completed worker result requires a published artifact")
        elif self.state not in {OptimizerState.FAILED, OptimizerState.CANCELLED}:
            raise ValueError("worker result must be terminal")
        elif self.output_path is not None or self.output_bytes != 0 or self.file_count != 0:
            raise ValueError("failed worker result cannot publish an artifact")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": OPTIMIZER_SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "state": self.state.value,
            "exit_code": self.exit_code,
            "output_path": self.output_path,
            "output_bytes": self.output_bytes,
            "file_count": self.file_count,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "error_code": self.error_code.value if self.error_code else None,
        }


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()


class ArtifactTransaction:
    def __init__(
        self,
        source: Path,
        output: Path,
        *,
        maximum_output_bytes: int,
        maximum_files: int = DEFAULT_MAX_ARTIFACT_FILES,
    ) -> None:
        if maximum_output_bytes <= 0 or not 0 < maximum_files <= DEFAULT_MAX_ARTIFACT_FILES:
            raise ValueError("artifact limits must be positive and bounded")
        self.source = source.expanduser().resolve(strict=True)
        self.output = validate_immutable_output_path(self.source, output)
        self.maximum_output_bytes = maximum_output_bytes
        self.maximum_files = maximum_files
        self.workspace: Path | None = None
        self._promoted = False

    def __enter__(self) -> "ArtifactTransaction":
        parent = self.output.parent
        if not parent.is_dir():
            raise OptimizationPathError("artifact output parent must already exist")
        workspace = Path(tempfile.mkdtemp(prefix=f".{self.output.name}.work-", dir=parent))
        workspace.chmod(0o700)
        self.workspace = workspace
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        if not self._promoted and self.workspace is not None:
            shutil.rmtree(self.workspace, ignore_errors=True)

    def promote(self) -> tuple[int, int]:
        workspace = self._require_workspace()
        if self._promoted:
            raise RuntimeError("artifact transaction was already promoted")
        file_count, output_bytes = _validate_and_sync_tree(
            workspace,
            self.maximum_files,
            self.maximum_output_bytes,
        )
        lock_path = self.output.parent / f".{self.output.name}.promotion.lock"
        try:
            lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            raise OptimizationPathError("artifact promotion is already in progress") from error
        try:
            if self.output.exists():
                raise OptimizationPathError("immutable artifact output appeared during conversion")
            os.rename(workspace, self.output)
            _fsync_directory(self.output.parent)
        finally:
            os.close(lock_descriptor)
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            _fsync_directory(self.output.parent)
        self._promoted = True
        return file_count, output_bytes

    def _require_workspace(self) -> Path:
        if self.workspace is None:
            raise RuntimeError("artifact transaction is not active")
        return self.workspace


class IsolatedConversionWorker:
    def __init__(
        self,
        events: OptimizerEventBus | None = None,
        *,
        log_tail_bytes: int = DEFAULT_LOG_TAIL_BYTES,
        poll_interval: float = 0.05,
        terminate_grace_seconds: float = 2.0,
    ) -> None:
        if not 1024 <= log_tail_bytes <= 1024 * 1024:
            raise ValueError("worker log tail must be between 1 KiB and 1 MiB")
        if poll_interval <= 0 or terminate_grace_seconds <= 0:
            raise ValueError("worker timing values must be positive")
        self.events = events or OptimizerEventBus()
        self.log_tail_bytes = log_tail_bytes
        self.poll_interval = poll_interval
        self.terminate_grace_seconds = terminate_grace_seconds

    def run(
        self,
        *,
        plan_id: str,
        source: Path,
        output: Path,
        command: Sequence[str],
        maximum_output_bytes: int,
        cancellation: CancellationToken | None = None,
        timeout_seconds: float | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> WorkerResult:
        _validate_command(command)
        if not plan_id:
            raise ValueError("worker plan ID cannot be empty")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("worker timeout must be positive")
        worker_environment = _worker_environment(environment)
        token = cancellation or CancellationToken()
        self.events.publish(plan_id, "prepare", OptimizerState.READY, 0.0, "optimizer.ready")
        with ArtifactTransaction(
            source,
            output,
            maximum_output_bytes=maximum_output_bytes,
        ) as transaction:
            if token.is_cancelled:
                return self._cancelled_result(plan_id, "", "")
            workspace = transaction._require_workspace()
            self.events.publish(
                plan_id,
                "convert",
                OptimizerState.RUNNING,
                0.1,
                "optimizer.worker.running",
            )
            try:
                process = subprocess.Popen(
                    tuple(command),
                    cwd=workspace,
                    env=worker_environment,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                    close_fds=True,
                    start_new_session=True,
                )
            except OSError as error:
                return self._failed_result(
                    plan_id,
                    None,
                    "",
                    str(error),
                    OptimizerErrorCode.WORKER_CRASHED,
                )
            stdout = _BoundedPipe(process.stdout, self.log_tail_bytes)
            stderr = _BoundedPipe(process.stderr, self.log_tail_bytes)
            stdout.start()
            stderr.start()
            started = time.monotonic()
            cancelled = False
            timed_out = False
            while process.poll() is None:
                if token.is_cancelled:
                    cancelled = True
                    _stop_process(process, self.terminate_grace_seconds)
                    break
                if timeout_seconds is not None and time.monotonic() - started >= timeout_seconds:
                    timed_out = True
                    _stop_process(process, self.terminate_grace_seconds)
                    break
                time.sleep(self.poll_interval)
            exit_code = process.wait()
            stdout.join()
            stderr.join()
            stdout_tail = stdout.text()
            stderr_tail = stderr.text()
            if cancelled:
                return self._cancelled_result(plan_id, stdout_tail, stderr_tail, exit_code)
            if timed_out or exit_code != 0:
                detail = stderr_tail or ("worker timed out" if timed_out else "worker exited")
                return self._failed_result(
                    plan_id,
                    exit_code,
                    stdout_tail,
                    detail,
                    (
                        OptimizerErrorCode.WORKER_TIMEOUT
                        if timed_out
                        else OptimizerErrorCode.WORKER_CRASHED
                    ),
                )
            self.events.publish(
                plan_id,
                "validate",
                OptimizerState.RUNNING,
                0.8,
                "optimizer.worker.validating",
            )
            try:
                file_count, output_bytes = transaction.promote()
            except (ArtifactValidationError, OSError, OptimizationPathError) as error:
                return self._failed_result(
                    plan_id,
                    exit_code,
                    stdout_tail,
                    str(error),
                    OptimizerErrorCode.ARTIFACT_VALIDATION_FAILED,
                )
        self.events.publish(
            plan_id,
            "promote",
            OptimizerState.COMPLETED,
            1.0,
            "optimizer.worker.completed",
        )
        return WorkerResult(
            plan_id=plan_id,
            state=OptimizerState.COMPLETED,
            exit_code=exit_code,
            output_path=str(output.expanduser().resolve(strict=True)),
            output_bytes=output_bytes,
            file_count=file_count,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
        )

    def _cancelled_result(
        self,
        plan_id: str,
        stdout: str,
        stderr: str,
        exit_code: int | None = None,
    ) -> WorkerResult:
        self.events.publish(
            plan_id,
            "convert",
            OptimizerState.CANCELLED,
            0.0,
            "optimizer.worker.cancelled",
            OptimizerErrorCode.CANCELLED.value,
        )
        return WorkerResult(
            plan_id,
            OptimizerState.CANCELLED,
            exit_code,
            None,
            0,
            0,
            stdout,
            stderr,
            OptimizerErrorCode.CANCELLED,
        )

    def _failed_result(
        self,
        plan_id: str,
        exit_code: int | None,
        stdout: str,
        stderr: str,
        error_code: OptimizerErrorCode,
    ) -> WorkerResult:
        self.events.publish(
            plan_id,
            "convert",
            OptimizerState.FAILED,
            0.0,
            "optimizer.worker.failed",
            error_code.value,
        )
        return WorkerResult(
            plan_id,
            OptimizerState.FAILED,
            exit_code,
            None,
            0,
            0,
            stdout,
            stderr,
            error_code,
        )


class _BoundedPipe:
    def __init__(self, pipe: object, capacity: int) -> None:
        self._pipe = pipe
        self._capacity = capacity
        self._tail = bytearray()
        self._thread = threading.Thread(target=self._drain, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self) -> None:
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            raise RuntimeError("worker output drain did not finish")

    def text(self) -> str:
        return bytes(self._tail).decode("utf-8", errors="replace")

    def _drain(self) -> None:
        pipe = self._pipe
        if pipe is None or not hasattr(pipe, "read"):
            return
        try:
            while True:
                chunk = pipe.read(4096)
                if not chunk:
                    return
                self._tail.extend(chunk)
                overflow = len(self._tail) - self._capacity
                if overflow > 0:
                    del self._tail[:overflow]
        finally:
            pipe.close()


def _validate_command(command: Sequence[str]) -> None:
    if not command or len(command) > 128:
        raise ValueError("worker command must contain between 1 and 128 arguments")
    if any(not isinstance(argument, str) or not argument or "\0" in argument for argument in command):
        raise ValueError("worker command arguments must be non-empty strings without NUL")
    if sum(len(argument.encode("utf-8")) for argument in command) > 64 * 1024:
        raise ValueError("worker command exceeds the argument byte limit")


def _worker_environment(overrides: Mapping[str, str] | None) -> dict[str, str]:
    allowed = {name: os.environ[name] for name in ("PATH", "TMPDIR", "LANG") if name in os.environ}
    if overrides:
        if len(overrides) > 32 or any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or not key
            or "=" in key
            or "\0" in key
            or "\0" in value
            for key, value in overrides.items()
        ):
            raise ValueError("worker environment overrides are invalid or unbounded")
        if sum(len(key.encode()) + len(value.encode()) for key, value in overrides.items()) > 64 * 1024:
            raise ValueError("worker environment exceeds the byte limit")
        allowed.update(overrides)
    return allowed


def _stop_process(process: subprocess.Popen[bytes], grace_seconds: float) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _validate_and_sync_tree(root: Path, max_files: int, max_bytes: int) -> tuple[int, int]:
    file_count = 0
    entry_count = 0
    total_bytes = 0

    def visit(directory: Path, depth: int) -> None:
        nonlocal entry_count, file_count, total_bytes
        if depth > 256:
            raise ArtifactValidationError("artifact directory depth exceeds its limit")
        with os.scandir(directory) as entries:
            for entry in entries:
                entry_count += 1
                if entry_count > max_files:
                    raise ArtifactValidationError("artifact exceeds its entry limit")
                entry_stat = entry.stat(follow_symlinks=False)
                candidate = Path(entry.path)
                if stat.S_ISLNK(entry_stat.st_mode):
                    raise ArtifactValidationError("artifact entries cannot be symbolic links")
                if stat.S_ISDIR(entry_stat.st_mode):
                    visit(candidate, depth + 1)
                    continue
                if not stat.S_ISREG(entry_stat.st_mode):
                    raise ArtifactValidationError("artifact entries must be regular files")
                file_count += 1
                total_bytes += entry_stat.st_size
                if total_bytes > max_bytes:
                    raise ArtifactValidationError("artifact exceeds its byte limit")
                flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(candidate, flags)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        _fsync_directory(directory)

    visit(root, 0)
    if file_count == 0:
        raise ArtifactValidationError("artifact workspace is empty")
    return file_count, total_bytes


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
