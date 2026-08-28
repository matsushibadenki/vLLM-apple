from __future__ import annotations

import hashlib
import math
import os
import resource
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .checkpoint import (
    CheckpointError,
    CheckpointManifest,
    CheckpointStage,
    CheckpointStore,
    ResumeAction,
    decide_resume,
    execution_fingerprint,
)
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
    output_hash: str | None
    elapsed_milliseconds: int
    peak_child_rss_bytes: int
    stdout_tail: str
    stderr_tail: str
    error_code: OptimizerErrorCode | None = None

    def __post_init__(self) -> None:
        if (
            not self.plan_id
            or self.output_bytes < 0
            or self.file_count < 0
            or self.elapsed_milliseconds < 0
            or self.peak_child_rss_bytes < 0
        ):
            raise ValueError("invalid worker result")
        if self.state == OptimizerState.COMPLETED:
            if (
                not self.output_path
                or self.file_count <= 0
                or self.error_code is not None
                or self.output_hash is None
                or len(self.output_hash) != 64
            ):
                raise ValueError("completed worker result requires a published artifact")
        elif self.state not in {OptimizerState.FAILED, OptimizerState.CANCELLED}:
            raise ValueError("worker result must be terminal")
        elif (
            self.output_path is not None
            or self.output_bytes != 0
            or self.file_count != 0
            or self.output_hash is not None
        ):
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
            "output_hash": self.output_hash,
            "elapsed_milliseconds": self.elapsed_milliseconds,
            "peak_child_rss_bytes": self.peak_child_rss_bytes,
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
        resume_workspace: Path | None = None,
    ) -> None:
        if maximum_output_bytes <= 0 or not 0 < maximum_files <= DEFAULT_MAX_ARTIFACT_FILES:
            raise ValueError("artifact limits must be positive and bounded")
        self.source = source.expanduser().resolve(strict=True)
        self.output = validate_immutable_output_path(self.source, output)
        self.maximum_output_bytes = maximum_output_bytes
        self.maximum_files = maximum_files
        self.resume_workspace = resume_workspace
        self.workspace: Path | None = None
        self._promoted = False
        self.output_hash: str | None = None

    def __enter__(self) -> "ArtifactTransaction":
        parent = self.output.parent
        if not parent.is_dir():
            raise OptimizationPathError("artifact output parent must already exist")
        if self.resume_workspace is not None:
            workspace = _validated_resume_workspace(self.resume_workspace, self.output)
        else:
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
        file_count, output_bytes, output_hash = _validate_and_sync_tree(
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
        self.output_hash = output_hash
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
        checkpoint_store: CheckpointStore | None = None,
        source_fingerprint: str | None = None,
        resume: bool = False,
        produced_subdirectory: str | None = None,
    ) -> WorkerResult:
        if checkpoint_store is None:
            if resume:
                raise CheckpointError("resume requires a checkpoint store")
            return self._run_once(
                plan_id=plan_id,
                source=source,
                output=output,
                command=command,
                maximum_output_bytes=maximum_output_bytes,
                cancellation=cancellation,
                timeout_seconds=timeout_seconds,
                environment=environment,
                produced_subdirectory=produced_subdirectory,
            )
        if source_fingerprint is None or len(source_fingerprint) < 16:
            raise CheckpointError("checkpointed worker requires a source fingerprint")
        source_resolved = source.expanduser().resolve(strict=True)
        if checkpoint_store.root == source_resolved or source_resolved in checkpoint_store.root.parents:
            raise CheckpointError("checkpoint root cannot be inside the source model")
        fingerprint_command = tuple(command) + (
            (f"produced_subdirectory={produced_subdirectory}",)
            if produced_subdirectory is not None
            else ()
        )
        fingerprint = execution_fingerprint(fingerprint_command, environment)
        with checkpoint_store.acquire(plan_id):
            checkpoint = checkpoint_store.load(plan_id)
            action = ResumeAction.RESTART_CONVERSION
            if checkpoint is None:
                checkpoint = CheckpointManifest.create(
                    plan_id=plan_id,
                    source_path=source_resolved,
                    source_fingerprint=source_fingerprint,
                    output_path=output,
                    execution_fingerprint=fingerprint,
                    maximum_output_bytes=maximum_output_bytes,
                )
                checkpoint_store.save(checkpoint)
            else:
                if not resume:
                    raise CheckpointError("checkpoint already exists; explicit resume is required")
                action = decide_resume(
                    checkpoint,
                    source_path=source_resolved,
                    source_fingerprint=source_fingerprint,
                    output_path=output,
                    execution_fingerprint_value=fingerprint,
                    maximum_output_bytes=maximum_output_bytes,
                )
                if action == ResumeAction.ALREADY_COMPLETED:
                    return self._completed_checkpoint_result(checkpoint)
                if action == ResumeAction.VERIFY_PROMOTED:
                    return self._reconcile_promoted(checkpoint_store, checkpoint)
                if action == ResumeAction.RESTART_CONVERSION:
                    checkpoint = checkpoint.next_attempt()
                    checkpoint_store.save(checkpoint)

            def converted(
                workspace: Path,
                elapsed_milliseconds: int,
                peak_child_rss_bytes: int,
            ) -> None:
                nonlocal checkpoint
                checkpoint = checkpoint.transition(
                    CheckpointStage.CONVERTED,
                    workspace_path=workspace,
                    elapsed_milliseconds=elapsed_milliseconds,
                    peak_child_rss_bytes=peak_child_rss_bytes,
                )
                checkpoint_store.save(checkpoint)

            def terminal(result: WorkerResult) -> None:
                nonlocal checkpoint
                if result.state == OptimizerState.COMPLETED:
                    checkpoint = checkpoint.transition(
                        CheckpointStage.COMPLETED,
                        output_bytes=result.output_bytes,
                        file_count=result.file_count,
                        output_hash=result.output_hash,
                        elapsed_milliseconds=result.elapsed_milliseconds,
                        peak_child_rss_bytes=result.peak_child_rss_bytes,
                    )
                else:
                    stage = (
                        CheckpointStage.CANCELLED
                        if result.state == OptimizerState.CANCELLED
                        else CheckpointStage.FAILED
                    )
                    checkpoint = checkpoint.transition(
                        stage,
                        last_error_code=(result.error_code.value if result.error_code else "unknown"),
                        elapsed_milliseconds=result.elapsed_milliseconds,
                        peak_child_rss_bytes=result.peak_child_rss_bytes,
                    )
                checkpoint_store.save(checkpoint)

            return self._run_once(
                plan_id=plan_id,
                source=source_resolved,
                output=output,
                command=command,
                maximum_output_bytes=maximum_output_bytes,
                cancellation=cancellation,
                timeout_seconds=timeout_seconds,
                environment=environment,
                produced_subdirectory=produced_subdirectory,
                resume_workspace=(
                    Path(checkpoint.workspace_path)
                    if action == ResumeAction.RESUME_VALIDATION
                    and checkpoint.workspace_path is not None
                    else None
                ),
                on_converted=converted,
                on_terminal=terminal,
                prior_elapsed_milliseconds=checkpoint.elapsed_milliseconds,
                prior_peak_child_rss_bytes=checkpoint.peak_child_rss_bytes,
            )

    def _run_once(
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
        resume_workspace: Path | None = None,
        on_converted: Callable[[Path, int, int], None] | None = None,
        on_terminal: Callable[[WorkerResult], None] | None = None,
        prior_elapsed_milliseconds: int = 0,
        prior_peak_child_rss_bytes: int = 0,
        produced_subdirectory: str | None = None,
    ) -> WorkerResult:
        _validate_command(command)
        if not plan_id:
            raise ValueError("worker plan ID cannot be empty")
        if timeout_seconds is not None and (
            not math.isfinite(timeout_seconds) or timeout_seconds <= 0
        ):
            raise ValueError("worker timeout must be positive")
        worker_environment = _worker_environment(environment)
        token = cancellation or CancellationToken()
        run_started = time.monotonic()
        peak_child_rss_bytes = prior_peak_child_rss_bytes

        def measurements() -> tuple[int, int]:
            elapsed = prior_elapsed_milliseconds + max(
                0,
                math.ceil((time.monotonic() - run_started) * 1000),
            )
            return elapsed, max(peak_child_rss_bytes, _children_peak_rss_bytes())

        self.events.publish(plan_id, "prepare", OptimizerState.READY, 0.0, "optimizer.ready")
        with ArtifactTransaction(
            source,
            output,
            maximum_output_bytes=maximum_output_bytes,
            resume_workspace=resume_workspace,
        ) as transaction:
            if token.is_cancelled:
                elapsed, peak_rss = measurements()
                result = self._cancelled_result(plan_id, "", "", elapsed=elapsed, peak_rss=peak_rss)
                if on_terminal:
                    on_terminal(result)
                return result
            workspace = transaction._require_workspace()
            if resume_workspace is None:
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
                    result = self._failed_result(
                        plan_id,
                        None,
                        "",
                        str(error),
                        OptimizerErrorCode.WORKER_CRASHED,
                    )
                    if on_terminal:
                        on_terminal(result)
                    return result
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
                peak_child_rss_bytes = max(peak_child_rss_bytes, _children_peak_rss_bytes())
                stdout.join()
                stderr.join()
                stdout_tail = stdout.text()
                stderr_tail = stderr.text()
                if cancelled:
                    elapsed, peak_rss = measurements()
                    result = self._cancelled_result(
                        plan_id,
                        stdout_tail,
                        stderr_tail,
                        exit_code,
                        elapsed=elapsed,
                        peak_rss=peak_rss,
                    )
                    if on_terminal:
                        on_terminal(result)
                    return result
                if timed_out or exit_code != 0:
                    detail = stderr_tail or ("worker timed out" if timed_out else "worker exited")
                    elapsed, peak_rss = measurements()
                    result = self._failed_result(
                        plan_id,
                        exit_code,
                        stdout_tail,
                        detail,
                        (
                            OptimizerErrorCode.WORKER_TIMEOUT
                            if timed_out
                            else OptimizerErrorCode.WORKER_CRASHED
                        ),
                        elapsed=elapsed,
                        peak_rss=peak_rss,
                    )
                    if on_terminal:
                        on_terminal(result)
                    return result
                if produced_subdirectory is not None:
                    try:
                        _flatten_produced_subdirectory(workspace, produced_subdirectory)
                    except (ArtifactValidationError, OSError) as error:
                        elapsed, peak_rss = measurements()
                        result = self._failed_result(
                            plan_id,
                            exit_code,
                            stdout_tail,
                            str(error),
                            OptimizerErrorCode.ARTIFACT_VALIDATION_FAILED,
                            elapsed=elapsed,
                            peak_rss=peak_rss,
                        )
                        if on_terminal:
                            on_terminal(result)
                        return result
                if on_converted:
                    elapsed, peak_rss = measurements()
                    on_converted(workspace, elapsed, peak_rss)
                if token.is_cancelled:
                    elapsed, peak_rss = measurements()
                    result = self._cancelled_result(
                        plan_id,
                        stdout_tail,
                        stderr_tail,
                        exit_code,
                        elapsed=elapsed,
                        peak_rss=peak_rss,
                    )
                    if on_terminal:
                        on_terminal(result)
                    return result
            else:
                exit_code = 0
                stdout_tail = ""
                stderr_tail = ""
                self.events.publish(
                    plan_id,
                    "resume",
                    OptimizerState.RUNNING,
                    0.75,
                    "optimizer.worker.resuming_validation",
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
                elapsed, peak_rss = measurements()
                result = self._failed_result(
                    plan_id,
                    exit_code,
                    stdout_tail,
                    str(error),
                    OptimizerErrorCode.ARTIFACT_VALIDATION_FAILED,
                    elapsed=elapsed,
                    peak_rss=peak_rss,
                )
                if on_terminal:
                    on_terminal(result)
                return result
        self.events.publish(
            plan_id,
            "promote",
            OptimizerState.COMPLETED,
            1.0,
            "optimizer.worker.completed",
        )
        elapsed, peak_rss = measurements()
        result = WorkerResult(
            plan_id=plan_id,
            state=OptimizerState.COMPLETED,
            exit_code=exit_code,
            output_path=str(output.expanduser().resolve(strict=True)),
            output_bytes=output_bytes,
            file_count=file_count,
            output_hash=transaction.output_hash,
            elapsed_milliseconds=elapsed,
            peak_child_rss_bytes=peak_rss,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
        )
        if on_terminal:
            on_terminal(result)
        return result

    def _completed_checkpoint_result(self, checkpoint: CheckpointManifest) -> WorkerResult:
        self.events.publish(
            checkpoint.plan_id,
            "resume",
            OptimizerState.COMPLETED,
            1.0,
            "optimizer.worker.already_completed",
        )
        return WorkerResult(
            plan_id=checkpoint.plan_id,
            state=OptimizerState.COMPLETED,
            exit_code=0,
            output_path=checkpoint.output_path,
            output_bytes=checkpoint.output_bytes,
            file_count=checkpoint.file_count,
            output_hash=checkpoint.output_hash,
            elapsed_milliseconds=checkpoint.elapsed_milliseconds,
            peak_child_rss_bytes=checkpoint.peak_child_rss_bytes,
            stdout_tail="",
            stderr_tail="",
        )

    def _reconcile_promoted(
        self,
        store: CheckpointStore,
        checkpoint: CheckpointManifest,
    ) -> WorkerResult:
        started = time.monotonic()
        file_count, output_bytes, output_hash = _validate_and_sync_tree(
            Path(checkpoint.output_path),
            DEFAULT_MAX_ARTIFACT_FILES,
            checkpoint.maximum_output_bytes,
        )
        completed = checkpoint.transition(
            CheckpointStage.COMPLETED,
            output_bytes=output_bytes,
            file_count=file_count,
            output_hash=output_hash,
            elapsed_milliseconds=checkpoint.elapsed_milliseconds
            + max(0, math.ceil((time.monotonic() - started) * 1000)),
            peak_child_rss_bytes=max(
                checkpoint.peak_child_rss_bytes,
                _children_peak_rss_bytes(),
            ),
        )
        store.save(completed)
        return self._completed_checkpoint_result(completed)

    def _cancelled_result(
        self,
        plan_id: str,
        stdout: str,
        stderr: str,
        exit_code: int | None = None,
        *,
        elapsed: int = 0,
        peak_rss: int = 0,
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
            plan_id=plan_id,
            state=OptimizerState.CANCELLED,
            exit_code=exit_code,
            output_path=None,
            output_bytes=0,
            file_count=0,
            output_hash=None,
            elapsed_milliseconds=elapsed,
            peak_child_rss_bytes=peak_rss,
            stdout_tail=stdout,
            stderr_tail=stderr,
            error_code=OptimizerErrorCode.CANCELLED,
        )

    def _failed_result(
        self,
        plan_id: str,
        exit_code: int | None,
        stdout: str,
        stderr: str,
        error_code: OptimizerErrorCode,
        *,
        elapsed: int = 0,
        peak_rss: int = 0,
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
            plan_id=plan_id,
            state=OptimizerState.FAILED,
            exit_code=exit_code,
            output_path=None,
            output_bytes=0,
            file_count=0,
            output_hash=None,
            elapsed_milliseconds=elapsed,
            peak_child_rss_bytes=peak_rss,
            stdout_tail=stdout,
            stderr_tail=stderr,
            error_code=error_code,
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


def _validate_and_sync_tree(root: Path, max_files: int, max_bytes: int) -> tuple[int, int, str]:
    root_info = root.lstat()
    if (
        stat.S_ISLNK(root_info.st_mode)
        or not stat.S_ISDIR(root_info.st_mode)
        or root_info.st_uid != os.getuid()
        or stat.S_IMODE(root_info.st_mode) & 0o077
    ):
        raise ArtifactValidationError("artifact root must be a private owned directory")
    file_count = 0
    entry_count = 0
    total_bytes = 0
    file_digests: list[bytes] = []

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
                    opened = os.fstat(descriptor)
                    if (
                        opened.st_dev != entry_stat.st_dev
                        or opened.st_ino != entry_stat.st_ino
                        or opened.st_size != entry_stat.st_size
                    ):
                        raise ArtifactValidationError("artifact changed during validation")
                    relative = str(candidate.relative_to(root)).encode("utf-8")
                    file_digest = hashlib.sha256(b"vllm-apple-artifact-file-v1\0")
                    file_digest.update(len(relative).to_bytes(4, "big"))
                    file_digest.update(relative)
                    file_digest.update(opened.st_size.to_bytes(8, "big"))
                    while True:
                        chunk = os.read(descriptor, 8 * 1024 * 1024)
                        if not chunk:
                            break
                        file_digest.update(chunk)
                    file_digests.append(file_digest.digest())
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        _fsync_directory(directory)

    visit(root, 0)
    if file_count == 0:
        raise ArtifactValidationError("artifact workspace is empty")
    digest = hashlib.sha256(b"vllm-apple-artifact-tree-v1\0")
    for file_digest in sorted(file_digests):
        digest.update(file_digest)
    return file_count, total_bytes, digest.hexdigest()


def _children_peak_rss_bytes() -> int:
    value = max(0, int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss))
    return value if sys.platform == "darwin" else value * 1024


def _flatten_produced_subdirectory(workspace: Path, name: str) -> None:
    if (
        not name
        or name in {".", ".."}
        or Path(name).name != name
        or len(name.encode("utf-8")) > 255
    ):
        raise ArtifactValidationError("produced subdirectory name is unsafe")
    with os.scandir(workspace) as scanned:
        entries = list(scanned)
    if len(entries) != 1 or entries[0].name != name:
        raise ArtifactValidationError("worker did not produce the expected artifact directory")
    produced = workspace / name
    info = produced.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise ArtifactValidationError("produced artifact directory is unsafe")
    with os.scandir(produced) as scanned:
        children = list(scanned)
    if not children:
        raise ArtifactValidationError("produced artifact directory is empty")
    if len(children) > DEFAULT_MAX_ARTIFACT_FILES:
        raise ArtifactValidationError("produced artifact exceeds its entry limit")
    for child in children:
        os.rename(child.path, workspace / child.name)
    produced.rmdir()
    _fsync_directory(workspace)


def _validated_resume_workspace(workspace: Path, output: Path) -> Path:
    candidate = workspace.expanduser()
    try:
        info = candidate.lstat()
    except FileNotFoundError as error:
        raise ArtifactValidationError("resume workspace is unavailable") from error
    resolved = candidate.resolve(strict=True)
    expected_prefix = f".{output.name}.work-"
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) & 0o077
        or resolved.parent != output.parent.resolve(strict=True)
        or not resolved.name.startswith(expected_prefix)
        or len(resolved.name) == len(expected_prefix)
    ):
        raise ArtifactValidationError("resume workspace is unsafe")
    return resolved


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
