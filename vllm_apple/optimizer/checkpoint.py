from __future__ import annotations

import hashlib
import json
import fcntl
import os
import stat
import tempfile
import threading
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

from .safety import validate_immutable_output_path
from .types import OPTIMIZER_SCHEMA_VERSION


MAX_CHECKPOINT_BYTES = 64 * 1024


class CheckpointError(ValueError):
    pass


class CheckpointLeaseError(CheckpointError):
    pass


class CheckpointStage(str, Enum):
    PREPARED = "prepared"
    CONVERTED = "converted"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResumeAction(str, Enum):
    RESTART_CONVERSION = "restart_conversion"
    RESUME_VALIDATION = "resume_validation"
    VERIFY_PROMOTED = "verify_promoted"
    ALREADY_COMPLETED = "already_completed"


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    plan_id: str
    source_path: str
    source_fingerprint: str
    output_path: str
    execution_fingerprint: str
    maximum_output_bytes: int
    stage: CheckpointStage
    attempt: int
    updated_at: str
    workspace_path: str | None = None
    output_bytes: int = 0
    file_count: int = 0
    output_hash: str | None = None
    elapsed_milliseconds: int = 0
    peak_child_rss_bytes: int = 0
    last_error_code: str | None = None

    def __post_init__(self) -> None:
        if not self.plan_id or len(self.source_fingerprint) < 16:
            raise CheckpointError("invalid checkpoint plan or source fingerprint")
        if len(self.execution_fingerprint) != 64:
            raise CheckpointError("invalid checkpoint execution fingerprint")
        if not Path(self.source_path).is_absolute() or not Path(self.output_path).is_absolute():
            raise CheckpointError("checkpoint source and output paths must be absolute")
        if self.maximum_output_bytes <= 0 or self.attempt <= 0:
            raise CheckpointError("checkpoint budget and attempt must be positive")
        if (
            self.output_bytes < 0
            or self.file_count < 0
            or self.elapsed_milliseconds < 0
            or self.peak_child_rss_bytes < 0
            or not self.updated_at
        ):
            raise CheckpointError("invalid checkpoint result metadata")
        try:
            timestamp = datetime.fromisoformat(self.updated_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise CheckpointError("checkpoint timestamp is invalid") from error
        if timestamp.tzinfo is None:
            raise CheckpointError("checkpoint timestamp must include a timezone")
        if self.last_error_code is not None and (
            not self.last_error_code or len(self.last_error_code) > 128
        ):
            raise CheckpointError("checkpoint error code is invalid")
        if self.workspace_path is not None and not Path(self.workspace_path).is_absolute():
            raise CheckpointError("checkpoint workspace path must be absolute")
        if self.stage == CheckpointStage.CONVERTED:
            if self.workspace_path is None or self.last_error_code is not None:
                raise CheckpointError("converted checkpoint requires a workspace without an error")
        elif self.stage == CheckpointStage.COMPLETED:
            if (
                self.workspace_path is not None
                or self.output_bytes <= 0
                or self.file_count <= 0
                or self.output_hash is None
                or len(self.output_hash) != 64
                or self.last_error_code is not None
            ):
                raise CheckpointError("completed checkpoint requires artifact metadata")
        elif self.output_bytes != 0 or self.file_count != 0 or self.output_hash is not None:
            raise CheckpointError("unfinished checkpoint cannot contain artifact result metadata")

    @classmethod
    def create(
        cls,
        *,
        plan_id: str,
        source_path: Path,
        source_fingerprint: str,
        output_path: Path,
        execution_fingerprint: str,
        maximum_output_bytes: int,
        attempt: int = 1,
    ) -> "CheckpointManifest":
        source = source_path.expanduser().resolve(strict=True)
        output = validate_immutable_output_path(source, output_path)
        return cls(
            plan_id=plan_id,
            source_path=str(source),
            source_fingerprint=source_fingerprint,
            output_path=str(output),
            execution_fingerprint=execution_fingerprint,
            maximum_output_bytes=maximum_output_bytes,
            stage=CheckpointStage.PREPARED,
            attempt=attempt,
            updated_at=_now(),
        )

    def transition(
        self,
        stage: CheckpointStage,
        *,
        workspace_path: Path | None = None,
        output_bytes: int = 0,
        file_count: int = 0,
        output_hash: str | None = None,
        elapsed_milliseconds: int = 0,
        peak_child_rss_bytes: int = 0,
        last_error_code: str | None = None,
    ) -> "CheckpointManifest":
        return replace(
            self,
            stage=stage,
            updated_at=_now(),
            workspace_path=(
                str(workspace_path.expanduser().resolve(strict=True))
                if workspace_path is not None
                else None
            ),
            output_bytes=output_bytes,
            file_count=file_count,
            output_hash=output_hash,
            elapsed_milliseconds=elapsed_milliseconds,
            peak_child_rss_bytes=peak_child_rss_bytes,
            last_error_code=last_error_code,
        )

    def next_attempt(self) -> "CheckpointManifest":
        return replace(
            self,
            stage=CheckpointStage.PREPARED,
            attempt=self.attempt + 1,
            updated_at=_now(),
            workspace_path=None,
            output_bytes=0,
            file_count=0,
            output_hash=None,
            elapsed_milliseconds=0,
            peak_child_rss_bytes=0,
            last_error_code=None,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": OPTIMIZER_SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "source_path": self.source_path,
            "source_fingerprint": self.source_fingerprint,
            "output_path": self.output_path,
            "execution_fingerprint": self.execution_fingerprint,
            "maximum_output_bytes": self.maximum_output_bytes,
            "stage": self.stage.value,
            "attempt": self.attempt,
            "updated_at": self.updated_at,
            "workspace_path": self.workspace_path,
            "output_bytes": self.output_bytes,
            "file_count": self.file_count,
            "output_hash": self.output_hash,
            "elapsed_milliseconds": self.elapsed_milliseconds,
            "peak_child_rss_bytes": self.peak_child_rss_bytes,
            "last_error_code": self.last_error_code,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "CheckpointManifest":
        fields = {
            "schema_version",
            "plan_id",
            "source_path",
            "source_fingerprint",
            "output_path",
            "execution_fingerprint",
            "maximum_output_bytes",
            "stage",
            "attempt",
            "updated_at",
            "workspace_path",
            "output_bytes",
            "file_count",
            "output_hash",
            "elapsed_milliseconds",
            "peak_child_rss_bytes",
            "last_error_code",
        }
        if set(payload) != fields or payload.get("schema_version") != OPTIMIZER_SCHEMA_VERSION:
            raise CheckpointError("unsupported or malformed checkpoint schema")
        string_fields = (
            "plan_id",
            "source_path",
            "source_fingerprint",
            "output_path",
            "execution_fingerprint",
            "updated_at",
        )
        nullable_strings = ("workspace_path", "output_hash", "last_error_code")
        integer_fields = (
            "maximum_output_bytes",
            "attempt",
            "output_bytes",
            "file_count",
            "elapsed_milliseconds",
            "peak_child_rss_bytes",
        )
        if any(not isinstance(payload[name], str) for name in string_fields):
            raise CheckpointError("checkpoint string fields are invalid")
        if any(
            payload[name] is not None and not isinstance(payload[name], str)
            for name in nullable_strings
        ):
            raise CheckpointError("checkpoint nullable string fields are invalid")
        if any(
            not isinstance(payload[name], int) or isinstance(payload[name], bool)
            for name in integer_fields
        ):
            raise CheckpointError("checkpoint integer fields are invalid")
        try:
            stage = CheckpointStage(payload["stage"])
        except (TypeError, ValueError) as error:
            raise CheckpointError("checkpoint stage is invalid") from error
        return cls(
            plan_id=payload["plan_id"],
            source_path=payload["source_path"],
            source_fingerprint=payload["source_fingerprint"],
            output_path=payload["output_path"],
            execution_fingerprint=payload["execution_fingerprint"],
            maximum_output_bytes=payload["maximum_output_bytes"],
            stage=stage,
            attempt=payload["attempt"],
            updated_at=payload["updated_at"],
            workspace_path=payload["workspace_path"],
            output_bytes=payload["output_bytes"],
            file_count=payload["file_count"],
            output_hash=payload["output_hash"],
            elapsed_milliseconds=payload["elapsed_milliseconds"],
            peak_child_rss_bytes=payload["peak_child_rss_bytes"],
            last_error_code=payload["last_error_code"],
        )


class CheckpointStore:
    def __init__(self, root: Path) -> None:
        expanded = root.expanduser()
        if expanded.is_symlink():
            raise CheckpointError("checkpoint root cannot be a symbolic link")
        candidate = expanded.resolve(strict=False)
        if candidate.exists():
            info = candidate.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise CheckpointError("checkpoint root must be a real directory")
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
                raise CheckpointError("checkpoint root must be private and owned by the current user")
        else:
            if not candidate.parent.is_dir():
                raise CheckpointError("checkpoint root parent must already exist")
            candidate.mkdir(mode=0o700)
            _fsync_directory(candidate.parent)
        self.root = candidate.resolve(strict=True)
        self._lock = threading.Lock()
        self._active_leases: set[str] = set()

    def path_for(self, plan_id: str) -> Path:
        if not plan_id or len(plan_id.encode("utf-8")) > 4096:
            raise CheckpointError("checkpoint plan ID is empty or too large")
        name = hashlib.sha256(plan_id.encode("utf-8")).hexdigest()
        return self.root / f"{name}.json"

    def acquire(self, plan_id: str) -> "CheckpointLease":
        checkpoint_path = self.path_for(plan_id)
        lease_path = checkpoint_path.with_suffix(".lease")
        with self._lock:
            if plan_id in self._active_leases:
                raise CheckpointLeaseError("checkpoint plan is already leased in this process")
            flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(lease_path, flags, 0o600)
            except OSError as error:
                raise CheckpointLeaseError("checkpoint lease cannot be opened safely") from error
            try:
                info = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) & 0o077
                ):
                    raise CheckpointLeaseError(
                        "checkpoint lease must be private and owned by the current user"
                    )
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise CheckpointLeaseError("checkpoint plan is already leased") from error
                metadata_payload = json.dumps(
                    {"plan_id": plan_id, "pid": os.getpid(), "acquired_at": _now()},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                os.ftruncate(descriptor, 0)
                os.lseek(descriptor, 0, os.SEEK_SET)
                written = 0
                while written < len(metadata_payload):
                    written += os.write(descriptor, metadata_payload[written:])
                os.fsync(descriptor)
                self._active_leases.add(plan_id)
            except BaseException:
                os.close(descriptor)
                raise
        return CheckpointLease(self, plan_id, lease_path, descriptor)

    def _release(self, plan_id: str, descriptor: int) -> None:
        with self._lock:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
                self._active_leases.discard(plan_id)

    def save(self, checkpoint: CheckpointManifest) -> Path:
        destination = self.path_for(checkpoint.plan_id)
        encoded = (
            json.dumps(checkpoint.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        if len(encoded) > MAX_CHECKPOINT_BYTES:
            raise CheckpointError("checkpoint exceeds its byte limit")
        with self._lock:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                dir=self.root,
            )
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "wb", buffering=0) as handle:
                    descriptor = -1
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_name, destination)
                _fsync_directory(self.root)
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass
        return destination

    def load(self, plan_id: str) -> CheckpointManifest | None:
        path = self.path_for(plan_id)
        try:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError:
            return None
        except OSError as error:
            raise CheckpointError("checkpoint cannot be opened safely") from error
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise CheckpointError("checkpoint must be a regular file")
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
                raise CheckpointError("checkpoint file must be private and owned by the current user")
            if info.st_size > MAX_CHECKPOINT_BYTES:
                raise CheckpointError("checkpoint exceeds its byte limit")
            with os.fdopen(descriptor, "rb", buffering=0) as handle:
                descriptor = -1
                encoded = handle.read(MAX_CHECKPOINT_BYTES + 1)
            if len(encoded) > MAX_CHECKPOINT_BYTES:
                raise CheckpointError("checkpoint exceeds its byte limit")
            payload = json.loads(encoded.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CheckpointError("checkpoint is not readable JSON") from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if not isinstance(payload, dict):
            raise CheckpointError("checkpoint payload must be an object")
        checkpoint = CheckpointManifest.from_dict(payload)
        if checkpoint.plan_id != plan_id:
            raise CheckpointError("checkpoint plan ID does not match its lookup key")
        return checkpoint


class CheckpointLease:
    def __init__(
        self,
        store: CheckpointStore,
        plan_id: str,
        path: Path,
        descriptor: int,
    ) -> None:
        self.store = store
        self.plan_id = plan_id
        self.path = path
        self._descriptor = descriptor
        self._closed = False
        self._close_lock = threading.Lock()

    def __enter__(self) -> "CheckpointLease":
        if self._closed:
            raise CheckpointLeaseError("checkpoint lease is already closed")
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()

    def close(self) -> None:
        with self._close_lock:
            if not self._closed:
                self._closed = True
                self.store._release(self.plan_id, self._descriptor)


def execution_fingerprint(
    command: Sequence[str],
    environment: Mapping[str, str] | None = None,
) -> str:
    if not command or any(not isinstance(value, str) or not value for value in command):
        raise CheckpointError("execution command must contain non-empty strings")
    if len(command) > 128 or sum(len(value.encode()) for value in command) > 64 * 1024:
        raise CheckpointError("execution command exceeds its limit")
    if environment and (
        len(environment) > 32
        or any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or not key
            or "=" in key
            or "\0" in key
            or "\0" in value
            for key, value in environment.items()
        )
    ):
        raise CheckpointError("execution environment is invalid or unbounded")
    if environment and sum(
        len(key.encode()) + len(value.encode()) for key, value in environment.items()
    ) > 64 * 1024:
        raise CheckpointError("execution environment exceeds its byte limit")
    payload = {
        "command": list(command),
        "environment": dict(sorted((environment or {}).items())),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def decide_resume(
    checkpoint: CheckpointManifest,
    *,
    source_path: Path,
    source_fingerprint: str,
    output_path: Path,
    execution_fingerprint_value: str,
    maximum_output_bytes: int,
) -> ResumeAction:
    source_resolved = source_path.expanduser().resolve(strict=True)
    output_resolved = output_path.expanduser().resolve(strict=False)
    bindings = (
        checkpoint.source_path == str(source_resolved),
        checkpoint.source_fingerprint == source_fingerprint,
        checkpoint.output_path == str(output_resolved),
        checkpoint.execution_fingerprint == execution_fingerprint_value,
        checkpoint.maximum_output_bytes == maximum_output_bytes,
    )
    if not all(bindings):
        raise CheckpointError("checkpoint does not match the requested conversion")
    if checkpoint.stage == CheckpointStage.CONVERTED:
        workspace_exists = (
            checkpoint.workspace_path is not None and Path(checkpoint.workspace_path).exists()
        )
        try:
            output_info = output_resolved.lstat()
            output_is_real_directory = stat.S_ISDIR(output_info.st_mode) and not stat.S_ISLNK(
                output_info.st_mode
            )
        except FileNotFoundError:
            output_is_real_directory = False
        if output_is_real_directory and not workspace_exists:
            return ResumeAction.VERIFY_PROMOTED
        validate_immutable_output_path(source_resolved, output_resolved)
        _validate_resume_workspace(checkpoint)
        return ResumeAction.RESUME_VALIDATION
    if checkpoint.stage == CheckpointStage.COMPLETED:
        output = Path(checkpoint.output_path)
        try:
            info = output.lstat()
        except FileNotFoundError as error:
            raise CheckpointError("completed checkpoint artifact is unavailable") from error
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise CheckpointError("completed checkpoint artifact is unavailable")
        return ResumeAction.ALREADY_COMPLETED
    validate_immutable_output_path(source_resolved, output_resolved)
    return ResumeAction.RESTART_CONVERSION


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_resume_workspace(checkpoint: CheckpointManifest) -> None:
    if checkpoint.workspace_path is None:
        raise CheckpointError("converted checkpoint workspace is unavailable")
    workspace = Path(checkpoint.workspace_path)
    try:
        info = workspace.lstat()
    except FileNotFoundError as error:
        raise CheckpointError("converted checkpoint workspace is unavailable") from error
    output = Path(checkpoint.output_path)
    expected_prefix = f".{output.name}.work-"
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) & 0o077
        or workspace.parent.resolve(strict=True) != output.parent.resolve(strict=True)
        or not workspace.name.startswith(expected_prefix)
        or len(workspace.name) == len(expected_prefix)
    ):
        raise CheckpointError("converted checkpoint workspace is unsafe")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
