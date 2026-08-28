from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from .backend import BackendStartupError
from .hardware import default_application_support
from .scheduler import ExecutionPlanAdmissionError, MemoryCapacityError

RUNTIME_FAILURE_SCHEMA_VERSION = 1
MAX_DIAGNOSTIC_LOG_LINES = 80
MAX_DIAGNOSTIC_LINE_BYTES = 4096
MAX_DIAGNOSTIC_BYTES = 64 * 1024


class RuntimeFailureCode(str, Enum):
    BACKEND_STARTUP_FAILED = "backend_startup_failed"
    BACKEND_EXITED = "backend_exited"
    BACKEND_READINESS_TIMEOUT = "backend_readiness_timeout"
    BACKEND_INCOMPATIBLE = "backend_incompatible"
    MEMORY_CAPACITY_EXCEEDED = "memory_capacity_exceeded"
    EXECUTION_PLAN_REJECTED = "execution_plan_rejected"
    INTERNAL_ERROR = "internal_error"


class RuntimeRecoverability(str, Enum):
    RETRYABLE = "retryable"
    USER_ACTION_REQUIRED = "user_action_required"
    FATAL = "fatal"


@dataclass(frozen=True, slots=True)
class RuntimeFailure:
    schema_version: int
    code: RuntimeFailureCode
    message_key: str
    recoverability: RuntimeRecoverability
    detail_fingerprint: str

    def __post_init__(self) -> None:
        if self.schema_version != RUNTIME_FAILURE_SCHEMA_VERSION:
            raise ValueError("unsupported runtime failure schema")
        if not self.message_key.startswith("runtime.error.") or len(self.message_key) > 128:
            raise ValueError("invalid runtime failure message key")
        if len(self.detail_fingerprint) != 24 or any(
            value not in "0123456789abcdef" for value in self.detail_fingerprint
        ):
            raise ValueError("invalid runtime failure detail fingerprint")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "code": self.code.value,
            "message_key": self.message_key,
            "recoverability": self.recoverability.value,
            "detail_fingerprint": self.detail_fingerprint,
        }


def classify_runtime_failure(error: BaseException | str) -> RuntimeFailure:
    detail = str(error)
    if isinstance(error, BackendStartupError):
        backend_code = getattr(error, "code", "backend_startup_failed")
        mapping = {
            "backend_exited": (
                RuntimeFailureCode.BACKEND_EXITED,
                RuntimeRecoverability.USER_ACTION_REQUIRED,
            ),
            "backend_readiness_timeout": (
                RuntimeFailureCode.BACKEND_READINESS_TIMEOUT,
                RuntimeRecoverability.RETRYABLE,
            ),
        }
        code, recoverability = mapping.get(
            backend_code,
            (RuntimeFailureCode.BACKEND_STARTUP_FAILED, RuntimeRecoverability.RETRYABLE),
        )
    elif isinstance(error, MemoryCapacityError):
        code = RuntimeFailureCode.MEMORY_CAPACITY_EXCEEDED
        recoverability = RuntimeRecoverability.USER_ACTION_REQUIRED
    elif isinstance(error, ExecutionPlanAdmissionError):
        code = RuntimeFailureCode.EXECUTION_PLAN_REJECTED
        recoverability = RuntimeRecoverability.USER_ACTION_REQUIRED
    elif "incompatible vLLM-Metal environment" in detail:
        code = RuntimeFailureCode.BACKEND_INCOMPATIBLE
        recoverability = RuntimeRecoverability.USER_ACTION_REQUIRED
    else:
        code = RuntimeFailureCode.INTERNAL_ERROR
        recoverability = RuntimeRecoverability.FATAL
    fingerprint = hashlib.sha256(detail.encode("utf-8", errors="replace")).hexdigest()[:24]
    return RuntimeFailure(
        RUNTIME_FAILURE_SCHEMA_VERSION,
        code,
        f"runtime.error.{code.value}",
        recoverability,
        fingerprint,
    )


def persist_crash_diagnostic(
    failure: RuntimeFailure,
    log_lines: tuple[str, ...] = (),
    *,
    root: Path | None = None,
) -> Path:
    bounded = tuple(
        line.encode("utf-8", errors="replace")[:MAX_DIAGNOSTIC_LINE_BYTES]
        for line in log_lines[-MAX_DIAGNOSTIC_LOG_LINES:]
    )
    log_digest = hashlib.sha256(b"\0".join(bounded)).hexdigest()
    created_at = datetime.now(timezone.utc).isoformat()
    diagnostic_id = hashlib.sha256(
        f"{created_at}\0{failure.detail_fingerprint}\0{log_digest}".encode("utf-8")
    ).hexdigest()[:24]
    payload = {
        "schema_version": 1,
        "diagnostic_id": diagnostic_id,
        "created_at": created_at,
        "failure": failure.to_dict(),
        "recent_log_line_count": len(bounded),
        "recent_log_digest": log_digest,
    }
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if len(encoded) > MAX_DIAGNOSTIC_BYTES:
        raise ValueError("runtime crash diagnostic exceeded 64 KiB")
    directory = root or default_application_support() / "diagnostics"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    attributes = directory.lstat()
    if not stat.S_ISDIR(attributes.st_mode) or attributes.st_uid != os.getuid():
        raise ValueError("runtime diagnostic directory is unsafe")
    os.chmod(directory, 0o700)
    destination = directory / f"{diagnostic_id}.json"
    descriptor, temporary_name = tempfile.mkstemp(prefix=".diagnostic-", dir=directory)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination
