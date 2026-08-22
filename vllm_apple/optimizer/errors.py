from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .types import OPTIMIZER_SCHEMA_VERSION


class OptimizerErrorCode(str, Enum):
    INVALID_MODEL = "invalid_model"
    INVALID_PLAN = "invalid_plan"
    UNSAFE_PATH = "unsafe_path"
    INSUFFICIENT_MEMORY = "insufficient_memory"
    INSUFFICIENT_DISK = "insufficient_disk"
    PROFILER_FAILED = "profiler_failed"
    ADAPTER_UNAVAILABLE = "adapter_unavailable"
    WORKER_CRASHED = "worker_crashed"
    WORKER_TIMEOUT = "worker_timeout"
    ARTIFACT_VALIDATION_FAILED = "artifact_validation_failed"
    CANCELLED = "cancelled"


class Recoverability(str, Enum):
    RETRYABLE = "retryable"
    USER_ACTION_REQUIRED = "user_action_required"
    NOT_RETRYABLE = "not_retryable"


@dataclass(frozen=True, slots=True)
class OptimizerFailure(Exception):
    code: OptimizerErrorCode
    message_key: str
    recoverability: Recoverability
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": OPTIMIZER_SCHEMA_VERSION,
            "code": self.code.value,
            "message_key": self.message_key,
            "recoverability": self.recoverability.value,
            "detail": self.detail[:2048],
        }
