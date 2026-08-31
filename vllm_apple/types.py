from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


GIB = 1024**3
MIB = 1024**2


class RuntimeState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    PROFILING = "profiling"
    LOADING_MODEL = "loading_model"
    READY = "ready"
    DEGRADED = "degraded"
    FAILED = "failed"
    STOPPING = "stopping"


class MemoryPressure(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class Priority(str, Enum):
    REALTIME = "realtime"
    INTERACTIVE = "interactive"
    NORMAL = "normal"
    BACKGROUND = "background"


class Backend(str, Enum):
    CPU = "cpu"
    MLX_GPU = "mlx_gpu"
    METAL = "metal"


@dataclass(frozen=True, slots=True)
class MemoryInfo:
    total_bytes: int
    available_bytes: int
    process_resident_bytes: int = 0
    pressure: MemoryPressure = MemoryPressure.UNKNOWN
    source: str = "unknown"

    def __post_init__(self) -> None:
        if self.total_bytes <= 0:
            raise ValueError("total_bytes must be positive")
        if not 0 <= self.available_bytes <= self.total_bytes:
            raise ValueError("available_bytes must be between zero and total_bytes")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["pressure"] = self.pressure.value
        return result


@dataclass(frozen=True, slots=True)
class HardwareInfo:
    platform: str
    architecture: str
    soc: str
    physical_cpu_count: int
    logical_cpu_count: int
    gpu_core_count: int | None
    memory: MemoryInfo
    is_apple_silicon: bool
    os_version: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["memory"] = self.memory.to_dict()
        return result


@dataclass(frozen=True, slots=True)
class ModelMemorySpec:
    model_id: str
    weights_bytes: int
    kv_bytes_per_token: int
    workspace_bytes: int = 0
    model_max_context: int | None = None

    def __post_init__(self) -> None:
        if self.weights_bytes < 0 or self.kv_bytes_per_token < 0 or self.workspace_bytes < 0:
            raise ValueError("invalid model memory specification")
        if self.model_max_context is not None and self.model_max_context <= 0:
            raise ValueError("model_max_context must be positive")
        if self.kv_bytes_per_token == 0 and self.model_max_context is None:
            raise ValueError("bounded-state models require model_max_context")

    def as_state_memory_spec(self) -> StateMemorySpec:
        """Return the generalized state model without changing the legacy API."""
        return StateMemorySpec(
            model_id=self.model_id,
            architecture="transformer",
            weights_bytes=self.weights_bytes,
            kv_bytes_per_token=self.kv_bytes_per_token,
            scratch_workspace_bytes=self.workspace_bytes,
            model_max_context=self.model_max_context,
        )


@dataclass(frozen=True, slots=True)
class StateMemorySpec:
    """Memory shape for Transformer, recurrent, windowed, and hybrid models."""

    model_id: str
    architecture: str
    weights_bytes: int
    kv_bytes_per_token: int = 0
    recurrent_state_bytes: int = 0
    prefix_state_bytes: int = 0
    attention_window_bytes_per_token: int = 0
    attention_window_tokens: int | None = None
    sparse_index_bytes_per_token: int = 0
    sparse_retrieval_bytes_per_token: int = 0
    sparse_retrieval_tokens: int | None = None
    expert_storage_bytes: int = 0
    expert_working_set_bytes: int = 0
    ngram_storage_bytes: int = 0
    ngram_working_set_bytes: int = 0
    mtp_storage_bytes: int = 0
    mtp_working_set_bytes: int = 0
    scratch_workspace_bytes: int = 0
    model_max_context: int | None = None

    def __post_init__(self) -> None:
        if not self.model_id or not self.architecture:
            raise ValueError("model_id and architecture must not be empty")
        byte_fields = (
            self.weights_bytes,
            self.kv_bytes_per_token,
            self.recurrent_state_bytes,
            self.prefix_state_bytes,
            self.attention_window_bytes_per_token,
            self.sparse_index_bytes_per_token,
            self.sparse_retrieval_bytes_per_token,
            self.expert_storage_bytes,
            self.expert_working_set_bytes,
            self.ngram_storage_bytes,
            self.ngram_working_set_bytes,
            self.mtp_storage_bytes,
            self.mtp_working_set_bytes,
            self.scratch_workspace_bytes,
        )
        if any(value < 0 for value in byte_fields):
            raise ValueError("state memory byte counts must not be negative")
        if self.kv_bytes_per_token == 0 and self.model_max_context is None:
            raise ValueError("bounded-state models require model_max_context")
        if self.attention_window_tokens is not None and self.attention_window_tokens <= 0:
            raise ValueError("attention_window_tokens must be positive")
        if self.sparse_retrieval_tokens is not None and self.sparse_retrieval_tokens <= 0:
            raise ValueError("sparse_retrieval_tokens must be positive")
        if self.model_max_context is not None and self.model_max_context <= 0:
            raise ValueError("model_max_context must be positive")

    @property
    def fixed_bytes(self) -> int:
        return (
            self.weights_bytes
            + self.recurrent_state_bytes
            + self.prefix_state_bytes
            + self.expert_working_set_bytes
            + self.ngram_working_set_bytes
            + self.mtp_working_set_bytes
            + self.scratch_workspace_bytes
        )

    def state_bytes(self, tokens: int) -> int:
        if tokens < 0:
            raise ValueError("tokens must not be negative")
        window_tokens = tokens
        if self.attention_window_tokens is not None:
            window_tokens = min(tokens, self.attention_window_tokens)
        retrieval_tokens = tokens
        if self.sparse_retrieval_tokens is not None:
            retrieval_tokens = min(tokens, self.sparse_retrieval_tokens)
        return (
            self.recurrent_state_bytes
            + self.prefix_state_bytes
            + tokens * self.kv_bytes_per_token
            + window_tokens * self.attention_window_bytes_per_token
            + tokens * self.sparse_index_bytes_per_token
            + retrieval_tokens * self.sparse_retrieval_bytes_per_token
        )

    def total_bytes(self, tokens: int) -> int:
        return (
            self.weights_bytes
            + self.expert_working_set_bytes
            + self.ngram_working_set_bytes
            + self.mtp_working_set_bytes
            + self.scratch_workspace_bytes
            + self.state_bytes(tokens)
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ContextTier:
    name: str
    max_tokens: int
    kv_budget_bytes: int


@dataclass(frozen=True, slots=True)
class ContextRecommendation:
    model_id: str
    allocatable_bytes: int
    os_reserve_bytes: int
    safety_headroom_bytes: int
    workspace_bytes: int
    tiers: tuple[ContextTier, ...]
    limiting_factor: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "allocatable_bytes": self.allocatable_bytes,
            "os_reserve_bytes": self.os_reserve_bytes,
            "safety_headroom_bytes": self.safety_headroom_bytes,
            "workspace_bytes": self.workspace_bytes,
            "tiers": [asdict(tier) for tier in self.tiers],
            "limiting_factor": self.limiting_factor,
        }


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    profile_version: int
    runtime_version: str
    created_at: str
    hardware: HardwareInfo
    context: ContextRecommendation | None = None
    capabilities: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_version": self.profile_version,
            "runtime_version": self.runtime_version,
            "created_at": self.created_at,
            "hardware": self.hardware.to_dict(),
            "context": self.context.to_dict() if self.context else None,
            "capabilities": list(self.capabilities),
            "metadata": dict(self.metadata),
        }
