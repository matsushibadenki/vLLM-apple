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
        if self.weights_bytes < 0 or self.kv_bytes_per_token <= 0 or self.workspace_bytes < 0:
            raise ValueError("invalid model memory specification")
        if self.model_max_context is not None and self.model_max_context <= 0:
            raise ValueError("model_max_context must be positive")


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

