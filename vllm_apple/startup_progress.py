from __future__ import annotations

from dataclasses import asdict, dataclass

STARTUP_PROGRESS_SCHEMA_VERSION = 1
STARTUP_STAGES = frozenset(
    {"initializing", "profiling", "loading_model", "ready", "degraded", "failed", "stopping", "stopped"}
)


@dataclass(frozen=True, slots=True)
class StartupProgress:
    schema_version: int
    stage: str
    completed_units: int
    total_units: int
    message_key: str

    def __post_init__(self) -> None:
        if self.schema_version != STARTUP_PROGRESS_SCHEMA_VERSION or self.stage not in STARTUP_STAGES:
            raise ValueError("invalid startup progress stage")
        if not 1 <= self.total_units <= 1000 or not 0 <= self.completed_units <= self.total_units:
            raise ValueError("invalid startup progress units")
        if not self.message_key or len(self.message_key) > 128:
            raise ValueError("invalid startup progress message key")

    @property
    def percent(self) -> int:
        return self.completed_units * 100 // self.total_units

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "percent": self.percent}
