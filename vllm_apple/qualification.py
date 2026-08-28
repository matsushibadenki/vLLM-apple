from __future__ import annotations

import time
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .backend import BackendConfig, BackendProcess
from .promotion_probe import PromotionProbeConfig, run_serving_promotion_probe
from .soak import SoakConfig, run_soak


@dataclass(frozen=True, slots=True)
class QualificationConfig:
    model: str
    executable: Path
    port: int = 8001
    max_model_len: int | None = None
    startup_timeout_seconds: float = 600
    duration_seconds: float = 1800
    warmup_seconds: float = 30
    concurrency: int = 4
    request_timeout_seconds: float = 300
    max_rss_growth_bytes: int = 256 * 1024 * 1024
    require_30_minute_window: bool = True

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model cannot be empty")
        for value in (
            self.startup_timeout_seconds,
            self.duration_seconds,
            self.request_timeout_seconds,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("qualification durations must be positive and finite")
        if not math.isfinite(self.warmup_seconds) or self.warmup_seconds < 0:
            raise ValueError("warmup must be non-negative and finite")
        if not 1 <= self.concurrency <= 256:
            raise ValueError("concurrency must be between 1 and 256")
        if self.max_rss_growth_bytes < 0:
            raise ValueError("RSS growth limit cannot be negative")
        if self.require_30_minute_window and self.duration_seconds < 1800:
            raise ValueError("real-model certification requires at least 1800 seconds")


def qualify_model(
    config: QualificationConfig,
    *,
    process_factory: Callable[[BackendConfig], BackendProcess] = BackendProcess,
) -> dict[str, object]:
    backend = process_factory(
        BackendConfig(
            model=config.model,
            executable=config.executable,
            port=config.port,
            max_model_len=config.max_model_len,
            startup_timeout=config.startup_timeout_seconds,
        )
    )
    load_started = time.monotonic()
    load_seconds = 0.0
    soak: dict[str, object] | None = None
    promotion: dict[str, object] | None = None
    shutdown_clean = False
    try:
        backend.start()
        load_seconds = time.monotonic() - load_started
        pid = backend.pid
        if pid is None:
            raise RuntimeError("backend started without a process ID")
        promotion = run_serving_promotion_probe(
            PromotionProbeConfig(
                base_url=backend.base_url,
                model=config.model,
                timeout_seconds=config.request_timeout_seconds,
            )
        )
        if not promotion["passed"]:
            raise RuntimeError("backend sampling/streaming promotion probe failed")
        soak = run_soak(
            SoakConfig(
                base_url=backend.base_url,
                duration_seconds=config.duration_seconds,
                warmup_seconds=config.warmup_seconds,
                concurrency=config.concurrency,
                request_timeout_seconds=config.request_timeout_seconds,
                mode="chat-mixed",
                model=config.model,
                target_pid=pid,
                max_rss_growth_bytes=config.max_rss_growth_bytes,
                require_30_minute_window=config.require_30_minute_window,
            )
        )
    finally:
        backend.stop()
        shutdown_clean = not backend.running
    return {
        "schema_version": 1,
        "model": config.model,
        "backend": "vllm_metal",
        "load_seconds": round(load_seconds, 3),
        "shutdown_clean": shutdown_clean,
        "promotion_probe": promotion,
        "soak": soak,
        "passed": bool(
            promotion and promotion["passed"] and soak and soak["passed"] and shutdown_clean
        ),
    }
