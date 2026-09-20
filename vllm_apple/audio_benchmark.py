"""Bounded dropout, latency, and real-time-factor audio benchmark."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True, slots=True)
class AudioBenchmarkConfig:
    sample_rate: int = 16_000
    channels: int = 1
    chunk_frames: int = 320
    total_frames: int = 160_000
    maximum_dropout_ratio: float = 0.0
    maximum_p95_latency_seconds: float = 0.02
    maximum_realtime_factor: float = 1.0

    def __post_init__(self) -> None:
        if (
            type(self.sample_rate) is not int
            or not 8_000 <= self.sample_rate <= 384_000
            or type(self.channels) is not int
            or not 1 <= self.channels <= 8
            or type(self.chunk_frames) is not int
            or not 1 <= self.chunk_frames <= self.sample_rate * 10
            or type(self.total_frames) is not int
            or not self.chunk_frames <= self.total_frames <= self.sample_rate * 60 * 60
            or any(not math.isfinite(value) or value < 0 for value in (
                self.maximum_dropout_ratio,
                self.maximum_p95_latency_seconds,
                self.maximum_realtime_factor,
            ))
            or self.maximum_dropout_ratio > 1
        ):
            raise ValueError("invalid audio benchmark configuration")


@dataclass(frozen=True, slots=True)
class AudioBenchmarkReport:
    schema_version: int
    chunks: int
    total_frames: int
    processed_frames: int
    dropped_frames: int
    failures: int
    dropout_ratio: float
    p50_latency_seconds: float
    p95_latency_seconds: float
    maximum_latency_seconds: float
    realtime_factor: float
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


def run_audio_benchmark(
    config: AudioBenchmarkConfig,
    process: Callable[[int, tuple[float, ...], int, bool], int],
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> AudioBenchmarkReport:
    """Run deterministic PCM chunks through a caller-owned streaming pipeline."""
    latencies = []
    processed_frames = 0
    failures = 0
    chunks = (config.total_frames + config.chunk_frames - 1) // config.chunk_frames
    for sequence in range(chunks):
        begin_frame = sequence * config.chunk_frames
        frame_count = min(config.chunk_frames, config.total_frames - begin_frame)
        samples = tuple(
            ((begin_frame + frame) % 97 - 48) / 48.0
            for frame in range(frame_count)
            for _ in range(config.channels)
        )
        started = clock()
        try:
            completed = process(sequence, samples, frame_count, sequence == chunks - 1)
            if type(completed) is not int or not 0 <= completed <= frame_count:
                raise ValueError("audio benchmark callback returned an invalid frame count")
            processed_frames += completed
        except Exception:
            failures += 1
        elapsed = clock() - started
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("audio benchmark clock moved backwards")
        latencies.append(elapsed)
    ordered = sorted(latencies)
    p50 = _percentile(ordered, 0.50)
    p95 = _percentile(ordered, 0.95)
    dropped = config.total_frames - processed_frames
    dropout = dropped / config.total_frames
    total_processing = sum(latencies)
    audio_duration = config.total_frames / config.sample_rate
    realtime_factor = total_processing / audio_duration
    passed = (
        failures == 0
        and dropout <= config.maximum_dropout_ratio
        and p95 <= config.maximum_p95_latency_seconds
        and realtime_factor <= config.maximum_realtime_factor
    )
    return AudioBenchmarkReport(
        schema_version=1,
        chunks=chunks,
        total_frames=config.total_frames,
        processed_frames=processed_frames,
        dropped_frames=dropped,
        failures=failures,
        dropout_ratio=round(dropout, 9),
        p50_latency_seconds=round(p50, 9),
        p95_latency_seconds=round(p95, 9),
        maximum_latency_seconds=round(max(latencies), 9),
        realtime_factor=round(realtime_factor, 9),
        passed=passed,
    )


def _percentile(ordered: list[float], fraction: float) -> float:
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]
