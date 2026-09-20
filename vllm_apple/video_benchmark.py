"""Validated throughput and normalized-memory benchmark for video decode paths."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True, slots=True)
class VideoDecodeMeasurement:
    decoded_frames: int
    video_duration_seconds: float
    elapsed_seconds: float
    retained_buffer_bytes: int
    peak_rss_bytes: int | None = None

    def __post_init__(self) -> None:
        if (
            type(self.decoded_frames) is not int
            or self.decoded_frames <= 0
            or not math.isfinite(self.video_duration_seconds)
            or self.video_duration_seconds <= 0
            or not math.isfinite(self.elapsed_seconds)
            or self.elapsed_seconds <= 0
            or type(self.retained_buffer_bytes) is not int
            or self.retained_buffer_bytes < 0
            or self.peak_rss_bytes is not None
            and (type(self.peak_rss_bytes) is not int or self.peak_rss_bytes <= 0)
        ):
            raise ValueError("invalid video decode measurement")


@dataclass(frozen=True, slots=True)
class VideoBenchmarkThresholds:
    minimum_frames_per_second: float = 1.0
    minimum_video_seconds_per_second: float = 1.0
    maximum_buffer_bytes_per_video_minute: int = 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.minimum_frames_per_second)
            or self.minimum_frames_per_second <= 0
            or not math.isfinite(self.minimum_video_seconds_per_second)
            or self.minimum_video_seconds_per_second <= 0
            or type(self.maximum_buffer_bytes_per_video_minute) is not int
            or self.maximum_buffer_bytes_per_video_minute <= 0
        ):
            raise ValueError("invalid video benchmark thresholds")


@dataclass(frozen=True, slots=True)
class VideoBenchmarkReport:
    schema_version: int
    decoded_frames: int
    video_duration_seconds: float
    elapsed_seconds: float
    frames_per_second: float
    video_seconds_per_second: float
    retained_buffer_bytes: int
    buffer_bytes_per_video_minute: int
    peak_rss_bytes: int | None
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def run_video_benchmark(
    measure: Callable[[], VideoDecodeMeasurement],
    thresholds: VideoBenchmarkThresholds = VideoBenchmarkThresholds(),
) -> VideoBenchmarkReport:
    measurement = measure()
    if not isinstance(measurement, VideoDecodeMeasurement):
        raise TypeError("video benchmark adapter returned an invalid measurement")
    frames_per_second = measurement.decoded_frames / measurement.elapsed_seconds
    video_seconds_per_second = measurement.video_duration_seconds / measurement.elapsed_seconds
    memory_per_minute = math.ceil(
        measurement.retained_buffer_bytes * 60 / measurement.video_duration_seconds
    )
    passed = (
        frames_per_second >= thresholds.minimum_frames_per_second
        and video_seconds_per_second >= thresholds.minimum_video_seconds_per_second
        and memory_per_minute <= thresholds.maximum_buffer_bytes_per_video_minute
    )
    return VideoBenchmarkReport(
        schema_version=1,
        decoded_frames=measurement.decoded_frames,
        video_duration_seconds=measurement.video_duration_seconds,
        elapsed_seconds=measurement.elapsed_seconds,
        frames_per_second=round(frames_per_second, 6),
        video_seconds_per_second=round(video_seconds_per_second, 6),
        retained_buffer_bytes=measurement.retained_buffer_bytes,
        buffer_bytes_per_video_minute=memory_per_minute,
        peak_rss_bytes=measurement.peak_rss_bytes,
        passed=passed,
    )
