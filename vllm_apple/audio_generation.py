"""Private, bounded qualification contract for speech and music generation."""
from __future__ import annotations

import hashlib
import math
import os
import stat
import time
import wave
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

MAX_AUDIO_PROMPT_BYTES = 32 * 1024
MAX_GENERATED_AUDIO_BYTES = 1 * 1024 * 1024 * 1024
MAX_GENERATED_AUDIO_SECONDS = 1800


class AudioGenerationKind(str, Enum):
    SPEECH = "speech"
    MUSIC = "music"


@dataclass(frozen=True, slots=True)
class AudioGenerationRequest:
    kind: AudioGenerationKind
    prompt: str
    seed: int
    sample_rate: int
    channels: int
    duration_seconds: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.kind, AudioGenerationKind)
            or not isinstance(self.prompt, str)
            or not self.prompt.strip()
            or len(self.prompt.encode()) > MAX_AUDIO_PROMPT_BYTES
            or type(self.seed) is not int
            or not 0 <= self.seed < 2**63
            or type(self.sample_rate) is not int
            or not 8_000 <= self.sample_rate <= 192_000
            or type(self.channels) is not int
            or not 1 <= self.channels <= 8
            or not isinstance(self.duration_seconds, (int, float))
            or isinstance(self.duration_seconds, bool)
            or not math.isfinite(self.duration_seconds)
            or not 0 < self.duration_seconds <= MAX_GENERATED_AUDIO_SECONDS
        ):
            raise ValueError("invalid audio generation request")

    @property
    def prompt_sha256(self) -> str:
        return hashlib.sha256(self.prompt.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class AudioWorkerTelemetry:
    peak_rss_bytes: int
    memory_pressure: str
    thermal_state: str
    backend: str
    backend_version: str

    def __post_init__(self) -> None:
        if (
            type(self.peak_rss_bytes) is not int
            or self.peak_rss_bytes <= 0
            or self.memory_pressure not in {"normal", "warning", "critical"}
            or self.thermal_state not in {"nominal", "fair", "serious", "critical"}
            or not self.backend
            or not self.backend_version
            or len(self.backend) > 128
            or len(self.backend_version) > 128
        ):
            raise ValueError("invalid audio worker telemetry")


class AudioGenerationBackend(Protocol):
    def generate(
        self, request: AudioGenerationRequest, output_path: Path
    ) -> AudioWorkerTelemetry: ...


@dataclass(frozen=True, slots=True)
class AudioGenerationQualificationReport:
    kind: AudioGenerationKind
    prompt_sha256: str
    seed: int
    sample_rate: int
    channels: int
    frames: int
    duration_seconds: float
    output_bytes: int
    output_sha256: str
    wall_time_seconds: float
    telemetry: AudioWorkerTelemetry
    private_cleanup_verified: bool
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "scope": "audio_generation_qualification",
            "kind": self.kind.value,
            "prompt_sha256": self.prompt_sha256,
            "seed": self.seed,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "frames": self.frames,
            "duration_seconds": self.duration_seconds,
            "output_bytes": self.output_bytes,
            "output_sha256": self.output_sha256,
            "wall_time_seconds": self.wall_time_seconds,
            "peak_rss_bytes": self.telemetry.peak_rss_bytes,
            "memory_pressure": self.telemetry.memory_pressure,
            "thermal_state": self.telemetry.thermal_state,
            "backend": self.telemetry.backend,
            "backend_version": self.telemetry.backend_version,
            "private_cleanup_verified": self.private_cleanup_verified,
            "stores_prompt": False,
            "stores_output": False,
            "passed": self.passed,
        }


def qualify_audio_generation(
    backend: AudioGenerationBackend,
    request: AudioGenerationRequest,
    private_root: str | Path,
    *,
    maximum_output_bytes: int = MAX_GENERATED_AUDIO_BYTES,
) -> AudioGenerationQualificationReport:
    root_candidate = Path(private_root).expanduser().absolute()
    if root_candidate.is_symlink():
        raise ValueError("audio generation private root is unsafe")
    root = root_candidate.resolve(strict=True)
    root_info = root.lstat()
    if (
        not stat.S_ISDIR(root_info.st_mode)
        or root_info.st_uid != os.getuid()
        or stat.S_IMODE(root_info.st_mode) & 0o077
        or not 1 <= maximum_output_bytes <= MAX_GENERATED_AUDIO_BYTES
    ):
        raise ValueError("audio generation private root is unsafe")
    output = root / "generated.wav"
    if output.exists() or output.is_symlink():
        raise ValueError("audio generation output path already exists")
    started = time.monotonic()
    telemetry = None
    metadata = None
    try:
        telemetry = backend.generate(request, output)
        if not isinstance(telemetry, AudioWorkerTelemetry):
            raise TypeError("audio generation backend returned invalid telemetry")
        metadata = _inspect_private_wav(output, maximum_output_bytes)
        if (
            metadata["sample_rate"] != request.sample_rate
            or metadata["channels"] != request.channels
            or metadata["duration_seconds"] > request.duration_seconds + 1 / request.sample_rate
        ):
            raise ValueError("generated audio does not match its request")
    finally:
        try:
            info = output.lstat()
        except FileNotFoundError:
            info = None
        if info is not None and stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid():
            output.unlink()
    cleanup = not output.exists() and not output.is_symlink()
    assert telemetry is not None and metadata is not None
    wall = time.monotonic() - started
    return AudioGenerationQualificationReport(
        request.kind,
        request.prompt_sha256,
        request.seed,
        metadata["sample_rate"],
        metadata["channels"],
        metadata["frames"],
        metadata["duration_seconds"],
        metadata["output_bytes"],
        metadata["output_sha256"],
        wall,
        telemetry,
        cleanup,
        cleanup and telemetry.memory_pressure == "normal"
        and telemetry.thermal_state in {"nominal", "fair"},
    )


def _inspect_private_wav(path: Path, maximum_output_bytes: int) -> dict[str, object]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) & 0o077
            or not 44 <= before.st_size <= maximum_output_bytes
        ):
            raise ValueError("generated WAV file is unsafe")
        digest = hashlib.sha256()
        with os.fdopen(os.dup(descriptor), "rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(os.dup(descriptor), "rb") as source, wave.open(source, "rb") as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            frames = wav.getnframes()
            if wav.getcomptype() != "NONE" or wav.getsampwidth() != 2:
                raise ValueError("generated WAV must contain PCM S16LE")
            if not 1 <= channels <= 8 or not 8_000 <= sample_rate <= 192_000 or frames <= 0:
                raise ValueError("generated WAV metadata is invalid")
            duration = frames / sample_rate
            if duration > MAX_GENERATED_AUDIO_SECONDS:
                raise ValueError("generated WAV duration exceeds limit")
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("generated WAV changed during inspection")
        return {
            "sample_rate": sample_rate,
            "channels": channels,
            "frames": frames,
            "duration_seconds": duration,
            "output_bytes": before.st_size,
            "output_sha256": digest.hexdigest(),
        }
    finally:
        os.close(descriptor)
