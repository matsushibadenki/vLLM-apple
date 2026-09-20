"""Bounded FFmpeg VideoToolbox decoder integration for local video input."""
from __future__ import annotations

import json
import math
import mmap
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


_CODECS = frozenset({"h264", "hevc", "vp9", "av1"})


@dataclass(frozen=True, slots=True)
class VideoStreamInfo:
    codec: str
    width: int
    height: int
    frame_rate: float
    duration_seconds: float | None


@dataclass(frozen=True, slots=True)
class DecodedVideoFrames:
    stream: VideoStreamInfo
    pixel_format: str
    frames: tuple[bytes, ...]
    hardware_accelerator: str


class MappedDecodedVideoFrames:
    """Anonymous-file-backed decoded frames with zero-copy CPU frame views."""

    def __init__(
        self,
        stream: VideoStreamInfo,
        storage: BinaryIO,
        mapping: mmap.mmap,
        frame_count: int,
    ) -> None:
        self.stream = stream
        self.pixel_format = "bgra"
        self.hardware_accelerator = "videotoolbox"
        self.storage = "anonymous_mmap"
        self.gpu_staging_compatible = True
        self._storage = storage
        self._mapping = mapping
        self.frame_count = frame_count
        self.frame_bytes = stream.width * stream.height * 4
        self._closed = False

    def frame_view(self, index: int) -> memoryview:
        if self._closed:
            raise RuntimeError("mapped video frames are closed")
        if type(index) is not int or not 0 <= index < self.frame_count:
            raise IndexError("video frame index is out of range")
        begin = index * self.frame_bytes
        return memoryview(self._mapping)[begin:begin + self.frame_bytes].toreadonly()

    def close(self) -> None:
        if self._closed:
            return
        self._mapping.close()
        self._storage.close()
        self._closed = True

    def __enter__(self) -> "MappedDecodedVideoFrames":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class FFmpegVideoToolboxDecoder:
    ffmpeg: Path = Path("/opt/homebrew/bin/ffmpeg")
    ffprobe: Path = Path("/opt/homebrew/bin/ffprobe")
    maximum_input_bytes: int = 512 * 1024 * 1024
    maximum_pixels_per_frame: int = 3840 * 2160
    maximum_frames: int = 256
    maximum_output_bytes: int = 256 * 1024 * 1024
    timeout_seconds: float = 30

    def __post_init__(self) -> None:
        if (
            not 1 <= self.maximum_input_bytes <= 1 << 40
            or not 1 <= self.maximum_pixels_per_frame <= 16_384 * 16_384
            or not 1 <= self.maximum_frames <= 4096
            or not 1 <= self.maximum_output_bytes <= 1 << 40
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("invalid VideoToolbox decoder limits")

    def inspect(self, source: Path) -> VideoStreamInfo:
        path = self._validated_source(source)
        completed = subprocess.run(
            [
                str(self.ffprobe), "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,width,height,avg_frame_rate,duration",
                "-of", "json", str(path),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=True,
            timeout=self.timeout_seconds,
        )
        if len(completed.stdout) > 64 * 1024:
            raise RuntimeError("video metadata exceeded its output bound")
        payload = json.loads(completed.stdout)
        streams = payload.get("streams") if isinstance(payload, dict) else None
        if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], dict):
            raise ValueError("video must contain a readable primary video stream")
        stream = streams[0]
        codec = stream.get("codec_name")
        width = stream.get("width")
        height = stream.get("height")
        if (
            codec not in _CODECS
            or type(width) is not int
            or type(height) is not int
            or width <= 0
            or height <= 0
            or width * height > self.maximum_pixels_per_frame
        ):
            raise ValueError("video stream is unsupported or exceeds its pixel budget")
        frame_rate = _parse_rate(stream.get("avg_frame_rate"))
        duration = _parse_duration(stream.get("duration"))
        return VideoStreamInfo(codec, width, height, frame_rate, duration)

    def decode(self, source: Path, *, maximum_frames: int | None = None) -> DecodedVideoFrames:
        path = self._validated_source(source)
        stream = self.inspect(path)
        frame_limit = self.maximum_frames if maximum_frames is None else maximum_frames
        if type(frame_limit) is not int or not 1 <= frame_limit <= self.maximum_frames:
            raise ValueError("invalid video decode frame limit")
        frame_bytes = stream.width * stream.height * 4
        if frame_bytes * frame_limit > self.maximum_output_bytes:
            raise ValueError("decoded video exceeds its output byte budget")
        completed = subprocess.run(
            [
                str(self.ffmpeg), "-v", "error", "-hwaccel", "videotoolbox",
                "-i", str(path), "-map", "0:v:0", "-frames:v", str(frame_limit),
                "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgra", "pipe:1",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=True,
            timeout=self.timeout_seconds,
        )
        output = completed.stdout
        if len(output) > self.maximum_output_bytes or len(output) % frame_bytes:
            raise RuntimeError("decoded video output is invalid or exceeded its bound")
        frames = tuple(
            output[offset:offset + frame_bytes]
            for offset in range(0, len(output), frame_bytes)
        )
        if not frames or len(frames) > frame_limit:
            raise RuntimeError("VideoToolbox decoder returned an invalid frame count")
        return DecodedVideoFrames(stream, "bgra", frames, "videotoolbox")

    def decode_mapped(
        self,
        source: Path,
        *,
        maximum_frames: int | None = None,
    ) -> MappedDecodedVideoFrames:
        """Decode directly into anonymous mmap storage without heap frame copies."""
        path = self._validated_source(source)
        stream = self.inspect(path)
        frame_limit = self.maximum_frames if maximum_frames is None else maximum_frames
        if type(frame_limit) is not int or not 1 <= frame_limit <= self.maximum_frames:
            raise ValueError("invalid video decode frame limit")
        frame_bytes = stream.width * stream.height * 4
        if frame_bytes * frame_limit > self.maximum_output_bytes:
            raise ValueError("decoded video exceeds its output byte budget")
        storage = tempfile.TemporaryFile(prefix="vllm-apple-video-")
        try:
            subprocess.run(
                [
                    str(self.ffmpeg), "-v", "error", "-hwaccel", "videotoolbox",
                    "-i", str(path), "-map", "0:v:0", "-frames:v", str(frame_limit),
                    "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgra", "pipe:1",
                ],
                stdin=subprocess.DEVNULL,
                stdout=storage,
                stderr=subprocess.PIPE,
                check=True,
                timeout=self.timeout_seconds,
            )
            size = storage.tell()
            if size > self.maximum_output_bytes or size == 0 or size % frame_bytes:
                raise RuntimeError("mapped video output is invalid or exceeded its bound")
            frame_count = size // frame_bytes
            if frame_count > frame_limit:
                raise RuntimeError("VideoToolbox decoder returned an invalid frame count")
            storage.flush()
            mapping = mmap.mmap(storage.fileno(), size, access=mmap.ACCESS_READ)
            return MappedDecodedVideoFrames(stream, storage, mapping, frame_count)
        except Exception:
            storage.close()
            raise

    def _validated_source(self, source: Path) -> Path:
        if not isinstance(source, Path) or source.is_symlink():
            raise ValueError("video source must be a regular local file")
        resolved = source.resolve(strict=True)
        if not resolved.is_file() or not 1 <= resolved.stat().st_size <= self.maximum_input_bytes:
            raise ValueError("video source size is invalid")
        return resolved


def _parse_rate(value: object) -> float:
    if not isinstance(value, str) or "/" not in value:
        raise ValueError("video frame rate is invalid")
    numerator, denominator = value.split("/", 1)
    try:
        rate = int(numerator) / int(denominator)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("video frame rate is invalid") from error
    if not math.isfinite(rate) or not 0 < rate <= 1000:
        raise ValueError("video frame rate is invalid")
    return rate


def _parse_duration(value: object) -> float | None:
    if value in {None, "N/A"}:
        return None
    try:
        duration = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("video duration is invalid") from error
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("video duration is invalid")
    return duration
