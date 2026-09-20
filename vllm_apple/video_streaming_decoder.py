"""Incremental FFmpeg VideoToolbox stream decoder feeding the frame scheduler."""
from __future__ import annotations

import math
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

from .video_frame_scheduler import ScheduledVideoFrame, VideoFrameDecision, VideoFrameScheduler


_INPUT_FORMATS = frozenset({"mpegts", "h264", "hevc"})


@dataclass(frozen=True, slots=True)
class StreamingVideoDecoderReport:
    input_chunks: int
    input_bytes: int
    decoded_frames: int
    scheduler_rejections: int
    stderr_truncated: bool
    exit_code: int
    passed: bool


class IncrementalFFmpegVideoDecoder:
    """One ordered compressed stream decoded concurrently into scheduled BGRA frames."""

    def __init__(
        self,
        scheduler: VideoFrameScheduler[bytes],
        *,
        width: int,
        height: int,
        frame_rate: float,
        input_format: str = "mpegts",
        ffmpeg: Path = Path("/opt/homebrew/bin/ffmpeg"),
        maximum_input_bytes: int = 512 * 1024 * 1024,
        maximum_frames: int = 4096,
        maximum_stderr_bytes: int = 64 * 1024,
        shutdown_timeout_seconds: float = 10,
        process_factory=subprocess.Popen,
    ) -> None:
        if (
            not isinstance(scheduler, VideoFrameScheduler)
            or type(width) is not int
            or type(height) is not int
            or width <= 0
            or height <= 0
            or width * height > 3840 * 2160
            or not math.isfinite(frame_rate)
            or not 0 < frame_rate <= 1000
            or input_format not in _INPUT_FORMATS
            or not 1 <= maximum_input_bytes <= 1 << 40
            or not 1 <= maximum_frames <= 65_536
            or not 1024 <= maximum_stderr_bytes <= 1024 * 1024
            or not math.isfinite(shutdown_timeout_seconds)
            or shutdown_timeout_seconds <= 0
        ):
            raise ValueError("invalid streaming video decoder configuration")
        self._scheduler = scheduler
        self._width = width
        self._height = height
        self._frame_rate = frame_rate
        self._frame_bytes = width * height * 4
        self._maximum_input_bytes = maximum_input_bytes
        self._maximum_frames = maximum_frames
        self._maximum_stderr_bytes = maximum_stderr_bytes
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._process = process_factory(
            [
                str(ffmpeg), "-v", "error", "-hwaccel", "videotoolbox",
                "-f", input_format, "-i", "pipe:0", "-map", "0:v:0",
                "-frames:v", str(maximum_frames), "-fps_mode", "passthrough",
                "-f", "rawvideo", "-pix_fmt", "bgra", "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if self._process.stdin is None or self._process.stdout is None or self._process.stderr is None:
            self._process.kill()
            raise RuntimeError("streaming video decoder pipes are unavailable")
        self._next_sequence = 0
        self._input_chunks = 0
        self._input_bytes = 0
        self._decoded_frames = 0
        self._scheduler_rejections = 0
        self._stderr = bytearray()
        self._stderr_truncated = False
        self._reader_error: Exception | None = None
        self._finalized = False
        self._lock = threading.RLock()
        self._stdout_thread = threading.Thread(target=self._read_frames, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    def append(self, sequence: int, chunk: bytes, *, final: bool = False) -> None:
        with self._lock:
            if self._finalized:
                raise RuntimeError("streaming video decoder is finalized")
            if type(sequence) is not int or sequence != self._next_sequence:
                raise ValueError(f"expected video decode chunk sequence {self._next_sequence}")
            if (
                not isinstance(chunk, bytes)
                or not chunk
                or self._input_bytes + len(chunk) > self._maximum_input_bytes
            ):
                raise ValueError("streaming video decoder input exceeds its budget")
            self._process.stdin.write(chunk)
            self._process.stdin.flush()
            self._next_sequence += 1
            self._input_chunks += 1
            self._input_bytes += len(chunk)
            if final:
                self._process.stdin.close()
                self._finalized = True

    def finish(self) -> StreamingVideoDecoderReport:
        with self._lock:
            if not self._finalized:
                self._process.stdin.close()
                self._finalized = True
        try:
            exit_code = self._process.wait(timeout=self._shutdown_timeout_seconds)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
            exit_code = -1
        self._stdout_thread.join(timeout=self._shutdown_timeout_seconds)
        self._stderr_thread.join(timeout=self._shutdown_timeout_seconds)
        passed = (
            exit_code == 0
            and self._reader_error is None
            and self._decoded_frames > 0
            and self._scheduler_rejections == 0
            and not self._stdout_thread.is_alive()
            and not self._stderr_thread.is_alive()
        )
        return StreamingVideoDecoderReport(
            self._input_chunks,
            self._input_bytes,
            self._decoded_frames,
            self._scheduler_rejections,
            self._stderr_truncated,
            exit_code,
            passed,
        )

    def drain_ready(self, media_time_seconds: float) -> tuple[VideoFrameDecision[bytes], ...]:
        return self._scheduler.drain_late(media_time_seconds)

    def _read_frames(self) -> None:
        try:
            while self._decoded_frames < self._maximum_frames:
                frame = _read_exact(self._process.stdout, self._frame_bytes)
                if not frame:
                    break
                if len(frame) != self._frame_bytes:
                    raise RuntimeError("streaming decoder returned a partial video frame")
                index = self._decoded_frames
                accepted = self._scheduler.submit(ScheduledVideoFrame(
                    f"frame-{index}",
                    index / self._frame_rate,
                    1 / self._frame_rate,
                    frame,
                    keyframe=index == 0,
                ))
                self._decoded_frames += 1
                if not accepted:
                    self._scheduler_rejections += 1
        except Exception as error:
            self._reader_error = error

    def _read_stderr(self) -> None:
        while True:
            chunk = self._process.stderr.read(4096)
            if not chunk:
                return
            remaining = self._maximum_stderr_bytes - len(self._stderr)
            if remaining > 0:
                self._stderr.extend(chunk[:remaining])
            if len(chunk) > remaining:
                self._stderr_truncated = True


def _read_exact(stream: object, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            break
        chunks.extend(chunk)
    return bytes(chunks)
