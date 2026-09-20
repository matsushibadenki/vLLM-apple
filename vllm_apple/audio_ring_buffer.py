"""Preallocated single-producer/single-consumer audio ring buffer."""
from __future__ import annotations

import math
from array import array
from dataclasses import dataclass
from typing import MutableSequence, Sequence


@dataclass(frozen=True, slots=True)
class AudioRingBufferSnapshot:
    capacity_frames: int
    channels: int
    available_frames: int
    free_frames: int
    written_frames: int
    read_frames: int
    overflow_frames: int
    underrun_frames: int


class AudioRingBuffer:
    """Bounded SPSC float32 buffer; producer and consumer methods never block."""

    def __init__(self, capacity_frames: int, *, channels: int = 1) -> None:
        if (
            type(capacity_frames) is not int
            or not 2 <= capacity_frames <= 10_000_000
            or type(channels) is not int
            or not 1 <= channels <= 32
        ):
            raise ValueError("invalid audio ring buffer shape")
        self._capacity_frames = capacity_frames
        self._channels = channels
        self._samples = array("f", [0.0]) * (capacity_frames * channels)
        self._read_position = 0
        self._write_position = 0
        self._available_frames = 0
        self._written_frames = 0
        self._read_frames = 0
        self._overflow_frames = 0
        self._underrun_frames = 0

    @property
    def capacity_frames(self) -> int:
        return self._capacity_frames

    @property
    def channels(self) -> int:
        return self._channels

    def write_from(self, samples: Sequence[float], frame_count: int) -> int:
        """Copy complete interleaved frames; reject excess rather than overwrite unread data."""
        if (
            type(frame_count) is not int
            or frame_count < 0
            or len(samples) < frame_count * self._channels
        ):
            raise ValueError("invalid audio producer input")
        accepted = min(frame_count, self._capacity_frames - self._available_frames)
        for frame in range(accepted):
            target_frame = (self._write_position + frame) % self._capacity_frames
            source_offset = frame * self._channels
            target_offset = target_frame * self._channels
            for channel in range(self._channels):
                value = float(samples[source_offset + channel])
                if not math.isfinite(value):
                    raise ValueError("audio samples must be finite")
                self._samples[target_offset + channel] = value
        self._write_position = (self._write_position + accepted) % self._capacity_frames
        self._available_frames += accepted
        self._written_frames += accepted
        self._overflow_frames += frame_count - accepted
        return accepted

    def read_into(self, output: MutableSequence[float], frame_count: int) -> int:
        """Copy available complete frames into caller-owned storage without blocking."""
        if (
            type(frame_count) is not int
            or frame_count < 0
            or len(output) < frame_count * self._channels
        ):
            raise ValueError("invalid audio consumer output")
        consumed = min(frame_count, self._available_frames)
        for frame in range(consumed):
            source_frame = (self._read_position + frame) % self._capacity_frames
            source_offset = source_frame * self._channels
            target_offset = frame * self._channels
            for channel in range(self._channels):
                output[target_offset + channel] = self._samples[source_offset + channel]
        self._read_position = (self._read_position + consumed) % self._capacity_frames
        self._available_frames -= consumed
        self._read_frames += consumed
        self._underrun_frames += frame_count - consumed
        return consumed

    def clear(self) -> None:
        """Discard unread frames; call only while producer and consumer are stopped."""
        self._read_position = 0
        self._write_position = 0
        self._available_frames = 0

    @property
    def snapshot(self) -> AudioRingBufferSnapshot:
        available = self._available_frames
        return AudioRingBufferSnapshot(
            capacity_frames=self._capacity_frames,
            channels=self._channels,
            available_frames=available,
            free_frames=self._capacity_frames - available,
            written_frames=self._written_frames,
            read_frames=self._read_frames,
            overflow_frames=self._overflow_frames,
            underrun_frames=self._underrun_frames,
        )
