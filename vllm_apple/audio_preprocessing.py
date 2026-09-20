"""Bounded streaming audio resampling and lightweight feature extraction."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


class StreamingLinearResampler:
    """Chunk-stable interleaved linear resampler with bounded input calls."""

    def __init__(self, input_rate: int, output_rate: int, *, channels: int = 1) -> None:
        if (
            type(input_rate) is not int
            or type(output_rate) is not int
            or not 1 <= input_rate <= 384_000
            or not 1 <= output_rate <= 384_000
            or type(channels) is not int
            or not 1 <= channels <= 32
        ):
            raise ValueError("invalid audio resampler configuration")
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.channels = channels
        self._step = input_rate / output_rate
        self._frames: list[tuple[float, ...]] = []
        self._position = 0.0
        self._total_input_frames = 0
        self._total_output_frames = 0

    def process(
        self,
        samples: Sequence[float],
        frame_count: int,
        *,
        final: bool = False,
    ) -> tuple[float, ...]:
        if (
            type(frame_count) is not int
            or not 0 <= frame_count <= 1_000_000
            or len(samples) < frame_count * self.channels
        ):
            raise ValueError("invalid resampler input")
        for frame in range(frame_count):
            values = tuple(
                float(samples[frame * self.channels + channel])
                for channel in range(self.channels)
            )
            if any(not math.isfinite(value) for value in values):
                raise ValueError("audio samples must be finite")
            self._frames.append(values)
        self._total_input_frames += frame_count

        output: list[float] = []
        while self._position + 1 < len(self._frames):
            self._append_interpolated(output)
        discard = min(int(self._position), max(0, len(self._frames) - 1))
        if discard:
            del self._frames[:discard]
            self._position -= discard

        if final:
            target_frames = round(
                self._total_input_frames * self.output_rate / self.input_rate
            )
            while self._total_output_frames < target_frames and self._frames:
                self._append_interpolated(output, clamp=True)
            self.reset()
        return tuple(output)

    def _append_interpolated(self, output: list[float], *, clamp: bool = False) -> None:
        lower = min(int(self._position), len(self._frames) - 1)
        upper = min(lower + 1, len(self._frames) - 1)
        fraction = self._position - int(self._position)
        for channel in range(self.channels):
            first = self._frames[lower][channel]
            second = self._frames[upper][channel]
            output.append(first + (second - first) * fraction)
        self._position += self._step
        self._total_output_frames += 1
        if clamp and self._position >= len(self._frames):
            self._position = float(len(self._frames) - 1)

    def reset(self) -> None:
        self._frames.clear()
        self._position = 0.0
        self._total_input_frames = 0
        self._total_output_frames = 0


@dataclass(frozen=True, slots=True)
class AudioFeatureFrame:
    start_sample: int
    values: tuple[float, ...]


class StreamingLogBandEncoder:
    """Online log-energy features with bounded retained state and no full-audio capture."""

    def __init__(self, *, frame_length: int = 400, hop_length: int = 160, bands: int = 20) -> None:
        if (
            type(frame_length) is not int
            or not 2 <= frame_length <= 8192
            or type(hop_length) is not int
            or not 1 <= hop_length <= frame_length
            or type(bands) is not int
            or not 1 <= bands <= min(256, frame_length)
        ):
            raise ValueError("invalid audio feature configuration")
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.bands = bands
        self._samples: list[float] = []
        self._start_sample = 0

    def process(self, samples: Sequence[float]) -> tuple[AudioFeatureFrame, ...]:
        if len(samples) > 4_000_000:
            raise ValueError("audio feature chunk exceeds its bound")
        converted = [float(value) for value in samples]
        if any(not math.isfinite(value) for value in converted):
            raise ValueError("audio samples must be finite")
        self._samples.extend(converted)
        frames = []
        while len(self._samples) >= self.frame_length:
            window = self._samples[:self.frame_length]
            values = []
            for band in range(self.bands):
                begin = band * self.frame_length // self.bands
                end = (band + 1) * self.frame_length // self.bands
                energy = sum(value * value for value in window[begin:end]) / (end - begin)
                values.append(math.log(max(energy, 1e-12)))
            frames.append(AudioFeatureFrame(self._start_sample, tuple(values)))
            del self._samples[:self.hop_length]
            self._start_sample += self.hop_length
        return tuple(frames)

    @property
    def retained_samples(self) -> int:
        return len(self._samples)

    def reset(self) -> None:
        self._samples.clear()
        self._start_sample = 0
