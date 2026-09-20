"""Validated subprocess adapter for the native CVPixelBuffer-to-Metal bridge."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VideoMetalBridgeReport:
    schema_version: int
    hardware_decode_supported: bool
    decoded_frames: int
    texture_bindings: int
    binding_failures: int
    width: int
    height: int
    pixel_format: str
    always_copies_sample_data: bool
    passed: bool


@dataclass(frozen=True, slots=True)
class NativeVideoMetalBridge:
    executable: Path
    timeout_seconds: float = 30
    maximum_output_bytes: int = 16 * 1024

    def run(self, source: Path, *, maximum_frames: int = 256) -> VideoMetalBridgeReport:
        if (
            not self.executable.is_file()
            or not isinstance(source, Path)
            or source.is_symlink()
            or not source.is_file()
            or type(maximum_frames) is not int
            or not 1 <= maximum_frames <= 4096
        ):
            raise ValueError("invalid native video Metal bridge request")
        completed = subprocess.run(
            [str(self.executable), str(source.resolve()), str(maximum_frames)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=True,
            timeout=self.timeout_seconds,
        )
        if len(completed.stdout) > self.maximum_output_bytes:
            raise RuntimeError("native video Metal bridge output exceeded its bound")
        payload = json.loads(completed.stdout)
        expected = set(VideoMetalBridgeReport.__dataclass_fields__)
        if not isinstance(payload, dict) or set(payload) != expected:
            raise RuntimeError("native video Metal bridge returned an invalid report")
        report = VideoMetalBridgeReport(**payload)
        if (
            report.schema_version != 1
            or report.decoded_frames < 1
            or report.texture_bindings < 0
            or report.binding_failures < 0
            or report.texture_bindings + report.binding_failures != report.decoded_frames
            or report.width < 1
            or report.height < 1
            or report.pixel_format != "bgra8Unorm"
            or report.passed != (
                report.hardware_decode_supported
                and not report.always_copies_sample_data
                and report.binding_failures == 0
                and report.texture_bindings == report.decoded_frames
            )
        ):
            raise RuntimeError("native video Metal bridge report is inconsistent")
        return report
