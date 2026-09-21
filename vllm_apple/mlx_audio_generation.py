"""Version-fixed private subprocess adapter for MLX Audio generation."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import uuid
from pathlib import Path
from typing import Callable

from .audio_generation import (
    AudioGenerationRequest,
    AudioWorkerTelemetry,
)
from .process_inference_engine import _validated_python_executable

MAX_WORKER_OUTPUT_BYTES = 64 * 1024


class MLXAudioGenerationBackend:
    def __init__(
        self,
        *,
        python_executable: Path,
        model: Path,
        workspace_root: Path,
        backend_version: str,
        voice: str = "af_heart",
        language: str = "en",
        timeout_seconds: float = 300,
        python_path: tuple[Path, ...] = (),
    ) -> None:
        self.python = _validated_python_executable(python_executable)
        self.model = model.expanduser().resolve(strict=True)
        self.workspace = workspace_root.expanduser().resolve(strict=True)
        if (
            not self.model.is_dir()
            or not self.model.is_relative_to(self.workspace)
            or not backend_version
            or not voice
            or not language
            or not 1 <= len(python_path) <= 8
            or not 0 < timeout_seconds <= 1800
        ):
            raise ValueError("invalid MLX Audio backend configuration")
        paths = tuple(path.expanduser().resolve(strict=True) for path in python_path)
        if any(not path.is_dir() for path in paths):
            raise ValueError("MLX Audio Python path is invalid")
        self.backend_version = backend_version
        self.voice = voice
        self.language = language
        self.timeout_seconds = timeout_seconds
        self.python_path = paths

    def generate(
        self, request: AudioGenerationRequest, output_path: Path
    ) -> AudioWorkerTelemetry:
        def speech_payload(
            current: AudioGenerationRequest, destination: Path
        ) -> dict[str, object]:
            return {
                "kind": "speech",
                "model": str(self.model),
                "output": str(destination.absolute()),
                "prompt": current.prompt,
                "voice": self.voice,
                "language": self.language,
                "max_tokens": min(
                    4096, max(64, int(current.duration_seconds * 64))
                ),
            }

        if request.kind.value != "speech":
            raise ValueError("MLX speech backend requires a speech request")
        return self._generate_with_payload(request, output_path, speech_payload)

    def _generate_with_payload(
        self,
        request: AudioGenerationRequest,
        output_path: Path,
        payload_factory: Callable[
            [AudioGenerationRequest, Path], dict[str, object]
        ],
    ) -> AudioWorkerTelemetry:
        output_path = output_path.expanduser().resolve(strict=False)
        if (
            output_path.parent.is_symlink()
            or not output_path.parent.resolve(strict=True).is_relative_to(self.workspace)
            or output_path.exists()
            or output_path.is_symlink()
        ):
            raise ValueError("MLX Audio output path is unsafe")
        request_path = output_path.parent / f"request-{uuid.uuid4().hex}.json"
        descriptor = os.open(request_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            encoded = json.dumps(
                payload_factory(request, output_path), separators=(",", ":")
            ).encode()
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        environment = {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "PYTHONPATH": os.pathsep.join(str(path) for path in self.python_path),
        }
        process = subprocess.Popen(
            [str(self.python), "-m", "vllm_apple.mlx_audio_generation_worker",
             "--request", str(request_path)],
            cwd=self.workspace,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2)
            raise TimeoutError("MLX Audio generation timed out")
        finally:
            request_path.unlink(missing_ok=True)
        if (
            process.returncode != 0
            or not 1 <= len(stdout) <= MAX_WORKER_OUTPUT_BYTES
            or len(stderr) > MAX_WORKER_OUTPUT_BYTES
        ):
            raise RuntimeError("MLX Audio worker failed")
        payload = json.loads(stdout)
        if (
            payload.get("schema_version") != 1
            or payload.get("backend_version") != self.backend_version
        ):
            raise RuntimeError("MLX Audio worker identity changed")
        return AudioWorkerTelemetry(
            payload["peak_rss_bytes"], payload["memory_pressure"],
            payload["thermal_state"], payload["backend"], payload["backend_version"],
        )


class MLXMusicGenerationBackend(MLXAudioGenerationBackend):
    """Private subprocess adapter for lyric-conditioned MLX music generation."""

    def __init__(
        self,
        *,
        lyrics: str = "[instrumental]",
        steps: int = 8,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if not lyrics.strip() or not 1 <= steps <= 100:
            raise ValueError("invalid MLX music backend configuration")
        self.lyrics = lyrics
        self.steps = steps

    def _request_payload(
        self, request: AudioGenerationRequest, output_path: Path
    ) -> dict[str, object]:
        return {
            "kind": "music",
            "model": str(self.model),
            "output": str(output_path.absolute()),
            "prompt": request.prompt,
            "lyrics": self.lyrics,
            "duration_seconds": request.duration_seconds,
            "steps": self.steps,
            "seed": request.seed,
        }

    def generate(
        self, request: AudioGenerationRequest, output_path: Path
    ) -> AudioWorkerTelemetry:
        if request.kind.value != "music":
            raise ValueError("MLX music backend requires a music request")
        return self._generate_with_payload(request, output_path, self._request_payload)
