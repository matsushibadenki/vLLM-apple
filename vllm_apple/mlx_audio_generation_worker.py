"""One-shot private MLX Audio generation worker."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import resource
import stat
from pathlib import Path

MAX_REQUEST_BYTES = 64 * 1024


def _consume_request(path: Path) -> dict[str, object]:
    candidate = path.expanduser().absolute()
    if candidate.is_symlink():
        raise ValueError("MLX Audio request must not be a symlink")
    descriptor = os.open(candidate, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) != 0o600
            or not 1 <= before.st_size <= MAX_REQUEST_BYTES
        ):
            raise ValueError("MLX Audio request is unsafe")
        encoded = os.read(descriptor, MAX_REQUEST_BYTES + 1)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("MLX Audio request changed while being read")
    finally:
        os.close(descriptor)
    candidate.unlink()
    payload = json.loads(encoded)
    if not isinstance(payload, dict):
        raise ValueError("MLX Audio request is invalid")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    arguments = parser.parse_args()
    payload = _consume_request(arguments.request)
    kind = payload.get("kind")
    speech_fields = {
        "kind", "model", "output", "prompt", "voice", "language", "max_tokens"
    }
    music_fields = {
        "kind", "model", "output", "prompt", "lyrics",
        "duration_seconds", "steps", "seed",
    }
    if kind not in {"speech", "music"} or set(payload) != (
        speech_fields if kind == "speech" else music_fields
    ):
        raise ValueError("MLX Audio request schema is invalid")
    model = Path(payload["model"]).resolve(strict=True)
    output = Path(payload["output"]).absolute()
    if (
        not model.is_dir()
        or output.exists()
        or output.is_symlink()
        or not output.parent.is_dir()
        or not isinstance(payload["prompt"], str)
        or not payload["prompt"].strip()
    ):
        raise ValueError("MLX Audio request values are invalid")

    import importlib.metadata

    with open(os.devnull, "w", encoding="utf-8") as sink, contextlib.redirect_stdout(sink):
        if kind == "speech":
            if (
                not isinstance(payload["voice"], str)
                or not isinstance(payload["language"], str)
                or type(payload["max_tokens"]) is not int
                or not 1 <= payload["max_tokens"] <= 4096
            ):
                raise ValueError("MLX speech request values are invalid")
            from mlx_audio.tts.generate import generate_audio

            generate_audio(
                text=payload["prompt"],
                model=str(model),
                max_tokens=payload["max_tokens"],
                voice=payload["voice"],
                lang_code=payload["language"],
                output_path=str(output.parent),
                file_prefix=output.stem,
                audio_format="wav",
                join_audio=True,
                play=False,
                verbose=False,
            )
        else:
            if (
                not isinstance(payload["lyrics"], str)
                or not payload["lyrics"].strip()
                or type(payload["duration_seconds"]) not in {int, float}
                or not 0 < payload["duration_seconds"] <= 300
                or type(payload["steps"]) is not int
                or not 1 <= payload["steps"] <= 100
                or type(payload["seed"]) is not int
                or not 0 <= payload["seed"] <= 2**32 - 1
            ):
                raise ValueError("MLX music request values are invalid")
            from mlx_audio.music.generate import generate_music

            generate_music(
                caption=payload["prompt"],
                lyrics=payload["lyrics"],
                model=str(model),
                duration=float(payload["duration_seconds"]),
                steps=payload["steps"],
                seed=payload["seed"],
                output_path=output,
                verbose=False,
            )
    if not output.is_file() or output.is_symlink():
        raise RuntimeError("MLX Audio did not produce its bound WAV")
    output.chmod(0o600)
    from .hardware import detect_hardware

    hardware = detect_hardware()
    print(json.dumps({
        "schema_version": 1,
        "backend": f"mlx-audio-{'kokoro' if kind == 'speech' else 'minimax-music3'}",
        "backend_version": importlib.metadata.version("mlx-audio"),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "memory_pressure": hardware.memory.pressure.value,
        "thermal_state": hardware.thermal_state.value,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
