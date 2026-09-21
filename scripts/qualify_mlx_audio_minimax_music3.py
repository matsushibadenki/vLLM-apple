#!/usr/bin/env python3
"""Qualify private MiniMax Music3 generation through Homebrew MLX Audio."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm_apple.audio_generation import (  # noqa: E402
    AudioGenerationKind,
    AudioGenerationRequest,
    qualify_audio_generation,
)
from vllm_apple.hardware import detect_hardware  # noqa: E402
from vllm_apple.mlx_audio_generation import MLXMusicGenerationBackend  # noqa: E402
from vllm_apple.qualification import save_qualification_report  # noqa: E402

MINIMUM_AVAILABLE_BYTES = 14_000_000_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--python-path", type=Path, action="append", required=True)
    parser.add_argument("--backend-version", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=4)
    arguments = parser.parse_args()
    if (
        len(arguments.model_revision) != 40
        or any(c not in "0123456789abcdef" for c in arguments.model_revision)
        or not 1 <= arguments.duration <= 30
        or not 1 <= arguments.steps <= 30
    ):
        raise ValueError("invalid MiniMax Music3 qualification configuration")
    hardware = detect_hardware()
    if (
        hardware.memory.available_bytes < MINIMUM_AVAILABLE_BYTES
        or hardware.memory.pressure.value != "normal"
        or hardware.thermal_state.value not in {"nominal", "fair"}
    ):
        raise RuntimeError("MiniMax Music3 memory or thermal admission failed")
    model = arguments.model.expanduser().resolve(strict=True)
    shards = sorted(model.glob("model-*.safetensors"))
    if len(shards) != 2:
        raise ValueError("MiniMax Music3 requires exactly two weight shards")
    private = arguments.private_root.expanduser().absolute()
    private.mkdir(mode=0o700)
    backend = MLXMusicGenerationBackend(
        python_executable=arguments.python,
        model=model,
        workspace_root=arguments.workspace_root,
        backend_version=arguments.backend_version,
        python_path=tuple(arguments.python_path),
        lyrics="[instrumental]",
        steps=arguments.steps,
        timeout_seconds=1800,
    )
    prompts = (
        "Warm acoustic instrumental, 96 BPM, gentle guitar and piano.",
        "Bright electronic instrumental, 112 BPM, soft synth and percussion.",
    )
    samples = []
    try:
        for index, prompt in enumerate(prompts):
            result = qualify_audio_generation(
                backend,
                AudioGenerationRequest(
                    AudioGenerationKind.MUSIC,
                    prompt,
                    index + 7,
                    44_100,
                    2,
                    arguments.duration,
                ),
                private,
            )
            samples.append(result.to_dict())
    finally:
        cleanup = private.is_dir() and not any(private.iterdir())
        if cleanup:
            private.rmdir()
    report = {
        "schema_version": 1,
        "scope": "minimax_music3_mlx_audio_qualification",
        "sample_count": len(samples),
        "samples": samples,
        "backend_version": arguments.backend_version,
        "model_revision": arguments.model_revision,
        "model_license": "minimax-music3-community-license",
        "quantization": {"mode": "affine", "bits": 4, "group_size": 64},
        "weight_sha256": {shard.name: _sha256(shard) for shard in shards},
        "admission": {
            "minimum_available_bytes": MINIMUM_AVAILABLE_BYTES,
            "available_bytes": hardware.memory.available_bytes,
            "memory_pressure": hardware.memory.pressure.value,
            "thermal_state": hardware.thermal_state.value,
        },
        "private_cleanup_verified": cleanup and not private.exists(),
        "stores_prompt": False,
        "stores_output": False,
    }
    report["passed"] = (
        len(samples) == len(prompts)
        and all(sample["passed"] for sample in samples)
        and len({sample["output_sha256"] for sample in samples}) == len(samples)
        and report["private_cleanup_verified"]
    )
    save_qualification_report(report, arguments.report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
