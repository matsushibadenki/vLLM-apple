#!/usr/bin/env python3
"""Qualify private Kokoro speech generation through Homebrew MLX Audio."""
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
from vllm_apple.mlx_audio_generation import MLXAudioGenerationBackend  # noqa: E402
from vllm_apple.qualification import save_qualification_report  # noqa: E402


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
    arguments = parser.parse_args()
    if (
        len(arguments.model_revision) != 40
        or any(character not in "0123456789abcdef" for character in arguments.model_revision)
    ):
        raise ValueError("Kokoro model revision must be a lowercase commit SHA")
    private = arguments.private_root.expanduser().absolute()
    private.mkdir(mode=0o700)
    backend = MLXAudioGenerationBackend(
        python_executable=arguments.python,
        model=arguments.model,
        workspace_root=arguments.workspace_root,
        backend_version=arguments.backend_version,
        python_path=tuple(arguments.python_path),
    )
    prompts = (
        "Hello from vLLM Apple. This is the first private speech qualification sample.",
        "Apple Silicon performs the second bounded speech generation sample locally.",
    )
    samples = []
    try:
        for index, prompt in enumerate(prompts):
            result = qualify_audio_generation(
                backend,
                AudioGenerationRequest(
                    AudioGenerationKind.SPEECH, prompt, index, 24_000, 1, 30.0
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
        "scope": "kokoro_mlx_audio_speech_qualification",
        "sample_count": len(samples),
        "samples": samples,
        "backend_version": arguments.backend_version,
        "model_revision": arguments.model_revision,
        "model_license": "apache-2.0",
        "weight_sha256": _sha256(arguments.model / "kokoro-v1_0.safetensors"),
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
