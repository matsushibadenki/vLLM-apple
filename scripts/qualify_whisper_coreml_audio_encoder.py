#!/usr/bin/env python3
"""Run and bind the Whisper tiny Core ML audio-encoder qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm_apple.hardware import detect_hardware  # noqa: E402
from vllm_apple.qualification import save_qualification_report  # noqa: E402

MAX_OUTPUT_BYTES = 64 * 1024


def _artifact_digest(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256(b"vllm-apple-whisper-coreml-artifact-v1\0")
    count = total = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Core ML encoder artifact must not contain symlinks")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        count += 1
        total += size
    if count < 1:
        raise ValueError("Core ML encoder artifact is empty")
    return digest.hexdigest(), count, total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--swift", type=Path, default=Path("/usr/bin/swift"))
    parser.add_argument(
        "--swift-script",
        type=Path,
        default=Path("scripts/qualify_whisper_coreml_audio_encoder.swift"),
    )
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    if (
        len(arguments.model_revision) != 40
        or any(c not in "0123456789abcdef" for c in arguments.model_revision)
    ):
        raise ValueError("model revision must be a lowercase commit SHA")
    model = arguments.model.expanduser().resolve(strict=True)
    swift = arguments.swift.expanduser().resolve(strict=True)
    script = arguments.swift_script.expanduser().resolve(strict=True)
    if not model.is_dir() or not swift.is_file() or not script.is_file():
        raise ValueError("Whisper Core ML qualification input is invalid")
    artifact_sha256, artifact_files, artifact_bytes = _artifact_digest(model)
    result = subprocess.run(
        (str(swift), str(script), str(model)),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    if (
        result.returncode != 0
        or not 1 <= len(result.stdout) <= MAX_OUTPUT_BYTES
        or len(result.stderr) > MAX_OUTPUT_BYTES
    ):
        raise RuntimeError("Whisper Core ML encoder qualification failed")
    payload = json.loads(result.stdout)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("scope") != "whisper_coreml_audio_encoder_qualification"
        or payload.get("compute_units") != "cpu_and_neural_engine"
        or payload.get("input_shape") != [1, 80, 1, 3000]
        or payload.get("output_shape") != [1, 384, 1, 1500]
        or payload.get("sample_count") != 3
        or payload.get("passed") is not True
    ):
        raise RuntimeError("Whisper Core ML encoder report is invalid")
    hardware = detect_hardware()
    payload.update({
        "model": "openai/whisper-tiny",
        "model_revision": arguments.model_revision,
        "model_license": "mit",
        "artifact_sha256": artifact_sha256,
        "artifact_file_count": artifact_files,
        "artifact_bytes": artifact_bytes,
        "memory_pressure": hardware.memory.pressure.value,
        "soc": hardware.soc,
        "os_version": hardware.os_version,
        "stores_audio": False,
        "stores_embedding": False,
    })
    payload["passed"] = (
        payload["passed"]
        and payload["memory_pressure"] == "normal"
        and payload["thermal_state"] in {"nominal", "fair"}
    )
    save_qualification_report(payload, arguments.report)
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
