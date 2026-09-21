#!/usr/bin/env python3
"""Compile, execute, and bind the MobileCLIP S0 Core ML qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm_apple.hardware import detect_hardware  # noqa: E402
from vllm_apple.qualification import save_qualification_report  # noqa: E402

MAX_OUTPUT_BYTES = 64 * 1024
MODEL_FILES = ("mobileclip_s0_image.mlpackage", "mobileclip_s0_text.mlpackage")


def _artifact_digest(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256(b"vllm-apple-mobileclip-coreml-artifacts-v1\0")
    count = total = 0
    for model_name in MODEL_FILES:
        model = root / model_name
        if not model.is_dir():
            raise ValueError(f"missing MobileCLIP artifact: {model_name}")
        for path in sorted(model.rglob("*")):
            if path.is_symlink():
                raise ValueError("MobileCLIP artifacts must not contain symlinks")
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
    if count < 2:
        raise ValueError("MobileCLIP artifacts are empty")
    return digest.hexdigest(), count, total


def _validated_payload(raw: bytes) -> dict[str, object]:
    payload = json.loads(raw)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("scope") != "mobileclip_s0_coreml_embedding_qualification"
        or payload.get("compute_units") != "cpu_and_neural_engine"
        or payload.get("image_input") != {"name": "image", "shape": [256, 256, 3]}
        or payload.get("text_input")
        != {"name": "text", "shape": [1, 77], "dtype": "int32"}
        or payload.get("output")
        != {"name": "final_emb_1", "shape": [1, 512], "dtype": "float32"}
        or payload.get("sample_count") != 3
        or payload.get("passed") is not True
    ):
        raise RuntimeError("MobileCLIP Core ML report is invalid")
    for key in ("image_samples", "text_samples"):
        samples = payload.get(key)
        if not isinstance(samples, list) or len(samples) != 3:
            raise RuntimeError("MobileCLIP Core ML sample report is invalid")
        digests = set()
        for sample in samples:
            if not isinstance(sample, dict):
                raise RuntimeError("MobileCLIP Core ML sample is invalid")
            digest = sample.get("output_sha256")
            norm = sample.get("l2_norm")
            if (
                sample.get("output_count") != 512
                or not isinstance(digest, str)
                or len(digest) != 64
                or not isinstance(norm, (int, float))
                or not 0.01 < float(norm) < 100.0
            ):
                raise RuntimeError("MobileCLIP Core ML embedding is invalid")
            digests.add(digest)
        if len(digests) != 3:
            raise RuntimeError("MobileCLIP Core ML embeddings are not input-distinct")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--swift", type=Path, default=Path("/usr/bin/swift"))
    parser.add_argument("--xcrun", type=Path, default=Path("/usr/bin/xcrun"))
    parser.add_argument(
        "--swift-script",
        type=Path,
        default=Path("scripts/qualify_mobileclip_coreml.swift"),
    )
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    if (
        len(arguments.model_revision) != 40
        or any(c not in "0123456789abcdef" for c in arguments.model_revision)
    ):
        raise ValueError("model revision must be a lowercase commit SHA")
    model_root = arguments.model_root.expanduser().resolve(strict=True)
    swift = arguments.swift.expanduser().resolve(strict=True)
    xcrun = arguments.xcrun.expanduser().resolve(strict=True)
    script = arguments.swift_script.expanduser().resolve(strict=True)
    if not model_root.is_dir() or not swift.is_file() or not xcrun.is_file() or not script.is_file():
        raise ValueError("MobileCLIP Core ML qualification input is invalid")
    artifact_sha256, artifact_files, artifact_bytes = _artifact_digest(model_root)
    private_root = arguments.private_root.expanduser().resolve()
    private_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    private_root.chmod(0o700)
    compile_root = private_root / "compiled"
    if compile_root.exists():
        shutil.rmtree(compile_root)
    compile_root.mkdir(mode=0o700)
    try:
        for model_name in MODEL_FILES:
            result = subprocess.run(
                (str(xcrun), "coremlc", "compile", str(model_root / model_name), str(compile_root)),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300,
                check=False,
            )
            if result.returncode != 0 or len(result.stderr) > MAX_OUTPUT_BYTES:
                raise RuntimeError(f"MobileCLIP Core ML compilation failed: {model_name}")
        result = subprocess.run(
            (
                str(swift),
                str(script),
                str(compile_root / "mobileclip_s0_image.mlmodelc"),
                str(compile_root / "mobileclip_s0_text.mlmodelc"),
            ),
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
            raise RuntimeError("MobileCLIP Core ML qualification failed")
        payload = _validated_payload(result.stdout)
    finally:
        shutil.rmtree(compile_root, ignore_errors=True)
    hardware = detect_hardware()
    payload.update({
        "model": "apple/coreml-mobileclip",
        "profile": "mobileclip_s0",
        "model_revision": arguments.model_revision,
        "model_license": "apple-ascl",
        "artifact_sha256": artifact_sha256,
        "artifact_file_count": artifact_files,
        "artifact_bytes": artifact_bytes,
        "memory_pressure": hardware.memory.pressure.value,
        "soc": hardware.soc,
        "os_version": hardware.os_version,
        "temporary_cleanup_verified": not compile_root.exists(),
    })
    payload["passed"] = (
        payload["passed"] is True
        and payload["memory_pressure"] == "normal"
        and payload["thermal_state"] in {"nominal", "fair"}
        and payload["temporary_cleanup_verified"] is True
    )
    save_qualification_report(payload, arguments.report)
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
