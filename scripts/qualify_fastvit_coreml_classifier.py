#!/usr/bin/env python3
"""Compile, execute, and bind the FastViT-T8 Core ML classifier qualification."""
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


def _artifact_digest(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256(b"vllm-apple-fastvit-coreml-artifact-v1\0")
    count = total = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("FastViT artifact must not contain symlinks")
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
        raise ValueError("FastViT artifact is empty")
    return digest.hexdigest(), count, total


def _validated_payload(raw: bytes) -> dict[str, object]:
    payload = json.loads(raw)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("scope") != "fastvit_t8_coreml_classifier_qualification"
        or payload.get("compute_units") != "cpu_and_neural_engine"
        or payload.get("input") != {"name": "image", "shape": [256, 256, 3]}
        or payload.get("outputs")
        != {
            "label": "classLabel",
            "probabilities": "classLabel_probs",
            "probability_array": "classLabelProbs",
            "class_count": 1000,
            "unique_label_count": 999,
        }
        or payload.get("sample_count") != 3
        or payload.get("passed") is not True
    ):
        raise RuntimeError("FastViT Core ML report is invalid")
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != 3:
        raise RuntimeError("FastViT sample report is invalid")
    digests = set()
    for sample in samples:
        if (
            not isinstance(sample, dict)
            or sample.get("class_count") != 1000
            or sample.get("unique_label_count") != 999
            or not isinstance(sample.get("top_label"), str)
            or not isinstance(sample.get("top_probability"), (int, float))
            or not 0.999 <= float(sample.get("probability_sum", 0)) <= 1.001
            or not isinstance(sample.get("probability_sha256"), str)
            or len(str(sample["probability_sha256"])) != 64
        ):
            raise RuntimeError("FastViT classification output is invalid")
        digests.add(sample["probability_sha256"])
    if len(digests) != 3:
        raise RuntimeError("FastViT distributions are not input-distinct")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--swift", type=Path, default=Path("/usr/bin/swift"))
    parser.add_argument("--xcrun", type=Path, default=Path("/usr/bin/xcrun"))
    parser.add_argument(
        "--swift-script", type=Path,
        default=Path("scripts/qualify_fastvit_coreml_classifier.swift"),
    )
    arguments = parser.parse_args()
    if len(arguments.model_revision) != 40 or any(
        c not in "0123456789abcdef" for c in arguments.model_revision
    ):
        raise ValueError("model revision must be a lowercase commit SHA")
    model = arguments.model.expanduser().resolve(strict=True)
    swift = arguments.swift.expanduser().resolve(strict=True)
    xcrun = arguments.xcrun.expanduser().resolve(strict=True)
    script = arguments.swift_script.expanduser().resolve(strict=True)
    artifact_sha256, artifact_files, artifact_bytes = _artifact_digest(model)
    private_root = arguments.private_root.expanduser().resolve()
    private_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    private_root.chmod(0o700)
    compiled = private_root / "compiled"
    shutil.rmtree(compiled, ignore_errors=True)
    compiled.mkdir(mode=0o700)
    try:
        result = subprocess.run(
            (str(xcrun), "coremlc", "compile", str(model), str(compiled)),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=300, check=False,
        )
        if result.returncode != 0 or len(result.stderr) > MAX_OUTPUT_BYTES:
            raise RuntimeError("FastViT Core ML compilation failed")
        result = subprocess.run(
            (str(swift), str(script), str(compiled / "FastViTT8F16.mlmodelc")),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=120, check=False,
        )
        if result.returncode != 0 or not 1 <= len(result.stdout) <= MAX_OUTPUT_BYTES:
            raise RuntimeError("FastViT Core ML qualification failed")
        payload = _validated_payload(result.stdout)
    finally:
        shutil.rmtree(compiled, ignore_errors=True)
    hardware = detect_hardware()
    payload.update({
        "model": "apple/coreml-FastViT-T8",
        "model_revision": arguments.model_revision,
        "model_license": "apple-ascl",
        "artifact_sha256": artifact_sha256,
        "artifact_file_count": artifact_files,
        "artifact_bytes": artifact_bytes,
        "memory_pressure": hardware.memory.pressure.value,
        "soc": hardware.soc,
        "os_version": hardware.os_version,
        "temporary_cleanup_verified": not compiled.exists(),
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
