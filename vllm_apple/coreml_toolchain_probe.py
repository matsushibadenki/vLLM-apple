"""Isolated Core ML Tools and Xcode compiler qualification probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path


_PREDICTION_PROGRAM = r'''
import CoreML
import Foundation

let configuration = MLModelConfiguration()
configuration.computeUnits = .cpuAndNeuralEngine
let model = try MLModel(
    contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]),
    configuration: configuration
)
let input = try MLMultiArray(shape: [1, 4], dataType: .float16)
for (index, value) in [1.0, 2.0, 3.0, 4.0].enumerated() {
    input[index] = NSNumber(value: value)
}
let provider = try MLDictionaryFeatureProvider(dictionary: [
    "pixel_values": MLFeatureValue(multiArray: input)
])
let started = DispatchTime.now().uptimeNanoseconds
let prediction = try model.prediction(from: provider)
let elapsed = DispatchTime.now().uptimeNanoseconds - started
guard let output = prediction.featureValue(for: "hidden_states")?.multiArrayValue else {
    throw NSError(domain: "vllm-apple", code: 1)
}
let payload: [String: Any] = [
    "latency_nanoseconds": elapsed,
    "output_values": (0..<output.count).map { output[$0].doubleValue }
]
let encoded = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
print(String(data: encoded, encoding: .utf8)!)
'''.strip()


def run_coreml_toolchain_probe(output_directory: Path) -> dict[str, object]:
    """Build and compile a deterministic FP16 MIL program in a new directory."""
    destination = output_directory.expanduser().resolve(strict=False)
    if destination.exists() or not destination.parent.is_dir():
        raise ValueError("Core ML toolchain probe destination must be new")
    try:
        import coremltools as ct
        import numpy as np
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types
    except ImportError as error:
        raise RuntimeError("Core ML Tools probe requires coremltools and NumPy") from error

    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    os.chmod(temporary, 0o700)
    package = temporary / "probe.mlpackage"
    compiled_root = temporary / "compiled"
    compiled_root.mkdir(mode=0o700)
    try:
        weights = np.asarray(
            [[1.0, 0.0, 0.0, 1.0], [0.5, -0.5, 0.25, -0.25]], dtype=np.float16
        )
        bias = np.asarray([0.125, -0.125], dtype=np.float16)

        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, 4), dtype=types.fp16)],
            opset_version=ct.target.macOS15,
        )
        def program(pixel_values):
            return mb.linear(
                x=pixel_values,
                weight=weights,
                bias=bias,
                name="hidden_states",
            )

        model = ct.convert(
            program,
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS15,
        )
        model.short_description = "vLLM-Apple deterministic Core ML toolchain probe"
        model.user_defined_metadata["vllm-apple.probe"] = "coreml-toolchain-v1"
        model.save(str(package))
        command = [
            "/usr/bin/xcrun",
            "coremlcompiler",
            "compile",
            str(package),
            str(compiled_root),
            "--platform",
            "macOS",
            "--deployment-target",
            "15.0",
        ]
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
            check=False,
            env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip()[-2048:]
            raise RuntimeError(f"Core ML compiler rejected probe package: {detail}")
        compiled = tuple(compiled_root.glob("*.mlmodelc"))
        if len(compiled) != 1 or not compiled[0].is_dir():
            raise RuntimeError("Core ML compiler did not produce one mlmodelc directory")
        compiler_version = subprocess.run(
            ["/usr/bin/xcrun", "coremlcompiler", "version"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
            env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
        )
        version = compiler_version.stdout.strip()
        if compiler_version.returncode != 0 or not version or len(version) > 128:
            raise RuntimeError("Core ML compiler version is unavailable")
        prediction = subprocess.run(
            ["/usr/bin/swift", "-e", _PREDICTION_PROGRAM, str(compiled[0])],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
            check=False,
            env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
        )
        if prediction.returncode != 0 or len(prediction.stdout) > 4096:
            detail = prediction.stderr.strip()[-2048:]
            raise RuntimeError(f"Core ML probe prediction failed: {detail}")
        try:
            prediction_payload = json.loads(prediction.stdout)
            values = prediction_payload["output_values"]
            latency = prediction_payload["latency_nanoseconds"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise RuntimeError("Core ML probe prediction output is invalid") from error
        expected = (5.125, -0.875)
        if (
            not isinstance(values, list)
            or len(values) != 2
            or type(latency) is not int
            or latency <= 0
            or any(
                type(value) not in (int, float) or abs(value - reference) > 0.01
                for value, reference in zip(values, expected)
            )
        ):
            raise RuntimeError("Core ML probe prediction is numerically incorrect")
        package_digest, package_files = _tree_digest(package)
        compiled_digest, compiled_files = _tree_digest(compiled[0])
        report = {
            "schema_version": 1,
            "passed": True,
            "coremltools_version": ct.__version__,
            "coremlcompiler_version": version,
            "platform": platform.system(),
            "platform_release": platform.release(),
            "architecture": platform.machine(),
            "minimum_deployment_target": "macOS15",
            "compute_precision": "fp16",
            "input_shape": [1, 4],
            "output_shape": [1, 2],
            "prediction_values": values,
            "prediction_latency_nanoseconds": latency,
            "compute_units": "cpu_and_neural_engine",
            "package_sha256": package_digest,
            "package_file_count": package_files,
            "compiled_sha256": compiled_digest,
            "compiled_file_count": compiled_files,
        }
        (temporary / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
        return report
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _tree_digest(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Core ML toolchain probe output contains a symlink")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        files += 1
    if not files:
        raise ValueError("Core ML toolchain probe output is empty")
    return digest.hexdigest(), files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    print(json.dumps(run_coreml_toolchain_probe(arguments.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
