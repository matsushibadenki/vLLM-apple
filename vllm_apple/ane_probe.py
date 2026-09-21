"""Public-Core-ML-only ANE surface probing without claiming execution support."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .execution import ExecutionBackend
from .kernel_probe import KernelMeasurement, KernelProbeConfig, KernelProbeResult, run_kernel_probe
from .model_integrity import verify_model_integrity

ANE_SURFACE_SCHEMA_VERSION = 1
_SURFACE_PROGRAM = r'''
import CoreML
import Foundation
let configuration = MLModelConfiguration()
configuration.computeUnits = .cpuAndNeuralEngine
let accepted = configuration.computeUnits == .cpuAndNeuralEngine
let payload: [String: Any] = [
    "coreml_framework": true,
    "cpu_and_neural_engine": accepted
]
let data = try! JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
print(String(data: data, encoding: .utf8)!)
'''.strip()

_MODEL_PROGRAM = r'''
import CoreML
import Foundation
let request = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))) as! [String: Any]
let values = request["input_values"] as! [NSNumber]
let array = try! MLMultiArray(shape: [NSNumber(value: values.count)], dataType: .float32)
for (index, value) in values.enumerated() { array[index] = value }
let configuration = MLModelConfiguration()
configuration.computeUnits = .cpuAndNeuralEngine
let model = try! MLModel(contentsOf: URL(fileURLWithPath: request["model_path"] as! String), configuration: configuration)
let provider = try! MLDictionaryFeatureProvider(dictionary: [request["input_name"] as! String: MLFeatureValue(multiArray: array)])
let started = DispatchTime.now().uptimeNanoseconds
let prediction = try! model.prediction(from: provider)
let elapsed = DispatchTime.now().uptimeNanoseconds - started
let output = prediction.featureValue(for: request["output_name"] as! String)!.multiArrayValue!
let result: [String: Any] = ["latency_nanoseconds": elapsed, "output_values": (0..<output.count).map { output[$0].doubleValue }]
let data = try! JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
print(String(data: data, encoding: .utf8)!)
'''.strip()


@dataclass(frozen=True, slots=True)
class CoreMLANESurfaceResult:
    platform: str
    architecture: str
    coreml_framework: bool
    cpu_and_neural_engine: bool
    execution_qualified: bool
    reason: str
    schema_version: int = ANE_SURFACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ANE_SURFACE_SCHEMA_VERSION:
            raise ValueError("unsupported ANE surface schema version")
        if self.execution_qualified:
            raise ValueError("surface probe cannot qualify ANE execution")
        if self.reason not in {
            "surface_available_model_probe_required",
            "non_apple_platform",
            "coreml_framework_unavailable",
            "probe_error",
        }:
            raise ValueError("invalid ANE surface reason")
        if self.reason == "surface_available_model_probe_required" and not (
            self.coreml_framework and self.cpu_and_neural_engine
        ):
            raise ValueError("ANE surface result contradicts its reason")

    @property
    def evidence_id(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CoreMLANESurfaceProbe:
    swift_executable: Path = Path("/usr/bin/swift")
    timeout_seconds: float = 10
    maximum_output_bytes: int = 4096

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.timeout_seconds)
            or not 0 < self.timeout_seconds <= 60
            or not 128 <= self.maximum_output_bytes <= 64 * 1024
        ):
            raise ValueError("invalid ANE surface probe bounds")

    def probe(
        self,
        *,
        platform_name: str | None = None,
        architecture: str | None = None,
    ) -> CoreMLANESurfaceResult:
        platform_name = platform_name or platform.system()
        architecture = architecture or platform.machine()
        if platform_name != "Darwin" or architecture != "arm64":
            return CoreMLANESurfaceResult(
                platform_name, architecture, False, False, False,
                "non_apple_platform",
            )
        try:
            completed = subprocess.run(
                (str(self.swift_executable), "-e", _SURFACE_PROGRAM),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=True,
                timeout=self.timeout_seconds,
            )
            if len(completed.stdout) > self.maximum_output_bytes:
                raise ValueError("ANE surface probe output exceeded its bound")
            payload = json.loads(completed.stdout)
            if (
                not isinstance(payload, dict)
                or set(payload) != {"coreml_framework", "cpu_and_neural_engine"}
                or type(payload["coreml_framework"]) is not bool
                or type(payload["cpu_and_neural_engine"]) is not bool
            ):
                raise ValueError("ANE surface probe output is invalid")
            available = payload["coreml_framework"] and payload["cpu_and_neural_engine"]
            return CoreMLANESurfaceResult(
                platform_name,
                architecture,
                payload["coreml_framework"],
                payload["cpu_and_neural_engine"],
                False,
                "surface_available_model_probe_required"
                if available else "coreml_framework_unavailable",
            )
        except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError):
            return CoreMLANESurfaceResult(
                platform_name, architecture, False, False, False, "probe_error"
            )


@dataclass(frozen=True, slots=True)
class CoreMLANEModelProbeConfig:
    model_path: Path
    integrity_manifest_path: Path
    model_root_sha256: str
    input_name: str
    output_name: str
    input_values: tuple[float, ...]
    expected_values: tuple[float, ...]
    baseline_latency_nanoseconds: int
    maximum_absolute_error: float = 1e-5

    def __post_init__(self) -> None:
        if (
            len(self.model_root_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.model_root_sha256)
        ):
            raise ValueError("invalid ANE fixture digest")
        if any(not 1 <= len(value) <= 128 for value in (self.input_name, self.output_name)):
            raise ValueError("invalid ANE fixture feature name")
        if (
            not 1 <= len(self.input_values) <= 4096
            or not 1 <= len(self.expected_values) <= 4096
            or any(not math.isfinite(value) for value in (*self.input_values, *self.expected_values))
            or type(self.baseline_latency_nanoseconds) is not int
            or self.baseline_latency_nanoseconds <= 0
            or not math.isfinite(self.maximum_absolute_error)
            or self.maximum_absolute_error < 0
        ):
            raise ValueError("invalid ANE fixture bounds")


@dataclass(frozen=True, slots=True)
class CoreMLPrediction:
    values: tuple[float, ...]
    latency_nanoseconds: int


def run_coreml_prediction(
    config: CoreMLANEModelProbeConfig,
    input_values: tuple[float, ...],
    *,
    swift_executable: Path = Path("/usr/bin/swift"),
    timeout_seconds: float = 30,
    maximum_output_bytes: int = 128 * 1024,
) -> CoreMLPrediction:
    """Run one bounded fixed-graph prediction through public Core ML APIs."""
    if (
        len(input_values) != len(config.input_values)
        or not input_values
        or any(type(value) not in (int, float) or not math.isfinite(value) for value in input_values)
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= 120
        or not 128 <= maximum_output_bytes <= 1024 * 1024
    ):
        raise ValueError("invalid Core ML prediction bounds")
    request = {
        "model_path": str(config.model_path.resolve(strict=True)),
        "input_name": config.input_name,
        "output_name": config.output_name,
        "input_values": list(input_values),
    }
    descriptor, request_path = tempfile.mkstemp(prefix="ane-execution-", suffix=".json")
    try:
        with open(descriptor, "w", encoding="utf-8", closefd=True) as handle:
            json.dump(request, handle, sort_keys=True, separators=(",", ":"))
        completed = subprocess.run(
            (str(swift_executable), "-e", _MODEL_PROGRAM, request_path),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=True,
            timeout=timeout_seconds,
        )
        if len(completed.stdout) > maximum_output_bytes:
            raise ValueError("Core ML prediction output exceeded its bound")
        payload = json.loads(completed.stdout)
        if not isinstance(payload, dict) or set(payload) != {
            "latency_nanoseconds", "output_values"
        }:
            raise ValueError("Core ML prediction output is invalid")
        raw_values = payload["output_values"]
        latency = payload["latency_nanoseconds"]
        if not isinstance(raw_values, list):
            raise ValueError("Core ML prediction output is invalid")
        values = tuple(raw_values)
        if (
            not 1 <= len(values) <= 4096
            or type(latency) is not int
            or latency <= 0
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)
        ):
            raise ValueError("Core ML prediction output is invalid")
        return CoreMLPrediction(values, latency)
    finally:
        Path(request_path).unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class CoreMLANEModelProbe:
    swift_executable: Path = Path("/usr/bin/swift")
    timeout_seconds: float = 30
    maximum_output_bytes: int = 128 * 1024

    def probe(
        self,
        config: CoreMLANEModelProbeConfig,
        *,
        hardware_fingerprint: str,
        environment_fingerprint: str,
        samples: int = 1,
    ) -> KernelProbeResult:
        manifest = verify_model_integrity(config.model_path, config.integrity_manifest_path)
        if manifest.get("root_sha256") != config.model_root_sha256:
            raise ValueError("ANE fixture integrity digest mismatch")
        expected_digest = self._values_digest(config.expected_values)

        def baseline() -> KernelMeasurement:
            return KernelMeasurement(
                expected_digest,
                config.baseline_latency_nanoseconds,
                config.expected_values,
            )

        def candidate() -> KernelMeasurement:
            prediction = run_coreml_prediction(
                config,
                config.input_values,
                swift_executable=self.swift_executable,
                timeout_seconds=self.timeout_seconds,
                maximum_output_bytes=self.maximum_output_bytes,
            )
            if len(prediction.values) != len(config.expected_values):
                raise ValueError("ANE model probe output is invalid")
            return KernelMeasurement(
                self._values_digest(prediction.values),
                prediction.latency_nanoseconds,
                prediction.values,
            )

        result = run_kernel_probe(
            KernelProbeConfig(
                hardware_fingerprint,
                environment_fingerprint,
                ExecutionBackend.COREML_DRAFT,
                f"coreml_fixed_graph@{config.model_root_sha256[:16]}",
                samples=samples,
                maximum_slowdown_ratio=1000,
                maximum_absolute_error=config.maximum_absolute_error,
            ),
            baseline,
            candidate,
        )
        if verify_model_integrity(config.model_path, config.integrity_manifest_path) != manifest:
            raise ValueError("ANE fixture changed during probing")
        return result

    @staticmethod
    def _values_digest(values: tuple[float, ...]) -> str:
        return hashlib.sha256(
            json.dumps(values, separators=(",", ":")).encode()
        ).hexdigest()
