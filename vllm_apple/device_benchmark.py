"""Bounded, capability-gated CPU/GPU/ANE microbenchmark contracts."""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import stat
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from .device_capability import DeviceCapabilityRegistry, DeviceEligibilityRequest
from .execution import ExecutionBackend, WorkloadPhase


DEVICE_BENCHMARK_SCHEMA_VERSION = 1
MAX_DEVICE_BENCHMARK_SAMPLES = 64
MAX_BENCHMARK_DIMENSIONS = 8
MAX_DEVICE_BENCHMARK_BYTES = 256 * 1024
MAX_DEVICE_BENCHMARK_SUITE_REPORTS = 64

_REPRESENTATIVE_DIMENSIONS = {
    "vector_add": (256,),
    "matmul": (8, 8, 8),
    "kv_copy": (128,),
    "attention": (16, 8),
    "paged_attention": (14, 8),
    "mla": (16, 4, 8),
}


@dataclass(frozen=True, slots=True)
class DeviceBenchmarkConfig:
    operator: str
    backend: ExecutionBackend
    phase: WorkloadPhase
    precision: str
    dimensions: tuple[int, ...]
    batch_size: int
    samples: int = 5

    def __post_init__(self) -> None:
        if (
            not isinstance(self.operator, str)
            or not 1 <= len(self.operator) <= 128
            or not isinstance(self.precision, str)
            or not 1 <= len(self.precision) <= 128
            or not isinstance(self.backend, ExecutionBackend)
            or not isinstance(self.phase, WorkloadPhase)
            or not 1 <= len(self.dimensions) <= MAX_BENCHMARK_DIMENSIONS
            or any(type(value) is not int or not 1 <= value <= 1_048_576 for value in self.dimensions)
            or type(self.batch_size) is not int
            or not 1 <= self.batch_size <= 65_536
            or type(self.samples) is not int
            or not 1 <= self.samples <= MAX_DEVICE_BENCHMARK_SAMPLES
        ):
            raise ValueError("invalid device benchmark configuration")


@dataclass(frozen=True, slots=True)
class DeviceBenchmarkMeasurement:
    total_nanoseconds: int
    execution_nanoseconds: int
    conversion_nanoseconds: int
    synchronization_nanoseconds: int
    work_items: int
    peak_memory_bytes: int | None
    energy_microjoules: float | None
    output_digest: str

    def __post_init__(self) -> None:
        components = (
            self.execution_nanoseconds,
            self.conversion_nanoseconds,
            self.synchronization_nanoseconds,
        )
        if (
            type(self.total_nanoseconds) is not int
            or self.total_nanoseconds <= 0
            or any(type(value) is not int or value < 0 for value in components)
            or sum(components) > self.total_nanoseconds
            or type(self.work_items) is not int
            or self.work_items <= 0
            or (
                self.peak_memory_bytes is not None
                and (type(self.peak_memory_bytes) is not int or self.peak_memory_bytes < 0)
            )
            or (
                self.energy_microjoules is not None
                and (
                    isinstance(self.energy_microjoules, bool)
                    or not isinstance(self.energy_microjoules, (int, float))
                    or not math.isfinite(self.energy_microjoules)
                    or self.energy_microjoules < 0
                )
            )
            or len(self.output_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.output_digest)
        ):
            raise ValueError("invalid device benchmark measurement")


@dataclass(frozen=True, slots=True)
class DeviceBenchmarkReport:
    hardware_fingerprint: str
    environment_fingerprint: str
    capability_id: str
    config: DeviceBenchmarkConfig
    measurements: tuple[DeviceBenchmarkMeasurement, ...]
    cold_load_nanoseconds: int | None
    schema_version: int = DEVICE_BENCHMARK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version != DEVICE_BENCHMARK_SCHEMA_VERSION
            or not self.hardware_fingerprint
            or not self.environment_fingerprint
            or len(self.capability_id) != 24
            or any(character not in "0123456789abcdef" for character in self.capability_id)
            or len(self.measurements) != self.config.samples
            or (
                self.cold_load_nanoseconds is not None
                and (type(self.cold_load_nanoseconds) is not int or self.cold_load_nanoseconds <= 0)
            )
        ):
            raise ValueError("invalid device benchmark report")
        digests = {measurement.output_digest for measurement in self.measurements}
        if len(digests) != 1:
            raise ValueError("device benchmark output changed between samples")

    @property
    def report_id(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]

    def to_dict(self) -> dict[str, object]:
        totals = tuple(value.total_nanoseconds for value in self.measurements)
        execution = tuple(value.execution_nanoseconds for value in self.measurements)
        conversion = tuple(value.conversion_nanoseconds for value in self.measurements)
        synchronization = tuple(
            value.synchronization_nanoseconds for value in self.measurements
        )
        work_items = sum(value.work_items for value in self.measurements)
        total_seconds = sum(totals) / 1_000_000_000
        peak_values = tuple(
            value.peak_memory_bytes
            for value in self.measurements
            if value.peak_memory_bytes is not None
        )
        energy_values = tuple(
            float(value.energy_microjoules)
            for value in self.measurements
            if value.energy_microjoules is not None
        )
        config = asdict(self.config)
        config["backend"] = self.config.backend.value
        config["phase"] = self.config.phase.value
        config["dimensions"] = list(self.config.dimensions)
        return {
            "schema_version": self.schema_version,
            "hardware_fingerprint": self.hardware_fingerprint,
            "environment_fingerprint": self.environment_fingerprint,
            "capability_id": self.capability_id,
            "config": config,
            "sample_count": len(self.measurements),
            "measurements": [asdict(value) for value in self.measurements],
            "cold_load_nanoseconds": self.cold_load_nanoseconds,
            "median_total_nanoseconds": int(statistics.median(totals)),
            "median_execution_nanoseconds": int(statistics.median(execution)),
            "median_conversion_nanoseconds": int(statistics.median(conversion)),
            "median_synchronization_nanoseconds": int(statistics.median(synchronization)),
            "throughput_work_items_per_second": work_items / total_seconds,
            "peak_memory_bytes": max(peak_values) if peak_values else None,
            "energy_microjoules": sum(energy_values) if len(energy_values) == len(self.measurements) else None,
            "output_digest": self.measurements[0].output_digest,
        }


@dataclass(frozen=True, slots=True)
class DeviceBenchmarkSuite:
    hardware_fingerprint: str
    environment_fingerprint: str
    reports: tuple[DeviceBenchmarkReport, ...]

    def __post_init__(self) -> None:
        if (
            not self.hardware_fingerprint
            or not self.environment_fingerprint
            or not 1 <= len(self.reports) <= MAX_DEVICE_BENCHMARK_SUITE_REPORTS
            or any(
                report.hardware_fingerprint != self.hardware_fingerprint
                or report.environment_fingerprint != self.environment_fingerprint
                for report in self.reports
            )
        ):
            raise ValueError("invalid device benchmark suite")
        identities = {
            (
                report.config.backend,
                report.config.operator,
                report.config.phase,
                report.config.precision,
                report.config.dimensions,
                report.config.batch_size,
            )
            for report in self.reports
        }
        if len(identities) != len(self.reports):
            raise ValueError("device benchmark suite contains duplicate reports")

    @property
    def suite_id(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "hardware_fingerprint": self.hardware_fingerprint,
                    "environment_fingerprint": self.environment_fingerprint,
                    "reports": [report.report_id for report in self.reports],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:24]


def representative_device_benchmark_configs(
    registry: DeviceCapabilityRegistry,
    *,
    samples: int = 5,
    dimension_overrides: dict[str, tuple[int, ...]] | None = None,
) -> tuple[DeviceBenchmarkConfig, ...]:
    """Build a deterministic suite only from currently usable capabilities."""
    overrides = dimension_overrides or {}
    if len(overrides) > MAX_DEVICE_BENCHMARK_SUITE_REPORTS:
        raise ValueError("too many device benchmark dimension overrides")
    configs: list[DeviceBenchmarkConfig] = []
    for capability in registry.snapshot():
        if capability.status != "available":
            continue
        operator = capability.operators[0]
        dimensions = overrides.get(operator, _REPRESENTATIVE_DIMENSIONS.get(operator))
        if dimensions is None:
            continue
        phase = next(
            (
                value
                for value in (
                    WorkloadPhase.AUXILIARY,
                    WorkloadPhase.PREFILL,
                    WorkloadPhase.DECODE,
                )
                if value in capability.phases
            ),
            None,
        )
        if phase is None or "fp32" not in capability.precisions:
            continue
        configs.append(DeviceBenchmarkConfig(
            operator,
            capability.backend,
            phase,
            "fp32",
            dimensions,
            1,
            samples,
        ))
        if len(configs) > MAX_DEVICE_BENCHMARK_SUITE_REPORTS:
            raise ValueError("device benchmark suite is too large")
    return tuple(sorted(
        configs,
        key=lambda value: (value.backend.value, value.operator, value.dimensions),
    ))


def run_device_benchmark_suite(
    registry: DeviceCapabilityRegistry,
    configs: tuple[DeviceBenchmarkConfig, ...],
    operations: dict[
        tuple[ExecutionBackend, str],
        Callable[[int], DeviceBenchmarkMeasurement],
    ],
    *,
    cold_loads: dict[tuple[ExecutionBackend, str], Callable[[], int]] | None = None,
) -> DeviceBenchmarkSuite:
    if not 1 <= len(configs) <= MAX_DEVICE_BENCHMARK_SUITE_REPORTS:
        raise ValueError("device benchmark suite configuration is invalid")
    reports: list[DeviceBenchmarkReport] = []
    load_operations = cold_loads or {}
    for config in configs:
        key = (config.backend, config.operator)
        operation = operations.get(key)
        if operation is None:
            raise ValueError("device benchmark suite operation is missing")
        reports.append(run_device_microbenchmark(
            config,
            registry,
            operation,
            cold_load=load_operations.get(key),
        ))
    return DeviceBenchmarkSuite(
        registry.hardware_fingerprint,
        registry.environment_fingerprint,
        tuple(reports),
    )


def run_device_microbenchmark(
    config: DeviceBenchmarkConfig,
    registry: DeviceCapabilityRegistry,
    operation: Callable[[int], DeviceBenchmarkMeasurement],
    *,
    cold_load: Callable[[], int] | None = None,
) -> DeviceBenchmarkReport:
    """Measure only the exact profile-bound backend capability requested."""
    decision = registry.decide(DeviceEligibilityRequest(
        config.operator,
        config.phase,
        config.precision,
        (config.backend,),
    ))
    if decision.selected is not config.backend:
        raise RuntimeError("device benchmark backend was not selected")
    capability = next(
        entry
        for entry in registry.snapshot()
        if entry.backend is config.backend and entry.operators == (config.operator,)
    )
    load_nanoseconds = cold_load() if cold_load is not None else None
    if load_nanoseconds is not None and (
        type(load_nanoseconds) is not int or load_nanoseconds <= 0
    ):
        raise ValueError("invalid device benchmark cold load duration")
    measurements = tuple(operation(index) for index in range(config.samples))
    if any(not isinstance(value, DeviceBenchmarkMeasurement) for value in measurements):
        raise ValueError("device benchmark operation returned an invalid measurement")
    return DeviceBenchmarkReport(
        registry.hardware_fingerprint,
        registry.environment_fingerprint,
        capability.capability_id,
        config,
        measurements,
        load_nanoseconds,
    )


class NativeCPUBenchmarkAdapter:
    """Deterministic standard-library kernels for the CPU benchmark suite."""

    def operation(
        self, config: DeviceBenchmarkConfig
    ) -> Callable[[int], DeviceBenchmarkMeasurement]:
        return lambda sample_index: self.measure(config, sample_index)

    def measure(
        self, config: DeviceBenchmarkConfig, _sample_index: int
    ) -> DeviceBenchmarkMeasurement:
        if config.backend is not ExecutionBackend.CPU or config.batch_size > 64:
            raise ValueError("CPU benchmark configuration is unsupported")
        if config.operator == "vector_add" and len(config.dimensions) == 1:
            count = config.dimensions[0]
            if count > 4096:
                raise ValueError("CPU vector benchmark shape is too large")
            left = tuple(float(index % 31) for index in range(count))
            right = tuple(float(index % 17) for index in range(count))

            def operation():
                output = ()
                for _ in range(config.batch_size):
                    output = tuple(a + b for a, b in zip(left, right))
                return output, count * config.batch_size
        elif config.operator == "matmul" and len(config.dimensions) == 3:
            rows, inner, columns = config.dimensions
            if max(config.dimensions) > 64 or rows * inner * columns > 262_144:
                raise ValueError("CPU matrix benchmark shape is too large")
            left = tuple(
                tuple(float((row * inner + index) % 13) for index in range(inner))
                for row in range(rows)
            )
            right_columns = tuple(
                tuple(float((index * columns + column) % 11) for index in range(inner))
                for column in range(columns)
            )

            def operation():
                output = ()
                for _ in range(config.batch_size):
                    output = tuple(
                        tuple(sum(a * b for a, b in zip(row, column)) for column in right_columns)
                        for row in left
                    )
                return output, rows * inner * columns * config.batch_size
        elif config.operator == "kv_copy" and len(config.dimensions) == 1:
            count = config.dimensions[0]
            if count > 65_536:
                raise ValueError("CPU KV benchmark shape is too large")
            source = tuple(float(index % 251) for index in range(count))

            def operation():
                output = ()
                for _ in range(config.batch_size):
                    output = tuple(source)
                return output, count * config.batch_size
        else:
            raise ValueError("CPU benchmark operator or shape is unsupported")
        started = time.perf_counter_ns()
        output, work_items = operation()
        elapsed = max(1, time.perf_counter_ns() - started)
        digest = hashlib.sha256(
            json.dumps(output, separators=(",", ":")).encode()
        ).hexdigest()
        return DeviceBenchmarkMeasurement(
            elapsed, elapsed, 0, 0, work_items, None, None, digest
        )


class BoundedCPUReferenceBenchmarkAdapter:
    """CPU baseline for an accelerator's exact, deterministic workload identity."""

    def __init__(
        self,
        operator: str,
        reference: Callable[[tuple[float, ...]], tuple[float, ...]],
        input_values: tuple[float, ...],
        *,
        maximum_values: int = 4096,
    ) -> None:
        if (
            not isinstance(operator, str)
            or not 1 <= len(operator) <= 128
            or not callable(reference)
            or type(maximum_values) is not int
            or not 1 <= maximum_values <= 4096
            or not 1 <= len(input_values) <= maximum_values
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in input_values)
        ):
            raise ValueError("invalid bounded CPU reference benchmark")
        self.operator = operator
        self.reference = reference
        self.input_values = tuple(float(value) for value in input_values)
        self.maximum_values = maximum_values

    def operation(
        self, config: DeviceBenchmarkConfig
    ) -> Callable[[int], DeviceBenchmarkMeasurement]:
        if (
            config.backend is not ExecutionBackend.CPU
            or config.operator != self.operator
            or config.phase is not WorkloadPhase.AUXILIARY
            or config.precision != "fp32"
            or config.batch_size != 1
            or config.dimensions != (len(self.input_values),)
        ):
            raise ValueError("CPU reference benchmark configuration is unsupported")

        def measure(_sample_index: int) -> DeviceBenchmarkMeasurement:
            started = time.perf_counter_ns()
            output = self.reference(self.input_values)
            elapsed = max(1, time.perf_counter_ns() - started)
            if (
                not isinstance(output, tuple)
                or not 1 <= len(output) <= self.maximum_values
                or any(
                    type(value) not in (int, float) or not math.isfinite(value)
                    for value in output
                )
            ):
                raise ValueError("CPU reference benchmark returned invalid output")
            normalized = tuple(float(value) for value in output)
            digest = hashlib.sha256(
                json.dumps(normalized, separators=(",", ":")).encode()
            ).hexdigest()
            return DeviceBenchmarkMeasurement(
                elapsed, elapsed, 0, 0, len(normalized), None, None, digest
            )

        return measure


class CoreMLFixedGraphBenchmarkAdapter:
    """Bridge a loaded fixed graph into the common end-to-end benchmark schema."""

    def __init__(self, backend, resource) -> None:
        from .coreml_backend import CoreMLFixedGraphBackend, CoreMLFixedGraphResource

        if not isinstance(backend, CoreMLFixedGraphBackend) or not isinstance(
            resource, CoreMLFixedGraphResource
        ):
            raise ValueError("invalid Core ML benchmark resource")
        self.backend = backend
        self.resource = resource

    def operation(
        self,
        config: DeviceBenchmarkConfig,
        input_values: tuple[float, ...],
        *,
        expected_values: tuple[float, ...] | None = None,
        maximum_absolute_error: float = 0,
    ) -> Callable[[int], DeviceBenchmarkMeasurement]:
        return lambda sample_index: self.measure(
            config, sample_index, input_values=input_values,
            expected_values=expected_values,
            maximum_absolute_error=maximum_absolute_error,
        )

    def measure(
        self,
        config: DeviceBenchmarkConfig,
        _sample_index: int,
        *,
        input_values: tuple[float, ...],
        expected_values: tuple[float, ...] | None = None,
        maximum_absolute_error: float = 0,
    ) -> DeviceBenchmarkMeasurement:
        if (
            config.backend is not ExecutionBackend.COREML_DRAFT
            or config.operator != self.resource.operator
            or config.phase is not WorkloadPhase.AUXILIARY
            or config.precision != "fp32"
            or config.batch_size != 1
            or config.dimensions != (len(input_values),)
        ):
            raise ValueError("Core ML benchmark configuration is unsupported")
        if (
            not math.isfinite(maximum_absolute_error)
            or maximum_absolute_error < 0
            or expected_values is not None
            and (
                len(expected_values) != config.dimensions[0]
                or any(
                    type(value) not in (int, float) or not math.isfinite(value)
                    for value in expected_values
                )
            )
        ):
            raise ValueError("Core ML benchmark reference is invalid")
        started = time.perf_counter_ns()
        result = self.backend.execute(self.resource, input_values)
        total = max(1, time.perf_counter_ns() - started)
        execution = min(total, result.latency_nanoseconds)
        normalized = tuple(float(value) for value in result.values)
        if expected_values is not None:
            reference = tuple(float(value) for value in expected_values)
            if any(
                abs(expected - actual) > maximum_absolute_error
                for expected, actual in zip(reference, normalized)
            ):
                raise ValueError("Core ML benchmark output mismatch")
            normalized = reference
        digest = hashlib.sha256(
            json.dumps(normalized, separators=(",", ":")).encode()
        ).hexdigest()
        return DeviceBenchmarkMeasurement(
            total,
            execution,
            0,
            0,
            len(result.values),
            None,
            None,
            digest,
        )


class NativeKernelBenchmarkAdapter:
    """Measure MLX/Metal adapters while retaining subprocess end-to-end overhead."""

    def __init__(self, adapter, backend: ExecutionBackend) -> None:
        if backend not in {ExecutionBackend.NATIVE_MLX, ExecutionBackend.NATIVE_METAL}:
            raise ValueError("native kernel benchmark backend is unsupported")
        if not callable(getattr(adapter, "measure_operator", None)):
            raise ValueError("native kernel benchmark adapter is invalid")
        self.adapter = adapter
        self.backend = backend

    def operation(
        self, config: DeviceBenchmarkConfig
    ) -> Callable[[int], DeviceBenchmarkMeasurement]:
        if config.backend is not self.backend:
            raise ValueError("native kernel benchmark backend does not match")
        work_items = config.batch_size * math.prod(config.dimensions)

        def measure(_sample_index: int) -> DeviceBenchmarkMeasurement:
            started = time.perf_counter_ns()
            result = self.adapter.measure_operator(config.operator)
            total = max(1, time.perf_counter_ns() - started)
            execution = min(total, result.latency_nanoseconds)
            return DeviceBenchmarkMeasurement(
                total,
                execution,
                0,
                0,
                work_items,
                None,
                None,
                result.output_digest,
            )

        return measure


def save_device_benchmark(report: DeviceBenchmarkReport, path: Path) -> Path:
    payload = report.to_dict()
    payload["report_id"] = report.report_id
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    if len(encoded) > MAX_DEVICE_BENCHMARK_BYTES:
        raise ValueError("device benchmark exceeded 256 KiB")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != os.getuid()
        or parent.st_mode & 0o077
    ):
        raise ValueError("device benchmark directory must be private")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return path


def load_device_benchmark(
    path: Path,
    *,
    hardware_fingerprint: str,
    environment_fingerprint: str,
    capability_id: str,
) -> DeviceBenchmarkReport:
    attributes = path.lstat()
    if (
        not stat.S_ISREG(attributes.st_mode)
        or attributes.st_uid != os.getuid()
        or attributes.st_mode & 0o077
        or attributes.st_size > MAX_DEVICE_BENCHMARK_BYTES
    ):
        raise ValueError("device benchmark must be a bounded private regular file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version", "report_id", "hardware_fingerprint",
        "environment_fingerprint", "capability_id", "config", "sample_count",
        "measurements", "cold_load_nanoseconds", "median_total_nanoseconds",
        "median_execution_nanoseconds", "median_conversion_nanoseconds",
        "median_synchronization_nanoseconds", "throughput_work_items_per_second",
        "peak_memory_bytes", "energy_microjoules", "output_digest",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("invalid device benchmark fields")
    if (
        payload["hardware_fingerprint"] != hardware_fingerprint
        or payload["environment_fingerprint"] != environment_fingerprint
        or payload["capability_id"] != capability_id
    ):
        raise ValueError("device benchmark identity mismatch")
    try:
        config_value = payload["config"]
        measurement_values = payload["measurements"]
        if not isinstance(config_value, dict) or set(config_value) != {
            "operator", "backend", "phase", "precision", "dimensions",
            "batch_size", "samples",
        }:
            raise ValueError("invalid device benchmark configuration fields")
        if not isinstance(measurement_values, list):
            raise ValueError("invalid device benchmark measurements")
        config = DeviceBenchmarkConfig(
            operator=config_value["operator"],
            backend=ExecutionBackend(config_value["backend"]),
            phase=WorkloadPhase(config_value["phase"]),
            precision=config_value["precision"],
            dimensions=tuple(config_value["dimensions"]),
            batch_size=config_value["batch_size"],
            samples=config_value["samples"],
        )
        measurement_fields = {
            "total_nanoseconds", "execution_nanoseconds", "conversion_nanoseconds",
            "synchronization_nanoseconds", "work_items", "peak_memory_bytes",
            "energy_microjoules", "output_digest",
        }
        if any(not isinstance(value, dict) or set(value) != measurement_fields for value in measurement_values):
            raise ValueError("invalid device benchmark measurement fields")
        report = DeviceBenchmarkReport(
            hardware_fingerprint=payload["hardware_fingerprint"],
            environment_fingerprint=payload["environment_fingerprint"],
            capability_id=payload["capability_id"],
            config=config,
            measurements=tuple(
                DeviceBenchmarkMeasurement(**value) for value in measurement_values
            ),
            cold_load_nanoseconds=payload["cold_load_nanoseconds"],
            schema_version=payload["schema_version"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid device benchmark values") from error
    canonical = report.to_dict()
    if (
        payload["report_id"] != report.report_id
        or payload["sample_count"] != canonical["sample_count"]
        or any(payload[name] != canonical[name] for name in (
            "median_total_nanoseconds", "median_execution_nanoseconds",
            "median_conversion_nanoseconds", "median_synchronization_nanoseconds",
            "throughput_work_items_per_second", "peak_memory_bytes",
            "energy_microjoules", "output_digest",
        ))
    ):
        raise ValueError("device benchmark derived values do not match measurements")
    return report


def load_profile_device_benchmark(
    path: Path,
    *,
    hardware_fingerprint: str,
    environment_fingerprint: str,
) -> DeviceBenchmarkReport:
    """Load a strict report when its capability ID is not known by the caller yet."""
    attributes = path.lstat()
    if (
        not stat.S_ISREG(attributes.st_mode)
        or attributes.st_uid != os.getuid()
        or attributes.st_mode & 0o077
        or attributes.st_size > MAX_DEVICE_BENCHMARK_BYTES
    ):
        raise ValueError("device benchmark must be a bounded private regular file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("invalid device benchmark fields")
    capability_id = payload.get("capability_id")
    if not isinstance(capability_id, str):
        raise ValueError("invalid device benchmark capability ID")
    return load_device_benchmark(
        path,
        hardware_fingerprint=hardware_fingerprint,
        environment_fingerprint=environment_fingerprint,
        capability_id=capability_id,
    )
