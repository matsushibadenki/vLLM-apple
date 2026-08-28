import hashlib
import importlib.metadata
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from vllm_apple.execution import AppleChipProfile, ExecutionBackend
from vllm_apple.kernel_probe import KernelMeasurement, KernelProbeConfig, run_kernel_probe
from vllm_apple.profile import build_profile
from vllm_apple.runtime_probe import RuntimeProbeCoordinator, discover_runtime_versions
from vllm_apple.scheduler import ScheduleRequest
from vllm_apple.service import RuntimeService
from vllm_apple.types import Backend, HardwareInfo, MemoryInfo


def result(
    backend: ExecutionBackend,
    operator: str,
    hardware_fingerprint: str,
    environment_fingerprint: str,
    *,
    passing: bool,
):
    expected = hashlib.sha256(b"expected").hexdigest()
    actual = expected if passing else hashlib.sha256(b"wrong").hexdigest()
    return run_kernel_probe(
        KernelProbeConfig(
            hardware_fingerprint,
            environment_fingerprint,
            backend,
            operator,
            samples=1,
            maximum_slowdown_ratio=2,
        ),
        lambda: KernelMeasurement(expected, 100),
        lambda: KernelMeasurement(actual, 100),
    )


class FakeMLXAdapter:
    def __init__(self, passing: bool = True) -> None:
        self.passing = passing

    def probe_suite(self, *, hardware_fingerprint: str, environment_fingerprint: str, samples: int):
        return (
            result(
                ExecutionBackend.NATIVE_MLX,
                "matmul",
                hardware_fingerprint,
                environment_fingerprint,
                passing=self.passing,
            ),
        )


class FakeMetalAdapter:
    def probe_vector_add(
        self, *, hardware_fingerprint: str, environment_fingerprint: str, samples: int
    ):
        return result(
            ExecutionBackend.NATIVE_METAL,
            "vector_add",
            hardware_fingerprint,
            environment_fingerprint,
            passing=True,
        )

    def probe_suite(
        self, *, hardware_fingerprint: str, environment_fingerprint: str, samples: int
    ):
        return (
            self.probe_vector_add(
                hardware_fingerprint=hardware_fingerprint,
                environment_fingerprint=environment_fingerprint,
                samples=samples,
            ),
        )


def chip() -> AppleChipProfile:
    return AppleChipProfile(
        profile_version=1,
        hardware_fingerprint="hardware",
        soc="Apple Test",
        total_memory_bytes=16 * 1024**3,
        backends=(ExecutionBackend.NATIVE_MLX, ExecutionBackend.CPU),
        platform="Darwin",
        architecture="arm64",
        os_version="test-os",
    )


def service() -> RuntimeService:
    hardware = HardwareInfo(
        platform="Darwin",
        architecture="arm64",
        soc="Apple Test",
        physical_cpu_count=8,
        logical_cpu_count=8,
        gpu_core_count=10,
        memory=MemoryInfo(total_bytes=16 * 1024**3, available_bytes=8 * 1024**3),
        is_apple_silicon=True,
        os_version="test-os",
    )
    return RuntimeService(profile=build_profile(hardware))


class RuntimeProbeCoordinatorTests(unittest.TestCase):
    def test_version_discovery_is_bounded_and_does_not_import_mlx(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="Metal Toolchain 1\n", stderr=""
        )
        with (
            patch("vllm_apple.runtime_probe.subprocess.run", return_value=completed),
            patch("vllm_apple.runtime_probe.importlib.metadata.version", return_value="0.31.0"),
        ):
            versions = discover_runtime_versions("vllm-metal-1")
        self.assertEqual(versions.toolchain_version, "Metal Toolchain 1")
        self.assertEqual(versions.mlx_version, "0.31.0")
        self.assertEqual(versions.backend_version, "vllm-metal-1")

        with (
            patch(
                "vllm_apple.runtime_probe.subprocess.run",
                side_effect=OSError("missing"),
            ),
            patch(
                "vllm_apple.runtime_probe.importlib.metadata.version",
                side_effect=importlib.metadata.PackageNotFoundError,
            ),
        ):
            unavailable = discover_runtime_versions(None)
        self.assertEqual(unavailable.toolchain_version, "unavailable")
        self.assertEqual(unavailable.mlx_version, "unavailable")
        self.assertEqual(unavailable.backend_version, "unavailable")

    def coordinator(self, *, mlx_passing: bool = True) -> RuntimeProbeCoordinator:
        return RuntimeProbeCoordinator(
            chip(),
            toolchain_version="swift-test",
            mlx_version="mlx-test",
            backend_version="backend-test",
            mlx_adapter=FakeMLXAdapter(mlx_passing),  # type: ignore[arg-type]
            metal_adapter=FakeMetalAdapter(),  # type: ignore[arg-type]
        )

    def test_probe_registry_is_atomically_installed_at_safe_point(self) -> None:
        runtime = service()
        report = self.coordinator().probe_and_install(runtime, samples=1)
        self.assertTrue(report.dispatcher_applied)
        self.assertEqual(len(report.results), 2)
        self.assertEqual(
            runtime.scheduler.choose_backend(ScheduleRequest("matmul", 1, batch_size=8)),
            Backend.MLX_GPU,
        )

    def test_matching_cache_skips_native_probe_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "probe.json"
            first = RuntimeProbeCoordinator(
                chip(),
                toolchain_version="swift-test",
                mlx_version="mlx-test",
                backend_version="backend-test",
                mlx_adapter=FakeMLXAdapter(),  # type: ignore[arg-type]
                metal_adapter=FakeMetalAdapter(),  # type: ignore[arg-type]
                cache_path=cache_path,
            ).probe_and_install(service(), samples=1)
            self.assertEqual(first.cache_status, "miss")

            mlx = FakeMLXAdapter()
            metal = FakeMetalAdapter()
            with (
                patch.object(mlx, "probe_suite", side_effect=AssertionError("cache miss")),
                patch.object(
                    metal,
                    "probe_suite",
                    side_effect=AssertionError("cache miss"),
                ),
            ):
                second = RuntimeProbeCoordinator(
                    chip(),
                    toolchain_version="swift-test",
                    mlx_version="mlx-test",
                    backend_version="backend-test",
                    mlx_adapter=mlx,  # type: ignore[arg-type]
                    metal_adapter=metal,  # type: ignore[arg-type]
                    cache_path=cache_path,
                ).probe_and_install(service(), samples=1)
            self.assertEqual(second.cache_status, "hit")
            self.assertEqual(second.results, first.results)

    def test_active_request_defers_dispatcher_until_completion(self) -> None:
        runtime = service()
        reservation = runtime.admit_schedule(ScheduleRequest("matmul", 1, batch_size=8))
        report = self.coordinator(mlx_passing=False).probe_and_install(runtime, samples=1)
        self.assertFalse(report.dispatcher_applied)
        runtime.complete_schedule(reservation)
        self.assertEqual(
            runtime.scheduler.choose_backend(ScheduleRequest("matmul", 1, batch_size=8)),
            Backend.CPU,
        )


if __name__ == "__main__":
    unittest.main()
