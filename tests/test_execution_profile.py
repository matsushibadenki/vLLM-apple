import json
import stat
import tempfile
import unittest
from pathlib import Path

from tests.schema_validator import validate_instance
from tests.test_schemas import load_schema
from vllm_apple.compat import BackendCompatibility
from vllm_apple.execution import ExecutionBackend
from vllm_apple.execution_profile import (
    detect_apple_chip_profile,
    load_chip_profile,
    save_chip_profile,
)
from vllm_apple.types import HardwareInfo, MemoryInfo


def hardware(*, apple: bool = True) -> HardwareInfo:
    return HardwareInfo(
        platform="Darwin" if apple else "Linux",
        architecture="arm64" if apple else "x86_64",
        soc="Apple Test" if apple else "Test CPU",
        physical_cpu_count=4,
        logical_cpu_count=8,
        gpu_core_count=10 if apple else None,
        memory=MemoryInfo(total_bytes=16 * 1024**3, available_bytes=8 * 1024**3),
        is_apple_silicon=apple,
        os_version="test-os",
    )


def compatibility(*, compatible: bool = True) -> BackendCompatibility:
    return BackendCompatibility(
        executable="/test/vllm" if compatible else None,
        python_version="3.12.0" if compatible else None,
        vllm_version="1.0" if compatible else None,
        vllm_metal_version="1.0" if compatible else None,
        compatible=compatible,
        issues=() if compatible else ("vllm_executable_not_found",),
        transformers_version="5.12.1" if compatible else None,
        platform_module="vllm_metal.platform" if compatible else None,
        platform_class="MetalPlatform" if compatible else None,
        platform_is_cpu=False if compatible else None,
    )


class ExecutionProfileTests(unittest.TestCase):
    def test_detection_is_deterministic_and_reports_only_usable_backends(self) -> None:
        first = detect_apple_chip_profile(
            hardware(), compatibility(), mlx_available=True
        )
        second = detect_apple_chip_profile(
            hardware(), compatibility(), mlx_available=True
        )
        self.assertEqual(first, second)
        self.assertEqual(
            first.backends,
            (
                ExecutionBackend.VLLM_METAL,
                ExecutionBackend.NATIVE_MLX,
                ExecutionBackend.CPU,
            ),
        )
        validate_instance(
            first.to_dict(), load_schema("runtime/apple-chip-profile-v1.schema.json")
        )

    def test_transformers_change_invalidates_hardware_capability_fingerprint(self) -> None:
        baseline = compatibility()
        changed = BackendCompatibility(
            executable=baseline.executable,
            python_version=baseline.python_version,
            vllm_version=baseline.vllm_version,
            vllm_metal_version=baseline.vllm_metal_version,
            compatible=True,
            issues=(),
            transformers_version="5.15.0",
            platform_module=baseline.platform_module,
            platform_class=baseline.platform_class,
            platform_is_cpu=baseline.platform_is_cpu,
        )
        self.assertNotEqual(
            detect_apple_chip_profile(
                hardware(), baseline, mlx_available=True
            ).hardware_fingerprint,
            detect_apple_chip_profile(
                hardware(), changed, mlx_available=True
            ).hardware_fingerprint,
        )

    def test_incompatible_backend_falls_back_to_cpu_and_records_issue(self) -> None:
        profile = detect_apple_chip_profile(
            hardware(), compatibility(compatible=False), mlx_available=False
        )
        self.assertEqual(profile.backends, (ExecutionBackend.CPU,))
        self.assertEqual(profile.capability_issues, ("vllm_executable_not_found",))

    def test_non_apple_host_never_claims_apple_backends(self) -> None:
        profile = detect_apple_chip_profile(
            hardware(apple=False), compatibility(), mlx_available=True
        )
        self.assertEqual(profile.backends, (ExecutionBackend.CPU,))
        self.assertEqual(profile.precisions, ("fp32",))

    def test_profile_round_trip_is_private_and_atomic(self) -> None:
        profile = detect_apple_chip_profile(
            hardware(), compatibility(), mlx_available=False
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chip.json"
            self.assertEqual(save_chip_profile(profile, path), path)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(load_chip_profile(path), profile)

            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["profile_version"] = 2
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_chip_profile(path)

            payload["profile_version"] = 1
            payload["total_memory_bytes"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_chip_profile(path)


if __name__ == "__main__":
    unittest.main()
