import hashlib
import json
import stat
import tempfile
import unittest
from pathlib import Path

from tests.schema_validator import validate_instance
from tests.test_schemas import load_schema
from vllm_apple.execution import ExecutionBackend
from vllm_apple.kernel_probe import (
    KernelCapabilityRegistry,
    KernelMeasurement,
    KernelProbeCache,
    KernelProbeConfig,
    build_environment_fingerprint,
    run_kernel_probe,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class KernelProbeTests(unittest.TestCase):
    def test_probe_cache_is_private_atomic_strict_and_profile_bound(self) -> None:
        result = run_kernel_probe(
            self.config(),
            lambda: KernelMeasurement(digest("same"), 100),
            lambda: KernelMeasurement(digest("same"), 100),
        )
        registry = KernelCapabilityRegistry("hardware-a", "macos-mlx-metal-a")
        registry.record(result)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.json"
            cache = KernelProbeCache(path, "hardware-a", "macos-mlx-metal-a")
            cache.save(registry)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            validate_instance(
                json.loads(path.read_text()),
                load_schema("runtime/kernel-probe-cache-v1.schema.json"),
            )
            self.assertEqual(cache.load().snapshot(), registry.snapshot())

            payload = json.loads(path.read_text())
            payload["results"][0]["samples_completed"] = True
            path.write_text(json.dumps(payload))
            path.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "integer"):
                cache.load()

            cache.save(registry)
            legacy = json.loads(path.read_text())
            legacy["probe_suite_version"] = 2
            path.write_text(json.dumps(legacy))
            path.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "unsupported"):
                cache.load()

            cache.save(registry)
            wrong = KernelProbeCache(path, "other-hardware", "macos-mlx-metal-a")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                wrong.load()

            timed = KernelProbeCache(
                path,
                "hardware-a",
                "macos-mlx-metal-a",
                max_age_seconds=10,
                clock=lambda: 1_000,
            )
            timed.save(registry)
            expired = KernelProbeCache(
                path,
                "hardware-a",
                "macos-mlx-metal-a",
                max_age_seconds=10,
                clock=lambda: 1_011,
            )
            with self.assertRaisesRegex(ValueError, "expired"):
                expired.load()

    def test_environment_fingerprint_changes_with_toolchain(self) -> None:
        values = {
            "platform": "Darwin-arm64",
            "os_version": "26.0",
            "toolchain_version": "metal-1",
            "mlx_version": "0.30",
            "backend_version": "1.0",
        }
        first = build_environment_fingerprint(**values)
        second = build_environment_fingerprint(
            **{**values, "toolchain_version": "metal-2"}
        )
        self.assertEqual(len(first), 24)
        self.assertNotEqual(first, second)

    def config(self, **changes: object) -> KernelProbeConfig:
        values = {
            "hardware_fingerprint": "hardware-a",
            "environment_fingerprint": "macos-mlx-metal-a",
            "backend": ExecutionBackend.NATIVE_METAL,
            "operator": "paged_attention",
            "samples": 3,
            "maximum_slowdown_ratio": 1.25,
            **changes,
        }
        return KernelProbeConfig(**values)  # type: ignore[arg-type]

    def test_matching_probe_is_schema_valid_and_usable(self) -> None:
        result = run_kernel_probe(
            self.config(),
            lambda: KernelMeasurement(digest("same"), 100),
            lambda: KernelMeasurement(digest("same"), 110),
        )
        self.assertTrue(result.passed)
        self.assertFalse(result.quarantined)
        self.assertEqual(result.slowdown_ratio, 1.1)
        validate_instance(
            result.to_dict(), load_schema("runtime/kernel-probe-result-v1.schema.json")
        )
        registry = KernelCapabilityRegistry("hardware-a", "macos-mlx-metal-a")
        registry.record(result)
        self.assertTrue(registry.is_usable(result.backend, result.operator))

    def test_correctness_failure_is_quarantined_and_sticky(self) -> None:
        failed = run_kernel_probe(
            self.config(),
            lambda: KernelMeasurement(digest("reference"), 100),
            lambda: KernelMeasurement(digest("wrong"), 50),
        )
        self.assertEqual(failed.reason, "correctness_mismatch")
        registry = KernelCapabilityRegistry("hardware-a", "macos-mlx-metal-a")
        registry.record(failed)
        self.assertFalse(registry.is_usable(failed.backend, failed.operator))

        passed = run_kernel_probe(
            self.config(),
            lambda: KernelMeasurement(digest("same"), 100),
            lambda: KernelMeasurement(digest("same"), 100),
        )
        with self.assertRaisesRegex(ValueError, "sticky"):
            registry.record(passed)

    def test_performance_regression_and_probe_error_fail_closed(self) -> None:
        slow = run_kernel_probe(
            self.config(),
            lambda: KernelMeasurement(digest("same"), 100),
            lambda: KernelMeasurement(digest("same"), 200),
        )
        self.assertEqual(slow.reason, "performance_regression")
        self.assertTrue(slow.quarantined)

        def broken() -> KernelMeasurement:
            raise RuntimeError("driver failure details are not persisted")

        errored = run_kernel_probe(
            self.config(),
            lambda: KernelMeasurement(digest("same"), 100),
            broken,
        )
        self.assertEqual(errored.reason, "probe_error")
        self.assertNotIn("driver", str(errored.to_dict()))

    def test_bounded_numeric_tolerance_handles_native_float_error(self) -> None:
        reference = KernelMeasurement(
            digest("reference"), 100, (0.1, 0.2)
        )
        measured = KernelMeasurement(
            digest("measured"), 110, (0.100001, 0.199999)
        )
        passed = run_kernel_probe(
            self.config(maximum_absolute_error=1e-5),
            lambda: reference,
            lambda: measured,
        )
        self.assertTrue(passed.passed)

        failed = run_kernel_probe(
            self.config(maximum_absolute_error=1e-7),
            lambda: reference,
            lambda: measured,
        )
        self.assertEqual(failed.reason, "correctness_mismatch")

    def test_registry_rejects_cross_profile_results_and_stays_bounded(self) -> None:
        result = run_kernel_probe(
            self.config(),
            lambda: KernelMeasurement(digest("same"), 100),
            lambda: KernelMeasurement(digest("same"), 100),
        )
        wrong = KernelCapabilityRegistry("other-hardware", "macos-mlx-metal-a")
        with self.assertRaisesRegex(ValueError, "does not match"):
            wrong.record(result)

        registry = KernelCapabilityRegistry("hardware-a", "macos-mlx-metal-a", capacity=1)
        registry.record(result)
        second = run_kernel_probe(
            self.config(operator="rope"),
            lambda: KernelMeasurement(digest("same"), 100),
            lambda: KernelMeasurement(digest("same"), 100),
        )
        with self.assertRaisesRegex(ValueError, "full"):
            registry.record(second)


if __name__ == "__main__":
    unittest.main()
