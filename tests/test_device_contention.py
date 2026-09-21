import tempfile
import time
import unittest
from pathlib import Path

from tests.test_scheduler import hardware
from vllm_apple.device_contention import (
    ContentionBenchmarkConfig,
    ContentionProfile,
    default_contention_profile_path,
    default_contention_profile_paths,
    install_contention_profile,
    load_contention_profile,
    load_contention_profile_with_fallback,
    promote_contention_profile,
    run_contention_benchmark,
    save_contention_profile,
)
from vllm_apple.device_resources import (
    BandwidthContentionEvidence,
    DeviceResourceRequest,
    UnifiedDeviceResourceLedger,
    contention_profile_id,
)
from vllm_apple.execution import ExecutionBackend
from vllm_apple.profile import build_profile
from vllm_apple.service import RuntimeService


class DeviceContentionTests(unittest.TestCase):
    def setUp(self):
        self.config = ContentionBenchmarkConfig(
            "profile", ExecutionBackend.CPU, ExecutionBackend.NATIVE_MLX, 3
        )

    def test_parallel_benchmark_preserves_outputs_and_measures_improvement(self):
        def first():
            time.sleep(0.003)
            return "a" * 64

        def second():
            time.sleep(0.003)
            return "b" * 64

        evidence = run_contention_benchmark(self.config, first, second)
        self.assertTrue(evidence.outputs_match)
        self.assertTrue(evidence.qualified)
        self.assertLess(evidence.parallel_latency_nanoseconds,
                        evidence.sequential_latency_nanoseconds)

    def test_output_mismatch_fails_closed(self):
        calls = 0

        def changing():
            nonlocal calls
            calls += 1
            return ("a" if calls == 1 else "c") * 64

        evidence = run_contention_benchmark(
            self.config, changing, lambda: "b" * 64
        )
        self.assertFalse(evidence.outputs_match)
        self.assertFalse(evidence.qualified)
        with self.assertRaisesRegex(ValueError, "invalid contention profile"):
            ContentionProfile("profile", (evidence,))

    def test_private_profile_round_trip_tamper_rejection_and_install(self):
        evidence = run_contention_benchmark(
            self.config,
            lambda: (time.sleep(0.002), "a" * 64)[1],
            lambda: (time.sleep(0.002), "b" * 64)[1],
        )
        profile = ContentionProfile("profile", (evidence,))
        ledger = UnifiedDeviceResourceLedger(
            unified_memory_bytes=100, cpu_threads=2, gpu_command_queues=1,
            ane_tasks=1, bandwidth_slots=2, contention_profile_id="profile",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_contention_profile(profile, Path(directory) / "profile.json")
            loaded = load_contention_profile(path, profile_id="profile")
            self.assertEqual(loaded.profile_digest, profile.profile_digest)
            self.assertEqual(install_contention_profile(ledger, loaded), 1)
            first = ledger.reserve(DeviceResourceRequest.for_backend(
                ExecutionBackend.CPU, 10
            ))
            second = ledger.reserve(DeviceResourceRequest.for_backend(
                ExecutionBackend.NATIVE_MLX, 10
            ))
            ledger.release(second.reservation_id)
            ledger.release(first.reservation_id)
            encoded = path.read_text().replace('"sample_count":3', '"sample_count":4')
            path.write_text(encoded)
            path.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                load_contention_profile(path, profile_id="profile")

    def test_default_path_is_profile_specific(self):
        with tempfile.TemporaryDirectory() as directory:
            path = default_contention_profile_path(
                "a" * 64, application_support=Path(directory)
            )
            self.assertEqual(path.parent.name, "device-contention")
            self.assertEqual(path.name, f"{'a' * 64}.json")

    def test_promotion_preserves_valid_current_and_fallback_recovers(self):
        first = BandwidthContentionEvidence(
            "a" * 64, ExecutionBackend.CPU, ExecutionBackend.NATIVE_MLX,
            100, 90, 3, True,
        )
        second = BandwidthContentionEvidence(
            "a" * 64, ExecutionBackend.CPU, ExecutionBackend.COREML_DRAFT,
            100, 80, 3, True,
        )
        with tempfile.TemporaryDirectory() as directory:
            current, last_good = default_contention_profile_paths(
                "a" * 64, application_support=Path(directory)
            )
            promote_contention_profile(
                ContentionProfile("a" * 64, (first,)), current, last_good
            )
            promoted = ContentionProfile("a" * 64, (second,))
            promote_contention_profile(promoted, current, last_good)
            current.write_text("broken")
            current.chmod(0o600)
            restored, source = load_contention_profile_with_fallback(
                current, last_good, profile_id="a" * 64
            )
            self.assertEqual(source, "last_known_good")
            self.assertEqual(restored.evidence, (first,))

    def test_runtime_installs_matching_qualified_profile_at_startup(self):
        device = hardware()
        profile_id = contention_profile_id(
            device.soc, device.os_version, device.architecture
        )
        evidence = BandwidthContentionEvidence(
            profile_id, ExecutionBackend.CPU, ExecutionBackend.NATIVE_MLX,
            100, 90, 3, True,
        )
        service = RuntimeService(
            profile=build_profile(device),
            contention_profile=ContentionProfile(profile_id, (evidence,)),
        )
        first = service.scheduler.device_resources.reserve(
            DeviceResourceRequest.for_backend(ExecutionBackend.CPU, 0)
        )
        second = service.scheduler.device_resources.reserve(
            DeviceResourceRequest.for_backend(ExecutionBackend.NATIVE_MLX, 0)
        )
        service.scheduler.device_resources.release(second.reservation_id)
        service.scheduler.device_resources.release(first.reservation_id)


if __name__ == "__main__":
    unittest.main()
