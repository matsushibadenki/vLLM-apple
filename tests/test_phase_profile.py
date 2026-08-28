import threading
import unittest

from tests.schema_validator import validate_instance
from tests.test_schemas import load_schema
from vllm_apple.phase_profile import ExecutionPhaseProfiler, PhaseMeasurement


class ExecutionPhaseProfilerTests(unittest.TestCase):
    def test_phase_metrics_are_separate_deterministic_and_schema_valid(self) -> None:
        profiler = ExecutionPhaseProfiler("hardware", "model", "vllm_metal")
        profiler.record(PhaseMeasurement(0, 100_000_000, 300_000_000, 10, 5, 1024))
        profiler.record(PhaseMeasurement(0, 200_000_000, 600_000_000, 20, 5, 2048))
        first = profiler.snapshot()
        second = profiler.snapshot()
        self.assertEqual(first, second)
        self.assertEqual(first["prefill"]["ttft"]["mean_ms"], 150.0)
        self.assertEqual(first["decode"]["tpot"]["mean_ms"], 75.0)
        self.assertEqual(first["decode"]["tokens_per_second"], 13.333)
        self.assertEqual(first["peak_memory_bytes"], 2048)
        validate_instance(
            first, load_schema("runtime/execution-phase-profile-v1.schema.json")
        )

    def test_storage_stays_constant_under_concurrent_load(self) -> None:
        profiler = ExecutionPhaseProfiler("hardware", "model", "cpu")

        def record_many() -> None:
            for _ in range(10_000):
                profiler.record(PhaseMeasurement(0, 1, 3, 1, 2, 128))

        workers = [threading.Thread(target=record_many) for _ in range(4)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        snapshot = profiler.snapshot()
        self.assertEqual(snapshot["sample_count"], 40_000)
        self.assertEqual(snapshot["storage"]["latency_bucket_count"], 26)
        self.assertEqual(snapshot["storage"]["raw_sample_count"], 0)

    def test_invalid_measurement_is_rejected(self) -> None:
        for values in (
            (2, 1, 3, 1, 1, 0),
            (0, 2, 1, 1, 1, 0),
            (0, 1, 2, -1, 1, 0),
            (0, 1, 2, 1, 0, 0),
            (0, 1, 2, 1, 1, -1),
        ):
            with self.assertRaises(ValueError):
                PhaseMeasurement(*values)


if __name__ == "__main__":
    unittest.main()
