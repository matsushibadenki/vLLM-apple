import hashlib
import threading
import time
import unittest

from vllm_apple.kernel_probe import KernelMeasurement
from vllm_apple.kernel_stress import run_multi_model_command_stress


class KernelStressTests(unittest.TestCase):
    def test_bounded_cross_model_concurrency_and_stable_digests(self):
        barrier = threading.Barrier(2, timeout=1)
        first = set()
        lock = threading.Lock()

        def measure(model_id, iteration):
            with lock:
                initial = model_id not in first
                first.add(model_id)
            if initial:
                barrier.wait()
            digest = hashlib.sha256(model_id.encode()).hexdigest()
            return KernelMeasurement(digest, 1)

        report = run_multi_model_command_stress(
            ("model-a", "model-b"), measure,
            iterations_per_model=3, maximum_concurrency=2,
        )
        self.assertTrue(report.passed)
        self.assertEqual(report.peak_concurrency, 2)
        self.assertEqual(report.command_count, 6)

    def test_failure_and_digest_change_fail_report_without_deadlock(self):
        def measure(model_id, iteration):
            if model_id == "model-a" and iteration == 1:
                raise RuntimeError("injected")
            digest = hashlib.sha256(f"{model_id}-{iteration > 0}".encode()).hexdigest()
            time.sleep(0.001)
            return KernelMeasurement(digest, 1)

        report = run_multi_model_command_stress(
            ("model-a", "model-b"), measure,
            iterations_per_model=2, maximum_concurrency=1,
        )
        self.assertFalse(report.passed)
        self.assertEqual(report.failures, 1)
        self.assertEqual(report.digest_mismatches, 1)
        self.assertEqual(report.peak_concurrency, 1)

    def test_rejects_duplicate_models_and_unbounded_inputs(self):
        def measure(model_id, iteration):
            return KernelMeasurement("a" * 64, 1)

        with self.assertRaises(ValueError):
            run_multi_model_command_stress(("same", "same"), measure)
        with self.assertRaises(ValueError):
            run_multi_model_command_stress(("a", "b"), measure, maximum_concurrency=9)


if __name__ == "__main__":
    unittest.main()
