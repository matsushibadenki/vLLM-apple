import unittest

from vllm_apple.cpu_probe import NativeCPUProbeAdapter
from vllm_apple.execution import ExecutionBackend


class NativeCPUProbeAdapterTests(unittest.TestCase):
    def test_bounded_suite_produces_operator_scoped_passing_evidence(self):
        results = NativeCPUProbeAdapter().probe_suite(
            hardware_fingerprint="m4-test",
            environment_fingerprint="python-test",
            samples=2,
        )
        self.assertEqual(
            {result.operator for result in results},
            {"vector_add", "matmul", "kv_copy"},
        )
        self.assertTrue(all(result.backend is ExecutionBackend.CPU for result in results))
        self.assertTrue(all(result.passed for result in results))
        self.assertTrue(all(result.samples_completed == 2 for result in results))


if __name__ == "__main__":
    unittest.main()
