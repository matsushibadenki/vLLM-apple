import threading
import unittest
from unittest.mock import patch

from vllm_apple.phase_probe import PhaseProbeConfig, PhaseProbeError, StreamProbeResult
from vllm_apple.phase_profile import PhaseMeasurement
from vllm_apple.text_benchmark import run_text_benchmark


class TextBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.config = PhaseProbeConfig("http://127.0.0.1:19090", "model", "M4", backend="mlx")

    @staticmethod
    def result(quality=True, done=20_000_000):
        return StreamProbeResult(PhaseMeasurement(0, 5_000_000, 10_000_000, 9, 2, 0, done),
                                 quality, 0)

    def test_failures_quality_and_slo_remain_in_denominator(self):
        responses = [self.result(), self.result(False), self.result(done=90_000_000),
                     PhaseProbeError("backend_unavailable", "private failure"),
                     self.result(done=None)]
        with patch("vllm_apple.text_benchmark.measure_stream", side_effect=responses), patch(
            "vllm_apple.text_benchmark.time.monotonic_ns", side_effect=[0, 1_000_000_000]
        ):
            report = run_text_benchmark(self.config, requests=5, e2e_slo_ms=50)
        self.assertEqual(report["completed"], 4)
        self.assertEqual(report["failed"], 1)
        self.assertEqual(report["quality_passed"], 3)
        self.assertEqual(report["slo_quality_passed"], 1)
        self.assertEqual(report["goodput_tokens_per_second"], 2)
        self.assertEqual(report["output_tokens_per_second"], 8)
        self.assertEqual(report["errors"], {"backend_unavailable": 1})
        self.assertEqual(report["phase_profile"]["transport"]["unavailable_sample_count"], 1)
        self.assertTrue(report["e2e_p99_reference_only"])
        self.assertNotIn("private failure", str(report))

    def test_concurrent_requests_have_bounded_workers_and_language_coverage(self):
        barrier = threading.Barrier(3)
        seen = []
        lock = threading.Lock()

        def measure(config, **kwargs):
            with lock:
                seen.append(config.prompt)
            barrier.wait(timeout=5)
            return self.result()

        with patch("vllm_apple.text_benchmark.measure_stream", side_effect=measure):
            report = run_text_benchmark(self.config, requests=9, concurrency=3)
        self.assertEqual(len(seen), 9)
        self.assertEqual(len(set(seen)), 3)
        self.assertEqual(report["completed"], 9)
        self.assertEqual([s["attempted"] for s in report["languages"].values()], [3, 3, 3])

    def test_workload_identity_ignores_endpoint_but_binds_slo_and_concurrency(self):
        from dataclasses import replace
        with patch("vllm_apple.text_benchmark.measure_stream", return_value=self.result()):
            first = run_text_benchmark(self.config, requests=3)
            second = run_text_benchmark(replace(self.config, base_url="http://127.0.0.1:2"),
                                        requests=3)
            changed = run_text_benchmark(self.config, requests=3, concurrency=2)
            slo = run_text_benchmark(self.config, requests=3, e2e_slo_ms=20)
        self.assertEqual(first["workload_sha256"], second["workload_sha256"])
        self.assertNotEqual(first["workload_sha256"], changed["workload_sha256"])
        self.assertNotEqual(first["workload_sha256"], slo["workload_sha256"])

    def test_custom_cases_bind_identity_and_labels(self):
        cases = (("long-a", "prefix 1+1", "2"), ("long-b", "prefix 2+2", "4"))
        with patch("vllm_apple.text_benchmark.measure_stream", return_value=self.result()):
            custom = run_text_benchmark(self.config, requests=4, cases=cases)
            default = run_text_benchmark(self.config, requests=4)
        self.assertEqual(list(custom["languages"]), ["long-a", "long-b"])
        self.assertEqual([item["attempted"] for item in custom["languages"].values()], [2, 2])
        self.assertNotEqual(custom["workload_sha256"], default["workload_sha256"])

    def test_all_failed_has_no_latency(self):
        with patch("vllm_apple.text_benchmark.measure_stream",
                   side_effect=PhaseProbeError("usage_missing", "missing")):
            result = run_text_benchmark(self.config, requests=3)
        self.assertEqual(result["failed"], 3)
        self.assertEqual(result["goodput_tokens_per_second"], 0)
        self.assertIsNone(result["e2e_p99_upper_bound_ms"])

    def test_invalid_bounds(self):
        for kwargs in ({"requests": True}, {"requests": 0}, {"concurrency": 33},
                       {"requests": 1, "concurrency": 2}, {"ttft_slo_ms": float("nan")},
                       {"e2e_slo_ms": float("inf")}, {"e2e_slo_ms": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run_text_benchmark(self.config, **kwargs)
        from dataclasses import replace
        with self.assertRaises(ValueError):
            run_text_benchmark(replace(self.config, timeout_seconds=float("nan")))
        invalid_cases = ((), (("duplicate", "p", "e"), ("duplicate", "p", "e")),
                         (("", "p", "e"),), (("label", "p", ""),),
                         (("x" * 65, "p", "e"),))
        for cases in invalid_cases:
            with self.subTest(cases=cases), self.assertRaises(ValueError):
                run_text_benchmark(self.config, cases=cases)


if __name__ == "__main__":
    unittest.main()
