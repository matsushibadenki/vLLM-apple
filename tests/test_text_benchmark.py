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
        self.assertEqual(report["route"], "mlx")
        self.assertEqual(report["phase_profile"]["transport"]["unavailable_sample_count"], 1)
        self.assertTrue(report["e2e_p99_reference_only"])
        self.assertNotIn("private failure", str(report))
        diagnostics = report["failure_diagnostics"]
        self.assertEqual(diagnostics["observed"], 3)
        self.assertEqual([s["reasons"] for s in diagnostics["samples"]],
                         [["quality_failed"], ["e2e_slo_exceeded"], ["stream_done_missing"]])
        self.assertNotIn("prompt", diagnostics["samples"][0])
        distributions = report["latency_distributions"]
        self.assertEqual(distributions["ttft"]["sample_count"], 4)
        self.assertEqual(distributions["e2e"]["sample_count"], 3)
        self.assertEqual(distributions["e2e"]["unavailable_attempt_count"], 2)
        self.assertEqual(sum(distributions["e2e"]["bucket_counts"]), 3)
        self.assertEqual(distributions["e2e"]["max_ms"], 90)

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
        self.assertIsNone(result["latency_distributions"]["e2e"]["mean_ms"])
        self.assertEqual(result["latency_distributions"]["ttft"]["unavailable_attempt_count"], 3)

    def test_cache_usage_coverage_and_validation(self):
        from dataclasses import replace

        from vllm_apple.phase_probe import _cached_prompt_tokens
        for value in (True, -1, 10, "2", None):
            self.assertIsNone(_cached_prompt_tokens(
                {"prompt_tokens_details": {"cached_tokens": value}}, 9))
        self.assertEqual(_cached_prompt_tokens(
            {"prompt_tokens_details": {"cached_tokens": 0}}, 9), 0)
        with patch("vllm_apple.text_benchmark.measure_stream", side_effect=[
            replace(self.result(), cached_prompt_tokens=6),
            replace(self.result(), cached_prompt_tokens=0), self.result(),
        ]):
            report = run_text_benchmark(self.config, requests=3)
        usage = report["prompt_cache_usage"]
        self.assertEqual(usage["observed_requests"], 2)
        self.assertEqual(usage["unavailable_attempts"], 1)
        self.assertEqual(usage["requests_with_reuse"], 1)
        self.assertEqual(usage["observed_token_reuse_ratio"], 6 / 18)

    def test_verified_identities_are_explicit_and_paired(self):
        with patch("vllm_apple.text_benchmark.measure_stream", return_value=self.result()):
            result = run_text_benchmark(
                self.config, requests=1, artifact_identity_sha256="a" * 64,
                backend_build_sha256="b" * 64,
            )
        self.assertTrue(result["artifact_identity_verified"])
        self.assertEqual(result["artifact_identity_sha256"], "a" * 64)
        with self.assertRaisesRegex(ValueError, "provided together"):
            run_text_benchmark(
                self.config, requests=1, artifact_identity_sha256="a" * 64
            )

    def test_warmup_is_excluded_from_measured_counts_and_binds_workload(self):
        with patch("vllm_apple.text_benchmark.measure_stream", return_value=self.result()) as measure:
            warmed = run_text_benchmark(self.config, requests=3, warmup_requests=3)
            cold = run_text_benchmark(self.config, requests=3)
        self.assertEqual(measure.call_count, 9)
        self.assertEqual(warmed["warmup"], {
            "attempted": 3, "completed": 3, "quality_passed": 3, "errors": {},
        })
        self.assertEqual(warmed["requests"], 3)
        self.assertEqual(warmed["cache_policy"], "backend_managed_conditioned")
        self.assertNotEqual(warmed["workload_sha256"], cold["workload_sha256"])
        self.assertEqual(warmed["latency_distributions"]["e2e"]["sample_count"], 3)

    def test_latency_distribution_boundaries_and_overflow(self):
        from vllm_apple.phase_profile import LATENCY_BUCKETS_NS
        boundary = LATENCY_BUCKETS_NS[0]
        # Use valid timestamps even for the smallest histogram boundary.
        responses = [StreamProbeResult(
            PhaseMeasurement(0, 0, 0, 1, 1, 0, boundary), True, 0,
        ), self.result(done=LATENCY_BUCKETS_NS[-1] + 1)]
        with patch("vllm_apple.text_benchmark.measure_stream", side_effect=responses):
            result = run_text_benchmark(self.config, requests=2)
        distribution = result["latency_distributions"]["e2e"]
        self.assertEqual(distribution["bucket_counts"][0], 1)
        self.assertEqual(distribution["bucket_counts"][-1], 1)
        self.assertIsNone(distribution["bucket_upper_bounds_ms"][-1])
        self.assertEqual(len(distribution["bucket_counts"]), len(LATENCY_BUCKETS_NS) + 1)

    def test_context_probes_are_outside_measured_window(self):
        events = []

        def observe(pid):
            events.append("observe")
            return {"target_process_age_seconds": 12}

        def measure(*args, **kwargs):
            events.append("request")
            return self.result()

        def clock():
            events.append("clock")
            return 0 if events.count("clock") == 1 else 1_000_000_000

        with patch("vllm_apple.text_benchmark.observe_benchmark_context", side_effect=observe), \
                patch("vllm_apple.text_benchmark.measure_stream", side_effect=measure), \
                patch("vllm_apple.text_benchmark.time.monotonic_ns", side_effect=clock):
            result = run_text_benchmark(
                self.config, requests=1, warmup_requests=1, collect_operating_context=True,
            )
        self.assertEqual(events, ["observe", "request", "observe", "clock",
                                  "request", "clock", "observe"])
        self.assertEqual(result["elapsed_seconds"], 1)
        self.assertEqual(len(result["operating_context"]), 3)

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

    def test_backend_observations_are_outside_timer(self):
        events = []
        def observe(config):
            events.append('observe')
            return {'status': 'unavailable'}
        def clock():
            events.append('clock')
            return 0 if events.count('clock') == 1 else 1_000_000_000
        with patch('vllm_apple.text_benchmark.observe_backend_memory', side_effect=observe), \
                patch('vllm_apple.text_benchmark.time.monotonic_ns', side_effect=clock), \
                patch('vllm_apple.text_benchmark.measure_stream', return_value=self.result()):
            result = run_text_benchmark(self.config, requests=1, collect_backend_memory=True)
        self.assertEqual(events, ['observe', 'clock', 'clock', 'observe'])
        self.assertEqual(len(result['backend_memory_observations']), 2)


if __name__ == "__main__":
    unittest.main()
