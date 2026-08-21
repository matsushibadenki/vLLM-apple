import os
import threading
import unittest

from vllm_apple.api import create_server
from vllm_apple.service import RuntimeService
from vllm_apple.soak import BoundedMetrics, SoakConfig, run_soak


class SoakRunnerTests(unittest.TestCase):
    def test_metrics_storage_stays_bounded(self) -> None:
        metrics = BoundedMetrics()
        for index in range(100_000):
            metrics.record(index % 12_000, f"error-{index}" if index % 10 == 0 else None)
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["requests"], 100_000)
        self.assertEqual(snapshot["storage"]["latency_buckets"], 12)
        self.assertLessEqual(snapshot["storage"]["error_keys"], 17)

    def test_health_soak_reports_throughput_and_rss(self) -> None:
        server = create_server("127.0.0.1", 0, RuntimeService())
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        try:
            result = run_soak(
                SoakConfig(
                    base_url=f"http://127.0.0.1:{server.server_port}",
                    duration_seconds=0.2,
                    warmup_seconds=0,
                    concurrency=2,
                    request_timeout_seconds=2,
                    target_pid=os.getpid(),
                    max_rss_growth_bytes=64 * 1024 * 1024,
                )
            )
            self.assertTrue(result["passed"])
            self.assertGreater(result["requests"], 0)
            self.assertGreater(result["requests_per_second"], 0)
            self.assertIsNotNone(result["rss"]["baseline_bytes"])
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2)

    def test_remote_target_requires_explicit_opt_in(self) -> None:
        with self.assertRaises(ValueError):
            SoakConfig(base_url="https://example.com")


if __name__ == "__main__":
    unittest.main()
