import unittest
import plistlib

from vllm_apple.backend_memory import (
    MAX_METRICS_BYTES,
    BackendMemorySample,
    IOGPUMemoryAdapter,
    KVCacheCapacityResolver,
    MemoryMetricsMonitor,
    parse_prometheus_memory_metrics,
)


class BackendMemoryMetricsTests(unittest.TestCase):
    def test_parser_prefers_current_kv_name_and_reads_resident_bytes(self) -> None:
        sample = parse_prometheus_memory_metrics(
            b"""
# TYPE process_resident_memory_bytes gauge
process_resident_memory_bytes 4096
vllm:gpu_cache_usage_perc{model_name="legacy"} 0.9
vllm:kv_cache_usage_perc{engine="0"} 0.25
vllm:kv_cache_usage_perc{engine="1"} 0.5
prompt_content{value="must-not-be-retained"} 7
"""
        )
        self.assertEqual(sample.resident_bytes, 4096)
        self.assertEqual(sample.kv_usage_ratio, 0.5)

    def test_parser_rejects_unbounded_payload_and_ignores_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            parse_prometheus_memory_metrics(b"x" * (MAX_METRICS_BYTES + 1))
        sample = parse_prometheus_memory_metrics(
            b"vllm:kv_cache_usage_perc NaN\nprocess_resident_memory_bytes -1\n"
        )
        self.assertIsNone(sample.resident_bytes)
        self.assertIsNone(sample.kv_usage_ratio)

    def test_verified_cache_config_converts_ratio_to_exact_bytes(self) -> None:
        resolver = KVCacheCapacityResolver(
            "0.28.0", kv_bytes_per_token=1024, model_type="qwen3"
        )
        sample = parse_prometheus_memory_metrics(
            b'vllm:cache_config_info{kv_cache_size_tokens="1000",block_size="16"} 1\n'
            b'vllm:kv_cache_usage_perc 0.25\n',
            resolver,
        )
        self.assertEqual(sample.kv_capacity_bytes, 1_024_000)
        self.assertEqual(sample.kv_used_bytes, 256_000)

    def test_capacity_contract_rejects_unknown_versions_and_ambiguous_engines(self) -> None:
        with self.assertRaises(ValueError):
            KVCacheCapacityResolver("0.23.0", 1024, "llama")
        with self.assertRaises(ValueError):
            KVCacheCapacityResolver("0.29.0", 1024, "llama")
        with self.assertRaises(ValueError):
            KVCacheCapacityResolver("0.28.0", 1024, "qwen3_next")
        resolver = KVCacheCapacityResolver("0.26.1+metal", 1024, "llama")
        sample = parse_prometheus_memory_metrics(
            b'vllm:cache_config_info{kv_cache_size_tokens="100"} 1\n'
            b'vllm:cache_config_info{kv_cache_size_tokens="100"} 1\n'
            b'vllm:kv_cache_usage_perc 0.5\n',
            resolver,
        )
        self.assertIsNone(sample.kv_capacity_bytes)
        self.assertEqual(sample.kv_usage_ratio, 0.5)

    def test_monitor_routes_one_sample_without_retaining_history(self) -> None:
        class Adapter:
            def sample(self) -> BackendMemorySample:
                return BackendMemorySample(8192, 0.75)

        class Sink:
            resident = None
            ratio = None

            def record_backend_resident_memory(self, value: int, *, source: str) -> None:
                self.resident = (value, source)

            def record_kv_cache_ratio(self, value: float, *, source: str) -> None:
                self.ratio = (value, source)

        sink = Sink()
        monitor = MemoryMetricsMonitor(Adapter(), sink)  # type: ignore[arg-type]
        monitor.poll_once()
        self.assertEqual(sink.resident, (8192, "vllm-prometheus"))
        self.assertEqual(sink.ratio, (0.75, "vllm-prometheus"))

    def test_monitor_prefers_exact_kv_bytes_over_ratio_only_sink(self) -> None:
        class Adapter:
            def sample(self) -> BackendMemorySample:
                return BackendMemorySample(
                    resident_bytes=None,
                    kv_usage_ratio=0.5,
                    kv_used_bytes=50,
                    kv_capacity_bytes=100,
                )

        class Sink:
            exact = None
            ratio = None

            def record_kv_cache_memory(
                self, used: int, capacity: int, *, source: str
            ) -> None:
                self.exact = (used, capacity, source)

            def record_kv_cache_ratio(self, value: float, *, source: str) -> None:
                self.ratio = (value, source)

        sink = Sink()
        MemoryMetricsMonitor(Adapter(), sink).poll_once()  # type: ignore[arg-type]
        self.assertEqual(sink.exact, (50, 100, "vllm-prometheus-cache-config-v1"))
        self.assertIsNone(sink.ratio)

    def test_iogpu_plist_parser_is_bounded_and_finds_nested_usage(self) -> None:
        payload = plistlib.dumps(
            [{"PerformanceStatistics": {"In use system memory": 1234}}]
        )
        self.assertEqual(IOGPUMemoryAdapter.parse(payload), 1234)
        self.assertIsNone(IOGPUMemoryAdapter.parse(plistlib.dumps([{"name": "gpu"}])))


if __name__ == "__main__":
    unittest.main()
