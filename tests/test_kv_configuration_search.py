import unittest

from vllm_apple.kv_configuration_search import (
    KVConfiguration,
    KVConfigurationMeasurement,
    KVSearchObjective,
    search_kv_configuration,
)


def measurement(precision, context, batch, latency, peak, digest="d" * 64, error=0.0):
    return KVConfigurationMeasurement(
        KVConfiguration(precision, context, batch, 16), 1, peak,
        error, error, 1.0 - error, (latency, latency + 1, latency - 1), digest,
    )


class KVConfigurationSearchTests(unittest.TestCase):
    def test_capacity_and_latency_objectives_choose_different_candidates(self):
        candidates = (
            measurement("fp16", 4096, 1, 100, 8192),
            measurement("int8", 8192, 2, 200, 32768),
        )
        common = dict(
            maximum_peak_memory_bytes=65536, maximum_absolute_error=.1,
            maximum_rmse=.1, minimum_cosine_similarity=.9,
            baseline_output_digest="d" * 64,
        )
        capacity = search_kv_configuration(
            candidates, objective=KVSearchObjective.CAPACITY, **common
        )
        latency = search_kv_configuration(
            candidates, objective=KVSearchObjective.LATENCY, **common
        )
        self.assertEqual(capacity.selected.precision, "int8")
        self.assertEqual(latency.selected.precision, "fp16")

    def test_quality_memory_and_digest_gates_fail_closed(self):
        candidates = (measurement("int8", 4096, 1, 100, 8192, error=.2),)
        with self.assertRaisesRegex(ValueError, "no KV"):
            search_kv_configuration(
                candidates, objective=KVSearchObjective.BALANCED,
                maximum_peak_memory_bytes=8192, maximum_absolute_error=.1,
                maximum_rmse=.1, minimum_cosine_similarity=.9,
                baseline_output_digest="d" * 64,
            )


if __name__ == "__main__":
    unittest.main()
