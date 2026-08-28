import json
import unittest
from pathlib import Path

from tests.schema_validator import validate_instance
from vllm_apple.promotion_probe import (
    PromotionProbeConfig,
    PromotionResponse,
    run_serving_promotion_probe,
)


class PromotionProbeTests(unittest.TestCase):
    def test_greedy_sampled_and_streaming_gate_passes(self) -> None:
        calls = []

        def transport(request, stream):
            calls.append((request, stream))
            text = "stable" if request["temperature"] == 0 else "sampled"
            return PromotionResponse(text, stream)

        report = run_serving_promotion_probe(
            PromotionProbeConfig("http://127.0.0.1:8001", "model"),
            transport=transport,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(len(calls), 5)
        self.assertEqual(calls[-1][0]["seed"], 1729)
        self.assertTrue(calls[-1][1])
        schema = json.loads(
            Path("schemas/runtime/serving-promotion-probe-v1.schema.json").read_text()
        )
        validate_instance(report, schema)

    def test_sampling_nondeterminism_fails_promotion(self) -> None:
        sampled_count = 0

        def transport(request, stream):
            nonlocal sampled_count
            if request["temperature"] == 0:
                return PromotionResponse("greedy", False)
            sampled_count += 1
            return PromotionResponse(f"sampled-{sampled_count}", stream)

        report = run_serving_promotion_probe(
            PromotionProbeConfig("http://localhost:8001", "model"),
            transport=transport,
        )
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["sampled_repeat_equal"])
        self.assertFalse(report["checks"]["sampled_stream_equal"])

    def test_stream_requires_done_marker(self) -> None:
        def transport(request, stream):
            return PromotionResponse("same", False)

        report = run_serving_promotion_probe(
            PromotionProbeConfig("http://[::1]:8001", "model"),
            transport=transport,
        )
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["stream_completed"])

    def test_remote_backend_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PromotionProbeConfig("https://example.com", "model")


if __name__ == "__main__":
    unittest.main()
