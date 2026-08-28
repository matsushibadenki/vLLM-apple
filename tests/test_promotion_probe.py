import json
import io
import unittest
from pathlib import Path

from tests.schema_validator import validate_instance
from vllm_apple.promotion_probe import (
    PromotionProbeConfig,
    PromotionResponse,
    run_serving_promotion_probe,
)
from vllm_apple.promotion_probe import _read_stream


class PromotionProbeTests(unittest.TestCase):
    def test_stream_reader_ignores_bounded_usage_only_chunk(self) -> None:
        response = io.BytesIO(
            b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
            b'data: {"choices":[],"usage":{"completion_tokens":1}}\n\n'
            b'data: [DONE]\n\n'
        )
        parsed = _read_stream(response)
        self.assertEqual(parsed.text, "ok")
        self.assertTrue(parsed.stream_completed)

    def test_greedy_only_backend_still_checks_repeat_and_stream_equivalence(self) -> None:
        calls = []

        def transport(request, stream):
            calls.append((request, stream))
            return PromotionResponse("stable", stream)

        report = run_serving_promotion_probe(
            PromotionProbeConfig(
                "http://127.0.0.1:8001",
                "model",
                supports_seeded_sampling=False,
            ),
            transport=transport,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["determinism_mode"], "greedy_only")
        self.assertTrue(all(call[0]["temperature"] == 0 for call in calls))

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
