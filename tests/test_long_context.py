import unittest

from tests.schema_validator import validate_instance
from tests.test_schemas import load_schema
from vllm_apple.long_context import (
    LongContextEvaluationError,
    LongContextEvaluator,
    LongContextObservation,
)


def observation(target: int, **overrides: object) -> LongContextObservation:
    values: dict[str, object] = {
        "target_tokens": target,
        "actual_prompt_tokens": target,
        "retrieval_score": 1.0,
        "ttft_ms": float(target),
        "tpot_ms": 10.0,
        "tokens_per_second": 100.0,
        "load_peak_rss_bytes": 400,
        "steady_state_rss_bytes": 300,
        "state_bytes": target * 2,
    }
    values.update(overrides)
    return LongContextObservation(**values)


class LongContextEvaluatorTests(unittest.TestCase):
    def test_1k_4k_16k_stages_are_deterministic_and_schema_valid(self) -> None:
        evaluator = LongContextEvaluator(
            model_id="model", hardware_fingerprint="hardware", memory_ceiling_bytes=500
        )
        first = evaluator.evaluate((1024, 4096, 16384), observation)
        second = evaluator.evaluate((1024, 4096, 16384), observation)
        self.assertEqual(first, second)
        self.assertTrue(first["passed"])
        self.assertEqual([stage["status"] for stage in first["stages"]], ["passed"] * 3)
        self.assertEqual(first["storage"], {"raw_prompt_count": 0, "raw_output_count": 0})
        validate_instance(
            first, load_schema("runtime/long-context-evaluation-v1.schema.json")
        )

    def test_memory_failure_stops_larger_context_before_measurement(self) -> None:
        measured: list[int] = []

        def measure(target: int) -> LongContextObservation:
            measured.append(target)
            return observation(
                target,
                steady_state_rss_bytes=600 if target == 4096 else 300,
            )

        report = LongContextEvaluator(
            model_id="model", hardware_fingerprint="hardware", memory_ceiling_bytes=500
        ).evaluate((1024, 4096, 16384), measure)
        self.assertEqual(measured, [1024, 4096])
        self.assertEqual(report["stages"][1]["error_code"], "memory_ceiling_exceeded")
        self.assertEqual(report["stages"][2]["status"], "skipped")

    def test_quality_and_token_mismatch_fail_closed(self) -> None:
        evaluator = LongContextEvaluator(
            model_id="model",
            hardware_fingerprint="hardware",
            memory_ceiling_bytes=500,
            minimum_retrieval_score=0.9,
        )
        quality = evaluator.evaluate(
            (1024,), lambda target: observation(target, retrieval_score=0.8)
        )
        self.assertEqual(quality["stages"][0]["error_code"], "retrieval_quality_failed")
        mismatch = evaluator.evaluate(
            (1024,), lambda target: observation(target, actual_prompt_tokens=800)
        )
        self.assertEqual(mismatch["stages"][0]["error_code"], "prompt_token_mismatch")

    def test_adapter_error_is_structured_and_later_stages_are_skipped(self) -> None:
        def fail(target: int) -> LongContextObservation:
            raise LongContextEvaluationError("backend_oom", f"failed at {target}")

        report = LongContextEvaluator(
            model_id="model", hardware_fingerprint="hardware", memory_ceiling_bytes=500
        ).evaluate((1024, 4096), fail)
        self.assertEqual(report["stages"][0]["error_code"], "backend_oom")
        self.assertEqual(report["stages"][1]["status"], "skipped")

    def test_stages_must_be_strictly_increasing_and_bounded(self) -> None:
        evaluator = LongContextEvaluator(
            model_id="model", hardware_fingerprint="hardware", memory_ceiling_bytes=500
        )
        for stages in ((), (4096, 1024), (1024, 1024), tuple(range(1, 18))):
            with self.assertRaises(ValueError):
                evaluator.evaluate(stages, observation)


if __name__ == "__main__":
    unittest.main()
