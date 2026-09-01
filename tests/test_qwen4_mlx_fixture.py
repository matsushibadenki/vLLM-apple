import unittest

from vllm_apple.qwen4_mlx_fixture import assess_qwen4_mlx_fixture, expected_qwen4_fixture


class Qwen4MLXFixtureTests(unittest.TestCase):
    def payload(self):
        expected = expected_qwen4_fixture()
        return {
            "schema_version": 1,
            "mlx_version": "0.31.0",
            "mlx_lm_version": "0.31.3",
            **expected,
            "fixture_tensor_bytes": 128,
        }

    def test_accepts_matching_bounded_fixture(self) -> None:
        result = assess_qwen4_mlx_fixture(self.payload())
        self.assertTrue(result["passed"])
        self.assertTrue(result["qsa_matches"])
        self.assertFalse(result["stores_tensor_values"])
        self.assertFalse(result["measures_peak_memory"])

    def test_rejects_qsa_mismatch_without_accepting_numeric_match(self) -> None:
        payload = self.payload()
        payload["qsa_selected_tokens"] = [0, 1, 4]
        result = assess_qwen4_mlx_fixture(payload)
        self.assertFalse(result["passed"])
        self.assertFalse(result["qsa_matches"])

    def test_rejects_oversized_allocation_claim(self) -> None:
        payload = self.payload()
        payload["fixture_tensor_bytes"] = 4097
        with self.assertRaisesRegex(ValueError, "tensor size"):
            assess_qwen4_mlx_fixture(payload)


if __name__ == "__main__":
    unittest.main()
