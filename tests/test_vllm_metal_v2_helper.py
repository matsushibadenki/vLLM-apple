import json
import unittest

from vllm_apple.vllm_metal_v2_adapter import build_v2_measurement_request
from vllm_apple.vllm_metal_v2_helper import (
    inspect_native_measurement_capability,
    invoke_native_measurement,
)
from vllm_apple.vllm_metal_v2_tuning import (
    V2DispatchConfiguration,
    V2PagedAttentionFamily,
    V2PagedAttentionShape,
)


class FakeOps:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def vllm_apple_measure_paged_attention_v2(self, request):
        self.requests.append(json.loads(request))
        return json.dumps(self.response, separators=(",", ":"))


class VLLMMetalV2HelperTests(unittest.TestCase):
    def request(self):
        shape = V2PagedAttentionShape(8192, 1, 1, 32, 8, 128, 16, 16)
        configuration = V2DispatchConfiguration(
            V2PagedAttentionFamily.SPLIT_KV,
            256,
            partition_size=512,
        )
        return build_v2_measurement_request(shape, configuration)

    def test_calls_native_extension_with_canonical_abi(self) -> None:
        ops = FakeOps(
            {
                "abi_version": 1,
                "passed": True,
                "latency_nanoseconds": 9876,
                "output_digest": "c" * 64,
            }
        )
        response = invoke_native_measurement(self.request(), get_ops=lambda: ops)
        self.assertEqual(response["latency_nanoseconds"], 9876)
        self.assertEqual(ops.requests[0]["configuration"]["family"], "split_kv")

    def test_fails_closed_when_native_symbol_is_missing(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "does not expose"):
            invoke_native_measurement(self.request(), get_ops=lambda: object())

    def test_rejects_invalid_native_response(self) -> None:
        ops = FakeOps(
            {
                "abi_version": 1,
                "passed": True,
                "latency_nanoseconds": 0,
                "output_digest": "d" * 64,
            }
        )
        with self.assertRaisesRegex(RuntimeError, "invalid JSON"):
            invoke_native_measurement(self.request(), get_ops=lambda: ops)

    def test_capability_reports_symbol_without_executing_it(self) -> None:
        compatible = inspect_native_measurement_capability(get_ops=lambda: FakeOps({}))
        self.assertTrue(compatible["compatible"])
        missing = inspect_native_measurement_capability(get_ops=lambda: object())
        self.assertFalse(missing["compatible"])
        self.assertEqual(missing["issue"], "measurement_symbol_missing")


if __name__ == "__main__":
    unittest.main()
