import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from tests.schema_validator import validate_instance
from vllm_apple.vllm_metal_v2_adapter import (
    V2MeasurementAdapterError,
    VLLMMetalV2MeasurementAdapter,
    build_v2_measurement_request,
    parse_v2_capability_response,
)
from vllm_apple.vllm_metal_v2_tuning import (
    V2DispatchConfiguration,
    V2PagedAttentionFamily,
    V2PagedAttentionShape,
)


class VLLMMetalV2MeasurementAdapterTests(unittest.TestCase):
    def test_capability_response_is_strict(self) -> None:
        parsed = parse_v2_capability_response(
            {
                "abi_version": 1,
                "compatible": False,
                "symbol": "vllm_apple_measure_paged_attention_v2",
                "issue": "measurement_symbol_missing",
            }
        )
        self.assertFalse(parsed["compatible"])
        with self.assertRaises(ValueError):
            parse_v2_capability_response({"compatible": True})

    def shape(self):
        return V2PagedAttentionShape(4096, 1, 1, 32, 8, 128, 16, 16)

    def helper(self, root: Path, body: str) -> Path:
        path = root / "measure"
        path.write_text("#!/bin/sh\n" + body)
        path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        return path

    def test_request_is_complete_and_versioned(self) -> None:
        configuration = V2DispatchConfiguration(V2PagedAttentionFamily.PER_TOKEN, 256)
        request = build_v2_measurement_request(self.shape(), configuration)
        self.assertEqual(request["abi_version"], 1)
        self.assertEqual(request["operation"], "measure_paged_attention_v2")
        self.assertEqual(request["configuration"]["family"], "per_token")
        self.assertEqual(request["shape"]["context_tokens"], 4096)
        schema = json.loads(
            Path("schemas/runtime/vllm-metal-v2-measurement-abi-v1.schema.json").read_text()
        )
        validate_instance(request, schema)

    def test_executes_isolated_helper_and_parses_measurement(self) -> None:
        response = json.dumps(
            {
                "abi_version": 1,
                "passed": True,
                "latency_nanoseconds": 1234,
                "output_digest": "a" * 64,
            },
            separators=(",", ":"),
        )
        with tempfile.TemporaryDirectory() as directory:
            helper = self.helper(Path(directory), f"read request\nprintf '%s' '{response}'\n")
            adapter = VLLMMetalV2MeasurementAdapter(helper)
            measured = adapter.measure(
                self.shape(),
                V2DispatchConfiguration(V2PagedAttentionFamily.PER_TOKEN, 256),
            )
        self.assertEqual(measured, (True, 1234, "a" * 64))
        schema = json.loads(
            Path("schemas/runtime/vllm-metal-v2-measurement-result-v1.schema.json").read_text()
        )
        validate_instance(json.loads(response), schema)

    def test_rejects_oversized_or_version_mismatched_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            oversized = self.helper(root, "printf '%02048d' 0\n")
            adapter = VLLMMetalV2MeasurementAdapter(oversized, maximum_output_bytes=1024)
            with self.assertRaises(V2MeasurementAdapterError):
                adapter.measure(
                    self.shape(),
                    V2DispatchConfiguration(V2PagedAttentionFamily.PER_TOKEN, 256),
                )
            mismatched = self.helper(
                root,
                "printf '%s' '{\"abi_version\":2,\"passed\":true,"
                "\"latency_nanoseconds\":1,\"output_digest\":\""
                + "b" * 64
                + "\"}'\n",
            )
            with self.assertRaises(V2MeasurementAdapterError):
                VLLMMetalV2MeasurementAdapter(mismatched).measure(
                    self.shape(),
                    V2DispatchConfiguration(V2PagedAttentionFamily.PER_TOKEN, 256),
                )

    def test_rejects_non_executable_helper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "measure"
            path.write_text("unused")
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
            with self.assertRaises(ValueError):
                VLLMMetalV2MeasurementAdapter(path)


if __name__ == "__main__":
    unittest.main()
