import hashlib
import tempfile
import unittest
from pathlib import Path

from tests import test_qwen4_adapter_loader as loader_tests
from vllm_apple.qwen4_adapter_contract import build_qwen4_adapter_contract
from vllm_apple.qwen4_conversion_protocol import build_qwen4_conversion_request
from vllm_apple.qwen4_conversion_worker import (
    ConvertedTensorEvidence,
    execute_qwen4_conversion_request,
)
from vllm_apple.qwen4_load_plan import build_qwen4_component_load_plan
from vllm_apple.qwen4_shard_stager import stage_qwen4_shards


class IdentityConverter:
    def __init__(self, *, consume: bool = True) -> None:
        self.consume = consume

    def convert(
        self,
        chunks,
        *,
        source_dtype,
        target_dtype,
        output_shape,
        reserved_bytes,
    ):
        digest = hashlib.sha256()
        output_bytes = 0
        if self.consume:
            for chunk in chunks:
                digest.update(chunk)
                output_bytes += len(chunk)
        return ConvertedTensorEvidence(
            backend="test",
            backend_version="1",
            output_shape=output_shape,
            output_bytes=output_bytes,
            output_digest=digest.hexdigest(),
        )


class Qwen4ConversionWorkerTests(unittest.TestCase):
    def fixture(self, root: Path):
        source = loader_tests.Qwen4AdapterLoaderTests().source(root)
        output = root / "output"
        stage_qwen4_shards(source, output, maximum_output_bytes=65536)
        contract = build_qwen4_adapter_contract(output, maximum_artifact_bytes=65536)
        plan = build_qwen4_component_load_plan(output, maximum_artifact_bytes=65536)
        tensor_name = next(iter(plan_contract_name(output)))
        request = build_qwen4_conversion_request(
            stage_root=output,
            tensor_name=tensor_name,
            contract_id=contract["contract_id"],
            load_plan_id=plan["load_plan_id"],
            target_dtype="BF16",
            maximum_artifact_bytes=65536,
            memory_capacity_bytes=32,
        )
        return request

    def test_rebuilds_plan_and_converts_inside_reservation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            response = execute_qwen4_conversion_request(
                self.fixture(Path(directory)), IdentityConverter()
            )
            self.assertTrue(response["passed"])
            self.assertEqual(response["output_bytes"], 2)
            self.assertEqual(response["peak_reserved_bytes"], 4)

    def test_rejects_request_with_stale_load_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = self.fixture(Path(directory))
            request["load_plan_id"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "rebuilt"):
                execute_qwen4_conversion_request(request, IdentityConverter())

    def test_rejects_converter_that_does_not_consume_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "complete"):
                execute_qwen4_conversion_request(
                    self.fixture(Path(directory)), IdentityConverter(consume=False)
                )


def plan_contract_name(stage: Path):
    import json

    index = json.loads((stage / "model.safetensors.index.json").read_text())
    return index["weight_map"]


if __name__ == "__main__":
    unittest.main()
