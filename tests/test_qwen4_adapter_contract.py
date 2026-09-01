import tempfile
import unittest
from pathlib import Path

from tests import test_qwen4_shard_stager as stager_tests
from vllm_apple.qwen4_adapter_contract import build_qwen4_adapter_contract
from vllm_apple.qwen4_shard_stager import stage_qwen4_shards


class Qwen4AdapterContractTests(unittest.TestCase):
    def test_contract_requires_and_binds_a_verified_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = stager_tests.Qwen4ShardStagerTests().source(root)
            output = root / "output"
            stage_qwen4_shards(source, output, maximum_output_bytes=4096)
            contract = build_qwen4_adapter_contract(
                output, maximum_artifact_bytes=4096
            )
            self.assertTrue(contract["stage_verified"])
            self.assertFalse(contract["loads_tensor_data"])
            self.assertFalse(contract["allocates_model_or_metal"])
            self.assertEqual(contract["shard_count"], 2)
            self.assertEqual(len(contract["shard_schedule"]), 2)
            self.assertEqual(len(contract["contract_id"]), 64)

    def test_contract_rejects_stage_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = stager_tests.Qwen4ShardStagerTests().source(root)
            output = root / "output"
            stage_qwen4_shards(source, output, maximum_output_bytes=4096)
            (output / "model-00001-of-00002.safetensors").write_bytes(b"mutated")
            with self.assertRaisesRegex(ValueError, "digest"):
                build_qwen4_adapter_contract(output, maximum_artifact_bytes=4096)


if __name__ == "__main__":
    unittest.main()
