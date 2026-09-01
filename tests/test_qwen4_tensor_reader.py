import json
import struct
import tempfile
import unittest
from pathlib import Path

from tests import test_qwen4_adapter_loader as loader_tests
from tests import test_qwen4_load_plan as load_plan_tests
from vllm_apple.qwen4_shard_stager import stage_qwen4_shards
from vllm_apple.qwen4_tensor_reader import Qwen4TensorReader


class Qwen4TensorReaderTests(unittest.TestCase):
    def test_streams_an_active_tensor_in_bounded_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = loader_tests.Qwen4AdapterLoaderTests().source(root)
            output = root / "output"
            stage_qwen4_shards(source, output, maximum_output_bytes=65536)
            reader = Qwen4TensorReader(
                output, maximum_artifact_bytes=65536, maximum_chunk_bytes=1
            )
            index = json.loads((output / "model.safetensors.index.json").read_text())
            tensor_name = next(iter(index["weight_map"]))
            chunks = list(reader.iter_tensor_chunks(tensor_name))
            self.assertEqual(chunks, [b"\0", b"\0"])
            self.assertTrue(reader.descriptor(tensor_name)["active"])

    def test_rechecks_shard_digest_before_streaming(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = loader_tests.Qwen4AdapterLoaderTests().source(root)
            output = root / "output"
            stage_qwen4_shards(source, output, maximum_output_bytes=65536)
            reader = Qwen4TensorReader(output, maximum_artifact_bytes=65536)
            index = json.loads((output / "model.safetensors.index.json").read_text())
            tensor_name, shard_name = next(iter(index["weight_map"].items()))
            shard = output / shard_name
            content = bytearray(shard.read_bytes())
            content[-1] ^= 1
            shard.write_bytes(content)
            with self.assertRaisesRegex(ValueError, "digest"):
                list(reader.iter_tensor_chunks(tensor_name))

    def test_rejects_tensor_disabled_by_requested_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = loader_tests.Qwen4AdapterLoaderTests().source(root)
            output = root / "output"
            stage_qwen4_shards(source, output, maximum_output_bytes=65536)
            reader = Qwen4TensorReader(output, maximum_artifact_bytes=65536)
            tensor_name = next(iter(reader._catalog["tensors"]))
            reader._catalog["tensors"][tensor_name]["active"] = False
            with self.assertRaisesRegex(ValueError, "disabled"):
                list(reader.iter_tensor_chunks(tensor_name))

    def test_streams_only_requested_packed_expert_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = load_plan_tests.packed_expert_source(root)
            index = json.loads((source / "model.safetensors.index.json").read_text())
            tensor_name = next(name for name in index["weight_map"] if ".experts." in name)
            shard = source / index["weight_map"][tensor_name]
            content = bytearray(shard.read_bytes())
            header_bytes = struct.unpack("<Q", content[:8])[0]
            header = json.loads(content[8 : 8 + header_bytes])
            start = 8 + header_bytes + header[tensor_name]["data_offsets"][0]
            content[start : start + 4] = b"\1\1\2\2"
            shard.write_bytes(content)
            output = root / "output"
            stage_qwen4_shards(source, output, maximum_output_bytes=65536)
            reader = Qwen4TensorReader(
                output, maximum_artifact_bytes=65536, maximum_chunk_bytes=1
            )
            self.assertEqual(
                list(reader.iter_tensor_axis0_slice(tensor_name, start=1, count=1)),
                [b"\2", b"\2"],
            )


if __name__ == "__main__":
    unittest.main()
