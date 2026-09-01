import json
import struct
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from tests import test_qwen4_shard_stager as stager_tests
from vllm_apple.qwen4_adapter_loader import inspect_qwen4_adapter_headers
from vllm_apple.qwen4_shard_stager import stage_qwen4_shards


def _write_safetensors(
    path: Path,
    names: list[str],
    *,
    invalid_offset: bool = False,
    shapes: dict[str, list[int]] | None = None,
) -> None:
    header = {}
    data = bytearray()
    for name in names:
        start = len(data)
        shape = (shapes or {}).get(name, [1])
        tensor_bytes = 2
        for dimension in shape:
            tensor_bytes *= dimension
        data.extend(b"\0" * tensor_bytes)
        end = len(data) + (1 if invalid_offset and not header else 0)
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [start, end]}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


class Qwen4AdapterLoaderTests(unittest.TestCase):
    def source(self, root: Path, *, invalid_offset: bool = False) -> Path:
        source = stager_tests.Qwen4ShardStagerTests().source(root)
        index = json.loads((source / "model.safetensors.index.json").read_text())
        names_by_shard = defaultdict(list)
        for name, shard in index["weight_map"].items():
            names_by_shard[shard].append(name)
        for shard, names in names_by_shard.items():
            _write_safetensors(
                source / shard,
                sorted(names),
                invalid_offset=invalid_offset and shard.endswith("00001-of-00002.safetensors"),
            )
        return source

    def test_inspects_headers_without_reading_tensor_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            output = root / "output"
            stage_qwen4_shards(source, output, maximum_output_bytes=65536)
            result = inspect_qwen4_adapter_headers(
                output, maximum_artifact_bytes=65536
            )
            self.assertTrue(result["passed"])
            self.assertFalse(result["reads_tensor_data"])
            self.assertEqual(result["dtype_counts"], {"BF16": result["tensor_count"]})

    def test_rejects_out_of_range_tensor_offset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root, invalid_offset=True)
            output = root / "output"
            stage_qwen4_shards(source, output, maximum_output_bytes=65536)
            with self.assertRaisesRegex(ValueError, "offset|byte length"):
                inspect_qwen4_adapter_headers(output, maximum_artifact_bytes=65536)

    def test_rejects_duplicate_header_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            shard = source / "model-00001-of-00002.safetensors"
            encoded = b'{"duplicate":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},' \
                b'"duplicate":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}'
            shard.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0\0")
            output = root / "output"
            stage_qwen4_shards(source, output, maximum_output_bytes=65536)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                inspect_qwen4_adapter_headers(output, maximum_artifact_bytes=65536)


if __name__ == "__main__":
    unittest.main()
