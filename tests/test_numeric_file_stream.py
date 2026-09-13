import hashlib
import os
import tempfile
import unittest
from pathlib import Path

from vllm_apple.numeric_file_stream import NVFP4FileTileProvider
from vllm_apple.numeric_formats import (
    NumericFormatDescriptor,
    TensorGeometry,
    convert_nvfp4_to_int8,
)
from vllm_apple.numeric_streaming import NumericDoubleBufferStream


class NumericFileStreamTests(unittest.TestCase):
    def fixture(self, root: Path):
        descriptor = NumericFormatDescriptor("nvfp4_e2m1", 34)
        geometry = TensorGeometry((2, 17), 1)
        packed = bytes([0x22] * 17)
        scales = bytes([56, 64, 72, 80])
        tensor = convert_nvfp4_to_int8(
            descriptor, packed, scales, 1, geometry=geometry)
        packed_path = root / "weight.nvfp4"
        scales_path = root / "weight.scales"
        packed_path.write_bytes(packed)
        scales_path.write_bytes(scales)
        packed_path.chmod(0o600)
        scales_path.chmod(0o600)
        provider = NVFP4FileTileProvider(
            packed_path,
            scales_path,
            descriptor,
            geometry=geometry,
            global_scale=1,
            packed_sha256=hashlib.sha256(packed).hexdigest(),
            scales_sha256=hashlib.sha256(scales).hexdigest(),
            scaled_payload_sha256=hashlib.sha256(tensor.payload).hexdigest(),
            source_digest=tensor.source_digest,
            target_digest=tensor.target_digest,
        )
        return provider, tensor, packed_path, scales_path

    def test_incrementally_converts_unaligned_tiles_without_materializing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            provider, tensor, _, _ = self.fixture(Path(directory))
            plan = provider.streaming_plan(7)
            stream = NumericDoubleBufferStream(plan, provider=provider)
            output = bytearray()
            while True:
                lease = stream.acquire_next()
                if lease is None:
                    break
                try:
                    output.extend(lease.view())
                finally:
                    lease.release()
            self.assertEqual(bytes(output), tensor.payload)
            self.assertEqual(stream.snapshot()["next_tile_index"], plan.tile_count)
            stream.close()

    def test_detects_same_size_source_mutation_before_next_tile(self):
        with tempfile.TemporaryDirectory() as directory:
            provider, _, packed_path, _ = self.fixture(Path(directory))
            stream = NumericDoubleBufferStream(provider.streaming_plan(8), provider=provider)
            lease = stream.acquire_next()
            lease.release()
            packed_path.write_bytes(bytes([0x44] * 17))
            packed_path.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "changed"):
                stream.acquire_next()
            stream.close()

    def test_rejects_symlink_permissions_and_content_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider, tensor, packed_path, scales_path = self.fixture(root)
            provider.close()
            link = root / "link.nvfp4"
            os.symlink(packed_path, link)
            arguments = dict(
                descriptor=NumericFormatDescriptor("nvfp4_e2m1", 34),
                geometry=TensorGeometry((2, 17), 1),
                global_scale=1,
                packed_sha256=hashlib.sha256(packed_path.read_bytes()).hexdigest(),
                scales_sha256=hashlib.sha256(scales_path.read_bytes()).hexdigest(),
                scaled_payload_sha256=hashlib.sha256(tensor.payload).hexdigest(),
                source_digest=tensor.source_digest,
                target_digest=tensor.target_digest,
            )
            with self.assertRaises(OSError):
                NVFP4FileTileProvider(link, scales_path, **arguments)
            packed_path.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "unsafe"):
                NVFP4FileTileProvider(packed_path, scales_path, **arguments)
            packed_path.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "binding"):
                NVFP4FileTileProvider(
                    packed_path, scales_path, **(arguments | {"target_digest": "0" * 64})
                )


if __name__ == "__main__":
    unittest.main()
