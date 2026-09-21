import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from vllm_apple.numeric_artifact import (
    MAX_NUMERIC_SOURCE_BYTES,
    NumericArtifactReader,
    read_numeric_source_file,
    write_nvfp4_file_numeric_artifact,
    write_nvfp4_numeric_artifact,
)
from vllm_apple.numeric_formats import NumericFormatDescriptor, TensorGeometry
from vllm_apple.numeric_precision import NumericPrecisionPolicy


class NumericArtifactTests(unittest.TestCase):
    def fixture(self, root: Path):
        root.chmod(0o700)
        path = root / "weight.json"
        created = write_nvfp4_numeric_artifact(
            path, NumericFormatDescriptor("nvfp4_e2m1", 3), bytes([0xA2, 2]),
            bytes([56, 64, 72]), 1, geometry=TensorGeometry((3, 1), 1),
            target_dtype="F16", precision_policy=NumericPrecisionPolicy())
        return path, created

    def test_roundtrip_rebuilds_and_binds_source_target_and_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, created = self.fixture(root)
            loaded = NumericArtifactReader(root).read(path.name, created.artifact_digest)
            self.assertEqual(loaded, created)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_roundtrip_preserves_two_axis_scale_and_nibble_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            descriptor = NumericFormatDescriptor(
                "nvfp4_e2m1", 289, packing="high_nibble_first")
            geometry = TensorGeometry((17, 17), (0, 1))
            packed = bytes([0x22] * 144 + [0x20])
            scales = bytes([56, 64, 72, 80])
            path = root / "two-axis.json"
            created = write_nvfp4_numeric_artifact(
                path, descriptor, packed, scales, 1, geometry=geometry,
                target_dtype="F16", precision_policy=NumericPrecisionPolicy())
            loaded = NumericArtifactReader(root).read(path.name, created.artifact_digest)
            self.assertEqual(loaded.tensor.plan.geometry.scale_axes, (0, 1))
            self.assertEqual(loaded.tensor.plan.format_plan.source.packing,
                             "high_nibble_first")
            self.assertEqual(loaded, created)

    def test_claim_removes_inbox_and_consume_removes_quarantine(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, created = self.fixture(root)
            reader = NumericArtifactReader(root)
            claimed = reader.claim(path.name, created.artifact_digest)
            self.assertFalse(path.exists())
            self.assertTrue(claimed.quarantined)
            quarantined = root / "quarantine" / claimed.artifact_name
            self.assertTrue(quarantined.exists())
            reader.consume(claimed)
            self.assertFalse(quarantined.exists())
            with self.assertRaises((FileNotFoundError, ValueError)):
                reader.consume(claimed)

    def test_file_bundle_claim_stream_and_consume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            name, digest = write_nvfp4_file_numeric_artifact(
                root / "weight.json",
                NumericFormatDescriptor("nvfp4_e2m1", 3),
                bytes([0xA2, 2]),
                bytes([56, 64, 72]),
                1,
                geometry=TensorGeometry((3, 1), 1),
                target_dtype="F16",
                precision_policy=NumericPrecisionPolicy(),
            )
            reader = NumericArtifactReader(root)
            claimed = reader.claim(name, digest)
            self.assertFalse((root / "weight.json").exists())
            self.assertFalse((root / "weight.packed").exists())
            self.assertFalse((root / "weight.scales").exists())
            plan = claimed.provider.streaming_plan(1)
            from vllm_apple.numeric_streaming import NumericDoubleBufferStream

            stream = NumericDoubleBufferStream(plan, provider=claimed.provider)
            output = bytearray()
            while (lease := stream.acquire_next()) is not None:
                try:
                    output.extend(lease.read())
                finally:
                    lease.release()
            self.assertEqual(bytes(output), bytes([2, 254, 2]))
            stream.close()
            reader.consume(claimed)
            self.assertEqual(list((root / "quarantine").iterdir()), [])

    def test_digest_mismatch_is_quarantined(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, created = self.fixture(root)
            reader = NumericArtifactReader(root)
            with self.assertRaisesRegex(ValueError, "quarantined"):
                reader.claim(path.name, "0" * 64)
            self.assertFalse(path.exists())
            quarantine = root / "quarantine"
            files = list(quarantine.iterdir())
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].name.startswith(created.artifact_digest))

    def test_digest_schema_and_path_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, created = self.fixture(root)
            reader = NumericArtifactReader(root)
            with self.assertRaises(ValueError):
                reader.read("../weight.json", created.artifact_digest)
            with self.assertRaises(ValueError):
                reader.read(path.name, "0" * 64)
            payload = json.loads(path.read_bytes())
            payload["target_digest"] = "0" * 64
            raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "binding"):
                reader.read(path.name, hashlib.sha256(raw).hexdigest())

    def test_rejects_duplicate_keys_permissions_and_unsafe_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            duplicate = root / "duplicate.json"
            duplicate.write_bytes(b'{"schema_version":1,"schema_version":1}')
            duplicate.chmod(0o600)
            digest = hashlib.sha256(duplicate.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "duplicate"):
                NumericArtifactReader(root).read(duplicate.name, digest)
            duplicate.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "unsafe"):
                NumericArtifactReader(root).read(duplicate.name, digest)
            with self.assertRaises(ValueError):
                write_nvfp4_numeric_artifact(
                    root / ".hidden.json", NumericFormatDescriptor("nvfp4_e2m1", 1),
                    bytes([0]), bytes([56]), 1, geometry=None, target_dtype="F16",
                    precision_policy=NumericPrecisionPolicy())
            root.chmod(0o755)
            with self.assertRaises(ValueError):
                NumericArtifactReader(root)

    def test_rejects_symlink_without_following_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, created = self.fixture(root)
            link = root / "link.json"
            os.symlink(path, link)
            with self.assertRaises(OSError):
                NumericArtifactReader(root).read(link.name, created.artifact_digest)

    def test_bounded_source_reader_rejects_symlink_and_oversize(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            source.write_bytes(b"value")
            self.assertEqual(read_numeric_source_file(source), b"value")
            link = root / "link.bin"
            os.symlink(source, link)
            with self.assertRaises(OSError):
                read_numeric_source_file(link)
            source.write_bytes(b"x" * (MAX_NUMERIC_SOURCE_BYTES + 1))
            with self.assertRaisesRegex(ValueError, "bounded"):
                read_numeric_source_file(source)

    def test_recover_stale_orphans_without_touching_active_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            stale = root / "stale.packed"
            recent = root / "recent.scales"
            active = root / "active.packed"
            manifest = root / "active.json"
            for path in (stale, recent, active, manifest):
                path.write_bytes(b"x")
                path.chmod(0o600)
            os.utime(stale, (1, 1))
            os.utime(active, (1, 1))
            reader = NumericArtifactReader(root)
            self.assertEqual(reader.recover_orphan_bundles(), 1)
            self.assertFalse(stale.exists())
            self.assertTrue(recent.exists())
            self.assertTrue(active.exists())
            quarantined = list((root / "quarantine").iterdir())
            self.assertEqual(len(quarantined), 1)
            self.assertTrue(quarantined[0].name.endswith(".packed"))


if __name__ == "__main__":
    unittest.main()
