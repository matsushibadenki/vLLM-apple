import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from vllm_apple.expert_manifest import ExpertManifest
from vllm_apple.expert_residency import ExpertKey
from vllm_apple.mlx_expert_backend import MLXFileExpertBackend


class ExpertManifestTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.file = self.root / "layer-0-expert-0.safetensors"
        self.file.write_bytes(b"fixture")
        self.path = self.root / "expert-manifest.json"
        self.payload = dict(schema_version=1, model_sha256="a" * 64,
                            quantization_bits=4, group_size=32,
                            experts=[dict(layer=0, expert=0, size_bytes=7,
                                          sha256=hashlib.sha256(b"fixture").hexdigest())])
        self.write()

    def write(self):
        self.path.write_text(json.dumps(self.payload))

    def load(self):
        return ExpertManifest.load(self.path, expected_model_sha256="a" * 64)

    def test_identity_settings_and_immutable_entries(self):
        manifest = self.load()
        manifest.verify_file(ExpertKey(0, 0), self.file)
        backend = MLXFileExpertBackend.from_manifest(
            self.root, expected_model_sha256="a" * 64, maximum_file_bytes=4096,
        )
        self.assertEqual((backend.quantization_bits, backend.group_size), (4, 32))
        with self.assertRaises(TypeError):
            manifest.entries[ExpertKey(0, 0)] = (1, "b" * 64)
        with self.assertRaises(ValueError):
            ExpertManifest.load(self.path, expected_model_sha256="b" * 64)

    def test_corruption_and_missing_entry_rejected_before_mlx(self):
        backend = MLXFileExpertBackend.from_manifest(
            self.root, expected_model_sha256="a" * 64, maximum_file_bytes=4096,
        )
        self.file.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            backend.load_expert(ExpertKey(0, 0))
        self.file.write_bytes(b"longer fixture")
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            backend.load_expert(ExpertKey(0, 0))
        with self.assertRaises(ValueError):
            self.load().verify_file(ExpertKey(0, 1), self.file)

    def test_duplicate_fields_entries_and_unsupported_schema(self):
        self.payload["experts"] *= 2
        self.write()
        with self.assertRaises(ValueError):
            self.load()
        self.path.write_text('{"schema_version":1,"schema_version":1}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.load()
        self.payload["schema_version"] = True
        self.write()
        with self.assertRaises(ValueError):
            self.load()

    def test_manifest_size_bound(self):
        self.path.write_bytes(b" " * (4 * 1024 * 1024 + 1))
        with self.assertRaises(ValueError):
            self.load()
