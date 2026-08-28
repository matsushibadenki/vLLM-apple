import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from tests.schema_validator import validate_instance
from vllm_apple.model_integrity import (
    ModelIntegrityError,
    build_model_integrity_manifest,
    save_model_integrity_manifest,
    verify_model_integrity,
)
from vllm_apple.cli import main


class ModelIntegrityTests(unittest.TestCase):
    def test_cli_create_and_verify_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "model"
            root.mkdir()
            (root / "config.json").write_text("{}")
            manifest = Path(directory) / "trusted.json"
            with redirect_stdout(StringIO()):
                created = main(
                    ["model-integrity-create", str(root), "--output", str(manifest)]
                )
                verified = main(
                    ["model-integrity-verify", str(root), "--manifest", str(manifest)]
                )
            self.assertEqual(created, 0)
            self.assertEqual(verified, 0)

    def test_manifest_streams_files_and_detects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "model"
            root.mkdir()
            (root / "config.json").write_text('{"model_type":"test"}')
            (root / "weights.safetensors").write_bytes(b"weights")
            manifest = build_model_integrity_manifest(root)
            output = Path(directory) / "trusted.json"
            save_model_integrity_manifest(manifest, output)

            verified = verify_model_integrity(root, output)
            self.assertEqual(verified["file_count"], 2)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            schema = json.loads(
                Path("schemas/runtime/model-integrity-manifest-v1.schema.json").read_text()
            )
            validate_instance(verified, schema)

            (root / "weights.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ModelIntegrityError, "verification failed"):
                verify_model_integrity(root, output)

    @unittest.skipUnless(hasattr(os, "symlink"), "symbolic links unavailable")
    def test_model_and_manifest_symlinks_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "model"
            root.mkdir()
            target = root / "weights.safetensors"
            target.write_bytes(b"weights")
            (root / "linked.safetensors").symlink_to(target)
            with self.assertRaisesRegex(ModelIntegrityError, "symbolic link"):
                build_model_integrity_manifest(root)

            (root / "linked.safetensors").unlink()
            manifest = build_model_integrity_manifest(root)
            trusted = Path(directory) / "trusted.json"
            save_model_integrity_manifest(manifest, trusted)
            linked_manifest = Path(directory) / "linked.json"
            linked_manifest.symlink_to(trusted)
            with self.assertRaisesRegex(ModelIntegrityError, "regular file|cannot be read"):
                verify_model_integrity(root, linked_manifest)

    def test_added_or_removed_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "model"
            root.mkdir()
            original = root / "config.json"
            original.write_text("{}")
            manifest_path = Path(directory) / "trusted.json"
            save_model_integrity_manifest(build_model_integrity_manifest(root), manifest_path)
            (root / "tokenizer.json").write_text("{}")
            with self.assertRaises(ModelIntegrityError):
                verify_model_integrity(root, manifest_path)
