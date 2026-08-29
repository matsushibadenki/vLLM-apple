from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vllm_apple.service import RuntimeService
from vllm_apple.vllm_metal_v2_preference import (
    load_native_v2_preference,
    save_native_v2_preference,
)


class NativeV2TuningPreferenceTests(unittest.TestCase):
    def test_private_atomic_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings" / "native-v2.json"
            save_native_v2_preference(False, path)
            self.assertFalse(load_native_v2_preference(path))
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            save_native_v2_preference(True, path)
            self.assertTrue(load_native_v2_preference(path))

    def test_loader_rejects_unknown_fields_and_public_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preference.json"
            path.write_text(json.dumps({"schema_version": 1, "enabled": True, "extra": 1}))
            path.chmod(0o600)
            with self.assertRaises(ValueError):
                load_native_v2_preference(path)
            path.write_text(json.dumps({"schema_version": 1, "enabled": True}))
            path.chmod(0o644)
            with self.assertRaises(ValueError):
                load_native_v2_preference(path)

    def test_runtime_restores_and_persists_control_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings" / "native-v2.json"
            save_native_v2_preference(False, path)
            first = RuntimeService()
            self.assertFalse(first.configure_native_v2_preference(path))
            self.assertEqual(first.native_v2_tuning.snapshot().status, "disabled")
            accepted, _ = first.control_native_v2_tuning("enable")
            self.assertTrue(accepted)

            restarted = RuntimeService()
            self.assertTrue(restarted.configure_native_v2_preference(path))
            self.assertEqual(restarted.native_v2_tuning.snapshot().status, "idle")


if __name__ == "__main__":
    unittest.main()
