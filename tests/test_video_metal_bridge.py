import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vllm_apple.video_metal_bridge import NativeVideoMetalBridge


class VideoMetalBridgeTests(unittest.TestCase):
    def fixture(self, root):
        executable = root / "bridge"
        executable.write_bytes(b"executable")
        source = root / "video.mp4"
        source.write_bytes(b"video")
        return executable, source

    def test_accepts_consistent_native_zero_copy_report(self):
        with tempfile.TemporaryDirectory() as directory:
            executable, source = self.fixture(Path(directory))
            payload = {
                "schema_version": 1,
                "hardware_decode_supported": True,
                "decoded_frames": 3,
                "texture_bindings": 3,
                "binding_failures": 0,
                "width": 64,
                "height": 64,
                "pixel_format": "bgra8Unorm",
                "always_copies_sample_data": False,
                "passed": True,
            }
            with patch(
                "vllm_apple.video_metal_bridge.subprocess.run",
                return_value=SimpleNamespace(stdout=json.dumps(payload).encode()),
            ):
                report = NativeVideoMetalBridge(executable).run(source, maximum_frames=3)
            self.assertTrue(report.passed)

    def test_rejects_self_inconsistent_native_report(self):
        with tempfile.TemporaryDirectory() as directory:
            executable, source = self.fixture(Path(directory))
            payload = {
                "schema_version": 1,
                "hardware_decode_supported": True,
                "decoded_frames": 3,
                "texture_bindings": 2,
                "binding_failures": 0,
                "width": 64,
                "height": 64,
                "pixel_format": "bgra8Unorm",
                "always_copies_sample_data": False,
                "passed": True,
            }
            with patch(
                "vllm_apple.video_metal_bridge.subprocess.run",
                return_value=SimpleNamespace(stdout=json.dumps(payload).encode()),
            ):
                with self.assertRaisesRegex(RuntimeError, "inconsistent"):
                    NativeVideoMetalBridge(executable).run(source)


if __name__ == "__main__":
    unittest.main()
