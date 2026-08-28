import json
import stat
import tempfile
import unittest
from pathlib import Path

from vllm_apple.profile import build_profile, save_profile
from vllm_apple.types import HardwareInfo, MemoryInfo


class ProfileTests(unittest.TestCase):
    def test_profile_is_atomically_saved_with_private_permissions(self) -> None:
        hardware = HardwareInfo(
            platform="Darwin",
            architecture="arm64",
            soc="Apple Test",
            physical_cpu_count=4,
            logical_cpu_count=4,
            gpu_core_count=None,
            memory=MemoryInfo(total_bytes=1024, available_bytes=512),
            is_apple_silicon=True,
            os_version="test",
        )
        profile = build_profile(hardware)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            saved = save_profile(profile, path)
            self.assertEqual(saved, path)
            self.assertEqual(json.loads(path.read_text())["hardware"]["soc"], "Apple Test")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)


if __name__ == "__main__":
    unittest.main()

