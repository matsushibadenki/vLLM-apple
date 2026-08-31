import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


SAMPLE = Path("samples/VLLMAppleChatXcode")


class XcodeSampleTests(unittest.TestCase):
    def test_project_declares_app_package_localizations_and_embed_phase(self) -> None:
        project = (SAMPLE / "project.yml").read_text(encoding="utf-8")
        self.assertIn("type: application", project)
        self.assertIn("path: ../../sdk/swift", project)
        self.assertIn("ENABLE_HARDENED_RUNTIME: YES", project)
        self.assertIn("Scripts/embed-daemon.sh", project)
        package = Path("samples/VLLMAppleChat/Package.swift").read_text(encoding="utf-8")
        self.assertIn('defaultLocalization: "en"', package)
        for language in ("en.lproj", "ja.lproj", "zh-Hans.lproj"):
            self.assertTrue(
                (Path("samples/VLLMAppleChat/Sources/VLLMAppleChat/Resources") / language).is_dir()
            )

    def test_embed_phase_places_executable_in_app_auxiliary_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            daemon = root / "source-daemon"
            daemon.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            daemon.chmod(0o700)
            environment = {
                **os.environ,
                "VLLM_APPLE_DAEMON_SOURCE": str(daemon),
                "TARGET_BUILD_DIR": str(root / "Build"),
                "EXECUTABLE_FOLDER_PATH": "VLLMAppleChat.app/Contents/MacOS",
                "CODE_SIGNING_ALLOWED": "NO",
            }
            subprocess.run(
                [str(SAMPLE / "Scripts/embed-daemon.sh")],
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            embedded = (
                root / "Build/VLLMAppleChat.app/Contents/MacOS/vllm-appled"
            )
            self.assertTrue(embedded.is_file())
            self.assertFalse(embedded.is_symlink())
            self.assertTrue(embedded.stat().st_mode & stat.S_IXUSR)

    def test_embed_phase_rejects_symlink_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            daemon = root / "daemon"
            daemon.write_text("#!/bin/sh\n", encoding="utf-8")
            daemon.chmod(0o700)
            link = root / "link"
            link.symlink_to(daemon)
            completed = subprocess.run(
                [str(SAMPLE / "Scripts/embed-daemon.sh")],
                env={
                    **os.environ,
                    "VLLM_APPLE_DAEMON_SOURCE": str(link),
                    "TARGET_BUILD_DIR": str(root / "Build"),
                    "EXECUTABLE_FOLDER_PATH": "App.app/Contents/MacOS",
                    "CODE_SIGNING_ALLOWED": "NO",
                },
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("not a symlink", completed.stderr)


if __name__ == "__main__":
    unittest.main()
