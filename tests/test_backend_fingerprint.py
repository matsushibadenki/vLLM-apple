import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vllm_apple.backend_fingerprint import fingerprint_backend, installed_fingerprint


class BackendFingerprintTests(unittest.TestCase):
    def test_inventory_content_change_invalidates_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launcher = root / "launcher"
            source = root / "engine.py"
            launcher.write_text("launcher")
            source.write_text("version one")
            distribution = SimpleNamespace(
                metadata={"Name": "synthetic-backend"}, version="1",
                files=[Path("engine.py")], read_text=lambda name: None,
                locate_file=lambda name: root / name,
            )
            with patch("importlib.metadata.distributions", return_value=[distribution]):
                first = installed_fingerprint(launcher)
                source.write_text("version two")
                second = installed_fingerprint(launcher)
            self.assertNotEqual(first, second)

    def test_missing_inventory_and_editable_install_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            launcher = Path(directory) / "launcher"
            launcher.write_text("launcher")
            for direct, message in ((None, "no file inventory"),
                                    ('{"dir_info":{"editable":true}}', "editable")):
                distribution = SimpleNamespace(files=None, read_text=lambda name: direct)
                with patch("importlib.metadata.distributions", return_value=[distribution]):
                    with self.assertRaisesRegex(ValueError, message):
                        installed_fingerprint(launcher)

    def test_nonpython_launcher_is_rejected_before_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "python").write_text("fixture")
            launcher = root / "server"
            launcher.write_text("#!/usr/bin/env python\n")
            with patch("vllm_apple.backend_fingerprint.subprocess.run") as run:
                with self.assertRaises(ValueError):
                    fingerprint_backend(launcher)
                run.assert_not_called()

    def test_brew_inventory_tracks_unlisted_file_addition_change_and_removal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
            launcher = root / "server"
            launcher.write_text("fixture")
            distribution = SimpleNamespace(
                metadata={"Name": "synthetic-brew"}, version="1", files=None,
                read_text=lambda name: "brew\n" if name == "INSTALLER" else None,
                locate_file=lambda name: root / name,
            )
            with patch("importlib.metadata.distributions", return_value=[distribution]), patch(
                "vllm_apple.backend_fingerprint.sys.prefix", str(root)
            ):
                first = installed_fingerprint(launcher)
                source = root / "unlisted.py"
                source.write_text("one")
                second = installed_fingerprint(launcher)
                source.write_text("two")
                third = installed_fingerprint(launcher)
                source.unlink()
                self.assertEqual(first, installed_fingerprint(launcher))
                self.assertEqual(len({first, second, third}), 3)

    def test_brew_inventory_rejects_shared_environment_and_external_symlinks(self):
        from vllm_apple.backend_fingerprint import _environment_inventory

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "env"
            root.mkdir()
            config = root / "pyvenv.cfg"
            config.write_text("include-system-site-packages = true\n")
            with self.assertRaisesRegex(ValueError, "isolated"):
                _environment_inventory(root)
            config.write_text("include-system-site-packages = false\n")
            external = Path(directory) / "outside.py"
            external.write_text("fixture")
            (root / "link.py").symlink_to(external)
            with self.assertRaisesRegex(ValueError, "external symlink"):
                _environment_inventory(root)

    def test_brew_inventory_rejects_entry_overflow(self):
        from vllm_apple.backend_fingerprint import _environment_inventory

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
            (root / "extra.py").write_text("fixture")
            with patch("vllm_apple.backend_fingerprint.MAX_FILES", 1):
                with self.assertRaisesRegex(ValueError, "exceeds limit"):
                    _environment_inventory(root)

    def test_brew_fallback_still_checks_pth_search_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
            (root / "external.pth").write_text("/external/source\n")
            launcher = root / "server"
            launcher.write_text("fixture")
            distribution = SimpleNamespace(
                metadata={"Name": "synthetic-brew"}, version="1", files=None,
                read_text=lambda name: "brew\n" if name == "INSTALLER" else None,
                locate_file=lambda name: root / name,
            )
            with patch("importlib.metadata.distributions", return_value=[distribution]), patch(
                "vllm_apple.backend_fingerprint.sys.prefix", str(root)
            ):
                with self.assertRaisesRegex(ValueError, "external backend search paths"):
                    installed_fingerprint(launcher)
