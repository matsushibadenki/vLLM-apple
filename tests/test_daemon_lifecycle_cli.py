import os
import plistlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm_apple.cli import main
from vllm_apple.daemon_lifecycle import (
    daemon_status,
    install_daemon,
    start_daemon,
    stop_daemon,
)


class DaemonLifecycleCLITests(unittest.TestCase):
    def test_install_is_private_atomic_and_refuses_implicit_overwrite(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "vllm_apple.daemon_lifecycle.sys.platform", "darwin"
        ), patch("vllm_apple.daemon_lifecycle.Path.home", return_value=Path(directory)):
            path = Path(directory) / "Library" / "LaunchAgents" / "test.daemon.plist"
            result = install_daemon(
                "model", label="test.daemon", plist_path=path,
                serve_arguments=(
                    "--socket-path", "/tmp/test.sock",
                    "--session-token-file", "/tmp/test.token",
                ))
            self.assertFalse(result.running)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            payload = plistlib.loads(path.read_bytes())
            self.assertEqual(payload["Label"], "test.daemon")
            self.assertEqual(payload["ProgramArguments"][-4:], [
                "--socket-path", "/tmp/test.sock",
                "--session-token-file", "/tmp/test.token",
            ])
            with self.assertRaises(FileExistsError):
                install_daemon("model", label="test.daemon", plist_path=path)

    def test_start_stop_and_status_use_user_domain_without_shell(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "vllm_apple.daemon_lifecycle.sys.platform", "darwin"
        ), patch("vllm_apple.daemon_lifecycle.Path.home", return_value=Path(directory)):
            path = Path(directory) / "test.daemon.plist"
            install_daemon("model", label="test.daemon", plist_path=path)
            completed = subprocess.CompletedProcess((), 0)
            with patch("vllm_apple.daemon_lifecycle.subprocess.run",
                       return_value=completed) as run:
                self.assertTrue(start_daemon(label="test.daemon", plist_path=path).running)
                self.assertFalse(stop_daemon(label="test.daemon", plist_path=path).running)
                self.assertTrue(daemon_status(label="test.daemon", plist_path=path).running)
            commands = [call.args[0] for call in run.call_args_list]
            domain = f"gui/{os.getuid()}"
            self.assertEqual(commands[0][:3], ("/bin/launchctl", "bootstrap", domain))
            self.assertEqual(commands[1], ("/bin/launchctl", "bootout", f"{domain}/test.daemon"))
            self.assertEqual(commands[2], ("/bin/launchctl", "print", f"{domain}/test.daemon"))
            self.assertTrue(all(call.kwargs.get("timeout") == 10 for call in run.call_args_list))

    def test_rejects_unsafe_label_and_plist(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "vllm_apple.daemon_lifecycle.sys.platform", "darwin"
        ), patch("vllm_apple.daemon_lifecycle.Path.home", return_value=Path(directory)):
            root = Path(directory)
            with self.assertRaises(ValueError):
                install_daemon(None, label="../bad", plist_path=root / "bad.plist")
            path = root / "test.daemon.plist"
            install_daemon(None, label="test.daemon", plist_path=path)
            path.chmod(0o644)
            with self.assertRaises(ValueError):
                daemon_status(label="test.daemon", plist_path=path)
            path.chmod(0o600)
            payload = plistlib.loads(path.read_bytes())
            payload["ProgramArguments"] = ["/tmp/untrusted"]
            path.write_bytes(plistlib.dumps(payload))
            path.chmod(0o600)
            with self.assertRaises(ValueError):
                start_daemon(label="test.daemon", plist_path=path)

    def test_install_defaults_to_authenticated_private_uds(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "vllm_apple.daemon_lifecycle.sys.platform", "darwin"
        ), patch("vllm_apple.daemon_lifecycle.Path.home", return_value=Path(directory)):
            path = Path(directory) / "test.daemon.plist"
            install_daemon(None, label="test.daemon", plist_path=path)
            arguments = plistlib.loads(path.read_bytes())["ProgramArguments"]
            self.assertIn("--socket-path", arguments)
            self.assertIn("--session-token-file", arguments)
            support = Path(directory) / "Library" / "Application Support" / "vLLM-Apple" / "daemon"
            self.assertEqual(support.stat().st_mode & 0o777, 0o700)
            with self.assertRaises(ValueError):
                install_daemon(
                    None, label="other.daemon", plist_path=Path(directory) / "other.daemon.plist",
                    serve_arguments=("--socket-path", "/tmp/only.sock"))

    def test_cli_install_and_status_exit_codes(self):
        # The CLI serializes the real dataclass; use the implementation once for
        # argument forwarding and mock only the external lifecycle boundary.
        from vllm_apple.daemon_lifecycle import DaemonLifecycleResult
        installed = DaemonLifecycleResult(
            "install", "test.daemon", "/tmp/test.daemon.plist", False)
        stopped = DaemonLifecycleResult(
            "status", "test.daemon", "/tmp/test.daemon.plist", False)
        with patch("vllm_apple.cli.install_daemon", return_value=installed) as install:
            self.assertEqual(main([
                "daemon-install", "model", "--label", "test.daemon",
                "--serve-argument=--socket-path", "--serve-argument=/tmp/a.sock",
            ]), 0)
        install.assert_called_once_with(
            "model", label="test.daemon", plist_path=None,
            serve_arguments=["--socket-path", "/tmp/a.sock"], force=False)
        with patch("vllm_apple.cli.daemon_status", return_value=stopped):
            self.assertEqual(main(["daemon-status", "--label", "test.daemon"]), 1)


if __name__ == "__main__":
    unittest.main()
