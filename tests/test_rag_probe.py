import importlib.util
import json
import signal
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class RagProbeTests(unittest.TestCase):
    def test_readiness_failure_records_failure_and_stops_owned_backend(self):
        path = Path("scripts/probe_rag_http.py")
        spec = importlib.util.spec_from_file_location("rag_http_probe", path)
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        process = Mock(returncode=0)
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed.json"
            with patch("sys.argv", [str(path), "--model", "unused", "--output", str(output)]), \
                 patch.object(probe.subprocess, "Popen", return_value=process), \
                 patch.object(probe, "urlopen", side_effect=RuntimeError("readiness failed")), \
                 patch.object(probe.socket, "socket") as socket_factory:
                socket_factory.return_value.__enter__.return_value.getsockname.return_value = (
                    "127.0.0.1", 12345)
                self.assertEqual(probe.main(), 1)
            report = json.loads(output.read_text())
            self.assertFalse(report["passed"])
            self.assertEqual(report["cases"], [])
            self.assertIn("readiness failed", report["error"])
            process.send_signal.assert_called_once_with(signal.SIGINT)
            process.wait.assert_called_once_with(timeout=15)
