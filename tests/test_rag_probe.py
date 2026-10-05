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

    def test_shutdown_timeout_is_separate_from_generation_error(self):
        path = Path("scripts/probe_rag_http.py")
        spec = importlib.util.spec_from_file_location("rag_shutdown_probe", path)
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        process = Mock(returncode=-9)
        process.poll.return_value = None
        process.wait.side_effect = [probe.subprocess.TimeoutExpired("backend", 15), -9]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "shutdown.json"
            response = Mock(status=200)
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            with patch("sys.argv", [str(path), "--model", "unused", "--output", str(output), "--shutdown-diagnostics"]), \
                 patch.object(probe.subprocess, "Popen", return_value=process), \
                 patch.object(probe, "urlopen", return_value=response), \
                 patch.object(probe, "answer_rag", side_effect=[
                     dict(status="references_valid", answer="12347 [S1]", token_budget_verified=True),
                     dict(status="abstained", generation_performed=False)] * 3), \
                 patch.object(probe.socket, "socket") as sockets:
                sockets.return_value.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 12345)
                self.assertEqual(probe.main(), 1)
            report = json.loads(output.read_text())
            self.assertIsNone(report['error'])
            self.assertTrue(report['shutdown']['forced_kill'])
            self.assertFalse(report['shutdown']['graceful'])
            self.assertFalse(report['passed'])
            self.assertTrue(report['shutdown']['stack_dump_requested'])
            self.assertEqual(process.send_signal.call_args_list[-1].args, (signal.SIGUSR1,))
            process.kill.assert_called_once()

    def test_model_identity_detects_weight_tokenizer_changes_and_missing_artifacts(self):
        from scripts.probe_rag_http import evidence_snapshot, verify_identity

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(evidence_snapshot(root)['model_complete'])
            for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json', 'model.safetensors'):
                (root / name).write_bytes(b'original')
            before = evidence_snapshot(root)
            self.assertTrue(verify_identity(before, evidence_snapshot(root))['model_identity_unchanged'])
            for name in ('model.safetensors', 'tokenizer.json'):
                (root / name).write_bytes(b'changed')
                self.assertFalse(verify_identity(before, evidence_snapshot(root))['model_identity_unchanged'])
                (root / name).write_bytes(b'original')
            changed_runtime = dict(before, packages=dict(before['packages'], mlx='unverified-version'))
            self.assertFalse(verify_identity(before, changed_runtime)['runtime_identity_unchanged'])
