import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm_apple.audio_generation import AudioGenerationKind, AudioGenerationRequest
from vllm_apple.mlx_audio_generation import (
    MLXAudioGenerationBackend,
    MLXMusicGenerationBackend,
)
from vllm_apple.mlx_audio_generation_worker import _consume_request


class FakeProcess:
    returncode = 0
    pid = 1

    def communicate(self, timeout):
        return json.dumps({
            "schema_version": 1,
            "backend": "mlx-audio-kokoro",
            "backend_version": "0.5.4",
            "peak_rss_bytes": 1000,
            "memory_pressure": "normal",
            "thermal_state": "nominal",
        }).encode(), b""


class FakeMusicProcess(FakeProcess):
    def communicate(self, timeout):
        return json.dumps({
            "schema_version": 1,
            "backend": "mlx-audio-minimax-music3",
            "backend_version": "0.5.4",
            "peak_rss_bytes": 1000,
            "memory_pressure": "normal",
            "thermal_state": "nominal",
        }).encode(), b""


class MLXAudioGenerationTests(unittest.TestCase):
    def test_private_request_is_consumed_and_not_exposed_in_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            model = root / "model"
            model.mkdir()
            site = root / "site"
            site.mkdir()
            backend = MLXAudioGenerationBackend(
                python_executable=Path(sys.executable),
                model=model,
                workspace_root=root,
                backend_version="0.5.4",
                python_path=(site,),
            )
            request = AudioGenerationRequest(
                AudioGenerationKind.SPEECH, "private words", 0, 24_000, 1, 10
            )
            with patch("vllm_apple.mlx_audio_generation.subprocess.Popen", return_value=FakeProcess()) as spawned:
                telemetry = backend.generate(request, root / "generated.wav")
            command = spawned.call_args.args[0]
            self.assertNotIn("private words", " ".join(command))
            self.assertEqual(telemetry.backend_version, "0.5.4")
            self.assertEqual(list(root.glob("request-*.json")), [])

    def test_worker_request_requires_0600_and_is_one_shot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text('{"value":1}')
            path.chmod(0o600)
            self.assertEqual(_consume_request(path), {"value": 1})
            self.assertFalse(path.exists())
            path.write_text("{}")
            path.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "unsafe"):
                _consume_request(path)
            os.unlink(path)

    def test_music_prompt_and_lyrics_are_not_exposed_in_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            model = root / "model"
            model.mkdir()
            site = root / "site"
            site.mkdir()
            backend = MLXMusicGenerationBackend(
                python_executable=Path(sys.executable),
                model=model,
                workspace_root=root,
                backend_version="0.5.4",
                python_path=(site,),
                lyrics="[instrumental]",
                steps=4,
            )
            request = AudioGenerationRequest(
                AudioGenerationKind.MUSIC, "private melody", 7, 44_100, 2, 5
            )
            with patch(
                "vllm_apple.mlx_audio_generation.subprocess.Popen",
                return_value=FakeMusicProcess(),
            ) as spawned:
                telemetry = backend.generate(request, root / "generated.wav")
            command = spawned.call_args.args[0]
            self.assertNotIn("private melody", " ".join(command))
            self.assertNotIn("[instrumental]", " ".join(command))
            self.assertEqual(telemetry.backend, "mlx-audio-minimax-music3")
            self.assertEqual(list(root.glob("request-*.json")), [])


if __name__ == "__main__":
    unittest.main()
