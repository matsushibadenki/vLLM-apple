import os
import tempfile
import unittest
import wave
from pathlib import Path

from vllm_apple.audio_generation import (
    AudioGenerationKind,
    AudioGenerationRequest,
    AudioWorkerTelemetry,
    qualify_audio_generation,
)


class Backend:
    def __init__(self, *, unsafe_mode=False, wrong_rate=False):
        self.unsafe_mode = unsafe_mode
        self.wrong_rate = wrong_rate

    def generate(self, request, output_path):
        with wave.open(str(output_path), "wb") as wav:
            wav.setnchannels(request.channels)
            wav.setsampwidth(2)
            wav.setframerate(request.sample_rate + (1 if self.wrong_rate else 0))
            wav.writeframes(bytes(request.channels * 2 * request.sample_rate))
        output_path.chmod(0o644 if self.unsafe_mode else 0o600)
        return AudioWorkerTelemetry(1000, "normal", "nominal", "fixture", "1")


class AudioGenerationTests(unittest.TestCase):
    def request(self, kind=AudioGenerationKind.SPEECH):
        return AudioGenerationRequest(kind, "private prompt", 7, 16_000, 1, 1.0)

    def test_speech_and_music_qualification_is_private_and_cleans_output(self):
        for kind in AudioGenerationKind:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                root.chmod(0o700)
                report = qualify_audio_generation(Backend(), self.request(kind), root)
                payload = report.to_dict()
                self.assertTrue(report.passed)
                self.assertEqual(report.frames, 16_000)
                self.assertFalse((root / "generated.wav").exists())
                self.assertNotIn("private prompt", str(payload))
                self.assertFalse(payload["stores_prompt"])
                self.assertFalse(payload["stores_output"])

    def test_unsafe_output_and_metadata_mismatch_are_deleted(self):
        for backend, message in (
            (Backend(unsafe_mode=True), "unsafe"),
            (Backend(wrong_rate=True), "does not match"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                root.chmod(0o700)
                with self.assertRaisesRegex(ValueError, message):
                    qualify_audio_generation(backend, self.request(), root)
                self.assertFalse((root / "generated.wav").exists())

    def test_private_root_and_request_bounds_fail_before_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o755)
            with self.assertRaisesRegex(ValueError, "root"):
                qualify_audio_generation(Backend(), self.request(), root)
        with tempfile.TemporaryDirectory() as directory:
            real_root = Path(directory) / "real"
            real_root.mkdir(mode=0o700)
            linked_root = Path(directory) / "linked"
            linked_root.symlink_to(real_root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "root"):
                qualify_audio_generation(Backend(), self.request(), linked_root)
        with self.assertRaises(ValueError):
            AudioGenerationRequest(
                AudioGenerationKind.MUSIC, "x", 0, 16_000, 1, 1801
            )

    def test_symlink_output_is_not_followed_or_deleted(self):
        class SymlinkBackend:
            def __init__(self, target):
                self.target = target

            def generate(self, request, output_path):
                output_path.symlink_to(self.target)
                return AudioWorkerTelemetry(1000, "normal", "nominal", "fixture", "1")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            target = root / "target"
            target.write_bytes(b"preserve")
            target.chmod(0o600)
            with self.assertRaises(OSError):
                qualify_audio_generation(SymlinkBackend(target), self.request(), root)
            self.assertEqual(target.read_bytes(), b"preserve")
            self.assertTrue((root / "generated.wav").is_symlink())
            os.unlink(root / "generated.wav")


if __name__ == "__main__":
    unittest.main()
