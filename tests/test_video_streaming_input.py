import hashlib
import tempfile
import unittest
from pathlib import Path

from vllm_apple.video_streaming_input import (
    StreamingVideoInputRegistry,
    StreamingVideoInputSession,
)


class VideoStreamingInputTests(unittest.TestCase):
    def test_ordered_chunks_finalize_private_digest_bound_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"first-second"
            session = StreamingVideoInputSession("stream", root)
            session.append(0, b"first-")
            update = session.append(
                1, b"second", final=True,
                expected_sha256=hashlib.sha256(payload).hexdigest(),
            )
            artifact = session.artifact
            self.assertTrue(update.final)
            self.assertEqual(artifact.path.read_bytes(), payload)
            self.assertEqual(artifact.path.stat().st_mode & 0o777, 0o600)
            session.close()
            self.assertFalse(artifact.path.exists())

    def test_sequence_budget_and_digest_mismatch_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = StreamingVideoInputSession(
                "stream", root, maximum_chunk_bytes=2, maximum_total_bytes=4
            )
            with self.assertRaisesRegex(ValueError, "sequence 0"):
                session.append(1, b"a")
            with self.assertRaisesRegex(ValueError, "budget"):
                session.append(0, b"abc")
            session.append(0, b"ab")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                session.append(1, b"cd", final=True, expected_sha256="0" * 64)
            self.assertFalse((root / ".stream.video-stream").exists())

    def test_registry_reaps_idle_session_and_unlinks_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            now = [0.0]
            registry = StreamingVideoInputRegistry(
                maximum_sessions=1, idle_timeout_seconds=5, clock=lambda: now[0]
            )
            session = registry.create("first", Path(directory))
            session.append(0, b"data")
            now[0] = 5
            self.assertEqual(registry.reap_idle(), ("first",))
            self.assertFalse((Path(directory) / ".first.video-stream").exists())
            registry.create("second", Path(directory))
            self.assertTrue(registry.close("second"))

    def test_artifact_is_unavailable_before_finalization(self):
        with tempfile.TemporaryDirectory() as directory:
            session = StreamingVideoInputSession("stream", Path(directory))
            session.append(0, b"data")
            with self.assertRaisesRegex(RuntimeError, "not finalized"):
                _ = session.artifact
            session.close()


if __name__ == "__main__":
    unittest.main()
