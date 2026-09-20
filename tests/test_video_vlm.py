import hashlib
import unittest

from vllm_apple.video_cache import VideoArtifactCache
from vllm_apple.video_temporal_sampler import TemporalVideoFrame
from vllm_apple.video_vlm import (
    VideoFrameEmbedding,
    VideoVLMIntegrator,
    VideoVLMRequest,
    embedding_digest,
)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


class Encoder:
    def __init__(self):
        self.calls = []

    def encode(self, frames):
        self.calls.append(tuple(frame.frame_id for frame in frames))
        return tuple(
            VideoFrameEmbedding(
                frame.frame_id, f"embedding-{frame.frame_id}", 4, digest(frame.frame_id)
            )
            for frame in frames
        )


class Language:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, embeddings):
        self.calls.append((prompt, tuple(item.frame_id for item in embeddings)))
        return "  video answer  "


def request():
    return VideoVLMRequest(
        " describe ", digest("video"), digest("transform"), digest("model"),
        tuple(TemporalVideoFrame(str(index), float(index), index) for index in range(5)),
        maximum_frames=3,
    )


class VideoVLMTests(unittest.TestCase):
    def test_samples_encodes_in_time_order_and_reuses_embedding_cache(self):
        encoder = Encoder()
        language = Language()
        integrator = VideoVLMIntegrator(
            encoder, language, VideoArtifactCache(maximum_bytes=1024)
        )
        first = integrator.run(request())
        second = integrator.run(request())
        self.assertEqual(first.selected_frame_ids, second.selected_frame_ids)
        self.assertEqual(first.cache_misses, 3)
        self.assertEqual(second.cache_hits, 3)
        self.assertEqual(len(encoder.calls), 1)
        self.assertEqual(language.calls[-1][1], second.selected_frame_ids)
        self.assertEqual(second.text, "video answer")

    def test_encoder_count_or_order_mismatch_fails_closed(self):
        class BrokenEncoder:
            def encode(self, frames):
                return tuple(reversed([
                    VideoFrameEmbedding(frame.frame_id, "x", 1, digest(frame.frame_id))
                    for frame in frames
                ]))

        integrator = VideoVLMIntegrator(
            BrokenEncoder(), Language(), VideoArtifactCache(maximum_bytes=1024)
        )
        with self.assertRaisesRegex(RuntimeError, "misordered"):
            integrator.run(request())

    def test_invalid_backend_response_and_empty_video_fail_closed(self):
        class EmptyLanguage:
            def generate(self, prompt, embeddings):
                return ""

        integrator = VideoVLMIntegrator(
            Encoder(), EmptyLanguage(), VideoArtifactCache(maximum_bytes=1024)
        )
        with self.assertRaisesRegex(RuntimeError, "invalid response"):
            integrator.run(request())
        empty = VideoVLMRequest(
            "prompt", digest("video"), digest("transform"), digest("model"), (), 1
        )
        with self.assertRaisesRegex(ValueError, "no frames"):
            integrator.run(empty)

    def test_embedding_digest_requires_nonempty_immutable_bytes(self):
        self.assertEqual(len(embedding_digest(b"values")), 64)
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            embedding_digest(b"")


if __name__ == "__main__":
    unittest.main()
