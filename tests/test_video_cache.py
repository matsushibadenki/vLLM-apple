import hashlib
import unittest

from vllm_apple.video_cache import VideoArtifactCache, VideoCacheKey, VideoCacheKind


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def key(kind, identifier, start=0, end=1):
    return VideoCacheKey(
        kind, digest(identifier), digest("transform"), digest("model"), start, end
    )


class VideoCacheTests(unittest.TestCase):
    def test_tier_budget_prevents_frames_from_evicting_embeddings(self):
        cache = VideoArtifactCache[str](
            maximum_entries=10,
            maximum_bytes=20,
            tier_maximum_bytes={VideoCacheKind.FRAME: 8, VideoCacheKind.EMBEDDING: 12},
        )
        cache.put(key(VideoCacheKind.EMBEDDING, "embedding"), "E", size_bytes=8)
        cache.put(key(VideoCacheKind.FRAME, "frame-1"), "F1", size_bytes=8)
        cache.put(key(VideoCacheKind.FRAME, "frame-2"), "F2", size_bytes=8)
        self.assertEqual(cache.get(key(VideoCacheKind.EMBEDDING, "embedding")), "E")
        self.assertIsNone(cache.get(key(VideoCacheKind.FRAME, "frame-1")))
        self.assertEqual(cache.get(key(VideoCacheKind.FRAME, "frame-2")), "F2")

    def test_key_separates_artifact_kind_and_time_range(self):
        frame = key(VideoCacheKind.FRAME, "same", 0, 10)
        patch = key(VideoCacheKind.PATCH, "same", 0, 10)
        later = key(VideoCacheKind.FRAME, "same", 10, 20)
        self.assertNotEqual(frame.digest, patch.digest)
        self.assertNotEqual(frame.digest, later.digest)

    def test_oversize_rejection_and_tier_clear_are_observable(self):
        cache = VideoArtifactCache[str](
            maximum_bytes=8,
            tier_maximum_bytes={VideoCacheKind.SCENE: 4},
        )
        self.assertFalse(cache.put(key(VideoCacheKind.SCENE, "large"), "x", size_bytes=5))
        cache.put(key(VideoCacheKind.SCENE, "small"), "x", size_bytes=4)
        cache.put(key(VideoCacheKind.PATCH, "patch"), "p", size_bytes=4)
        cache.clear(VideoCacheKind.SCENE)
        self.assertEqual(cache.snapshot.entries, 1)
        self.assertEqual(cache.snapshot.rejected_oversize, 1)

    def test_invalid_time_range_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "invalid video cache key"):
            key(VideoCacheKind.FRAME, "video", 1, 1)


if __name__ == "__main__":
    unittest.main()
