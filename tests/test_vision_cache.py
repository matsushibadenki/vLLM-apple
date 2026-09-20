import hashlib
import unittest

from vllm_apple.vision_cache import (
    VisionCacheKey,
    VisionEncoderCache,
    preprocessing_fingerprint,
)
from vllm_apple.vision_frontend import VisionPreprocessSpec


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def key(image: str, revision: str = "revision") -> VisionCacheKey:
    return VisionCacheKey(
        digest(image), digest(revision), digest("preprocess"), digest("encoder")
    )


class VisionEncoderCacheTests(unittest.TestCase):
    def test_lru_enforces_entry_and_byte_limits(self):
        cache = VisionEncoderCache[str](maximum_entries=2, maximum_bytes=10)
        self.assertTrue(cache.put(key("a"), "A", size_bytes=4))
        self.assertTrue(cache.put(key("b"), "B", size_bytes=4))
        self.assertEqual(cache.get(key("a")), "A")
        self.assertTrue(cache.put(key("c"), "C", size_bytes=4))
        self.assertIsNone(cache.get(key("b")))
        self.assertEqual(cache.get(key("a")), "A")
        self.assertEqual(cache.get(key("c")), "C")
        self.assertEqual(cache.stats.entries, 2)
        self.assertEqual(cache.stats.resident_bytes, 8)
        self.assertEqual(cache.stats.evictions, 1)

    def test_key_separates_model_revisions_and_oversize_is_not_cached(self):
        cache = VisionEncoderCache[str](maximum_entries=2, maximum_bytes=4)
        first = key("same", "one")
        second = key("same", "two")
        self.assertNotEqual(first.digest, second.digest)
        self.assertFalse(cache.put(first, "large", size_bytes=5))
        self.assertIsNone(cache.get(first))

    def test_get_or_compute_reuses_value_and_spec_fingerprint_is_stable(self):
        cache = VisionEncoderCache[str](maximum_entries=2, maximum_bytes=16)
        calls = []

        def compute():
            calls.append(True)
            return "embedding", 8

        self.assertEqual(cache.get_or_compute(key("a"), compute), "embedding")
        self.assertEqual(cache.get_or_compute(key("a"), compute), "embedding")
        self.assertEqual(len(calls), 1)
        spec = VisionPreprocessSpec(8, 8, 4, (0.5,) * 3, (0.5,) * 3)
        self.assertEqual(preprocessing_fingerprint(spec), preprocessing_fingerprint(spec))

    def test_invalid_non_digest_key_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            VisionCacheKey("image", digest("r"), digest("p"), digest("e"))


if __name__ == "__main__":
    unittest.main()
