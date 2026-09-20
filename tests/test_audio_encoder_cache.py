import hashlib
import unittest

from vllm_apple.audio_encoder_cache import (
    AudioEncoderCache,
    AudioEncoderCacheKey,
    audio_feature_fingerprint,
)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def key(segment="audio", *, start=0, end=16000, rate=16000):
    return AudioEncoderCacheKey(
        digest(segment), digest("encoder"), digest("features"),
        rate, 1, start, end,
    )


class AudioEncoderCacheTests(unittest.TestCase):
    def test_key_is_bound_to_segment_and_audio_configuration(self):
        self.assertNotEqual(key(start=0, end=100).digest, key(start=100, end=200).digest)
        self.assertNotEqual(key(rate=16000).digest, key(rate=48000).digest)

    def test_lru_enforces_entry_and_byte_limits(self):
        cache = AudioEncoderCache[str](maximum_entries=2, maximum_bytes=8)
        cache.put(key("a"), "A", size_bytes=4)
        cache.put(key("b"), "B", size_bytes=4)
        self.assertEqual(cache.get(key("a")), "A")
        cache.put(key("c"), "C", size_bytes=4)
        self.assertIsNone(cache.get(key("b")))
        self.assertEqual(cache.snapshot.entries, 2)
        self.assertEqual(cache.snapshot.evictions, 1)

    def test_oversize_is_rejected_and_compute_is_reused(self):
        cache = AudioEncoderCache[str](maximum_entries=2, maximum_bytes=4)
        self.assertFalse(cache.put(key(), "large", size_bytes=5))
        self.assertEqual(cache.snapshot.rejected_oversize, 1)
        calls = []

        def compute():
            calls.append(True)
            return "encoded", 4

        self.assertEqual(cache.get_or_compute(key(), compute), "encoded")
        self.assertEqual(cache.get_or_compute(key(), compute), "encoded")
        self.assertEqual(len(calls), 1)

    def test_feature_fingerprint_is_stable_and_validated(self):
        first = audio_feature_fingerprint(
            model_rate=16000, frame_length=400, hop_length=160, feature_bands=80
        )
        second = audio_feature_fingerprint(
            model_rate=16000, frame_length=400, hop_length=160, feature_bands=80
        )
        self.assertEqual(first, second)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            audio_feature_fingerprint(
                model_rate=0, frame_length=400, hop_length=160, feature_bands=80
            )


if __name__ == "__main__":
    unittest.main()
