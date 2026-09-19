import unittest

from vllm_apple.phase_probe import _TrimmedTextDigest


class TrimmedTextDigestTests(unittest.TestCase):
    def test_chunk_boundaries_preserve_stripped_exact_semantics(self):
        for text in ("青", " \n青  \n", "蓝\u3000", "cobalt \n", "co balt", "青です", "青\n説明", "   "):
            for expected in ("青", "蓝", "cobalt", "co balt"):
                for split in range(len(text) + 1):
                    digest = _TrimmedTextDigest()
                    digest.update(text[:split])
                    digest.update(text[split:])
                    self.assertEqual(digest.matches(expected), text.strip() == expected)

    def test_internal_whitespace_is_not_discarded(self):
        digest = _TrimmedTextDigest()
        for chunk in ("  co", " ", "balt", "\n"):
            digest.update(chunk)
        self.assertFalse(digest.matches("cobalt"))
        self.assertTrue(digest.matches("co balt"))
