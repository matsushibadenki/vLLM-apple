import os
from pathlib import Path
import tempfile
import threading
import time
import unittest

from vllm_apple.numeric_conversion_cache import (
    NumericConversionCache,
    NumericConversionCacheIdentity,
)


def identity(value="a"):
    return NumericConversionCacheIdentity(
        value * 64, "b" * 64, "c" * 64, "d" * 64, "e" * 64
    )


def write_output(path, value=b"converted"):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(value)


class NumericConversionCacheTests(unittest.TestCase):
    def test_single_flight_publish_and_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            root.mkdir(mode=0o700)
            cache = NumericConversionCache(root, signing_key=b"k" * 32)
            calls = []
            barrier = threading.Barrier(2)

            def converter(path):
                calls.append(1)
                time.sleep(0.02)
                write_output(path)

            results = []
            def run():
                barrier.wait()
                results.append(cache.get_or_create(identity(), converter))

            threads = [threading.Thread(target=run) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(len(calls), 1)
            self.assertEqual(results[0].output_sha256, results[1].output_sha256)
            self.assertEqual(cache.load(identity()).output_path.read_bytes(), b"converted")

    def test_tamper_quarantine_and_bounded_eviction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            root.mkdir(mode=0o700)
            cache = NumericConversionCache(
                root, signing_key=b"k" * 32, maximum_entries=1, maximum_bytes=64
            )
            first = cache.get_or_create(identity("a"), lambda path: write_output(path, b"one"))
            cache.get_or_create(identity("f"), lambda path: write_output(path, b"two"))
            self.assertFalse(first.output_path.exists())
            current = cache.load(identity("f"))
            current.output_path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "quarantined"):
                cache.load(identity("f"))

    def test_failed_conversion_rolls_back_and_can_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            root.mkdir(mode=0o700)
            cache = NumericConversionCache(root, signing_key=b"k" * 32)
            with self.assertRaisesRegex(RuntimeError, "failed"):
                cache.get_or_create(identity(), lambda _path: (_ for _ in ()).throw(
                    RuntimeError("failed")
                ))
            self.assertFalse(any((root / "entries").iterdir()))
            self.assertEqual(
                cache.get_or_create(identity(), write_output).output_path.read_bytes(),
                b"converted",
            )


if __name__ == "__main__":
    unittest.main()
