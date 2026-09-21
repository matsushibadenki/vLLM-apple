import hashlib
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

from vllm_apple.activation_statistics import (
    ActivationStatisticsStream,
    OnlineActivationStatistics,
)


class ActivationStatisticsTests(unittest.TestCase):
    def test_online_statistics_merge_without_retaining_values(self):
        statistics = OnlineActivationStatistics()
        statistics.update((-2.0, 0.0))
        statistics.update((2.0, 4.0))
        snapshot = statistics.snapshot()
        self.assertEqual(snapshot.count, 4)
        self.assertEqual(snapshot.mean, 1.0)
        self.assertEqual(snapshot.variance, 5.0)
        self.assertEqual(snapshot.minimum, -2.0)
        self.assertEqual(snapshot.maximum, 4.0)
        self.assertEqual(snapshot.absolute_maximum, 4.0)
        self.assertEqual(snapshot.zeros, 1)

    def test_non_finite_and_empty_updates_fail_closed(self):
        statistics = OnlineActivationStatistics()
        with self.assertRaises(ValueError):
            statistics.update(())
        with self.assertRaises(ValueError):
            statistics.update((math.nan,))

    def test_private_stream_stores_aggregate_and_fingerprint_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "private"
            fingerprint = hashlib.sha256(b"layer-1").hexdigest()
            statistics = OnlineActivationStatistics()
            statistics.update((1.0, 2.0, 3.0))
            path = root / "statistics.jsonl"
            with ActivationStatisticsStream(path, maximum_bytes=4096) as stream:
                stream.append(fingerprint, statistics.snapshot())
            record = json.loads(path.read_text())
            self.assertEqual(record["tensor_fingerprint"], fingerprint)
            self.assertNotIn("values", record)
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)

    def test_stream_capacity_and_fingerprint_are_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            stream = ActivationStatisticsStream(
                Path(directory) / "private" / "stats.jsonl", maximum_bytes=1
            )
            with self.assertRaises(ValueError):
                stream.append("not-a-digest", OnlineActivationStatistics().snapshot())
            with self.assertRaises(ValueError):
                stream.append("0" * 64, OnlineActivationStatistics().snapshot())
            stream.close()


if __name__ == "__main__":
    unittest.main()
