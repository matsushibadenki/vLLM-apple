import threading
import unittest

from vllm_apple.numeric_pipeline import NumericBandwidthLedger, NumericTilePipeline


class NumericPipelineTests(unittest.TestCase):
    def test_double_buffer_pipeline_preserves_order_and_releases_bandwidth(self):
        ledger = NumericBandwidthLedger(1000)
        report = NumericTilePipeline(
            ledger, reserved_bandwidth_bytes_per_second=600, buffer_count=2
        ).execute(
            tile_count=3,
            read_tile=lambda index: bytes([index + 1]) * 4,
            convert_tile=lambda value: value[::-1],
            consume_tile=lambda index, value: (index, value),
        )
        self.assertEqual([item[0] for item in report.results], [0, 1, 2])
        self.assertEqual(report.source_bytes, 12)
        self.assertEqual(report.converted_bytes, 12)
        self.assertEqual(report.completion_barriers, 3)
        self.assertEqual(ledger.snapshot()["reserved_bytes_per_second"], 0)

    def test_bandwidth_ceiling_rejects_contention(self):
        ledger = NumericBandwidthLedger(100)
        lease = ledger.reserve(60)
        with self.assertRaisesRegex(ValueError, "ceiling"):
            NumericTilePipeline(
                ledger, reserved_bandwidth_bytes_per_second=50
            ).execute(
                tile_count=1, read_tile=lambda _index: b"x",
                convert_tile=lambda value: value,
                consume_tile=lambda _index, value: value,
            )
        lease.release()

    def test_cancellation_and_invalid_tile_release_resources(self):
        ledger = NumericBandwidthLedger(100)
        cancellation = threading.Event()
        cancellation.set()
        pipeline = NumericTilePipeline(ledger, reserved_bandwidth_bytes_per_second=50)
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            pipeline.execute(
                tile_count=2, read_tile=lambda _index: b"x",
                convert_tile=lambda value: value,
                consume_tile=lambda _index, value: value,
                cancellation=cancellation,
            )
        self.assertEqual(ledger.snapshot()["reserved_bytes_per_second"], 0)
        with self.assertRaisesRegex(ValueError, "source tile"):
            pipeline.execute(
                tile_count=1, read_tile=lambda _index: b"",
                convert_tile=lambda value: value,
                consume_tile=lambda _index, value: value,
            )
        self.assertEqual(ledger.snapshot()["reserved_bytes_per_second"], 0)


if __name__ == "__main__":
    unittest.main()
