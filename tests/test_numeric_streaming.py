import hashlib
import threading
import unittest

from vllm_apple.numeric_streaming import (
    MAX_NUMERIC_TILE_BYTES,
    NumericDoubleBufferStream,
    NumericStreamingCancelled,
    NumericStreamingPlan,
)
from vllm_apple.qwen4_component_loader import Qwen4MemoryAdmission


class NumericStreamingTests(unittest.TestCase):
    @staticmethod
    def plan(source: bytes, *, tile_bytes: int = 4, buffer_count: int = 2):
        return NumericStreamingPlan(
            hashlib.sha256(source).hexdigest(),
            len(source),
            tile_bytes,
            buffer_count,
        )

    def test_double_buffer_ownership_prevents_overwrite_and_reuses_released_slot(self):
        source = b"abcdefghij"
        plan = self.plan(source)
        with NumericDoubleBufferStream(plan, source) as stream:
            first = stream.acquire_next()
            second = stream.acquire_next()
            first_view = first.view()
            self.assertEqual(first.read(), b"abcd")
            self.assertEqual(second.read(), b"efgh")
            with self.assertRaisesRegex(RuntimeError, "leased"):
                stream.acquire_next()
            first.release()
            self.assertEqual(first_view.tobytes(), b"\0\0\0\0")
            third = stream.acquire_next()
            self.assertEqual(third.read(), b"ij")
            with self.assertRaisesRegex(ValueError, "lease"):
                first.read()
            second.release()
            third.release()
            self.assertIsNone(stream.acquire_next())
            self.assertEqual(stream.snapshot()["in_flight_tiles"], 0)
        self.assertTrue(stream.snapshot()["closed"])

    def test_cancellation_invalidates_leases_and_zeroizes_buffers(self):
        source = b"abcdefgh"
        cancellation = threading.Event()
        stream = NumericDoubleBufferStream(self.plan(source), source, cancellation=cancellation)
        lease = stream.acquire_next()
        cancellation.set()
        with self.assertRaises(NumericStreamingCancelled):
            stream.poll_cancellation()
        with self.assertRaisesRegex(ValueError, "lease"):
            lease.read()
        snapshot = stream.snapshot()
        self.assertTrue(snapshot["cancelled"])
        self.assertTrue(snapshot["closed"])
        self.assertEqual(snapshot["in_flight_tiles"], 0)
        self.assertFalse(snapshot["stores_tensor_values"])

    def test_plan_rejects_unbounded_or_misaligned_configuration(self):
        source = b"abcdefgh"
        digest = hashlib.sha256(source).hexdigest()
        for values in (
            (digest, 8, 9, 2, 1),
            (digest, 8, 4, 3, 1),
            (digest, 8, 3, 2, 2),
            (digest, 8, MAX_NUMERIC_TILE_BYTES + 1, 2, 1),
            ("bad", 8, 4, 2, 1),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                NumericStreamingPlan(*values)
        with self.assertRaisesRegex(ValueError, "digest"):
            NumericDoubleBufferStream(self.plan(source), b"abcdxxxx")

    def test_memory_admission_counts_two_buffers_metadata_scratch_and_destination(self):
        source = b"abcdefghijkl"
        plan = self.plan(source, tile_bytes=4)
        admission = Qwen4MemoryAdmission(32)
        reservation = admission.reserve_numeric_stream(
            "numeric:tensor",
            {"component": "numeric_compatibility", "shape": [3]},
            target_dtype="F32",
            stream_plan=plan,
            metadata_bytes=2,
            scratch_bytes=3,
        )
        self.assertEqual(reservation.source_stream_bytes, 10)
        self.assertEqual(reservation.destination_bytes, 12)
        self.assertEqual(reservation.scratch_bytes, 3)
        self.assertEqual(reservation.reserved_bytes, 25)
        self.assertEqual(reservation.stream_tile_bytes, 4)
        self.assertEqual(reservation.stream_buffer_count, 2)
        with self.assertRaisesRegex(MemoryError, "admission"):
            Qwen4MemoryAdmission(24).reserve_numeric_stream(
                "numeric:tensor",
                {"component": "numeric_compatibility", "shape": [3]},
                target_dtype="F32",
                stream_plan=plan,
                metadata_bytes=2,
                scratch_bytes=3,
            )
        retained = admission.retain_destination(reservation)
        self.assertEqual(retained.reserved_bytes, 12)
        self.assertEqual(retained.stream_tile_bytes, 0)
        self.assertEqual(retained.stream_buffer_count, 0)
        admission.release(retained)


if __name__ == "__main__":
    unittest.main()
