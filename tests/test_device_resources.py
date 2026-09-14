import unittest

from vllm_apple.device_resources import (
    DeviceResourceCapacityError,
    DeviceResourceRequest,
    UnifiedDeviceResourceLedger,
)
from vllm_apple.execution import ExecutionBackend


class UnifiedDeviceResourceLedgerTests(unittest.TestCase):
    def setUp(self):
        self.ledger = UnifiedDeviceResourceLedger(
            unified_memory_bytes=100,
            cpu_threads=2,
            gpu_command_queues=1,
            ane_tasks=1,
            bandwidth_slots=2,
        )

    def test_reserves_and_releases_all_dimensions_atomically(self):
        request = DeviceResourceRequest.for_backend(ExecutionBackend.COREML_DRAFT, 60)
        reservation = self.ledger.reserve(request)
        snapshot = self.ledger.snapshot()
        self.assertEqual(snapshot["used"]["unified_memory_bytes"], 60)
        self.assertEqual(snapshot["used"]["ane_tasks"], 1)
        self.assertEqual(snapshot["used"]["bandwidth_slots"], 1)
        self.assertTrue(self.ledger.release(reservation.reservation_id))
        self.assertEqual(self.ledger.snapshot()["active_reservations"], 0)
        self.assertFalse(self.ledger.release(reservation.reservation_id))

    def test_exhaustion_does_not_partially_consume_other_resources(self):
        first = self.ledger.reserve(
            DeviceResourceRequest.for_backend(ExecutionBackend.NATIVE_METAL, 40)
        )
        before = self.ledger.snapshot()
        with self.assertRaises(DeviceResourceCapacityError):
            self.ledger.reserve(
                DeviceResourceRequest.for_backend(ExecutionBackend.NATIVE_MLX, 10)
            )
        self.assertEqual(self.ledger.snapshot(), before)
        self.ledger.release(first.reservation_id)

    def test_non_apple_zero_accelerator_capacity_fails_closed(self):
        ledger = UnifiedDeviceResourceLedger(
            unified_memory_bytes=100, cpu_threads=1,
            gpu_command_queues=0, ane_tasks=0, bandwidth_slots=1,
        )
        with self.assertRaisesRegex(DeviceResourceCapacityError, "ane_tasks"):
            ledger.reserve(
                DeviceResourceRequest.for_backend(ExecutionBackend.COREML_DRAFT, 1)
            )
