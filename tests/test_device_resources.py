import unittest

from vllm_apple.device_resources import (
    BandwidthContentionEvidence,
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
            contention_profile_id="profile",
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

    def test_transfer_replaces_backend_resources_atomically(self):
        reservation = self.ledger.reserve(
            DeviceResourceRequest.for_backend(ExecutionBackend.COREML_DRAFT, 60)
        )
        transferred = self.ledger.transfer(
            reservation.reservation_id,
            DeviceResourceRequest.for_backend(ExecutionBackend.NATIVE_MLX, 60),
        )
        self.assertEqual(transferred.reservation_id, reservation.reservation_id)
        used = self.ledger.snapshot()["used"]
        self.assertEqual(used["ane_tasks"], 0)
        self.assertEqual(used["gpu_command_queues"], 1)
        before = self.ledger.snapshot()
        with self.assertRaises(DeviceResourceCapacityError):
            self.ledger.transfer(
                reservation.reservation_id,
                DeviceResourceRequest(
                    ExecutionBackend.CPU, 101, cpu_threads=1,
                ),
            )
        self.assertEqual(self.ledger.snapshot(), before)

    def test_cross_device_parallelism_requires_measured_improvement(self):
        first = self.ledger.reserve(
            DeviceResourceRequest.for_backend(ExecutionBackend.NATIVE_MLX, 10)
        )
        with self.assertRaisesRegex(
            DeviceResourceCapacityError, "bandwidth_contention_unqualified"
        ):
            self.ledger.reserve(
                DeviceResourceRequest.for_backend(ExecutionBackend.CPU, 10)
            )
        rejected = BandwidthContentionEvidence(
            "profile", ExecutionBackend.NATIVE_MLX, ExecutionBackend.CPU,
            100, 99, 3, True,
        )
        self.assertFalse(self.ledger.install_contention_evidence(rejected))
        with self.assertRaisesRegex(ValueError, "profile mismatch"):
            self.ledger.install_contention_evidence(BandwidthContentionEvidence(
                "other", ExecutionBackend.NATIVE_MLX, ExecutionBackend.CPU,
                100, 90, 3, True,
            ))
        qualified = BandwidthContentionEvidence(
            "profile", ExecutionBackend.NATIVE_MLX, ExecutionBackend.CPU,
            100, 95, 3, True,
        )
        self.assertTrue(self.ledger.install_contention_evidence(qualified))
        second = self.ledger.reserve(
            DeviceResourceRequest.for_backend(ExecutionBackend.CPU, 10)
        )
        self.ledger.release(second.reservation_id)
        self.ledger.release(first.reservation_id)
