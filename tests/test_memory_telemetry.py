import unittest

from vllm_apple.memory_telemetry import UnifiedMemoryTelemetry
from vllm_apple.types import MemoryPressure


class UnifiedMemoryTelemetryTests(unittest.TestCase):
    def test_layers_remain_unknown_until_measured_and_peaks_are_monotonic(self) -> None:
        telemetry = UnifiedMemoryTelemetry(1000, 600, "fixture")
        initial = telemetry.snapshot()
        self.assertIsNone(initial.allocator_current_bytes)
        self.assertIsNone(initial.kv_used_bytes)
        self.assertIsNone(initial.allocator_resident_peak_delta_bytes)

        telemetry.update_os(
            available_bytes=500,
            control_resident_bytes=100,
            backend_resident_bytes=450,
        )
        telemetry.update_allocator(300, source="mlx")
        telemetry.update_kv_cache(40, 100, source="vllm")
        telemetry.update_os(
            available_bytes=550,
            control_resident_bytes=90,
            backend_resident_bytes=400,
        )
        telemetry.update_allocator(250, source="mlx")
        snapshot = telemetry.snapshot()
        self.assertEqual(snapshot.backend_resident_peak_bytes, 450)
        self.assertEqual(snapshot.allocator_peak_bytes, 300)
        self.assertEqual(snapshot.allocator_resident_peak_delta_bytes, 150)
        self.assertEqual(snapshot.kv_usage_ratio, 0.4)
        telemetry.update_iogpu(425, source="ioreg")
        self.assertEqual(telemetry.snapshot().iogpu_peak_bytes, 425)

    def test_invalid_samples_are_rejected_without_mutating_snapshot(self) -> None:
        telemetry = UnifiedMemoryTelemetry(1000, 600, "fixture")
        before = telemetry.snapshot()
        with self.assertRaises(ValueError):
            telemetry.update_kv_cache(101, 100, source="vllm")
        with self.assertRaises(ValueError):
            telemetry.update_os(available_bytes=1001, control_resident_bytes=1)
        self.assertEqual(telemetry.snapshot(), before)

    def test_ratio_only_sample_does_not_invent_kv_byte_capacity(self) -> None:
        telemetry = UnifiedMemoryTelemetry(1000, 600, "fixture")
        telemetry.update_kv_ratio(0.25, source="vllm-prometheus")
        snapshot = telemetry.snapshot()
        self.assertEqual(snapshot.kv_usage_ratio, 0.25)
        self.assertIsNone(snapshot.kv_used_bytes)
        self.assertIsNone(snapshot.kv_capacity_bytes)

    def test_pressure_transition_count_ignores_duplicate_level(self) -> None:
        telemetry = UnifiedMemoryTelemetry(1000, 600, "fixture", MemoryPressure.NORMAL)
        telemetry.update_pressure(MemoryPressure.WARNING)
        telemetry.update_pressure(MemoryPressure.WARNING)
        self.assertEqual(telemetry.snapshot().pressure, "warning")
        self.assertEqual(telemetry.snapshot().pressure_notifications, 1)


if __name__ == "__main__":
    unittest.main()
