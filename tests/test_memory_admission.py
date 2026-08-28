import unittest

from vllm_apple.memory_admission import (
    MemoryPressureAdmissionError,
    MemoryPressureAdmissionGate,
)
from vllm_apple.memory_telemetry import UnifiedMemoryTelemetry
from vllm_apple.scheduler import ScheduleRequest
from vllm_apple.types import MemoryPressure, Priority


class MemoryPressureAdmissionGateTests(unittest.TestCase):
    def memory(
        self,
        *,
        available: int = 500,
        pressure: MemoryPressure = MemoryPressure.NORMAL,
        backend: int | None = None,
        iogpu: int | None = None,
    ):
        telemetry = UnifiedMemoryTelemetry(1000, available, "fixture", pressure)
        if backend is not None:
            telemetry.update_backend_resident(backend)
        if iogpu is not None:
            telemetry.update_iogpu(iogpu, source="fixture")
        return telemetry.snapshot()

    def test_critical_rejects_normal_but_preserves_interactive_escape_hatch(self) -> None:
        gate = MemoryPressureAdmissionGate()
        with self.assertRaises(MemoryPressureAdmissionError):
            gate.admit(ScheduleRequest("decode", 0), self.memory(pressure=MemoryPressure.CRITICAL))
        gate.admit(
            ScheduleRequest("decode", 0, priority=Priority.INTERACTIVE),
            self.memory(pressure=MemoryPressure.CRITICAL),
        )
        self.assertEqual(gate.snapshot().rejected, 1)
        self.assertEqual(gate.snapshot().admitted, 1)

    def test_warning_rejects_only_background_and_normal_clears_reason(self) -> None:
        gate = MemoryPressureAdmissionGate()
        warning = self.memory(available=100)
        with self.assertRaises(MemoryPressureAdmissionError):
            gate.admit(
                ScheduleRequest("decode", 0, priority=Priority.BACKGROUND), warning
            )
        gate.refresh(self.memory())
        self.assertIsNone(gate.snapshot().last_rejection_reason)

    def test_rss_and_iogpu_are_views_not_additive_allocations(self) -> None:
        pressure = MemoryPressureAdmissionGate.effective_pressure(
            self.memory(backend=500, iogpu=500)
        )
        self.assertEqual(pressure, MemoryPressure.NORMAL)
        self.assertEqual(
            MemoryPressureAdmissionGate.effective_pressure(self.memory(iogpu=930)),
            MemoryPressure.CRITICAL,
        )

    def test_recovery_stages_ramp_batch_context_and_memory(self) -> None:
        now = [100.0]
        gate = MemoryPressureAdmissionGate(clock=lambda: now[0])
        critical = self.memory(pressure=MemoryPressure.CRITICAL)
        normal = self.memory()
        gate.refresh(critical)
        gate.refresh(normal)
        self.assertEqual(gate.snapshot().recovery_stage, 0)
        with self.assertRaisesRegex(MemoryPressureAdmissionError, "recovery_batch_limited"):
            gate.admit(ScheduleRequest("decode", 0, batch_size=2), normal)
        with self.assertRaisesRegex(MemoryPressureAdmissionError, "recovery_memory_limited"):
            gate.admit(ScheduleRequest("decode", 63), normal)

        now[0] += 6
        gate.admit(ScheduleRequest("decode", 0, batch_size=2), normal)
        with self.assertRaisesRegex(MemoryPressureAdmissionError, "recovery_context_limited"):
            gate.admit(ScheduleRequest("decode", 0, estimated_context_tokens=8193), normal)

        now[0] += 25
        self.assertIsNone(gate.snapshot().recovery_stage)
        gate.admit(
            ScheduleRequest("decode", 400, batch_size=32, estimated_context_tokens=100_000),
            normal,
        )

    def test_new_pressure_cancels_recovery_immediately(self) -> None:
        now = [0.0]
        gate = MemoryPressureAdmissionGate(clock=lambda: now[0])
        gate.refresh(self.memory(pressure=MemoryPressure.CRITICAL))
        gate.refresh(self.memory())
        self.assertEqual(gate.snapshot().recovery_stage, 0)
        gate.refresh(self.memory(pressure=MemoryPressure.WARNING))
        self.assertIsNone(gate.snapshot().recovery_stage)


if __name__ == "__main__":
    unittest.main()
