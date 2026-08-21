import unittest

from vllm_apple.scheduler import BasicScheduler, MemoryCapacityError, ScheduleRequest
from vllm_apple.types import Backend, HardwareInfo, MemoryInfo, Priority


def hardware(apple: bool = True) -> HardwareInfo:
    return HardwareInfo(
        platform="Darwin" if apple else "Linux",
        architecture="arm64",
        soc="Test",
        physical_cpu_count=8,
        logical_cpu_count=8,
        gpu_core_count=10,
        memory=MemoryInfo(total_bytes=1_000, available_bytes=800),
        is_apple_silicon=apple,
        os_version="test",
    )


class SchedulerTests(unittest.TestCase):
    def test_backend_choice_avoids_launch_overhead_for_tiny_decode(self) -> None:
        scheduler = BasicScheduler(hardware(), 500)
        self.assertEqual(scheduler.choose_backend(ScheduleRequest("gemv", 10)), Backend.CPU)
        self.assertEqual(
            scheduler.choose_backend(ScheduleRequest("gemm", 10, batch_size=8)), Backend.MLX_GPU
        )
        self.assertEqual(
            scheduler.choose_backend(ScheduleRequest("paged_attention", 10)), Backend.METAL
        )

    def test_reservations_never_exceed_capacity(self) -> None:
        scheduler = BasicScheduler(hardware(), 100)
        reservation = scheduler.admit(
            ScheduleRequest("attention", 80, priority=Priority.INTERACTIVE)
        )
        with self.assertRaises(MemoryCapacityError):
            scheduler.admit(ScheduleRequest("attention", 21))
        self.assertEqual(scheduler.memory.reserved_bytes, 80)
        scheduler.complete(reservation)
        self.assertEqual(scheduler.memory.reserved_bytes, 0)
        scheduler.complete(reservation)
        self.assertEqual(scheduler.memory.reserved_bytes, 0)


if __name__ == "__main__":
    unittest.main()

