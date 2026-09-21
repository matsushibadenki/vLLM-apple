import unittest

from vllm_apple.adaptive_state_allocation import (
    AdaptiveStateAllocator,
    AdaptiveStateCoordinator,
    AdaptiveStateKind,
    AdaptiveStateRecord,
)
from vllm_apple.scheduler import ScheduleRequest
from vllm_apple.service import RuntimeService
from vllm_apple.types import MemoryPressure


def record(identifier, age, *, precision="fp16", promoted=("fp16", "int8"), pinned=False):
    return AdaptiveStateRecord(
        identifier, AdaptiveStateKind.KV, 200, age, precision, promoted, pinned
    )


class AdaptiveStateAllocationTests(unittest.TestCase):
    def test_warning_reprecisions_oldest_promoted_state_to_budget(self):
        plan = AdaptiveStateAllocator().plan(
            (record("new", 1), record("old", 100)), MemoryPressure.WARNING
        )
        self.assertEqual(plan.source_bytes, 400)
        self.assertEqual(plan.target_bytes, 300)
        self.assertEqual(plan.bytes_released, 100)
        self.assertEqual(
            [(action.state_id, action.action) for action in plan.actions],
            [("new", "retain"), ("old", "reprecision")],
        )

    def test_critical_evicts_oldest_when_no_lower_precision_is_promoted(self):
        plan = AdaptiveStateAllocator().plan(
            (record("old", 100, promoted=("fp16",)), record("new", 1, promoted=("fp16",))),
            MemoryPressure.CRITICAL,
        )
        self.assertEqual(plan.target_bytes, 200)
        self.assertEqual(plan.actions[0].action, "evict")
        self.assertEqual(plan.actions[1].action, "retain")

    def test_pinned_state_is_never_reprecisioned_or_evicted(self):
        plan = AdaptiveStateAllocator().plan(
            (record("pinned", 100, precision="fp32", promoted=("fp32", "int8"), pinned=True),),
            MemoryPressure.CRITICAL,
        )
        self.assertEqual(plan.actions[0].action, "retain")
        self.assertEqual(plan.target_bytes, plan.source_bytes)

    def test_unknown_pressure_is_fail_soft_normal(self):
        plan = AdaptiveStateAllocator().plan(
            (record("state", 10),), MemoryPressure.UNKNOWN
        )
        self.assertEqual(plan.pressure, MemoryPressure.NORMAL)
        self.assertEqual(plan.actions[0].action, "retain")

    def test_duplicate_identity_and_unpromoted_precision_fail_closed(self):
        allocator = AdaptiveStateAllocator()
        item = record("same", 1)
        with self.assertRaises(ValueError):
            allocator.plan((item, item), MemoryPressure.WARNING)
        with self.assertRaises(ValueError):
            AdaptiveStateRecord(
                "bad", AdaptiveStateKind.KV, 1, 0, "fp16", ("int8",)
            )


class Transaction:
    def __init__(self, backend, plan, *, fail=False):
        self.backend = backend
        self.plan = plan
        self.fail = fail

    def commit(self):
        self.backend.calls.append(("commit", self.plan.target_bytes))
        if self.fail:
            raise RuntimeError("commit failed")

    def rollback(self):
        self.backend.calls.append(("rollback", self.plan.source_bytes))


class Backend:
    def __init__(self, *, fail=False):
        self.calls = []
        self.fail = fail

    def adaptive_state_records(self):
        return (record("old", 100), record("new", 1))

    def begin_adaptive_state(self, plan):
        self.calls.append(("begin", plan.source_bytes))
        return Transaction(self, plan, fail=self.fail)


class AdaptiveStateCoordinatorTests(unittest.TestCase):
    def test_commit_failure_rolls_back_and_is_observable(self):
        backend = Backend(fail=True)
        coordinator = AdaptiveStateCoordinator(backend)
        with self.assertRaisesRegex(RuntimeError, "commit failed"):
            coordinator.request(MemoryPressure.WARNING, safe_to_apply=True)
        self.assertEqual([call[0] for call in backend.calls], ["begin", "commit", "rollback"])
        self.assertEqual(coordinator.snapshot()["adaptive_state_rollbacks"], 1)

    def test_runtime_service_defers_until_scheduler_safe_point(self):
        backend = Backend()
        service = RuntimeService(adaptive_state_backend=backend)
        reservation = service.admit_schedule(ScheduleRequest("decode", 1))
        self.assertIsNone(service.apply_memory_pressure(MemoryPressure.WARNING))
        self.assertEqual(backend.calls, [])
        service.complete_schedule(reservation)
        self.assertEqual([call[0] for call in backend.calls], ["begin", "commit"])
        snapshot = service.snapshot().elastic_memory
        self.assertTrue(snapshot["adaptive_state_enabled"])
        self.assertEqual(snapshot["adaptive_state_applied"], 1)
        self.assertEqual(snapshot["adaptive_state_last_released_bytes"], 100)


if __name__ == "__main__":
    unittest.main()
