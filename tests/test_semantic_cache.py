import json
import threading
import unittest
from pathlib import Path

from tests.schema_validator import validate_instance
from vllm_apple.semantic_cache import (
    SemanticAnchor,
    SemanticAnchorCache,
    SemanticAnchorKind,
    semantic_prefix_fingerprint,
)
from vllm_apple.semantic_state import BackendStateReference, SemanticStateCoordinator
from vllm_apple.scheduler import ScheduleRequest
from tests.test_scheduler import execution_plan
from vllm_apple.service import RuntimeService
from vllm_apple.types import MemoryPressure


SESSION = "a" * 64


def anchor(
    tokens: tuple[int, ...],
    handle: str,
    *,
    state_bytes: int = 4,
    kind: SemanticAnchorKind = SemanticAnchorKind.TURN,
) -> SemanticAnchor:
    return SemanticAnchor(
        session_fingerprint=SESSION,
        prefix_fingerprint=semantic_prefix_fingerprint(tokens),
        token_position=len(tokens),
        kind=kind,
        state_handle=handle,
        state_bytes=state_bytes,
    )


class SemanticAnchorCacheTests(unittest.TestCase):
    def test_deepest_valid_prefix_is_reused_without_storing_prompt(self) -> None:
        cache = SemanticAnchorCache(capacity_entries=4, capacity_bytes=32)
        first = anchor((1, 2), "state-turn")
        second = anchor(
            (1, 2, 3, 4),
            "state-tool",
            kind=SemanticAnchorKind.TOOL_RESULT,
        )
        self.assertEqual(cache.put(first), ())
        self.assertEqual(cache.put(first), ())
        self.assertEqual(cache.put(second), ())
        reused = cache.deepest_reusable(
            SESSION,
            (
                (2, first.prefix_fingerprint),
                (4, second.prefix_fingerprint),
                (6, semantic_prefix_fingerprint((1, 2, 9, 9, 9, 9))),
            ),
        )
        self.assertEqual(reused, second)
        payload = second.to_dict()
        self.assertNotIn("tokens", payload)
        self.assertNotIn("prompt", payload)
        schema = json.loads(
            Path("schemas/semantic-anchor-v1.schema.json").read_text(encoding="utf-8")
        )
        validate_instance(payload, schema)

    def test_lru_resize_returns_handles_for_bounded_backend_release(self) -> None:
        cache = SemanticAnchorCache(capacity_entries=2, capacity_bytes=8)
        first = anchor((1,), "state-1")
        second = anchor((1, 2), "state-2")
        third = anchor((1, 2, 3), "state-3")
        cache.put(first)
        cache.put(second)
        cache.deepest_reusable(SESSION, ((1, first.prefix_fingerprint),))
        self.assertEqual(cache.put(third), (second,))
        self.assertEqual(cache.snapshot().resident_bytes, 8)
        self.assertEqual(cache.resize(1, 4), (first,))
        self.assertEqual(cache.snapshot().entry_count, 1)
        self.assertEqual(cache.clear(), (third,))
        self.assertEqual(cache.snapshot().resident_bytes, 0)

    def test_concurrent_insertions_never_exceed_hard_limits(self) -> None:
        cache = SemanticAnchorCache(capacity_entries=8, capacity_bytes=32)

        def insert(offset: int) -> None:
            for value in range(offset, offset + 32):
                cache.put(anchor((value + 1,), f"state-{value}"))

        threads = tuple(threading.Thread(target=insert, args=(index * 32,)) for index in range(4))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        snapshot = cache.snapshot()
        self.assertLessEqual(snapshot.entry_count, snapshot.capacity_entries)
        self.assertLessEqual(snapshot.resident_bytes, snapshot.capacity_bytes)
        self.assertGreater(snapshot.evictions, 0)


class RecordingSemanticBackend:
    def __init__(self) -> None:
        self.released: list[str] = []
        self.restorable: set[str] = set()
        self.fail_release = False

    def capture_semantic_state(
        self,
        session_fingerprint: str,
        prefix_fingerprint: str,
        token_position: int,
        kind: SemanticAnchorKind,
    ) -> BackendStateReference:
        handle = f"state-{token_position}"
        self.restorable.add(handle)
        return BackendStateReference(handle, 4)

    def restore_semantic_state(self, handle: str) -> bool:
        return handle in self.restorable

    def release_semantic_state(self, handle: str) -> None:
        if self.fail_release:
            raise RuntimeError("temporary release failure")
        self.released.append(handle)
        self.restorable.discard(handle)


class SemanticStateCoordinatorTests(unittest.TestCase):
    def test_failed_cache_admission_releases_new_backend_state(self) -> None:
        backend = RecordingSemanticBackend()
        coordinator = SemanticStateCoordinator(
            backend,
            SemanticAnchorCache(capacity_entries=1, capacity_bytes=3),
        )
        prefix = semantic_prefix_fingerprint((7,))
        with self.assertRaises(ValueError):
            coordinator.capture(SESSION, prefix, 1, SemanticAnchorKind.TURN)
        self.assertEqual(backend.released, ["state-1"])
        self.assertEqual(coordinator.snapshot()["entry_count"], 0)

    def test_service_capture_restore_resize_and_metrics(self) -> None:
        backend = RecordingSemanticBackend()
        events: list[tuple[str, dict[str, object]]] = []
        coordinator = SemanticStateCoordinator(
            backend,
            SemanticAnchorCache(capacity_entries=2, capacity_bytes=8),
            lambda name, payload: events.append((name, payload)),
        )
        service = RuntimeService(semantic_state=coordinator)
        first_prefix = semantic_prefix_fingerprint((1, 2))
        second_prefix = semantic_prefix_fingerprint((1, 2, 3, 4))
        service.capture_semantic_state(
            SESSION,
            first_prefix,
            2,
            SemanticAnchorKind.TURN,
        )
        service.capture_semantic_state(
            SESSION,
            second_prefix,
            4,
            SemanticAnchorKind.TOOL_RESULT,
        )
        restored = service.restore_semantic_state(
            SESSION,
            ((2, first_prefix), (4, second_prefix)),
        )
        self.assertTrue(restored.reused)
        self.assertEqual(restored.token_position, 4)
        self.assertEqual(service.resize_semantic_cache(1, 4), 1)
        self.assertEqual(backend.released, ["state-2"])
        metrics = service.snapshot().semantic_cache
        self.assertTrue(metrics["enabled"])
        self.assertEqual(metrics["captures"], 2)
        self.assertEqual(metrics["hits"], 1)
        self.assertEqual(events[-1][0], "semantic_cache.hit")

    def test_failed_restore_discards_stale_state_and_release_retries(self) -> None:
        backend = RecordingSemanticBackend()
        coordinator = SemanticStateCoordinator(
            backend,
            SemanticAnchorCache(capacity_entries=1, capacity_bytes=4),
        )
        prefix = semantic_prefix_fingerprint((9,))
        coordinator.capture(SESSION, prefix, 1, SemanticAnchorKind.THINKING)
        backend.restorable.clear()
        backend.fail_release = True
        result = coordinator.restore_deepest(SESSION, ((1, prefix),))
        self.assertFalse(result.reused)
        self.assertEqual(coordinator.snapshot()["pending_releases"], 1)
        backend.fail_release = False
        self.assertEqual(coordinator.retry_pending_releases(), 1)
        metrics = coordinator.snapshot()
        self.assertEqual(metrics["restore_failures"], 1)
        self.assertEqual(metrics["pending_releases"], 0)


class ElasticMemoryControllerTests(unittest.TestCase):
    def test_pressure_resizes_cache_and_normal_restores_capacity(self) -> None:
        backend = RecordingSemanticBackend()
        coordinator = SemanticStateCoordinator(
            backend,
            SemanticAnchorCache(capacity_entries=8, capacity_bytes=32),
        )
        service = RuntimeService(semantic_state=coordinator)
        for position in range(1, 9):
            service.capture_semantic_state(
                SESSION,
                semantic_prefix_fingerprint(tuple(range(1, position + 1))),
                position,
                SemanticAnchorKind.TURN,
            )
        warning = service.apply_memory_pressure(MemoryPressure.WARNING)
        self.assertIsNotNone(warning)
        self.assertEqual(warning.status, "applied")
        self.assertEqual(warning.target_entries, 4)
        self.assertEqual(warning.evicted_entries, 4)
        critical = service.apply_memory_pressure(MemoryPressure.CRITICAL)
        self.assertEqual(critical.target_entries, 1)
        self.assertEqual(coordinator.snapshot()["entry_count"], 1)
        normal = service.apply_memory_pressure(MemoryPressure.NORMAL)
        self.assertEqual(normal.target_entries, 8)
        metrics = service.snapshot().elastic_memory
        self.assertEqual(metrics["current_pressure"], "normal")
        self.assertEqual(metrics["current_capacity_entries"], 8)
        self.assertEqual(metrics["adjustments"], 3)

    def test_pressure_change_is_deferred_until_scheduler_safe_point(self) -> None:
        backend = RecordingSemanticBackend()
        coordinator = SemanticStateCoordinator(
            backend,
            SemanticAnchorCache(capacity_entries=8, capacity_bytes=32),
        )
        service = RuntimeService(semantic_state=coordinator)
        reservation = service.scheduler.admit(ScheduleRequest("decode", 1))
        deferred = service.apply_memory_pressure(MemoryPressure.CRITICAL)
        self.assertEqual(deferred.status, "deferred")
        self.assertEqual(coordinator.snapshot()["capacity_entries"], 8)
        service.scheduler.complete(reservation)
        applied = service.apply_pending_memory_pressure()
        self.assertIsNotNone(applied)
        self.assertEqual(applied.status, "applied")
        self.assertEqual(coordinator.snapshot()["capacity_entries"], 1)

    def test_service_applies_cache_and_plan_together_after_last_reservation(self) -> None:
        backend = RecordingSemanticBackend()
        coordinator = SemanticStateCoordinator(
            backend,
            SemanticAnchorCache(capacity_entries=8, capacity_bytes=32),
        )
        service = RuntimeService(semantic_state=coordinator)
        initial = execution_plan("d" * 24, 4)
        constrained = execution_plan("e" * 24, 1)
        self.assertEqual(service.request_execution_plan(initial).status, "applied")
        reservation = service.admit_schedule(ScheduleRequest("decode", 1))
        self.assertEqual(
            service.apply_memory_pressure(MemoryPressure.CRITICAL).status, "deferred"
        )
        self.assertEqual(service.request_execution_plan(constrained).status, "deferred")
        plan, elastic = service.complete_schedule(reservation)
        self.assertEqual(elastic.status, "applied")
        self.assertEqual(plan.status, "applied")
        snapshot = service.snapshot()
        self.assertEqual(snapshot.elastic_memory["current_capacity_entries"], 1)
        self.assertEqual(snapshot.execution_plan["active_plan_id"], constrained.plan_id)


if __name__ == "__main__":
    unittest.main()
