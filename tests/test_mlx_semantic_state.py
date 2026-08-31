import unittest

from vllm_apple.mlx_semantic_state import MLXPromptCacheStateAdapter
from vllm_apple.semantic_cache import SemanticAnchorKind
from vllm_apple.semantic_state import SemanticStateCoordinator
from vllm_apple.semantic_cache import SemanticAnchorCache


SESSION = "a" * 64
PREFIX = "b" * 64


class FakeArray:
    def __init__(self, size: int) -> None:
        self.nbytes = size


class MLXPromptCacheStateAdapterTests(unittest.TestCase):
    def adapter(self, *, capacity_entries: int = 2, capacity_bytes: int = 32):
        captured = []
        restored = []
        released = []

        def capture():
            snapshot = [FakeArray(8), FakeArray(4)]
            captured.append(snapshot)
            return snapshot

        adapter = MLXPromptCacheStateAdapter(
            capture,
            lambda snapshot: restored.append(snapshot) is None,
            lambda snapshot: released.append(snapshot),
            capacity_entries=capacity_entries,
            capacity_bytes=capacity_bytes,
        )
        return adapter, captured, restored, released

    def test_captures_restores_and_releases_opaque_mlx_state(self) -> None:
        adapter, captured, restored, released = self.adapter()
        reference = adapter.capture_semantic_state(
            SESSION, PREFIX, 2, SemanticAnchorKind.TURN
        )
        self.assertIsNotNone(reference)
        assert reference is not None
        self.assertEqual(reference.state_bytes, 12)
        self.assertTrue(reference.handle.startswith("mlx-"))
        self.assertTrue(adapter.restore_semantic_state(reference.handle))
        self.assertIs(restored[0], captured[0])
        adapter.release_semantic_state(reference.handle)
        self.assertIs(released[0], captured[0])
        self.assertFalse(adapter.restore_semantic_state(reference.handle))
        self.assertEqual(adapter.snapshot()["resident_bytes"], 0)

    def test_refuses_capture_before_exceeding_entry_or_byte_budget(self) -> None:
        adapter, captured, _, released = self.adapter(
            capacity_entries=1, capacity_bytes=12
        )
        first = adapter.capture_semantic_state(
            SESSION, PREFIX, 1, SemanticAnchorKind.TURN
        )
        second = adapter.capture_semantic_state(
            SESSION, "c" * 64, 2, SemanticAnchorKind.TURN
        )
        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertEqual(len(captured), 1)
        self.assertEqual(released, [])

        oversized, captured, _, released = self.adapter(capacity_bytes=11)
        self.assertIsNone(
            oversized.capture_semantic_state(
                SESSION, PREFIX, 1, SemanticAnchorKind.TURN
            )
        )
        self.assertEqual(released, captured)
        self.assertEqual(oversized.snapshot()["entry_count"], 0)

    def test_release_failure_preserves_ownership_for_coordinator_retry(self) -> None:
        snapshots = []
        fail = [True]

        def release(snapshot):
            if fail[0]:
                raise RuntimeError("busy")
            snapshots.append(snapshot)

        adapter = MLXPromptCacheStateAdapter(
            lambda: [FakeArray(4)],
            lambda _snapshot: True,
            release,
            capacity_entries=1,
            capacity_bytes=4,
        )
        coordinator = SemanticStateCoordinator(
            adapter, SemanticAnchorCache(capacity_entries=1, capacity_bytes=4)
        )
        coordinator.capture(SESSION, PREFIX, 1, SemanticAnchorKind.TURN)
        coordinator.close()
        self.assertEqual(coordinator.snapshot()["pending_releases"], 1)
        self.assertEqual(adapter.snapshot()["entry_count"], 1)
        fail[0] = False
        self.assertEqual(coordinator.retry_pending_releases(), 1)
        self.assertEqual(adapter.snapshot()["entry_count"], 0)
        self.assertEqual(len(snapshots), 1)

    def test_invalid_metadata_and_handles_fail_closed(self) -> None:
        adapter, _, _, _ = self.adapter()
        with self.assertRaises(ValueError):
            adapter.capture_semantic_state("bad", PREFIX, 1, SemanticAnchorKind.TURN)
        with self.assertRaises(ValueError):
            adapter.restore_semantic_state("mlx-not-a-handle")


if __name__ == "__main__":
    unittest.main()
