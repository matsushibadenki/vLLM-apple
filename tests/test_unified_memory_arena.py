import unittest

from vllm_apple.unified_memory_arena import UnifiedMemoryArena


class UnifiedMemoryArenaTests(unittest.TestCase):
    def test_aligned_zero_copy_suballocation_and_coalescing(self):
        arena = UnifiedMemoryArena(16 * 1024)
        first = arena.allocate(100, alignment=4096)
        second = arena.allocate(200, alignment=4096)
        self.assertEqual((first.offset, second.offset), (0, 4096))
        first.view[:4] = b"test"
        self.assertEqual(bytes(first.view[:4]), b"test")
        first.release()
        second.release()
        snapshot = arena.snapshot()
        self.assertEqual(snapshot.free_bytes, arena.capacity_bytes)
        self.assertEqual(snapshot.largest_free_range_bytes, arena.capacity_bytes)
        self.assertEqual(snapshot.fragmentation_bytes, 0)
        arena.close()

    def test_fragmentation_failure_and_recovery(self):
        arena = UnifiedMemoryArena(12 * 1024, zero_on_release=False)
        leases = [arena.allocate(4096) for _ in range(3)]
        leases[1].release()
        with self.assertRaisesRegex(ValueError, "aligned free range"):
            arena.allocate(5000, alignment=4096)
        self.assertEqual(arena.snapshot().failed_allocations, 1)
        leases[0].release()
        replacement = arena.allocate(5000, alignment=4096)
        replacement.release()
        leases[2].release()
        arena.close()

    def test_active_close_and_stale_release_fail_closed(self):
        arena = UnifiedMemoryArena(4096)
        lease = arena.allocate(1)
        with self.assertRaisesRegex(RuntimeError, "active"):
            arena.close()
        lease.release()
        lease.release()
        arena.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            arena.allocate(1)


if __name__ == "__main__":
    unittest.main()
