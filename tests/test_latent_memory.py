import unittest

from vllm_apple.latent_memory import LatentMemoryManager, LatentShape
from vllm_apple.unified_memory_arena import UnifiedMemoryArena


class LatentMemoryTests(unittest.TestCase):
    def test_reuses_and_zeroes_matching_idle_buffer(self):
        arena = UnifiedMemoryArena(8192)
        manager = LatentMemoryManager(arena, maximum_buffers=2)
        shape = LatentShape((1, 4, 4, 4), 2)
        lease = manager.acquire(shape)
        lease.view[:] = bytes([9]) * shape.byte_count
        lease.release()
        reused = manager.acquire(shape)
        self.assertEqual(bytes(reused.view), bytes(shape.byte_count))
        self.assertEqual(manager.snapshot()["hits"], 1)
        reused.release()
        manager.close()
        arena.close()

    def test_lru_evicts_idle_but_never_active_buffer(self):
        arena = UnifiedMemoryArena(8192)
        manager = LatentMemoryManager(arena, maximum_buffers=1)
        first = manager.acquire(LatentShape((1024,), 4))
        with self.assertRaisesRegex(ValueError, "pinned"):
            manager.acquire(LatentShape((2048,), 2))
        first.release()
        second = manager.acquire(LatentShape((2048,), 2))
        self.assertEqual(manager.snapshot()["evictions"], 1)
        second.release()
        manager.close()
        arena.close()

    def test_invalid_shape_and_active_close_fail_closed(self):
        with self.assertRaises(ValueError):
            LatentShape((0,), 2)
        arena = UnifiedMemoryArena(4096)
        manager = LatentMemoryManager(arena)
        lease = manager.acquire(LatentShape((1,), 1))
        with self.assertRaisesRegex(RuntimeError, "active"):
            manager.close()
        lease.release()
        manager.close()
        arena.close()


if __name__ == "__main__":
    unittest.main()
