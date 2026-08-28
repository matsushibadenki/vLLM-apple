import threading
import unittest

from vllm_apple.token_count_cache import TokenCountCache, TokenCountSingleFlight


class TokenCountCacheTests(unittest.TestCase):
    def test_evicts_least_recently_used_entry_at_capacity(self) -> None:
        cache = TokenCountCache(capacity=2)
        cache.put("first", 1)
        cache.put("second", 2)
        self.assertEqual(cache.get("first"), 1)
        cache.put("third", 3)

        self.assertIsNone(cache.get("second"))
        self.assertEqual(cache.get("first"), 1)
        self.assertEqual(cache.get("third"), 3)
        snapshot = cache.snapshot()
        self.assertEqual(snapshot.entries, 2)
        self.assertEqual(snapshot.evictions, 1)

    def test_expires_entries_using_monotonic_ttl(self) -> None:
        now = [10.0]
        cache = TokenCountCache(capacity=2, ttl_seconds=5.0, clock=lambda: now[0])
        cache.put("fingerprint", 42)
        now[0] = 14.999
        self.assertEqual(cache.get("fingerprint"), 42)
        now[0] = 15.0
        self.assertIsNone(cache.get("fingerprint"))

        snapshot = cache.snapshot()
        self.assertEqual(snapshot.entries, 0)
        self.assertEqual(snapshot.hits, 1)
        self.assertEqual(snapshot.misses, 1)
        self.assertEqual(snapshot.expirations, 1)

    def test_rejects_unbounded_or_invalid_entries(self) -> None:
        with self.assertRaises(ValueError):
            TokenCountCache(capacity=0)
        cache = TokenCountCache()
        with self.assertRaises(ValueError):
            cache.put("", 1)
        with self.assertRaises(ValueError):
            cache.put("fingerprint", -1)


class TokenCountSingleFlightTests(unittest.TestCase):
    def test_follower_receives_leader_result_and_entry_is_released(self) -> None:
        coordinator = TokenCountSingleFlight(capacity=2)
        leader = coordinator.join("same")
        follower = coordinator.join("same")
        result: list[int | None] = []
        thread = threading.Thread(target=lambda: result.append(coordinator.wait(follower, 1.0)))
        thread.start()
        coordinator.complete(leader, 23)
        thread.join(timeout=1.0)

        self.assertEqual(result, [23])
        snapshot = coordinator.snapshot()
        self.assertEqual(snapshot.active, 0)
        self.assertEqual(snapshot.leaders, 1)
        self.assertEqual(snapshot.followers, 1)

    def test_capacity_bypasses_distinct_work_without_tracking_it(self) -> None:
        coordinator = TokenCountSingleFlight(capacity=1)
        leader = coordinator.join("first")
        bypass = coordinator.join("second")
        self.assertTrue(bypass.leader)
        self.assertIsNone(bypass.state)
        self.assertEqual(coordinator.snapshot().bypasses, 1)
        coordinator.complete(bypass, 2)
        coordinator.complete(leader, 1)
        self.assertEqual(coordinator.snapshot().active, 0)

    def test_follower_timeout_is_bounded_and_counted(self) -> None:
        coordinator = TokenCountSingleFlight()
        leader = coordinator.join("same")
        follower = coordinator.join("same")
        self.assertIsNone(coordinator.wait(follower, 0.001))
        self.assertEqual(coordinator.snapshot().timeouts, 1)
        coordinator.complete(leader, None)


if __name__ == "__main__":
    unittest.main()
