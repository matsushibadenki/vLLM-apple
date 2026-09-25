import unittest

from vllm_apple.scheduler import PriorityScheduleQueue, ScheduleQueueFullError, ScheduleRequest
from vllm_apple.types import Priority


class PriorityScheduleQueueCleanupTests(unittest.TestCase):
    def test_cancel_churn_without_consumer_releases_internal_storage(self):
        queue = PriorityScheduleQueue(maximum_requests=2)
        for _ in range(10_000):
            token = queue.enqueue(ScheduleRequest("decode", 1))
            self.assertTrue(queue.cancel(token))
        self.assertEqual(queue.snapshot()["queued"], 0)
        self.assertEqual(len(queue._heap), 0)
        self.assertEqual(len(queue._enqueued_ns), 0)

    def test_cleanup_preserves_priority_and_fifo_behind_live_head(self):
        queue = PriorityScheduleQueue(maximum_requests=5)
        first = queue.enqueue(ScheduleRequest("decode", 1, Priority.NORMAL))
        background = queue.enqueue(ScheduleRequest("decode", 1, Priority.BACKGROUND))
        realtime = queue.enqueue(ScheduleRequest("decode", 1, Priority.REALTIME))
        second = queue.enqueue(ScheduleRequest("decode", 1, Priority.NORMAL))
        for _ in range(2000):
            token = queue.enqueue(ScheduleRequest("decode", 1, Priority.BACKGROUND))
            self.assertTrue(queue.cancel(token))
            self.assertLessEqual(len(queue._heap), 2 * 5 + 64)
        self.assertEqual(queue.peek()[0], realtime)
        for expected in (realtime, first, second, background):
            actual, _ = queue.dequeue(timeout=0)
            self.assertEqual(actual, expected)
            self.assertTrue(queue.finish_claim(actual))
        self.assertIsNone(queue.dequeue(timeout=0))

    def test_compaction_does_not_lose_or_reorder_a_restored_claim(self):
        queue = PriorityScheduleQueue(maximum_requests=3)
        first = queue.enqueue(ScheduleRequest("decode", 1))
        self.assertEqual(queue.claim_head(first)[0], first)
        second = queue.enqueue(ScheduleRequest("decode", 1))
        for _ in range(200):
            token = queue.enqueue(ScheduleRequest("decode", 1))
            with self.assertRaises(ScheduleQueueFullError):
                queue.enqueue(ScheduleRequest("decode", 1))
            queue.cancel(token)
        self.assertEqual(queue.snapshot()["dispatching"], 1)
        self.assertTrue(queue.restore_claim(first))
        self.assertEqual(queue.dequeue(timeout=0)[0], first)
        queue.finish_claim(first)
        self.assertEqual(queue.dequeue(timeout=0)[0], second)
        queue.finish_claim(second)
        self.assertIsNone(queue.dequeue(timeout=0))

    def test_cancelled_claim_cannot_be_restored_after_cleanup(self):
        queue = PriorityScheduleQueue(maximum_requests=2)
        claimed = queue.enqueue(ScheduleRequest("decode", 1))
        queue.dequeue(timeout=0)
        pending = queue.enqueue(ScheduleRequest("decode", 1))
        self.assertTrue(queue.cancel(pending))
        self.assertEqual(len(queue._heap), 0)
        self.assertTrue(queue.cancel(claimed))
        self.assertFalse(queue.restore_claim(claimed))
        self.assertFalse(queue.finish_claim(claimed))
        self.assertFalse(queue.cancel(claimed))
        self.assertEqual(queue.snapshot()["dispatching"], 0)


if __name__ == "__main__":
    unittest.main()
