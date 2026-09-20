import unittest

from vllm_apple.audio_deadline_scheduler import (
    AudioDeadlineScheduler,
    AudioScheduledTask,
    AudioSchedulingPriority,
)


def task(identifier, priority, deadline, duration=0.1):
    return AudioScheduledTask(identifier, priority, deadline, duration, identifier)


class AudioDeadlineSchedulerTests(unittest.TestCase):
    def test_realtime_precedes_other_classes_and_edf_orders_within_class(self):
        now = [0.0]
        scheduler = AudioDeadlineScheduler[str](clock=lambda: now[0])
        self.assertTrue(scheduler.submit(task("background", AudioSchedulingPriority.BACKGROUND, 10)))
        self.assertTrue(scheduler.submit(task("later", AudioSchedulingPriority.REALTIME, 5)))
        self.assertTrue(scheduler.submit(task("earlier", AudioSchedulingPriority.REALTIME, 3)))
        outcomes = [scheduler.run_next(str.upper) for _ in range(3)]
        self.assertEqual([outcome.task_id for outcome in outcomes], [
            "earlier", "later", "background"
        ])
        self.assertEqual(outcomes[0].result, "EARLIER")

    def test_admission_rejects_infeasible_deadline(self):
        scheduler = AudioDeadlineScheduler[str](clock=lambda: 10.0)
        self.assertFalse(scheduler.submit(
            task("late", AudioSchedulingPriority.REALTIME, 10.05, 0.1)
        ))
        self.assertEqual(scheduler.snapshot.rejected, 1)

    def test_expired_task_is_not_executed(self):
        now = [0.0]
        scheduler = AudioDeadlineScheduler[str](clock=lambda: now[0])
        self.assertTrue(scheduler.submit(task("audio", AudioSchedulingPriority.REALTIME, 1)))
        now[0] = 2.0
        calls = []
        outcome = scheduler.run_next(lambda payload: calls.append(payload))
        self.assertEqual(outcome.status, "deadline_missed")
        self.assertEqual(calls, [])
        self.assertEqual(scheduler.snapshot.deadline_misses, 1)

    def test_cancelled_task_is_skipped_and_capacity_is_recovered(self):
        scheduler = AudioDeadlineScheduler[str](maximum_tasks=1, clock=lambda: 0.0)
        self.assertTrue(scheduler.submit(task("first", AudioSchedulingPriority.REALTIME, 2)))
        self.assertTrue(scheduler.cancel("first"))
        self.assertFalse(scheduler.submit(task("second", AudioSchedulingPriority.REALTIME, 2)))
        self.assertIsNone(scheduler.run_next(str.upper))
        self.assertTrue(scheduler.submit(task("second", AudioSchedulingPriority.REALTIME, 2)))
        self.assertEqual(scheduler.snapshot.queue_depth, 1)


if __name__ == "__main__":
    unittest.main()
