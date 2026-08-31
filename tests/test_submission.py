import threading
import unittest

from tests.test_scheduler import hardware
from vllm_apple.scheduler import BasicScheduler, ScheduleRequest
from vllm_apple.submission import (
    GlobalSubmissionScheduler,
    SubmissionCancelledError,
)
from vllm_apple.types import Priority


class GlobalSubmissionSchedulerTests(unittest.TestCase):
    def test_backend_commands_run_off_the_application_thread_in_priority_order(self) -> None:
        scheduler = BasicScheduler(hardware(), 100)
        submissions = GlobalSubmissionScheduler(scheduler)
        application_thread = threading.get_ident()
        order = []
        background = submissions.submit(
            ScheduleRequest("decode", 1, Priority.BACKGROUND),
            lambda reservation, cancelled: order.append("background") or threading.get_ident(),
        )
        realtime = submissions.submit(
            ScheduleRequest("decode", 1, Priority.REALTIME),
            lambda reservation, cancelled: order.append("realtime") or threading.get_ident(),
        )
        submissions.start()
        realtime_thread = realtime.result(timeout=1)
        background_thread = background.result(timeout=1)
        self.assertEqual(order, ["realtime", "background"])
        self.assertNotEqual(realtime_thread, application_thread)
        self.assertEqual(realtime_thread, background_thread)
        self.assertTrue(submissions.shutdown())
        self.assertEqual(scheduler.memory.reserved_bytes, 0)

    def test_pending_cancellation_removes_command_without_reservation_leak(self) -> None:
        scheduler = BasicScheduler(hardware(), 100)
        submissions = GlobalSubmissionScheduler(scheduler)
        handle = submissions.submit(
            ScheduleRequest("decode", 80), lambda reservation, cancelled: "unused"
        )
        self.assertTrue(handle.cancel())
        with self.assertRaises(SubmissionCancelledError):
            handle.result(timeout=0)
        self.assertEqual(submissions.snapshot()["commands"], 0)
        self.assertEqual(scheduler.memory.reserved_bytes, 0)
        self.assertTrue(submissions.shutdown())

    def test_backend_exception_releases_reservation_and_does_not_stop_worker(self) -> None:
        scheduler = BasicScheduler(hardware(), 100)
        submissions = GlobalSubmissionScheduler(scheduler)

        def fail(reservation, cancelled):
            raise RuntimeError("backend failed")

        failed = submissions.submit(ScheduleRequest("decode", 80), fail)
        succeeded = submissions.submit(
            ScheduleRequest("decode", 80), lambda reservation, cancelled: "ok"
        )
        submissions.start()
        with self.assertRaisesRegex(RuntimeError, "backend failed"):
            failed.result(timeout=1)
        self.assertEqual(succeeded.result(timeout=1), "ok")
        self.assertEqual(scheduler.memory.reserved_bytes, 0)
        self.assertTrue(submissions.shutdown())

    def test_admission_failure_is_reported_and_worker_continues(self) -> None:
        scheduler = BasicScheduler(hardware(), 100)
        submissions = GlobalSubmissionScheduler(scheduler)
        rejected = submissions.submit(
            ScheduleRequest("decode", 101), lambda reservation, cancelled: "unreachable"
        )
        accepted = submissions.submit(
            ScheduleRequest("decode", 1), lambda reservation, cancelled: "ok"
        )
        submissions.start()
        with self.assertRaisesRegex(RuntimeError, "request needs 101 bytes"):
            rejected.result(timeout=1)
        self.assertEqual(accepted.result(timeout=1), "ok")
        self.assertTrue(submissions.shutdown())

    def test_running_command_uses_cooperative_cancellation_before_release(self) -> None:
        scheduler = BasicScheduler(hardware(), 100)
        submissions = GlobalSubmissionScheduler(scheduler)
        entered = threading.Event()

        def operation(reservation, cancelled):
            entered.set()
            self.assertTrue(cancelled.wait(timeout=1))
            return "cancelled-cleanly"

        handle = submissions.submit(ScheduleRequest("decode", 80), operation)
        submissions.start()
        self.assertTrue(entered.wait(timeout=1))
        self.assertFalse(handle.cancel())
        self.assertEqual(scheduler.memory.reserved_bytes, 80)
        self.assertTrue(handle.request_cancellation())
        self.assertEqual(handle.result(timeout=1), "cancelled-cleanly")
        self.assertEqual(scheduler.memory.reserved_bytes, 0)
        self.assertTrue(submissions.shutdown())


if __name__ == "__main__":
    unittest.main()
