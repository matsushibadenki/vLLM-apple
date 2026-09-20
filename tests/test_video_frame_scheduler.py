import unittest

from vllm_apple.video_frame_scheduler import ScheduledVideoFrame, VideoFrameScheduler


def frame(identifier, pts, *, keyframe=False):
    return ScheduledVideoFrame(identifier, pts, 1 / 30, identifier, keyframe)


class VideoFrameSchedulerTests(unittest.TestCase):
    def test_reorders_by_presentation_timestamp_and_waits_until_ready(self):
        scheduler = VideoFrameScheduler[str]()
        scheduler.submit(frame("later", 1.0))
        scheduler.submit(frame("earlier", 0.5))
        self.assertIsNone(scheduler.pop_ready(0.4))
        self.assertEqual(scheduler.pop_ready(0.5).frame.frame_id, "earlier")
        self.assertEqual(scheduler.pop_ready(1.0).frame.frame_id, "later")
        self.assertEqual(scheduler.snapshot.presented, 2)

    def test_drops_late_frames_before_returning_current_frame(self):
        scheduler = VideoFrameScheduler[str](maximum_lateness_seconds=0.1)
        scheduler.submit(frame("old-1", 0.0))
        scheduler.submit(frame("old-2", 0.1))
        scheduler.submit(frame("current", 1.0))
        decisions = scheduler.drain_late(1.0)
        self.assertEqual([item.action for item in decisions], [
            "dropped_late", "dropped_late", "present"
        ])
        self.assertEqual(scheduler.snapshot.late_drops, 2)

    def test_rejects_duplicate_capacity_and_excessive_reorder(self):
        scheduler = VideoFrameScheduler[str](maximum_frames=2, maximum_reorder_seconds=0.5)
        self.assertTrue(scheduler.submit(frame("a", 1.0)))
        self.assertFalse(scheduler.submit(frame("a", 1.1)))
        self.assertFalse(scheduler.submit(frame("too-old", 0.4)))
        self.assertTrue(scheduler.submit(frame("b", 1.2)))
        self.assertFalse(scheduler.submit(frame("full", 1.3)))
        self.assertEqual(scheduler.snapshot.rejected, 3)

    def test_cancelled_frame_is_skipped(self):
        scheduler = VideoFrameScheduler[str]()
        scheduler.submit(frame("cancel", 0))
        self.assertTrue(scheduler.cancel("cancel"))
        self.assertIsNone(scheduler.pop_ready(0))
        self.assertEqual(scheduler.snapshot.queue_depth, 0)


if __name__ == "__main__":
    unittest.main()
