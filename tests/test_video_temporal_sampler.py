import unittest

from vllm_apple.video_temporal_sampler import TemporalVideoFrame, sample_temporal_frames


def frame(index, *, keyframe=False, scene=0.0):
    return TemporalVideoFrame(str(index), float(index), index, keyframe, scene)


class VideoTemporalSamplerTests(unittest.TestCase):
    def test_preserves_boundaries_and_salient_frames_then_fills_time_gaps(self):
        frames = [frame(index) for index in range(11)]
        frames[3] = frame(3, scene=0.9)
        frames[7] = frame(7, keyframe=True)
        report = sample_temporal_frames(frames, maximum_frames=6, scene_change_threshold=0.5)
        identifiers = [item.frame_id for item in report.selected_frames]
        self.assertEqual(len(identifiers), 6)
        self.assertEqual(identifiers[0], "0")
        self.assertEqual(identifiers[-1], "10")
        self.assertIn("3", identifiers)
        self.assertIn("7", identifiers)
        self.assertEqual(report.dropped_frames, 5)

    def test_minimum_interval_limits_dense_scene_changes(self):
        frames = [frame(index, scene=1.0) for index in range(6)]
        report = sample_temporal_frames(
            frames,
            maximum_frames=6,
            minimum_interval_seconds=2.0,
            scene_change_threshold=0.5,
        )
        self.assertEqual(
            [item.presentation_seconds for item in report.selected_frames],
            [0.0, 2.0, 5.0],
        )

    def test_input_order_does_not_change_deterministic_result(self):
        frames = [frame(index) for index in range(8)]
        first = sample_temporal_frames(frames, maximum_frames=4)
        second = sample_temporal_frames(list(reversed(frames)), maximum_frames=4)
        self.assertEqual(first.selected_frames, second.selected_frames)

    def test_duplicate_ids_and_unbounded_configuration_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "identifiers"):
            sample_temporal_frames([frame(1), frame(1)], maximum_frames=1)
        with self.assertRaisesRegex(ValueError, "configuration"):
            sample_temporal_frames([], maximum_frames=0)


if __name__ == "__main__":
    unittest.main()
