import unittest

from vllm_apple.expert_selection_telemetry import (
    ExpertSelectionSample,
    ExpertSelectionTelemetry,
)


class ExpertSelectionTelemetryTests(unittest.TestCase):
    def test_bounded_aggregate_tracks_layer_expert_and_cache_hits(self):
        telemetry = ExpertSelectionTelemetry(maximum_samples=2)
        telemetry.record(ExpertSelectionSample(0, (1, 2), (.7, .3), 100, (1,)))
        telemetry.record(ExpertSelectionSample(0, (2, 3), (.6, .4), 200, (2, 3)))
        telemetry.record(ExpertSelectionSample(1, (1,), (1.0,), 300, ()))
        report = telemetry.snapshot()
        self.assertEqual(report["samples"], 2)
        self.assertEqual(report["dropped"], 1)
        self.assertEqual(report["selected_experts"], 3)
        self.assertEqual(report["cache_hits"], 2)
        self.assertEqual(report["latency_nanoseconds"]["median"], 250.0)
        self.assertEqual(report["per_expert"][0]["expert"], 2)

    def test_empty_snapshot_and_invalid_sample_fail_closed(self):
        self.assertIsNone(
            ExpertSelectionTelemetry().snapshot()["latency_nanoseconds"]["median"]
        )
        with self.assertRaises(ValueError):
            ExpertSelectionSample(0, (1, 1), (.5, .5), 1, ())
        with self.assertRaises(ValueError):
            ExpertSelectionSample(0, (1,), (float("nan"),), 1, ())


if __name__ == "__main__":
    unittest.main()
