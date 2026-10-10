import unittest
from unittest.mock import patch

from vllm_apple.expert_timing import PHASES, ExpertTimings, measure


class ExpertTimingTests(unittest.TestCase):
    def test_nested_spans_are_inclusive_and_failures_recorded(self):
        timings = ExpertTimings()
        with patch("vllm_apple.expert_timing.time.perf_counter_ns", side_effect=[0, 2, 5, 10]):
            with self.assertRaises(RuntimeError), timings.span("acquire"):
                with timings.span("mlx_load"):
                    raise RuntimeError("load failed")
        snapshot = timings.snapshot()
        self.assertEqual(snapshot["acquire"], dict(count=1, nanoseconds=10))
        self.assertEqual(snapshot["mlx_load"], dict(count=1, nanoseconds=3))
        self.assertEqual(set(snapshot), PHASES)
        snapshot["acquire"]["count"] = 999
        self.assertEqual(timings.snapshot()["acquire"]["count"], 1)

    def test_disabled_measure_and_unknown_phase(self):
        with measure(None, "acquire"):
            pass
        with self.assertRaises(ValueError):
            with ExpertTimings().span("arbitrary content"):
                self.fail("unknown phase must not run")
