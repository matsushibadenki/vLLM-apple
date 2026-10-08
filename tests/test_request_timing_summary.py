import unittest

from scripts.summarize_request_timings import summarize


class RequestTimingSummaryTests(unittest.TestCase):
    def test_missing_is_not_zero_and_backend_interval_is_separate(self):
        result = summarize([{'stages_ms': {'scheduler_dequeued': 10,
            'tokenize_started': 12, 'tokenize_finished': 15, 'first_response_ready': 115}},
            {'stages_ms': {}}])
        self.assertEqual(result['phases']['tokenize']['median_ms'], 3)
        self.assertEqual(result['phases']['tokenize']['missing'], 1)
        self.assertEqual(result['phases']['tokenize_end_to_first_response_ready']['max_ms'], 100)
        self.assertIsNone(result['phases']['first_sse_write']['median_ms'])
        self.assertFalse(result['qualification'])

    def test_negative_and_nonfinite_intervals_are_rejected(self):
        for value in (-1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                summarize([{'stages_ms': {'scheduler_dequeued': value}}])
