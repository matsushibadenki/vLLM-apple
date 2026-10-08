import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from vllm_apple.request_timing import RequestTimings, TimedSSEWriter, timed_tokenize


class RequestTimingTests(unittest.TestCase):
    def test_tokenize_records_success_and_failure_without_changing_result(self):
        history = RequestTimings()
        request = SimpleNamespace(_vllm_apple_timing=history.start())
        value = object()
        self.assertIs(timed_tokenize(lambda *a: value, None, request, None), value)
        request._vllm_apple_timing.finish()
        successful = history.snapshot()['records'][0]['stages_ms']
        self.assertLessEqual(successful['tokenize_started'], successful['tokenize_finished'])
        self.assertNotIn('tokenize_failed', successful)
        request._vllm_apple_timing = history.start()
        error = ValueError('tokenizer unavailable')

        def failed(*args):
            raise error

        with self.assertRaises(ValueError) as raised:
            timed_tokenize(failed, None, request, None)
        self.assertIs(raised.exception, error)
        request._vllm_apple_timing.finish()
        self.assertIn('tokenize_failed', history.snapshot()['records'][1]['stages_ms'])

    def test_tokenize_without_trace_forwards_arguments(self):
        arguments = (object(), SimpleNamespace(), object())
        self.assertEqual(timed_tokenize(lambda *a: a, *arguments), arguments)

    def test_first_marks_are_stable_and_monotonic_while_wall_clock_moves(self):
        history = RequestTimings()
        with patch('time.monotonic_ns', side_effect=[100, 200, 300, 400]), patch('time.time_ns', return_value=10):
            trace = history.start()
            trace.mark('first_response_ready')
            trace.mark('first_response_ready')
            trace.finish()
        record = history.snapshot()['records'][0]
        self.assertEqual(record['stages_ms']['first_response_ready'], .0001)
        self.assertEqual(record['stages_ms']['handler_finished'], .0003)
        record['stages_ms'].clear()
        self.assertTrue(history.snapshot()['records'][0]['stages_ms'])

    def test_history_is_bounded_finish_idempotent_and_traces_are_isolated(self):
        history = RequestTimings(capacity=2)
        first, second = history.start(), history.start()
        first.mark('scheduler_dequeued')
        second.mark('first_response_ready')
        first.finish()
        first.finish()
        second.finish()
        history.start().finish()
        snapshot = history.snapshot()
        self.assertEqual(snapshot['completed'], 3)
        self.assertEqual([r['sequence'] for r in snapshot['records']], [2, 3])
        self.assertNotIn('scheduler_dequeued', snapshot['records'][0]['stages_ms'])

    def test_writer_forwards_bytes_and_flush_and_excludes_headers_keepalive_done(self):
        history = RequestTimings()
        trace = history.start()
        original = io.BytesIO()
        writer = TimedSSEWriter(original, trace)
        parts = [b'HTTP/1.1 200 OK\r\n', b': keepalive 1/2\n\n', b'data: [DONE]\n\n']
        for part in parts:
            self.assertEqual(writer.write(part), len(part))
        self.assertNotIn('first_sse_write_started', trace._stages)
        parts.append(b'data: {"text":"secret"}\n\n')
        writer.write(parts[-1])
        writer.flush()
        trace.finish()
        self.assertEqual(original.getvalue(), b''.join(parts))
        snapshot = history.snapshot()
        self.assertIn('first_sse_write_returned', snapshot['records'][0]['stages_ms'])
        self.assertNotIn('secret', str(snapshot))
