import threading
import unittest
from unittest.mock import Mock, patch

from vllm_apple.generation_admission import (
    GenerationAdmission,
    GenerationAdmissionError,
    peer_disconnected,
)


class GenerationAdmissionTests(unittest.TestCase):
    def test_eof_probe_never_consumes_pending_request_bytes(self):
        connection = Mock()
        with patch('vllm_apple.generation_admission.select.select',
                   return_value=([connection], [], [])):
            connection.recv.return_value = b'x'
            self.assertFalse(peer_disconnected(connection))
            connection.recv.return_value = b''
            self.assertTrue(peer_disconnected(connection))
            connection.recv.side_effect = ConnectionResetError()
            self.assertTrue(peer_disconnected(connection))
        with patch('vllm_apple.generation_admission.select.select', return_value=([], [], [])):
            connection.recv.reset_mock()
            self.assertFalse(peer_disconnected(connection))
            connection.recv.assert_not_called()

    def test_full_queue_and_exception_release(self):
        gate = GenerationAdmission(0, .1)
        with gate.admit():
            with self.assertRaisesRegex(GenerationAdmissionError, 'queue_full'):
                with gate.admit():
                    self.fail('overload entered worker')
        with self.assertRaises(ValueError):
            with gate.admit():
                raise ValueError('handler failed')
        with gate.admit():
            pass

    def test_wait_timeout_releases_capacity_and_recovers(self):
        gate = GenerationAdmission(1, .02)
        errors = []
        def waiting_request():
            try:
                with gate.admit():
                    errors.append('unexpected admission')
            except GenerationAdmissionError as error:
                errors.append(str(error))
        with gate.admit():
            thread = threading.Thread(target=waiting_request)
            thread.start()
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
        self.assertEqual(errors, ['generation_queue_timeout'])
        with gate.admit():
            pass

    def test_invalid_bounds(self):
        for capacity, timeout in ((True, 1), (-1, 1), (129, 1), (1, 0),
                                  (1, float('nan')), (1, True), (1, 301)):
            with self.subTest(capacity=capacity, timeout=timeout), self.assertRaises(ValueError):
                GenerationAdmission(capacity, timeout)

    def test_cancelled_waiter_does_not_generate_and_releases_slot(self):
        gate = GenerationAdmission(1, 2)
        cancelled = threading.Event()
        checked = threading.Event()
        results = []
        def is_cancelled():
            checked.set()
            return cancelled.is_set()
        def waiting():
            try:
                with gate.admit(is_cancelled):
                    results.append('generated')
            except GenerationAdmissionError as error:
                results.append(str(error))
        with gate.admit():
            thread = threading.Thread(target=waiting)
            thread.start()
            self.assertTrue(checked.wait(1))
            cancelled.set()
            thread.join(1)
            self.assertFalse(thread.is_alive())
            with self.assertRaisesRegex(GenerationAdmissionError, 'queue_cancelled'):
                # A cancelled request released the waiting slot; this is not queue_full.
                with gate.admit(lambda: True):
                    pass
        self.assertEqual(results, ['generation_queue_cancelled'])
        with gate.admit():
            pass
