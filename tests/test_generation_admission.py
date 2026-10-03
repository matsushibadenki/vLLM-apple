import threading
import unittest
from unittest.mock import Mock, patch

from vllm_apple.generation_admission import (
    GenerationAdmission,
    GenerationAdmissionError,
    peer_disconnected,
)


class GenerationAdmissionTests(unittest.TestCase):
    def test_fifo_after_body_preparation_and_cancelled_head(self):
        for cancel_head in (False, True):
            gate = GenerationAdmission(3, 2)
            order = []
            entered = [threading.Event() for _ in range(3)]
            cancel = threading.Event()
            errors = []
            def worker(index):
                def cancelled():
                    entered[index].set()
                    return index == 0 and cancel.is_set()
                try:
                    with gate.admit(cancelled):
                        order.append(index)
                except GenerationAdmissionError as error:
                    errors.append(str(error))
            threads = []
            with gate.admit():
                for index in range(3):
                    thread = threading.Thread(target=worker, args=(index,))
                    threads.append(thread)
                    thread.start()
                    self.assertTrue(entered[index].wait(1))
                if cancel_head:
                    cancel.set()
            for thread in threads:
                thread.join(2)
                self.assertFalse(thread.is_alive())
            self.assertEqual(order, [1, 2] if cancel_head else [0, 1, 2])
            self.assertEqual(errors, ['generation_queue_cancelled'] if cancel_head else [])
            self.assertEqual(gate.snapshot()['inflight'], 0)

    def test_active_disconnect_is_distinct_from_waiting_cancel(self):
        gate = GenerationAdmission(0, 1)
        with self.assertRaises(BrokenPipeError):
            with gate.admit():
                raise BrokenPipeError()
        metrics = gate.snapshot()
        self.assertEqual(metrics['active_disconnects'], 1)
        self.assertEqual(metrics['cancelled'], 0)
        self.assertEqual(metrics['inflight'], 0)
        with gate.admit():
            pass

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

    def test_prepare_failure_releases_capacity(self):
        gate = GenerationAdmission(0, 1)
        def fail():
            raise ValueError('bad body')
        with self.assertRaises(ValueError):
            with gate.admit(prepare=fail):
                self.fail('invalid request entered generation')
        with gate.admit():
            pass

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
        self.assertEqual(gate.snapshot()['cancelled'], 2)
        self.assertEqual(gate.snapshot()['inflight'], 0)
        self.assertEqual(gate.snapshot()['active'], 0)
        with gate.admit():
            pass
