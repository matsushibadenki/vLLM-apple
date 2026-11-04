import http.client
import io
import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vllm_apple import main_thread_process_worker as worker_module
from vllm_apple.api import create_server
from vllm_apple.inference_request import (
    InferenceEngineBusy,
    InferenceRequestCancelled,
    InferenceRequestContext,
)
from vllm_apple.process_inference_engine import (
    MainThreadSubprocessInferenceEngine,
    _validated_python_executable,
)
from vllm_apple.scheduler import ScheduleRequest
from vllm_apple.service import RuntimeService


class ProcessInferenceEngineTests(unittest.TestCase):
    def test_active_python_is_trusted_even_with_hosted_toolcache_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "python"
            executable.write_text("#!/bin/sh\nexit 0\n")
            executable.chmod(0o777)
            with self.assertRaisesRegex(ValueError, "unsafe"):
                _validated_python_executable(executable)
            with patch.object(sys, "executable", str(executable)):
                self.assertEqual(_validated_python_executable(executable), executable)

    def test_root_owned_system_python_is_accepted(self):
        system_python = Path("/usr/bin/python3")
        if not system_python.exists() or system_python.stat().st_uid != 0:
            self.skipTest("root-owned system Python is unavailable")
        self.assertEqual(_validated_python_executable(system_python), system_python)

    def test_python_venv_symlink_is_preserved_after_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            link = Path(directory) / "python"
            link.symlink_to(Path(sys.executable).resolve())
            self.assertEqual(_validated_python_executable(link), link.absolute())

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.marker = root / "closed.txt"
        self.config = root / "config.json"
        self.config.write_text(json.dumps({"marker": str(self.marker)}))
        self.config.chmod(0o600)

    def tearDown(self):
        self.directory.cleanup()

    def engine(self, maximum_pending_requests=8):
        return MainThreadSubprocessInferenceEngine(
            "tests.process_engine_fixture:create_delegate",
            python_executable=Path(sys.executable),
            config_path=self.config,
            maximum_pending_requests=maximum_pending_requests,
            startup_timeout_seconds=5,
        )

    def context(self, seconds=5, cancellation=None):
        return InferenceRequestContext(
            "parent-request",
            time.monotonic() + seconds,
            cancellation or threading.Event(),
        )

    def test_child_owns_delegate_on_main_thread_and_preserves_order(self):
        engine = self.engine()
        try:
            first = engine.chat_completions_with_request_context(
                {"value": "a"}, None, self.context()
            )
            second = engine.chat_completions_with_request_context(
                {"value": "b"}, None, self.context()
            )
            self.assertNotEqual(first["pid"], os.getpid())
            self.assertTrue(first["main_thread"])
            self.assertEqual((first["sequence"], second["sequence"]), (1, 2))
            self.assertEqual(engine.models()[0]["id"], "process-fixture")
            diagnostics = engine.diagnostics()
            self.assertEqual(diagnostics["sequence"], 2)
            self.assertTrue(diagnostics["main_thread"])
        finally:
            self.assertTrue(engine.close())
        pid, main = self.marker.read_text().split(":")
        self.assertEqual(int(pid), first["pid"])
        self.assertEqual(main, "True")

    def test_parent_queue_saturation_is_bounded(self):
        engine = self.engine(maximum_pending_requests=1)
        outcome = []
        thread = threading.Thread(target=lambda: outcome.append(
            engine.chat_completions_with_request_context(
                {"delay": 0.3}, None, self.context()
            )
        ))
        thread.start()
        time.sleep(0.05)
        try:
            with self.assertRaises(InferenceEngineBusy):
                engine.chat_completions_with_request_context(
                    {"value": "rejected"}, None, self.context()
                )
            thread.join(timeout=2)
            self.assertEqual(len(outcome), 1)
        finally:
            engine.close()

    def test_active_cancellation_reaches_child_safe_point(self):
        engine = self.engine()
        cancellation = threading.Event()
        context = self.context(cancellation=cancellation)
        raised = []

        def invoke():
            try:
                engine.chat_completions_with_request_context(
                    {"delay": 2}, None, context
                )
            except BaseException as error:
                raised.append(error)

        thread = threading.Thread(target=invoke)
        thread.start()
        time.sleep(0.1)
        cancellation.set()
        thread.join(timeout=2)
        try:
            self.assertEqual(len(raised), 1)
            self.assertIsInstance(raised[0], InferenceRequestCancelled)
            recovered = engine.chat_completions_with_request_context(
                {"value": "after-cancel"}, None, self.context()
            )
            self.assertEqual(recovered["value"], "after-cancel")
        finally:
            engine.close()

    def wait_for(self, predicate):
        deadline = time.monotonic() + 3
        while not predicate() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(predicate())

    def test_cancel_keeps_capacity_until_child_acknowledges_or_exits(self):
        for exit_child in (False, True):
            with self.subTest(exit_child=exit_child):
                entered = Path(self.directory.name) / f"entered-{exit_child}"
                release = Path(self.directory.name) / f"release-{exit_child}"
                engine = self.engine(maximum_pending_requests=1)
                cancellation = threading.Event()
                errors = []

                def invoke():
                    try:
                        engine.chat_completions_with_request_context(
                            {"entered": str(entered), "release": str(release),
                             "exit": exit_child}, None, self.context(cancellation=cancellation)
                        )
                    except BaseException as error:
                        errors.append(error)

                caller = threading.Thread(target=invoke)
                caller.start()
                try:
                    self.wait_for(entered.exists)
                    cancellation.set()
                    caller.join(2)
                    self.assertFalse(caller.is_alive())
                    self.assertEqual(len(errors), 1)
                    self.assertIsInstance(errors[0], InferenceRequestCancelled)
                    completion = errors[0].completion
                    self.assertIsNotNone(completion)
                    self.assertFalse(completion.done())
                    with self.assertRaises(InferenceEngineBusy):
                        engine.chat_completions_with_request_context({}, None, self.context())
                    callback_done = threading.Event()
                    observations = []

                    def finished(_future):
                        # A callback may reenter admission without holding the pending lock.
                        with engine._pending_lock:
                            observations.append(len(engine._pending))
                        callback_done.set()

                    completion.add_done_callback(finished)
                    release.touch()
                    self.assertTrue(callback_done.wait(3))
                    self.assertEqual(observations, [0])
                    if not exit_child:
                        self.assertEqual(engine.chat_completions_with_request_context(
                            {"value": "recovered"}, None, self.context()
                        )["value"], "recovered")
                    else:
                        self.assertIsNotNone(engine._process.poll())
                finally:
                    release.touch()
                    caller.join(2)
                    engine.close()

    def test_queued_cancel_acknowledges_without_waiting_for_active_native_call(self):
        root = Path(self.directory.name)
        active_entered, active_release = root/'active-entered', root/'active-release'
        cancelled_entered = root/'cancelled-entered'
        engine = self.engine(maximum_pending_requests=2)
        cancellation = threading.Event()
        sent = threading.Event()
        outcomes = {}
        original_write = engine._write

        def write(payload):
            original_write(payload)
            if payload.get('request', {}).get('value') == 'cancelled':
                sent.set()

        def invoke(name, request, signal=None):
            try:
                outcomes[name] = engine.chat_completions_with_request_context(
                    request, None, self.context(seconds=10, cancellation=signal)
                )
            except BaseException as error:
                outcomes[name] = error

        active = threading.Thread(target=invoke, args=('active', {
            'entered': str(active_entered), 'release': str(active_release)}))
        cancelled = threading.Thread(target=invoke, args=('cancelled', {
            'value': 'cancelled', 'entered': str(cancelled_entered)}, cancellation))
        replacement = threading.Thread(target=invoke, args=('replacement', {'value': 'replacement'}))
        try:
            active.start()
            self.wait_for(active_entered.exists)
            with patch.object(engine, '_write', side_effect=write):
                cancelled.start()
                self.assertTrue(sent.wait(2))
                cancellation.set()
                cancelled.join(2)
            self.assertFalse(cancelled.is_alive())
            self.assertIsInstance(outcomes['cancelled'], InferenceRequestCancelled)
            self.wait_for(lambda: len(engine._pending) == 1)
            completion = outcomes['cancelled'].completion
            if completion is not None:
                self.wait_for(completion.done)
            self.assertTrue(active.is_alive())
            self.assertFalse(cancelled_entered.exists())
            replacement.start()
            self.wait_for(lambda: len(engine._pending) == 2)
            with self.assertRaises(InferenceEngineBusy):
                engine.chat_completions_with_request_context({}, None, self.context())
            active_release.touch()
            active.join(3)
            replacement.join(3)
            self.assertEqual(outcomes['replacement']['value'], 'replacement')
            self.assertEqual(outcomes['replacement']['sequence'], 2)
            self.assertFalse(cancelled_entered.exists())
        finally:
            active_release.touch()
            for caller in (active, cancelled, replacement):
                if caller.ident is not None:
                    caller.join(3)
            engine.close()

    def test_invalid_terminal_or_stdout_eof_releases_only_after_process_exit(self):
        for request in ({'invalid_terminal': {'type': 'cancel_ack'}},
                        {'invalid_terminal': {'type': 'result', 'result': []}},
                        {'invalid_terminal': {'type': 'error', 'code': 'cancel_received'}},
                        {'close_stdout': True}):
            with self.subTest(request=request):
                engine = self.engine(maximum_pending_requests=1)
                try:
                    with self.assertRaisesRegex(RuntimeError, 'worker_exited'):
                        engine.chat_completions_with_request_context(
                            dict(request, release=str(Path(self.directory.name)/'never-release')),
                            None, self.context(seconds=10)
                        )
                    self.assertIsNotNone(engine._process.poll())
                    self.assertFalse(engine.ready)
                    self.assertEqual(engine._pending, {})
                finally:
                    self.assertTrue(engine.close())

    def test_terminal_callback_can_close_its_owned_engine(self):
        engine = self.engine()
        finished = threading.Event()
        outcomes = []

        def on_complete(_future):
            try:
                outcomes.append(engine.close())
            except BaseException as error:
                outcomes.append(error)
            finally:
                finished.set()

        # Register before the terminal response so the reader invokes the callback.
        original_write = engine._write

        def write(payload):
            if payload.get('type') == 'request':
                with engine._pending_lock:
                    engine._pending[payload['id']].completion.add_done_callback(on_complete)
            original_write(payload)

        try:
            with patch.object(engine, '_write', side_effect=write):
                result = engine.chat_completions_with_request_context(
                    {'value': 'close-after-result'}, None, self.context()
                )
            self.assertEqual(result['value'], 'close-after-result')
            self.assertTrue(finished.wait(3))
            engine._reader.join(2)
            self.assertFalse(engine._reader.is_alive())
            self.assertEqual(outcomes, [True])
        finally:
            engine.close()

    def test_http_abandonment_retains_process_resources_until_ack(self):
        for mode in ("timeout", "disconnect", "protocol_failure"):
            with self.subTest(mode=mode):
                entered = Path(self.directory.name) / f"http-entered-{mode}"
                release = Path(self.directory.name) / f"http-release-{mode}"
                emit_invalid = Path(self.directory.name) / f"http-invalid-{mode}"
                engine = self.engine(maximum_pending_requests=1)
                service = RuntimeService(engine=engine)
                server = create_server("127.0.0.1", 0, service, socket_timeout=0.2)
                worker = threading.Thread(target=server.serve_forever, daemon=True)
                worker.start()
                request_body = {"model": "process-fixture", "messages": [],
                                "entered": str(entered), "release": str(release)}
                if mode == "protocol_failure":
                    request_body.update(invalid_terminal={"type": "cancel_ack"},
                                        emit_invalid=str(emit_invalid))
                body = json.dumps(request_body).encode()
                connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
                schedule_done, bytes_done = threading.Event(), threading.Event()
                complete_schedule = service.complete_schedule
                release_bytes = server.release_request_bytes

                def completed(reservation):
                    complete_schedule(reservation)
                    schedule_done.set()

                def released(amount):
                    release_bytes(amount)
                    bytes_done.set()

                try:
                    with patch.object(service, "chat_schedule_request", return_value=ScheduleRequest(
                        "paged_attention", 4096
                    )), patch.object(service, "complete_schedule", side_effect=completed) as schedule, \
                         patch.object(server, "release_request_bytes", side_effect=released) as byte_release:
                        connection.request("POST", "/v1/chat/completions", body,
                                           {"Content-Type": "application/json"})
                        self.wait_for(entered.exists)
                        if mode != "disconnect":
                            response = connection.getresponse()
                            self.assertEqual(response.status, 408)
                            response.read()
                        else:
                            connection.close()
                        self.wait_for(lambda: server.request_metrics()["active_requests"] == 0)
                        self.assertEqual(service.scheduler.memory.snapshot()["reserved_bytes"], 4096)
                        self.assertEqual(server.request_metrics()["inflight_request_bytes"], len(body))
                        schedule.assert_not_called()
                        byte_release.assert_not_called()
                        if mode == "protocol_failure":
                            emit_invalid.touch()
                        else:
                            release.touch()
                        self.assertTrue(schedule_done.wait(3))
                        self.assertTrue(bytes_done.wait(3))
                        self.assertEqual(service.scheduler.memory.snapshot()["active_reservations"], 0)
                        self.assertEqual(server.request_metrics()["inflight_request_bytes"], 0)
                        schedule.assert_called_once()
                        byte_release.assert_called_once()
                        if mode == "protocol_failure":
                            self.assertIsNotNone(engine._process.poll())
                            self.assertFalse(engine.ready)
                        else:
                            self.assertEqual(engine.chat_completions_with_request_context(
                                {"value": "after-ack"}, None, self.context()
                            )["value"], "after-ack")
                finally:
                    emit_invalid.touch()
                    release.touch()
                    connection.close()
                    server.shutdown()
                    server.server_close()
                    worker.join(2)
                    service.close()

    def test_diagnostics_timeout_stays_bounded_until_child_reply(self):
        entered = Path(self.directory.name) / "diagnostic-entered"
        release = Path(self.directory.name) / "diagnostic-release"
        engine = self.engine(maximum_pending_requests=2)
        outcomes = []
        caller = threading.Thread(target=lambda: outcomes.append(
            engine.chat_completions_with_request_context(
                {"entered": str(entered), "release": str(release)}, None, self.context()
            )
        ))
        caller.start()
        try:
            self.wait_for(entered.exists)
            with self.assertRaises(TimeoutError):
                engine.diagnostics(timeout_seconds=0.05)
            with self.assertRaises(InferenceEngineBusy):
                engine.diagnostics()
            release.touch()
            caller.join(3)
            self.assertEqual(len(outcomes), 1)
            self.wait_for(lambda: not engine._pending)
            self.assertEqual(engine.diagnostics()["sequence"], 1)
        finally:
            release.touch()
            caller.join(3)
            engine.close()

    def test_oversized_request_releases_unsent_admission(self):
        engine = self.engine(maximum_pending_requests=1)
        try:
            with self.assertRaisesRegex(ValueError, "too large"):
                engine.chat_completions_with_request_context(
                    {"value": "x" * (4 * 1024 * 1024)}, None, self.context()
                )
            self.assertEqual(engine.chat_completions_with_request_context(
                {"value": "after-rejection"}, None, self.context()
            )["value"], "after-rejection")
        finally:
            engine.close()

    def test_real_http_path_reaches_child_main_thread(self):
        engine = self.engine()
        service = RuntimeService(engine=engine)
        server = create_server("127.0.0.1", 0, service)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            body = {
                "model": "process-fixture",
                "messages": [{"role": "user", "content": "hello"}],
                "value": "through-http",
            }
            request = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=3) as response:
                result = json.load(response)
            self.assertTrue(result["main_thread"])
            self.assertEqual(result["value"], "through-http")
            self.assertNotEqual(result["pid"], os.getpid())
        finally:
            server.shutdown()
            server.server_close()
            service.close()
            thread.join(timeout=2)

    def test_http_timeout_cancels_child_and_recovers_request_slot(self):
        engine = self.engine()
        service = RuntimeService(engine=engine)
        server = create_server("127.0.0.1", 0, service, socket_timeout=0.15)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            body = {
                "model": "process-fixture",
                "messages": [{"role": "user", "content": "hello"}],
                "delay": 2,
            }
            request = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(request, timeout=2)
            self.assertEqual(raised.exception.code, 408)
            result = engine.chat_completions_with_request_context(
                {"value": "after-timeout"}, None, self.context()
            )
            self.assertEqual(result["value"], "after-timeout")
            self.assertEqual(server.request_metrics()["active_requests"], 0)
        finally:
            server.shutdown()
            server.server_close()
            service.close()
            thread.join(timeout=2)

    def test_client_disconnect_cancels_child_and_cleans_up(self):
        engine = self.engine()
        service = RuntimeService(engine=engine)
        server = create_server("127.0.0.1", 0, service, socket_timeout=2)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        body = json.dumps({
            "model": "process-fixture",
            "messages": [{"role": "user", "content": "hello"}],
            "delay": 2,
        }).encode()
        connection = socket.create_connection(("127.0.0.1", server.server_port))
        try:
            connection.sendall(
                b"POST /v1/chat/completions HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\nContent-Type: application/json\r\n"
                + f"Content-Length: {len(body)}\r\n\r\n".encode()
                + body
            )
        finally:
            connection.close()
        deadline = time.monotonic() + 3
        while (server.request_metrics()["active_requests"] != 0
               and time.monotonic() < deadline):
            time.sleep(0.02)
        try:
            self.assertEqual(server.request_metrics()["active_requests"], 0)
            result = engine.chat_completions_with_request_context(
                {"value": "after-disconnect"}, None, self.context()
            )
            self.assertEqual(result["value"], "after-disconnect")
        finally:
            server.shutdown()
            server.server_close()
            service.close()
            thread.join(timeout=2)


class WorkerQueueTests(unittest.TestCase):
    def test_reader_sets_deadline_at_enqueue_and_ignores_supplied_internal_deadline(self):
        identifier = 'a' * 32
        request = {'type': 'request', 'id': identifier, 'request': {},
                   'timeout_seconds': 0.25, '_deadline_monotonic': 999999}
        commands = worker_module._CommandQueue(2)
        stdin = SimpleNamespace(buffer=io.BytesIO((json.dumps(request) + '\n').encode()))
        with patch.object(worker_module.sys, 'stdin', stdin), \
             patch.object(worker_module.time, 'monotonic', return_value=100):
            worker_module._reader(commands, {}, threading.Lock())
        self.assertEqual(commands.get_nowait()['_deadline_monotonic'], 100.25)
        self.assertIsNone(commands.get_nowait())

    def test_expired_waiter_never_calls_delegate(self):
        identifier = 'b' * 32
        delegate = Mock()
        delegate.models.return_value = []

        def feed(commands, cancellations, _lock):
            cancellations[identifier] = threading.Event()
            commands.put_nowait({'type': 'request', 'id': identifier, 'request': {},
                                 '_deadline_monotonic': time.monotonic() - 1})
            commands.put_nowait(None)

        with patch.object(worker_module, '_factory', return_value=delegate), \
             patch.object(worker_module, '_reader', side_effect=feed), \
             patch.object(worker_module, '_write') as write:
            self.assertEqual(worker_module.run('fixture:create', None, 2), 0)
        delegate.chat_completions_with_request_context.assert_not_called()
        delegate.close.assert_called_once()
        write.assert_any_call({'type': 'error', 'id': identifier, 'code': 'request_timed_out'})


if __name__ == "__main__":
    unittest.main()
