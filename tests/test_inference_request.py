import http.client
import io
import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from concurrent.futures import Future
from contextlib import nullcontext
from unittest.mock import Mock, patch

from vllm_apple.api import _after_inference_completion, create_server
from vllm_apple.inference_request import (
    InferenceRequestCancelled,
    InferenceRequestContext,
)
from vllm_apple.managed_engine import ThreadAffineInferenceEngine
from vllm_apple.scheduler import ScheduleRequest
from vllm_apple.service import RuntimeService


class Signal:
    def __init__(self, value=False):
        self.value = value

    def is_set(self):
        return self.value


class ManagedEngine:
    ready = True

    def __init__(self):
        self.context = None
        self.close_count = 0

    def models(self):
        return []

    def chat_completions_with_request_context(self, request, kernel, context):
        context.raise_if_cancelled()
        self.context = context
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    def open_chat_stream(self, request):
        return nullcontext(io.BytesIO(b"data: [DONE]\n\n"))

    def close(self):
        self.close_count += 1
        return True


class ExpiringEngine(ManagedEngine):
    def chat_completions_with_request_context(self, request, kernel, context):
        time.sleep(0.03)
        context.raise_if_cancelled()


class InferenceRequestTests(unittest.TestCase):
    def test_release_handles_pending_and_already_finished_completion(self):
        for finished in (False, True):
            with self.subTest(finished=finished):
                completion = Future()
                release = Mock()
                if finished:
                    completion.set_result(None)
                _after_inference_completion(completion, release)
                if not finished:
                    release.assert_not_called()
                    completion.set_result(None)
                release.assert_called_once()
        release = Mock()
        _after_inference_completion(None, release)
        release.assert_called_once()

    def test_abandoned_native_http_request_keeps_reservations_until_owner_finishes(self):
        for mode in ('timeout', 'disconnect'):
            with self.subTest(mode=mode):
                entered, release, abandoned, bytes_released, schedule_released = (
                    threading.Event() for _ in range(5)
                )

                class BlockingDelegate(ManagedEngine):
                    def chat_completions_with_request_context(self, request, kernel, context):
                        entered.set()
                        release.wait(5)
                        context.raise_if_cancelled()
                        return {}

                class ObservedOwner(ThreadAffineInferenceEngine):
                    def chat_completions_with_request_context(self, *args):
                        try:
                            return super().chat_completions_with_request_context(*args)
                        except InferenceRequestCancelled:
                            abandoned.set()
                            raise

                engine = ObservedOwner(BlockingDelegate)
                service = RuntimeService(engine=engine)
                server = create_server('127.0.0.1', 0, service, socket_timeout=0.15)
                server_thread = threading.Thread(target=server.serve_forever, daemon=True)
                server_thread.start()
                complete = service.complete_schedule
                release_bytes = server.release_request_bytes

                def completed(reservation):
                    result = complete(reservation)
                    schedule_released.set()
                    return result

                def released(amount):
                    release_bytes(amount)
                    bytes_released.set()

                connection = None
                try:
                    with patch.object(service, 'chat_schedule_request', return_value=ScheduleRequest(
                        'paged_attention', 4096
                    )), patch.object(service, 'complete_schedule', side_effect=completed) as completion, \
                         patch.object(server, 'release_request_bytes', side_effect=released) as byte_completion:
                        body = json.dumps({'model': 'fixture', 'messages': []}).encode()
                        if mode == 'timeout':
                            request = urllib.request.Request(
                                f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
                                data=body, headers={'Content-Type': 'application/json'}, method='POST'
                            )
                            with self.assertRaises(urllib.error.HTTPError) as error:
                                urllib.request.urlopen(request, timeout=2)
                            self.assertEqual(error.exception.code, 408)
                            error.exception.close()
                        else:
                            connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=2)
                            connection.request('POST', '/v1/chat/completions', body,
                                               {'Content-Type': 'application/json'})
                            self.assertTrue(entered.wait(1))
                            connection.close()
                        self.assertTrue(abandoned.wait(1))
                        self.assertTrue(entered.is_set())
                        self.assertEqual(service.scheduler.memory.snapshot()['reserved_bytes'], 4096)
                        self.assertEqual(service.scheduler.memory.snapshot()['active_reservations'], 1)
                        self.assertEqual(server.request_metrics()['inflight_request_bytes'], len(body))
                        completion.assert_not_called()
                        byte_completion.assert_not_called()
                        release.set()
                        self.assertTrue(schedule_released.wait(1))
                        self.assertTrue(bytes_released.wait(1))
                        self.assertEqual(service.scheduler.memory.snapshot()['active_reservations'], 0)
                        self.assertEqual(server.request_metrics()['inflight_request_bytes'], 0)
                        completion.assert_called_once()
                        byte_completion.assert_called_once()
                        request = urllib.request.Request(
                            f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
                            data=body, headers={'Content-Type': 'application/json'}, method='POST'
                        )
                        with urllib.request.urlopen(request, timeout=2) as recovered:
                            self.assertEqual(recovered.status, 200)
                            json.load(recovered)
                        self.assertTrue(server.wait_for_drain(2))
                        self.assertEqual(completion.call_count, 2)
                        self.assertEqual(byte_completion.call_count, 2)
                        self.assertEqual(server.request_metrics()['inflight_request_bytes'], 0)
                finally:
                    release.set()
                    if connection is not None:
                        connection.close()
                    server.shutdown()
                    server.server_close()
                    server_thread.join(2)
                    self.assertTrue(engine.close())

    def test_service_closes_managed_engine_exactly_once(self):
        engine = ManagedEngine()
        service = RuntimeService(engine=engine)
        self.assertTrue(service.close())
        self.assertTrue(service.close())
        self.assertEqual(engine.close_count, 1)

    def test_context_distinguishes_live_cancelled_and_expired(self):
        signal = Signal()
        context = InferenceRequestContext("request-123", time.monotonic() + 1, signal)
        context.raise_if_cancelled()
        self.assertGreater(context.remaining_seconds, 0)
        signal.value = True
        with self.assertRaisesRegex(InferenceRequestCancelled, "disconnected"):
            context.raise_if_cancelled()

        expired = InferenceRequestContext("request-456", time.monotonic() - 1, Signal())
        with self.assertRaisesRegex(InferenceRequestCancelled, "timed out"):
            expired.raise_if_cancelled()

    def test_http_chat_passes_bounded_request_context_to_engine(self):
        engine = ManagedEngine()
        server = create_server("127.0.0.1", 0, RuntimeService(engine=engine))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                data=json.dumps({"model": "fixture", "messages": []}).encode(),
                headers={"Content-Type": "application/json", "X-Request-ID": "managed-12345678"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                payload = json.load(response)
            self.assertEqual(payload["choices"][0]["message"]["content"], "ok")
            self.assertEqual(engine.context.request_id, "managed-12345678")
            self.assertGreater(engine.context.remaining_seconds, 0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_http_chat_timeout_releases_request_slot(self):
        server = create_server(
            "127.0.0.1", 0, RuntimeService(engine=ExpiringEngine()),
            socket_timeout=0.01,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                data=json.dumps({"model": "fixture", "messages": []}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(request, timeout=2)
            self.assertEqual(raised.exception.code, 408)
            self.assertEqual(json.load(raised.exception)["error"]["code"], "request_cancelled")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.assertEqual(server.request_metrics()["active_requests"], 0)


if __name__ == "__main__":
    unittest.main()
