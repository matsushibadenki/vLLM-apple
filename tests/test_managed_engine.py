import threading
import time
import unittest
from unittest.mock import Mock, patch

from vllm_apple.inference_request import (
    InferenceEngineBusy,
    InferenceRequestCancelled,
    InferenceRequestContext,
)
from vllm_apple.managed_engine import ThreadAffineInferenceEngine
from vllm_apple.service import RuntimeService


class Delegate:
    ready = True

    def __init__(self, calls, entered=None, release=None):
        self.calls = calls
        self.entered = entered
        self.release = release
        self.calls.append(("create", threading.get_ident()))

    def models(self):
        self.calls.append(("models", threading.get_ident()))
        return [{"id": "fixture", "object": "model"}]

    def chat_completions_with_request_context(self, request, kernel, context):
        self.calls.append((request["name"], threading.get_ident()))
        if self.entered is not None:
            self.entered.set()
        if self.release is not None and request.get("block"):
            self.release.wait(timeout=2)
        context.raise_if_cancelled()
        return {"name": request["name"]}

    def close(self):
        self.calls.append(("close", threading.get_ident()))


def context(name, seconds=2):
    return InferenceRequestContext(name, time.monotonic() + seconds, threading.Event())


class ThreadAffineInferenceEngineTests(unittest.TestCase):
    def test_create_infer_models_and_close_stay_on_owner_thread(self):
        calls = []
        engine = ThreadAffineInferenceEngine(lambda: Delegate(calls))
        try:
            result = engine.chat_completions_with_request_context(
                {"name": "infer"}, None, context("request-owner")
            )
            self.assertEqual(result, {"name": "infer"})
            self.assertEqual(engine.models()[0]["id"], "fixture")
        finally:
            self.assertTrue(engine.close())
        self.assertTrue(calls)
        self.assertEqual({thread_id for _, thread_id in calls}, {engine.owner_thread_ident})

    def test_bounded_queue_rejects_excess_and_skips_cancelled_pending_work(self):
        calls = []
        entered = threading.Event()
        release = threading.Event()
        engine = ThreadAffineInferenceEngine(
            lambda: Delegate(calls, entered, release), maximum_pending_requests=1
        )
        outcomes = []
        queued = threading.Event()
        second_context = context("request-second")
        original_put = engine._commands.put_nowait

        def observed_put(command):
            original_put(command)
            if command is not None and command.request_context is second_context:
                queued.set()

        def invoke(request, request_context):
            try:
                outcomes.append(engine.chat_completions_with_request_context(
                    request, None, request_context
                ))
            except BaseException as error:
                outcomes.append(error)

        first = threading.Thread(
            target=invoke,
            args=({"name": "first", "block": True}, context("request-first")),
        )
        second = threading.Thread(
            target=invoke,
            args=({"name": "second"}, second_context),
        )
        try:
            first.start()
            self.assertTrue(entered.wait(timeout=1))
            with patch.object(engine._commands, "put_nowait", side_effect=observed_put):
                second.start()
                self.assertTrue(queued.wait(1))
            with self.assertRaises(InferenceEngineBusy):
                engine.chat_completions_with_request_context(
                    {"name": "third"}, None, context("request-third")
                )
            second_context.cancellation.set()
            second.join(timeout=1)
            self.assertFalse(second.is_alive())
            self.assertTrue(any(isinstance(item, InferenceRequestCancelled) for item in outcomes))
            release.set()
            first.join(timeout=1)
            self.assertFalse(first.is_alive())
        finally:
            release.set()
            engine.close()
        self.assertNotIn("second", [name for name, _ in calls])
        self.assertNotIn("third", [name for name, _ in calls])

    def test_full_queue_close_is_bounded_retryable_and_skips_pending_work(self):
        calls, outcomes = [], []
        entered, release, queued = (threading.Event() for _ in range(3))
        engine = ThreadAffineInferenceEngine(
            lambda: Delegate(calls, entered, release), maximum_pending_requests=1
        )
        original_put = engine._commands.put_nowait

        def observed_put(command):
            original_put(command)
            if command is not None and command.request.get('name') == 'pending':
                queued.set()

        def invoke(name, block=False):
            try:
                outcomes.append(engine.chat_completions_with_request_context(
                    {'name': name, 'block': block}, None, context(name)
                ))
            except BaseException as error:
                outcomes.append(error)

        active = threading.Thread(target=invoke, args=('active', True))
        pending = threading.Thread(target=invoke, args=('pending',))
        try:
            active.start()
            self.assertTrue(entered.wait(1))
            with patch.object(engine._commands, 'put_nowait', side_effect=observed_put):
                pending.start()
                self.assertTrue(queued.wait(1))
            self.assertFalse(engine.close(timeout_seconds=0.01))
            self.assertFalse(engine.ready)
            self.assertFalse(engine.close(timeout_seconds=0))
            pending.join(1)
            self.assertFalse(pending.is_alive())
            self.assertTrue(any(isinstance(item, InferenceRequestCancelled) for item in outcomes))
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                engine.chat_completions_with_request_context({}, None, context('late'))
            self.assertNotIn('close', [name for name, _ in calls])
        finally:
            release.set()
            active.join(1)
            if pending.ident is not None:
                pending.join(1)
            self.assertTrue(engine.close())
        self.assertNotIn('pending', [name for name, _ in calls])
        self.assertEqual([name for name, _ in calls].count('close'), 1)
        self.assertEqual({owner for _, owner in calls}, {engine.owner_thread_ident})

    def test_startup_timeout_closes_late_delegate_on_owner_thread(self):
        calls, errors = [], []
        entered, release, closed = (threading.Event() for _ in range(3))

        class LateDelegate(Delegate):
            def close(self):
                super().close()
                closed.set()

        def factory():
            entered.set()
            release.wait(2)
            return LateDelegate(calls)

        def construct():
            try:
                ThreadAffineInferenceEngine(factory, startup_timeout_seconds=0.02)
            except BaseException as error:
                errors.append(error)

        constructor = threading.Thread(target=construct)
        try:
            constructor.start()
            self.assertTrue(entered.wait(1))
            constructor.join(1)
            self.assertFalse(constructor.is_alive())
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], TimeoutError)
        finally:
            release.set()
            constructor.join(1)
        self.assertTrue(closed.wait(1))
        self.assertEqual([name for name, _ in calls], ['create', 'close'])
        self.assertEqual(len({owner for _, owner in calls}), 1)

    def test_partial_startup_failure_still_closes_delegate(self):
        calls = []
        closed = threading.Event()

        class BrokenDelegate(Delegate):
            def models(self):
                raise ValueError('invalid model metadata')

            def close(self):
                super().close()
                closed.set()

        with self.assertRaisesRegex(RuntimeError, 'startup failed'):
            ThreadAffineInferenceEngine(lambda: BrokenDelegate(calls))
        self.assertTrue(closed.wait(1))
        self.assertEqual([name for name, _ in calls], ['create', 'close'])
        self.assertEqual(len({owner for _, owner in calls}), 1)

    def test_owner_callback_can_request_close_without_self_join(self):
        calls, holder, result = [], {}, []

        class ClosingDelegate(Delegate):
            def chat_completions_with_request_context(self, request, kernel, request_context):
                result.append(holder['engine'].close(timeout_seconds=0))
                return {'closed': True}

        engine = ThreadAffineInferenceEngine(lambda: ClosingDelegate(calls))
        holder['engine'] = engine
        try:
            self.assertEqual(engine.chat_completions_with_request_context(
                {}, None, context('self-close')
            ), {'closed': True})
        finally:
            self.assertTrue(engine.close())
        self.assertEqual(result, [False])
        self.assertEqual([name for name, _ in calls].count('close'), 1)

    def test_close_rejects_invalid_timeouts_without_shutting_down(self):
        engine = ThreadAffineInferenceEngine(lambda: Delegate([]))
        try:
            for value in (-1, float('nan'), float('inf'), 1801):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    engine.close(timeout_seconds=value)
            self.assertTrue(engine.ready)
        finally:
            self.assertTrue(engine.close())

    def test_service_retries_incomplete_shutdown_and_caches_only_success(self):
        engine = Mock(ready=True)
        engine.close.side_effect = [False, True]
        service = RuntimeService(engine=engine)
        self.assertFalse(service.close())
        self.assertTrue(service.close())
        self.assertTrue(service.close())
        self.assertEqual(engine.close.call_count, 2)

    def test_service_does_not_claim_success_during_another_close(self):
        entered, release = threading.Event(), threading.Event()
        engine = Mock(ready=True)

        def close():
            entered.set()
            release.wait(2)
            return True

        engine.close.side_effect = close
        service = RuntimeService(engine=engine)
        results = []
        closer = threading.Thread(target=lambda: results.append(service.close()))
        try:
            closer.start()
            self.assertTrue(entered.wait(1))
            self.assertFalse(service.close())
        finally:
            release.set()
            closer.join(1)
        self.assertEqual(results, [True])
        self.assertTrue(service.close())
        engine.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
