import threading
import time
import unittest

from vllm_apple.inference_request import (
    InferenceEngineBusy,
    InferenceRequestCancelled,
    InferenceRequestContext,
)
from vllm_apple.managed_engine import ThreadAffineInferenceEngine


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
            args=({"name": "second"}, context("request-second", 0.05)),
        )
        try:
            first.start()
            self.assertTrue(entered.wait(timeout=1))
            second.start()
            time.sleep(0.01)
            with self.assertRaises(InferenceEngineBusy):
                engine.chat_completions_with_request_context(
                    {"name": "third"}, None, context("request-third")
                )
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


if __name__ == "__main__":
    unittest.main()
