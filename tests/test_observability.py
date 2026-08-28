import unittest

from vllm_apple.backend import OpenAIProxyEngine
from vllm_apple.events import EventBus
from vllm_apple.observability import (
    REQUEST_ID_HEADER,
    RequestLogRecord,
    StructuredRequestLog,
    current_request_id,
    request_scope,
    resolve_request_id,
)


class ObservabilityTests(unittest.TestCase):
    def test_request_ids_are_validated_and_context_is_restored(self) -> None:
        self.assertEqual(resolve_request_id("client-12345678"), "client-12345678")
        self.assertNotEqual(resolve_request_id("prompt text"), "prompt text")
        self.assertIsNone(current_request_id())
        with request_scope("request-12345678"):
            self.assertEqual(current_request_id(), "request-12345678")
        self.assertIsNone(current_request_id())

    def test_log_is_bounded_and_contains_no_request_body_field(self) -> None:
        log = StructuredRequestLog(capacity=2)
        for index in range(3):
            log.append(RequestLogRecord(str(index), "POST", "/v1/chat/completions", 200, 1.0, False))
        records = log.records()
        self.assertEqual([record.request_id for record in records], ["1", "2"])
        self.assertNotIn("body", records[-1].to_dict())
        self.assertNotIn("prompt", records[-1].to_dict())

    def test_request_id_reaches_backend_and_events(self) -> None:
        engine = OpenAIProxyEngine("http://127.0.0.1:1")
        bus = EventBus()
        with request_scope("request-abcdefgh"):
            request = engine._request("/v1/models")
            event = bus.publish("test.event", {})
        headers = {key.lower(): value for key, value in request.header_items()}
        self.assertEqual(headers[REQUEST_ID_HEADER.lower()], "request-abcdefgh")
        self.assertEqual(event.payload["request_id"], "request-abcdefgh")


if __name__ == "__main__":
    unittest.main()
