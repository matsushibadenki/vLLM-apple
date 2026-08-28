import asyncio
import json
import unittest
from pathlib import Path

from tests.schema_validator import validate_instance
from vllm_apple.backend_tuning import (
    BackendKernelTuningAdapter,
    KernelTuningASGIMiddleware,
    parse_kernel_tuning_headers,
)
from vllm_apple.kernel_context import (
    KERNEL_TUNING_ACCEPTED_HEADER,
    KERNEL_TUNING_CONTEXT_HEADER,
    KERNEL_TUNING_ID_HEADER,
    InferenceKernelContext,
    PagedAttentionKernelSelection,
)
from vllm_apple.kernel_profile import PagedAttentionShape
from vllm_apple.metal_probe import MetalThreadConfiguration
from vllm_apple.vllm_middleware import (
    VLLMAppleKernelTuningMiddleware,
    invoke_paged_attention_kernel,
)


class BackendKernelTuningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.shape = PagedAttentionShape(1024, 1, 32, 8, 128, 16, 64, 4194304)

    def context(self, identity: str, width: int) -> InferenceKernelContext:
        return InferenceKernelContext(
            identity * 24,
            "f" * 24,
            (
                PagedAttentionKernelSelection(
                    self.shape, MetalThreadConfiguration(width, width, width)
                ),
            ),
        )

    def test_strict_parser_round_trips_and_rejects_identity_mismatch(self) -> None:
        context = self.context("a", 128)
        schema = json.loads(
            Path("schemas/runtime/backend-kernel-tuning-context-v1.schema.json").read_text()
        )
        validate_instance(context.to_dict(), schema)
        self.assertEqual(parse_kernel_tuning_headers(context.to_http_headers()), context)
        headers = context.to_http_headers()
        headers[KERNEL_TUNING_ID_HEADER] = "b" * 24
        with self.assertRaises(ValueError):
            parse_kernel_tuning_headers(headers)

    def test_malformed_context_falls_back_without_reaching_invoker_as_tuned(self) -> None:
        adapter = BackendKernelTuningAdapter()
        observed = []

        def invoke(shape, configuration):
            observed.append((shape, configuration))
            return "fallback"

        with adapter.bind_headers(
            {
                KERNEL_TUNING_ID_HEADER: "a" * 24,
                KERNEL_TUNING_CONTEXT_HEADER: "{}",
            }
        ):
            result = adapter.invoke_paged_attention(invoke, self.shape)
        self.assertEqual(result, "fallback")
        self.assertIsNone(observed[0][1])
        self.assertEqual(adapter.snapshot().rejected_contexts, 1)
        self.assertEqual(adapter.snapshot().shape_misses, 1)

    def test_exact_shape_winner_reaches_kernel_invocation(self) -> None:
        adapter = BackendKernelTuningAdapter()
        context = self.context("a", 256)

        def invoke(shape, configuration, marker):
            return shape, configuration, marker

        with adapter.bind_headers(context.to_http_headers()):
            shape, configuration, marker = adapter.invoke_paged_attention(
                invoke, self.shape, "called"
            )
        self.assertEqual(shape, self.shape)
        self.assertEqual(configuration, MetalThreadConfiguration(256, 256, 256))
        self.assertEqual(marker, "called")
        self.assertEqual(adapter.snapshot().shape_hits, 1)
        self.assertIsNone(adapter.configuration_for(self.shape))

    def test_asgi_middleware_isolates_concurrent_request_contexts(self) -> None:
        adapter = BackendKernelTuningAdapter()
        observed: dict[str, list[MetalThreadConfiguration | None]] = {"a": [], "b": []}

        async def app(scope, receive, send):
            request_id = scope["request_id"]
            observed[request_id].append(adapter.configuration_for(self.shape))
            await asyncio.sleep(0)
            observed[request_id].append(adapter.configuration_for(self.shape))

        middleware = KernelTuningASGIMiddleware(app, adapter)

        async def run(identity: str, width: int) -> None:
            headers = self.context(identity, width).to_http_headers()
            scope = {
                "type": "http",
                "request_id": identity,
                "headers": [
                    (key.lower().encode("ascii"), value.encode("ascii"))
                    for key, value in headers.items()
                ],
            }
            await middleware(scope, lambda: None, lambda message: None)

        async def run_concurrently() -> None:
            await asyncio.gather(run("a", 64), run("b", 256))

        asyncio.run(run_concurrently())
        self.assertEqual(observed["a"], [MetalThreadConfiguration(64, 64, 64)] * 2)
        self.assertEqual(observed["b"], [MetalThreadConfiguration(256, 256, 256)] * 2)

    def test_parser_rejects_changed_shape_field_contract(self) -> None:
        context = self.context("a", 128)
        headers = context.to_http_headers()
        payload = json.loads(headers[KERNEL_TUNING_CONTEXT_HEADER])
        payload["shape_fields"][0] = "changed"
        headers[KERNEL_TUNING_CONTEXT_HEADER] = json.dumps(payload, separators=(",", ":"))
        with self.assertRaises(ValueError):
            parse_kernel_tuning_headers(headers)

    def test_vllm_entrypoint_joins_middleware_to_kernel_call_site(self) -> None:
        observed = []
        response_messages = []

        def invoke(shape, configuration):
            observed.append(configuration)
            return "completed"

        async def app(scope, receive, send):
            self.assertEqual(invoke_paged_attention_kernel(invoke, self.shape), "completed")
            await send({"type": "http.response.start", "status": 200, "headers": []})

        middleware = VLLMAppleKernelTuningMiddleware(app)
        headers = self.context("c", 128).to_http_headers()
        scope = {
            "type": "http",
            "headers": [
                (key.lower().encode("ascii"), value.encode("ascii"))
                for key, value in headers.items()
            ],
        }

        async def send(message):
            response_messages.append(message)

        asyncio.run(middleware(scope, lambda: None, send))
        self.assertEqual(observed, [MetalThreadConfiguration(128, 128, 128)])
        response_headers = dict(response_messages[0]["headers"])
        self.assertEqual(
            response_headers[KERNEL_TUNING_ACCEPTED_HEADER.lower().encode("ascii")],
            b"c" * 24,
        )

    def test_middleware_does_not_ack_context_that_no_kernel_consumed(self) -> None:
        response_messages = []

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})

        async def send(message):
            response_messages.append(message)

        middleware = VLLMAppleKernelTuningMiddleware(app)
        headers = self.context("d", 128).to_http_headers()
        scope = {
            "type": "http",
            "headers": [
                (key.lower().encode("ascii"), value.encode("ascii"))
                for key, value in headers.items()
            ],
        }
        asyncio.run(middleware(scope, lambda: None, send))
        self.assertEqual(response_messages[0]["headers"], [])


if __name__ == "__main__":
    unittest.main()
