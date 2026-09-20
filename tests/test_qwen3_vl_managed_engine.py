import base64
import io
import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace

import numpy as np
from PIL import Image

from vllm_apple.api import create_server
from vllm_apple.inference_request import InferenceRequestContext
from vllm_apple.managed_engine import ThreadAffineInferenceEngine
from vllm_apple.qwen3_vl_managed_engine import (
    Qwen3VLChatRuntime,
    Qwen3VLPersistentChatDelegate,
    _parse_chat_request,
)
from vllm_apple.service import RuntimeService


def png_bytes():
    output = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(output, format="PNG")
    return output.getvalue()


def request_body():
    encoded = base64.b64encode(png_bytes()).decode()
    return {
        "model": "fixture-qwen3-vl",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                {"type": "text", "text": "What color is this?"},
            ],
        }],
        "max_tokens": 8,
    }


class FakeImageProcessor:
    def __call__(self, *, images):
        return {
            "pixel_values": np.zeros((256, 1536), dtype=np.float16),
            "image_grid_thw": np.array([[1, 16, 16]]),
        }


class FakeEncoder:
    def __init__(self, calls):
        self.calls = calls

    def encode(self, pixel_path, transport):
        self.calls.append(("encode", threading.get_ident(), pixel_path.stat().st_size))
        return "encoded"

    def close(self):
        self.calls.append(("encoder_close", threading.get_ident(), 0))
        return True


class FakeBridge:
    def execute_inline(self, *, encode, consume, request_context, **kwargs):
        request_context.raise_if_cancelled()
        encode()
        bundle = SimpleNamespace(hidden_states="hidden", deepstack_visual_embeds=(1, 2, 3))
        return SimpleNamespace(output=consume(bundle))


class FakePipeline:
    def __init__(self, calls):
        self.calls = calls

    def close(self):
        self.calls.append(("pipeline_close", threading.get_ident(), 0))


def delegate(calls):
    original = SimpleNamespace(patch_embed="patch")
    model = SimpleNamespace(vision_tower=original, config="config")
    processor = SimpleNamespace(image_processor=FakeImageProcessor())
    runtime = Qwen3VLChatRuntime(
        SimpleNamespace(array=lambda value: value),
        np,
        Image.open,
        lambda processor, config, prompt, num_images: f"templated:{prompt}",
        lambda model, processor, prompt, **kwargs: (
            calls.append(("generate", threading.get_ident(), kwargs["max_tokens"]))
            or SimpleNamespace(text="red")
        ),
        requires_main_thread=False,
    )
    return Qwen3VLPersistentChatDelegate(
        model, processor, FakeBridge(), FakeEncoder(calls), FakePipeline(calls), runtime,
        model_id="fixture-qwen3-vl",
    )


class Qwen3VLManagedEngineTests(unittest.TestCase):
    def test_native_runtime_fails_closed_outside_main_thread(self):
        calls = []

        def unsafe_delegate():
            result = delegate(calls)
            object.__setattr__(result._runtime, "requires_main_thread", True)
            result.__init__(
                result._model, result._processor, result._bridge, result._encoder,
                result._pipeline, result._runtime, model_id="fixture-qwen3-vl",
            )
            return result

        with self.assertRaisesRegex(RuntimeError, "startup failed"):
            ThreadAffineInferenceEngine(unsafe_delegate)

    def test_parser_accepts_bounded_inline_image_and_rejects_remote_url(self):
        prompt, image = _parse_chat_request(request_body(), "fixture-qwen3-vl")
        self.assertEqual(prompt, "What color is this?")
        self.assertEqual(image, png_bytes())
        invalid = request_body()
        invalid["messages"][0]["content"][0]["image_url"]["url"] = "https://example.com/a.png"
        with self.assertRaisesRegex(ValueError, "inline PNG or JPEG"):
            _parse_chat_request(invalid, "fixture-qwen3-vl")

    def test_real_http_path_runs_delegate_and_cleanup_on_owner_thread(self):
        calls = []
        engine = ThreadAffineInferenceEngine(lambda: delegate(calls))
        server = create_server("127.0.0.1", 0, RuntimeService(engine=engine))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                data=json.dumps(request_body()).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                payload = json.load(response)
            self.assertEqual(payload["choices"][0]["message"]["content"], "red")
            self.assertEqual(payload["model"], "fixture-qwen3-vl")
        finally:
            server.shutdown()
            server.server_close()
            service_closed = server.service.close()
            thread.join(timeout=2)
        self.assertTrue(service_closed)
        self.assertEqual({call[1] for call in calls}, {engine.owner_thread_ident})
        self.assertIn(("encode", engine.owner_thread_ident, 256 * 1536 * 2), calls)

    def test_expired_request_never_reaches_encoder(self):
        calls = []
        engine = ThreadAffineInferenceEngine(lambda: delegate(calls))
        expired = InferenceRequestContext(
            "expired-request", time.monotonic() - 1, threading.Event()
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                engine.chat_completions_with_request_context(
                    request_body(), None, expired
                )
        finally:
            engine.close()
        self.assertFalse(any(call[0] == "encode" for call in calls))


if __name__ == "__main__":
    unittest.main()
