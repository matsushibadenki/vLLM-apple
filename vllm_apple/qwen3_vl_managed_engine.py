"""Managed Qwen3-VL chat delegate for persistent Core ML vision inference."""
from __future__ import annotations

import io
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .execution import ExecutionBackend
from .inference_request import InferenceRequestContext
from .qwen3_vl_embedding import Qwen3VLANEGPUPipeline
from .qwen3_vl_persistent_encoder import Qwen3VLPersistentEncoder
from .vision_frontend import parse_vision_chat_request


@dataclass(frozen=True, slots=True)
class Qwen3VLChatRuntime:
    mx: object
    np: object
    image_open: Callable[[io.BytesIO], object]
    apply_chat_template: Callable[..., str]
    generate: Callable[..., object]
    requires_main_thread: bool = True


class _FixedVisionTower:
    def __init__(self, original: object, result: tuple[object, list[object]]) -> None:
        self.patch_embed = original.patch_embed
        self._result = result

    def __call__(self, *_args: object, **_kwargs: object):
        return self._result


class Qwen3VLPersistentChatDelegate:
    """One-owner-thread OpenAI chat adapter over the persistent ANE/GPU path."""

    def __init__(
        self,
        model: object,
        processor: object,
        bridge: Qwen3VLANEGPUPipeline,
        encoder: Qwen3VLPersistentEncoder,
        pipeline: object,
        runtime: Qwen3VLChatRuntime,
        *,
        model_id: str,
        encoder_memory_bytes: int = 216_449_024,
        llm_memory_bytes: int = 1_000_000_000,
    ) -> None:
        if (
            not model_id
            or not callable(getattr(runtime.mx, "array", None))
            or not callable(getattr(runtime.np, "asarray", None))
            or not callable(getattr(processor, "image_processor", None))
            or not callable(getattr(pipeline, "close", None))
            or encoder_memory_bytes < 0
            or llm_memory_bytes < 0
        ):
            raise ValueError("invalid Qwen3-VL managed chat configuration")
        if runtime.requires_main_thread and threading.current_thread() is not threading.main_thread():
            raise RuntimeError(
                "Homebrew MLX Qwen3-VL requires main-process-thread isolation"
            )
        self._model = model
        self._processor = processor
        self._bridge = bridge
        self._encoder = encoder
        self._pipeline = pipeline
        self._runtime = runtime
        self._model_id = model_id
        self._encoder_memory_bytes = encoder_memory_bytes
        self._llm_memory_bytes = llm_memory_bytes
        self._owner_ident = threading.get_ident()
        self._closed = False

    @property
    def ready(self) -> bool:
        return not self._closed

    def models(self) -> list[dict[str, Any]]:
        self._require_owner()
        return [{"id": self._model_id, "object": "model", "owned_by": "vllm-apple"}]

    def chat_completions_with_request_context(
        self,
        request: dict[str, Any],
        _kernel_context: object | None,
        request_context: InferenceRequestContext,
    ) -> dict[str, Any]:
        self._require_owner()
        request_context.raise_if_cancelled()
        prompt_text, image_bytes = _parse_chat_request(request, self._model_id)
        workspace = Path(tempfile.mkdtemp(prefix="vllm-apple-qwen3-vl-request-"))
        workspace.chmod(0o700)
        original = self._model.vision_tower
        try:
            with self._runtime.image_open(io.BytesIO(image_bytes)) as opened:
                processed = self._processor.image_processor(
                    images=[opened.convert("RGB")]
                )
            pixels = self._runtime.np.asarray(processed["pixel_values"]).astype(
                self._runtime.np.float16
            ).reshape(256, 1536)
            grid = tuple(
                int(value)
                for value in self._runtime.np.asarray(
                    processed["image_grid_thw"]
                ).reshape(-1).tolist()
            )
            if grid != (1, 16, 16):
                raise ValueError("Qwen3-VL image does not match the fixed Core ML profile")
            pixel_path = workspace / "pixel_values.fp16"
            pixels.tofile(pixel_path)
            pixel_path.chmod(0o600)
            image_path = workspace / "image"
            with image_path.open("xb") as stream:
                stream.write(image_bytes)
            image_path.chmod(0o600)
            transport = workspace / "transport"
            transport.mkdir(mode=0o700)
            prompt = self._runtime.apply_chat_template(
                self._processor, self._model.config, prompt_text, num_images=1
            )

            def consume(bundle):
                request_context.raise_if_cancelled()
                candidate = (bundle.hidden_states, list(bundle.deepstack_visual_embeds))
                self._model.vision_tower = _FixedVisionTower(original, candidate)
                response = self._runtime.generate(
                    self._model,
                    self._processor,
                    prompt,
                    image=[str(image_path)],
                    max_tokens=_max_tokens(request),
                    temperature=0,
                    verbose=False,
                )
                return response.text

            result = self._bridge.execute_inline(
                grid_thw=grid,
                encoder_memory_bytes=self._encoder_memory_bytes,
                llm_backend=ExecutionBackend.NATIVE_MLX,
                llm_memory_bytes=self._llm_memory_bytes,
                encode=lambda: self._encoder.encode(pixel_path, transport),
                consume=consume,
                request_context=request_context,
            )
            return {
                "model": self._model_id,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": result.output},
                    "finish_reason": "stop",
                }],
            }
        finally:
            self._model.vision_tower = original
            shutil.rmtree(workspace, ignore_errors=True)

    def close(self) -> bool:
        self._require_owner()
        if self._closed:
            return True
        self._closed = True
        self._pipeline.close()
        return self._encoder.close()

    def _require_owner(self) -> None:
        if threading.get_ident() != self._owner_ident:
            raise RuntimeError("Qwen3-VL managed chat must run on its owner thread")


def load_qwen3_vl_chat_runtime() -> Qwen3VLChatRuntime:
    """Load optional Homebrew MLX dependencies only inside the model owner."""
    import mlx.core as mx
    import numpy as np
    from PIL import Image
    from mlx_vlm import apply_chat_template, generate

    return Qwen3VLChatRuntime(mx, np, Image.open, apply_chat_template, generate)


def _parse_chat_request(request: dict[str, Any], model_id: str) -> tuple[str, bytes]:
    try:
        parsed = parse_vision_chat_request(request, model_id, max_images=1)
    except ValueError as error:
        raise ValueError(str(error).replace("vision chat", "Qwen3-VL chat")) from error
    return parsed.prompt, parsed.images[0].data


def _max_tokens(request: dict[str, Any]) -> int:
    value = request.get("max_completion_tokens", request.get("max_tokens", 32))
    if type(value) is not int or not 1 <= value <= 256:
        raise ValueError("Qwen3-VL max tokens must be between 1 and 256")
    return value
