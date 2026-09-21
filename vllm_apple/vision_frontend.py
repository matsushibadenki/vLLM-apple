"""Bounded, model-neutral image input and preprocessing contracts."""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
from dataclasses import dataclass
from typing import Any, Callable

_IMAGE_PREFIXES = {
    "data:image/png;base64,": ("image/png", b"\x89PNG\r\n\x1a\n"),
    "data:image/jpeg;base64,": ("image/jpeg", b"\xff\xd8\xff"),
}


@dataclass(frozen=True, slots=True)
class VisionImageInput:
    media_type: str
    data: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class VisionChatInput:
    prompt: str
    images: tuple[VisionImageInput, ...]


@dataclass(frozen=True, slots=True)
class VisionPreprocessSpec:
    width: int
    height: int
    patch_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    def __post_init__(self) -> None:
        if (
            not 1 <= self.width <= 4096
            or not 1 <= self.height <= 4096
            or not 1 <= self.patch_size <= 256
            or self.width % self.patch_size
            or self.height % self.patch_size
            or len(self.mean) != 3
            or len(self.std) != 3
            or any(value <= 0 for value in self.std)
        ):
            raise ValueError("invalid vision preprocessing specification")


@dataclass(frozen=True, slots=True)
class VisionPreprocessResult:
    patches: object
    image_shape: tuple[int, int, int]
    patch_shape: tuple[int, int]
    patch_count: int


def parse_vision_chat_request(
    request: dict[str, Any],
    model_id: str,
    *,
    max_images: int = 8,
    max_image_bytes: int = 3 * 1024 * 1024,
    max_total_image_bytes: int = 12 * 1024 * 1024,
) -> VisionChatInput:
    """Parse bounded OpenAI-style chat content without fetching remote resources."""
    if (
        request.get("model") != model_id
        or request.get("stream") is True
        or not 1 <= max_images <= 32
        or max_image_bytes < 1
        or max_total_image_bytes < max_image_bytes
    ):
        raise ValueError("unsupported vision chat request")
    messages = request.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 32:
        raise ValueError("vision chat messages are invalid")

    texts: list[str] = []
    images: list[VisionImageInput] = []
    total_bytes = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user"}:
            raise ValueError("vision chat role is unsupported")
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
            continue
        if not isinstance(content, list) or not 1 <= len(content) <= 64:
            raise ValueError("vision chat content is invalid")
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("vision chat content part is invalid")
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
                continue
            if part.get("type") != "image_url":
                raise ValueError("vision chat content part is unsupported")
            image_url = part.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else None
            image = _decode_inline_image(url, max_image_bytes)
            images.append(image)
            total_bytes += len(image.data)
            if len(images) > max_images or total_bytes > max_total_image_bytes:
                raise ValueError("vision chat image budget exceeded")

    prompt = "\n".join(text.strip() for text in texts if text.strip())
    if not prompt or not images:
        raise ValueError("vision chat requires text and at least one image")
    return VisionChatInput(prompt, tuple(images))


def preprocess_vision_image(
    image: VisionImageInput,
    spec: VisionPreprocessSpec,
    *,
    image_open: Callable[[io.BytesIO], object],
    np: object,
) -> VisionPreprocessResult:
    """Decode, resize, normalize and patchify one RGB image with bounded shapes."""
    with image_open(io.BytesIO(image.data)) as opened:
        source = opened.convert("RGB")
        resized = source.resize((spec.width, spec.height))
        pixels = np.asarray(resized, dtype=np.float32)
    if tuple(pixels.shape) != (spec.height, spec.width, 3):
        raise ValueError("decoded image shape does not match preprocessing specification")
    pixels = pixels / np.float32(255.0)
    mean = np.asarray(spec.mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(spec.std, dtype=np.float32).reshape(1, 1, 3)
    normalized = (pixels - mean) / std
    rows = spec.height // spec.patch_size
    columns = spec.width // spec.patch_size
    patches = normalized.reshape(
        rows, spec.patch_size, columns, spec.patch_size, 3
    ).transpose(0, 2, 1, 3, 4).reshape(rows * columns, -1)
    return VisionPreprocessResult(
        patches=patches.astype(np.float16),
        image_shape=(spec.height, spec.width, 3),
        patch_shape=(spec.patch_size, spec.patch_size),
        patch_count=rows * columns,
    )


def _decode_inline_image(url: object, max_image_bytes: int) -> VisionImageInput:
    if not isinstance(url, str):
        raise ValueError("vision chat image URL is invalid")
    match = next(
        ((prefix, metadata) for prefix, metadata in _IMAGE_PREFIXES.items() if url.startswith(prefix)),
        None,
    )
    if match is None:
        raise ValueError("vision chat image must be an inline PNG or JPEG")
    prefix, (media_type, signature) = match
    encoded = url[len(prefix):]
    if not encoded or len(encoded) > ((max_image_bytes + 2) // 3) * 4 + 4:
        raise ValueError("vision chat image size is invalid")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("vision chat image encoding is invalid") from error
    if not 1 <= len(data) <= max_image_bytes or not data.startswith(signature):
        raise ValueError("vision chat image content is invalid")
    return VisionImageInput(media_type, data, hashlib.sha256(data).hexdigest())
