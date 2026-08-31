from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import BinaryIO, Callable

from .model import (
    MAX_MODEL_CONFIG_BYTES,
    MAX_MODEL_CONFIG_DEPTH,
    MAX_MODEL_CONFIG_NODES,
    ModelInspectionError,
    inspect_model_architecture,
)


HUGGING_FACE_HOST = "huggingface.co"
MAX_MODEL_IDENTIFIER_BYTES = 512
MAX_REVISION_BYTES = 256
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_REVISION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")


class HuggingFaceMetadataError(ModelInspectionError):
    pass


@dataclass(frozen=True, slots=True)
class HuggingFaceModelMetadata:
    model_id: str
    requested_revision: str
    resolved_revision: str
    config_sha256: str
    config_bytes: int
    architecture: str
    modes: tuple[str, ...]
    required_features: tuple[str, ...]
    config: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "model_id": self.model_id,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "config_sha256": self.config_sha256,
            "config_bytes": self.config_bytes,
            "architecture": self.architecture,
            "modes": list(self.modes),
            "required_features": list(self.required_features),
            "weights_downloaded": False,
            "memory_fit_evaluated": False,
            "config": self.config,
        }


def fetch_hugging_face_metadata(
    model_id: str,
    *,
    revision: str = "main",
    timeout_seconds: float = 10.0,
    opener: Callable[..., BinaryIO] = urllib.request.urlopen,
) -> HuggingFaceModelMetadata:
    _validate_identifier(model_id, revision)
    if not 0 < timeout_seconds <= 60:
        raise ValueError("metadata timeout must be between zero and 60 seconds")
    owner, name = model_id.split("/", 1)
    url = (
        f"https://{HUGGING_FACE_HOST}/"
        f"{urllib.parse.quote(owner, safe='')}/{urllib.parse.quote(name, safe='')}/resolve/"
        f"{urllib.parse.quote(revision, safe='')}/config.json"
    )
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "vllm-apple-metadata/1",
        },
        method="GET",
    )
    try:
        with opener(request, timeout=timeout_seconds) as response:
            final_url = response.geturl()
            parsed = urllib.parse.urlsplit(final_url)
            if parsed.scheme != "https" or parsed.hostname != HUGGING_FACE_HOST:
                raise HuggingFaceMetadataError("metadata redirect left the trusted host")
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared_bytes = int(content_length)
                except ValueError as error:
                    raise HuggingFaceMetadataError("metadata content length is invalid") from error
                if not 1 <= declared_bytes <= MAX_MODEL_CONFIG_BYTES:
                    raise HuggingFaceMetadataError("metadata response exceeds the byte limit")
            payload = response.read(MAX_MODEL_CONFIG_BYTES + 1)
            resolved = response.headers.get("X-Repo-Commit")
    except HuggingFaceMetadataError:
        raise
    except (OSError, TimeoutError, urllib.error.URLError) as error:
        raise HuggingFaceMetadataError("Hugging Face metadata request failed") from error
    if not 1 <= len(payload) <= MAX_MODEL_CONFIG_BYTES:
        raise HuggingFaceMetadataError("metadata response exceeds the byte limit")
    try:
        config = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise HuggingFaceMetadataError("metadata response is not valid UTF-8 JSON") from error
    if not isinstance(config, dict):
        raise HuggingFaceMetadataError("model config must be an object")
    _validate_config_structure(config)
    if not isinstance(resolved, str) or not _COMMIT.fullmatch(resolved.lower()):
        raise HuggingFaceMetadataError("metadata response lacks a trusted resolved revision")
    capability = inspect_model_architecture(config)
    return HuggingFaceModelMetadata(
        model_id=model_id,
        requested_revision=revision,
        resolved_revision=resolved.lower(),
        config_sha256=hashlib.sha256(payload).hexdigest(),
        config_bytes=len(payload),
        architecture=capability.architecture,
        modes=capability.modes,
        required_features=capability.required_features,
        config=config,
    )


def _validate_identifier(model_id: str, revision: str) -> None:
    if (
        len(model_id.encode("utf-8")) > MAX_MODEL_IDENTIFIER_BYTES
        or not _MODEL_ID.fullmatch(model_id)
        or ".." in model_id
    ):
        raise ValueError("Hugging Face model identifier is invalid")
    if (
        len(revision.encode("utf-8")) > MAX_REVISION_BYTES
        or not _REVISION.fullmatch(revision)
        or ".." in revision
        or revision.endswith("/")
    ):
        raise ValueError("Hugging Face revision is invalid")


def _validate_config_structure(config: dict[str, object]) -> None:
    nodes = 0
    stack: list[tuple[object, int]] = [(config, 1)]
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > MAX_MODEL_CONFIG_NODES or depth > MAX_MODEL_CONFIG_DEPTH:
            raise HuggingFaceMetadataError("model config structure exceeds bounded limits")
        if isinstance(value, dict):
            stack.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            stack.extend((item, depth + 1) for item in value)
