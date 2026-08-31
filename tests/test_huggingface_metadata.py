import io
import json
import unittest
import urllib.error
from email.message import Message

from tests.schema_validator import validate_instance
from tests.test_schemas import load_schema
from vllm_apple.huggingface_metadata import (
    HuggingFaceMetadataError,
    fetch_hugging_face_metadata,
)


class FakeResponse(io.BytesIO):
    def __init__(
        self,
        payload: bytes,
        *,
        url: str = "https://huggingface.co/api/resolve-cache/models/org/model/commit/config.json",
        commit: str | None = "a" * 40,
        declared_length: str | None = None,
    ) -> None:
        super().__init__(payload)
        self._url = url
        self.headers = Message()
        self.headers["Content-Length"] = declared_length or str(len(payload))
        if commit is not None:
            self.headers["X-Repo-Commit"] = commit

    def geturl(self) -> str:
        return self._url


class HuggingFaceMetadataTests(unittest.TestCase):
    def test_fetches_config_without_weights_and_binds_resolved_revision(self) -> None:
        payload = json.dumps(
            {
                "model_type": "deepseek_v3",
                "num_hidden_layers": 12,
                "kv_lora_rank": 256,
                "qk_rope_head_dim": 64,
            }
        ).encode()
        requests = []

        def open_request(request, *, timeout):
            requests.append((request, timeout))
            return FakeResponse(payload)

        report = fetch_hugging_face_metadata(
            "org/model", revision="refs/pr/2", timeout_seconds=3, opener=open_request
        )
        self.assertEqual(report.resolved_revision, "a" * 40)
        self.assertEqual(report.architecture, "deepseek_v3")
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0][1], 3)
        self.assertIn("/refs%2Fpr%2F2/config.json", requests[0][0].full_url)
        serialized = report.to_dict()
        self.assertFalse(serialized["weights_downloaded"])
        self.assertFalse(serialized["memory_fit_evaluated"])
        validate_instance(
            serialized, load_schema("runtime/huggingface-metadata-v1.schema.json")
        )

    def test_rejects_untrusted_redirect_oversize_and_missing_commit(self) -> None:
        payload = b'{"model_type":"test"}'
        with self.assertRaisesRegex(HuggingFaceMetadataError, "trusted host"):
            fetch_hugging_face_metadata(
                "org/model",
                opener=lambda *_args, **_kwargs: FakeResponse(
                    payload, url="https://example.invalid/config.json"
                ),
            )
        with self.assertRaisesRegex(HuggingFaceMetadataError, "byte limit"):
            fetch_hugging_face_metadata(
                "org/model",
                opener=lambda *_args, **_kwargs: FakeResponse(
                    payload, declared_length=str(1024 * 1024 + 1)
                ),
            )
        with self.assertRaisesRegex(HuggingFaceMetadataError, "resolved revision"):
            fetch_hugging_face_metadata(
                "org/model",
                opener=lambda *_args, **_kwargs: FakeResponse(payload, commit=None),
            )

    def test_rejects_invalid_identifiers_network_failure_and_deep_config(self) -> None:
        for model, revision in (("../model", "main"), ("org/model", "../main")):
            with self.assertRaises(ValueError):
                fetch_hugging_face_metadata(model, revision=revision)
        with self.assertRaisesRegex(HuggingFaceMetadataError, "request failed"):
            fetch_hugging_face_metadata(
                "org/model",
                opener=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    urllib.error.URLError("offline")
                ),
            )
        nested: object = "leaf"
        for _ in range(20):
            nested = {"nested": nested}
        with self.assertRaisesRegex(HuggingFaceMetadataError, "bounded limits"):
            fetch_hugging_face_metadata(
                "org/model",
                opener=lambda *_args, **_kwargs: FakeResponse(json.dumps(nested).encode()),
            )


if __name__ == "__main__":
    unittest.main()
