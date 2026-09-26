from __future__ import annotations

import contextlib
import io
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

from vllm_apple.cli import main
from vllm_apple.rag import ABSTAIN, MAX_INPUT_BYTES, answer_rag, prepare_rag


def payload(language="ja"):
    return {"question": "対応言語は？", "language": language,
            "documents": [{"id": "manual:1", "title": "Manual", "text": "English、日本語、简体中文。"}]}


class RagTests(unittest.TestCase):
    def test_languages_and_provenance(self):
        for language in ABSTAIN:
            plan = prepare_rag(payload(language))
            self.assertIn(ABSTAIN[language], plan["request"]["messages"][0]["content"])
            self.assertFalse(plan["token_budget_verified"])
            self.assertEqual(plan["sources"][0]["reference"], "S1")
            self.assertEqual(plan["sources"], prepare_rag(payload(language))["sources"])
        changed = payload()
        changed["documents"][0]["text"] += " changed"
        self.assertNotEqual(prepare_rag(changed)["sources"], prepare_rag(payload())["sources"])

    def test_budget_skips_whole_chunks_preserving_order(self):
        data = payload()
        data["documents"].insert(0, {"id": "large", "title": "Large", "text": "x" * 500})
        result = prepare_rag(data, max_source_bytes=150)
        self.assertEqual(result["omitted_document_ids"], ["large"])
        self.assertEqual(result["sources"][0]["id"], "manual:1")
        self.assertEqual(result["sources"][0]["reference"], "S1")
        self.assertLessEqual(result["source_bytes"], 150)
        context = json.loads(result["request"]["messages"][1]["content"])
        self.assertEqual(context["sources"][0]["text"], data["documents"][1]["text"])

    def test_reject_invalid_inputs(self):
        bad = [None, {}, {**payload(), "language": "fr"}, {**payload(), "documents": [{}]},
               {**payload(), "documents": payload()["documents"] * 2},
               {**payload(), "question": ""}, {**payload(), "extra": True}]
        for data in bad:
            with self.subTest(data=data), self.assertRaises(ValueError):
                prepare_rag(data)
        for options in ({"max_tokens": True}, {"max_tokens": 0}, {"max_source_bytes": -1}):
            with self.assertRaises(ValueError):
                prepare_rag(payload(), **options)

    def test_no_sources_abstains_without_network(self):
        with patch("http.client.HTTPConnection", side_effect=AssertionError("network")):
            for language in ABSTAIN:
                result = answer_rag(payload(language), base_url="http://127.0.0.1:1", max_source_bytes=1)
                self.assertEqual(result["answer"], ABSTAIN[language])
                self.assertFalse(result["generation_performed"])
                self.assertNotIn("request", result)

    def test_reject_remote_and_nonroot_endpoints(self):
        for url in ("https://example.com", "http://127.0.0.1/v1", "http://user@localhost",
                    "http://localhost?token=secret", "http://localhost#fragment"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                answer_rag(payload(), base_url=url)

    def test_cli_prepare_and_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.json"
            source.write_text(json.dumps(payload()), encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["rag", str(source)]), 0)
            self.assertIn("request", json.loads(output.getvalue()))
            source.write_text("{", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["rag", str(source)]), 2)


class RagHTTPTests(unittest.TestCase):
    def setUp(self):
        owner = self
        self.response = {"choices": [{"message": {"content": "3言語です。[S1]"}, "finish_reason": "stop"}]}
        self.status = 200
        self.raw = None
        self.requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append((self.path, body, self.headers.get("Authorization")))
                self.send_response(owner.status)
                self.end_headers()
                self.wfile.write(owner.raw if owner.raw is not None else json.dumps(owner.response).encode())

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_local_generation_and_reference_mapping(self):
        result = answer_rag(payload(), base_url=self.url, session_token="test-token")
        self.assertEqual(result["status"], "references_valid")
        self.assertFalse(result["grounding_verified"])
        self.assertEqual(result["citations"][0]["id"], "manual:1")
        self.assertNotIn("request", result)
        path, body, authorization = self.requests[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(authorization, "Bearer test-token")
        self.assertFalse(body["stream"])

    def test_diagnostics_do_not_claim_grounding(self):
        for answer, finish, expected in (
            ("回答 [S99]", "stop", "invalid_references"),
            ("回答", "stop", "missing_citations"),
            ("回答 [S1]", "length", "incomplete"),
            (ABSTAIN["ja"], "stop", "abstained"),
            ("誤った回答 [S1]", "stop", "references_valid"),
        ):
            with self.subTest(expected=expected):
                self.response["choices"][0] = {"message": {"content": answer}, "finish_reason": finish}
                result = answer_rag(payload(), base_url=self.url)
                self.assertEqual(result["status"], expected)
                self.assertFalse(result["grounding_verified"])

    def test_redirect_is_not_followed(self):
        self.status = 302
        with self.assertRaisesRegex(ValueError, "302"):
            answer_rag(payload(), base_url=self.url)
        self.assertEqual(len(self.requests), 1)

    def test_bounded_response_and_invalid_envelope(self):
        self.raw = b"x" * (MAX_INPUT_BYTES + 1)
        with self.assertRaisesRegex(ValueError, "byte limit"):
            answer_rag(payload(), base_url=self.url)
        self.raw = b'{"choices": []}'
        with self.assertRaises(ValueError):
            answer_rag(payload(), base_url=self.url)
