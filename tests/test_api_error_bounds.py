import unittest
from http import HTTPStatus
from unittest.mock import Mock

from vllm_apple.api import (
    MAX_ERROR_MESSAGE_CHARS,
    RuntimeRequestHandler,
    _bounded_error_message,
    _validate_cache_salt,
)


class APIErrorBoundsTests(unittest.TestCase):
    def test_small_error_is_unchanged(self) -> None:
        self.assertEqual(_bounded_error_message("invalid request"), "invalid request")

    def test_attacker_controlled_error_is_bounded(self) -> None:
        message = '"\\\x00' * 100_000
        bounded = _bounded_error_message(message)
        self.assertEqual(len(bounded), MAX_ERROR_MESSAGE_CHARS)
        self.assertTrue(bounded.endswith("...[truncated]"))
        self.assertNotEqual(bounded, message)

    def test_multibyte_error_is_bounded_without_corruption(self) -> None:
        bounded = _bounded_error_message("界" * 10_000)
        self.assertEqual(len(bounded), MAX_ERROR_MESSAGE_CHARS)
        self.assertTrue(bounded.endswith("...[truncated]"))

    def test_every_public_error_uses_the_bound(self) -> None:
        handler = object.__new__(RuntimeRequestHandler)
        handler._send = Mock()
        handler._error(HTTPStatus.BAD_REQUEST, "invalid_request", "x" * 10_000)
        payload = handler._send.call_args.args[1]
        self.assertEqual(len(payload["error"]["message"]), MAX_ERROR_MESSAGE_CHARS)

    def test_cache_salt_matches_upstream_vllm_boundary(self) -> None:
        for valid in (None, "safe-cache_salt-123", "a" * 128):
            _validate_cache_salt(valid)
        for invalid in ("", 1, "a" * 129, "bad@host", "../path", "bad\\path", "nul\0"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    _validate_cache_salt(invalid)
