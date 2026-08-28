import stat
import tempfile
import unittest
from pathlib import Path

from vllm_apple.auth import SessionAuthenticator, SessionTokenError, load_or_create_token_file


class AuthenticationTests(unittest.TestCase):
    def test_bearer_token_is_required_and_compared_exactly(self) -> None:
        token = "a" * 32
        authenticator = SessionAuthenticator(token)
        self.assertFalse(authenticator.authorize(None))
        self.assertFalse(authenticator.authorize("Bearer " + "b" * 32))
        self.assertTrue(authenticator.authorize("Bearer " + token))

    def test_token_file_is_private_and_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.token"
            first = load_or_create_token_file(path)
            second = load_or_create_token_file(path)
            self.assertEqual(first, second)
            self.assertGreaterEqual(len(first), 32)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_insecure_token_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.token"
            path.write_text("a" * 32, encoding="utf-8")
            path.chmod(0o644)
            with self.assertRaises(SessionTokenError):
                load_or_create_token_file(path)


if __name__ == "__main__":
    unittest.main()

