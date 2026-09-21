import unittest

from vllm_apple.heterogeneous_mlx_speculative import (
    _validate_request,
    commit_verified_tokens,
)


class HeterogeneousMLXSpeculativeTests(unittest.TestCase):
    def test_verifier_commits_only_agreed_prefix_and_correction(self) -> None:
        self.assertEqual(commit_verified_tokens((2, 3, 4), (2, 9, 4, 5)), (1, (2, 9)))
        self.assertEqual(commit_verified_tokens((2, 3), (8, 3, 5)), (0, (8,)))
        self.assertEqual(commit_verified_tokens((2, 3), (2, 3, 5)), (2, (2, 3, 5)))

    def test_rejects_unbounded_or_invalid_tokens(self) -> None:
        with self.assertRaisesRegex(ValueError, "verifier token sequence"):
            commit_verified_tokens((1, 2), (1, 2))
        with self.assertRaisesRegex(ValueError, "verifier token sequence"):
            commit_verified_tokens((1, -1), (1, 2, 3))
        with self.assertRaisesRegex(ValueError, "bounded heterogeneous"):
            _validate_request((1,), 65, 2)
        with self.assertRaisesRegex(ValueError, "bounded heterogeneous"):
            _validate_request((1,), 16, 9)


if __name__ == "__main__":
    unittest.main()
