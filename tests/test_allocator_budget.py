"""Allocator policy guards. Synthetic fixtures load no GPU weights."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vllm_apple.backend import BackendConfig, BackendConfigurationError


class AllocatorBudgetTests(unittest.TestCase):
    def test_disabled_policy_never_calls_allocator(self):
        from vllm_apple.mlx_kv_budget import configure_allocator

        mx = SimpleNamespace(set_cache_limit=Mock())
        self.assertEqual(
            configure_allocator(mx, None),
            dict(configured_limit_bytes=None, effective_limit_bytes=None),
        )
        mx.set_cache_limit.assert_not_called()

    def test_policy_sets_bytes_and_keeps_stricter_existing_limit(self):
        from vllm_apple.mlx_kv_budget import configure_allocator

        for previous in [0, 64 * 1024**2, 4 * 1024**3]:
            with self.subTest(previous=previous):
                mx = SimpleNamespace(set_cache_limit=Mock(return_value=previous))
                result = configure_allocator(mx, "256")
                self.assertEqual(result["configured_limit_bytes"], 256 * 1024**2)
                self.assertEqual(result["effective_limit_bytes"], min(previous, 256 * 1024**2))
                self.assertEqual(mx.set_cache_limit.call_args.args, (min(previous, 256 * 1024**2),))

    def test_invalid_values_never_touch_allocator(self):
        from vllm_apple.mlx_kv_budget import configure_allocator

        mx = SimpleNamespace(set_cache_limit=Mock())
        for raw in ["", "0", "1", "63", "4097", "+256", " 256", "256.0", "٢٥٦"]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                configure_allocator(mx, raw)
        mx.set_cache_limit.assert_not_called()

    def test_requires_reviewed_kv_launcher_and_snapshots_policy(self):
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "mlx_lm.server"
            (Path(d) / "python").touch()
            with (
                patch.dict(os.environ, {"VLLM_APPLE_ALLOCATOR_CACHE_MIB": "256"}, clear=True),
                self.assertRaises(BackendConfigurationError),
            ):
                BackendConfig("local", exe, backend_kind="mlx_lm")
            with patch.dict(
                os.environ,
                {"VLLM_APPLE_ALLOCATOR_CACHE_MIB": "256", "VLLM_APPLE_KV_CACHE_MIB": "1024"},
                clear=True,
            ):
                config = BackendConfig("local", exe, backend_kind="mlx_lm")
            self.assertEqual(config.allocator_cache_bytes, 256 * 1024**2)
            self.assertTrue(any(x.endswith("mlx_kv_budget.py") for x in config.command()))

    def test_invalid_policy_rejected_before_launch(self):
        for raw in ["0", "63", "4097", "256.0", "+256"]:
            with (
                self.subTest(raw=raw),
                patch.dict(
                    os.environ,
                    {"VLLM_APPLE_ALLOCATOR_CACHE_MIB": raw, "VLLM_APPLE_KV_CACHE_MIB": "1024"},
                    clear=True,
                ),
                self.assertRaises(BackendConfigurationError),
            ):
                BackendConfig("local", Path("/backend"), backend_kind="mlx_lm")
