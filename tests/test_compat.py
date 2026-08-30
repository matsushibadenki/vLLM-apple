import unittest
import json
import os
import tempfile
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from vllm_apple.compat import (
    _decode_mlx_probe,
    assess_backend_versions,
    assess_platform_selection,
    inspect_mlx_lm_backend,
)


class BackendVersionMatrixTests(unittest.TestCase):
    def test_mlx_probe_declares_only_bounded_explicit_features(self) -> None:
        version, features = _decode_mlx_probe(
            json.dumps(
                {
                    "version": "0.26.3",
                    "architecture_features": ["gated_deltanet", "gated_deltanet"],
                }
            )
        )
        self.assertEqual(version, "0.26.3")
        self.assertEqual(features, ("gated_deltanet",))
        with self.assertRaisesRegex(ValueError, "features are invalid"):
            _decode_mlx_probe(
                json.dumps({"version": "0.26.3", "architecture_features": ["Unsafe-Name"]})
            )

    def test_mlx_backend_exposes_probe_feature_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "mlx_lm.server"
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            os.chmod(executable, 0o700)
            payload = json.dumps(
                {
                    "version": "0.26.3",
                    "architecture_features": ["native_long_context", "gated_deltanet"],
                }
            )
            completed = CompletedProcess([], 0, stdout=payload, stderr="")
            with patch("vllm_apple.compat.subprocess.run", return_value=completed):
                result = inspect_mlx_lm_backend(executable)
        self.assertTrue(result.compatible)
        self.assertEqual(
            result.architecture_features,
            ("gated_deltanet", "native_long_context"),
        )

    def test_verified_vllm_metal_stack_is_accepted(self) -> None:
        self.assertEqual(
            assess_backend_versions(
                vllm_version="0.27.1",
                vllm_metal_version="0.3.0.dev20260827",
                transformers_version="5.12.1",
            ),
            (),
        )

    def test_vllm_028_and_transformers_515_are_unverified(self) -> None:
        issues = assess_backend_versions(
            vllm_version="0.28.0",
            vllm_metal_version="0.3.0",
            transformers_version="5.15.0",
        )
        self.assertEqual(
            issues,
            (
                "vllm_version_outside_verified_matrix",
                "transformers_version_outside_verified_matrix",
            ),
        )

    def test_development_suffix_does_not_hide_release_compatibility(self) -> None:
        self.assertEqual(
            assess_backend_versions(
                vllm_version="0.26.0+metal",
                vllm_metal_version="0.3.0.dev1",
                transformers_version="5.5.3rc1",
            ),
            (),
        )

    def test_unparseable_installed_version_fails_closed(self) -> None:
        self.assertEqual(
            assess_backend_versions(
                vllm_version="main",
                vllm_metal_version="0.3.0",
                transformers_version="5.12.1",
            ),
            ("vllm_version_unparseable",),
        )

    def test_metal_platform_selection_is_required(self) -> None:
        self.assertEqual(
            assess_platform_selection(
                platform_module="vllm_metal.platform",
                platform_class="MetalPlatform",
                platform_is_cpu=False,
            ),
            (),
        )
        self.assertEqual(
            assess_platform_selection(
                platform_module="vllm.platforms.cpu",
                platform_class="CpuPlatform",
                platform_is_cpu=True,
            ),
            ("vllm_metal_platform_not_selected",),
        )

    def test_unknown_platform_selection_fails_closed(self) -> None:
        self.assertEqual(
            assess_platform_selection(
                platform_module=None,
                platform_class=None,
                platform_is_cpu=None,
            ),
            ("vllm_platform_selection_unknown",),
        )


if __name__ == "__main__":
    unittest.main()
