import unittest

from vllm_apple.compat import assess_backend_versions, assess_platform_selection


class BackendVersionMatrixTests(unittest.TestCase):
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
