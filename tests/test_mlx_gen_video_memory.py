import unittest

from vllm_apple.mlx_gen_video_memory import estimate_mlx_gen_video_resident_bytes
from vllm_apple.types import GIB


class MLXGenVideoMemoryTests(unittest.TestCase):
    def test_low_ram_estimate_uses_largest_phase_not_all_components(self) -> None:
        artifact = {
            "components": [
                {"role": "denoiser", "artifact_bytes": 5 * GIB},
                {"role": "text_encoder", "artifact_bytes": 10 * GIB},
                {"role": "vae", "artifact_bytes": GIB},
            ]
        }
        estimate = estimate_mlx_gen_video_resident_bytes(
            artifact, width=640, height=384, frames=33
        )
        self.assertEqual(estimate, 10 * GIB + GIB + 640 * 384 * 4 * 33)
        self.assertLess(estimate, 12 * GIB)

    def test_missing_component_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing"):
            estimate_mlx_gen_video_resident_bytes(
                {"components": [{"role": "denoiser", "artifact_bytes": GIB}]},
                width=640,
                height=360,
                frames=33,
            )


if __name__ == "__main__":
    unittest.main()
