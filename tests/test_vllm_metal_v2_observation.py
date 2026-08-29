from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm_apple.vllm_metal_v2_observation import (
    load_v2_observations,
    record_v2_observed_shape,
)
from vllm_apple.vllm_metal_v2_tuning import V2PagedAttentionShape


class NativeV2ObservationTests(unittest.TestCase):
    def test_records_unique_shapes_in_private_atomic_artifact(self) -> None:
        shape = V2PagedAttentionShape(
            1024, 1024, 1, 8, 4, 256, 16, 10, "bfloat16", "bfloat16"
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "vllm_apple.vllm_metal_v2_observation.default_application_support",
            return_value=Path(directory),
        ):
            path = record_v2_observed_shape(
                shape, hardware_fingerprint="hardware", source_fingerprint="source"
            )
            record_v2_observed_shape(
                shape, hardware_fingerprint="hardware", source_fingerprint="source"
            )
            loaded = load_v2_observations(
                path, hardware_fingerprint="hardware", source_fingerprint="source"
            )
            self.assertEqual(loaded, (shape,))
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
