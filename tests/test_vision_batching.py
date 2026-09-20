import hashlib
import unittest

from vllm_apple.vision_batching import (
    VisionBatchCompatibility,
    VisionBatchLimits,
    VisionBatchRequest,
    plan_multimodal_batches,
)
from vllm_apple.vision_frontend import VisionImageInput


def image(value: bytes) -> VisionImageInput:
    return VisionImageInput("image/png", value, hashlib.sha256(value).hexdigest())


def compatibility(revision: str = "r") -> VisionBatchCompatibility:
    return VisionBatchCompatibility(revision, "preprocess", "encoder", (224, 224, 3))


def request(identifier: str, *, count: int = 1, revision: str = "r") -> VisionBatchRequest:
    return VisionBatchRequest(
        identifier,
        compatibility(revision),
        tuple(image(f"image-{identifier}-{index}".encode()) for index in range(count)),
        196,
    )


class VisionBatchingTests(unittest.TestCase):
    def test_groups_compatible_requests_and_preserves_stable_order(self):
        batches = plan_multimodal_batches([
            request("a"), request("other", revision="r2"), request("b", count=2)
        ])
        self.assertEqual([batch.request_ids for batch in batches], [("a", "b"), ("other",)])
        self.assertEqual(batches[0].image_count, 3)
        self.assertEqual(batches[0].patch_count, 3 * 196)

    def test_splits_before_limit_without_splitting_a_request(self):
        limits = VisionBatchLimits(
            maximum_requests=2,
            maximum_images=3,
            maximum_patches=600,
            maximum_encoded_bytes=1024,
        )
        batches = plan_multimodal_batches(
            [request("a", count=2), request("b", count=2)], limits
        )
        self.assertEqual([batch.request_ids for batch in batches], [("a",), ("b",)])

    def test_rejects_oversize_request_and_duplicate_identifier(self):
        limits = VisionBatchLimits(maximum_images=1)
        with self.assertRaisesRegex(ValueError, "exceeds batch limits"):
            plan_multimodal_batches([request("large", count=2)], limits)
        with self.assertRaisesRegex(ValueError, "identifiers must be unique"):
            plan_multimodal_batches([request("same"), request("same")])

    def test_empty_input_has_no_batches(self):
        self.assertEqual(plan_multimodal_batches([]), ())


if __name__ == "__main__":
    unittest.main()
