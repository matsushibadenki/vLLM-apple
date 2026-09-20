import base64
import unittest

from vllm_apple.vision_frontend import (
    VisionPreprocessSpec,
    parse_vision_chat_request,
    preprocess_vision_image,
)
from vllm_apple.vision_smoke import solid_png


PNG = solid_png((255, 0, 0))


def request(*images: bytes):
    content = [{"type": "text", "text": "  describe these images  "}]
    content.extend({
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + base64.b64encode(image).decode()},
    } for image in images)
    return {"model": "vision", "messages": [{"role": "user", "content": content}]}


class VisionFrontendTests(unittest.TestCase):
    def test_parses_multiple_bounded_images_and_records_digest(self):
        parsed = parse_vision_chat_request(request(PNG, PNG), "vision", max_images=2)
        self.assertEqual(parsed.prompt, "describe these images")
        self.assertEqual(len(parsed.images), 2)
        self.assertEqual(len(parsed.images[0].sha256), 64)
        self.assertEqual(parsed.images[0].media_type, "image/png")

    def test_rejects_remote_mislabeled_and_over_budget_images(self):
        remote = request(PNG)
        remote["messages"][0]["content"][1]["image_url"]["url"] = "https://example.com/a.png"
        with self.assertRaisesRegex(ValueError, "inline PNG or JPEG"):
            parse_vision_chat_request(remote, "vision")
        mislabeled = request(b"not-a-png")
        with self.assertRaisesRegex(ValueError, "content is invalid"):
            parse_vision_chat_request(mislabeled, "vision")
        with self.assertRaisesRegex(ValueError, "budget exceeded"):
            parse_vision_chat_request(request(PNG, PNG), "vision", max_images=1)

    def test_resize_normalize_patchify_has_stable_shape(self):
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            self.skipTest("optional vision runtime is unavailable")
        parsed = parse_vision_chat_request(request(PNG), "vision")
        spec = VisionPreprocessSpec(
            width=8,
            height=8,
            patch_size=4,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )
        result = preprocess_vision_image(
            parsed.images[0], spec, image_open=Image.open, np=np
        )
        self.assertEqual(result.image_shape, (8, 8, 3))
        self.assertEqual(result.patch_shape, (4, 4))
        self.assertEqual(result.patch_count, 4)
        self.assertEqual(result.patches.shape, (4, 48))
        self.assertEqual(result.patches.dtype, np.float16)

    def test_invalid_preprocess_geometry_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "preprocessing specification"):
            VisionPreprocessSpec(
                width=7,
                height=8,
                patch_size=4,
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            )


if __name__ == "__main__":
    unittest.main()
