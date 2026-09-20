import json
import tempfile
import unittest
from pathlib import Path

from vllm_apple.qwen3_vl_transport import load_qwen3_vl_coreml_transport


class Qwen3VLTransportTests(unittest.TestCase):
    def test_public_transport_directory_is_rejected_before_mlx_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "transport"
            root.mkdir(mode=0o755)
            (root / "manifest.json").write_text(json.dumps({}))
            with self.assertRaisesRegex(ValueError, "not private"):
                load_qwen3_vl_coreml_transport(
                    root, object(), object(), expected_graph_id="a" * 64
                )


if __name__ == "__main__":
    unittest.main()
