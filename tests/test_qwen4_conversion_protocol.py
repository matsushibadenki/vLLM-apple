import json
import os
import tempfile
import unittest
from pathlib import Path

from vllm_apple.qwen4_conversion_protocol import (
    Qwen4IsolatedConversionAdapter,
    build_qwen4_conversion_request,
)


class Qwen4ConversionProtocolTests(unittest.TestCase):
    def request(self, root: Path) -> dict[str, object]:
        return build_qwen4_conversion_request(
            stage_root=root,
            tensor_name="model.tensor.weight",
            contract_id="a" * 64,
            load_plan_id="b" * 64,
            target_dtype="BF16",
            maximum_artifact_bytes=1024,
            memory_capacity_bytes=512,
            axis0_slice=(1, 2),
        )

    def helper(self, root: Path, response: dict[str, object]) -> Path:
        helper = root / "helper"
        helper.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "json.load(sys.stdin)\n"
            f"print({json.dumps(json.dumps(response))})\n",
            encoding="utf-8",
        )
        helper.chmod(0o700)
        return helper

    def test_accepts_bounded_response_bound_to_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = self.request(root)
            response = {
                "abi_version": 1,
                "passed": True,
                "backend": "test",
                "backend_version": "1",
                "contract_id": "a" * 64,
                "load_plan_id": "b" * 64,
                "target_dtype": "BF16",
                "output_shape": [2, 4],
                "output_bytes": 16,
                "output_digest": "c" * 64,
                "peak_reserved_bytes": 32,
                "stores_tensor_values": False,
            }
            adapter = Qwen4IsolatedConversionAdapter(self.helper(root, response))
            self.assertEqual(adapter.convert(request)["output_digest"], "c" * 64)

    def test_rejects_response_not_bound_to_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = self.request(root)
            response = {
                "abi_version": 1,
                "passed": True,
                "backend": "test",
                "backend_version": "1",
                "contract_id": "d" * 64,
                "load_plan_id": "b" * 64,
                "target_dtype": "BF16",
                "output_shape": [1],
                "output_bytes": 2,
                "output_digest": "c" * 64,
                "peak_reserved_bytes": 2,
                "stores_tensor_values": False,
            }
            adapter = Qwen4IsolatedConversionAdapter(self.helper(root, response))
            with self.assertRaisesRegex(ValueError, "bound"):
                adapter.convert(request)

    def test_rejects_non_private_or_non_executable_helper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory) / "helper"
            helper.write_text("exit 0\n", encoding="utf-8")
            helper.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "executable"):
                Qwen4IsolatedConversionAdapter(helper)
            self.assertEqual(os.getuid(), helper.stat().st_uid)


if __name__ == "__main__":
    unittest.main()
