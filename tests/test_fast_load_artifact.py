import hashlib
import tempfile
import unittest
from pathlib import Path

from vllm_apple.execution import ExecutionBackend
from vllm_apple.fast_load_artifact import (
    FastLoadSourceTensor,
    KernelCompatibility,
    KernelCompatibilityIndex,
    build_fast_load_artifact,
    load_fast_load_artifact,
)
from vllm_apple.numeric_routing import NumericFormat


class FastLoadArtifactTests(unittest.TestCase):
    def test_page_aligned_copy_verify_and_kernel_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            source.write_bytes(b"prefix" + b"a" * 17 + b"b" * 19)
            layout = "c" * 64
            tensors = (
                FastLoadSourceTensor(
                    "first", source, 6, 17, hashlib.sha256(b"a" * 17).hexdigest(),
                    NumericFormat.INT8, layout,
                ),
                FastLoadSourceTensor(
                    "second", source, 23, 19, hashlib.sha256(b"b" * 19).hexdigest(),
                    NumericFormat.INT8, layout,
                ),
            )
            artifact = build_fast_load_artifact(tensors, root / "artifact", page_size=4096)
            self.assertEqual(tuple(item.offset for item in artifact.entries), (0, 4096))
            self.assertEqual(load_fast_load_artifact(artifact.root).artifact_id,
                             artifact.artifact_id)
            kernel = KernelCompatibility(
                ExecutionBackend.NATIVE_METAL, "metal.v1", "d" * 64, "gemv",
                NumericFormat.INT8, layout, 4096, 1024,
            )
            selected = KernelCompatibilityIndex((kernel,)).select(
                artifact.entries[1], backend=ExecutionBackend.NATIVE_METAL,
                backend_version="metal.v1", environment_sha256="d" * 64,
                operator="gemv",
            )
            self.assertEqual(selected.kernel_id, kernel.kernel_id)

    def test_source_tamper_and_incompatible_kernel_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            source.write_bytes(b"value")
            tensor = FastLoadSourceTensor(
                "tensor", source, 0, 5, "0" * 64, NumericFormat.INT8, "a" * 64
            )
            with self.assertRaisesRegex(ValueError, "digest"):
                build_fast_load_artifact((tensor,), root / "bad", page_size=4096)

            valid = FastLoadSourceTensor(
                "tensor", source, 0, 5, hashlib.sha256(b"value").hexdigest(),
                NumericFormat.INT8, "a" * 64,
            )
            artifact = build_fast_load_artifact((valid,), root / "good", page_size=4096)
            kernel = KernelCompatibility(
                ExecutionBackend.NATIVE_METAL, "metal.v1", "b" * 64, "gemv",
                NumericFormat.INT4, "a" * 64, 4096, 1024,
            )
            with self.assertRaisesRegex(ValueError, "compatible"):
                KernelCompatibilityIndex((kernel,)).select(
                    artifact.entries[0], backend=ExecutionBackend.NATIVE_METAL,
                    backend_version="metal.v1", environment_sha256="b" * 64,
                    operator="gemv",
                )


if __name__ == "__main__":
    unittest.main()
