import unittest

from vllm_apple.numeric_formats import NumericFormatDescriptor, convert_nvfp4_to_int8
from vllm_apple.numeric_precision import NumericPrecisionPolicy, PrecisionExecutionContract
from vllm_apple.qwen4_mlx_resident_backend import (
    MLXResidentTensorResource,
    Qwen4MLXNumericResidentBackend,
)


class Qwen4MLXResidentBackendTests(unittest.TestCase):
    def fixture(self):
        tensor = convert_nvfp4_to_int8(
            NumericFormatDescriptor("nvfp4_e2m1", 1), bytes([2]), bytes([56]), 1)
        return tensor, PrecisionExecutionContract(
            tensor.target_digest, "F16", NumericPrecisionPolicy())

    def test_rejects_before_importing_mlx(self):
        backend = Qwen4MLXNumericResidentBackend()
        tensor, contract = self.fixture()
        with self.assertRaisesRegex(ValueError, "only accepts"):
            backend.load(())
        with self.assertRaisesRegex(ValueError, "mismatch"):
            backend.load_scaled_int8(
                tensor, target_dtype="F32", execution_contract=contract, reserved_bytes=100)
        with self.assertRaisesRegex(MemoryError, "reservation"):
            backend.load_scaled_int8(
                tensor, target_dtype="F16", execution_contract=contract, reserved_bytes=8)

    def test_resource_release_is_explicit_and_single_use(self):
        backend = Qwen4MLXNumericResidentBackend()
        resource = MLXResidentTensorResource(object())
        backend.release(resource)
        self.assertTrue(resource.released)
        for invalid in (resource, object(), None):
            with self.assertRaises(ValueError):
                backend.release(invalid)


if __name__ == "__main__":
    unittest.main()
