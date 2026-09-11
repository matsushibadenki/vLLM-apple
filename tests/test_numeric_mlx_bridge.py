import struct
import unittest
from unittest.mock import patch

from vllm_apple.numeric_formats import (
    NumericFormatDescriptor, TensorGeometry, convert_nvfp4_to_int8,
)
from vllm_apple.qwen4_mlx_conversion_worker import Qwen4MLXCorrectnessConverter
from vllm_apple.numeric_precision import NumericPrecisionPolicy


class NumericMLXBridgeTests(unittest.TestCase):
    def test_policy_rejects_before_backend_and_checks_backend_digest(self):
        from types import SimpleNamespace
        converter = Qwen4MLXCorrectnessConverter()
        descriptor = NumericFormatDescriptor("nvfp4_e2m1", 1)
        rounded = convert_nvfp4_to_int8(descriptor, bytes([2]), bytes([56]), 1.0001)
        with patch.object(converter, "convert") as consume:
            with self.assertRaises(ValueError):
                converter.convert_scaled_int8(rounded, target_dtype="F16", reserved_bytes=4096,
                                              precision_policy=NumericPrecisionPolicy())
            consume.assert_not_called()
        exact = convert_nvfp4_to_int8(descriptor, bytes([2]), bytes([56]), 1)
        with patch.object(converter, "convert", return_value=SimpleNamespace(output_digest="wrong")):
            with self.assertRaisesRegex(ValueError, "differs"):
                converter.convert_scaled_int8(exact, target_dtype="F16", reserved_bytes=4096,
                                              precision_policy=NumericPrecisionPolicy())

    def test_bridge_preserves_shape_scale_and_reservation(self):
        tensor = convert_nvfp4_to_int8(
            NumericFormatDescriptor("nvfp4_e2m1", 3), bytes([0xA2, 2]),
            bytes([56, 64, 72]), 1, geometry=TensorGeometry((3, 1), 1),
        )
        converter = Qwen4MLXCorrectnessConverter()
        # 6 input bytes + 12 bridge + 24 worker intermediate + 24 output buffers.
        with patch.object(converter, "convert", return_value="evidence") as consume:
            self.assertEqual(converter.convert_scaled_int8(
                tensor, target_dtype="F32", reserved_bytes=66), "evidence")
            args, kwargs = consume.call_args
            self.assertEqual(struct.unpack("<3f", args[0][0]), (1, -2, 4))
            self.assertEqual(kwargs, dict(source_dtype="F32", target_dtype="F32",
                                         output_shape=(3, 1), reserved_bytes=48))
            consume.reset_mock()
            with self.assertRaises(MemoryError):
                converter.convert_scaled_int8(tensor, target_dtype="F32", reserved_bytes=65)
            consume.assert_not_called()

    def test_invalid_bridge_requests_do_not_reach_backend(self):
        tensor = convert_nvfp4_to_int8(
            NumericFormatDescriptor("nvfp4_e2m1", 1), bytes([7]), bytes([56]), 1e40,
        )
        converter = Qwen4MLXCorrectnessConverter()
        with patch.object(converter, "convert") as consume:
            for dtype, reservation in (("INT8", 1000), ("F32", True), ("F32", -1),
                                       ("F32", 1000)):
                with self.assertRaises(ValueError):
                    converter.convert_scaled_int8(tensor, target_dtype=dtype,
                                                  reserved_bytes=reservation)
            consume.assert_not_called()


if __name__ == "__main__":
    unittest.main()
