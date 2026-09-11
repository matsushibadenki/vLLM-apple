"""Opt-in real MLX test: VLLM_APPLE_TEST_MLX_NUMERIC=1 python -m unittest ..."""
import hashlib
import os
import struct
import unittest

from vllm_apple.numeric_formats import NumericFormatDescriptor, TensorGeometry, convert_nvfp4_to_int8
from vllm_apple.qwen4_mlx_conversion_worker import Qwen4MLXCorrectnessConverter
from vllm_apple.numeric_precision import NumericPrecisionPolicy


@unittest.skipUnless(os.environ.get("VLLM_APPLE_TEST_MLX_NUMERIC") == "1", "opt-in MLX device test")
class NumericMLXDeviceTests(unittest.TestCase):
    def test_rounding_and_small_values(self):
        converter = Qwen4MLXCorrectnessConverter()
        cases = {
            "F16": (1 + 2 ** -11, 1 + 3 * 2 ** -11, 2 ** -14,
                    2 ** -24, 2 ** -25, 65504.0),
            "BF16": (1 + 2 ** -8, 1 + 3 * 2 ** -8, 2 ** -126, 2 ** -16),
            "F32": (1 + 2 ** -24, 1 + 3 * 2 ** -24, 2 ** -126),
        }
        for dtype, values in cases.items():
            for value in values:
                with self.subTest(dtype=dtype, value=value):
                    tensor = convert_nvfp4_to_int8(NumericFormatDescriptor("nvfp4_e2m1", 2),
                                                  bytes([0xA2]), bytes([56]), value)
                    expected = bytearray()
                    for signed in (value, -value):
                        f32 = struct.pack("<f", signed)
                        rounded = struct.unpack("<f", f32)[0]
                        if dtype == "BF16":
                            bits = int.from_bytes(f32, "little")
                            # Round the F32 bridge to nearest, ties to even.
                            bits = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
                            expected.extend(struct.pack("<H", bits))
                        else:
                            expected.extend(struct.pack("<e" if dtype == "F16" else "<f", rounded))
                    evidence = converter.convert_scaled_int8(tensor, target_dtype=dtype,
                                                             reserved_bytes=4096)
                    self.assertEqual(evidence.output_digest, hashlib.sha256(expected).hexdigest())

    def test_overflow_and_nonfinite_source_are_rejected(self):
        converter = Qwen4MLXCorrectnessConverter()
        for dtype, scale in (("F16", 65520.0), ("BF16", 3.4028234663852886e38),
                             ("F32", 1e40)):
            tensor = convert_nvfp4_to_int8(NumericFormatDescriptor("nvfp4_e2m1", 1),
                                          bytes([2]), bytes([56]), scale)
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                converter.convert_scaled_int8(tensor, target_dtype=dtype, reserved_bytes=4096)
        for value in (float("inf"), float("-inf"), float("nan")):
            with self.assertRaises(ValueError):
                converter.convert((struct.pack("<f", value),), source_dtype="F32",
                                  target_dtype="F32", output_shape=(1,), reserved_bytes=4096)

    def test_real_backend_all_codes_and_scale_codes(self):
        # Independent scalar oracle: E2M1 magnitudes, E4M3FN positive scale.
        values = (0, .5, 1, 1.5, 2, 3, 4, 6)
        codes = [i % 16 for i in range(34)]
        packed = bytes(codes[i] | codes[i + 1] << 4 for i in range(0, 34, 2))
        descriptor = NumericFormatDescriptor("nvfp4_e2m1", 34)
        converter = Qwen4MLXCorrectnessConverter()
        for axis, shape in ((1, (2, 17)), (0, (17, 2))):
            geometry = TensorGeometry(shape, axis)
            for scale_code in range(127):
                scale_codes = [(scale_code + i) % 127 for i in range(4)]
                expected = []
                for i, code in enumerate(codes):
                    scale_index = ((i // 17) * 2 + (i % 17) // 16 if axis == 1
                                   else (i // 2 // 16) * 2 + i % 2)
                    s = scale_codes[scale_index]
                    exponent, mantissa = divmod(s, 8)
                    scale = mantissa / 512 if exponent == 0 else (1 + mantissa / 8) * 2 ** (exponent - 7)
                    value = values[code & 7] * (-1 if code & 8 else 1) * scale
                    # Only encoded +/-zero loses its sign in INT8 storage.
                    # A negative nonzero integer times zero scale still yields -0.
                    expected.append(0.0 if (code & 7) == 0 else value)
                tensor = convert_nvfp4_to_int8(descriptor, packed, bytes(scale_codes), 1,
                                              geometry=geometry)
                for dtype in ("F32", "F16", "BF16"):
                    with self.subTest(axis=axis, scale=scale_code, dtype=dtype):
                        raw = b"".join(struct.pack("<f", v)[2:] if dtype == "BF16" else
                                       struct.pack("<f" if dtype == "F32" else "<e", v)
                                       for v in expected)
                        evidence = converter.convert_scaled_int8(tensor, target_dtype=dtype,
                                                                 reserved_bytes=4096,
                                                                 precision_policy=NumericPrecisionPolicy())
                        self.assertEqual(evidence.backend, "mlx")
                        self.assertEqual(evidence.output_shape, shape)
                        self.assertEqual(evidence.output_bytes, len(raw))
                        self.assertEqual(evidence.output_digest, hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    unittest.main()
