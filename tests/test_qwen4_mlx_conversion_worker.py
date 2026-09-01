import unittest

from vllm_apple.qwen4_mlx_conversion_worker import Qwen4MLXCorrectnessConverter


class Qwen4MLXConversionWorkerTests(unittest.TestCase):
    def test_rejects_insufficient_reservation_before_importing_mlx(self) -> None:
        converter = Qwen4MLXCorrectnessConverter()
        with self.assertRaisesRegex(MemoryError, "reservation"):
            converter.convert(
                [b"\0\0"],
                source_dtype="BF16",
                target_dtype="BF16",
                output_shape=(1,),
                reserved_bytes=9,
            )

    def test_rejects_shape_mismatch_before_importing_mlx(self) -> None:
        converter = Qwen4MLXCorrectnessConverter()
        with self.assertRaisesRegex(ValueError, "shape"):
            converter.convert(
                [b"\0\0"],
                source_dtype="BF16",
                target_dtype="BF16",
                output_shape=(2,),
                reserved_bytes=64,
            )


if __name__ == "__main__":
    unittest.main()
