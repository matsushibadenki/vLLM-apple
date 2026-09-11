import unittest

from vllm_apple.numeric_precision import NumericPrecisionPolicy


class NumericPrecisionTests(unittest.TestCase):
    def test_exact_and_tolerance(self):
        exact = NumericPrecisionPolicy()
        self.assertEqual(exact.checked_bytes(1, "F16"), b"\0\x3c")
        with self.assertRaises(ValueError):
            exact.checked_bytes(1 + 2 ** -24, "F32")
        NumericPrecisionPolicy(absolute_tolerance=2 ** -24).checked_bytes(1 + 2 ** -24, "F32")
        NumericPrecisionPolicy(relative_tolerance=0.001).checked_bytes(-1.0001, "F16")

    def test_underflow_requires_both_permission_and_tolerance(self):
        for policy in (NumericPrecisionPolicy(absolute_tolerance=1),
                       NumericPrecisionPolicy(allow_underflow=True)):
            with self.assertRaises(ValueError):
                policy.checked_bytes(2 ** -25, "F16")
        self.assertEqual(NumericPrecisionPolicy(absolute_tolerance=1, allow_underflow=True)
                         .checked_bytes(-2 ** -25, "F16"), b"\0\x80")

    def test_invalid_and_overflow(self):
        for tolerance in (-1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                NumericPrecisionPolicy(absolute_tolerance=tolerance)
        with self.assertRaises(ValueError):
            NumericPrecisionPolicy(allow_underflow=1)
        for value, dtype in ((65520, "F16"), (3.4028234663852886e38, "BF16"),
                             (1e40, "F32"), (float("nan"), "F32"), (1, "INT8")):
            with self.assertRaises(ValueError):
                NumericPrecisionPolicy(absolute_tolerance=1e300).checked_bytes(value, dtype)
