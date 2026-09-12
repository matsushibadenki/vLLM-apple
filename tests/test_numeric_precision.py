import unittest
import json
from dataclasses import replace

from vllm_apple.numeric_precision import NumericPrecisionPolicy, PrecisionExecutionContract


class NumericPrecisionTests(unittest.TestCase):
    def test_policy_roundtrip_and_canonical_id(self):
        policy = NumericPrecisionPolicy(absolute_tolerance=1, relative_tolerance=-0.0)
        self.assertEqual(policy.policy_id, NumericPrecisionPolicy(absolute_tolerance=1.0).policy_id)
        self.assertEqual(policy, NumericPrecisionPolicy.from_dict(json.loads(json.dumps(policy.to_dict()))))
        self.assertNotEqual(policy.policy_id, replace(policy, allow_underflow=True).policy_id)
        for change in ({"schema_version": True}, {"schema_version": 2}, {"extra": 1},
                       {"absolute_tolerance": "1"}):
            with self.assertRaises(ValueError):
                NumericPrecisionPolicy.from_dict(policy.to_dict() | change)
        with self.assertRaises(ValueError):
            NumericPrecisionPolicy.from_dict({})

    def test_contract_binds_tensor_dtype_and_policy(self):
        contract = PrecisionExecutionContract("a" * 64, "F16", NumericPrecisionPolicy())
        self.assertEqual(contract, PrecisionExecutionContract.from_dict(
            json.loads(json.dumps(contract.to_dict()))))
        for change in ({"tensor_digest": "b" * 64}, {"target_dtype": "F32"},
                       {"policy": NumericPrecisionPolicy(absolute_tolerance=0.01)}):
            self.assertNotEqual(contract.contract_id, replace(contract, **change).contract_id)
        for change in ({"bridge": "unknown"}, {"tensor_digest": "bad"},
                       {"target_dtype": "INT8"}, {"policy": {}}, {"schema_version": True}):
            with self.assertRaises(ValueError):
                PrecisionExecutionContract.from_dict(contract.to_dict() | change)

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
