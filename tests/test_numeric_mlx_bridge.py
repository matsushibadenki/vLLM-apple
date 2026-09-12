import hashlib
import struct
import unittest
from dataclasses import replace
from unittest.mock import patch

from vllm_apple.numeric_formats import (
    NumericFormatDescriptor, TensorGeometry, convert_nvfp4_to_int8,
)
from vllm_apple.qwen4_mlx_conversion_worker import Qwen4MLXCorrectnessConverter
from vllm_apple.numeric_precision import NumericPrecisionPolicy, PrecisionExecutionContract


class NumericMLXBridgeTests(unittest.TestCase):
    def test_contract_mismatch_never_reaches_backend(self):
        tensor = convert_nvfp4_to_int8(NumericFormatDescriptor("nvfp4_e2m1", 1),
                                      bytes([2]), bytes([56]), 1)
        converter = Qwen4MLXCorrectnessConverter()
        for digest, dtype, policy in (("0" * 64, "F16", NumericPrecisionPolicy()),
                                      (tensor.target_digest, "F32", NumericPrecisionPolicy()),
                                      (tensor.target_digest, "F16", NumericPrecisionPolicy(1))):
            with patch.object(converter, "convert") as consume:
                with self.assertRaises(ValueError):
                    converter.convert_scaled_int8(
                        tensor, target_dtype="F16", reserved_bytes=4096,
                        precision_policy=NumericPrecisionPolicy(),
                        execution_contract=PrecisionExecutionContract(digest, dtype, policy))
                consume.assert_not_called()

    def test_checked_evidence_exposes_bounded_diagnostics(self):
        from vllm_apple.qwen4_conversion_worker import ConvertedTensorEvidence
        tensor = convert_nvfp4_to_int8(NumericFormatDescriptor("nvfp4_e2m1", 1),
                                      bytes([2]), bytes([56]), 1)
        policy = NumericPrecisionPolicy()
        contract = PrecisionExecutionContract(tensor.target_digest, "F16", policy)
        output_digest = hashlib.sha256(b"\0\x3c").hexdigest()
        backend = ConvertedTensorEvidence("test", "1", (1,), 2, output_digest)
        converter = Qwen4MLXCorrectnessConverter()
        with patch.object(converter, "convert", return_value=backend):
            evidence = converter.convert_scaled_int8(
                tensor, target_dtype="F16", reserved_bytes=4096,
                execution_contract=contract)
        self.assertEqual(evidence.precision_diagnostics(), {
            "schema_version": 1, "checked": True, "contract_id": contract.contract_id,
            "policy_id": policy.policy_id, "stores_tensor_values": False,
        })
        self.assertNotIn(tensor.target_digest, evidence.precision_diagnostics().values())
        for change in ({"precision_policy_id": "bad", "precision_checked": True},
                       {"precision_contract_id": "0" * 64},
                       {"precision_policy_id": "0" * 64},
                       {"precision_checked": 1}):
            with self.assertRaises(ValueError):
                replace(backend, **change)

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
