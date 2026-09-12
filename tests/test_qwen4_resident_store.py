import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tests import test_qwen4_adapter_loader as loader_tests
from vllm_apple.qwen4_component_loader import Qwen4MemoryAdmission
from vllm_apple.qwen4_resident_store import Qwen4ResidentStore, ResidentBackendAllocation
from vllm_apple.numeric_formats import NumericFormatDescriptor, TensorGeometry, convert_nvfp4_to_int8
from vllm_apple.numeric_precision import NumericPrecisionPolicy, PrecisionExecutionContract
from vllm_apple.numeric_streaming import NumericStreamingCancelled, NumericStreamingPlan
from vllm_apple.qwen4_shard_stager import stage_qwen4_shards
from vllm_apple.qwen4_tensor_reader import Qwen4TensorReader


class FakeResidentBackend:
    def __init__(self, *, fail_release: bool = False, invalid_evidence: bool = False) -> None:
        self.fail_release = fail_release
        self.invalid_evidence = invalid_evidence
        self.resources = []

    def load(self, chunks, *, source_dtype, target_dtype, output_shape, reserved_bytes):
        raw = b"".join(chunks)
        resource = object()
        self.resources.append(resource)
        return ResidentBackendAllocation(
            resource=resource,
            backend="test",
            backend_version="1",
            output_shape=output_shape,
            output_bytes=0 if self.invalid_evidence else len(raw),
            output_digest=hashlib.sha256(raw).hexdigest(),
        )

    def load_scaled_int8(self, tensor, *, target_dtype, execution_contract, reserved_bytes):
        resource = object()
        self.resources.append(resource)
        shape = tensor.geometry.shape if tensor.geometry is not None else (len(tensor.payload),)
        output_bytes = len(tensor.payload) * {"BF16": 2, "F16": 2, "F32": 4}[target_dtype]
        return ResidentBackendAllocation(
            resource=resource,
            backend="test",
            backend_version="1",
            output_shape=shape,
            output_bytes=0 if self.invalid_evidence else output_bytes,
            output_digest=hashlib.sha256(tensor.payload).hexdigest(),
            precision_contract_id=execution_contract.contract_id,
            precision_policy_id=execution_contract.policy.policy_id,
            precision_checked=True,
        )

    def load_scaled_int8_streaming(
        self, tensor, stream, *, target_dtype, execution_contract, reserved_bytes
    ):
        raw = bytearray()
        while True:
            lease = stream.acquire_next()
            if lease is None:
                break
            try:
                raw.extend(lease.read())
            finally:
                lease.release()
        if bytes(raw) != tensor.payload:
            raise ValueError("streamed payload mismatch")
        return replace(self.load_scaled_int8(
            tensor,
            target_dtype=target_dtype,
            execution_contract=execution_contract,
            reserved_bytes=reserved_bytes,
        ), numeric_streaming_plan_id=stream.plan.plan_id)

    def release(self, resource):
        if self.fail_release:
            raise RuntimeError("release failed")
        self.resources.remove(resource)


class Qwen4ResidentStoreTests(unittest.TestCase):
    @staticmethod
    def numeric_tensor():
        return convert_nvfp4_to_int8(
            NumericFormatDescriptor("nvfp4_e2m1", 3), bytes([0x22, 2]),
            bytes([56, 64, 72]), 1, geometry=TensorGeometry((3, 1), 1),
        )

    def test_scaled_int8_residency_retains_only_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeResidentBackend()
            store, _ = self.fixture(Path(directory), backend)
            tensor = self.numeric_tensor()
            contract = PrecisionExecutionContract(
                tensor.target_digest, "F16", NumericPrecisionPolicy())
            handle = store.load_scaled_int8(
                tensor, target_dtype="F16", execution_contract=contract)
            snapshot = store.snapshot()
            self.assertEqual(snapshot["memory"]["reserved_bytes"], 6)
            self.assertEqual(snapshot["resident_components"], {"numeric_compatibility": 1})
            store.unload(handle)
            self.assertEqual(store.snapshot()["memory"]["reserved_bytes"], 0)
            self.assertEqual(backend.resources, [])

    def test_scaled_int8_rejects_mismatch_and_invalid_backend_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeResidentBackend()
            store, _ = self.fixture(Path(directory), backend)
            tensor = self.numeric_tensor()
            contract = PrecisionExecutionContract(
                tensor.target_digest, "F16", NumericPrecisionPolicy())
            for bad in (
                PrecisionExecutionContract("0" * 64, "F16", contract.policy),
                PrecisionExecutionContract(tensor.target_digest, "F32", contract.policy),
            ):
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    store.load_scaled_int8(tensor, target_dtype="F16", execution_contract=bad)
            self.assertEqual(store.snapshot()["memory"]["reserved_bytes"], 0)
            backend.invalid_evidence = True
            with self.assertRaisesRegex(ValueError, "evidence"):
                store.load_scaled_int8(tensor, target_dtype="F16", execution_contract=contract)
            self.assertEqual(store.snapshot()["memory"]["reserved_bytes"], 0)
            self.assertEqual(backend.resources, [])

    def test_scaled_int8_streaming_counts_buffers_and_retains_only_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeResidentBackend()
            store, _ = self.fixture(Path(directory), backend)
            tensor = self.numeric_tensor()
            contract = PrecisionExecutionContract(
                tensor.target_digest, "F16", NumericPrecisionPolicy())
            plan = NumericStreamingPlan(
                hashlib.sha256(tensor.payload).hexdigest(), len(tensor.payload), 1, 2)
            handle = store.load_scaled_int8_streaming(
                tensor,
                stream_plan=plan,
                target_dtype="F16",
                execution_contract=contract,
            )
            self.assertEqual(store.snapshot()["memory"]["reserved_bytes"], 6)
            store.unload(handle)
            self.assertEqual(store.snapshot()["memory"]["reserved_bytes"], 0)

    def test_scaled_int8_streaming_cancel_releases_reservation(self) -> None:
        class CancellingBackend(FakeResidentBackend):
            def load_scaled_int8_streaming(self, tensor, stream, **kwargs):
                cancellation.set()
                stream.poll_cancellation()

        with tempfile.TemporaryDirectory() as directory:
            cancellation = __import__("threading").Event()
            store, _ = self.fixture(Path(directory), CancellingBackend())
            tensor = self.numeric_tensor()
            contract = PrecisionExecutionContract(
                tensor.target_digest, "F16", NumericPrecisionPolicy())
            plan = NumericStreamingPlan(
                hashlib.sha256(tensor.payload).hexdigest(), len(tensor.payload), 1, 2)
            with self.assertRaises(NumericStreamingCancelled):
                store.load_scaled_int8_streaming(
                    tensor,
                    stream_plan=plan,
                    target_dtype="F16",
                    execution_contract=contract,
                    cancellation=cancellation,
                )
            self.assertEqual(store.snapshot()["memory"]["reserved_bytes"], 0)

    def fixture(self, root: Path, backend: FakeResidentBackend):
        source = loader_tests.Qwen4AdapterLoaderTests().source(root)
        output = root / "output"
        stage_qwen4_shards(source, output, maximum_output_bytes=65536)
        reader = Qwen4TensorReader(output, maximum_artifact_bytes=65536)
        index = json.loads((output / "model.safetensors.index.json").read_text())
        tensor_name = next(iter(index["weight_map"]))
        admission = Qwen4MemoryAdmission(32)
        return Qwen4ResidentStore(reader, admission, backend), tensor_name

    def test_retains_only_destination_until_explicit_unload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeResidentBackend()
            store, tensor_name = self.fixture(Path(directory), backend)
            handle = store.load(tensor_name, target_dtype="BF16")
            self.assertEqual(store.snapshot()["memory"]["reserved_bytes"], 2)
            self.assertEqual(store.snapshot()["resident_tensors"], 1)
            store.unload(handle)
            self.assertEqual(store.snapshot()["memory"]["reserved_bytes"], 0)
            self.assertEqual(backend.resources, [])

    def test_keeps_reservation_when_backend_release_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeResidentBackend(fail_release=True)
            store, tensor_name = self.fixture(Path(directory), backend)
            handle = store.load(tensor_name, target_dtype="BF16")
            with self.assertRaisesRegex(RuntimeError, "release"):
                store.unload(handle)
            self.assertEqual(store.snapshot()["resident_tensors"], 1)
            self.assertEqual(store.snapshot()["memory"]["reserved_bytes"], 2)

    def test_quarantines_failed_allocation_when_cleanup_also_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeResidentBackend(fail_release=True, invalid_evidence=True)
            store, tensor_name = self.fixture(Path(directory), backend)
            with self.assertRaisesRegex(RuntimeError, "quarantined"):
                store.load(tensor_name, target_dtype="BF16")
            self.assertEqual(store.snapshot()["quarantined_tensors"], 1)
            self.assertEqual(store.snapshot()["memory"]["reserved_bytes"], 4)
            backend.fail_release = False
            self.assertEqual(store.retry_quarantined_releases(), 1)
            self.assertEqual(store.snapshot()["memory"]["reserved_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
