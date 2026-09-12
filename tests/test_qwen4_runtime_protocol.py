import unittest
from types import SimpleNamespace

from vllm_apple.numeric_formats import NumericFormatDescriptor, convert_nvfp4_to_int8
from vllm_apple.numeric_precision import NumericPrecisionPolicy, PrecisionExecutionContract
from vllm_apple.qwen4_runtime_protocol import (
    Qwen4RuntimeCommandService,
    build_qwen4_numeric_runtime_request,
    parse_qwen4_runtime_request,
)


class FakeStore:
    def __init__(self) -> None:
        self.loads = 0
        self.handles = set()

    def load(self, tensor_name, *, target_dtype, scratch_bytes=0, axis0_slice=None):
        self.loads += 1
        handle = f"{self.loads:032x}"
        self.handles.add(handle)
        return handle

    def load_scaled_int8(self, tensor, *, target_dtype, execution_contract,
                         component="numeric_compatibility", scratch_bytes=0):
        self.loads += 1
        self.last_numeric = (tensor, target_dtype, execution_contract, component, scratch_bytes)
        handle = f"{self.loads:032x}"
        self.handles.add(handle)
        return handle

    def unload(self, handle):
        self.handles.remove(handle)

    def snapshot(self):
        return {
            "schema_version": 1,
            "resident_tensors": len(self.handles),
            "quarantined_tensors": 0,
            "resident_components": {},
            "memory": {},
            "stores_tensor_names": False,
        }

    def retry_quarantined_releases(self):
        return 0

    def shutdown(self):
        self.handles.clear()
        return self.snapshot()


class Qwen4RuntimeProtocolTests(unittest.TestCase):
    @staticmethod
    def numeric_fixture():
        tensor = convert_nvfp4_to_int8(
            NumericFormatDescriptor("nvfp4_e2m1", 1), bytes([2]), bytes([56]), 1)
        contract = PrecisionExecutionContract(
            tensor.target_digest, "F16", NumericPrecisionPolicy())
        loaded = SimpleNamespace(
            tensor=tensor, execution_contract=contract, artifact_digest="b" * 64)
        reader = SimpleNamespace(
            claim=lambda name, digest: loaded,
            consume=lambda value: None,
        )
        return tensor, contract, reader

    def test_numeric_load_is_ordered_replay_safe_and_dtype_bound(self) -> None:
        tensor, contract, reader = self.numeric_fixture()
        store = FakeStore()
        service = Qwen4RuntimeCommandService("a" * 32, store, reader)
        request = build_qwen4_numeric_runtime_request(
            session_id="a" * 32, sequence=1, request_id="1" * 32,
            artifact_name="weight.json", artifact_digest="b" * 64,
            target_dtype="F16", scratch_bytes=7)
        first = service.handle(request)
        self.assertEqual(first["result"]["artifact_state"], "consumed")
        self.assertEqual(service.handle(dict(request)), first)
        self.assertEqual(store.loads, 1)
        self.assertEqual(store.last_numeric, (
            tensor, "F16", contract, "numeric_compatibility", 7))
        mismatch = build_qwen4_numeric_runtime_request(
            session_id="a" * 32, sequence=2, request_id="2" * 32,
            artifact_name="weight.json", artifact_digest="b" * 64,
            target_dtype="F32")
        self.assertFalse(service.handle(mismatch)["passed"])
        self.assertEqual(store.loads, 1)

    def test_numeric_load_schema_and_disabled_reader_fail_closed(self) -> None:
        request = build_qwen4_numeric_runtime_request(
            session_id="a" * 32, sequence=1, request_id="1" * 32,
            artifact_name="weight.json", artifact_digest="b" * 64,
            target_dtype="F16")
        response = Qwen4RuntimeCommandService("a" * 32, FakeStore()).handle(request)
        self.assertEqual(response["error_code"], "operation_failed")
        for change in ({"artifact_name": "../weight.json"}, {"artifact_name": ".hidden.json"},
                       {"artifact_digest": "bad"}, {"target_dtype": "INT8"},
                       {"scratch_bytes": True}, {"extra": 1}):
            with self.assertRaises(ValueError):
                parse_qwen4_runtime_request(request | change)

    def test_numeric_backend_failure_leaves_claim_unconsumed(self) -> None:
        _, _, fixture_reader = self.numeric_fixture()
        consumed = []
        reader = SimpleNamespace(
            claim=fixture_reader.claim,
            consume=lambda value: consumed.append(value),
        )
        store = FakeStore()

        def fail(*args, **kwargs):
            raise ValueError("backend failed")

        store.load_scaled_int8 = fail
        request = build_qwen4_numeric_runtime_request(
            session_id="a" * 32, sequence=1, request_id="1" * 32,
            artifact_name="weight.json", artifact_digest="b" * 64,
            target_dtype="F16")
        response = Qwen4RuntimeCommandService("a" * 32, store, reader).handle(request)
        self.assertFalse(response["passed"])
        self.assertEqual(consumed, [])

    def request(self, sequence, operation, **values):
        return {
            "abi_version": 1,
            "session_id": "a" * 32,
            "sequence": sequence,
            "request_id": f"{sequence:032x}",
            "operation": operation,
            **values,
        }

    def test_replays_identical_request_without_loading_twice(self) -> None:
        store = FakeStore()
        service = Qwen4RuntimeCommandService("a" * 32, store)
        request = self.request(
            1,
            "load",
            tensor_name="model.tensor",
            target_dtype="BF16",
            scratch_bytes=0,
            axis0_slice=None,
        )
        first = service.handle(request)
        second = service.handle(dict(request))
        self.assertEqual(first, second)
        self.assertEqual(store.loads, 1)

    def test_rejects_sequence_gap_and_content_reuse(self) -> None:
        service = Qwen4RuntimeCommandService("a" * 32, FakeStore())
        with self.assertRaisesRegex(ValueError, "contiguous"):
            service.handle(self.request(2, "status"))
        service.handle(self.request(1, "status"))
        with self.assertRaisesRegex(ValueError, "different content"):
            service.handle(self.request(1, "retry_quarantine"))

    def test_status_unload_and_shutdown_are_ordered(self) -> None:
        store = FakeStore()
        service = Qwen4RuntimeCommandService("a" * 32, store)
        loaded = service.handle(
            self.request(
                1,
                "load",
                tensor_name="model.tensor",
                target_dtype="BF16",
                scratch_bytes=0,
                axis0_slice=None,
            )
        )
        handle = loaded["result"]["handle"]
        self.assertEqual(service.handle(self.request(2, "status"))["result"]["resident_tensors"], 1)
        service.handle(self.request(3, "unload", handle=handle))
        self.assertTrue(service.handle(self.request(4, "shutdown"))["result"]["shutdown"])
        with self.assertRaisesRegex(ValueError, "closed"):
            service.handle(self.request(5, "status"))


if __name__ == "__main__":
    unittest.main()
