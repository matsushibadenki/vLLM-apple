import unittest

from vllm_apple.qwen4_runtime_protocol import Qwen4RuntimeCommandService


class FakeStore:
    def __init__(self) -> None:
        self.loads = 0
        self.handles = set()

    def load(self, tensor_name, *, target_dtype, scratch_bytes=0, axis0_slice=None):
        self.loads += 1
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
