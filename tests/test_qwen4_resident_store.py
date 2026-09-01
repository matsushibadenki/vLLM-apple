import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tests import test_qwen4_adapter_loader as loader_tests
from vllm_apple.qwen4_component_loader import Qwen4MemoryAdmission
from vllm_apple.qwen4_resident_store import Qwen4ResidentStore, ResidentBackendAllocation
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

    def release(self, resource):
        if self.fail_release:
            raise RuntimeError("release failed")
        self.resources.remove(resource)


class Qwen4ResidentStoreTests(unittest.TestCase):
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
