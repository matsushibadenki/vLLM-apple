import json
import tempfile
import unittest
from pathlib import Path

from tests import test_qwen4_adapter_loader as loader_tests
from tests import test_qwen4_load_plan as load_plan_tests
from vllm_apple.qwen4_component_loader import Qwen4ComponentLoader, Qwen4MemoryAdmission
from vllm_apple.qwen4_shard_stager import stage_qwen4_shards
from vllm_apple.qwen4_tensor_reader import Qwen4TensorReader


class Qwen4ComponentLoaderTests(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Qwen4ComponentLoader, str]:
        source = loader_tests.Qwen4AdapterLoaderTests().source(root)
        output = root / "output"
        stage_qwen4_shards(source, output, maximum_output_bytes=65536)
        reader = Qwen4TensorReader(
            output, maximum_artifact_bytes=65536, maximum_chunk_bytes=1
        )
        index = json.loads((output / "model.safetensors.index.json").read_text())
        tensor_name = next(iter(index["weight_map"]))
        admission = Qwen4MemoryAdmission(16)
        return Qwen4ComponentLoader(reader, admission), tensor_name

    def test_reserves_destination_stream_and_scratch_then_releases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            loader, tensor_name = self.fixture(Path(directory))
            with loader.open_tensor(tensor_name, target_dtype="F32", scratch_bytes=2) as lease:
                self.assertEqual(lease.reservation.destination_bytes, 4)
                self.assertEqual(lease.reservation.source_stream_bytes, 1)
                self.assertEqual(lease.reservation.reserved_bytes, 7)
                self.assertEqual(list(lease.chunks), [b"\0", b"\0"])
                self.assertEqual(loader.admission.snapshot()["reserved_bytes"], 7)
            self.assertEqual(loader.admission.snapshot()["reserved_bytes"], 0)

    def test_releases_reservation_when_consumer_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            loader, tensor_name = self.fixture(Path(directory))
            with self.assertRaisesRegex(RuntimeError, "consumer"):
                with loader.open_tensor(tensor_name, target_dtype="BF16"):
                    raise RuntimeError("consumer failed")
            self.assertEqual(loader.admission.snapshot()["active_reservations"], 0)

    def test_rejects_overlapping_loads_above_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            loader, tensor_name = self.fixture(Path(directory))
            with loader.open_tensor(tensor_name, target_dtype="F32", scratch_bytes=8):
                with self.assertRaisesRegex(MemoryError, "admission"):
                    with loader.open_tensor(tensor_name, target_dtype="F32", scratch_bytes=8):
                        self.fail("overcommitted reservation was accepted")
            self.assertEqual(loader.admission.snapshot()["remaining_bytes"], 16)

    def test_reserves_only_the_requested_expert_slice_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = load_plan_tests.packed_expert_source(root)
            output = root / "output"
            stage_qwen4_shards(source, output, maximum_output_bytes=65536)
            reader = Qwen4TensorReader(output, maximum_artifact_bytes=65536)
            tensor_name = next(
                name for name in reader._catalog["tensors"] if ".experts." in name
            )
            loader = Qwen4ComponentLoader(reader, Qwen4MemoryAdmission(16))
            with loader.open_tensor_axis0_slice(
                tensor_name, start=1, count=1, target_dtype="F32"
            ) as lease:
                self.assertEqual(lease.reservation.destination_bytes, 4)
                self.assertEqual(lease.reservation.source_stream_bytes, 2)

    def test_atomically_reduces_conversion_reservation_to_resident_destination(self) -> None:
        admission = Qwen4MemoryAdmission(32)
        reservation = admission.reserve(
            "tensor",
            {"component": "token_embedding", "shape": [2]},
            target_dtype="F32",
            source_stream_bytes=4,
            scratch_bytes=6,
        )
        retained = admission.retain_destination(reservation)
        self.assertEqual(retained.reserved_bytes, 8)
        self.assertEqual(retained.source_stream_bytes, 0)
        self.assertEqual(retained.scratch_bytes, 0)
        self.assertEqual(admission.snapshot()["reserved_bytes"], 8)
        with self.assertRaisesRegex(ValueError, "changed"):
            admission.retain_destination(reservation)
        admission.release(retained)


if __name__ == "__main__":
    unittest.main()
