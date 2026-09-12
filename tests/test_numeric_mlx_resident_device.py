"""Opt-in: VLLM_APPLE_TEST_MLX_NUMERIC=1 python -m unittest this.module."""
import os
import hashlib
import socket
import tempfile
import threading
import unittest
from pathlib import Path

from tests import test_qwen4_adapter_loader as loader_tests
from vllm_apple.numeric_formats import (
    NumericFormatDescriptor,
    TensorGeometry,
    convert_nvfp4_to_int8,
)
from vllm_apple.numeric_artifact import NumericArtifactReader, write_nvfp4_numeric_artifact
from vllm_apple.numeric_precision import NumericPrecisionPolicy, PrecisionExecutionContract
from vllm_apple.numeric_streaming import NumericStreamingPlan
from vllm_apple.qwen4_component_loader import Qwen4MemoryAdmission
from vllm_apple.qwen4_mlx_resident_backend import Qwen4MLXNumericResidentBackend
from vllm_apple.qwen4_resident_store import Qwen4ResidentStore
from vllm_apple.qwen4_shard_stager import stage_qwen4_shards
from vllm_apple.qwen4_tensor_reader import Qwen4TensorReader
from vllm_apple.qwen4_runtime_protocol import (
    Qwen4RuntimeCommandService,
    build_qwen4_numeric_streaming_runtime_request,
)
from vllm_apple.qwen4_runtime_transport import (
    Qwen4RuntimeUnixServer,
    receive_qwen4_runtime_frame,
    send_qwen4_runtime_frame,
)


@unittest.skipUnless(os.environ.get("VLLM_APPLE_TEST_MLX_NUMERIC") == "1", "opt-in MLX device test")
class NumericMLXResidentDeviceTests(unittest.TestCase):
    def test_private_artifact_over_socket_to_mlx_residency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            source = loader_tests.Qwen4AdapterLoaderTests().source(root)
            stage = root / "stage"
            stage_qwen4_shards(source, stage, maximum_output_bytes=65536)
            artifacts = root / "numeric"
            artifacts.mkdir(mode=0o700)
            created = write_nvfp4_numeric_artifact(
                artifacts / "weight.json", NumericFormatDescriptor("nvfp4_e2m1", 3),
                bytes([0xA2, 2]), bytes([56, 64, 72]), 1,
                geometry=TensorGeometry((3, 1), 1), target_dtype="F16",
                precision_policy=NumericPrecisionPolicy())
            store = Qwen4ResidentStore(
                Qwen4TensorReader(stage, maximum_artifact_bytes=65536),
                Qwen4MemoryAdmission(4096), Qwen4MLXNumericResidentBackend())
            service = Qwen4RuntimeCommandService(
                "a" * 32, store, NumericArtifactReader(artifacts))
            server_socket, client_socket = socket.socketpair()
            server = Qwen4RuntimeUnixServer("/tmp/not-bound.sock", service)
            thread = threading.Thread(target=server.serve_connection, args=(server_socket,))
            thread.start()
            try:
                request = build_qwen4_numeric_streaming_runtime_request(
                    session_id="a" * 32, sequence=1, request_id="1" * 32,
                    artifact_name="weight.json", artifact_digest=created.artifact_digest,
                    target_dtype="F16", tile_bytes=1, buffer_count=2)
                send_qwen4_runtime_frame(client_socket, request)
                loaded = receive_qwen4_runtime_frame(client_socket)
                self.assertTrue(loaded["passed"])
                self.assertEqual(loaded["result"]["artifact_state"], "consumed")
                self.assertFalse((artifacts / "weight.json").exists())
                self.assertEqual(list((artifacts / "quarantine").iterdir()), [])
                self.assertEqual(store.snapshot()["resident_tensors"], 1)
                send_qwen4_runtime_frame(client_socket, {
                    "abi_version": 1, "session_id": "a" * 32, "sequence": 2,
                    "request_id": "2" * 32, "operation": "unload",
                    "handle": loaded["result"]["handle"],
                })
                self.assertTrue(receive_qwen4_runtime_frame(client_socket)["passed"])
                self.assertEqual(store.snapshot()["memory"]["reserved_bytes"], 0)
                send_qwen4_runtime_frame(client_socket, {
                    "abi_version": 1, "session_id": "a" * 32, "sequence": 3,
                    "request_id": "3" * 32, "operation": "shutdown",
                })
                self.assertTrue(receive_qwen4_runtime_frame(client_socket)["passed"])
            finally:
                client_socket.close()
                thread.join(timeout=2)
                server_socket.close()
            self.assertFalse(thread.is_alive())

    def test_nvfp4_to_mlx_residency_and_release(self):
        import mlx.core as mx
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = loader_tests.Qwen4AdapterLoaderTests().source(root)
            stage = root / "stage"
            stage_qwen4_shards(source, stage, maximum_output_bytes=65536)
            store = Qwen4ResidentStore(
                Qwen4TensorReader(stage, maximum_artifact_bytes=65536),
                Qwen4MemoryAdmission(4096),
                Qwen4MLXNumericResidentBackend(),
            )
            tensor = convert_nvfp4_to_int8(
                NumericFormatDescriptor("nvfp4_e2m1", 6), bytes([0xA2, 0x46, 0xEC]),
                bytes([56, 64]), 1,
                geometry=TensorGeometry((2, 3), 1),
            )
            for dtype, destination_bytes in (("F16", 12), ("BF16", 12), ("F32", 24)):
                with self.subTest(dtype=dtype):
                    contract = PrecisionExecutionContract(
                        tensor.target_digest, dtype, NumericPrecisionPolicy())
                    handle = store.load_scaled_int8(
                        tensor, target_dtype=dtype, execution_contract=contract)
                    record = store._records[handle]
                    resource = record.allocation.resource
                    try:
                        inspected = resource.array.astype(mx.float32)
                        mx.eval(inspected)
                        self.assertEqual(np.asarray(inspected).tolist(),
                                         [[1, -1, 4], [4, -4, -8]])
                        self.assertEqual(store.snapshot()["memory"]["reserved_bytes"],
                                         destination_bytes)
                    finally:
                        store.unload(handle)
                    self.assertTrue(resource.released)
                    self.assertEqual(store.snapshot()["memory"]["reserved_bytes"], 0)

            streaming_contract = PrecisionExecutionContract(
                tensor.target_digest, "F16", NumericPrecisionPolicy())
            streaming_plan = NumericStreamingPlan(
                hashlib.sha256(tensor.payload).hexdigest(), len(tensor.payload), 2, 2)
            streaming_handle = store.load_scaled_int8_streaming(
                tensor,
                stream_plan=streaming_plan,
                target_dtype="F16",
                execution_contract=streaming_contract,
            )
            self.assertEqual(store.snapshot()["memory"]["reserved_bytes"], 12)
            store.unload(streaming_handle)

            rounded = convert_nvfp4_to_int8(
                NumericFormatDescriptor("nvfp4_e2m1", 1), bytes([2]), bytes([56]), 1.0001)
            strict = PrecisionExecutionContract(
                rounded.target_digest, "F16", NumericPrecisionPolicy())
            with self.assertRaisesRegex(ValueError, "tolerance"):
                store.load_scaled_int8(rounded, target_dtype="F16", execution_contract=strict)
            self.assertEqual(store.snapshot()["memory"]["reserved_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
