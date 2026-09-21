import unittest
from dataclasses import dataclass

from vllm_apple.device_capability import (
    ComputeDevice,
    DeviceCapability,
    DeviceCapabilityRegistry,
)
from vllm_apple.device_pipeline import (
    ANEAuxiliaryWorkload,
    AsyncEncoderLLMPipeline,
    require_ane_auxiliary_route,
)
from vllm_apple.device_resources import UnifiedDeviceResourceLedger
from vllm_apple.execution import ExecutionBackend, WorkloadPhase
from vllm_apple.qwen3_vl_ane import Qwen3VLVisionANEAdapterSpec
from vllm_apple.qwen3_vl_coreml import Qwen3VLCoreMLConversionManifest
from vllm_apple.qwen3_vl_embedding import (
    Qwen3VLANEGPUPipeline,
    Qwen3VLCoreMLPipelineOutput,
    Qwen3VLVisionEmbeddingBundle,
    build_vllm_metal_qwen3_vl_encode_result,
    validate_qwen3_vl_vision_embeddings,
)


class FakeTensor:
    def __init__(self, shape):
        self.shape = shape


@dataclass
class FakeVLLMMetalResult:
    hidden_states: object
    deepstack_visual_embeds: object


class Qwen3VLEmbeddingTests(unittest.TestCase):
    def setUp(self):
        fingerprint = "a" * 64
        self.source = Qwen3VLVisionANEAdapterSpec(
            "b" * 40, fingerprint, f"vision_encoder@{fingerprint[:16]}",
            2, 16, 32, 4, 16, 2, 2, 24, (0, 1), 20, 100,
            "affine-int4", False,
        )
        self.conversion = Qwen3VLCoreMLConversionManifest(
            fingerprint, "b" * 40, "c" * 64, "pixel_values", "image_embeddings",
            (1, 3, 2, 16, 16), (1, 4, 24), "fp16", "coremltools-8.1",
        )
        self.registry = DeviceCapabilityRegistry("m4-test", "environment-test")
        self.capability = DeviceCapability(
            ExecutionBackend.COREML_DRAFT, ComputeDevice.ANE, "coremltools-8.1",
            "m4-test", "environment-test", (self.source.operator,),
            (WorkloadPhase.AUXILIARY,), ("fp16",), "available", "probe_passed",
            ("d" * 24,),
        )
        self.registry.record(self.capability)
        self.route = require_ane_auxiliary_route(
            self.registry,
            workload=ANEAuxiliaryWorkload.VISION_ENCODER,
            operator=self.source.operator,
            precision="fp16",
        )
        self.ledger = UnifiedDeviceResourceLedger(
            unified_memory_bytes=100,
            cpu_threads=2,
            gpu_command_queues=1,
            ane_tasks=1,
            bandwidth_slots=2,
        )

    def test_validates_main_and_every_deepstack_output(self):
        bundle = validate_qwen3_vl_vision_embeddings(
            self.source,
            self.route,
            grid_thw=(1, 4, 4),
            hidden_states=FakeTensor((4, 24)),
            deepstack_visual_embeds=(FakeTensor((4, 24)), FakeTensor((4, 24))),
        )
        self.assertEqual(bundle.token_count, 4)
        self.assertEqual(bundle.hidden_size, 24)
        self.assertEqual(bundle.capability_id, self.capability.capability_id)

    def test_rejects_grid_hidden_and_deepstack_mismatch(self):
        with self.assertRaisesRegex(ValueError, "spatial-merge"):
            validate_qwen3_vl_vision_embeddings(
                self.source, self.route, grid_thw=(1, 3, 4),
                hidden_states=FakeTensor((4, 24)),
                deepstack_visual_embeds=(FakeTensor((4, 24)), FakeTensor((4, 24))),
            )
        with self.assertRaisesRegex(ValueError, "hidden-state"):
            validate_qwen3_vl_vision_embeddings(
                self.source, self.route, grid_thw=(1, 4, 4),
                hidden_states=FakeTensor((3, 24)),
                deepstack_visual_embeds=(FakeTensor((4, 24)), FakeTensor((4, 24))),
            )
        with self.assertRaisesRegex(ValueError, "deep-stack"):
            validate_qwen3_vl_vision_embeddings(
                self.source, self.route, grid_thw=(1, 4, 4),
                hidden_states=FakeTensor((4, 24)),
                deepstack_visual_embeds=(FakeTensor((4, 24)),),
            )

    def test_async_bridge_hands_complete_bundle_to_gpu_llm(self):
        observed = []
        with AsyncEncoderLLMPipeline(self.ledger, self.registry) as pipeline:
            bridge = Qwen3VLANEGPUPipeline(
                pipeline,
                source=self.source,
                conversion=self.conversion,
                route=self.route,
            )
            future = bridge.submit(
                grid_thw=(1, 4, 4),
                encoder_memory_bytes=20,
                llm_backend=ExecutionBackend.VLLM_METAL,
                llm_memory_bytes=30,
                encode=lambda: (
                    FakeTensor((4, 24)),
                    (FakeTensor((4, 24)), FakeTensor((4, 24))),
                ),
                consume=lambda bundle: observed.append(bundle) or "generated",
            )
            result = future.result(timeout=2)
        self.assertEqual(result.output, "generated")
        self.assertEqual(len(observed), 1)
        self.assertEqual(len(observed[0].deepstack_visual_embeds), 2)
        self.assertEqual(self.ledger.snapshot()["active_reservations"], 0)

    def test_invalid_encoder_output_never_reaches_gpu_and_releases_resources(self):
        consumed = []
        with AsyncEncoderLLMPipeline(self.ledger, self.registry) as pipeline:
            bridge = Qwen3VLANEGPUPipeline(
                pipeline,
                source=self.source,
                conversion=self.conversion,
                route=self.route,
            )
            future = bridge.submit(
                grid_thw=(1, 4, 4),
                encoder_memory_bytes=20,
                llm_backend=ExecutionBackend.NATIVE_MLX,
                llm_memory_bytes=30,
                encode=lambda: (FakeTensor((4, 24)), ()),
                consume=lambda bundle: consumed.append(bundle),
            )
            with self.assertRaisesRegex(ValueError, "deep-stack"):
                future.result(timeout=2)
        self.assertEqual(consumed, [])
        self.assertEqual(self.ledger.snapshot()["active_reservations"], 0)

    def test_coreml_output_requires_exact_graph_provenance_and_order(self):
        graph_id = "e" * 64
        observed = []
        with AsyncEncoderLLMPipeline(self.ledger, self.registry) as pipeline:
            bridge = Qwen3VLANEGPUPipeline(
                pipeline,
                source=self.source,
                conversion=self.conversion,
                route=self.route,
                coreml_graph_id=graph_id,
            )
            output = Qwen3VLCoreMLPipelineOutput(
                FakeTensor((4, 24)),
                (FakeTensor((4, 24)), FakeTensor((4, 24))),
                (1, 4, 4),
                graph_id,
            )
            result = bridge.submit(
                grid_thw=(1, 4, 4),
                encoder_memory_bytes=20,
                llm_backend=ExecutionBackend.VLLM_METAL,
                llm_memory_bytes=30,
                encode=lambda: output,
                consume=lambda bundle: observed.append(bundle) or "generated",
            ).result(timeout=2)
        self.assertEqual(result.output, "generated")
        self.assertEqual(len(observed[0].deepstack_visual_embeds), 2)

    def test_coreml_output_rejects_missing_or_changed_graph_provenance(self):
        with AsyncEncoderLLMPipeline(self.ledger, self.registry) as pipeline:
            bridge = Qwen3VLANEGPUPipeline(
                pipeline,
                source=self.source,
                conversion=self.conversion,
                route=self.route,
                coreml_graph_id="e" * 64,
            )
            future = bridge.submit(
                grid_thw=(1, 4, 4),
                encoder_memory_bytes=20,
                llm_backend=ExecutionBackend.VLLM_METAL,
                llm_memory_bytes=30,
                encode=lambda: (
                    FakeTensor((4, 24)),
                    (FakeTensor((4, 24)), FakeTensor((4, 24))),
                ),
                consume=lambda bundle: bundle,
            )
            with self.assertRaisesRegex(ValueError, "provenance is missing"):
                future.result(timeout=2)

    def test_inline_bridge_validates_provenance_and_runs_consumer(self):
        graph_id = "e" * 64
        output = Qwen3VLCoreMLPipelineOutput(
            FakeTensor((4, 24)),
            (FakeTensor((4, 24)), FakeTensor((4, 24))),
            (1, 4, 4),
            graph_id,
        )
        with AsyncEncoderLLMPipeline(self.ledger, self.registry) as pipeline:
            bridge = Qwen3VLANEGPUPipeline(
                pipeline,
                source=self.source,
                conversion=self.conversion,
                route=self.route,
                coreml_graph_id=graph_id,
            )
            result = bridge.execute_inline(
                grid_thw=(1, 4, 4),
                encoder_memory_bytes=20,
                llm_backend=ExecutionBackend.NATIVE_MLX,
                llm_memory_bytes=30,
                encode=lambda: output,
                consume=lambda bundle: bundle.token_count,
            )
        self.assertEqual(result.output, 4)
        self.assertEqual(self.ledger.snapshot()["active_reservations"], 0)

    def test_builds_exact_vllm_metal_qwen3_vl_result(self):
        hidden = FakeTensor((4, 24))
        deepstack = (FakeTensor((4, 24)), FakeTensor((4, 24)))
        bundle = Qwen3VLVisionEmbeddingBundle(
            hidden, deepstack, (1, 4, 4), 4, 24, "capability"
        )
        result = build_vllm_metal_qwen3_vl_encode_result(
            bundle, result_type=FakeVLLMMetalResult
        )
        self.assertIs(result.hidden_states, hidden)
        self.assertEqual(tuple(result.deepstack_visual_embeds), deepstack)


if __name__ == "__main__":
    unittest.main()
