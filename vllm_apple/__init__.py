"""Public package surface for vLLM-Apple."""

from .audio_ring_buffer import AudioRingBuffer, AudioRingBufferSnapshot
from .audio_preprocessing import (
    AudioFeatureFrame,
    StreamingLinearResampler,
    StreamingLogBandEncoder,
)
from .audio_streaming_state import (
    AudioStreamSnapshot,
    AudioStreamUpdate,
    StreamingAudioSession,
    StreamingAudioSessionRegistry,
)
from .audio_deadline_scheduler import (
    AudioDeadlineScheduler,
    AudioScheduledTask,
    AudioSchedulerSnapshot,
    AudioSchedulingPriority,
    AudioTaskOutcome,
)
from .asr_integration import (
    ASRBackend,
    ASRSubmission,
    ASRTranscript,
    ASRWorkItem,
    CallableASRBackend,
    StreamingASRIntegrator,
)
from .audio_encoder_cache import (
    AudioEncoderCache,
    AudioEncoderCacheKey,
    AudioEncoderCacheSnapshot,
    audio_feature_fingerprint,
)
from .speech_to_speech import (
    DialogueBackend,
    DialogueResponse,
    EchoDialogueBackend,
    SpeechSynthesisBackend,
    SpeechToSpeechPipeline,
    SpeechToSpeechResult,
    SynthesizedSpeech,
)
from .audio_benchmark import (
    AudioBenchmarkConfig,
    AudioBenchmarkReport,
    run_audio_benchmark,
)
from .video_decoder import (
    DecodedVideoFrames,
    FFmpegVideoToolboxDecoder,
    MappedDecodedVideoFrames,
    VideoStreamInfo,
)
from .video_metal_bridge import (
    NativeVideoMetalBridge,
    VideoMetalBridgeReport,
)
from .video_frame_scheduler import (
    ScheduledVideoFrame,
    VideoFrameDecision,
    VideoFrameScheduler,
    VideoFrameSchedulerSnapshot,
)
from .video_temporal_sampler import (
    TemporalSamplingReport,
    TemporalVideoFrame,
    sample_temporal_frames,
)
from .video_cache import (
    VideoArtifactCache,
    VideoCacheKey,
    VideoCacheKind,
    VideoCacheSnapshot,
    VideoCacheTierSnapshot,
)
from .video_vlm import (
    VideoFrameEmbedding,
    VideoFrameEncoder,
    VideoLanguageBackend,
    VideoVLMIntegrator,
    VideoVLMRequest,
    VideoVLMResult,
    embedding_digest,
)
from .video_streaming_input import (
    FinalizedVideoArtifact,
    StreamingVideoInputRegistry,
    StreamingVideoInputSession,
    StreamingVideoUpdate,
)
from .video_streaming_decoder import (
    IncrementalFFmpegVideoDecoder,
    StreamingVideoDecoderReport,
)
from .video_benchmark import (
    VideoBenchmarkReport,
    VideoBenchmarkThresholds,
    VideoDecodeMeasurement,
    run_video_benchmark,
)

from .backend_tuning import (
    BackendKernelTuningAdapter,
    BackendTuningSnapshot,
    KernelTuningASGIMiddleware,
    PagedAttentionKernelInvoker,
    parse_kernel_tuning_headers,
)
from .ane_probe import (
    CoreMLANEModelProbe,
    CoreMLANEModelProbeConfig,
    CoreMLANESurfaceProbe,
    CoreMLANESurfaceResult,
    CoreMLPrediction,
    run_coreml_prediction,
)
from .coreml_backend import (
    CoreMLFixedGraphBackend,
    CoreMLFixedGraphResource,
    CoreMLFixedGraphResult,
)
from .coreml_worker import CoreMLPersistentWorker
from .coreml_worker_cache import (
    CoreMLWorkerCache,
    CoreMLWorkerCacheKey,
    CoreMLWorkerLease,
)
from .coreml_artifact_cache import (
    CoreMLArtifactCache,
    CoreMLArtifactCacheEntry,
    CoreMLArtifactCacheIdentity,
)
from .backend_engine import (
    BackendEngine,
    BackendEngineAttempt,
    BackendEngineDescriptor,
    BackendEngineFailure,
    BackendEngineRegistry,
    BackendEngineRequest,
    BackendEngineResult,
)
from .backend_composition import (
    BackendRegistryInferenceEngine,
    BackendEngineRegistration,
    ManagedInferenceBackendEngine,
    ProductionBackendComposition,
)
from .operator_graph_dispatch import (
    OperatorGraphDispatcher,
    OperatorGraphNode,
    OperatorGraphResult,
)
from .workload_performance import (
    EndToEndPerformanceProfile,
    EndToEndPerformanceSample,
    EndToEndPhase,
    EndToEndPromotionDecision,
    build_end_to_end_performance_profile,
    evaluate_end_to_end_promotion,
)
from .adaptive_state_allocation import (
    AdaptiveStateAction,
    AdaptiveStateAllocator,
    AdaptiveStateBackend,
    AdaptiveStateCoordinator,
    AdaptiveStateDecision,
    AdaptiveStateKind,
    AdaptiveStatePlan,
    AdaptiveStateRecord,
    AdaptiveStateTransaction,
)
from .measured_state_backend import (
    MeasuredStateBackendAdapter,
    StatePrecisionGate,
    StatePrecisionMeasurement,
)
from .numeric_routing import (
    NumericCapability,
    NumericEligibilityDecision,
    NumericEligibilityMatrix,
    NumericEligibilityRequest,
    NumericFormat,
    NumericRouteDecision,
    NumericRouteProfile,
    NumericRouteStrategy,
    NumericTensorRole,
    choose_numeric_route,
)
from .numeric_conversion_cache import (
    NumericConversionCache,
    NumericConversionCacheEntry,
    NumericConversionCacheIdentity,
)
from .numeric_promotion import (
    NumericPromotionDecision,
    NumericPromotionEvidence,
    NumericPromotionThresholds,
    evaluate_numeric_promotion,
)
from .fault_injection import (
    DeterministicFaultInjector,
    FaultAction,
    FaultPoint,
    FaultRule,
    InjectedFault,
)
from .mtls_authorization import (
    ClientCertificatePolicy,
    ClientCertificatePolicyStore,
)
from .process_inference_engine import MainThreadSubprocessInferenceEngine
from .elastic_memory import ElasticMemoryController, ElasticMemoryDecision
from .device_capability import (
    ComputeDevice,
    DeviceCapability,
    DeviceCapabilityRegistry,
    DeviceEligibilityDecision,
    DeviceEligibilityRequest,
    compose_device_capability_registry,
    device_capability_from_probe,
)
from .device_benchmark import (
    BoundedCPUReferenceBenchmarkAdapter,
    DeviceBenchmarkConfig,
    DeviceBenchmarkMeasurement,
    DeviceBenchmarkReport,
    DeviceBenchmarkSuite,
    CoreMLFixedGraphBenchmarkAdapter,
    NativeCPUBenchmarkAdapter,
    NativeKernelBenchmarkAdapter,
    representative_device_benchmark_configs,
    run_device_benchmark_suite,
    load_device_benchmark,
    load_profile_device_benchmark,
    run_device_microbenchmark,
    save_device_benchmark,
)
from .device_selection import (
    DevicePlacementCandidate,
    DevicePlacementDecision,
    select_measured_device_backend,
)
from .device_placement import (
    DevicePlacement,
    DevicePlacementPlan,
    build_device_placement_plan,
    default_device_placement_paths,
    load_device_placement_plan,
    load_device_placement_with_fallback,
    promote_device_placement_plan,
    save_device_placement_plan,
)
from .device_resources import (
    BandwidthContentionEvidence,
    DeviceResourceCapacityError,
    DeviceResourceRequest,
    DeviceResourceReservation,
    UnifiedDeviceResourceLedger,
    contention_profile_id,
)
from .device_pipeline import (
    ANEAuxiliaryRoute,
    ANEAuxiliaryWorkload,
    AsyncEncoderLLMPipeline,
    DevicePipelineExecutor,
    DevicePipelineResult,
    DevicePipelineStage,
    EncoderLLMPipelineResult,
    require_ane_auxiliary_route,
)
from .device_contention import (
    ContentionBenchmarkConfig,
    ContentionProfile,
    default_contention_profile_path,
    default_contention_profile_paths,
    install_contention_profile,
    load_contention_profile,
    load_contention_profile_with_fallback,
    promote_contention_profile,
    run_contention_benchmark,
    save_contention_profile,
)
from .execution import AppleChipProfile, AppleExecutionPlan, AppleExecutionPlanner
from .execution_profile import detect_apple_chip_profile, load_chip_profile, save_chip_profile
from .context_reevaluation import ContextCapacityReevaluator, ContextReevaluationSnapshot
from .cpu_probe import NativeCPUProbeAdapter
from .kernel_context import InferenceKernelContext, PagedAttentionKernelSelection
from .kernel_probe import (
    KernelCapabilityRegistry,
    KernelMeasurement,
    KernelProbeCache,
    KernelProbeConfig,
    KernelProbeResult,
    build_environment_fingerprint,
    run_kernel_probe,
)
from .kernel_profile import (
    ModelKernelShapeProfile,
    PagedAttentionShape,
    build_model_kernel_shape_profile,
)
from .kernel_stress import MultiModelStressReport, run_multi_model_command_stress
from .graph_fusion import (
    CapabilityGatedGraphFusionPass,
    FusionRule,
    GraphFusionResult,
)
from .long_context import (
    LongContextEvaluator,
    LongContextObservation,
    save_long_context_report,
)
from .long_context_backend import MLXLongContextAdapter, VLLMLongContextAdapter
from .kv_calibration import (
    KVCalibration,
    default_calibration_report_path,
    discover_latest_kv_calibration,
    load_kv_calibration,
)
from .memory_budget import (
    MemoryBudgetComponent,
    MemoryBudgetSnapshot,
    UnifiedMemoryBudgetLedger,
)
from .metal_probe import (
    MetalShapeTuningDecision,
    MetalThreadConfiguration,
    NativeMetalProbeAdapter,
)
from .metal_tuning import (
    MetalTuningReport,
    default_metal_tuning_path,
    discover_metal_tuning_report,
    load_metal_tuning_report,
    save_metal_tuning_report,
    tune_metal_shape_profile,
)
from .mlx_probe import NativeMLXProbeAdapter, build_mlx_probe_registry
from .mlx_phase3_probe import MLXPhase3ProbeAdapter
from .mlx_vision_probe import MLXVisionFusionProbeAdapter
from .mlx_vision_benchmark import (
    MLXVisionBenchmarkAdapter,
    VisionBatchBenchmark,
    VisionBenchmarkMeasurement,
    VisionBenchmarkReport,
)
from .operator_dispatch import (
    OperatorDispatchDecision,
    OperatorDispatcher,
    OperatorDispatchRequest,
)
from .phase_profile import ExecutionPhaseProfiler, PhaseMeasurement
from .promotion_probe import (
    PromotionProbeConfig,
    PromotionProbeError,
    PromotionResponse,
    run_serving_promotion_probe,
)
from .runtime_probe import (
    RuntimeEnvironmentVersions,
    RuntimeProbeCoordinator,
    RuntimeProbeReport,
    discover_runtime_versions,
)
from .runtime_errors import (
    RuntimeFailure,
    RuntimeFailureCode,
    RuntimeRecoverability,
    classify_runtime_failure,
    persist_crash_diagnostic,
)
from .vision_frontend import (
    VisionChatInput,
    VisionImageInput,
    VisionPreprocessResult,
    VisionPreprocessSpec,
    parse_vision_chat_request,
    preprocess_vision_image,
)
from .vision_cache import (
    VisionCacheKey,
    VisionCacheStats,
    VisionEncoderCache,
    preprocessing_fingerprint,
)
from .vision_batching import (
    VisionBatch,
    VisionBatchCompatibility,
    VisionBatchLimits,
    VisionBatchRequest,
    plan_multimodal_batches,
)
from .qwen3_vl_ane import (
    Qwen3VLVisionANEAdapterSpec,
    inspect_qwen3_vl_vision_for_ane,
)
from .qwen3_vl_coreml import (
    Qwen3VLCoreMLConversionManifest,
    load_qwen3_vl_coreml_conversion,
    qualify_qwen3_vl_coreml_conversion,
    save_qwen3_vl_coreml_conversion,
)
from .qwen3_vl_conversion_plan import (
    Qwen3VLCoreMLConversionPlan,
    build_qwen3_vl_coreml_conversion_plan,
)
from .qwen3_vl_conversion_worker import stage_qwen3_vl_coreml_weights
from .qwen3_vl_graph_spec import (
    Qwen3VLCoreMLGraphProfile,
    Qwen3VLCoreMLGraphSpec,
    build_qwen3_vl_coreml_graph_spec,
)
from .qwen3_vl_patch_coreml import (
    build_qwen3_vl_patch_coreml,
    qualify_qwen3_vl_patch_coreml,
)
from .qwen3_vl_mlp_coreml import (
    build_qwen3_vl_mlp_coreml,
    qualify_qwen3_vl_mlp_coreml,
)
from .qwen3_vl_attention_coreml import (
    build_qwen3_vl_attention_coreml,
    qualify_qwen3_vl_attention_coreml,
)
from .qwen3_vl_block_coreml import (
    build_qwen3_vl_block_coreml,
    qualify_qwen3_vl_block_coreml,
)
from .qwen3_vl_tower_coreml import (
    build_qwen3_vl_tower_blocks_coreml,
    qualify_qwen3_vl_tower_blocks_coreml,
)
from .qwen3_vl_deepstack_coreml import (
    build_qwen3_vl_deepstack_coreml,
    build_qwen3_vl_final_coreml,
    qualify_qwen3_vl_deepstack_coreml,
    qualify_qwen3_vl_final_coreml,
)
from .qwen3_vl_pipeline_coreml import qualify_qwen3_vl_segment_pipeline_coreml
from .qwen3_vl_pipeline_coreml import export_qwen3_vl_segment_pipeline_coreml
from .qwen3_vl_transport import load_qwen3_vl_coreml_transport
from .qwen3_vl_compute_plan import inspect_qwen3_vl_coreml_compute_plans
from .qwen3_vl_persistent_worker import Qwen3VLPersistentWorker
from .qwen3_vl_persistent_encoder import (
    Qwen3VLPersistentEncoder,
    publish_qwen3_vl_persistent_transport_manifest,
)
from .qwen3_vl_embedding import (
    build_vllm_metal_qwen3_vl_encode_result,
    Qwen3VLCoreMLPipelineOutput,
    Qwen3VLANEGPUPipeline,
    Qwen3VLVisionEmbeddingBundle,
    validate_qwen3_vl_vision_embeddings,
)
from .semantic_cache import (
    SemanticAnchor,
    SemanticAnchorCache,
    SemanticAnchorKind,
    SemanticCacheSnapshot,
    semantic_prefix_fingerprint,
)
from .semantic_state import (
    BackendStateReference,
    SemanticRestoreResult,
    SemanticStateBackend,
    SemanticStateCoordinator,
)
from .mlx_semantic_state import MLXPromptCacheStateAdapter
from .shape_benchmark import (
    MetalShapeBenchmark,
    default_metal_shape_benchmark_path,
    load_metal_shape_benchmark,
    run_metal_shape_benchmark,
    save_metal_shape_benchmark,
)
from .version import API_VERSION, SCHEMA_VERSION, __version__
from .vllm_metal_integration import (
    VLLMMetalIntegrationInspection,
    inspect_vllm_metal_integration,
)
from .vllm_metal_v2_tuning import (
    V2DispatchConfiguration,
    V2PagedAttentionFamily,
    V2PagedAttentionShape,
    V2ShapeTuningDecision,
    VLLMMetalV2TuningProfile,
    build_v2_tuning_profile,
    build_v2_environment_fingerprint,
    candidate_configurations,
    load_v2_tuning_profile,
    inspect_v2_tuning_quarantine,
    quarantine_v2_tuning_profile,
    restore_quarantined_v2_profile,
    save_v2_tuning_profile,
    tune_v2_model_profile,
    tune_v2_observed_shapes,
    tune_v2_shape,
)
from .vllm_metal_v2_observation import (
    default_v2_observation_path,
    load_v2_observations,
    record_v2_observed_shape,
)
from .vllm_metal_v2_orchestration import (
    NativeV2IdleTuningCoordinator,
    NativeV2ObservationMonitor,
    V2IdleTuningSnapshot,
)
from .vllm_metal_v2_preference import (
    default_native_v2_preference_path,
    load_native_v2_preference,
    save_native_v2_preference,
)
from .scheduling_preference import (
    default_scheduling_preference_path,
    load_scheduling_preference,
    save_scheduling_preference,
)
from .vllm_metal_v2_adapter import (
    V2MeasurementAdapterError,
    VLLMMetalV2MeasurementAdapter,
    build_v2_measurement_request,
    parse_v2_measurement_request,
    parse_v2_measurement_response,
)

__all__ = [
    "AudioRingBuffer",
    "AudioRingBufferSnapshot",
    "AudioFeatureFrame",
    "AudioEncoderCache",
    "AudioEncoderCacheKey",
    "AudioEncoderCacheSnapshot",
    "AudioDeadlineScheduler",
    "AudioBenchmarkConfig",
    "AudioBenchmarkReport",
    "AudioScheduledTask",
    "AudioSchedulerSnapshot",
    "AudioSchedulingPriority",
    "AudioTaskOutcome",
    "audio_feature_fingerprint",
    "run_audio_benchmark",
    "run_video_benchmark",
    "ASRBackend",
    "ASRSubmission",
    "ASRTranscript",
    "ASRWorkItem",
    "CallableASRBackend",
    "DialogueBackend",
    "DialogueResponse",
    "DecodedVideoFrames",
    "EchoDialogueBackend",
    "FFmpegVideoToolboxDecoder",
    "FinalizedVideoArtifact",
    "IncrementalFFmpegVideoDecoder",
    "MappedDecodedVideoFrames",
    "NativeVideoMetalBridge",
    "AudioStreamSnapshot",
    "AudioStreamUpdate",
    "StreamingLinearResampler",
    "StreamingLogBandEncoder",
    "StreamingAudioSession",
    "StreamingAudioSessionRegistry",
    "StreamingASRIntegrator",
    "StreamingVideoInputRegistry",
    "StreamingVideoInputSession",
    "StreamingVideoUpdate",
    "StreamingVideoDecoderReport",
    "SpeechSynthesisBackend",
    "SpeechToSpeechPipeline",
    "SpeechToSpeechResult",
    "SynthesizedSpeech",
    "TemporalSamplingReport",
    "TemporalVideoFrame",
    "ANEAuxiliaryRoute",
    "ANEAuxiliaryWorkload",
    "AsyncEncoderLLMPipeline",
    "API_VERSION",
    "SCHEMA_VERSION",
    "AppleChipProfile",
    "AppleExecutionPlan",
    "AppleExecutionPlanner",
    "AdaptiveStateAction",
    "AdaptiveStateAllocator",
    "AdaptiveStateBackend",
    "AdaptiveStateCoordinator",
    "AdaptiveStateDecision",
    "AdaptiveStateKind",
    "AdaptiveStatePlan",
    "AdaptiveStateRecord",
    "AdaptiveStateTransaction",
    "MeasuredStateBackendAdapter",
    "StatePrecisionGate",
    "StatePrecisionMeasurement",
    "NumericCapability",
    "NumericConversionCache",
    "NumericConversionCacheEntry",
    "NumericConversionCacheIdentity",
    "NumericEligibilityDecision",
    "NumericEligibilityMatrix",
    "NumericEligibilityRequest",
    "NumericFormat",
    "NumericPromotionDecision",
    "NumericPromotionEvidence",
    "NumericPromotionThresholds",
    "NumericRouteDecision",
    "NumericRouteProfile",
    "NumericRouteStrategy",
    "NumericTensorRole",
    "BackendKernelTuningAdapter",
    "BackendEngine",
    "BackendEngineAttempt",
    "BackendEngineDescriptor",
    "BackendEngineFailure",
    "BackendEngineRegistry",
    "BackendEngineRequest",
    "BackendEngineResult",
    "BackendRegistryInferenceEngine",
    "BackendEngineRegistration",
    "ManagedInferenceBackendEngine",
    "ProductionBackendComposition",
    "OperatorGraphDispatcher",
    "OperatorGraphNode",
    "OperatorGraphResult",
    "EndToEndPerformanceProfile",
    "EndToEndPerformanceSample",
    "EndToEndPhase",
    "EndToEndPromotionDecision",
    "build_end_to_end_performance_profile",
    "evaluate_end_to_end_promotion",
    "BackendStateReference",
    "BackendTuningSnapshot",
    "ContextCapacityReevaluator",
    "ClientCertificatePolicy",
    "ClientCertificatePolicyStore",
    "CoreMLANESurfaceProbe",
    "CoreMLArtifactCache",
    "CoreMLArtifactCacheEntry",
    "CoreMLArtifactCacheIdentity",
    "CoreMLANESurfaceResult",
    "CoreMLANEModelProbe",
    "CoreMLANEModelProbeConfig",
    "CoreMLFixedGraphBackend",
    "CoreMLFixedGraphResource",
    "CoreMLFixedGraphResult",
    "CoreMLPersistentWorker",
    "CoreMLWorkerCache",
    "CoreMLWorkerCacheKey",
    "CoreMLWorkerLease",
    "CoreMLFixedGraphBenchmarkAdapter",
    "CoreMLPrediction",
    "ContextReevaluationSnapshot",
    "ComputeDevice",
    "DeviceCapability",
    "DeviceCapabilityRegistry",
    "DeviceBenchmarkConfig",
    "DeviceBenchmarkMeasurement",
    "DeviceBenchmarkReport",
    "DeviceBenchmarkSuite",
    "DeterministicFaultInjector",
    "BoundedCPUReferenceBenchmarkAdapter",
    "DeviceEligibilityDecision",
    "DeviceEligibilityRequest",
    "FaultAction",
    "FaultPoint",
    "FaultRule",
    "InjectedFault",
    "MainThreadSubprocessInferenceEngine",
    "Qwen3VLVisionANEAdapterSpec",
    "Qwen3VLCoreMLConversionManifest",
    "Qwen3VLCoreMLConversionPlan",
    "Qwen3VLCoreMLGraphProfile",
    "Qwen3VLCoreMLGraphSpec",
    "Qwen3VLANEGPUPipeline",
    "Qwen3VLCoreMLPipelineOutput",
    "Qwen3VLVisionEmbeddingBundle",
    "Qwen3VLPersistentEncoder",
    "Qwen3VLPersistentWorker",
    "DevicePlacementCandidate",
    "DevicePlacementDecision",
    "DevicePlacement",
    "DevicePlacementPlan",
    "DeviceResourceCapacityError",
    "DevicePipelineExecutor",
    "DevicePipelineResult",
    "DevicePipelineStage",
    "EncoderLLMPipelineResult",
    "BandwidthContentionEvidence",
    "ContentionBenchmarkConfig",
    "ContentionProfile",
    "default_contention_profile_path",
    "default_contention_profile_paths",
    "DeviceResourceRequest",
    "DeviceResourceReservation",
    "UnifiedDeviceResourceLedger",
    "install_contention_profile",
    "inspect_qwen3_vl_vision_for_ane",
    "inspect_qwen3_vl_coreml_compute_plans",
    "build_qwen3_vl_coreml_conversion_plan",
    "build_vllm_metal_qwen3_vl_encode_result",
    "build_qwen3_vl_coreml_graph_spec",
    "build_qwen3_vl_patch_coreml",
    "build_qwen3_vl_mlp_coreml",
    "build_qwen3_vl_attention_coreml",
    "build_qwen3_vl_block_coreml",
    "build_qwen3_vl_tower_blocks_coreml",
    "build_qwen3_vl_deepstack_coreml",
    "build_qwen3_vl_final_coreml",
    "stage_qwen3_vl_coreml_weights",
    "load_qwen3_vl_coreml_conversion",
    "contention_profile_id",
    "load_contention_profile",
    "load_contention_profile_with_fallback",
    "promote_contention_profile",
    "run_contention_benchmark",
    "require_ane_auxiliary_route",
    "qualify_qwen3_vl_coreml_conversion",
    "qualify_qwen3_vl_patch_coreml",
    "qualify_qwen3_vl_mlp_coreml",
    "qualify_qwen3_vl_attention_coreml",
    "qualify_qwen3_vl_block_coreml",
    "qualify_qwen3_vl_tower_blocks_coreml",
    "qualify_qwen3_vl_deepstack_coreml",
    "qualify_qwen3_vl_final_coreml",
    "qualify_qwen3_vl_segment_pipeline_coreml",
    "export_qwen3_vl_segment_pipeline_coreml",
    "load_qwen3_vl_coreml_transport",
    "publish_qwen3_vl_persistent_transport_manifest",
    "save_contention_profile",
    "save_qwen3_vl_coreml_conversion",
    "validate_qwen3_vl_vision_embeddings",
    "ElasticMemoryController",
    "ElasticMemoryDecision",
    "ExecutionPhaseProfiler",
    "InferenceKernelContext",
    "KernelCapabilityRegistry",
    "KernelMeasurement",
    "KernelProbeCache",
    "KernelProbeConfig",
    "KernelProbeResult",
    "KernelTuningASGIMiddleware",
    "CapabilityGatedGraphFusionPass",
    "FusionRule",
    "GraphFusionResult",
    "LongContextEvaluator",
    "LongContextObservation",
    "MetalShapeBenchmark",
    "MetalShapeTuningDecision",
    "MetalThreadConfiguration",
    "MetalTuningReport",
    "MLXPhase3ProbeAdapter",
    "MLXVisionFusionProbeAdapter",
    "MLXVisionBenchmarkAdapter",
    "MultiModelStressReport",
    "MemoryBudgetComponent",
    "MemoryBudgetSnapshot",
    "NativeCPUProbeAdapter",
    "NativeCPUBenchmarkAdapter",
    "NativeKernelBenchmarkAdapter",
    "MLXPromptCacheStateAdapter",
    "ModelKernelShapeProfile",
    "NativeMLXProbeAdapter",
    "NativeMetalProbeAdapter",
    "NativeV2IdleTuningCoordinator",
    "NativeV2ObservationMonitor",
    "OperatorDispatchDecision",
    "OperatorDispatchRequest",
    "OperatorDispatcher",
    "PagedAttentionKernelInvoker",
    "PagedAttentionKernelSelection",
    "PagedAttentionShape",
    "PhaseMeasurement",
    "PromotionProbeConfig",
    "PromotionProbeError",
    "PromotionResponse",
    "RuntimeEnvironmentVersions",
    "RuntimeFailure",
    "RuntimeFailureCode",
    "RuntimeRecoverability",
    "RuntimeProbeCoordinator",
    "RuntimeProbeReport",
    "SemanticAnchor",
    "SemanticAnchorCache",
    "SemanticAnchorKind",
    "SemanticCacheSnapshot",
    "SemanticRestoreResult",
    "SemanticStateBackend",
    "SemanticStateCoordinator",
    "ScheduledVideoFrame",
    "UnifiedMemoryBudgetLedger",
    "V2DispatchConfiguration",
    "V2PagedAttentionFamily",
    "V2PagedAttentionShape",
    "V2ShapeTuningDecision",
    "VLLMLongContextAdapter",
    "MLXLongContextAdapter",
    "KVCalibration",
    "load_kv_calibration",
    "default_calibration_report_path",
    "discover_latest_kv_calibration",
    "save_long_context_report",
    "VLLMMetalIntegrationInspection",
    "VLLMMetalV2TuningProfile",
    "V2MeasurementAdapterError",
    "V2IdleTuningSnapshot",
    "VLLMMetalV2MeasurementAdapter",
    "VisionCacheKey",
    "VisionCacheStats",
    "VisionBatch",
    "VisionBatchCompatibility",
    "VisionBatchLimits",
    "VisionBatchRequest",
    "VisionBatchBenchmark",
    "VisionBenchmarkMeasurement",
    "VisionBenchmarkReport",
    "VisionChatInput",
    "VisionEncoderCache",
    "VisionImageInput",
    "VisionPreprocessResult",
    "VisionPreprocessSpec",
    "VideoStreamInfo",
    "VideoMetalBridgeReport",
    "VideoFrameDecision",
    "VideoFrameEmbedding",
    "VideoFrameEncoder",
    "VideoFrameScheduler",
    "VideoFrameSchedulerSnapshot",
    "VideoArtifactCache",
    "VideoBenchmarkReport",
    "VideoBenchmarkThresholds",
    "VideoCacheKey",
    "VideoCacheKind",
    "VideoCacheSnapshot",
    "VideoCacheTierSnapshot",
    "VideoDecodeMeasurement",
    "VideoLanguageBackend",
    "VideoVLMIntegrator",
    "VideoVLMRequest",
    "VideoVLMResult",
    "__version__",
    "build_environment_fingerprint",
    "choose_numeric_route",
    "evaluate_numeric_promotion",
    "build_device_placement_plan",
    "classify_runtime_failure",
    "compose_device_capability_registry",
    "build_mlx_probe_registry",
    "build_model_kernel_shape_profile",
    "build_v2_tuning_profile",
    "build_v2_environment_fingerprint",
    "build_v2_measurement_request",
    "parse_v2_measurement_request",
    "parse_v2_measurement_response",
    "quarantine_v2_tuning_profile",
    "restore_quarantined_v2_profile",
    "candidate_configurations",
    "default_metal_shape_benchmark_path",
    "default_device_placement_paths",
    "default_metal_tuning_path",
    "default_v2_observation_path",
    "default_native_v2_preference_path",
    "default_scheduling_preference_path",
    "detect_apple_chip_profile",
    "device_capability_from_probe",
    "embedding_digest",
    "discover_metal_tuning_report",
    "discover_runtime_versions",
    "inspect_vllm_metal_integration",
    "inspect_v2_tuning_quarantine",
    "load_chip_profile",
    "load_device_benchmark",
    "load_profile_device_benchmark",
    "load_device_placement_plan",
    "load_device_placement_with_fallback",
    "load_metal_shape_benchmark",
    "load_metal_tuning_report",
    "load_v2_tuning_profile",
    "load_v2_observations",
    "load_native_v2_preference",
    "load_scheduling_preference",
    "parse_kernel_tuning_headers",
    "parse_vision_chat_request",
    "plan_multimodal_batches",
    "preprocess_vision_image",
    "preprocessing_fingerprint",
    "promote_device_placement_plan",
    "persist_crash_diagnostic",
    "run_kernel_probe",
    "run_multi_model_command_stress",
    "run_coreml_prediction",
    "run_device_microbenchmark",
    "run_device_benchmark_suite",
    "representative_device_benchmark_configs",
    "run_metal_shape_benchmark",
    "run_serving_promotion_probe",
    "record_v2_observed_shape",
    "save_chip_profile",
    "save_device_benchmark",
    "save_device_placement_plan",
    "save_metal_shape_benchmark",
    "save_metal_tuning_report",
    "save_v2_tuning_profile",
    "sample_temporal_frames",
    "save_native_v2_preference",
    "save_scheduling_preference",
    "semantic_prefix_fingerprint",
    "select_measured_device_backend",
    "tune_metal_shape_profile",
    "tune_v2_shape",
    "tune_v2_model_profile",
    "tune_v2_observed_shapes",
]
