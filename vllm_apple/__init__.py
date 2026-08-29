"""Public package surface for vLLM-Apple."""

from .backend_tuning import (
    BackendKernelTuningAdapter,
    BackendTuningSnapshot,
    KernelTuningASGIMiddleware,
    PagedAttentionKernelInvoker,
    parse_kernel_tuning_headers,
)
from .elastic_memory import ElasticMemoryController, ElasticMemoryDecision
from .execution import AppleChipProfile, AppleExecutionPlan, AppleExecutionPlanner
from .execution_profile import detect_apple_chip_profile, load_chip_profile, save_chip_profile
from .context_reevaluation import ContextCapacityReevaluator, ContextReevaluationSnapshot
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
    candidate_configurations,
    load_v2_tuning_profile,
    inspect_v2_tuning_quarantine,
    quarantine_v2_tuning_profile,
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
from .vllm_metal_v2_adapter import (
    V2MeasurementAdapterError,
    VLLMMetalV2MeasurementAdapter,
    build_v2_measurement_request,
    parse_v2_measurement_request,
    parse_v2_measurement_response,
)

__all__ = [
    "API_VERSION",
    "SCHEMA_VERSION",
    "AppleChipProfile",
    "AppleExecutionPlan",
    "AppleExecutionPlanner",
    "BackendKernelTuningAdapter",
    "BackendStateReference",
    "BackendTuningSnapshot",
    "ContextCapacityReevaluator",
    "ContextReevaluationSnapshot",
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
    "LongContextEvaluator",
    "LongContextObservation",
    "MetalShapeBenchmark",
    "MetalShapeTuningDecision",
    "MetalThreadConfiguration",
    "MetalTuningReport",
    "MemoryBudgetComponent",
    "MemoryBudgetSnapshot",
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
    "__version__",
    "build_environment_fingerprint",
    "classify_runtime_failure",
    "build_mlx_probe_registry",
    "build_model_kernel_shape_profile",
    "build_v2_tuning_profile",
    "build_v2_measurement_request",
    "parse_v2_measurement_request",
    "parse_v2_measurement_response",
    "quarantine_v2_tuning_profile",
    "candidate_configurations",
    "default_metal_shape_benchmark_path",
    "default_metal_tuning_path",
    "default_v2_observation_path",
    "default_native_v2_preference_path",
    "detect_apple_chip_profile",
    "discover_metal_tuning_report",
    "discover_runtime_versions",
    "inspect_vllm_metal_integration",
    "inspect_v2_tuning_quarantine",
    "load_chip_profile",
    "load_metal_shape_benchmark",
    "load_metal_tuning_report",
    "load_v2_tuning_profile",
    "load_v2_observations",
    "load_native_v2_preference",
    "parse_kernel_tuning_headers",
    "persist_crash_diagnostic",
    "run_kernel_probe",
    "run_metal_shape_benchmark",
    "run_serving_promotion_probe",
    "record_v2_observed_shape",
    "save_chip_profile",
    "save_metal_shape_benchmark",
    "save_metal_tuning_report",
    "save_v2_tuning_profile",
    "save_native_v2_preference",
    "semantic_prefix_fingerprint",
    "tune_metal_shape_profile",
    "tune_v2_shape",
    "tune_v2_model_profile",
    "tune_v2_observed_shapes",
]
