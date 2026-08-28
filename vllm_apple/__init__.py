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
from .long_context import LongContextEvaluator, LongContextObservation
from .long_context_backend import VLLMLongContextAdapter
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
    save_v2_tuning_profile,
    tune_v2_model_profile,
    tune_v2_shape,
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
    "ModelKernelShapeProfile",
    "NativeMLXProbeAdapter",
    "NativeMetalProbeAdapter",
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
    "RuntimeProbeCoordinator",
    "RuntimeProbeReport",
    "SemanticAnchor",
    "SemanticAnchorCache",
    "SemanticAnchorKind",
    "SemanticCacheSnapshot",
    "SemanticRestoreResult",
    "SemanticStateBackend",
    "SemanticStateCoordinator",
    "V2DispatchConfiguration",
    "V2PagedAttentionFamily",
    "V2PagedAttentionShape",
    "V2ShapeTuningDecision",
    "VLLMLongContextAdapter",
    "VLLMMetalIntegrationInspection",
    "VLLMMetalV2TuningProfile",
    "V2MeasurementAdapterError",
    "VLLMMetalV2MeasurementAdapter",
    "__version__",
    "build_environment_fingerprint",
    "build_mlx_probe_registry",
    "build_model_kernel_shape_profile",
    "build_v2_tuning_profile",
    "build_v2_measurement_request",
    "parse_v2_measurement_request",
    "parse_v2_measurement_response",
    "candidate_configurations",
    "default_metal_shape_benchmark_path",
    "default_metal_tuning_path",
    "detect_apple_chip_profile",
    "discover_metal_tuning_report",
    "discover_runtime_versions",
    "inspect_vllm_metal_integration",
    "load_chip_profile",
    "load_metal_shape_benchmark",
    "load_metal_tuning_report",
    "load_v2_tuning_profile",
    "parse_kernel_tuning_headers",
    "run_kernel_probe",
    "run_metal_shape_benchmark",
    "run_serving_promotion_probe",
    "save_chip_profile",
    "save_metal_shape_benchmark",
    "save_metal_tuning_report",
    "save_v2_tuning_profile",
    "semantic_prefix_fingerprint",
    "tune_metal_shape_profile",
    "tune_v2_shape",
    "tune_v2_model_profile",
]
