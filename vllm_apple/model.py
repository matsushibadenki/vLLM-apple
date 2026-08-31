from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import GIB, HardwareInfo, ModelMemorySpec, StateMemorySpec


DEFAULT_UNINSPECTED_CONTEXT = 4096
MAX_MODEL_CONFIG_BYTES = 1024 * 1024
MAX_MODEL_CONFIG_NODES = 4096
MAX_MODEL_CONFIG_DEPTH = 16
_WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".bin")


class ModelInspectionError(RuntimeError):
    pass


class ModelCapabilityError(ModelInspectionError):
    pass


@dataclass(frozen=True, slots=True)
class ModelArchitectureCapability:
    architecture: str
    required_features: tuple[str, ...]
    modes: tuple[str, ...] = ("text",)
    native_context_tokens: int | None = None
    extended_context_tokens: int | None = None

    @property
    def requires_hybrid_backend(self) -> bool:
        return bool(self.required_features)


@dataclass(frozen=True, slots=True)
class ModelMemoryFit:
    artifact_bytes: int
    estimated_resident_bytes: int
    hard_ceiling_bytes: int
    context_tokens: int
    fits: bool


def assess_model_memory_fit(
    model: InspectedModel,
    hardware: HardwareInfo,
    *,
    context_tokens: int,
) -> ModelMemoryFit:
    if context_tokens <= 0:
        raise ValueError("model fit context must be positive")
    state = model.state_memory_spec or model.memory_spec.as_state_memory_spec()
    emergency_margin = max(GIB, int(hardware.memory.total_bytes * 0.08))
    hard_ceiling = max(0, hardware.memory.available_bytes - emergency_margin)
    resident = state.total_bytes(context_tokens)
    return ModelMemoryFit(
        artifact_bytes=model.memory_spec.weights_bytes,
        estimated_resident_bytes=resident,
        hard_ceiling_bytes=hard_ceiling,
        context_tokens=context_tokens,
        fits=resident <= hard_ceiling,
    )


@dataclass(frozen=True, slots=True)
class InspectedModel:
    model_id: str
    path: Path
    config: dict[str, Any]
    memory_spec: ModelMemorySpec
    kv_dtype_bytes: int
    architecture_capability: ModelArchitectureCapability = ModelArchitectureCapability(
        "unknown", ()
    )
    state_memory_spec: StateMemorySpec | None = None


def _bounded_config(path: Path) -> dict[str, Any]:
    try:
        attributes = path.stat()
        if not path.is_file() or not 1 <= attributes.st_size <= MAX_MODEL_CONFIG_BYTES:
            raise ModelInspectionError("config.json size is outside the bounded limit")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ModelInspectionError("config.json was not found") from error
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelInspectionError("config.json is not readable JSON") from error
    if not isinstance(payload, dict):
        raise ModelInspectionError("model config must be an object")
    nodes = 0
    stack: list[tuple[object, int]] = [(payload, 1)]
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > MAX_MODEL_CONFIG_NODES or depth > MAX_MODEL_CONFIG_DEPTH:
            raise ModelInspectionError("model config structure exceeds bounded limits")
        if isinstance(value, dict):
            stack.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            stack.extend((item, depth + 1) for item in value)
    return payload


def inspect_model_metadata(path: str | Path) -> tuple[dict[str, Any], ModelArchitectureCapability]:
    """Inspect bounded local config metadata without requiring model weights."""
    root = Path(path).expanduser()
    config_path = root / "config.json" if root.is_dir() else root
    config = _bounded_config(config_path)
    return config, inspect_model_architecture(config)


def inspect_model_architecture(config: dict[str, Any]) -> ModelArchitectureCapability:
    model_type = config.get("model_type")
    architectures = config.get("architectures")
    architecture = model_type if isinstance(model_type, str) else "unknown"
    names = [architecture]
    if isinstance(architectures, list):
        names.extend(value for value in architectures if isinstance(value, str))
    normalized = " ".join(names).lower()
    if "qwen4_exp" in normalized or "qwen3.8-flash-next" in normalized:
        text = config.get("text_config", config)
        if not isinstance(text, dict):
            raise ModelInspectionError("qwen4_exp text_config is missing")
        language_model_only = config.get("language_model_only", True)
        if not isinstance(language_model_only, bool):
            raise ModelInspectionError("qwen4_exp language_model_only must be boolean")
        modes = ["text"]
        features = [
            "gated_deltanet",
            "qwen_sparse_attention",
            "mixture_of_experts",
            "gated_residual",
            "ngram_embedding",
        ]
        mtp_layers = text.get("mtp_num_hidden_layers", 0)
        if not isinstance(mtp_layers, int) or isinstance(mtp_layers, bool) or mtp_layers < 0:
            raise ModelInspectionError("qwen4_exp MTP layer count is invalid")
        if mtp_layers > 0:
            modes.append("mtp")
            features.append("multi_token_prediction")
        if not language_model_only:
            if not isinstance(config.get("vision_config"), dict):
                raise ModelInspectionError("qwen4_exp vision_config is missing")
            modes.append("vision")
            features.append("vision_encoder")
        native_context = text.get("max_position_embeddings")
        if (
            not isinstance(native_context, int)
            or isinstance(native_context, bool)
            or not 1 <= native_context <= 16_777_216
        ):
            raise ModelInspectionError("qwen4_exp native context is invalid")
        features.append("native_long_context")
        extended_context = None
        rope = text.get("rope_parameters", text.get("rope_scaling"))
        if isinstance(rope, dict) and str(rope.get("rope_type") or "").lower() == "yarn":
            factor = rope.get("factor")
            if not isinstance(factor, (int, float)) or isinstance(factor, bool) or factor <= 1:
                raise ModelInspectionError("qwen4_exp YaRN factor is invalid")
            extended_context = min(16_777_216, int(native_context * factor))
            modes.append("yarn")
            features.append("yarn_extended_context")
        return ModelArchitectureCapability(
            architecture="qwen4_exp",
            required_features=tuple(features),
            modes=tuple(modes),
            native_context_tokens=native_context,
            extended_context_tokens=extended_context,
        )
    return ModelArchitectureCapability(architecture, ())


def ensure_model_backend_compatible(
    model: InspectedModel,
    *,
    backend: str = "vllm_metal",
    available_features: frozenset[str] | None = None,
    requested_modes: frozenset[str] | None = None,
) -> None:
    ensure_architecture_backend_compatible(
        model.architecture_capability,
        backend=backend,
        available_features=available_features,
        requested_modes=requested_modes,
    )


def ensure_architecture_backend_compatible(
    capability: ModelArchitectureCapability,
    *,
    backend: str = "vllm_metal",
    available_features: frozenset[str] | None = None,
    requested_modes: frozenset[str] | None = None,
) -> None:
    supported_modes = frozenset(capability.modes)
    if requested_modes is not None:
        unsupported_modes = requested_modes - supported_modes
        if unsupported_modes:
            modes = ",".join(sorted(unsupported_modes))
            raise ModelCapabilityError(f"model_missing_requested_modes:{modes}")
    required_features = capability.required_features
    if requested_modes is not None:
        optional_features = {
            "vision_encoder": "vision",
            "multi_token_prediction": "mtp",
            "yarn_extended_context": "yarn",
        }
        required_features = tuple(
            feature
            for feature in required_features
            if optional_features.get(feature) in requested_modes
            or feature not in optional_features
        )
    if available_features is None:
        available_features = (
            frozenset()
            if backend in {"vllm_metal", "mlx_lm"}
            else frozenset(required_features)
        )
    missing_features = tuple(
        feature for feature in required_features if feature not in available_features
    )
    if missing_features:
        missing = ",".join(missing_features)
        raise ModelCapabilityError(f"backend_missing_model_capabilities:{missing}")


def qwen4_exp_recurrent_state_bytes(config: dict[str, Any]) -> int:
    text = config.get("text_config", config)
    if not isinstance(text, dict) or text.get("model_type") != "qwen4_exp_text":
        raise ModelInspectionError("qwen4_exp text_config is missing")
    layer_types = text.get("layer_types")
    layers = text.get("num_hidden_layers")
    if (
        not isinstance(layer_types, list)
        or not isinstance(layers, int)
        or isinstance(layers, bool)
        or not 1 <= layers <= 512
        or len(layer_types) != layers
        or any(value not in {"linear_attention", "full_attention"} for value in layer_types)
    ):
        raise ModelInspectionError("qwen4_exp layer_types are invalid")

    def bounded_positive(name: str, maximum: int = 65_536) -> int:
        value = text.get(name)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 1 <= value <= maximum
        ):
            raise ModelInspectionError(f"qwen4_exp {name} is invalid")
        return value

    key_heads = bounded_positive("linear_num_key_heads")
    value_heads = bounded_positive("linear_num_value_heads")
    key_dim = bounded_positive("linear_key_head_dim")
    value_dim = bounded_positive("linear_value_head_dim")
    conv_kernel = bounded_positive("linear_conv_kernel_dim", 64)
    dtype = text.get("mamba_ssm_dtype")
    dtype_bytes = {"float32": 4, "float16": 2, "bfloat16": 2}.get(dtype)
    if dtype_bytes is None:
        raise ModelInspectionError("qwen4_exp mamba_ssm_dtype is unsupported")
    linear_layers = sum(value == "linear_attention" for value in layer_types)
    matrix_elements = value_heads * key_dim * value_dim
    conv_channels = 2 * key_heads * key_dim + value_heads * value_dim
    conv_elements = conv_channels * (conv_kernel - 1)
    return linear_layers * (matrix_elements + conv_elements) * dtype_bytes


def qwen4_exp_sparse_state(
    config: dict[str, Any], *, dtype_bytes: int
) -> tuple[int, int, int, int]:
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        raise ModelInspectionError("qwen4_exp text_config is missing")
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list):
        raise ModelInspectionError("qwen4_exp layer_types are invalid")
    full_layers = sum(value == "full_attention" for value in layer_types)
    if full_layers <= 0:
        raise ModelInspectionError("qwen4_exp requires full_attention layers")

    def bounded_positive(name: str, maximum: int = 65_536) -> int:
        value = text.get(name)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 1 <= value <= maximum
        ):
            raise ModelInspectionError(f"qwen4_exp {name} is invalid")
        return value

    kv_heads = bounded_positive("num_key_value_heads")
    head_dim = bounded_positive("head_dim")
    indexer_kv_heads = bounded_positive("indexer_kv_heads")
    indexer_heads = bounded_positive("indexer_n_heads")
    indexer_dim = bounded_positive("indexer_head_dim")
    compress_ratio = bounded_positive("indexer_compress_ratio", 1024)
    retrieval_tokens = bounded_positive("indexer_budget", 1_048_576)
    kv_bytes = 2 * full_layers * kv_heads * head_dim * dtype_bytes
    raw_index_bytes = full_layers * indexer_kv_heads * indexer_dim * dtype_bytes
    index_bytes = (raw_index_bytes + compress_ratio - 1) // compress_ratio
    retrieval_bytes = full_layers * indexer_heads * indexer_dim * dtype_bytes
    return kv_bytes, index_bytes, retrieval_bytes, retrieval_tokens


def qwen4_exp_expert_memory(config: dict[str, Any]) -> tuple[int, int]:
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        raise ModelInspectionError("qwen4_exp text_config is missing")

    def bounded_positive(name: str, maximum: int = 1_048_576) -> int:
        value = text.get(name)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 1 <= value <= maximum
        ):
            raise ModelInspectionError(f"qwen4_exp {name} is invalid")
        return value

    layers = bounded_positive("num_hidden_layers", 512)
    hidden = bounded_positive("hidden_size")
    intermediate = bounded_positive("moe_intermediate_size")
    experts = bounded_positive("num_experts", 4096)
    routed = bounded_positive("num_experts_per_tok", experts)
    if routed > experts:
        raise ModelInspectionError("qwen4_exp routed expert count exceeds total experts")
    shared_intermediate = bounded_positive("shared_expert_intermediate_size")
    routed_parameters = 3 * hidden * intermediate
    shared_parameters = 3 * hidden * shared_intermediate
    storage = layers * _qwen_parameter_bytes(
        config, text, experts * routed_parameters + shared_parameters
    )
    working = layers * _qwen_parameter_bytes(
        config, text, routed * routed_parameters + shared_parameters
    )
    return storage, working


def _qwen_parameter_bytes(
    config: dict[str, Any], text: dict[str, Any], parameters: int
) -> int:
    quantization = config.get("quantization_config")
    bits = quantization.get("bits") if isinstance(quantization, dict) else None
    method = quantization.get("quant_method") if isinstance(quantization, dict) else None
    if isinstance(bits, int) and not isinstance(bits, bool) and 1 <= bits <= 16:
        return (parameters * bits + 7) // 8
    dtype_bytes = 1 if method == "fp8" else _kv_dtype_bytes(text)
    return parameters * dtype_bytes


def qwen4_exp_ngram_memory(config: dict[str, Any]) -> tuple[int, int]:
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        raise ModelInspectionError("qwen4_exp text_config is missing")

    def bounded_positive(name: str, maximum: int) -> int:
        value = text.get(name)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 1 <= value <= maximum
        ):
            raise ModelInspectionError(f"qwen4_exp {name} is invalid")
        return value

    vocabulary = bounded_positive("ngram_vocab_size_base", 100_000_000)
    hidden = bounded_positive("hidden_size", 1_048_576)
    bounded_positive("ngram_size", 16)
    bounded_positive("heads_per_ngram", 4096)
    parts = bounded_positive("split_ngram_parts", 4096)
    storage = _qwen_parameter_bytes(config, text, vocabulary * hidden)
    working = (storage + parts - 1) // parts
    return storage, working


def qwen4_exp_hyper_connection_scratch(config: dict[str, Any]) -> int:
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        raise ModelInspectionError("qwen4_exp text_config is missing")
    values = []
    for name, maximum in (("hidden_size", 1_048_576), ("hc_count", 64), ("hc_lowrank", 65_536)):
        value = text.get(name)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 1 <= value <= maximum
        ):
            raise ModelInspectionError(f"qwen4_exp {name} is invalid")
        values.append(value)
    hidden, count, lowrank = values
    return count * (hidden + lowrank) * _kv_dtype_bytes(text)


def qwen4_exp_mtp_memory(config: dict[str, Any]) -> tuple[int, int]:
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        raise ModelInspectionError("qwen4_exp text_config is missing")
    mtp = text.get("mtp")
    layers = text.get("mtp_num_hidden_layers")
    if (
        not isinstance(mtp, dict)
        or mtp.get("hybrid") is not True
        or not isinstance(layers, int)
        or isinstance(layers, bool)
        or not 1 <= layers <= 16
        or mtp.get("num_hidden_layers") != layers
        or mtp.get("layer_types") != ["full_attention"] * layers
    ):
        raise ModelInspectionError("qwen4_exp MTP metadata is invalid")

    def positive(name: str, maximum: int = 1_048_576) -> int:
        value = text.get(name)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 1 <= value <= maximum
        ):
            raise ModelInspectionError(f"qwen4_exp {name} is invalid")
        return value

    hidden = positive("hidden_size")
    query_heads = positive("num_attention_heads")
    kv_heads = positive("num_key_value_heads")
    head_dim = positive("head_dim")
    experts = positive("num_experts", 4096)
    routed = positive("num_experts_per_tok", experts)
    intermediate = positive("moe_intermediate_size")
    shared_intermediate = positive("shared_expert_intermediate_size")
    attention_parameters = 2 * hidden * query_heads * head_dim
    attention_parameters += 2 * hidden * kv_heads * head_dim
    routed_parameters = 3 * hidden * intermediate
    shared_parameters = 3 * hidden * shared_intermediate
    storage_parameters = attention_parameters + experts * routed_parameters + shared_parameters
    working_parameters = attention_parameters + routed * routed_parameters + shared_parameters
    if text.get("mtp_use_dedicated_embeddings") is True:
        embedding_parameters = positive("vocab_size", 100_000_000) * hidden
        storage_parameters += embedding_parameters
        working_parameters += embedding_parameters
    elif text.get("mtp_use_dedicated_embeddings") is not False:
        raise ModelInspectionError("qwen4_exp MTP embedding policy is invalid")
    return (
        layers * _qwen_parameter_bytes(config, text, storage_parameters),
        layers * _qwen_parameter_bytes(config, text, working_parameters),
    )


def _hugging_face_cache_path(model_id: str, cache_root: Path | None = None) -> Path | None:
    if "/" not in model_id or model_id.startswith("/"):
        return None
    root = cache_root or (Path.home() / ".cache" / "huggingface" / "hub")
    repository = root / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if not repository.is_dir():
        return None
    candidates = [path for path in repository.iterdir() if (path / "config.json").is_file()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def resolve_model_path(model_id: str, cache_root: Path | None = None) -> Path | None:
    direct = Path(model_id).expanduser()
    if direct.is_dir():
        return direct.resolve()
    return _hugging_face_cache_path(model_id, cache_root)


def _positive_config_int(config: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = config.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    raise ModelInspectionError(f"missing positive model config value: {' or '.join(keys)}")


def _kv_dtype_bytes(config: dict[str, Any]) -> int:
    dtype = str(config.get("torch_dtype") or config.get("dtype") or "float16").lower()
    if dtype in {"float32", "fp32"}:
        return 4
    if dtype in {"float16", "fp16", "half", "bfloat16", "bf16"}:
        return 2
    if dtype in {"float8", "fp8", "uint8", "int8"}:
        return 1
    # Unknown storage quantization must not be applied to KV. Most MLX language
    # models still use 16-bit runtime KV even when weights are packed to 4/8 bit.
    return 2


def _bounded_positive_config_int(
    config: dict[str, Any], name: str, maximum: int = 16_777_216
) -> int:
    value = config.get(name)
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 1 <= value <= maximum
    ):
        raise ModelInspectionError(f"{name} must be a bounded positive integer")
    return value


def _state_layer_counts(config: dict[str, Any], layers: int) -> tuple[int, int]:
    layer_types = config.get("layer_types")
    if layer_types is None:
        return 0, layers
    if not isinstance(layer_types, list) or len(layer_types) != layers:
        raise ModelInspectionError("hybrid layer_types must match num_hidden_layers")
    attention = 0
    recurrent = 0
    for layer_type in layer_types:
        if not isinstance(layer_type, str):
            raise ModelInspectionError("hybrid layer type must be a string")
        normalized = layer_type.lower()
        if normalized in {"mamba", "ssm", "state_space", "linear_attention"}:
            recurrent += 1
        elif "attention" in normalized:
            attention += 1
        else:
            raise ModelInspectionError(f"unsupported hybrid layer type: {layer_type}")
    return attention, recurrent


def _recurrent_state_bytes(config: dict[str, Any], recurrent_layers: int) -> int:
    if recurrent_layers == 0:
        return 0
    raw_intermediate = config.get("intermediate_size")
    if isinstance(raw_intermediate, int) and not isinstance(raw_intermediate, bool):
        intermediate = _bounded_positive_config_int(config, "intermediate_size")
    else:
        hidden = _bounded_positive_config_int(config, "hidden_size")
        expand = _bounded_positive_config_int(config, "expand", 64)
        intermediate = hidden * expand
        if intermediate > 16_777_216:
            raise ModelInspectionError("derived state-space intermediate size is too large")
    state_size = _bounded_positive_config_int(config, "state_size", 65_536)
    convolution_kernel = _bounded_positive_config_int(config, "conv_kernel", 4_096)
    dtype = str(config.get("ssm_state_dtype") or config.get("mamba_ssm_dtype") or "float32")
    dtype_bytes = {
        "float32": 4,
        "fp32": 4,
        "float16": 2,
        "fp16": 2,
        "bfloat16": 2,
        "bf16": 2,
    }.get(dtype.lower())
    if dtype_bytes is None:
        raise ModelInspectionError("state-space state dtype is unsupported")
    convolution_channels = intermediate
    groups = config.get("n_groups")
    if groups is not None:
        group_count = _bounded_positive_config_int(config, "n_groups", 65_536)
        convolution_channels += 2 * group_count * state_size
    elements_per_layer = intermediate * state_size + convolution_channels * convolution_kernel
    return recurrent_layers * elements_per_layer * dtype_bytes


def _mla_bytes_per_token(config: dict[str, Any], attention_layers: int, dtype_bytes: int) -> int:
    latent_rank = _bounded_positive_config_int(config, "kv_lora_rank", 65_536)
    rope_dimension = _bounded_positive_config_int(config, "qk_rope_head_dim", 65_536)
    return attention_layers * (latent_rank + rope_dimension) * dtype_bytes


def _weight_bytes(path: Path) -> int:
    total = 0
    seen_files: set[tuple[int, int]] = set()
    for candidate in path.rglob("*"):
        if not candidate.is_file() or not candidate.name.endswith(_WEIGHT_SUFFIXES):
            continue
        stat = candidate.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity not in seen_files:
            seen_files.add(identity)
            total += stat.st_size
    if total <= 0:
        raise ModelInspectionError("no supported model weight files found")
    return total


def inspect_model(
    model_id: str | Path,
    cache_root: Path | None = None,
    *,
    backend: str | None = None,
) -> InspectedModel:
    normalized_model_id = str(model_id)
    path = resolve_model_path(normalized_model_id, cache_root)
    if path is None:
        raise ModelInspectionError("model is neither a local directory nor present in HF cache")
    config_path = path / "config.json"
    config = _bounded_config(config_path)
    architecture_capability = inspect_model_architecture(config)
    if backend is not None:
        ensure_architecture_backend_compatible(
            architecture_capability, backend=backend
        )

    shape_config = config.get("text_config", config)
    if not isinstance(shape_config, dict):
        raise ModelInspectionError("model text_config must be an object")
    layers = _positive_config_int(shape_config, "num_hidden_layers", "n_layer")
    dtype_bytes = _kv_dtype_bytes(shape_config)
    max_context = None
    for key in ("max_position_embeddings", "model_max_length", "n_positions", "seq_length"):
        value = shape_config.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            max_context = value
            break

    recurrent_state_bytes = 0
    derived_architecture = architecture_capability.architecture
    has_state_space = "state_size" in shape_config and "conv_kernel" in shape_config and (
        "intermediate_size" in shape_config
        or ("hidden_size" in shape_config and "expand" in shape_config)
    )
    has_mla = "kv_lora_rank" in shape_config or "qk_rope_head_dim" in shape_config
    attention_layers = layers
    recurrent_layers = 0
    if has_state_space:
        attention_layers, recurrent_layers = _state_layer_counts(shape_config, layers)
        recurrent_state_bytes = _recurrent_state_bytes(shape_config, recurrent_layers)
        derived_architecture = "hybrid_state_space" if attention_layers else "state_space"

    if has_mla:
        if "kv_lora_rank" not in shape_config or "qk_rope_head_dim" not in shape_config:
            raise ModelInspectionError("MLA latent cache metadata is incomplete")
        kv_bytes_per_token = _mla_bytes_per_token(
            shape_config, attention_layers, dtype_bytes
        )
        derived_architecture = "hybrid_mla_state_space" if recurrent_layers else "mla"
    elif attention_layers:
        attention_heads = _positive_config_int(shape_config, "num_attention_heads", "n_head")
        kv_heads_value = shape_config.get("num_key_value_heads", attention_heads)
        if (
            not isinstance(kv_heads_value, int)
            or isinstance(kv_heads_value, bool)
            or kv_heads_value <= 0
        ):
            raise ModelInspectionError("num_key_value_heads must be positive")
        head_dim_value = shape_config.get("head_dim")
        if (
            isinstance(head_dim_value, int)
            and not isinstance(head_dim_value, bool)
            and head_dim_value > 0
        ):
            head_dim = head_dim_value
        else:
            hidden_size = _positive_config_int(
                shape_config, "hidden_size", "n_embd", "d_model"
            )
            if hidden_size % attention_heads:
                raise ModelInspectionError("hidden_size is not divisible by num_attention_heads")
            head_dim = hidden_size // attention_heads
        kv_bytes_per_token = 2 * attention_layers * kv_heads_value * head_dim * dtype_bytes
    else:
        kv_bytes_per_token = 0
    sparse_index_bytes = 0
    sparse_retrieval_bytes = 0
    sparse_retrieval_tokens = None
    if architecture_capability.architecture == "qwen4_exp":
        (
            kv_bytes_per_token,
            sparse_index_bytes,
            sparse_retrieval_bytes,
            sparse_retrieval_tokens,
        ) = qwen4_exp_sparse_state(config, dtype_bytes=dtype_bytes)
    if kv_bytes_per_token == 0 and max_context is None:
        raise ModelInspectionError("bounded state-space model requires a context limit")

    spec = ModelMemorySpec(
        model_id=normalized_model_id,
        weights_bytes=_weight_bytes(path),
        kv_bytes_per_token=kv_bytes_per_token,
        model_max_context=max_context,
    )
    state_spec = spec.as_state_memory_spec()
    if recurrent_state_bytes or has_mla:
        state_spec = StateMemorySpec(
            model_id=normalized_model_id,
            architecture=derived_architecture,
            weights_bytes=spec.weights_bytes,
            kv_bytes_per_token=spec.kv_bytes_per_token,
            recurrent_state_bytes=recurrent_state_bytes,
            model_max_context=spec.model_max_context,
        )
    if architecture_capability.architecture == "qwen4_exp":
        expert_storage_bytes, expert_working_set_bytes = qwen4_exp_expert_memory(config)
        ngram_storage_bytes, ngram_working_set_bytes = qwen4_exp_ngram_memory(config)
        mtp_storage_bytes, mtp_working_set_bytes = qwen4_exp_mtp_memory(config)
        state_spec = StateMemorySpec(
            model_id=normalized_model_id,
            architecture="qwen4_exp",
            weights_bytes=max(
                0, spec.weights_bytes - expert_storage_bytes - ngram_storage_bytes
                - mtp_storage_bytes
            ),
            kv_bytes_per_token=spec.kv_bytes_per_token,
            recurrent_state_bytes=qwen4_exp_recurrent_state_bytes(config),
            sparse_index_bytes_per_token=sparse_index_bytes,
            sparse_retrieval_bytes_per_token=sparse_retrieval_bytes,
            sparse_retrieval_tokens=sparse_retrieval_tokens,
            expert_storage_bytes=expert_storage_bytes,
            expert_working_set_bytes=expert_working_set_bytes,
            ngram_storage_bytes=ngram_storage_bytes,
            ngram_working_set_bytes=ngram_working_set_bytes,
            mtp_storage_bytes=mtp_storage_bytes,
            mtp_working_set_bytes=mtp_working_set_bytes,
            scratch_workspace_bytes=qwen4_exp_hyper_connection_scratch(config),
            model_max_context=spec.model_max_context,
        )
    return InspectedModel(
        model_id=normalized_model_id,
        path=path,
        config=config,
        memory_spec=spec,
        kv_dtype_bytes=dtype_bytes,
        architecture_capability=architecture_capability,
        state_memory_spec=state_spec,
    )
