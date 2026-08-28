from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path

VLLM_METAL_INTEGRATION_SCHEMA_VERSION = 1
MAX_INTEGRATION_SOURCE_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class VLLMMetalIntegrationInspection:
    schema_version: int
    source_fingerprint: str
    topology: str
    native_v2_detected: bool
    python_callsite_hook: bool
    dynamic_thread_configuration_abi: bool
    compatible: bool
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_fingerprint": self.source_fingerprint,
            "topology": self.topology,
            "native_v2_detected": self.native_v2_detected,
            "python_callsite_hook": self.python_callsite_hook,
            "dynamic_thread_configuration_abi": self.dynamic_thread_configuration_abi,
            "compatible": self.compatible,
            "issues": list(self.issues),
        }


def inspect_vllm_metal_integration(source_root: Path) -> VLLMMetalIntegrationInspection:
    package_root = _resolve_package_root(source_root)
    python_source = _read_bounded_source(package_root / "attention" / "impls" / "sdpa.py")
    cpp_source = _read_bounded_source(package_root / "metal" / "paged_ops.cpp")

    native_v2 = (
        all(
            marker in cpp_source
            for marker in (
                "dispatch_paged_attention_v2_online",
                "paged_attention_primitive",
                "dispatch_paged_attention_tiled",
            )
        )
        and "ops.paged_attention_primitive" in python_source
    )
    python_hook = all(
        marker in python_source
        for marker in (
            "vllm_apple.vllm_middleware",
            "invoke_paged_attention_kernel",
        )
    )
    dynamic_abi = all(
        marker in cpp_source
        for marker in (
            "VLLM_APPLE_THREAD_CONFIG_ABI_V1",
            "score_width",
            "softmax_width",
            "output_width",
        )
    )
    issues: list[str] = []
    if not native_v2:
        issues.append("native_v2_topology_not_detected")
    if not python_hook:
        issues.append("python_callsite_hook_missing")
    if not dynamic_abi:
        issues.append("dynamic_thread_configuration_abi_missing")
    fingerprint = hashlib.sha256((python_source + "\0" + cpp_source).encode("utf-8")).hexdigest()[
        :24
    ]
    return VLLMMetalIntegrationInspection(
        VLLM_METAL_INTEGRATION_SCHEMA_VERSION,
        fingerprint,
        "native_v2" if native_v2 else "unknown",
        native_v2,
        python_hook,
        dynamic_abi,
        not issues,
        tuple(issues),
    )


def _resolve_package_root(source_root: Path) -> Path:
    root = source_root.expanduser().resolve()
    candidates = (root, root / "vllm_metal")
    for candidate in candidates:
        if (candidate / "attention" / "impls" / "sdpa.py").is_file() and (
            candidate / "metal" / "paged_ops.cpp"
        ).is_file():
            return candidate
    raise ValueError("vLLM-Metal source root does not contain the native v2 call sites")


def _read_bounded_source(path: Path) -> str:
    attributes = path.lstat()
    if (
        not stat.S_ISREG(attributes.st_mode)
        or attributes.st_uid != os.getuid()
        or not 1 <= attributes.st_size <= MAX_INTEGRATION_SOURCE_BYTES
    ):
        raise ValueError("integration source must be a bounded current-user regular file")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("integration source must be UTF-8") from error
