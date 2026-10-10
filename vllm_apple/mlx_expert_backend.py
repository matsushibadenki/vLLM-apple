"""Opt-in dense or affine-quantized SwiGLU experts from independent files.

This does not wrap a resident SwitchGLU bank: evicting slices of a retained bank
would not release the original weights. No model hooks are installed here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .expert_execution import ResidentExpertExecutor
from .expert_manifest import ExpertManifest
from .expert_residency import ExpertKey, ExpertResource
from .expert_selection_telemetry import ExpertSelectionSample
from .expert_timing import measure

_NAMES = frozenset({"gate_proj.weight", "up_proj.weight", "down_proj.weight"})
_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_QUANTIZED_NAMES = frozenset(
    f"{projection}.{part}" for projection in _PROJECTIONS
    for part in ("weight", "scales", "biases")
)


@dataclass
class _Weights:
    tensors: dict


class MLXFileExpertBackend:
    """Each file contains one bias-free dense or affine-quantized SwiGLU expert.

    Files are named layer-L-expert-E.safetensors. The root must be prepared
    offline; original model shards are never retained or modified by this adapter.
    """

    def __init__(self, root: Path, *, maximum_file_bytes: int,
                 quantization_bits: int | None = None, group_size: int = 64, timings=None):
        if type(maximum_file_bytes) is not int or maximum_file_bytes <= 0:
            raise ValueError("invalid expert file limit")
        if (quantization_bits is not None
                and (type(quantization_bits) is not int or quantization_bits not in (4, 8))):
            raise ValueError("expected affine 4-bit or 8-bit quantization")
        if type(group_size) is not int or group_size not in (32, 64, 128):
            raise ValueError("invalid quantization group size")
        self.quantization_bits = quantization_bits
        self.group_size = group_size
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("expert root must be a directory")
        self.maximum_file_bytes = maximum_file_bytes
        self._manifest = None
        self.timings = timings

    @classmethod
    def from_manifest(cls, root: Path, *, expected_model_sha256: str,
                      maximum_file_bytes: int, timings=None):
        manifest = ExpertManifest.load(
            Path(root) / "expert-manifest.json", expected_model_sha256=expected_model_sha256,
        )
        backend = cls(root, maximum_file_bytes=maximum_file_bytes,
                      quantization_bits=manifest.quantization_bits, group_size=manifest.group_size,
                      timings=timings)
        backend._manifest = manifest
        return backend

    def load_expert(self, key: ExpertKey) -> ExpertResource:
        if not isinstance(key, ExpertKey):
            raise ValueError("invalid expert key")
        path = self.root / f"layer-{key.layer}-expert-{key.expert}.safetensors"
        with measure(self.timings, "file_stat"):
            if path.is_symlink() or not path.is_file():
                raise ValueError("expert file must be a regular non-symlink file")
            if not 0 < path.stat().st_size <= self.maximum_file_bytes:
                raise ValueError("expert file exceeds limit")
        if self._manifest is not None:
            if (self.quantization_bits != self._manifest.quantization_bits
                    or self.group_size != self._manifest.group_size):
                raise ValueError("expert settings differ from manifest")
            self._manifest.verify_file(key, path, timings=self.timings)
        import mlx.core as mx

        with measure(self.timings, "mlx_load"):
            tensors = mx.load(str(path))
        if self.quantization_bits is not None:
            self._validate_quantized(tensors, mx)
            with measure(self.timings, "weight_eval"):
                mx.eval(*tensors.values())
            return ExpertResource(_Weights(tensors), sum(t.nbytes for t in tensors.values()))
        if not isinstance(tensors, dict) or set(tensors) != _NAMES:
            raise ValueError("expected three dense SwiGLU weights")
        gate, up, down = (tensors[name] for name in (
            "gate_proj.weight", "up_proj.weight", "down_proj.weight"
        ))
        if (any(len(t.shape) != 2 or min(t.shape) <= 0
                or t.dtype not in (mx.float32, mx.float16, mx.bfloat16)
                for t in (gate, up, down))
                or gate.shape != up.shape or down.shape != (gate.shape[1], gate.shape[0])):
            raise ValueError("incompatible dense SwiGLU weights")
        with measure(self.timings, "weight_eval"):
            mx.eval(gate, up, down)
        return ExpertResource(_Weights(tensors), sum(t.nbytes for t in tensors.values()))

    def _validate_quantized(self, tensors, mx):
        if not isinstance(tensors, dict) or set(tensors) != _QUANTIZED_NAMES:
            raise ValueError("expected affine packed weights, scales and biases")
        dimensions = []
        for projection in _PROJECTIONS:
            weight, scales, biases = (tensors[f"{projection}.{part}"]
                                      for part in ("weight", "scales", "biases"))
            if (any(len(t.shape) != 2 or min(t.shape) <= 0
                    for t in (weight, scales, biases))
                    or weight.dtype != mx.uint32
                    or scales.dtype not in (mx.float32, mx.float16, mx.bfloat16)
                    or biases.dtype != scales.dtype or scales.shape != biases.shape
                    or weight.shape[0] != scales.shape[0]
                    or weight.shape[1] * (32 // self.quantization_bits)
                    != scales.shape[1] * self.group_size):
                raise ValueError("incompatible affine quantized weights")
            dimensions.append((weight.shape[0], scales.shape[1] * self.group_size))
        gate, up, down = dimensions
        if gate != up or down != (gate[1], gate[0]):
            raise ValueError("incompatible quantized SwiGLU dimensions")

    def _linear(self, x, tensors, projection, mx):
        if self.quantization_bits is None:
            return x @ tensors[f"{projection}.weight"].T
        return mx.quantized_matmul(
            x, tensors[f"{projection}.weight"], tensors[f"{projection}.scales"],
            tensors[f"{projection}.biases"], transpose=True,
            group_size=self.group_size, bits=self.quantization_bits, mode="affine",
        )

    def release_expert(self, resource: ExpertResource) -> None:
        if not isinstance(resource.handle, _Weights):
            raise ValueError("foreign expert resource")
        resource.handle.tensors.clear()

    def execute_selected(
        self, executor: ResidentExpertExecutor, x, *, phase: str, layer: int,
        selected_experts: tuple[int, ...], routing_weights: tuple[float, ...],
    ):
        """Evaluate one routed token group sharing the same router selection.

        Batch rows with different selections must be grouped by the caller. This
        initial reference path is not a fused production MoE kernel.
        """
        if not executor.manager.uses_backend(self):
            raise ValueError("executor uses a different expert backend")
        import mlx.core as mx

        if (not isinstance(x, mx.array) or len(x.shape) != 2 or min(x.shape) <= 0
                or x.dtype not in (mx.float32, mx.float16, mx.bfloat16)):
            raise ValueError("expert input must be a nonempty matrix")

        def consume(resources, weights):
            output = None
            for resource, weight in zip(resources, weights):
                tensors = resource.handle.tensors
                input_dims = (tensors["gate_proj.weight"].shape[-1]
                              if self.quantization_bits is None else
                              tensors["gate_proj.scales"].shape[-1] * self.group_size)
                if x.shape[-1] != input_dims:
                    raise ValueError("expert input dimension mismatch")
                with measure(self.timings, "expert_build"):
                    g = self._linear(x, tensors, "gate_proj", mx)
                    u = self._linear(x, tensors, "up_proj", mx)
                    value = self._linear((g * mx.sigmoid(g)) * u, tensors, "down_proj", mx)
                    value = value * weight
                    output = value if output is None else output + value
            with measure(self.timings, "expert_eval"):
                mx.eval(output)
            return output

        return executor.execute(
            phase=phase, layer=layer, selected_experts=selected_experts,
            routing_weights=routing_weights, consume=consume,
        )

    def execute_routed(
        self, executor: ResidentExpertExecutor, x, *, phase: str, layer: int,
        selections: tuple[tuple[int, ...], ...],
        weights: tuple[tuple[float, ...], ...],
    ):
        """Reference path for authoritative per-token selections, in row order.

        The caller supplies router decisions; no top-k, normalization or prediction
        is performed. Sequential rows bound simultaneous residency to one top-k
        group. This intentionally favors correctness over throughput.
        """
        import mlx.core as mx

        if (not isinstance(x, mx.array) or len(x.shape) != 2 or min(x.shape) <= 0
                or x.dtype not in (mx.float32, mx.float16, mx.bfloat16)
                or not isinstance(selections, tuple) or not isinstance(weights, tuple)
                or len(selections) != x.shape[0] or len(weights) != x.shape[0]
                or x.shape[0] > 4096 or phase not in {"prefill", "decode"}
                or not executor.manager.uses_backend(self)):
            raise ValueError("invalid routed expert batch")
        for selected, routing in zip(selections, weights):
            if not isinstance(selected, tuple) or not isinstance(routing, tuple):
                raise ValueError("router rows must be immutable tuples")
            ExpertSelectionSample(layer, selected, routing, 1, ())
        results = [
            self.execute_selected(
                executor, x[row:row + 1], phase=phase, layer=layer,
                selected_experts=selected, routing_weights=routing,
            )
            for row, (selected, routing) in enumerate(zip(selections, weights))
        ]
        result = mx.concatenate(results, axis=0)
        mx.eval(result)
        return result

    def execute_unaggregated(
        self, executor: ResidentExpertExecutor, x, *, phase: str, layer: int,
        selected_experts: tuple[int, ...],
    ):
        """Pin a complete router group and materialize its outputs once."""
        import mlx.core as mx

        if (not executor.manager.uses_backend(self) or not isinstance(x, mx.array)
                or len(x.shape) != 2 or min(x.shape) <= 0
                or x.dtype not in (mx.float32, mx.float16, mx.bfloat16)
                or x.nbytes * len(selected_experts) > 64 * 1024 * 1024):
            raise ValueError("invalid grouped expert input")

        def consume(resources, weights):
            outputs = []
            for resource in resources:
                tensors = resource.handle.tensors
                input_dims = (tensors["gate_proj.weight"].shape[-1]
                              if self.quantization_bits is None else
                              tensors["gate_proj.scales"].shape[-1] * self.group_size)
                if x.shape[-1] != input_dims:
                    raise ValueError("expert input dimension mismatch")
                with measure(self.timings, "expert_build"):
                    g = self._linear(x, tensors, "gate_proj", mx)
                    u = self._linear(x, tensors, "up_proj", mx)
                    outputs.append(self._linear((g * mx.sigmoid(g)) * u, tensors, "down_proj", mx))
            with measure(self.timings, "output_assembly"):
                output = mx.stack(outputs, axis=-2)
            with measure(self.timings, "expert_eval"):
                mx.eval(output)
            return output

        return executor.execute(
            phase=phase, layer=layer, selected_experts=selected_experts,
            routing_weights=(1.0,) * len(selected_experts), consume=consume,
        )
