from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

from ..hardware import detect_hardware
from ..model import inspect_model
from .planner import build_dry_run_plan
from .types import OptimizationObjective, ResourceBudget

MAX_OPTIMIZER_PAYLOAD_BYTES = 1_048_576


class OptimizerDaemonController:
    """Side-effect-free optimizer control exposed to a sandboxed UI."""

    def __init__(self, model_roots: tuple[Path, ...], output_roots: tuple[Path, ...]) -> None:
        self._model_roots = self._validate_roots(model_roots, writable=False)
        self._output_roots = self._validate_roots(output_roots, writable=True)

    @staticmethod
    def _validate_roots(roots: tuple[Path, ...], *, writable: bool) -> tuple[Path, ...]:
        if not roots or len(roots) > 16:
            raise ValueError("optimizer daemon requires 1 to 16 allowed roots")
        result = []
        for root in roots:
            info = os.lstat(root)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise ValueError("optimizer daemon roots must be real directories")
            if info.st_uid != os.getuid():
                raise ValueError("optimizer daemon roots must be owned by the current user")
            if writable and not os.access(root, os.W_OK):
                raise ValueError("optimizer output roots must be writable")
            result.append(root.resolve(strict=True))
        return tuple(result)

    @staticmethod
    def _within(path: Path, roots: tuple[Path, ...]) -> bool:
        return any(path == root or root in path.parents for root in roots)

    def handle(self, operation: str, payload: bytes) -> bytes:
        if operation != "plan":
            raise ValueError("optimizer daemon operation is not enabled")
        if len(payload) > MAX_OPTIMIZER_PAYLOAD_BYTES:
            raise ValueError("optimizer daemon payload is too large")
        try:
            request = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("optimizer daemon payload must be JSON") from error
        if not isinstance(request, dict) or set(request) != {
            "model_path",
            "output_path",
            "objective",
            "maximum_memory_bytes",
            "maximum_disk_bytes",
            "maximum_duration_seconds",
            "license",
        }:
            raise ValueError("invalid optimizer plan payload")
        model_path = Path(self._bounded_string(request["model_path"], "model_path")).resolve(
            strict=True
        )
        output_path = Path(
            self._bounded_string(request["output_path"], "output_path")
        ).resolve(strict=False)
        if not self._within(model_path, self._model_roots):
            raise ValueError("model path is outside allowed roots")
        if not self._within(output_path, self._output_roots):
            raise ValueError("output path is outside allowed roots")
        memory = self._positive_integer(request["maximum_memory_bytes"], "memory")
        disk = self._positive_integer(request["maximum_disk_bytes"], "disk")
        duration_value = request["maximum_duration_seconds"]
        duration = (
            None
            if duration_value is None
            else self._positive_integer(duration_value, "duration")
        )
        license_value = request["license"]
        if license_value is not None:
            license_value = self._bounded_string(license_value, "license", maximum=256)
        plan = build_dry_run_plan(
            inspect_model(model_path),
            detect_hardware(),
            output_path,
            OptimizationObjective(request["objective"]),
            ResourceBudget(memory, disk, duration),
            license_name=license_value,
        )
        encoded = json.dumps(plan.to_dict(), separators=(",", ":"), sort_keys=True).encode()
        if len(encoded) > MAX_OPTIMIZER_PAYLOAD_BYTES:
            raise ValueError("optimizer daemon response payload is too large")
        return encoded

    @staticmethod
    def _bounded_string(value: Any, name: str, *, maximum: int = 4096) -> str:
        if not isinstance(value, str) or not value or len(value.encode()) > maximum:
            raise ValueError(f"invalid optimizer {name}")
        return value

    @staticmethod
    def _positive_integer(value: Any, name: str) -> int:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            or value > 2**63 - 1
        ):
            raise ValueError(f"invalid optimizer {name} budget")
        return value
