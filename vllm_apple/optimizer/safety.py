from __future__ import annotations

from pathlib import Path


class OptimizationPathError(ValueError):
    pass


def validate_immutable_output_path(source: Path, output: Path) -> Path:
    source_resolved = source.expanduser().resolve(strict=True)
    if not source_resolved.is_dir():
        raise OptimizationPathError("source model must be a directory")
    output_resolved = output.expanduser().resolve(strict=False)
    if output_resolved.exists():
        raise OptimizationPathError("immutable artifact output already exists")
    if output_resolved == Path(output_resolved.anchor):
        raise OptimizationPathError("filesystem root cannot be an artifact output")
    if _contains(source_resolved, output_resolved) or _contains(output_resolved, source_resolved):
        raise OptimizationPathError("source and output paths must not overlap")
    return output_resolved


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True
