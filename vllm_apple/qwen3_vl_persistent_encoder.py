"""Verified persistent Core ML worker bridge into the Qwen3-VL MLX transport."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

from .device_pipeline import ANEAuxiliaryRoute
from .qwen3_vl_ane import Qwen3VLVisionANEAdapterSpec
from .qwen3_vl_embedding import Qwen3VLCoreMLPipelineOutput
from .qwen3_vl_persistent_worker import Qwen3VLPersistentWorker
from .qwen3_vl_transport import load_qwen3_vl_coreml_transport


_OUTPUT_NAMES = ("deepstack_0", "deepstack_1", "deepstack_2", "final")
_OUTPUT_BYTES = 64 * 2048 * 2


def publish_qwen3_vl_persistent_transport_manifest(
    transport_root: Path,
    worker_result: dict[str, object],
    *,
    graph_id: str,
) -> dict[str, object]:
    """Bind worker-reported digests to the private payload files atomically."""
    root = transport_root.expanduser().resolve(strict=True)
    digests = worker_result.get("output_digests")
    if (
        len(graph_id) != 64
        or any(character not in "0123456789abcdef" for character in graph_id)
        or not root.is_dir()
        or stat.S_IMODE(root.stat().st_mode) & 0o077
        or not isinstance(digests, list)
        or len(digests) != 4
        or any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in digests
        )
        or {path.name for path in root.iterdir()}
        != {f"{name}.fp16" for name in _OUTPUT_NAMES}
    ):
        raise ValueError("persistent Qwen3-VL transport identity is invalid")
    records = []
    for name, expected_digest in zip(_OUTPUT_NAMES, digests, strict=True):
        path = root / f"{name}.fp16"
        if (
            path.is_symlink()
            or not path.is_file()
            or stat.S_IMODE(path.stat().st_mode) & 0o077
            or path.stat().st_size != _OUTPUT_BYTES
        ):
            raise ValueError("persistent Qwen3-VL payload is invalid")
        actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_digest != expected_digest:
            raise ValueError("persistent Qwen3-VL payload digest changed")
        records.append(
            {
                "name": name,
                "file": path.name,
                "shape": [64, 2048],
                "dtype": "float16",
                "bytes": _OUTPUT_BYTES,
                "sha256": actual_digest,
            }
        )
    manifest = {
        "schema_version": 1,
        "graph_id": graph_id,
        "grid_thw": [1, 16, 16],
        "records": records,
    }
    encoded = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    descriptor = os.open(
        root / "manifest.json",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        (root / "manifest.json").unlink(missing_ok=True)
        raise
    return manifest


class Qwen3VLPersistentEncoder:
    """Produce a provenance-bound MLX embedding bundle from one resident worker."""

    def __init__(
        self,
        worker: Qwen3VLPersistentWorker,
        source: Qwen3VLVisionANEAdapterSpec,
        route: ANEAuxiliaryRoute,
        *,
        graph_id: str,
    ) -> None:
        if (
            not isinstance(worker, Qwen3VLPersistentWorker)
            or not isinstance(source, Qwen3VLVisionANEAdapterSpec)
            or not isinstance(route, ANEAuxiliaryRoute)
            or route.operator != source.operator
            or len(graph_id) != 64
            or any(character not in "0123456789abcdef" for character in graph_id)
        ):
            raise ValueError("persistent Qwen3-VL encoder identity is invalid")
        self._worker = worker
        self._source = source
        self._route = route
        self._graph_id = graph_id

    def encode(
        self, pixel_values_file: Path, transport_root: Path
    ) -> Qwen3VLCoreMLPipelineOutput:
        result = self._worker.predict(pixel_values_file, transport_root)
        publish_qwen3_vl_persistent_transport_manifest(
            transport_root, result, graph_id=self._graph_id
        )
        return load_qwen3_vl_coreml_transport(
            transport_root,
            self._source,
            self._route,
            expected_graph_id=self._graph_id,
        )

    def close(self) -> bool:
        return self._worker.close()
