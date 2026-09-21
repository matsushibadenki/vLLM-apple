#!/usr/bin/env python3
"""Build and qualify the five-artifact Qwen3-VL Core ML runtime profile."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm_apple.qwen3_vl_ane import inspect_qwen3_vl_vision_for_ane  # noqa: E402
from vllm_apple.qwen3_vl_conversion_plan import (  # noqa: E402
    build_qwen3_vl_coreml_conversion_plan,
)
from vllm_apple.qwen3_vl_conversion_worker import (  # noqa: E402
    stage_qwen3_vl_coreml_weights,
)
from vllm_apple.qwen3_vl_deepstack_coreml import (  # noqa: E402
    build_qwen3_vl_deepstack_coreml,
    qualify_qwen3_vl_deepstack_coreml,
)
from vllm_apple.qwen3_vl_patch_coreml import (  # noqa: E402
    build_qwen3_vl_patch_coreml,
    qualify_qwen3_vl_patch_coreml,
)


def build_profile(model: Path, revision: str, output: Path) -> dict[str, object]:
    """Atomically publish a fully qualified 1x16x16 FP16 profile."""
    model = model.expanduser().resolve(strict=True)
    destination = output.expanduser().absolute()
    if destination.exists() or destination.is_symlink() or not destination.parent.is_dir():
        raise ValueError("Qwen3-VL profile destination must be new")
    source = inspect_qwen3_vl_vision_for_ane(model, model_revision=revision)
    plan = build_qwen3_vl_coreml_conversion_plan(
        model, source, fixed_grid_profiles=((1, 16, 16),)
    )
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    os.chmod(temporary, 0o700)
    try:
        staged = temporary / "staged-weights"
        stage_qwen3_vl_coreml_weights(
            model,
            staged,
            source,
            plan,
            maximum_output_bytes=plan.target_tensor_bytes,
        )
        patch = temporary / "patch"
        build_qwen3_vl_patch_coreml(staged, patch, source, plan)
        qualifications = [qualify_qwen3_vl_patch_coreml(patch)]
        partitions = []
        for index, start_layer in enumerate((0, 6, 12)):
            partition = temporary / f"segment-{start_layer}-{start_layer + 5}"
            build_qwen3_vl_deepstack_coreml(
                staged,
                partition,
                source,
                plan,
                deepstack_position=index,
                start_layer=start_layer,
            )
            qualifications.append(qualify_qwen3_vl_deepstack_coreml(partition))
            partitions.append(partition)
        final = temporary / "segment-18-23"
        build_qwen3_vl_deepstack_coreml(
            staged,
            final,
            source,
            plan,
            start_layer=18,
            final_merger=True,
        )
        qualifications.append(qualify_qwen3_vl_deepstack_coreml(final))
        partitions.append(final)
        reports = [json.loads((root / "report.json").read_bytes()) for root in (patch, *partitions)]
        graph_ids = {report["graph_id"] for report in reports}
        if len(graph_ids) != 1 or not all(item.get("passed") is True for item in qualifications):
            raise RuntimeError("Qwen3-VL Core ML profile qualification failed")
        shutil.rmtree(staged)
        manifest = {
            "schema_version": 1,
            "scope": "qwen3_vl_coreml_runtime_profile",
            "model_revision": revision,
            "source_artifact_fingerprint": source.artifact_fingerprint,
            "conversion_plan_id": plan.plan_id,
            "graph_id": next(iter(graph_ids)),
            "grid_thw": [1, 16, 16],
            "precision": "fp16",
            "compiled_models": [
                str(root.relative_to(temporary) / report["compiled_model"])
                for root, report in zip((patch, *partitions), reports, strict=True)
            ],
            "partitions": [report["partition"] for report in reports],
            "qualifications": qualifications,
            "passed": True,
        }
        manifest_path = temporary / "profile.json"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        os.chmod(manifest_path, 0o600)
        os.replace(temporary, destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(build_profile(arguments.model, arguments.revision, arguments.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
