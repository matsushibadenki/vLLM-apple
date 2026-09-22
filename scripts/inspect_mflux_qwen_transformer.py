#!/usr/bin/env python3
"""Record a load-free transformer block staging inventory for Qwen-Image-2512."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.inspect_mflux_qwen_streaming import _deployable_tree_identity  # noqa: E402
from vllm_apple.mflux_qwen_transformer_plan import (  # noqa: E402
    inspect_mflux_qwen_transformer_staging,
)
from vllm_apple.qualification import save_qualification_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.model.expanduser().resolve(strict=True)
    plan = inspect_mflux_qwen_transformer_staging(root)
    root_digest, file_count, artifact_bytes = _deployable_tree_identity(root)
    report = {
        "schema_version": 1,
        "scope": "mflux_qwen_image_transformer_load_free_staging_inventory",
        "candidate_id": "qwen-image-2512",
        "artifact_root_sha256": root_digest,
        "artifact_file_count": file_count,
        "artifact_bytes": artifact_bytes,
        "plan": plan.to_dict(),
        "weight_load_started": False,
        "memory_fit_qualified": False,
        "transformer_forward_qualified": False,
        "image_generation_qualified": False,
        "stores_prompt": False,
        "stores_output": False,
        "passed": True,
    }
    save_qualification_report(report, arguments.report)
    print(json.dumps({
        "passed": True,
        "block_count": plan.block_count,
        "tensor_count": plan.tensor_count,
        "quantized_weight_tensor_count": plan.quantized_weight_tensor_count,
        "static_plus_maximum_block_bytes": plan.static_plus_maximum_block_bytes,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
