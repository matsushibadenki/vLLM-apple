"""Generate the deterministic x -> 2x Core ML neural-network probe fixture."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    output = arguments.output.expanduser().absolute()
    if output.suffix != ".mlmodel" or output.exists():
        raise SystemExit("output must be a new .mlmodel path")
    output.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from coremltools.models import datatypes
    from coremltools.models.neural_network import NeuralNetworkBuilder
    from coremltools.models.utils import save_spec

    features = [("input", datatypes.Array(4))]
    builder = NeuralNetworkBuilder(features, [("output", datatypes.Array(4))])
    builder.add_scale(
        name="double",
        W=np.array([2.0], dtype=np.float32),
        b=None,
        has_bias=False,
        input_name="input",
        output_name="output",
        shape_scale=[1],
    )
    builder.spec.description.metadata.shortDescription = (
        "vLLM-Apple deterministic ANE correctness probe: output = input * 2"
    )
    save_spec(builder.spec, str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
