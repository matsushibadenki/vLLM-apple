"""Generate the deterministic x -> 2x Core ML neural-network probe fixture."""
from __future__ import annotations

import argparse
from pathlib import Path

from vllm_apple.ane_reference import (
    MAX_ENCODER_LAYERS,
    MAX_ENCODER_WIDTH,
    encoder_bias,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preset", choices=("scale", "encoder"), default="scale")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    arguments = parser.parse_args()
    output = arguments.output.expanduser().absolute()
    if output.suffix != ".mlmodel" or output.exists():
        raise SystemExit("output must be a new .mlmodel path")
    output.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from coremltools.models import datatypes
    from coremltools.models.neural_network import NeuralNetworkBuilder
    from coremltools.models.utils import save_spec

    if arguments.preset == "scale":
        features = [("input", datatypes.Array(4))]
        builder = NeuralNetworkBuilder(features, [("output", datatypes.Array(4))])
        builder.add_scale(
            name="double", W=np.array([2.0], dtype=np.float32), b=None,
            has_bias=False, input_name="input", output_name="output", shape_scale=[1],
        )
        description = "vLLM-Apple deterministic ANE correctness probe: output = input * 2"
    else:
        if not 1 <= arguments.width <= MAX_ENCODER_WIDTH or not 1 <= arguments.layers <= MAX_ENCODER_LAYERS:
            raise SystemExit("encoder shape exceeds its bound")
        features = [("input", datatypes.Array(arguments.width))]
        builder = NeuralNetworkBuilder(
            features, [("output", datatypes.Array(arguments.width))]
        )
        previous = "input"
        rows = np.arange(arguments.width, dtype=np.int64)[:, None]
        columns = np.arange(arguments.width, dtype=np.int64)[None, :]
        for layer in range(arguments.layers):
            dense = f"dense_{layer}"
            activated = "output" if layer == arguments.layers - 1 else f"relu_{layer}"
            weights = (
                (((layer + 1) * 7 + rows * 17 + columns * 13) % 29 - 14)
                / 16384.0
            ).astype(np.float32)
            bias = np.array(
                [encoder_bias(layer, row) for row in range(arguments.width)],
                dtype=np.float32,
            )
            builder.add_inner_product(
                name=dense, W=weights, b=bias,
                input_channels=arguments.width, output_channels=arguments.width,
                has_bias=True, input_name=previous, output_name=dense,
            )
            builder.add_activation(
                name=f"activation_{layer}", non_linearity="RELU",
                input_name=dense, output_name=activated,
            )
            previous = activated
        description = (
            f"vLLM-Apple deterministic representative encoder: "
            f"width={arguments.width}, layers={arguments.layers}"
        )
    builder.spec.description.metadata.shortDescription = description
    save_spec(builder.spec, str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
