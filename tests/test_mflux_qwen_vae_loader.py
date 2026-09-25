import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from vllm_apple.mflux_qwen_vae_loader import inspect_mflux_qwen_vae_decoder


class MFluxQwenVAELoaderTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("numpy") and importlib.util.find_spec("safetensors"),
        "NumPy and safetensors are required",
    )
    def test_decoder_inventory_and_binding(self) -> None:
        import numpy as np
        from safetensors.numpy import save_file

        with TemporaryDirectory() as directory:
            component = Path(directory) / "vae"
            component.mkdir()
            arrays = {
                "decoder.conv_in.conv3d.weight": np.ones((2, 2), dtype=np.float16),
                "post_quant_conv.conv3d.bias": np.ones((2,), dtype=np.float16),
                "encoder.conv_in.conv3d.weight": np.ones((2, 2), dtype=np.float16),
            }
            save_file(arrays, component / "0.safetensors")
            index = component / "model.safetensors.index.json"
            index.write_text(json.dumps({
                "metadata": {"quantization_level": "4"},
                "weight_map": {name: "0.safetensors" for name in arrays},
            }))
            with self.assertRaisesRegex(ValueError, "descriptor is invalid"):
                inspect_mflux_qwen_vae_decoder(Path(directory))
            payload = json.loads(index.read_text())
            payload["weight_map"]["decoder.conv_in.conv3d.weight"] = "../unsafe.safetensors"
            index.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "shard binding"):
                inspect_mflux_qwen_vae_decoder(Path(directory))

    def test_missing_index_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            original_import = __import__

            def import_without_safetensors(name, *args, **kwargs):
                if name == "safetensors":
                    raise ModuleNotFoundError(name)
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=import_without_safetensors):
                with self.assertRaises(FileNotFoundError):
                    inspect_mflux_qwen_vae_decoder(Path(directory))
