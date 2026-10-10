---
license: apache-2.0
tags:
- mlx
- qwen
- image-generation
- text-to-image
- apple-silicon
- diffusion
base_model: Qwen/Qwen-Image-2512
library_name: mflux
---

# Qwen-Image-2512-4bit-MLX

MLX-optimized 4-bit quantized version of [Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) for Apple Silicon.

## Quick Start

```bash
pip install mflux

mflux-generate-qwen \
  --model mlx-community/Qwen-Image-2512-4bit \
  --prompt "A photorealistic cat wearing a tiny top hat" \
  --steps 20
```

## Model Details

- **Base Model:** Qwen/Qwen-Image-2512 (Dec 31, 2025)
- **Quantization:** 4-bit
- **Framework:** MLX (Apple Silicon optimized)
- **Converted with:** mflux 0.14.0

## Other Quantizations

- [8-bit](https://huggingface.co/mlx-community/Qwen-Image-2512-8bit) (34GB)
- [6-bit](https://huggingface.co/mlx-community/Qwen-Image-2512-6bit) (29GB)
- [5-bit](https://huggingface.co/mlx-community/Qwen-Image-2512-5bit) (27GB)
- [4-bit](https://huggingface.co/mlx-community/Qwen-Image-2512-4bit) (24GB)
- [3-bit](https://huggingface.co/mlx-community/Qwen-Image-2512-3bit) (22GB)

## License

Apache 2.0 (same as base model)
