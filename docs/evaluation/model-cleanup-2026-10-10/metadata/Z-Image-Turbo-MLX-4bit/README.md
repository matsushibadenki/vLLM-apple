---
library_name: mlx
license: apache-2.0
base_model: Tongyi-MAI/Z-Image-Turbo
tags:
  - mlx
  - diffusers
  - safetensors
  - text-to-image
  - apple-silicon
  - image-generation
pipeline_tag: text-to-image
language:
  - en
  - zh
---

# Z-Image-Turbo — MLX (4-bit Quantized)

> MLX conversion of [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) for Apple Silicon.

This is the **4-bit quantized** MLX conversion. Linear layer weights are quantized to 4-bit with group_size=64. VAE remains in float16 to preserve image quality.

**Model size: 6.48 GB**

## All Available MLX Variants

| Variant | Size | Quantization | Link |
|---------|------|-------------|------|
| **Full Precision (fp16)** | 20.54 GB | None | [andrevp/Z-Image-Turbo-MLX](https://huggingface.co/andrevp/Z-Image-Turbo-MLX) |
| **8-bit** | 11.37 GB | 8-bit, group_size=64 | [andrevp/Z-Image-Turbo-MLX-8bit](https://huggingface.co/andrevp/Z-Image-Turbo-MLX-8bit) |
| **4-bit** | 6.48 GB | 4-bit, group_size=64 | [andrevp/Z-Image-Turbo-MLX-4bit](https://huggingface.co/andrevp/Z-Image-Turbo-MLX-4bit) |
| **2-bit** | 4.04 GB | 2-bit, group_size=64 | [andrevp/Z-Image-Turbo-MLX-2bit](https://huggingface.co/andrevp/Z-Image-Turbo-MLX-2bit) |

## About Z-Image-Turbo

Z-Image is an efficient 6B-parameter image generation foundation model using a **Scalable Single-Stream Diffusion Transformer (S3-DiT)** architecture. Z-Image-Turbo is the distilled variant with only **8 NFEs** (Number of Function Evaluations), achieving sub-second inference latency.

### Key Features

- **Photorealistic image generation** with state-of-the-art quality
- **Bilingual text rendering** (English & Chinese)
- **Strong instruction adherence**
- **8-step inference** — distilled via Decoupled-DMD + Reinforcement Learning (DMDR)
- **No CFG required** — guidance_scale=0.0

## Architecture

| Component | Architecture | Parameters |
|-----------|-------------|------------|
| **Text Encoder** | Qwen3 (36 layers, hidden_size=2560, GQA with 32/8 heads) | ~7.8 GB (fp16) |
| **Transformer** | ZImageTransformer2DModel (30 layers, dim=3840, 30 heads) | ~12.3 GB (fp16) |
| **VAE** | AutoencoderKL (from Flux, 16 latent channels) | ~160 MB (fp16) |
| **Tokenizer** | Qwen2Tokenizer (vocab_size=151,936) | — |
| **Scheduler** | FlowMatchEulerDiscreteScheduler | — |

The S3-DiT architecture concatenates text tokens, visual semantic tokens, and image VAE tokens at the sequence level as a unified input stream, maximizing parameter efficiency compared to dual-stream approaches.

## Quantization Details

| Parameter | Value |
|-----------|-------|
| Bits | 4 |
| Group Size | 64 |
| Quantized Components | Text Encoder (Qwen3), Transformer (ZImageTransformer2DModel) |
| Non-Quantized Components | VAE (AutoencoderKL) — kept at float16 for image quality |
| Quantized Tensors | 526 Linear layer weight tensors |
| Method | MLX group quantization (`mlx.core.quantize`) |

Only 2D weight tensors from Linear layers are quantized. Normalization layers, biases, embeddings,
and position encodings remain in float16.

## Component Sizes

| Component | Original (bf16) | This Variant (4-bit Quantized) |
|-----------|----------------|----------------------------------|
| Text Encoder | 7.8 GB | ~2.8 GB |
| Transformer | 24.6 GB | ~3.5 GB |
| VAE | 160 MB | 160 MB |
| **Total** | **~32.6 GB** | **6.48 GB** |

## Original Model

- **Source**: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- **Authors**: Tongyi MAI Team (Alibaba)
- **License**: Apache 2.0
- **Papers**:
  - Z-Image: [arXiv:2511.22699](https://arxiv.org/abs/2511.22699)
  - Decoupled-DMD: [arXiv:2511.22677](https://arxiv.org/abs/2511.22677)
  - DMDR: [arXiv:2511.13649](https://arxiv.org/abs/2511.13649)

## Original Usage (PyTorch/CUDA)

```python
import torch
from diffusers import ZImagePipeline

pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

prompt = "Young Chinese woman in red Hanfu, intricate embroidery, ancient temple backdrop"

image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=9,   # Results in 8 DiT forwards
    guidance_scale=0.0,      # No CFG for Turbo models
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("example.png")
```

## Conversion Details

- Converted using MLX {0.30.6} on Apple Silicon
- Weights converted from bfloat16 to float16
- SafeTensors format (MLX-compatible)
- All weight keys preserved and verified
- VAE kept at float16 across all quantization levels
- Verified: no NaN/Inf values, all shapes consistent, all index files valid

## Citation

```bibtex
@article{z-image2025,
    title={Z-Image: An Efficient Image Generation Foundation Model with Scalable Single Stream Diffusion Transformer},
    author={Tongyi MAI Team},
    journal={arXiv preprint arXiv:2511.22699},
    year={2025}
}

@article{decoupled-dmd2025,
    title={Decoupled Consistency Model Distillation},
    author={Liu et al.},
    journal={arXiv preprint arXiv:2511.22677},
    year={2025}
}

@article{dmdr2025,
    title={DMDR: Fusing DMD with Reinforcement Learning},
    author={Jiang et al.},
    journal={arXiv preprint arXiv:2511.13649},
    year={2025}
}
```
