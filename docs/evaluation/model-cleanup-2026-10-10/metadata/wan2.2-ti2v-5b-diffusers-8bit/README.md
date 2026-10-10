---
license: apache-2.0
base_model: Wan-AI/Wan2.2-TI2V-5B-Diffusers
pipeline_tag: text-to-video
library_name: mlx-gen
tags:
- mlx
- mlx-gen
- mflux
- apple-silicon
- 8-bit
- mixed-q8-bf16
- wan
- wan2.2
- video-generation
- text-to-video
- image-to-video
---
# wan2.2-ti2v-5b-diffusers-8bit

This repository contains mixed q8/BF16 MLX-Gen saved weights for
[`Wan-AI/Wan2.2-TI2V-5B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers).
It is designed for local Apple Silicon inference with
[`mlx-gen`](https://github.com/lpalbou/mlx-gen).

It uses the mflux/MLX saved-weight layout with MLX quantization tensors. It is not a Diffusers or
Transformers `from_pretrained()` checkpoint.

## Source Model

Original model: [`Wan-AI/Wan2.2-TI2V-5B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers).

This quantized derivative follows the Apache 2.0 license of the source model.

## Quantization

This is a mixed q8/BF16 checkpoint:

- q8 for quantizable Wan transformer attention and feed-forward linears.
- BF16 for the Wan VAE.
- BF16 for Wan transformer `condition_embedder.*` and `proj_out`.
- BF16 for the UMT5 text encoder, scheduler metadata, tokenizer files, norms, convolutions, and
  other non-quantizable parameters.

The upstream TI2V-5B source snapshot is not uniformly 16-bit on disk: the transformer and VAE
safetensors are FP32, while the UMT5 text encoder is BF16. MLX-Gen loads Wan transformer/VAE
weights at BF16 runtime precision.

## Measurements

Measured on 2026-06-04 with `mlx-gen 0.18.10` on an Apple M5 Max with 128 GiB unified memory.

Validation profile: `1280x704`, 17 frames, 20 denoising steps, guidance `5`, 24 fps, seed `321`,
explicit empty negative prompt. This is a large normal-cache profile, not a `--low-ram` profile and
not comparable to the A14B short low-RAM rows as a model-size memory statement.

| Layout | Storage | Wan MLX Model | MLX Active After Generation | Full-Process Physical Peak | Max RSS | MLX Peak | Total Time | Output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Upstream source snapshot | 31.9 GiB | 10.6 GiB | 10.3 GiB | 102.7 GiB | 13.7 GiB | 58.5 GiB | 216.2 s | [base-source.mp4](validation/ti2v5b-clean/base-source.mp4) |
| Prepared BF16 package | 21.2 GiB | 10.6 GiB | 10.3 GiB | 102.6 GiB | 14.5 GiB | 58.5 GiB | 261.6 s | [prepared-bf16.mp4](validation/ti2v5b-clean/prepared-bf16.mp4) |
| This mixed q8/BF16 package | 16.9 GiB | 6.3 GiB | 6.1 GiB | 103.7 GiB | 13.8 GiB | 54.2 GiB | 243.4 s | [mixed-q8-bf16.mp4](validation/ti2v5b-clean/mixed-q8-bf16.mp4) |

This package reduces storage, logical model bytes, active MLX model bytes, and MLX allocator peak in
the validation profile. It did not reduce full-process physical peak memory in this profile because
transient video-generation allocations dominated the run.

The source and prepared BF16 package produced byte-identical decoded MP4 frames. This mixed q8/BF16
package stayed visually in the same family with mean frame MAE `1.66` versus source/BF16.

`Storage` is the Hugging Face repository total. `Wan MLX Model` is the loaded Wan transformer plus
VAE tensor footprint measured from MLX arrays; it excludes the UMT5 text encoder and video/save
buffers. `MLX Active After Generation` is the live MLX allocator footprint after `generate_video()`
returns, before cleanup. `Full-Process Physical Peak` is Darwin `phys_footprint` sampled from model
initialization through MP4 save and health validation. `Max RSS` can under-report Apple
unified-memory/Metal pressure, and `MLX Peak` is only the MLX allocator high-water mark.

Validation assets:

- [contact-sheet.png](validation/ti2v5b-clean/contact-sheet.png)
- [metrics.json](validation/ti2v5b-clean/metrics.json)

## Usage

```bash
python -m pip install -U mlx-gen

mlxgen download --model AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit

mlxgen generate \
  --model AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit \
  --prompt "A short cinematic video of a glowing orange glass sphere floating above calm teal water, soft reflections, gentle camera movement" \
  --negative-prompt "" \
  --width 1280 \
  --height 704 \
  --frames 17 \
  --steps 20 \
  --guidance 5 \
  --fps 24 \
  --seed 321 \
  --output video.mp4
```

TI2V-5B also supports first-frame image-to-video in MLX-Gen when one input image is supplied.

## Attribution

MLX-Gen is based on [mflux](https://github.com/filipstrand/mflux) by Filip Strand and the original
mflux contributors.

Quantized and contributed by [@lpalbou](https://huggingface.co/lpalbou).
