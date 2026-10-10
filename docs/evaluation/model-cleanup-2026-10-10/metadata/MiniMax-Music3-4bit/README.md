---
library_name: mlx-audio
pipeline_tag: text-to-audio
base_model: MiniMaxAI/MiniMax-Music3
license: other
license_name: minimax-music3-community-license
license_link: https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/main/LICENSE
inference: false
tags:
  - mlx
  - mlx-audio
  - apple-silicon
  - music-generation
  - text-to-music
  - text-to-audio
  - quantized
  - 4bit
---

# MiniMax Music 3 · MLX 4-bit

Native MLX affine 4-bit weights for
[`MiniMaxAI/MiniMax-Music3`](https://huggingface.co/MiniMaxAI/MiniMax-Music3),
converted for lyric-conditioned song generation on Apple Silicon with
[`mlx-audio`](https://github.com/Blaizzy/mlx-audio).

> Community conversion, not an official MiniMax release. All model credit goes
> to MiniMax. Review the original model card and license before use.

Other MLX variants:
[`BF16`](https://huggingface.co/mlx-community/MiniMax-Music3-bf16)
· [`8-bit`](https://huggingface.co/mlx-community/MiniMax-Music3-8bit)
· [`6-bit`](https://huggingface.co/mlx-community/MiniMax-Music3-6bit)
· [`MXFP8`](https://huggingface.co/mlx-community/MiniMax-Music3-mxfp8)
· [`MXFP4 (experimental)`](https://huggingface.co/mlx-community/MiniMax-Music3-mxfp4)
· [`NVFP4 (experimental)`](https://huggingface.co/mlx-community/MiniMax-Music3-nvfp4)

## Install

MiniMax Music 3 support was merged upstream in
[`Blaizzy/mlx-audio#888`](https://github.com/Blaizzy/mlx-audio/pull/888).
Until a PyPI release includes it, install the upstream merge commit directly:

```bash
python -m pip install "mlx-audio @ git+https://github.com/Blaizzy/mlx-audio.git@784b29e2691a93ca7483147d86f61859dfaa6296"
```

## Generate

```bash
python -m mlx_audio.music.generate \
  --model mlx-community/MiniMax-Music3-4bit \
  --caption "Warm acoustic pop, 96 BPM, intimate female vocal" \
  --lyrics $'[verse]\nMorning light across the room\n[chorus]\nSing with me' \
  --duration 30 \
  --steps 30 \
  --seed 7 \
  --output song.wav
```

```python
from mlx_audio.music import load

model = load("mlx-community/MiniMax-Music3-4bit")
result = next(
    model.generate(
        text="Warm acoustic pop, 96 BPM, intimate female vocal",
        lyrics="[verse]\nMorning light across the room\n[chorus]\nSing with me",
        duration=30,
        steps=30,
        seed=7,
    )
)
print(result.audio.shape, result.sample_rate)  # stereo, 44100 Hz
```

Lyrics are required by the checkpoint contract. Use `[instrumental]` explicitly
for instrumental generation. Duration is a requested upper bound: the
autoregressive stage may emit its end token early. Style, tempo, instrument, and
vocal controls are probabilistic rather than strict.

## Quantization and verification

- Affine 4-bit weight quantization with group size 64.
- Large linears in the global language model, RVQ depth decoder, and flow
  transformer are quantized. Embeddings, output heads, convolutions, condition
  encoder, and vocoder remain dense for fidelity.
- The two weight shards total 9,199,141,298 bytes.
- Strict loading and real generation produced finite 44.1 kHz stereo audio.
- The mlx-audio regression suite passed 1,742 tests with 34 expected skips; the
  focused music, converter, and registry suite passed 43 tests and 3 subtests.

Converted with `mlx-audio` 0.4.8 development commit `c2fa486` and MLX 0.31.2.

## License

The weights remain subject to the
[`MiniMax-Music3 Community License`](https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/main/LICENSE),
including its acceptable-use and commercial terms. The full license text is
included in this repository.
