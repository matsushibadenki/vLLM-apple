---
license: apache-2.0
language:
- en
base_model:
- yl4579/StyleTTS2-LJSpeech
pipeline_tag: text-to-speech
library_name: mlx
---
# mlx-community/Kokoro-82M-6bit
This model was converted to MLX format from [`hexagrad/Kokoro-82M`]() using mlx-audio version **0.0.1**.
Refer to the [original model card](https://huggingface.co/hexagrad/Kokoro-82M) for more details on the model.
## Use with mlx

```bash
pip install -U mlx-audio
```

```bash
python -m mlx_audio.tts.generate --model mlx-community/Kokoro-82M-6bit --text "Hello world"
```