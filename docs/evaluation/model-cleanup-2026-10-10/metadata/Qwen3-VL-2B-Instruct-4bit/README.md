---
license: apache-2.0
pipeline_tag: image-text-to-text
library_name: transformers
tags:
- mlx
---

# mlx-community/Qwen3-VL-2B-Instruct-4bit
This model was converted to MLX format from [`Qwen/Qwen3-VL-2B-Instruct`]() using mlx-vlm version **0.3.4**.
Refer to the [original model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) for more details on the model.
## Use with mlx

```bash
pip install -U mlx-vlm
```

```bash
python -m mlx_vlm.generate --model mlx-community/Qwen3-VL-2B-Instruct-4bit --max-tokens 100 --temperature 0.0 --prompt "Describe this image." --image <path_to_image>
```
