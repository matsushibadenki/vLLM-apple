---
license: gemma
library_name: transformers
pipeline_tag: image-text-to-text
extra_gated_heading: Access Gemma on Hugging Face
extra_gated_prompt: To access Gemma on Hugging Face, you’re required to review and
  agree to Google’s usage license. To do this, please ensure you’re logged in to Hugging
  Face and click below. Requests are processed immediately.
extra_gated_button_content: Acknowledge license
base_model: google/gemma-3-4b-pt
base_model_relation: quantized
tags:
- mlx
---

# mlx-community/gemma-3-4b-it-4bit
This model was converted to MLX format from [`google/gemma-3-4b-it`]() using mlx-vlm version **0.1.18**.
Refer to the [original model card](https://huggingface.co/google/gemma-3-4b-it) for more details on the model.
## Use with mlx

```bash
pip install -U mlx-vlm
```

```bash
python -m mlx_vlm.generate --model mlx-community/gemma-3-4b-it-4bit --max-tokens 100 --temperature 0.0 --prompt "Describe this image." --image <path_to_image>
```
