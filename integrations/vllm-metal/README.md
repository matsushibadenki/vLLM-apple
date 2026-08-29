# vLLM-Metal integration patches

`813e738d-native-v2-measurement.patch` targets vLLM-Metal commit
`813e738d95840bd66b60248ad4a557485320d896`.

Apply it from that vLLM-Metal checkout:

```sh
git apply --check /path/to/vLLM-apple/integrations/vllm-metal/813e738d-native-v2-measurement.patch
git apply /path/to/vLLM-apple/integrations/vllm-metal/813e738d-native-v2-measurement.patch
VLLM_METAL_BUILD_FROM_SOURCE=1 python -m vllm_metal.metal.build
```

The patch only exposes the isolated measurement ABI and a thread-local family
override. With the override unset, normal vLLM-Metal dispatch is unchanged.
