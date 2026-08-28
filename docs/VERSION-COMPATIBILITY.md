# Runtime Version Compatibility

最終更新：2026-08-28

`vllm-apple doctor`は推論processを起動する前に、同じvirtual environment内のPython、vLLM、
vLLM-Metal、Transformersを検査する。verified範囲外は動作不能と断定するのではなく、未検証構成を
Metal推論へ昇格させないfail-closed判定である。

package metadataだけでなく、`vllm.platforms.current_platform`のmodule、class、`is_cpu()`も検査する。
vLLM-Metalが導入済みでもMetal pluginが選択されず`CpuPlatform`へ落ちた場合は
`vllm_metal_platform_not_selected`を返す。

| Component | Verified range | Status |
| --- | --- | --- |
| Python | 3.12 / 3.13 | `[Done]` |
| vLLM | `>=0.24.0,<0.28.0` | `[Done]` |
| vLLM-Metal | `>=0.2.0,<0.4.0` | `[Done]` |
| Transformers | `>=5.5.3,<5.13.0` | `[Done]` |

vLLM 0.28.0はTransformers 5.15.0へ更新された一方、確認対象のvLLM-Metal 0.3系はMetal platform
検出の既知問題によりTransformersを`<5.13`へ制限している。このため、`0.28.0 + 0.3.x + 5.15.0`
を自動的に対応済みとは扱わない。

versionは先頭のrelease tripletで比較し、`0.3.0.dev...`や`0.26.0+metal`のsuffixはidentityとして
保持しつつrange判定を妨げない。解析不能なversion文字列は明示issueを返す。

## Promotion gate

vLLM 0.28系をverified範囲へ昇格する条件：

1. vLLM-MetalがTransformers 5.15系でMetal platformを選択すること
2. 小型safetensors modelがCPU fallbackなしで起動すること
3. greedyとtemperature samplingのcorrectness testが通ること
4. prefix cache、streaming終了、long-context memory stabilityが通ること
5. native v2 Paged Attentionのsource fingerprint別profileを再生成すること

`qualify-model`はsoak開始前に、同一prompt/seedによるgreedy再現性、temperature 0.7のsampling再現性、
samplingのstream/non-stream digest一致、SSE `[DONE]`を検査する。生成本文はreportへ保持せずdigestだけを
記録し、いずれかが失敗した場合は30分試験へ進まずbackendを停止する。
