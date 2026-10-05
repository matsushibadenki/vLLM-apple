# Upstream review — 2026-10-05

参照：2026-10-05 08:25:57 JST（2026-10-04 23:25:57 UTC）。公式release／tag／commit／licenseを確認。
P1 r5が動作中のためruntime・依存・GPU負荷は変更しない。本reviewは公開情報の採用判断であり、
新versionの動作・品質・速度を認定しない。コードの直接流用は行っていない。

| Project | 確認したversion／commit | License | 採用判断 |
| --- | --- | --- | --- |
| MLX | [v0.32.3](https://github.com/ml-explore/mlx/releases/tag/v0.32.3)／[64ea011cb65f14d9ce2737e60db9a4ae91ed7441](https://github.com/ml-explore/mlx/commit/64ea011cb65f14d9ce2737e60db9a4ae91ed7441) | [MIT](https://raw.githubusercontent.com/ml-explore/mlx/v0.32.3/LICENSE) | [Next] array buffer容量取得・stream修正・GQA kernel修正を分離環境で比較する。現MLX 0.32.1から自動更新しない。M5／NAX固有の検証は[pending] |
| MLX-LM | [v0.32.0 tag](https://github.com/ml-explore/mlx-lm/releases/tag/v0.32.0)／[a9bd8af5c02118882af735cef60705d2efce9fd0](https://github.com/ml-explore/mlx-lm/commit/a9bd8af5c02118882af735cef60705d2efce9fd0) | [MIT](https://raw.githubusercontent.com/ml-explore/mlx-lm/v0.32.0/LICENSE) | 現installed versionと同名tagを確認。release一覧はv0.31.3を先頭表示しており、tagとpublished releaseを混同しない。package内容とtagの一致は未確認、現source hash gateを維持 |
| vLLM-Metal | [stable v0.30.0](https://github.com/vllm-project/vllm-metal/releases/tag/v0.30.0)／[15f0b215c89825928ac796ab8face335f714163f](https://github.com/vllm-project/vllm-metal/commit/15f0b215c89825928ac796ab8face335f714163f) | [Apache-2.0](https://github.com/vllm-project/vllm-metal/blob/v0.30.0/LICENSE) | [Next] core 0.30.0との分離環境でunified KV・paged経路を比較。MLXを0.32.1へ厳密固定するABI契約を保持。旧memory／paged flagの削除を移行時に確認。dev releaseは標準採用しない |
| llama.cpp | [b11399 prerelease](https://github.com/ggml-org/llama.cpp/releases/tag/b11399)／[2ca15f5404760548c39e7b92bd43116a09414a1a](https://github.com/ggml-org/llama.cpp/commit/2ca15f5404760548c39e7b92bd43116a09414a1a) | [MIT](https://raw.githubusercontent.com/ggml-org/llama.cpp/2ca15f5404760548c39e7b92bd43116a09414a1a/LICENSE) | 当該releaseの変更はCUDA swizzling。現在のM4のP1 TTFT対策として採用しない。Metal／GGUF比較は独立artifactで実測する |
| vllm-mlx | [v0.5.0](https://github.com/waybarrios/vllm-mlx/releases/tag/v0.5.0)／b064502（release表示の短縮commitのみ確認） | [Apache-2.0](https://raw.githubusercontent.com/waybarrios/vllm-mlx/v0.5.0/LICENSE) | [Later] conversation prefix trie・正常stream完了時のcache保存race修正を参考候補にする。現P2はnative LRUPromptCache経路なので同じbugとは断定しない。直接流用前に完全commitとcache-on/off数値・E2Eを確認 |

MLX v0.32.3のbuffer容量APIはP0／P2の論理KV容量と物理bufferの区別を強化する候補。
GQA batch offsetなどの修正もあるが、既存Gemma mask互換修正やP2のlogit差と同一原因とは
確認できていない。新versionの採用には同じmodel／shape・fresh process・cache条件での
correctnessとE2E再測定を必要とする。[公式変更履歴](https://github.com/ml-explore/mlx/releases/tag/v0.32.3)

vLLM-Metal stable 0.30.0のrelease記述ではunified KV storage、MLX 0.32.1固定、legacy serving
flag削除が重要。hybrid GDNのprefix parity未達も明記されている。新releaseの機能説明を
このrepositoryの能力・認定matrixへ自動転記しない。
[公式互換性・既知境界](https://github.com/vllm-project/vllm-metal/releases/tag/v0.30.0)

## English

Official releases, tags, commits and licenses were reviewed while P1 r5 was running.
No runtime, dependency or GPU workload changed. Prioritize an isolated comparison of
Metal 0.30.0 and MLX buffer accounting after the current test. Metal's exact MLX 0.32.1
ABI pin prevents blindly upgrading to MLX 0.32.3. MLX-LM's release-list and tag views
differ; neither proves installed source equivalence. All new-version capabilities and
performance remain unqualified. M5-specific testing requires the target hardware.

## 简体中文

在P1 r5运行时检查了官方release、tag、commit和license，没有修改runtime、依赖或GPU负载。
当前试验结束后，优先在隔离环境比较Metal 0.30.0及MLX buffer容量计量。
Metal严格固定MLX 0.32.1，不能直接升级为0.32.3。MLX-LM列表和tag显示不同，不能据此
认定installed source一致。新version的能力与性能均未认证。M5专用试验需要目标实机。
