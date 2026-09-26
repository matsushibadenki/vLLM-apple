# Upstream review — 2026-09-26

公式releaseページを確認した。以下は取得ページに表示された版であり、索引の遅延もあり得る。特にMLX-LMの表示とローカル認定済み0.32.0は一致しないため、「最新」を根拠にdowngradeしない。外部ソースコードの転載・依存更新は今回行っていない。採用時には完全commit SHAと当該版licenseを確認する。

| Project / source | 表示version / commit短縮値 | 判断 |
| --- | --- | --- |
| [MLX](https://github.com/ml-explore/mlx/releases/tag/v0.32.2) | 0.32.2 / `1f8e74e` | GQA decodeのK/V読み出し改善を次の比較候補にする。MetalのABI固定依存とは別環境で評価する |
| [MLX-LM](https://github.com/ml-explore/mlx-lm/releases/tag/v0.31.3) | 0.31.3 / `ed1fca4` | thread-local生成streamとbatch cache修正を確認。取得ページの版がローカル0.32.0より古く、更新可否の判断は保留 |
| [vLLM-Metal](https://github.com/vllm-project/vllm-metal/releases/tag/v0.30.0) | stable 0.30.0 / `15f0b21` | unified KV storage、DLPack、window prefillが高優先候補。MLX 0.32.1固定ABIとKV dtype制約を維持して比較する |
| [vLLM-Metal prerelease](https://github.com/vllm-project/vllm-metal/releases/tag/v0.30.0.dev20260925203833) | dev20260925203833 / `f7d3766` | multimodal cache入力の保護など。標準text比較より優先度は低い |
| [llama.cpp](https://github.com/ggml-org/llama.cpp/releases/tag/b11192) | b11192 / `171e884` | cpp-httplib更新。GGUF比較backend候補として追跡。今回のPython HTTP実装への直接移植対象ではない |
| [vllm-mlx](https://github.com/waybarrios/vllm-mlx/releases/tag/v0.5.0) | 0.5.0 / `b064502` | prefix trieとstream完了時のcache保存race修正を確認。実KV再利用はP2で検証する |

今回の実装はP0の通信完了計測を独立実装したもの。上流kernelやcacheを移植したとは扱わない。backend更新の効果と終了通知待ち時間を区別するため、既存phase probeへend-to-endとstream tailを追加した。実モデル比較、上流版の実機昇格、共通goodput reportは次の作業として残す。

このタスクに毎週月曜・木曜09:00の定期確認を設定。更新確認後はROADMAP順に1件ずつ実装・検証し、意味のある完了・失敗・対応事項だけ通知する。

English: Official releases were reviewed; no upstream source or dependency was adopted
in this change. Transport instrumentation prepares reproducible comparisons. Version
listing discrepancies remain unresolved, and runtime qualification is still required.

简体中文：已检查官方发布记录，本次未移植上游代码或更新依赖。新增通信计时用于准备
可复现的比较。版本列表差异仍待核实，后端升级必须通过实际运行验证。
