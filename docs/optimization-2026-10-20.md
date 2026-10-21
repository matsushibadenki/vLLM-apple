# 定期評価 / Recurring evaluation / 定期评估 — 2026-10-20

Problem: prefill host待ちがGPU実行かhost overheadか未分離だった。
Root cause: 今回の限定traceではGPU executionが支配的。帯域／compute律速の判別は未取得。
Evidence: [GPU report](evaluation/recurring-gpu-profile-2026-10-20-r1.json)、[profilerなしHTTP](evaluation/recurring-optimization-gemma2-2026-10-20-r1.json)、[host gate](evaluation/recurring-optimization-host-preflight-2026-10-20-r1.json)。専用試験競合なし、低負荷gate通過。旧r11 failedは別途確認。
Changed files: 本記録・評価JSON・ROADMAP追記、非versioned raw trace保存のみ。
Change: 既存experiments.p3_mlx.profileで30秒Metal traceを取得後、別workerの通常HTTP品質・速度を測定。runtimeコード変更なし。
Why it should improve performance: 今回は診断を完結し、根拠のない同期削除を避ける。
Before: host eval待ちだけではGPU/host帰属不明。
After: 12 prefill samplesの代表wall1233.572 ms、GPU active1216.862 ms、比率98.49%。decode12 samplesはwall23.917 ms、GPU active22.351 ms、比率89.25%。trace集計はphaseのmedianベースで、比率はsampleごとの統計と混同しない。quantized matmul/fused dequantizationのsampled shader時間shareはprefill77.26%、decode84.42%。shader shareとGPU wallの分母は異なる。copy/indexing全体をKV copyと解釈しない。
CPU impact: prefill graph encode代表2.025 ms、calling CPU86.464 ms。host waitを全て不要な同期と解釈しない。
GPU impact: target PID13961のdepth-zero Active Compute区間union、mach clock照合。GPU単独traceによる限定診断で高速化認定なし。
Memory impact: allocator peakは各report参照。RSSや長時間安定性の資格ではない。
I/O impact: trace約229MiBをqualification-resultsへ保持しversioned JSONと分離。
Energy impact: 未測定。AC Power／nominal thermal／automatic power modeで観測。
Correctness verification: GPU workerのidentity不変・正常終了。別profilerなしHTTPでwarmup3、長文12、短文30の45/45算術trimmed-exact品質・SLO合格、正常終了。短文は英語・日本語・简体中文各10件。一般生成・perplexity・コード・数学一般・長文理解・stress・長時間は未認定。
Regression risk: runtime変更なし。profiler条件とHTTP条件を同一workloadと扱わず、trace時間からHTTP改善率を計算しない。低負荷gateは開始時確認で、全期間のhost独占を保証しない。
Keep / Revert: 証跡のみKeep。新しい最適化候補を採用しない。

HTTP短文goodput34.858 tokens/s、E2E mean225.882 ms、p50/p95 bucket上限250/500 ms、max301.948 ms。TTFT／TPOT／p99参考上限はraw report参照。単一取得で速度変化を認定しない。

Commands:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python -m experiments.p3_mlx.profile --python /opt/homebrew/opt/vllm-metal/libexec/bin/python --model models/gemma-2-2b-it-4bit --output-directory /private/tmp/vllm-apple-gpu-profile-2026-10-20-r1
VLLM_APPLE_P1_EFFICIENCY=compact PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python scripts/qualify_gemma2_batch_mask.py --python /opt/homebrew/opt/vllm-metal/libexec/bin/python --model models/gemma-2-2b-it-4bit --output /private/tmp/recurring-optimization-gemma2-2026-10-20-r1.json --port 19221 --sustained-requests 30 --long-requests 12 --duration-seconds 0 --decode-concurrency 2 --prompt-concurrency 2 --prefill-step-size 512 --long-concurrency 1 --p1-profile
```

Raw directoryは取得後qualification-results/recurring-gpu-profile-2026-10-20-r1へ移動。report内の元/tmp pathは取得時のprovenanceとして保持し、対応basenameは移動先にある。

🟢 [Done] 現在identityのGPU/host診断と別worker品質確認。🟠 [Next] 支配する量子化演算を対象に帯域／compute counterを確認し一つの仮説へ絞る。🔴 [Later] 広い品質・shape・stress・長時間認定。⭕️ [Pending] 未保有SoC／独立Mac実測。

English: A fresh target-PID Metal trace attributes98.49% of the representative prefill wall time to GPU activity. Quantized matmul/fused dequantization dominates sampled shader time. This is diagnostic, not a speedup result or proof of bandwidth/compute limitation. A separate unprofiled HTTP worker passed all45 arithmetic quality/SLO checks and shut down cleanly. No runtime patch adopted.

简体中文：新的目标PID Metal trace显示，prefill代表wall time中GPU活动占98.49%，量化矩阵运算／融合反量化主导shader采样时间。仅为诊断，不代表速度提升，也不认证带宽或compute瓶颈。独立无profiler HTTP worker的45件算术质量／SLO全部通过，正常退出。不采用runtime修改。
