# 定期評価 / Recurring evaluation / 定期评估 — 2026-10-12

Problem: warmup尾遅延と長時間資格が未解決。
Root cause: 未確定。途中の既存実装変更と日ごとの変動を今回の最適化効果にしない。
Evidence: [実機report](evaluation/recurring-optimization-gemma2-2026-10-12-r1.json)、[要約](evaluation/recurring-optimization-summary-2026-10-12-r1.json)。プロセス一覧で競合なし、p1_stability_statusの旧r11はfailed。
Changed files: 本記録・評価JSONのみ。ユーザーの既存差分を保持。
Change: M4/Gemma 2B 4bit、compact、prefill512、decode/prompt concurrency2、long concurrency1、短文30・長文12・30秒windowを再計測。runtime変更なし。
Why it should improve performance: 今回は速度改善patchを採用せず、現在の基準と未達を記録。
Before: 10/9短文goodput20.497 tokens/s、E2E mean388.704 ms。
After: 短文goodput37.628 tokens/s、E2E mean210.829 ms、TTFT mean125.447 ms、TTFT p50/p95 bucket上限250/250 ms、max171.323 ms。E2E p50/p95上限250/250 ms、max261.553 ms。p99は少数標本の参考値としてJSONに保存。実装identityの一致判定は要約JSONに記録し、速度差の因果関係を認定しない。
CPU impact: 今回の変更なし。host/backend phase診断はraw report内、CPU削減率は未認定。
GPU impact: kernel変更なし。GPU単独時間未測定。
Memory impact: window peak RSS2,112,061,440 bytes、差分+134,381,568 bytes。30秒を長時間安定性・leak認定にしない。
I/O impact: 既存localhost HTTP評価と証跡保存。
Energy impact: 未測定。
Correctness verification: warmup3、長文12、短文30、window153の計198件算術trimmed-exact品質合格。短文は英語・日本語・简体中文各10件合格。warmup SLO2/3、TTFT max1081.087 msで1000 ms上限を超過。その他の当該SLO全件合格、正常終了。総合passed=trueでもwarmup未達を隠さない。
Regression risk: runtime patchなし。perplexity、一般生成、数学一般、コード、長文理解、8時間安定性は今回未評価。cacheはbackend管理で完全制御ではない。small/medium/large/stressの網羅的認定なし。
Keep / Revert: 証跡のみKeep。推測による最適化を追加しない。

Command:

```sh
VLLM_APPLE_P1_EFFICIENCY=compact PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python scripts/qualify_gemma2_batch_mask.py --python /opt/homebrew/opt/vllm-metal/libexec/bin/python --model models/gemma-2-2b-it-4bit --output /private/tmp/recurring-optimization-gemma2-2026-10-12-r1.json --port 19179 --sustained-requests 30 --long-requests 12 --duration-seconds 30 --decode-concurrency 2 --prompt-concurrency 2 --prefill-step-size 512 --long-concurrency 1 --p1-profile
```

🟢 [Done] 現行基準測定。🟠 [Next] 同じ実装identityで独立反復しwarmup境界を診断。🔴 [Later] 広い品質と長時間認定。⭕️ [Pending] GPU単独時間・電力collectorによる実測。

English: All 198 arithmetic quality checks passed; one of three warmup requests missed the 1-second TTFT limit (1081.087 ms). Other measured SLO checks passed and shutdown was clean. No runtime patch or speedup attribution. Repeat with identical implementation identity before adopting an optimization; broad quality and long-run stability remain unqualified.

简体中文：198件算术质量检查全部通过；3件warmup中1件TTFT超过1秒上限（1081.087 ms）。其他测得SLO全部通过，正常退出。未修改runtime，不认证速度提升。采用优化前需固定实现identity并独立重复；通用质量和长期稳定性仍未认证。
