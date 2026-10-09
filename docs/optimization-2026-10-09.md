# 定期評価 / Recurring evaluation / 定期评估 — 2026-10-09

Problem: P1の長時間資格は未達。短時間の精度・速度を再測定し、診断対象を絞る。
Root cause: 未確定。今回もprefill evalのhost待ちが支配的だが、GPU単独時間や根因とは認定しない。
Evidence: [実機結果](evaluation/recurring-optimization-gemma2-2026-10-09-r1.json)、[比較](evaluation/recurring-optimization-comparison-2026-10-09-r1.json)。プロセス一覧に競合する試験なし。p1_stability_statusでr11の既存失敗を確認。
Changed files: 本記録、評価JSON、ROADMAPへの短い追記のみ。既存の未コミット変更は保持。
Change: M4/32 GiB、Gemma 2B 4bit、compact、prefill512、decode/prompt並行数2で基準を再測定。runtime sources・runner・model filesの記録は前回と一致。
Why it should improve performance: 今回は改善を採用せず、比較可能な診断証跡を追加する。

| 測定 | Before | After | 差分 |
| --- | ---: | ---: | ---: |
| 短文30要求 goodput tokens/s | 18.008 | 20.497 | +13.82% |
| 短文 E2E mean ms | 441.551 | 388.704 | −11.97% |
| 短文 TTFT mean ms | 282.098 | 289.895 | +2.76% |
| 短文 E2E p95 bucket上限 ms | 1000 | 1000 | 0% |
| 短文 E2E maximum ms | 579.133 | 558.179 | −3.62% |
| 長文12要求 E2E mean ms | 607.615 | 608.583 | +0.16% |
| warmup品質 / SLO合格 | 3/3 / 3/3 | 3/3 / 2/3 | SLO 1件未達 |

単一再測定・コード変更なしのため、上の差分を高速化の効果と認定しない。cacheはbackend管理で完全には制御されない。median/p95はbucket上限、p99は少数標本の参考値（JSON参照）。30秒windowは前回102件、今回99件なので総数を直接比較しない。

CPU impact: prefill eval 37回、host mean 231.845 ms、calling-thread CPU mean 19.561 ms。CPU値は他thread/GPUを含まない。
GPU impact: GPU単独時間は欠測。kernel変更なし。
Memory impact: window peak RSS 2,205,138,944 bytes、window RSS差分 −78,692,352 bytes。30秒の減少はleak不存在や長時間plateauの証明ではない。
I/O impact: 既存localhost HTTP評価とreport保存。実装変更なし。
Energy impact: 未測定。
Correctness verification: warmup3、長文12、短文30、window99要求すべて算術trimmed-exact品質合格。英語・日本語・简体中文の短文各10/10。長文・短文・windowのSLO全件合格、warmupは1件TTFT 1001.743 msで1000 ms上限未達。runner総合passed=trueでもこの未達を隠さない。正常停止。test_process_activity_delta 2件通過。
Regression risk: runtime変更なし。一般生成品質、perplexity、コード、数学一般、長文理解、8時間安定性は認定しない。
Keep / Revert: 証跡のみKeep。最適化patchなし。今回の変動から最適化を推測しない。

実行コマンド：

```sh
VLLM_APPLE_P1_EFFICIENCY=compact PYTHONPATH=. .venv/bin/python scripts/qualify_gemma2_batch_mask.py --python /opt/homebrew/opt/vllm-metal/libexec/bin/python --model models/gemma-2-2b-it-4bit --output docs/evaluation/recurring-optimization-gemma2-2026-10-09-r1.json --port 19169 --sustained-requests 30 --long-requests 12 --duration-seconds 30 --decode-concurrency 2 --prompt-concurrency 2 --prefill-step-size 512 --long-concurrency 1 --p1-profile
.venv/bin/python -m unittest discover -s tests -p test_process_activity_delta.py
```

🟢 [Done] 比較可能な短時間基準の再測定。🟠 [Next] 独立反復でwarmup境界とprefill待ちを確認し、一つの原因仮説を検証する。🔴 [Later] 長時間安定性認定。⭕️ [Pending] 電力・GPU単独時間は対応collectorによる実測が必要。

English: Repeated the unchanged M4/Gemma baseline; all 144 arithmetic quality checks passed. One of three warmup requests missed the 1-second TTFT limit by 1.743 ms. Other measured SLO checks passed and shutdown was clean. The single-run goodput difference is not an optimization result. Keep evidence only; repeat before attributing prefill waits or adopting changes. General quality, power and long-run stability remain unqualified.

简体中文：重测未修改的M4/Gemma基线，144件算术质量检查全部通过。3件warmup中1件TTFT超过1秒上限1.743 ms；其他测得的SLO全部通过，正常停止。单次goodput差异不代表优化效果。仅保留证据，先独立重复测量，再定位prefill等待并验证修改。通用质量、功耗和长期稳定性仍未认证。
