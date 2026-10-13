# 定期評価 / Recurring evaluation / 定期评估 — 2026-10-13

Problem: 固定条件の再現性とwarmup境界が未確認だった。
Root cause: 速度変動の原因は未確定。prefill evalのhost待ちが支配的だがGPU単独時間・allocation原因とは断定しない。
Evidence: [3回の要約](evaluation/recurring-optimization-summary-2026-10-13-r1.json)、各raw report recurring-optimization-gemma2-2026-10-13-r1/r2/r3.json。開始時プロセス一覧に競合なし、p1_stability_statusで旧r11のfailedを確認。
Changed files: 本記録、評価JSON、ROADMAPの短い追記のみ。既存ユーザー差分を保持。
Change: 独立worker再起動によるbaseline3回。モデル・runtime sources・runner一致=True/True/True。M4/Gemma2 4bit、compact、prefill512、decode/prompt concurrency2、long concurrency1、warmup3/長文12/短文30/30秒window固定。
Why it should improve performance: 速度改善patchなし。計測変動を把握し、根拠不足の最適化を避ける。
Before: 10/12単一測定、短文goodput37.628 tokens/s、warmup SLO2/3。
After: 短文goodput38.928/39.106/35.562 tokens/s、中央値38.928、範囲/中央値9.11%。短文E2E mean203.136/202.253/222.548 ms、p95 bucket上限250/250/500 ms、max241.788/246.855/441.907 ms。median・TTFT・TPOT・p99上限は要約JSON。bucket上限を正確なpercentileと扱わず少数標本p99は参考値。速度改善を認定しない。
CPU impact: run1 prefill eval host mean277.855 ms、clear_cache0.953 ms、cache_fetch0.252 ms。これらは入れ子なので加算しない。CPU削減未測定。
GPU impact: kernel変更なし、GPU単独時間未測定。
Memory impact: window peak RSS2,137,751,552/2,180,907,008/2,180,055,040 bytes。window RSSは3回とも減少。短時間減少をleak不存在と認定しない。
I/O impact: 既存localhost HTTPとreport保存。cacheはbackend管理で完全制御されない。
Energy impact: 未測定。
Correctness verification: 全489件の算術trimmed-exact品質および当該SLO合格。warmupは各3/3。短文は各runで英語・日本語・简体中文各10/10。各総合passed=true、正常終了。runtime patchなしにつき新規回帰テストなし。
Regression risk: 5%以内のrun変動条件を満たさず候補採用なし。一般生成品質・perplexity・コード・数学一般・長文理解・stress・8時間資格は未評価。3回の単一baselineでP3候補gate通過を主張しない。
Keep / Revert: 証跡のみKeep。最適化patchなし。

Command: VLLM_APPLE_P1_EFFICIENCY=compact PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python scripts/qualify_gemma2_batch_mask.py --python /opt/homebrew/opt/vllm-metal/libexec/bin/python --model models/gemma-2-2b-it-4bit --output /private/tmp/recurring-optimization-gemma2-2026-10-13-rN.json --port 1918N --sustained-requests 30 --long-requests 12 --duration-seconds 30 --decode-concurrency 2 --prompt-concurrency 2 --prefill-step-size 512 --long-concurrency 1 --p1-profile（N=1,2,3、順次独立worker）

🟢 [Done] 同identityで3回基準測定。🟠 [Next] 変動の環境要因とprefill待ちを切り分け、原因仮説一つを検証。🔴 [Later] 広い品質・stress・長時間認定。⭕️ [Pending] GPU単独時間と電力の対応collectorによる実測。

English: All 489 arithmetic and SLO checks passed across three independently restarted workers; clean shutdown throughout. Short-workload goodput ranged from 35.562 to 39.106 tokens/s (9.11% range/median). Baseline variability exceeds the existing 5% criterion, so no optimization was adopted. General quality, power and long-run stability remain unqualified.

简体中文：3次独立重启worker，全部489件算术质量及SLO通过，正常退出。短文goodput为35.562–39.106 tokens/s，范围/中位数9.11%，超过现有5%变动条件，因此不采用优化。通用质量、功耗及长期稳定性仍未认证。
