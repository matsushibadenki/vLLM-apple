# 定期最適化評価 / Optimization evaluation / 优化评估 — 2026-10-15

Problem: 既存profileでprefill evalのhost待ちが支配的。chunk縮小で改善するかを検証。
Root cause: GPU単独の根因は未確定。仮説は512→256で長いeval待ちを分割できること。
Evidence: [比較JSON](evaluation/recurring-prefill-comparison-2026-10-15.json)、同名raw6件。低負荷gate通過、専用試験の競合なし。旧r11はstatus helperでfailed。
Changed files: 本報告・評価JSON・ROADMAP追記のみ。既存runtimeの変更なし。
Change: 512/256を交互に各3回、独立workerで測定。compact/Gemma2 4bit/M4、decode・prompt concurrency2、long concurrency1、warmup3/long12/short30、duration0で固定。各runのモデル・runtime・runner identity一致は比較JSON参照。
Why it should improve performance: chunk縮小がprefill待ちを改善する仮説を試したが、結果は採用を支持しない。
Before: 512長文12要求の所要時間5.677/5.713/6.232秒、中央値5.713秒。
After: 256で5.662/5.759/6.630秒、中央値5.759秒（約0.8%悪化）。短文30要求は512=2.989/3.080/3.601秒、256=3.054/3.126/3.283秒。run範囲は重なり、5%以上の再現性ある改善なし。TTFT p95 bucket上限は長文5000 ms、短文250 msで両設定共通。median/p95/maximum、TPOT、goodput、p99参考上限、allocator peak、cache再利用は比較JSONに記録。bucket上限を正確なpercentileと扱わない。
CPU impact: host phase測定はraw内。chunk分割のCPU総量削減は未認定。
GPU impact: GPU単独時間未測定。chunk設定のみ変更、kernelコード変更なし。
Memory impact: phase_profile peakはMLX allocator側指標でRSS peakと混同しない。duration0につき長時間RSS推移は未評価。
I/O impact: 既存localhost HTTPとreport保存、cacheはbackend管理で完全制御ではない。
Energy impact: 未測定。
Correctness verification: 初期品質270/270合格、長文・短文SLO全件合格、全6worker正常終了。短文は各runで英語・日本語・简体中文各10件。warmup個別SLOは比較JSONに保持。一般品質・perplexity・コード・数学一般・長文理解・stress・8時間安定性は未評価。
Regression risk: 変更を標準採用しない。既存P3の5%以上改善・変動5%以内・run範囲分離等を緩めない。期間0の今回を過去30秒window全体と直接比較しない。
Keep / Revert: candidate256はReject、既定512を保持。runtime patchなし。証跡のみKeep。

Command: VLLM_APPLE_P1_EFFICIENCY=compact PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python scripts/qualify_gemma2_batch_mask.py --python /opt/homebrew/opt/vllm-metal/libexec/bin/python --model models/gemma-2-2b-it-4bit --output /private/tmp/recurring-prefill-C-2026-10-15-rN.json --port P --sustained-requests 30 --long-requests 12 --duration-seconds 0 --decode-concurrency 2 --prompt-concurrency 2 --prefill-step-size C --long-concurrency 1 --p1-profile。N=1..3、Cは512/256の交互順、P=19200+2*N+(C==256)。各run timeout240秒。

🟢 [Done] chunk縮小仮説を実測し不採用。🟠 [Next] 同workloadのGPU/host待ち切り分け。🔴 [Later] 広い品質・stress・長時間認定。⭕️ [Pending] GPU単独時間・電力collectorでの実測。

English: Alternating three independent runs per chunk size preserved all 270 initial arithmetic quality checks. Reducing prefill512 to256 did not produce reproducible improvement; long-workload median elapsed time worsened about0.8%. Retain512, adopt no runtime patch. General quality, power and long-run stability are unqualified.

简体中文：交替独立测试两种chunk各3次，270件初始算术质量全部通过。prefill512改为256未带来可复现改善，长文所需时间中位数约恶化0.8%。保持512，不采用runtime修改。通用质量、功耗及长期稳定性未认证。
