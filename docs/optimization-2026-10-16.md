# 定期評価 / Recurring evaluation / 定期评估 — 2026-10-16

Problem: 前回不採用のchunk縮小を繰り返さず、GPU/host待ちを切り分ける必要がある。
Root cause: 今回の低負荷条件を満たさない。Adobe IllustratorのCPU約98%を観測。過去の速度変動の根因とは断定しない。
Evidence: [10秒の負荷測定](evaluation/recurring-optimization-host-preflight-2026-10-16-r1.json)。既存wait_for_quiet gate failed、10 samples。専用推論／qualificationはprocess一覧に見つからず、旧r11はstatus helperでfailed。
Changed files: 本記録と負荷JSONのみ。既存差分・ROADMAP・runtimeを保持。
Change: 推論benchmarkとコード変更を次回へ延期。GPU-PROFILING-SELECTION.mdとexperiments.p3_mlx.profileの既存collectorを再確認。
Why it should improve performance: 今回は高速化なし。負荷混入を避け、次回の診断経路を明確化。
Before: 10/15の512/256各3回は改善なし、512保持。
After: 新たな候補を採用せず、既存Metal trace経路でのGPU/host診断を次工程とする。既存r3 GPU証拠を現在sourceの資格に流用しない。
CPU impact: counter読み取りのみ。改善率未測定。
GPU impact: 新規GPU処理なし。
Memory impact: 新規モデルloadなし。
I/O impact: 指示書・差分・roadmap・最新評価・collector文書の読み取りと証跡保存。
Energy impact: 未測定。
Correctness verification: 新規精度・速度測定なし。host gate失敗をモデル品質回帰として扱わない。
Regression risk: runtime変更なし。他アプリを停止・操作しない。
Keep / Revert: 証跡のみKeep。最適化patchなし。

Commands: ps -axo pid,etime,commandで競合確認。scripts/p1_stability_status.pyで旧r11確認。PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /opt/homebrew/opt/vllm-metal/libexec/bin/pythonからscripts.strata_measurement.wait_for_quiet(os.getpid(), timeout=10, required_samples=3)を実行。既存閾値（aggregate<=200%、単一process<=75%、新規終了0、連続3回）を維持。

🟢 [Done] 競合・host負荷と既存collectorの確認。🟠 [Next] 低負荷時にexperiments.p3_mlx.profileで現在identityのGPU/host計測を実行。既存Metal collectorは実装済みなので、collector不足を理由とした[Pending]にはしない。🔴 [Later] 広い品質・stress・長時間認定。⭕️ [Pending] 未保有SoC／独立Macの実測。

English: The ten-second host-load gate failed while Adobe Illustrator used about98% of one CPU core. Defer inference measurements and code changes; retain evidence only. The existing Metal trace collector is implemented and is the next diagnostic route once the host is quiet. No new quality or speed results; no attribution of earlier variability.

简体中文：10秒host负载gate未通过，Adobe Illustrator占用约98%的单核CPU。推迟推论测量及代码修改，仅保留证据。已有Metal trace collector已实现，下一步在低负载时使用该路径诊断。不新增质量或速度结论，不认定此前波动原因。
