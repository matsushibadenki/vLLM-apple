# P1 workload activity windows — 2026-10-09

## 日本語

[Done] 同じbenchmark windowの前後に既に取得しているresourcesから、process累積counterの差分を保存する。追加HTTP／OS取得／GPU同期は不要。fault、page-in、copy-on-write、context switchとRSS差分を最大64 windowに保持し、workload SHA、cycle、品質／SLO件数を添える。

同PID・Darwin source一致・整数の非負counterを要求し、PID変更、counter減少／overflow、欠測をavailable=falseとして拒否する。RSSの減少は正当な負値として保持。counterリセットをゼロ活動と解釈しない。異なるworkloadをまとめて同条件比較しない。snapshotは非atomic・process全体なので、CPU/GPU待ちやmemory pressureの根因とは認定しない。

Problem: 前回の累積counter一時点だけではworkload中の活動を比較できなかった。
Root cause: 診断collectorが前後snapshotを差分へ接続していなかった。r11遅延／RSS未達の原因は未確定。
Changed files: process_activity_delta.py、qualify_gemma2_batch_mask.py、対象tests。
Change: 既存の前後resourcesを再利用して差分を計算し、有界履歴へ記録。
Before: 累積counterのみ。After: 同PIDのworkload window差分。
CPU impact: 定数個のcounter比較と辞書作成。改善率は未測定。
GPU impact: 呼出・同期・kernel変更なし。
Memory impact: 最大64件の数値履歴、RSS減少の改善は未認定。
I/O impact: 新しいHTTP／OS callなし。既存reportへ数値を追加。
Energy impact: 未測定。
Regression risk: counter差分は診断用途、合格gateを変更しない。古いreportの欠測を補わない。
Keep / Revert: 診断としてKeep。推論速度・P1長時間資格ではない。

[Next] 同workloadで繰り返し比較し、eval遅延とRSS増加の説明に使う。長時間campaignは09:00 JST開始。定期更新はユーザーの指定で停止したまま。

## English

[Done] Reuse existing before/after benchmark resource snapshots to retain bounded process-activity deltas with workload hashes, cycle and quality/SLO counts. No extra HTTP, OS sampling or GPU synchronization. At most 64 windows are retained. Require the same valid PID, Darwin source and nonnegative integer counters; missing samples, resets and overflow remain unavailable. Legitimate RSS decreases are preserved. Process-wide, non-atomic snapshots do not establish causality. [Next] Repeat identical workloads and correlate eval waits/RSS. Long campaigns start at 09:00 JST; recurring updates remain stopped.

## 简体中文

[Done] 复用benchmark前后现有resources，保存最多64个process活动差分window，并记录workload SHA、cycle、质量／SLO次数。不增加HTTP、OS采样或GPU同步。要求同一有效PID、Darwin source及非负整数counter；缺失、reset或overflow明确不可用。保留合法RSS负差分。非atomic、process整体snapshot不能认证根因。[Next] 重复相同workload，对照eval等待及RSS。长时间测试09:00 JST开始，定期更新保持停止。

## 実機結果 / Real-model results / 实际结果

M4／Gemma 2B／compact／prefill512、30秒の診断負荷10 window、102/102品質・SLO合格。初期warmup3・長文12・短文30も全件合格。正常停止、runtime/model identity不変。10 windowの差分はすべてavailable=true。
短文workloadは8 window（各12要求）、page-in増分すべて0、fault増分中央値1920、RSS差分+409,600～+9,977,856 bytes。長文workloadは2 window（各3要求）、page-in増分0、fault中央値145、RSS差分−1,228,800～+671,744 bytes。異なるworkloadは別集計。
今回のwindow内ではpage-inを伴わないRSS増加を観測した。r11原因、cache再allocation、memory leak、不在証明、plateauを断定しない。30秒診断を30分／8時間資格にしない。

English: Ten 30-second diagnostic workload windows passed all 102 quality/SLO checks, plus all 45 initial checks; clean shutdown and unchanged identities. All deltas were available. Page-in deltas were zero in every window; short-workload RSS deltas were positive. This does not identify allocations/leaks or establish r11 causality/stability.
简体中文：30秒诊断包含10 window，102/102质量／SLO通过，初始45件也全部通过；正常停止及identity不变。全部差分可用，page-in增量均为0，短文workload RSS差分为正。不能因此认定allocation／leak或r11原因／长期稳定性。

- [Raw report](evaluation/p1-activity-windows-m4-2026-10-09.json)
- [Grouped audit](evaluation/p1-activity-windows-audit-2026-10-09.json)

Correctness verification / 検証 / 验证：対象2 tests、全回帰1482 tests（11 skip）、Ruff・diff check成功。[Receipt](evaluation/p1-activity-windows-validation-2026-10-09.json).
