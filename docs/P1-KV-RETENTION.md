# P1 KV retention and rejected budget candidate — 2026-10-09

## 日本語

🟢 [Done] P1 resourcesにLRUPromptCacheの保持entry数、既存nbytes会計、max_bytesを追加。cacheが既に持つ数値を参照し、tensor走査・copy・GPU同期なし。P1初期化時のcache参照を利用。通常経路へ新しいcacheを作らない。snapshotは非atomicで、共有bufferのunique物理容量や生成中の全KV量を意味しない。

256 MiB設定の実機試験は長文12/12、短文30/30品質・SLO合格、warmup品質3/3・SLO2/3。正常停止・runtime/model identity不変。最終LRUは4 entry、50,692,096 bytes（48.34375 MiB）。上限256 MiBがそのまま256 MiBを保持するわけではない。長文の観測入力token再利用率91.06%、短文87.38%。allocator cacheとLRU KVは別の予算。

64 MiB候補は新workerで試験したが180秒のrunner timeoutで最終reportを生成できず、未認定。途中のrunning／passed=false checkpointはそのまま保持。SIGINTの試みはtimeout後に到着しrunner不在だったため、正常制御停止とは記録しない。workerも終了確認済み。原因、品質／SLOの最終失敗、memory削減を断定しない。

Problem: 実KV保持量が不明なまま、RSS増加をcache予算に結びつける恐れがあった。
Root cause: LRU数値会計をresourcesへ接続していなかった。r11未達と今回timeoutの根因は未確定。
Evidence: baseline raw report、未完了candidate checkpoint、別ファイルのtimeout監査。
Changed files: mlx_gemma2_compat.py、docs／証跡。
Change: 既存cacheの数値snapshotを公開。64 MiB候補のコード／optionは撤回。
Before: allocatorだけを参照。After: LRU保持KV会計を別表示。
CPU impact: 3種類のLRU queueのentry数と既存整数会計を参照。速度改善率未測定。
GPU impact: 追加呼出／同期なし。元kernelを変更しない。
Memory impact: backendが保持するcache参照のみ。RSS／KV削減は未認定。
I/O impact: resourcesに小さな数値を追加。
Energy impact: 未測定。
Correctness verification: baseline実モデルの数値取得、品質・SLO・正常停止・identity。warmup未達を明記。
Regression risk: P1限定、非atomicと対象範囲を明示。既存256 MiB gate／defaultを維持。
Keep / Revert: 保持量表示をKeep。64 MiB最適化候補をRevert。効果を確認できない設定を残さない。

🟠 [Next] 活動中KVとLRU保持を区別し、同条件でallocation／eval待ちを診断。今回baselineも以前の試験より遅いため、単回比較を変更効果と認定しない。長時間試験は09:00 JST。定期更新は停止したまま。

## English

🟢 [Done] P1 resources now expose retained LRU entries, existing byte accounting and the byte limit, without tensor traversal, copies or GPU synchronization. These non-atomic counters are neither unique physical bytes nor all active-generation KV.

The default 256 MiB trial passed 42/42 measured long/short quality/SLO checks, with clean shutdown and unchanged identities. Warmup quality was 3/3, SLO 2/3. Final LRU accounting was 4 entries, 48.34375 MiB. Observed token reuse was 91.06% for long and 87.38% for short prompts.

The experimental 64 MiB runner timed out at 180 seconds before final validation. Its running checkpoint remains intact; a separate timeout audit records runner/worker exit. A late SIGINT found no runner, so this was not a clean controlled stop. No final candidate quality, SLO, memory or root-cause claim. Revert the experimental budget option and retain default 256 MiB. Keep only the retention snapshot. 🟠 [Next] Separate active KV, retained LRU and allocation/eval waits. No speed or stability promotion; long campaigns start at 09:00 JST, recurring updates remain stopped.

## 简体中文

🟢 [Done] P1 resources增加LRU保持entry数、已有byte会计及上限，无tensor遍历、copy或GPU同步。非atomic数值不是unique物理容量，也不包含生成中的全部KV。

默认256 MiB测试长／短文42/42质量及SLO通过，正常停止、identity不变。warmup质量3/3、SLO2/3。最终LRU为4 entry、48.34375 MiB；长文token复用91.06%、短文87.38%。

64 MiB实验在180秒runner timeout时仍未完成最终验证，保留原running checkpoint并另存timeout审核。迟到的SIGINT没有找到runner，不能记录为正常控制停止；worker已退出。不认证候选最终质量、SLO、memory或根因。撤回64 MiB option，保持默认256 MiB，仅保留保持量snapshot。🟠 [Next] 区分活动KV、LRU保持及allocation／eval等待。未晋升速度／稳定性；长时间测试09:00 JST，定期更新保持停止。

## Evidence

- [256 MiB completed trial](evaluation/p1-kv-budget-256-m4-2026-10-09.json)
- [64 MiB unfinished checkpoint](evaluation/p1-kv-budget-64-m4-2026-10-09.json)
- [Timeout audit](evaluation/p1-kv-budget-64-timeout-2026-10-09.json)

検証 / Validation / 验证：全回帰1483 tests（11 skip）、Ruff・diff check成功。[Receipt](evaluation/p1-kv-retention-validation-2026-10-09.json).
