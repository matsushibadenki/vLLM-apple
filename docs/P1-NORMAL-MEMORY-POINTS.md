# P1 normal-workload memory points — 2026-10-09

## 日本語

🟢 [Done] P1 qualifierにwarmup／正常長文／正常短文の直後、fault開始前の固定測定点を追加。registry active／queued=0、HTTP active=1（この読取りだけ）を確認して保存する。benchmark起点PIDと観測PIDが一致することを検証し、worker交換・欠測・非整数PIDをsame_worker=falseとして扱う。workload SHA、品質／SLO件数、時刻、KV会計、allocator、RSSを保存。GPU計算／eval／synchronizeは追加しない。

反復比較toolはafter_shortの有効な同worker測定点を必須にし、古いfault後snapshotで補完しない。正常負荷後のRSSはtrendやunique KV物理容量ではなく、各値は非atomic。

| 測定点 | 保持entry | LRU会計 bytes | RSS bytes | allocator active bytes |
| --- | ---: | ---: | ---: | ---: |
| warmup後 | 3 | 7,774,208 | 1,452,605,440 | 1,535,844,894 |
| 長文後 | 1 | 221,192,192 | 1,496,793,088 | 1,967,710,750 |
| 短文後 | 3 | 7,774,208 | 918,503,424 | 1,592,877,602 |
| fault後（旧測定点） | 4 | 37,167,104 | 別scope | 別scope |

長文後のLRUは約211 MiBで、64 MiB予算には収まらない。旧fault後の約35.45 MiBだけを見て64 MiBで十分と判断しない。64 MiB試験timeoutの根因自体は未確定。この実機試験は既定4 entry／256 MiB、compact／prefill512。正常要求の品質42/42、SLO39/42（長文11/12、短文28/30）、warmup品質3/3・SLO1/3で不合格。全3点の同worker・workload一致、正常停止・実行中identity不変は確認。SLO未達を計測機能の合格と混同しない。

Problem: fault後のLRU値を正常workloadの保持量と比較していた。
Root cause: 測定pointのscopeが異なっていた。P1遅延・資源未達のruntime根因は未確定。
Evidence: 下記raw reportとmetadata監査。
Changed files: qualify_gemma2_batch_mask.py、compare_p1_cache_entries.py、対象tests。
Change: 3つのdrained固定point、PID／workload接続、古いpointへのfallback禁止。
Before: fault試験後のみ。After: warmup／長文／短文直後を別保存。
CPU impact: 既存resources集計を各pointで取得。改善率未測定。
GPU impact: 既存allocator読取りAPIを使用、追加compute／eval／syncなし。
Memory impact: reportあたり固定3点の数値metadata。RSS改善未認定。
I/O impact: 3点でHTTP読取りを追加（回収待ちretryは5秒上限）、request latency測定の外で行う。
Energy impact: 未測定。
Correctness verification: drain、PID不一致／型拒否、欠測拒否、workload一致、SLO失敗をそのまま保持。実機で全3点を取得。
Regression risk: 計測費用とタイミングの影響をゼロとはしない。標準API経路は変更せず、既存gateを緩めない。
Keep / Revert: scope修正をKeep。速度・メモリ最適化やP1長時間資格の認定ではない。

🟠 [Next] この測定点で長文KV保持とeval／allocation待ちを比較。既定4 entry／256 MiBを維持。長時間campaignは09:00 JST、定期更新は停止したまま。

## English

🟢 [Done] Added drained memory samples immediately after warmup, long and short normal workloads, before faults. Store workload hashes, quality/SLO counts, timestamps, LRU accounting, allocator and process counters. Require the benchmark's starting PID; worker changes and missing/invalid observations are not accepted. The repeat comparator requires the normal after-short point and never substitutes older post-fault samples. No added GPU compute/eval/synchronization; resource reads occur outside request measurement.

Real default-four/256 MiB verification obtained all three same-worker points with matching workload hashes. Long-workload retained KV accounting was 221,192,192 bytes, versus 7,774,208 after short workload and 37,167,104 after faults. A 64 MiB budget cannot retain that observed long-workload entry; this does not prove the earlier timeout's cause. Quality passed 42/42 but SLO 39/42, and warmup SLO 1/3; the inference qualification remains failed despite successful diagnostics and clean shutdown. 🟠 [Next] Compare allocation/eval waits with the fixed points. Default remains unchanged, long campaigns start at 09:00 JST, recurring updates remain stopped.

## 简体中文

🟢 [Done] 在warmup、正常长文、正常短文结束且请求回收后、fault前增加固定memory测量点，记录workload SHA、质量／SLO、时刻、LRU会计、allocator及process counter。要求与benchmark起点PID一致；worker变更或缺失／非法观察不能通过。比较tool必须使用正常短文后的point，不用旧fault后值补齐。不增加GPU compute／eval／同步，resources读取在request计测外。

实际默认4 entry／256 MiB取得全部3个同worker、workload一致的point。长文保持KV会计221,192,192 bytes，短文后7,774,208，fault后37,167,104。64 MiB不能容纳该长文entry，但不能由此认定旧timeout根因。质量42/42、SLO39/42、warmup SLO1/3，推理认证仍失败；计测成功及正常停止不代表资格通过。🟠 [Next] 用固定point比较allocation／eval等待。默认不变，长时间测试09:00 JST，定期更新保持停止。

## Evidence

- [Raw real-model report](evaluation/p1-normal-memory-points-m4-2026-10-09.json)
- [Metadata audit](evaluation/p1-normal-memory-audit-2026-10-09.json)

## Regression diagnosis / 回帰の切り分け / 回归诊断

初回全回帰はsubprocess起動timeout、daemon再起動接続、device contention判定の3件で失敗。単独16 testsは合格。再確認はbounded HTTPのBrokenPipeErrorのみ失敗した。
header deadlineでサーバーがtrickle接続を閉じた後にtest側が送信し得るため、testが期待されるBrokenPipe／ConnectionResetを許容するよう修正。runtime deadline、expiry2件、peak2、rejected1件、slot回収と200応答の検証は維持。変更対象4 tests合格。その他の失敗の原因はこの修正と同一と断定しない。
English: The first full run had three failures; the isolated 16-test recheck passed. A second full run hit expected socket closure during the header-deadline test. Handle only BrokenPipe/ConnectionReset in the test while retaining strict expiration, capacity and recovery assertions; four targeted tests passed. Do not attribute the other failures to this socket race.
简体中文：首次全回归3项失败，单独16 tests复查通过。第二次全回归在header deadline测试遇到预期socket关闭。test仅容许BrokenPipe／ConnectionReset，保持expiry、容量及恢复验证，4个目标tests通过。不能将其他失败归因于该socket竞态。

## Validation status / 最終検証状態 / 最终验证状态

対象3 tests、header deadlineを含む対象4 tests成功、Ruff・diff check成功。最終全回帰1484 tests（27 skip）は2 failures／4 errorsで未合格。subprocess起動timeout、daemon起動接続、device contention判定、generative adapter timeout／process-group PermissionErrorが残る。最初の16-test単独再確認成功を全回帰合格へ広げない。GPU／test worker不在を確認。
🟠 [Next] 起動・process-group回収の失敗を切り分け、全回帰を合格させる。今回のheader deadline修正後に当該BrokenPipe failureは再発していないが、他の失敗が解決したとは扱わない。
English: Feature/header tests and Ruff passed, but the final full run had two failures and four errors among 1484 tests (27 skipped). Startup, daemon, contention and generative-process cleanup problems remain. No full-regression certification. Workers were absent after validation.
简体中文：目标／header tests和Ruff通过，但最终1484 tests（27 skip）仍有2 failures／4 errors。起动、daemon、contention及generative process回收问题未解决，不能认证全回归通过。确认验证后worker不在。
[Validation receipt](evaluation/p1-normal-memory-validation-2026-10-09.json).
