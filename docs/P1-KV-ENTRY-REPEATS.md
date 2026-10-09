# P1 KV entry repeat campaign — 2026-10-09

## 日本語

🟢 [Done] `compare_p1_cache_entries.py`を追加。順序4→3、3→4、4→3で各設定3 fresh workerを計画し、bounded runnerでserial実行する。新しいoutput directoryだけを使用し、raw reportとSHAを保持。全6回が完了・設定／runtime／runner／model identity一致のときだけ集計する。合格・性能・安定性の自動昇格なし。timeoutまたは非ゼロexitで後続を中止。

今回campaignは2回で停止。4 entryは品質42/42・SLO42/42、warmup品質3/3・SLO2/3。3 entryは品質42/42・SLO39/42（長文11/12、短文28/30）、warmup品質3/3・SLO0/3。両方正常停止・identity不変／比較間一致。runner／worker不在を確認した。残り4回を開始せず、6回分のmedianを捏造しない。

| 設定 | 短文E2E mean / max | 長文E2E mean / max | 最終LRU会計 |
| --- | ---: | ---: | ---: |
| 4 entry | 3618.002 / 5389.171 ms | 2834.529 / 7697.248 ms | 37,167,104 bytes |
| 3 entry | 2541.719 / 6620.293 ms | 1955.635 / 10320.011 ms | 34,717,696 bytes |

3 entryの平均は低いが最大値とSLOが悪化。平均だけで選ばない。4 entryも以前の平均424.798 msから大きく変動し、今回値を以前の値と混ぜて改善率を作らない。SLO未達をcache設定の因果と断定しない。
最終LRU sampleは故障試験後の値で、cache eviction／completion timingを含む。以前の31.5%差を固定負荷のKV削減率として認定しない。今回の2最終sampleも一般のmemory改善率ではない。まず固定正常要求直後の測定点を揃える。

Problem: 単回候補比較から採用判断できなかった。
Root cause: 速度・保持量の変動とSLO未達。P1未達のruntime根因は未確定。
Evidence: [campaign監査](evaluation/p1-kv-entry-repeats-m4-2026-10-09/audit.json)、raw report／failure_diagnostics。
Changed files: compare_p1_cache_entries.py、対象tests、docs。
Change: 順序交互・fresh worker・fail-closed集計・後続停止の再現可能な比較tool。
Before: 単回手動比較。After: 同identity反復、失敗で停止。
CPU impact: orchestration／offline集計のみ、改善率未測定。
GPU impact: 比較の既存推論のみ、kernel変更なし。
Memory impact: 有界trial数6、RSS／KV削減未認定。
I/O impact: 各raw report／logと監査を保存。
Energy impact: 未測定。
Correctness verification: incomplete／設定不一致／identity変更の拒否、非昇格、実campaignの失敗停止・shutdown。
Regression risk: 失敗や欠測からwinnerを作らない。3 entryは標準採用しない。
Keep / Revert: 比較toolをKeep、既定4維持。今回の3 entry採用はReject。

🟠 [Next] 正常負荷直後のKV会計と資源sampleを固定し、eval待ち・allocationを同条件診断。長時間試験09:00 JST、定期更新は停止したまま。

## English

🟢 [Done] Added a reproducible six-trial alternating fresh-worker comparator with identity validation, bounded cleanup and no auto-promotion. The campaign stopped after the second trial failed. Four entries passed 42/42 quality/SLO, with warmup SLO 2/3. Three entries passed quality 42/42 but SLO 39/42, with warmup SLO 0/3. Both shut down cleanly with unchanged and matching identities; four remaining trials did not start. No six-trial median or winner is reported. Means improved while maxima/SLO worsened, and baseline timing changed substantially from the earlier campaign. Final LRU samples follow fault tests, so they do not certify fixed-workload memory savings. 🟠 [Next] Fix the normal-workload memory sampling point before attribution. Default four remains; no speed/stability qualification.

## 简体中文

🟢 [Done] 增加6轮交替全新worker比较tool，验证identity、有界回收、不自动晋升。第2轮失败后停止：4 entry质量／SLO 42/42、warmup SLO2/3；3 entry质量42/42、SLO39/42、warmup SLO0/3。两者正常停止、identity不变且一致，剩余4轮未启动。未生成6轮median或winner。平均下降但最大延迟／SLO恶化，基线也与旧试验大幅变化。最终LRU sample位于fault试验后，不能认证固定workload的memory节省。🟠 [Next] 固定正常负荷后的memory测量点，再分析根因。保持默认4，未认证速度／稳定性。

Validation / 検証 / 验证：対象7 tests、現venv Python 3.12.9の全回帰1483 tests（27 skip）、offline Ruff・diff check成功。以前の環境1486 tests／11 skipと同じcoverageとは扱わない。[Receipt](evaluation/p1-kv-entry-repeats-m4-2026-10-09/validation.json).
