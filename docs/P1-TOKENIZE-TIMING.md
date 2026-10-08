# P1 tokenize timing and SLO gate — 2026-10-08

## 日本語

[Done] P1の単一process経路でtokenize開始／終了を要求traceへ記録。失敗時はtokenize_failedを残し、元のexceptionを維持。
sourceを固定したupstream CompletionRequestへtraceを一時的に関連付け、handler終了時に解除する。
distributed経路にはlockを持つtraceを渡さず、tokenize区間を欠測として扱う。
[区間集計ツール](../scripts/summarize_request_timings.py)は欠測をゼロにせず、負・非有限intervalを拒否する。

[実モデルsmoke](evaluation/p1-tokenize-timing-smoke-m4-2026-10-08.json)は品質warmup 3/3、長文12/12、短文30/30。
SLOはwarmup 2/3、長文8/12、短文30/30。正常停止・runtime/model identity不変を確認。
長文4件のSLO超過があり、速度・安定性は未認定。

[保持52件の区間集計](evaluation/p1-tokenize-timing-summary-2026-10-08.json)：faultを含む混合履歴で、workload別ではない。

| 区間 | samples | median ms | p95 ms | maximum ms |
| --- | --- | --- | --- | --- |
| queue作成→dequeue | 52 | 27.846 | 360.463 | 973.758 |
| dequeue→tokenize開始 | 51 | 0.019 | 0.088 | 2.260 |
| tokenize | 51 | 0.453 | 5.219 | 16.174 |
| tokenize終了→初回Response準備 | 48 | 722.295 | 12905.347 | 18566.587 |
| Response準備→HTTP受領 | 48 | 2.124 | 6.395 | 10.323 |
| 最初のSSE write | 50 | 0.059 | 0.333 | 1.520 |

p99もJSONへ保存し、sample不足の参考値と明記。区間ごとに欠測件数が異なるため、別集団の中央値を足し合わせない。
大きな遅延はこの試験ではtokenize以降に観測された。この区間にはcache処理、prefill、decode、同期、thread待ち等を含み、GPU時間とは断定しない。
前回smokeとの時間差を計測機能の効果や悪化率として認定しない。計測負荷の独立比較は未実施。

**判定の修正:** 今回のcollectorは長文SLO 8/12でもpassed=trueを返していた。
最終判定が初期の長文／短文benchmarkの完了件数と品質だけを確認し、SLOを確認していなかったため。
両benchmarkでrequests／completed／quality／SLO件数が予定件数と一致し、failed=0であることを必須にした。
欠測も拒否。既存の長時間window SLO gateは維持する。warmupは別表示で、今回もSLO2/3を明示する。
元のreportは書き換えず、新しい判定でこの長文結果が拒否されることを確認した。

[Done] 対象19 tests成功。全Python回帰1473 tests成功（11 skip）、Ruff・diff check成功。
全回帰の初回は既存contention profileの2 ms計測に依存する試験で1 error。
当該6 testsと全回帰の再実行は成功。今回の変更でその不安定性を解消したとは扱わない。
[検証receipt](evaluation/p1-tokenize-gate-validation-2026-10-08.json)に初回失敗も保存した。
[Next] tokenize後のcache処理／prefill／decode／同期を分離し、要求別identityをcollectorへ接続する。
同条件で計測負荷も比較し、支配時間の根因を確認してから最小修正を選ぶ。長時間campaignは09:00 JST開始。

## English

[Done] Added per-request tokenization start/end/error timing on the reviewed single-process route and a missing-aware phase summary.
Tokenization median was 0.453 ms; tokenization-end to first Response readiness median was 722.295 ms, maximum 18.57 s.
The latter includes backend work and waits; it is not GPU-only time. Retained traces include faults and are not grouped by workload.
Actual quality: warmup 3/3, long 12/12, short 30/30; SLO: 2/3, 8/12, 30/30 respectively.
Fixed a collector acceptance gap: initial long/short benchmarks must now pass all quality/SLO counts, complete every request and have zero failures.
The original report is preserved; its failing long result is rejected by the revised predicate.
Targeted 19 tests and full 1473-test regression passed (11 skipped in the full run), plus Ruff.
[Next] Split post-tokenization backend phases, connect collector identity and compare instrumentation overhead. No speed/stability qualification yet.

## 简体中文

[Done] 在已审核单process路径增加请求级tokenize开始／结束／异常计时，并提供不补零的区间汇总。
tokenize中位0.453 ms，tokenize结束到首个Response准备中位722.295 ms、最大18.57秒。
后者含backend计算和等待，不代表GPU专用时间；保留记录含fault，未按workload分组。
实际质量：warmup 3/3、长文12/12、短文30/30；SLO分别2/3、8/12、30/30。
修复collector判定遗漏：初始长文／短文必须全部质量及SLO通过，全部完成且failed=0，缺失证据拒绝。
保留原始report，新判定拒绝其失败的长文结果。19项针对性测试及1473项全回归成功（全回归跳过11项），Ruff通过。
[Next] 分离tokenize后的backend phase，连接collector identity并比较计测开销。速度及稳定性尚未认证。
