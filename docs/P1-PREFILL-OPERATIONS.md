# P1 prefill operation timing — 2026-10-08

## 日本語

[Done] source検証済みの隔離P1 workerで、generate module内の既存`mx.eval`／`mx.clear_cache`をprefill呼出中だけ計測する。generate専用proxyとthread-local contextを使い、元MLX moduleを変更しない。引数・戻り値・例外・呼出回数を維持し、GPU同期／eval／解放を追加しない。通常profileなしの経路は変更しない。履歴は各16件、本文やtensorを保存しない。

M4／Gemma-2-2b-it-4bit／compact／prefill512の短時間検証はwarmup 3/3、長文12/12、短文30/30の品質・SLO合格。正常停止・runtime/model identity不変。全回帰1479 tests（11 skip）、Ruff・diff check成功。

| 対象 | 呼出数 | 平均wall | p95上限bucket | 最大wall |
| --- | ---: | ---: | ---: | ---: |
| prefill eval | 24 | 183.607 ms | 1000 ms | 829.155 ms |
| prefill clear_cache | 24 | 0.642 ms | 1 ms | 1.007 ms |

保持された遅いprefill 4件（738.243／766.618／803.409／830.523 ms）の内部に、eval 736.894／765.151／801.916／829.155 msがそれぞれ完全に含まれる。今回の遅い4件ではeval待ちが主な区間。cache解放は主要な遅延区間ではないため削減案を採用しない。これはhost callの時間照合であり、GPU kernel時間やr11の根因ではない。wall−thread CPUをGPU時間としない。nested計測を合算しない。

Problem: 非空prefillのどの既存operationで待つか不明だった。
Root cause: r11の原因は未確定。今回の遅いprefillは既存eval区間がほぼ全時間を占めた。
Evidence: 下記raw report／時刻照合／合成overhead。
Changed files: `mlx_operation_timing.py`、`mlx_gemma2_compat.py`、対象test。
Change: bounded・thread-local・P1限定のoperation計測。
Why: 未計測のcache解放削減を避け、次の改善対象を絞る。
Before: prefill全体のみ。After: evalとclear_cacheを分離。
CPU impact: no-op eval＋clear_cacheを含む合成10,000 prompt呼出×9回の中央値0.625→21.250 ms、約2.06 µs/callの追加wall。CPU0.626→21.251 ms。p95／p99（9回のnearest rank）は最大値で、wall0.683→21.541 ms。推論遅延率ではない。
GPU impact: 呼出や同期の追加なし。GPU実時間の改善未測定。
Memory impact: 固定上限の数値履歴とcontext。RSS改善未認定。
I/O impact: 既存resources snapshotに数値統計を追加、本文／tensorなし。
Energy impact: 未測定。
Correctness: 引数、結果、例外後context復元、別thread除外、idempotent、実モデル45/45と全回帰。
Regression risk: generate moduleのproxy参照に追加CPU費用。source固定、P1 profile限定、標準経路へ昇格しない。
Keep / Revert: 診断としてKeep。速度改善や長時間資格として扱わない。

[Next] eval待ち内のGPU実行・host scheduling・memory pressureを適切なprofileで分離し、RSS資源未達も調査する。長時間campaignは09:00 JST開始。
[Pending] 実電力samplerまたは外部電力計によるjoule/request。

## English

[Done] Isolated, source-verified P1 workers now time existing prefill eval and clear_cache calls through a generate-module proxy and thread-local context. The original MLX module is untouched; no GPU calls or synchronization are added. Results, arguments, exceptions and call counts are preserved. Each operation retains at most 16 numeric slow samples.

Real M4/Gemma 2B compact/prefill512 checks passed 45/45 quality/SLO, clean shutdown and unchanged identities. All 1479 regression tests passed with 11 skips. In four retained slow prefill windows, eval occupied nearly the entire 738–831 ms window; clear_cache peaked at 1.007 ms. This identifies the observed interval, not GPU-only time or r11 causality. Keep profiling and decline speculative cache-release removal. Synthetic instrumentation cost was about 2.06 microseconds per no-op prompt call; inference speed, RSS and energy savings remain unqualified. [Next] Separate GPU execution, scheduling and memory pressure within eval waits. Long campaigns start at 09:00 JST. [Pending] Electrical energy measurement.

## 简体中文

[Done] 在source验证后的独立P1 worker中，通过generate专用proxy及thread-local context，分离现有prefill eval和clear_cache计时。不修改原MLX module，不添加GPU调用或同步；保留参数、结果、异常及调用次数。每项最多保留16个慢调用数值记录。

实际M4/Gemma 2B、compact/prefill512测试45/45质量与SLO通过，正常停止及identity不变。全回归1479 tests通过，11 skip。4个慢prefill窗口738–831 ms几乎全部位于eval内；clear_cache最大1.007 ms。仅定位本次host区间，不认证GPU时间或r11根因，暂不采用删除cache释放。合成计时额外开销约2.06 µs/空操作prompt调用；推理速度、RSS及功耗未认证。[Next] 分离eval中的GPU执行、调度及内存压力；长时间测试09:00 JST开始。[Pending] 实际电能测量。

## Evidence

- [Raw real-model report](evaluation/p1-prefill-operations-m4-2026-10-08.json)
- [Nested-window audit](evaluation/p1-prefill-operations-audit-2026-10-08.json)
- [Instrumentation overhead](evaluation/p1-prefill-operation-overhead-2026-10-08.json)
- [Validation receipt](evaluation/p1-prefill-operations-validation-2026-10-08.json)
