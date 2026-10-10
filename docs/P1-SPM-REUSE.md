# P1 response latency and RSS: immutable SPM table reuse — 2026-10-10

## 日本語

🟢 [Done] 要求ごとのSPM変換表再構築を除去した。配置済みMLX-LM 0.32.0の`TokenizerWrapper.detokenizer`は要求ごとに新しいstreamを作り、そのconstructorが256,000語の変換表を作り直す。実tokenizerの15反復ではconstructor中央値121.386 ms、一時allocation peak 42.18 MiBだった。これは要求ごとのCPU費用とallocation churnの直接原因であり、過去r11の全遅延の唯一の原因と断定しない。

`vllm_apple/spm_tokenmap_reuse.py`はtokenizerごとにimmutable tupleを一度保持する。tokens/text/offset/未完成UTF-8 buffer/trim_spaceは各要求専用のまま。全256,000項目、逐次byte出力、finalizeの一致を実tokenizerで検証した。warm constructorは中央値0.958 µs、計測peak 272 bytes。cold初回の表構築費用は残る。これはconstructorの計測であり、推論全体の改善率ではない。

適用範囲はsource checkoutのP1 workerだけ。既定on、比較時のみ`VLLM_APPLE_P1_SPM_REUSE=off`。native tokenizer source SHA256 `c9eea380fd7e1a624f8873a2d2298fd75283285750f74a9348b655659017ee09`以外は拒否する。MLX 0.32.1、MLX-LM 0.32.0、M4/32 GiB、Gemma 2 2B 4-bitで検証。MLX-LMの[MIT license本文](licenses/MLX-LM-MIT.txt)を保持し、Apple Inc.の帰属をsourceに記載。installed packageや依存versionは変更していない。tokenizer vocabularyは不変というP1契約であり、変更時は新しいtokenizerを作る。既存wheelや別modelへ認定を拡大しない。

[6 fresh workerの比較](evaluation/p1-spm-reuse-m4-2026-10-10/comparison.json)はoff/on/on/off→off/on、各8 cycleで短文12件(concurrency 2)、active cancel、長文3件(concurrency 1)。compact/prefill512、prompt cache 4 entries/256 MiB、同workloadと同修正codeを使用。正常要求720/720件で品質・SLO合格、全worker正常停止。warmupとprime要求はこの件数に含めない。

| 計測範囲 | off (3 workers) | on (3 workers) |
| --- | --- | --- |
| 短文平均E2E（各worker同数要求の平均） | 451.115 ms | 209.279 ms |
| 長文平均E2E | 1823.564 ms | 2112.188 ms |
| 反復後半4点のRSS増加 | 8.000–9.016 MiB | 0.016–0.609 MiB |
| 後半RSS slope | 1017–1530 MB/hour | −7.25–91.69 MB/hour |
| 16 MiB/hour plateau判定 | 0/3 | 1/3 |

短文平均はこの条件で53.61%小さい。長文速度改善は確認できず、長文を含めた全用途の高速化は認定しない。RSS増加は減少したが、短時間slope判定はonでも2/3未達。8点の外挿を30分/8時間の資格にしない。電力は未測定。

🟢 [Done] HTTP header Timerのcallbackがhandlerを循環参照していたため、thread終了後にTimer参照を外す。GCを止めた200 HTTP要求では残存handler 200→0、traced live allocation 1,528,900→46,155 bytes。通常GC下の推論RSS改善率とは別の計測。deadlineや回収基準は維持。

KV aggregate予算の分離案と長文優先保持案は実測で採用せず、実装optionも削除した。GPU traceは計測負荷があり、CPU→GPU待ちの観測だけで根因を断定しない。vmmapの異常な未割当値からswap原因を推定しない。元のraw reportを保持する。

🟢 [Done] [修正後90秒report](evaluation/p1-spm-final-m4-2026-10-10.json)：通常負荷420/420品質・SLO、active cancel 41/41、slow consumer 5/5、queued cancel 4/4、timeout 4/4。全epoch資源、awake、runtime/model identity、正常停止は合格。RSS後半slope −19.84 MB/hour、後半増加 −311,296 bytes。worker crashは0件で未検証。warmup 3/3、初期長文12/12・短文30/30も合格。認定scopeは90秒に限定。全回帰1491 tests成功（31 skip）、Ruff成功。

🟠 [Next] 修正後の30分→全gate合格時のみ8時間試験で最終判定する。開始は2026-10-11 09:00 JSTの単発予約。30分でSLOが合格でも全epochのRSS／allocator／thread／FD／registry基準が未達なら8時間を開始しないよう、runnerにも進行gateを追加した。定期更新は復活させない。稼働中はruntime編集、GPU/CPU benchmark、回帰、依存更新を避ける。RSS/SLO/品質/全epoch資源/awake/identity/shutdownの証拠が揃うまでP1完了としない。awake qualificationを実sleep/wakeやdaemon全体へ拡張しない。

## English

🟢 [Done] The pinned MLX-LM SPM constructor rebuilt a 256,000-entry vocabulary table per request: median 121.386 ms and 42.18 MiB transient allocation peak. Reuse an immutable table owned by each tokenizer, while keeping all streaming state separate. All vocabulary entries and incremental output match; warmed construction takes 0.958 µs with a 272-byte measured peak. Cold construction remains. The patch is restricted to source-pinned P1; unknown source hashes fail closed, and dependencies are unchanged.

Six fresh-worker trials pass quality/SLO for 720/720 normal requests. Short mean E2E changes from 451.115 to 209.279 ms. Long mean E2E does not improve (1823.564→2112.188 ms). Recent RSS growth changes from 8.000–9.016 to 0.016–0.609 MiB, but two of three enabled trials still fail the short-run RSS slope gate. This is neither long-run qualification nor a power measurement. Also break the finished HTTP Timer/handler cycle: 200 retained handlers become zero with cyclic GC disabled.

The final regression suite passes 1491 tests (31 skipped); the 90-second check passes 420/420 mixed normal requests plus resource/awake/identity/shutdown checks. The runner now refuses the eight-hour stage when the thirty-minute resource gates fail.

🟠 [Next] A one-off campaign starts October 11 at 09:00 JST: 30 minutes, then eight hours only on a full pass. Keep all existing gates and evidence; do not certify P1 before final quality/SLO/resources/awake/identity/shutdown audit. Recurring updates remain stopped.

## 简体中文

🟢 [Done] 固定版本MLX-LM的SPM constructor为每次请求重建256,000项词表：中位数121.386 ms、临时allocation峰值42.18 MiB。现在每个tokenizer只保留一份immutable词表，每次请求的tokens/text/offset/UTF-8 buffer仍独立。全部词表项及逐步输出一致；warm初始化0.958 µs、计测峰值272 bytes，cold首次构建仍保留。仅用于source-pinned P1，未知source hash拒绝运行，未修改依赖。

6个fresh worker共720/720正常请求通过质量/SLO。短文平均E2E从451.115降至209.279 ms；长文未改善（1823.564→2112.188 ms）。反复后半RSS增加从8.000–9.016降至0.016–0.609 MiB，但启用后的3轮仍有2轮不满足短期RSS斜率gate。不能认定长期稳定性或功耗。另修复HTTP Timer/handler循环引用；关闭循环GC的试验中残留handler 200→0。

最终全量回归1491项通过（31 skip）；90秒混合正常负载420/420及资源/awake/identity/shutdown通过。30分钟资源gate未通过时，runner拒绝启动8小时。

🟠 [Next] 10月11日09:00 JST单次启动30分钟试验，全部gate通过后才进入8小时。最终审核质量/SLO、各worker epoch资源、awake、identity和shutdown之前，不将P1标记完成；不恢复定期更新。
