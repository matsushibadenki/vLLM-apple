# P1 backend phase timing — 2026-10-08

## 日本語

[Done] sourceを固定したMLX-LM 0.32.0／MLX 0.32.1のP1 profileで4つのhost methodを計測。
`/vllm-apple/resources`の`backend_phases`へcall数、host wall／calling-thread CPU統計、例外数を保存する。
遅いsampleは各phase16件（計64件）に制限し、tensor／prompt／出力を参照・保存しない。
既存の結果・例外・同期をそのまま維持し、GPU同期を計測目的で追加しない。

| phase | 境界 |
| --- | --- |
| cache_fetch | LRUPromptCache.fetch_nearest_cache |
| prefill_prompt | PromptProcessingBatch.prompt（空batchの呼出も含む） |
| prefill_transition | PromptProcessingBatch.generate（内部でpromptを呼ぶ場合あり） |
| decode_next | GenerationBatch.next |

generate sourceは既存hashで検証し、cache sourceもSHA-256
`440709018cc528ee1e4e42e61ff8713ed2e0079566d9e8fa58eed3a92d334404`以外を拒否する。
配置済みupstream fileは変更せず、未確認versionを自動対応にしない。

[実モデルsmoke](evaluation/p1-backend-phases-smoke-m4-2026-10-08.json)：compact／prefill512、
長文12/12・短文30/30品質／SLO合格。warmupは品質3/3、SLO2/3。
collectorの修正済み長文／短文SLO gateに合格し、fault確認、正常停止、runtime/model identity不変を確認。

| phase | calls | wall平均 ms | wall p95上限 ms | wall最大 ms | thread CPU平均 ms | 250 ms以上 |
| --- | --- | --- | --- | --- | --- | --- |
| cache_fetch | 50 | 0.305 | 1 | 4.249 | 0.303 | 0 |
| prefill_prompt | 299 | 40.683 | 100 | 1824.601 | 3.202 | 8 |
| prefill_transition | 41 | 15.228 | 50 | 32.775 | 8.351 | 0 |
| decode_next | 282 | 15.080 | 50 | 62.340 | 5.582 | 0 |

この試験では長いmethod待ちはprefill_promptに現れた。r11の17件の根因を証明したものではない。
prefill内にはGPU実行、既存mx.eval同期、cache解放、padding等を含む。wall−thread CPUをGPU時間とみなさない。
空batchが含まれる平均・p50は実prefillだけの性能値ではない。p95は既存histogramのbucket上限で、正確な分位点ではない。
phaseが入れ子になる場合があり、平均／最大を足して要求時間としない。requestへの帰属・kernel timelineは未取得。
前回smokeからの遅延差を改善率として扱わず、長時間安定性・省電力は未認定。

[Done] 対象15 tests成功（1 skip）、全Python回帰1475 tests成功（11 skip）、Ruff・diff check成功。
最初のGPU実行申請はr11稼働中と判断され自動承認レビューに拒否された。
read-onlyでr11終了・runner不在・専用job回収・競合worker不在を確認後、同じ短時間試験の申請が承認された。

[Next] 非空prefillを区別し、GPU実行／同期／cache解放の時間を調べる。既存prefill設定候補を同じworkloadで比較する。
計測のE2E影響も独立比較し、品質・SLO・memoryを維持する候補だけ採用する。長時間campaignは09:00 JST開始。

## English

[Done] Added bounded host-method timing for cache fetch, prefill prompt, prefill-to-generation transition and decode.
Verified pinned generate/cache source; no upstream files or GPU synchronization were changed.
The short MLX smoke passed long 12/12 and short 30/30 quality/SLO, clean shutdown and unchanged identities;
warmup SLO remained 2/3. Prefill prompt had eight calls over 250 ms, maximum 1.825 s;
cache fetch/decode maxima were 4.249/62.340 ms. Calls include empty prefill batches and may nest.
These are shared host timings, not per-request attribution or GPU kernel times. Full regression: 1475 tests, 11 skipped; Ruff passed.
[Next] Separate nonempty prefill, synchronization/cache-release costs and compare existing prefill candidates plus instrumentation overhead.
Speed, energy and long-run stability remain unqualified.

## 简体中文

[Done] 为cache获取、prefill、prefill到生成转换及decode增加有界host method计时。
固定并检查generate／cache source，没有修改upstream文件或新增GPU同步。
实际短期MLX试验长文12/12、短文30/30质量及SLO通过，正常停止及identity不变；warmup SLO仍为2/3。
prefill有8次超过250 ms，最大1.825秒；cache／decode最大4.249／62.340 ms。
统计含空prefill batch且method可能嵌套，不代表请求归属或GPU kernel时间。
全回归1475项成功（跳过11项），Ruff通过。
[Next] 分离非空prefill、同步及cache释放成本，比较既有prefill候选和计测开销。速度、电能及长期稳定性尚未认证。

Follow-up / 続報 / 后续：[Nonempty prefill and candidate comparison](P1-PREFILL-COMPARISON-2026-10-08.md). The original report includes empty calls; later reports exclude them. Do not compare their phase means as a speed improvement.
