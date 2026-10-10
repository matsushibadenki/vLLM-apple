# P2 continuous batching / actual KV reuse — 2026-10-10

## 2026-10-10 update / 更新 / 更新

🟢 [Done] [数値差と並列応答の改善](P2-OPTIMIZATION-2026-10-10.md)：限定float32 profileで16/16 KV数値条件に合格。3 fresh workersずつの同精度比較でc4 goodput 12.440→44.097 tokens/s、改善側c4品質・SLO 180/180。prefixなしc1を別の3 workersで検証し、TPOT p95 13.816–13.994 msは5%悪化上限14.025 ms以内。固定workloadのROADMAP条件を満たす。P2専用既定はfloat32／50 ms budget／SPM・immutable RMSNorm共有、dtypeをcache identityへ含める。下記10月4日の数値と失敗記録は履歴として保持する。

🟠 [Next] cache有効c1の最悪p95 gateは未達。float16対照ではTPOTが悪化し、標準経路のreplacementは未認定。一般chat／coding／Agent、長context・SWA、cancel・fairnessと標準採用は引き続き[Next]。同token数のfloat32 KV保存幅はfloat16の2倍で、RSS・省電力改善は未認定。

English: 🟢 [Done] The fixed float32 workload passes 16 numerical cases, 3.545× c4 goodput and the unchanged uncached-c1 5% p95 gate. 🟠 [Next] Cache-enabled c1 tails, float16 replacement, broader workload/SWA/cancel/fairness and standard promotion remain unqualified. See the linked report; October 4 results below are retained history.

简体中文：🟢 [Done] 固定float32 workload通过16项数值测试，并发4 goodput提升3.545倍，且不使用prefix复用的c1符合原5% p95限制。🟠 [Next] cache启用c1尾延迟、float16替换、广泛workload／SWA／取消／公平性及标准晋升仍未认证。详见链接报告；下列10月4日结果保留为历史记录。

## 2026-10-04 implementation and evidence / 実装・証拠の履歴

## 日本語

[Done] `experiments.p2_mlx.server`に実験用backendを実装した。MLX-LM既存のtoken scheduler、chunked prefill、実LRU prompt cacheを再利用する。HTTP requestを直列化するadapterではなく、backendの`BatchGenerator.next()`が同一stepに返したUID数で同時decodeを確認する。標準daemon／P1 runtimeは変更していない。

対象は配置済みGemma 2 2B 4-bit、MLX 0.32.1、MLX-LM 0.32.0。Gemma 2／server／scheduler／cacheの確認済みsource hashに限定する。model、tokenizer、template関連ファイルの内容hash、template policy、position origin 0、process単位のcache saltからcache namespaceを作る。別model・adapter・draftは拒否する。request単位のsalt切替は未対応で400とし、既存cacheへ混在させない。

native cacheのexact token prefixを使用し、返された残りtokensが元入力の正確なsuffixであることを確認する。backendのdeepcopyによるcache分離、prefix pruning／容量eviction、trimを利用する。近似prompt類似度でKVを流用しない。cache内の大きなtensorはbackendに保持する。

上限：context（prompt＋最大出力）4,096 tokens、output 512 tokens、decode concurrency 4、prompt concurrency 2、prefill step 512、LRU 256 MiB、GPU allocator 8 GiB・allocator cache 256 MiB、HTTP接続16本。これらは実験profileの上限で、最大contextと最大並列度の同時認定ではない。P1 profileの資格を継承しない。

`/vllm-apple/p2`は実decode step数・複数sequence step数・最大幅、prefill提出tokens、実cache hit／miss／再利用tokens、論理cache容量・entry数・除去entry数を公開する。除去理由はprefix pruningと容量evictionを分離できず、その限界を明記する。snapshotはnon-atomic。prefill提出数はmodelに渡す予定のtokensであり、GPU kernel実行時間や実行済みtokensの完全な計数ではない。

### 実機結果

- [HTTP probe](evaluation/p2-mlx-http-m4-2026-10-04-r2.json)：最大同時decode幅4、複数sequenceのdecode step 39回。prefix末尾を3種類に編集したfirst passと再入場で、prefill提出数が545 tokens減少した。最初のpass内でも共通prefixを再利用しており、純粋なcold baselineではない。
- first pass 3/3、再入場3/3、三言語c4 30/30が品質合格。c4のSLOは28/30で未達。機能検証は合格だが総合gateは不合格、性能優位は未認定。warmup 3件は別集計。sample不足のためp99は参考値。
- [最終sourceのHTTP確認](evaluation/p2-mlx-http-m4-2026-10-04-r3.json)でもdecode幅4・prefill提出削減545・品質36/36を確認したが、c4 SLOは22/30だった。loopback、logprob出力量、request template変更の上限を追加した現コードでも性能gateは未達。終了時backend exit 0を確認した。
- [実model KV数値対照](evaluation/p2-real-kv-parity-m4-2026-10-04-r2.json)：26層の実KVCache／float16で19-token prefixを再利用した。同じ分割をfresh KVで再計算した対照とlogit差0、branchコピー後も保存prefix offset 19、trim後missを確認した。
- 一方、全promptを一度に計算する対照とは最大logit差0.0390625／0.03125で、事前の`atol=rtol=1e-3`基準を満たさなかった。argmaxは双方一致したが、数値gateを緩めず未認定として保存する。SWA長context・wraparound・hybrid stateの正確性へ外挿しない。
- [初回失敗](evaluation/p2-mlx-http-failed-m4-2026-10-04.json)はwrapperの`__len__`不足でbackend owner threadが終了した。契約を修正し、失敗を保存した。CPU testはidentity分離、exact prefix、budget release、cache無効時のmissを確認する。

### 再現

repository rootから、ほかのGPU試験が停止していることを確認して実行する。

```sh
PYTHONPATH=. /opt/homebrew/opt/vllm-metal/libexec/bin/python \
  -m experiments.p2_mlx.server --model models/gemma-2-2b-it-4bit \
  --host 127.0.0.1 --port 19148 --decode-concurrency 4 \
  --prompt-concurrency 2 --prefill-step-size 512 --prompt-cache-size 4
PYTHONPATH=. .venv/bin/python scripts/probe_p2_mlx.py \
  --model models/gemma-2-2b-it-4bit --output /tmp/p2-new-http-report.json
```

cache無効baselineは別のfresh processで`--disable-prefix-cache`を追加する。saltによるprocess単位の分離には`--cache-salt`を指定する。HTTP backendを終了してから数値対照を実行する。

```sh
PYTHONPATH=. /opt/homebrew/opt/vllm-metal/libexec/bin/python \
  -m experiments.p2_mlx.probe_kv --model models/gemma-2-2b-it-4bit \
  --output /tmp/p2-new-kv-report.json
```

[Next] cache無効c1／c4と有効profileの独立3 runでp95／goodputを比較し、c1悪化5%以内・c4改善20%以上のgateを確認する。現在のSLO未達とfull-versus-segmented数値差を調査し、長context、cancel後のcache保存、session分離、Agent再入場・fairness／starvationを検証する。実装はopt-inに留め、P2全体は[Next]。未保有SoC／メモリ環境は[Later][pending]。

全CPU回帰1,411 tests合格（11 skip）、変更PythonのRuff・diff検査合格。実model数値試験とHTTP試験は別々に実行し、試験backendは回収する。

## English

[Done] An isolated experimental backend now reuses the native MLX token scheduler, chunked prefill and real KV prompt cache. Cache identity binds artifact/tokenizer/template content, template policy, position origin and process-scoped salt. Unsupported request-scoped salts, other models, adapters and drafts are rejected. The production/P1 routes are unchanged.

The M4 HTTP probe observed actual decode width four and 39 multi-sequence steps. Re-entry reduced submitted prefill by 545 tokens. All 36 measured requests passed quality; concurrency-four latency passed only 28/30, so the overall gate failed. The first pass already shares prefixes and is not a cold-cache baseline. Counters describe submissions, not GPU execution timings.

The final-source repeat passed 36/36 quality requests but only 22/30 concurrency-four SLO requests; the backend exited cleanly. Full CPU regression passed 1,411 tests with 11 skips.

Real float16 KV copies matched a fresh segmented reference exactly and retained the original prefix offsets after branching. Eviction produced a miss. Full-prompt computation differed by up to 0.0390625 and failed the predefined numeric tolerance despite matching argmax. Numeric and performance qualification remain false; no tolerance was relaxed.

[Next] Independent cache-off/on performance runs, numerical investigation, long-context/SWA behavior, cancellation, session isolation and agent fairness remain required. P2 stays experimental and [Next]. Additional hardware is [Later][pending].

## 简体中文

[Done] 独立实验后端已接入MLX原生token scheduler、chunked prefill和真实KV prompt cache。cache identity包含模型／tokenizer／template内容hash、template policy、position origin及process级salt。request级salt、其他model、adapter和draft会被拒绝，标准／P1路径未改动。

M4 HTTP测试观察到真实decode宽度4及39次多sequence step。再入场减少545个prefill提交tokens。36个测量请求均通过质量，但c4延迟仅28/30通过，总gate不合格。首次pass内部已复用公共prefix，不是cold-cache基准。计数表示提交量，不是GPU实际执行时间。

最终source的再次验证质量36/36通过，但c4 SLO仅22/30通过；backend正常退出。全CPU回归1,411 tests通过、11 skip。

真实float16 KV副本与同样分段的fresh参考logit差为0，分支后原prefix offset未变，eviction后为miss。与一次计算完整prompt的参考最大差为0.0390625，虽然argmax一致，仍未通过预先设定的数值容差。未放宽gate，不认证数值或性能。

[Next] 继续独立cache-off／on性能对比、数值差调查、长context／SWA、取消、session隔离及Agent公平性测试。P2保持实验性[Next]；未持有硬件为[Later][pending]。
