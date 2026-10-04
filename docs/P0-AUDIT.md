# P0 audit — 2026-10-04

## 日本語

[Done] 手元のApple M4／32 GiBで、同じGemma 2 2B 4-bit artifactを使うMLX-LMとvLLM-Metalの基準測定と実行経路監査を完了した。これは比較の出発点であり、性能優位、全モデル対応、batching、持続運用の認定ではない。

各backendを独立したprocessで3回起動し、各回warmup 3件＋測定102件、concurrency 1、temperature 0、出力上限16 tokensで実行した。その他のsampling設定はbackendの既定値で、同一sampling実装とは認定しない。順序はMetal→MLX→MLX→Metal→Metal→MLX。英語・日本語・简体中文の固定算術問題を各backendでそれぞれ102件、合計306件測定し、全612件が品質・SLO gateに合格した。SLOはclient TTFT 1,000 ms、E2E 5,000 ms。一般的なchat、coding、tool-use品質へは外挿しない。

| Backend | 実行version | goodput中央値 tokens/s | 3 runの相対幅 |
| --- | --- | ---: | ---: |
| MLX-LM direct | mlx-lm 0.32.0／mlx 0.32.1 | 18.3411 | 1.76% |
| vLLM-Metal direct | vllm-metal 0.29.0／vllm 0.29.0+cpu | 25.7337 | 6.19% |

Metalの相対幅は既存の性能比較許容値5%を超える。速度差を安定した改善として認定しない。p99は各回102件で1,000未満のため参考値。processはfreshだがOS file cacheを消去しておらず、warmup後のbackend管理prefix cacheを使用する。cold model load比較ではない。温度・電源・load／メモリ観測はreportに保存し、変動原因の因果関係は未確定。

### 実行経路と能力の監査

起動時にbackendがmodel／KVを所有する。MLX-LMは`mlx_lm.server.ModelProvider`から`ResponseGenerator`へ生成を渡し、HTTP handlerがSSEを返す。wrapperは同じbackendへPOSTを直列に渡し、admission／取消観測／telemetryを追加する。daemonは`BackendProcess`を介してHTTPをproxyする。vLLMは`OpenAIServingChat.create_chat_completion`→engine client `generate`→V1 scheduler→`MetalWorker.execute_model`→`MetalModelRunner.execute_model`を通り、token結果をOpenAI APIへ返す。model tensor／KVの所有者はbackendであり、control planeのqueueはtoken schedulerではない。

| 能力 | MLX-LM direct | MLX wrapper／daemon | vLLM-Metal direct |
| --- | --- | --- | --- |
| Streaming・usage | 実機合格：今回306件 | wrapper実機合格：既存限定試験。daemon旧buildの比較証跡あり | 実機合格：今回306件 |
| Cancel | 今回の基準試験では未検証 | wrapper待機取消・stream切断後の枠回収は実機合格。GPU停止latency未検証 | 未検証 |
| Prefix reuse | backend報告cached tokensを各回1,972/2,074取得。KV数値正確性未認定 | usage／容量観測あり。hit・eviction完全性未認定 | cached usage欠測。起動flagだけでは認定しない |
| Continuous batching | Gemma 2の過去c2試験は失敗。未認定 | adapterによる直列処理のみ。backend batchingではない | 今回c1のみ。未認定 |
| Chunked prefill | 未認定 | 未認定 | 起動logに有効表示あり。実動作・品質は未認定 |
| KV precision | 今回の生成KV dtypeは未取得 | 容量からdtypeを推測しない | BF16設定logはKV dtypeの証明ではない |
| Structured output | 未検証 | 未検証 | 未検証 |

llama.cpp／vllm-mlxは今回の環境では実行せず、比較対象の候補として残す。未検証と未対応は区別し、能力flagから実機合格へ自動昇格しない。wrapperの[取消証跡](evaluation/wrapper-active-cancel-m4-2026-10-03.json)、[待機取消](evaluation/wrapper-queue-cancel-m4-2026-10-03.json)、[c2直列処理](evaluation/text-benchmark-c2-serialized-memory-m4-2026-10-03.json)はその試験条件だけに適用する。

### 証拠・再現・計測境界

[Identity](evaluation/p0-2026-10-04/identity.json)はmodel、tokenizer、chat templateを含む配置ファイルのSHA-256、4-bit／group size 64のconfig、backend Python source SHA-256、依存version、OSを保存する。upstream commit／model revisionは不明で、content hashで同一性を固定した。native binaryの全内容hashは未収集で、完全な配布build認定ではない。modelはGemma利用条件、MLX-LMはMIT、vLLM／vLLM-MetalはApache-2.0。外部コードの流用は今回行っていない。

[Baseline audit](evaluation/p0-2026-10-04/baseline-audit.json)は2 backend・各3 run・同一workload／artifact／hardware・backend内同一build・全件品質／SLO・三言語各100件以上を検証する。別process起動はcollectorの記録に依存し、reportだけから独立性を証明するものではない。[共通索引](evaluation/p0-2026-10-04/evidence-index.json)はphase／比較／qualification／soakを元reportのhashとscopeへ接続し、旧資格を現在のruntimeへ転用しない。旧direct／daemon比較の相対幅18.52%は失敗結果のまま保存する。

client TTFTはqueue＋tokenize＋prefill＋通信を含む。client TPOTは到着interval由来でGPU decode時間ではない。最終contentから`[DONE]`までのtailもserialization単独ではない。backend内部queue、tokenize、prefill、decode、model load、serializationは独立計測できず、共通索引ではnullとする。client phase名を内部kernel計測に読み替えない。内部instrumentationとCPU／cache／energyによる変動原因の特定は[Next]の計測拡張である。

[実MLX cache監査](evaluation/p0-2026-10-04/cache-metadata.json)ではKVCacheの確保容量131,072 bytesとmetadata値が一致した。active stateは8,192 bytesで、容量と区別する。同一objectの重複は除けるが、parent＋viewの共有storageは二重計上し得る。physical storageはnull。1,000回の単独metadata取得は合計0.663 msで、推論中の負荷や性能改善は測定していない。

autotunerには異なるprofile ID・同じpolicy・同じhardwareの確認roundを要求する契約を追加した。確認winnerが一致しても独立取得とE2E改善の認定は行わず、自動適用はfalse。実測候補選択とE2E回帰への接続は[Next]として残す。

再現時は配置済みmodelと上記Homebrew環境を使用する。collectorは既存出力を上書きせず、port 19140／19141が使用中なら停止し、起動したprocess群を終了する。

```sh
PYTHONPATH=. .venv/bin/python scripts/collect_p0_baselines.py --output docs/evaluation/p0-new-run
python3 -m vllm_apple.p0_audit docs/evaluation/p0-new-run/mlx_lm-r*.json docs/evaluation/p0-new-run/vllm_metal-r*.json
PYTHONPATH=. /opt/homebrew/opt/vllm-metal/libexec/bin/python scripts/probe_real_mlx_cache.py
```

[Next] P1でHTTP全体の資源上限、長時間cancel／queueとp95待ち時間、8時間soakを検証する。P2でKV正確性・eviction・batchingを検証する。比較matrixの長context・高並列・別architectureは今後の対象で、今回の合格scopeには含めない。[Later][pending] 未保有SoC／メモリ構成の試験は環境取得後に行う。

実装検証：最終全回帰1,404 tests合格（11 skip）、変更PythonファイルのRuffとdiff whitespace検査合格。保存した実MLX cache probeでも容量検証を再現した。試験用backend processは終了済み。

## English

[Done] P0 now has an audited execution path and a reproducible local baseline: two backends, three fresh processes per backend, 102 measured requests per run, and 102 completions per language per backend. All 612 requests passed the same arithmetic quality and latency gates. The model artifact and workload are identical. This covers M4/32 GiB, Gemma 2 2B 4-bit, concurrency one and short output only.

Median goodput was 18.3411 tokens/s for MLX-LM and 25.7337 for vLLM-Metal. Metal's 6.19% spread exceeds the 5% stability policy; neither performance leadership nor general capability certification is granted. Each run has fewer than 1,000 samples, so p99 remains a reference. Cached usage is available for MLX and missing for Metal. Backend flags do not establish batching, chunked-prefill or KV-precision correctness.

The linked identity, baseline audit and evidence index retain hashes, versions, failures, scope and unavailable phases. Source/version binding does not hash every native binary or identify upstream commits. Client TTFT includes multiple internal phases; missing internal timings remain null. Historical soak and proxy evidence does not certify the present runtime. Real KV metadata measures allocated logical capacity, not unique physical storage; shared views can overcount. Autotuner confirmation binds policy and separate profile IDs but cannot prove independent acquisition or E2E benefit and never applies settings automatically.

[Next] Internal timing instrumentation, attribution of variance, sustained cancellation/resource tests and real optimization confirmation remain explicit follow-up work in P1–P3. [Later][pending] Unavailable hardware requires a future test environment. Use the commands above to reproduce without overwriting existing evidence.

## 简体中文

[Done] P0已完成执行路径审计及本机可复现基准：两个后端各启动三个独立进程，每次测量102个请求，每个后端各语言累计102个完成请求。612个请求均通过相同算术质量及延迟门槛。模型文件和工作负载完全相同；结论仅覆盖M4／32 GiB、Gemma 2 2B 4-bit、并发1和短输出。

MLX-LM与vLLM-Metal的有效吞吐中位数分别为18.3411和25.7337 tokens/s。Metal的6.19%波动超过5%稳定性规则，因此不认证性能领先。每次样本不足1,000，p99仅供参考。MLX有cached usage，Metal缺测。配置开关不能证明批处理、分块prefill或KV精度正确。

identity、baseline audit和证据索引保留hash、version、失败、适用范围及缺测信息；未固定上游commit或全部native binary内容。client TTFT包含多个内部阶段，内部独立计时保持null。历史soak与proxy证据不自动认证当前runtime。真实KV metadata表示逻辑分配容量，共享view可能重复计数。自动调优确认绑定相同policy及不同profile ID，但不证明独立采集或端到端收益，也不自动应用设置。

[Next] P1–P3继续内部计时、波动归因、持续取消／资源测试与实测优化确认。[Later][pending] 未持有硬件待环境具备后测试。以上命令可重现测量且不会覆盖原证据。
