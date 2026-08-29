# vLLM-Metal Native v2 Integration Contract

最終確認：2026-08-28  
確認対象：`vllm-project/vllm-metal` commit `813e738d95840bd66b60248ad4a557485320d896`

## 現在の判定

現行vLLM-Metalはnative v2 Paged Attentionを実装しているが、vLLM-Appleの3-stage
`score_width` / `softmax_width` / `output_width`を直接受け取るABIは持たない。
したがって、headerを受信しただけではtuning適用済みと判断しない。

確認したdispatch familyは次の通り。

- NAX prefill：128 threads固定
- tiled prefill：head sizeに応じて32 / 64 / 128 threads
- per-token decode：256 threads固定
- split-KV decode：pass 1とreduceが256 threads固定

これらは単一の3-stage kernelではなく、kernel family選択、function constant、threadgroup memory、
partition判定と結び付いている。既存winnerをthread数だけ置換するとcorrectness、occupancy、shared
memory budgetを壊す可能性がある。

## v1 acceptance gate

`vllm-apple vllm-metal-integration-inspect <source-root>`は以下をboundedに検査する。

1. `sdpa.py`から`paged_attention_primitive`へ到達するnative v2 call site
2. C++側のonline、tiled、primitive dispatch topology
3. bundled Python call-site hook
4. `VLLM_APPLE_THREAD_CONFIG_ABI_V1`と3幅を受け取るC++ ABI

3と4が存在しない現行upstreamは`compatible=false`となる。source本文は保存せず、24桁の
fingerprintとcapability flagだけを出力する。fileはcurrent user所有のregular file、各2 MiB以下に
制限する。

## 次のABI

native v2専用profileは別schemaとして実装した。以下を独立shapeとして扱う。

- prefill family：NAX / tiled / per-token fallback
- decode family：single-pass / split-KV
- head size、block size、query tokens、KV length、concurrency
- threads、tile、partition threshold、threadgroup memory

correctness gateを通過したfamily単位のwinnerだけをC++ dispatchへ渡す。既存の3-stage reportを
暗黙変換しない。

候補はupstream eligibilityと一致する場合だけ生成される。各候補は最大9 sampleの中央値を使い、
correctnessに失敗した候補とsample間でdigestが変化した候補を除外する。最速から2%以内は同等とし、
追加workspaceとdispatch complexityが小さいfamilyを決定的に選ぶ。

profileはhardware fingerprintとvLLM-Metal source fingerprintへ結び付け、最大16 shape、512 KiBに
制限する。0600 fileへatomic保存し、load時にcandidate eligibility、sample中央値、winner、profile
IDを再計算する。

## native measurement ABI v1

`VLLMMetalV2MeasurementAdapter`はkernel benchmarkをcontrol processから分離し、current-user所有の
実行fileを引数なしで起動する。stdinへ1件のbounded JSON requestを渡し、stdoutから1件のJSON
responseだけを読む。stderrは推論daemonのmemoryへ保持しない。

requestは`abi_version=1`、`operation=measure_paged_attention_v2`、完全なshape、family別configurationを
含む。responseは`abi_version`、CPU/MLX referenceとの比較結果`passed`、正の
`latency_nanoseconds`、lowercase SHA-256 `output_digest`だけを許可する。timeout、64 KiB以下の出力、
終了code、厳密なfield集合を検証する。

同じcandidate内でdigestが揺れる場合に加え、correctnessを通過したper-token fallbackと異なるdigestを
返すspecialized familyもwinnerから除外する。これによりC++ helperの誤配線を性能値だけで採用しない。

`vllm-apple-v2-measure`はこのprotocolを実装する隔離helperである。vLLM-Metalのnative extensionが
`vllm_apple_measure_paged_attention_v2(str) -> str`を公開する場合だけ呼び出す。関数はcanonical request
JSONを受け取り、result schema準拠JSONを返す。symbolがない現行upstreamでは終了code 2でfail closedし、
通常の自動dispatchをfamily別計測の代用にはしない。

helperの`--capabilities`はkernelやmodel bufferを確保せず、extension loadとsymbolのcallable性だけを
bounded JSONで返す。`vllm-metal-v2-capability`およびtuning CLIはこのhandshakeをbenchmarkより先に
実行する。

`integrations/vllm-metal/813e738d-native-v2-measurement.patch`は確認対象commitへ次を追加する。

- helper threadだけに作用し、空文字で必ず解除するfamily強制制御
- NAX / tiled / per-token / split-KVのshape compatibility gate
- bounded Python fixtureを呼ぶ`vllm_apple_measure_paged_attention_v2` symbol

通常の推論threadではoverrideが空のため、upstreamの自動dispatchを変更しない。fixtureは256 MiBを
上限とし、巨大なCPU attention行列を作らず、zero keyとKV-head別constant valueから既知の期待出力を
生成する。各計測はMLX arrayを評価してGPU完了まで同期し、reference比較後だけlatencyとdigestを返す。

2026-08-29にApple M4（10 GPU cores）、Python 3.12、vLLM-Metal `813e738d`でnative extensionと
全Metal libraryをsource buildし、capability `compatible=true`を確認した。Gemma 2 2B IT 4-bitの
float16 KV fixtureを1K/4K context、decode/prefill各3 sampleで測定し、全候補がreferenceと同一digestで
correctnessを通過した。winnerは1K decodeがper-token、4K decodeがsplit-KV、16-token prefillは
1K/4Kともper-tokenだった。profile `56d785bba017e1a14312a61e`はhardware/source fingerprint別の
private Application Support領域へatomic保存した。

次の境界は、このprofileを起動時にbounded探索し、完全一致shapeのwinnerだけをrequest-localに
production dispatchへ適用することである。未一致、破損、fingerprint不一致はupstream自動dispatchへ
fail closedする。
