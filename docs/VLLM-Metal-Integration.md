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

familyはthread-global stateではなくlazy MLX Primitiveのimmutable fieldへ格納する。これによりPythonで
graphを構築した後にGPU評価されても選択が保持され、別request・別layerへの漏洩やlayerごとの強制同期を
生じない。profile未発見、破損、hardware/source fingerprint不一致、shape missでは空値となり、upstreamの
自動dispatchを変更しない。fixtureは256 MiBを
上限とし、巨大なCPU attention行列を作らず、zero keyとKV-head別constant valueから既知の期待出力を
生成する。各計測はMLX arrayを評価してGPU完了まで同期し、reference比較後だけlatencyとdigestを返す。

2026-08-29にApple M4（10 GPU cores）、Python 3.12、vLLM-Metal `813e738d`でnative extensionと
全Metal libraryをsource buildし、capability `compatible=true`を確認した。Gemma 2 2B IT 4-bitの
float16 KV fixtureを1K/4K context、decode/prefill各3 sampleで測定し、全候補がreferenceと同一digestで
correctnessを通過した。winnerは1K decodeがper-token、4K decodeがsplit-KV、16-token prefillは
1K/4Kともper-tokenだった。profile `56d785bba017e1a14312a61e`はhardware/source fingerprint別の
private Application Support領域へatomic保存した。

profileは最大64候補のprivate directoryからbounded探索し、物理Mac専用fingerprintとインストール済み
vLLM-Metal source fingerprintを再検証する。完全一致shapeのwinnerだけを各Primitiveへ渡す。M4実環境で
4K decodeが`split_kv`へ解決され、未登録2K shapeが空値へfail closedすることを確認した。

2026-08-29にpatched vLLM-Metal 0.27.1 serverをGemma 2 2B IT 4-bit、BF16、4K context、
`gpu-memory-utilization=0.25`で起動し、1,024 prompt tokensから2 tokensを生成した。実server shapeは
prefill `(context=1024, query=1024)`、decode `(context=1025/1026, query=1)`だった。synthetic
`query=16` profileは安全にmissし、production-aligned BF16 profile `588d5dbe0e795e3dddde5492`を
生成後、最初のdecodeで`family=per_token`のEngineCore hitを確認した。responseはHTTP 200で正常終了し、
prompt 1,024 / completion 2 tokensだった。

telemetryはprofile load、各family初回hit、最大8種類のshape missだけをvLLM EngineCore loggerへ出す。
prompt本文とtoken IDは記録しない。API/EngineCore分離processでも観測でき、同一familyのlayer反復でlogを
増幅しない。

production missは、hardware/source fingerprint別のprivate observation artifactへ最大16種類、64 KiBまで
保存する。artifactはcurrent user所有のregular fileだけを受け付け、0600 fileと0700 directoryへatomicに
更新する。同一shapeのlayer反復では再書き込みせず、prompt本文とtoken IDは保存しない。

`vllm-apple vllm-metal-v2-tune --observed-shapes <path>`は、現在の物理hardware/source fingerprintと
厳密に一致するartifactだけを読み、実際に観測したquery/context長を計測する。小さな追加tuning runが既存の
decode coverageを上書きしないよう、profile探索は有効なincremental profileをshape単位で統合し、新しい
decisionを優先しながら最大16 shapeの決定的なcomposite profileを構築する。

Gemma 2 2B ITのproduction prefill `(context=1024, query=1024)`を観測artifactから再計測し、追加profile
`b1e858be26d9a517a3c02163`を生成した。統合profile `3026b74e2c53de801fb3d9b9`では、このprefillと既存の
decode `(context=1025, query=1)`がともに`per_token`へ解決され、shape hit 2、miss 0を確認した。

次の境界は、推論requestと競合しないidle時だけこのtuningを起動し、完成profileをdaemon/Mac appの
scheduler safe pointで反映するorchestrationである。

このorchestrationのscheduler境界としてexclusive maintenance leaseを実装した。active reservationが
存在する場合は開始せず、計測中は新しいadmissionを構造化された`runtime_maintenance`応答で抑止する。
coordinatorはsingle-flightで、計測または適用が失敗しても`finally`でleaseを解放し、例外本文をeventへ
残さずerror classだけを公開する。profile適用callbackもlease内で実行されるため、古いprofileを使うrequestと
新しいprofileを使うrequestが混在しない。

daemonは`--vllm-metal-source-root`で検証対象source identityを固定し、measurement helperは
`--vllm-metal-v2-helper`またはPATH上の`vllm-apple-v2-measure`から発見する。完全一致するobservationが
存在するときだけidle coordinatorを起動し、生成profileをatomic保存した後、同じlease内でmanaged backendを
recycleする。再起動後のEngineCoreは統合profileを新規loadする。source/helper/observationが不足する場合は
構造化eventを残して通常推論を継続し、`--disable-native-v2-idle-tuning`で明示的に無効化できる。

observation artifactは5秒間隔のbounded pollingで監視し、最大64 KiBのprivate regular fileだけをSHA-256で
比較する。新しい内容が10秒間安定した場合だけ起動し、同一digestは再計測しない。schedulerがbusyで開始
できない場合はdigestを消費済みにせず次回pollで再試行する。起動済みdigestは計測失敗時にも自動反復せず、
新しいshape追加時だけ次のtuning/recycleを許可するため、継続的なGPU負荷や再起動loopを生じない。

次はこの状態とeventをSwift SDKでtyped decodeし、Mac appから自動tuningの有効状態、待機、実行、適用、
失敗を確認・制御できるcontractを公開する。

runtime snapshotへ`native_v2_tuning`を追加し、`status`、bounded `run_id`、適用profile ID、本文を含まない
error codeを公開した。Swift SDKは`NativeV2TuningState`としてHTTP/UDSの両transportでdecodeし、古いdaemonで
fieldが欠落する場合はidleへfallbackする。Mac sampleはevent streamもtyped decodeして、英語・日本語・
简体中文で待機、実行、適用、失敗を即時表示する。自動tuning自体は既存daemon flagで無効化できる。

`POST /v1/native-v2-tuning`はsession token認証後、fieldが`action`だけの`enable`、`disable`、`retry`を
受け付ける。disableは実行中kernelを強制終了せず次回起動を止め、enableはartifact監視を再開する。retryは
最後に検証済みのtune/apply closureがある場合だけsingle-flightで再実行し、未準備またはbusyなら409を返す。
Swift SDKはHTTP/UDSの両方でtyped control resultを返し、Mac sampleは英語・日本語・简体中文の操作UIから
状態を即時更新する。

enable/disable preferenceはApplication Supportの`settings/native-v2-tuning.json`へ保存する。artifactは
schema versionとbooleanだけを許し、最大4 KiB、current-user所有、0600 file、0700 directory、fsync後の
atomic replaceを要求する。daemon flagによる明示disableを最優先し、それ以外は起動時に設定を復元する。
破損・unsafe artifactは有効側へfail safeし、保存失敗でも明示disable自体は維持する。HTTP/UDS controlからの
変更は永続化成功後だけruntime状態へ反映するため、画面表示と再起動後の設定が食い違わない。

profile適用は保存、backend recycle、readiness確認を一つのmaintenance transactionとして扱う。最初の
restartが失敗した場合、新profileを同じprivate directory下の`quarantine/`へatomic moveして通常探索から
除外し、残っているlast-known-good incremental profile集合でもう一度だけrestartする。rollback成功時は
runtimeをREADYへ戻し、tuning runだけをFAILEDとして報告する。rollbackまたはquarantineに失敗した場合だけ
runtime全体をFAILEDにする。quarantineはcurrent-user所有の0600 profile、0700 directory、最大64件に制限する。

daemon起動時と新規隔離後にquarantine directoryをbounded走査し、最大64件の件数とmtime/profile IDで決まる
直近24桁IDだけをcoordinatorへ復元する。runtime snapshotとeventは本文やpathを含まずこのsummaryだけを
公開する。Swift SDKは旧daemonでfieldが欠落しても0件へfallbackし、Mac sampleは英語・日本語・简体中文で
隔離件数を表示し、直近IDはhelp診断だけに利用する。

`vllm-apple vllm-metal-v2-restore <profile-id>`は現在の物理hardware/source fingerprintを再検証し、隔離済み
profileの全shapeを実helperで再計測する。candidate eligibility、CPU/MLX correctness、cross-family digest、中央値、
2% tie-breakの全gateに合格した新しいprofileだけをactive directoryへ保存する。失敗時は隔離artifactを維持し、
成功時も元artifactをprivateな`restored/`へ移して監査可能にする。`quarantine/`と`restored/`はいずれも64件を
hard capとし、自動削除による診断情報の喪失を避ける。

認証済み`POST /v1/native-v2-tuning`の`restore` actionはpathを受け取らず、24桁の隔離profile IDだけを受け取る。
daemonは再計測と保存をexclusive maintenance lease内で実行し、managed backendのrecycleとreadiness確認が
成功した場合だけruntimeをREADYへ戻す。失敗時は新profileを再隔離し、last-known-goodで一度だけrollbackする。
Swift SDKはHTTP/UDSの両transportでtyped restore APIを公開し、Mac sampleは直近の隔離profileを英語・日本語・
简体中文の操作から復帰できる。成功後はquarantine summaryも再走査して画面へ反映する。

native v2 hardware identityは物理構成に加えて、OS、Metal toolchain、MLXから作る24桁environment fingerprintを
合成する。環境検出はprocess内で一度だけ行い、fingerprint生成ごとのsubprocess起動を避ける。いずれかのversionが
変わるとprofileとobservationの探索directoryが変わるため、旧artifactは削除せずfail closedで無効化される。
新環境ではproduction shapeを再収集した後、既存のidle tuningとreadiness gateを通して再benchmarkする。

runtime memory ledgerはweights、KV、recurrent、prefix、sliding-window、MoE experts、scratch、Core MLを独立した
additive componentとして扱い、Metal heap/RSSだけをoverlap envelopeとして加算しない。`StateMemorySpec`の固定領域は
起動時に常駐計上し、KV/windowはrequestのcontext tokensとbatchからadmission前に投影してscheduler reservationへ
反映する。semantic prefixとmodel prefix、scheduler scratchとmodel workspaceはそれぞれ合算し、上限超過はbackendへ
渡す前に拒否する。legacy `ModelMemorySpec`は同じstate contractへ変換される。

次はQwen3.8-Flash-Nextのbounded metadata inspectionを追加し、hybrid architectureの必須機能がbackendにない場合は
model load前に明示的なcapability errorとして拒否する。
