# vLLM-Apple Runtime Roadmap

最終更新：2026-09-10

本ロードマップは、[Design-Specifications.md](Design-Specifications.md)を実装可能な単位へ分解し、現在のコードベースに対する進捗を示す。

## ステータス

- `[Done]` implemented in the current codebase
- `[Next]` high-priority unfinished work
- `[Later]` planned, but not the closest next step

`[Done]` は設計済みではなく、現在のコードベースに実装と検証が存在する項目だけに付与する。

## 現在地

Phase 1のcontrol plane、メモリ安全性基盤、AppleExecutionPlanner、StateMemorySpec、
prefill/decode別profile、Swift SDK、3言語macOS sample、Gemma実modelの30分安定性まで実装済み。

2026-09-15のstatus監査で、後続実装と実機記録が存在した古い`[Next]`を`[Done]`へ更新した。
現在ローカルで進める優先項目は、contention profileのreload／rollbackをSwift SDKのtyped操作と
Mac app三言語UIへ接続し、利用者が再起動なしで安全に管理できるようにすることである。
外部項目の最優先は大容量Apple SiliconでのQwen3.8-Flash-Next text-only qualificationと、専用runnerでの
vLLM 0.28.x昇格試験である。HomebrewのvLLM-Metal 0.29.xは、固定revisionの認定stackとは別candidateとして
Metal platform選択を先に確認する。
設計判断は
[Architecture-Decision-Apple-Execution.md](Architecture-Decision-Apple-Execution.md)に固定する。

### vLLM-Metal実行経路の昇格手順

2026-09-19: Homebrew 0.29.0でMetalPlatform選択、Gemma 2 2B IT 4-bitのGPUロード、
non-stream応答とSSE完走を実機確認。`+cpu` suffix自体はMetal利用不可を意味しない。
一度Transformers 5.12.1 + tokenizers 0.22.2でtext smokeを確認した後、
画像・音声の依存要件を満たす5.17.0 + 0.23.2へ戻し、pip check成功とMetal選択を確認。
通常version matrixへの昇格はqualification完了後に判断する。

- `[Done]` Homebrew候補のMetal GPUでGemma non-stream / SSE実応答確認
- `[Done]` 画像・音声を含むパッケージ依存関係の整合（pip check成功、モデル推論の認定とは別）
- `[Done]` Homebrew 0.29.0 / Transformers 5.17.0の30秒text負荷試験（142/142成功、47.832 decode tok/s、平均TTFT 87.699 ms）。証跡: [short report](evaluation/homebrew-029-text-short-2026-09-19.json)。品質試験省略・30分未達のため完全認定ではない。
- `[Done]` 多言語品質試験の不一致原因を末尾空白・改行と確認。`exact`は維持し、品質smokeは明示的な`trimmed_exact`で前後空白だけを除外。内部空白・追加説明は拒否し、本文非保存のincremental hash比較を維持。Gemmaの英語・日本語・简体中文実SSEで合格、関連20テスト成功。
- `[Done]` Homebrew 0.29.0 / Transformers 5.17.0、Gemma 2 2B IT 4-bit、context 1024、concurrency 1、`VLLM_METAL_MEMORY_FRACTION=0.25`で30分qualification成功（1800.253秒、6,775/6,775成功、失敗0、3.763 req/s）。3言語`trimmed_exact`、sampling/streaming、正常shutdownが合格。監視対象processのRSS peak増加409,600 bytes、終了時増加0。別途3 sampleのphase probeは42.841 decode tok/s、平均TTFT 91.105 ms。証跡: [30-minute report](evaluation/homebrew-029-text-30min-2026-09-19.json)。KV capacity再評価はunavailableで未検証、画像・音声や全stackの認定へは拡張しない。
- `[Next]` 画像の長時間安定性と音声の実推論確認
- `[Done]` Gemma 3 4B IT 4bitの固定revision取得・14ファイル照合、Homebrew 0.29.0でMLX-VLMロード確認。実画像要求はmultimodal encoder adapter未準備でEngineCore停止を再現し、不合格として記録: [attempt](evaluation/gemma-3-4b-it-vision-attempt-2026-09-19.json)。chat templateを備えるだけではbackend画像対応を保証しない。
- `[Done]` 配布版のQwen3-VL adapter登録とforward_readyを先に確認し、Qwen3-VL-2B-Instruct-4bit（revision `9c4f5209e57b31f4b9dfba735de3fb983739c9cc`、16ファイル照合済み）で画像smokeを実測。Homebrew 0.29.0、context 2048、memory fraction 0.25で3言語×赤青の6ケースすべて合格、正常shutdown。証跡: [vision smoke](evaluation/qwen3-vl-2b-vision-smoke-2026-09-19.json)。長時間認定・一般的画像理解能力の認定は含まない。
- `[Done]` `vision-smoke` CLI: 32x32赤・青PNGをOpenAI image_urlに埋め込み、3言語6ケースを画像付きSSEで検証。同一質問の画像差分を用い、画像・生成本文は非保存。PNG送信・画素・判定の回帰テストを追加。長時間vision認定とは別扱い。
- `[Done]` text-only qualification runnerによるvision誤認定を起動前に拒否（image-input probeが未実装のため）。
- `[Done]` ローカルGemma 3 4B PTの実weight headerでvision/projector tensor 439件を確認。Homebrew 0.29.0でMLX-VLMロード成功。`skip_vision=true`だけではweight欠落と判断できない。画像付きchatはtokenizerのchat template未定義によりHTTP 400となり、画像品質は未測定。
- `[Next]` 合格したQwen3-VLの画像入力を伴う長時間qualificationへ接続。音声モデルはローカルmodels一覧には未配置。
- `[Done]` phase/vision probeでSSEの`[DONE]`を必須化し、本文とusageがあっても途中切断は不合格とする。関連26テスト成功。

Homebrewの最新版を無条件に実行経路へ昇格させず、次の順序で検証する。

1. `[Done]` backend executable、Python、vLLM、vLLM-Metal、Transformersのversionと、Metalが利用可能か、vLLMが実際に選択したplatformを`doctor`で取得する。
2. `[Done]` platformがCPUの場合は、Metal利用可能性と「未選択」を別々に報告し、推論を開始しない。`doctor`は`metal_available=true`かつCPU platform選択を`vllm_metal_available_but_not_selected`として識別し、managed serveは起動前に拒否する。
3. `[Done]` 固定revisionの認定stackで、Gemma non-stream、SSE、memory、30分安定性を専用runnerで再検証。Homebrew 0.29.0 / Transformers 5.17.0、Gemma 2 2B IT 4-bitで6,775/6,775成功、失敗0、3言語品質と正常shutdownを確認した。
4. `[Done]` Homebrew 0.29.0 candidateのGemma smoke・短時間・30分qualificationを独立証跡として保存（上記条件に限定）。
5. `[Later]` candidateが3回連続で同じ証跡を満たした場合だけ、対応version matrixと標準backendへ昇格する。

この手順により、vLLM-Metalの実装済み機能と、手元のHomebrew版が実際にMetal platformを選択できることを混同しない。

```text
Swift / CLI
    ↓
VLLMAppleKit / Control API
    ↓
vllm-appled
    ↓
vLLM + vLLM-Metal
    ↓
MLX / Metal / Unified Memory
```

### Apple SoC runtimeとしての設計原則

本projectは「vLLMをMLXへ接続するだけ」のbackendではなく、CPU、GPU、将来のANE、Unified
Memoryを一つの推論装置として扱うruntime control/execution planeを目指す。ただし、GPU演算器数、
memory帯域、非公開ANE命令、OS kernel内部schedulerなどのhardware/OS限界は補える対象に含めない。

通常のtensor演算はMLXをcorrectness/performance baselineとし、Paged Attention、KV compaction、
quantized GEMV/GEMM、RoPE/RMSNorm fusion、MoE routingなどのLLM固有hot pathだけを実測後に
Custom Metal候補とする。Core ML/ANEは固定graph化しやすいVision/Audio encoderやdraft処理から
段階導入し、動的LLM decodeを無条件に移さない。

```text
Graph / Memory Planner + Global Scheduler
                    ↓
           Operator Dispatcher
        ┌───────────┼───────────┐
       MLX      Custom Metal   Core ML
   correctness     hot path    fixed graph
        └───────────┼───────────┘
              Unified Memory
```

## Phase 1 — Foundation and Mac Integration

### Hardware and memory

- `[Done]` Apple Silicon、architecture、OS、CPU数の検出
- `[Done]` Unified Memory総量の検出
- `[Done]` `vm_stat`を利用した現在のavailable memory検出
- `[Done]` memory pressureの基礎判定
- `[Done]` 検出APIが利用できない場合の保守的fallback
- `[Done]` GPU core数とSoC名の検出精度向上
- `[Done]` framework allocator値とOS resident観測を分離するsource-aware二層memory telemetry（unknownはnull、latest値+monotonic peakのみを保持）
- `[Done]` vLLM Prometheus adapterからの非同期自動sample投入（1秒poll、1 MiB/20,000行/4 KiB行上限、last-good保持）
- `[Done]` IOGPU統計とMLX allocator adapterからの自動sample投入（backend内active/cache/peak、bounded ioreg、取得不能時null）
- `[Done]` native macOS memory pressure notificationをtelemetryとelastic controllerへ接続（重複coalesce、scheduler safe point、登録失敗時fallback）
- `[Done]` pressure、RSS、IOGPUを使った新規workload admission抑制（Unified Memory viewはmax、既存work非cancel、interactive escape hatch）
- `[Done]` pressure回復後の段階的batch/context ramp-up（0/5/15/30秒、transient memoryも12.5/25/50/100%で解除）
- `[Done]` tokenizer実測prompt tokensをadmission context見積もりへ接続（vLLM `/tokenize`、64 KiB count scan、token ID非保持、fallback counter）
- `[Done]` tokenize latency cacheと同一prompt fingerprintのbounded再利用（SHA-256 keyのみ、256件LRU、5分TTL）
- `[Done]` 同時到着した同一tokenize要求を1回へ集約するbounded single-flight（最大64 active key、5.5秒wait、失敗共有）
- `[Done]` weights、KV、prefix、scratch、Metal heap、Core ML bufferの統合budget ledger（unknown明示、monotonic peak、Metal overlap非加算）
- `[Done]` model manifestからweights実測値を自動投入し、budget overcommitをadmissionへ接続
- `[Done]` KV ratioのみのbackendでcapacity bytesを取得するversion-gated adapter（vLLM 0.24–0.28、単一cache config、2 TiB hard limit）
- `[Done]` backend load後の実KV capacityとinspected weights footprintによるcontext再評価（起動設定非変更、admission上限のみ縮小）
- `[Done]` context再評価結果のcoalesced SSE通知とMac app警告表示（英語、日本語、简体中文）
- `[Done]` 実model qualificationへcontext reduced判定とprivate/atomic report保存を統合
- `[Done]` qualification reportのSwift typed decodeとMac app履歴表示（bounded、破損・oversize・symlink fail-soft）
- `[Done]` self-hosted qualification成果物をMac SDKのbounded readerで再検証するCI gate
- `[Done]` self-hosted Apple Silicon runnerで実model 30分qualification workflowを実行
- `[Done]` NSProcessInfo thermal stateとactive電源系統のpower mode検出（旧profile/Swift decode互換、unknown fail-soft）

### Automatic context calculation

- `[Done]` model weight、KV bytes/token、workspaceを入力とするcontext計算
- `[Done]` physical memoryとcurrent available memoryの小さい方をhard limitに採用
- `[Done]` OS reserve、safety headroom、workspaceの除外
- `[Done]` SAFE、BALANCED、AGGRESSIVE tierの生成
- `[Done]` token block単位への安全な切り下げ
- `[Done]` model max contextの適用
- `[Done]` local / Hugging Face cache model path resolution
- `[Done]` standard Transformer / GQA metadataからKV bytes/tokenを自動算出
- `[Done]` weight shardの重複を避けたmemory size集計
- `[Done]` model metadataからmodel max contextを検出
- `[Done]` inspect不能時の4096 token保守的fallback
- `[Done]` managed backendへのbalanced context自動適用
- `[Done]` MLA latent KV、state-space recurrent/conv state、hybrid layer構成固有のstate memory計算
- `[Done]` 未cache Hugging Face modelのweightを取得しないbounded metadata fetchとrevision binding
- `[Done]` model load前後でのcontext再評価
- `[Done]` 最大64件のadmitted workload履歴とthermal状態を用いた動的context調整。実KV capacity／設定上限を超えず、fairは75〜87.5%、seriousは50〜75%、criticalは50%へ16-token単位で縮退し、nominal復帰時は安全上限まで回復。拒否requestは履歴へ混入させず、変更をruntime eventへ公開

### Runtime profile

- `[Done]` immutable runtime profile model
- `[Done]` versioned profile schema
- `[Done]` private permissionによるprofile保存
- `[Done]` `fsync`とatomic replaceによる破損防止
- `[Done]` profile load、validation、migration
- `[Done]` hardware/model別profile cache
- `[Done]` benchmark結果を含むprofile versioning。benchmark report ID、capability ID、hardware/environment fingerprint、exact workload identity、TTLをversioned device-placement planへ結合し、private atomic保存、strict derived-value再計算、last-known-good rollbackとsafe-point適用を実装

### Basic scheduler

- `[Done]` operator単位のbackend選択interface
- `[Done]` CPU、MLX GPU、Metalの基礎routing
- `[Done]` transient memoryのhard admission limit
- `[Done]` thread-safe reservationとrelease
- `[Done]` request priority model
- `[Done]` queueingとpriority arbitration
- `[Done]` cancellation時のreservation自動解放
- `[Done]` backend failure時のMetal → MLX → CPU fallback
- `[Done]` application threadからbackend command queueを隔離するglobal submission scheduler
- `[Later]` prefill、decode、sampling、encoder別のoperator dispatch
- `[Later]` Vision/Audio encoderのCore ML/ANE routingとGPU LLM pipeline連携
- `[Done]` profiler実測値によるbackend選択。同一operator/phase/precision/shape/batchのCPU baselineとaccelerator reportを比較し、digest一致、最低sample数、peak memory非悪化、cold-load償却後5%以上のlatency改善を満たすbackendだけをversioned placement planへ昇格。hardware/environment identity、TTL、quarantine、last-known-good、safe-point reloadまでruntimeに統合済み

### Edge-native adaptive execution

FreeTokenのedge-native設計を参考にするが、CUDA/PCIe実装は取り込まず、Apple Unified Memoryと
Metal向けに独立実装する。外部engineへのruntime依存は追加しない。

- `[Done]` tool call、tool result、thinking、turn境界のbounded semantic anchor cache
- `[Done]` raw promptを保存しないprefix fingerprintと最深有効anchor探索
- `[Done]` entry/state byte二重上限、thread-safe LRU、runtime resize
- `[Done]` eviction時にbackend stateを確実に解放できるownership contract
- `[Done]` daemon RuntimeServiceとbackend KV/recurrent state adapter contractの接続
- `[Done]` capture/restore/release、stale state破棄、bounded release retry queue
- `[Done]` semantic cache hit/miss/capture/eviction/release failure runtime metrics
- `[Done]` MLX prompt cache固有のopaque KV state capture/restore/release adapter
- `[Done]` scheduler safe pointでのsemantic cache elastic memory再配分
- `[Done]` Normal/Warning/Criticalによる1x/1/2x/1/8x budgetとNormal復元
- `[Done]` active reservation中のbudget変更保留とpending適用event/metrics
- `[Done]` macOS memory pressure notificationからelastic controllerへの自動接続
- `[Later]` KV/expert/workspaceを含む統合elastic memory再配分
- `[Later]` prefill/decode別のCPU・Metal・Unified Memory bandwidth profile
- `[Later]` MoE `(layer, expert)` working-set LRUとMetal residency adapter
- `[Later]` layer double-buffered prefetchと不足時のon-demand fallback
- `[Later]` page-aligned fast-load optimizer artifactとkernel compatibility index

### Daemon and API

- `[Done]` `vllm-appled` headless daemon foundation
- `[Done]` localhost-only default bind
- `[Done]` runtime state model
- `[Done]` health、readiness、hardware、profile、runtime endpoint
- `[Done]` versioned response metadata
- `[Done]` OpenAI形式のmodels/chat endpoint foundation
- `[Done]` request bodyの4MiB hard limit
- `[Done]` bounded request threads、listen backlog、socket timeout
- `[Done]` active、peak、completed、rejected requestのbounded server metrics
- `[Done]` inference未接続時の構造化503 response
- `[Done]` vLLM-Metal managed process adapter
- `[Done]` backend environmentとPython/vLLM/vLLM-Metal version診断
- `[Done]` loopback-only inference backend起動
- `[Done]` backend readiness、exit、timeout監視
- `[Done]` bounded stdout/stderr log capture
- `[Done]` OpenAI models/chat proxy
- `[Done]` 4KiB単位でflushするSSE token streaming proxy
- `[Done]` client切断時のupstream stream解放
- `[Done]` backend terminate、timeout後killによるshutdown
- `[Done]` 固定revisionの認定stackを用いた実model vLLM-Metal互換性検証
- `[Done]` Homebrew vLLM-Metal candidateのMetal platform選択と実model smoke。0.29.0でMetalPlatform選択、Gemma 2 2B IT 4-bitのnon-stream / SSE、30分qualificationまで確認済み
- `[Done]` graceful request drainを伴うshutdown
- `[Done]` Unix Domain Socket HTTP transport
- `[Done]` UDS pathのowner/type検証と0600 permission
- `[Done]` constant-time Bearer session token認証
- `[Done]` atomic 0600 session token file
- `[Done]` 256 eventのbounded runtime event ring
- `[Done]` 最大8 clientのbounded event subscription
- `[Done]` 遅延subscriberへの`stream.gap`通知
- `[Done]` `Last-Event-ID`対応SSE runtime event stream
- `[Done]` 明示opt-in remote modeのTLSとBearer authentication。非loopback bindは`--allow-remote`、証明書＋owner-only秘密鍵、session tokenがすべて揃う場合だけmodel load/listen前gateを通過し、TLS 1.2以上のserver socketへ昇格する。欠落、片側TLS identity、symlink／公開秘密鍵はfail-closed

### CLI

- `[Done]` `vllm-apple hardware`
- `[Done]` `vllm-apple context`
- `[Done]` `vllm-apple profile`
- `[Done]` `vllm-apple serve`
- `[Done]` `vllm-apple serve <model>` command path
- `[Done]` managed backend port、startup timeout、max model context options
- `[Done]` `doctor` command
- `[Done]` UDS、session token、session token file options
- `[Done]` automatic model inspectionとrecommended configuration表示
- `[Done]` structured startup progress
- `[Done]` macOS per-user launchd向けdaemon install、start、stop、status command。owner-only atomic plist、既存定義の暗黙上書き拒否、認証付きprivate UDS既定値、label/plist ownership検証、shellなし10秒上限のlaunchctl lifecycleを実装

### Swift SDK and Mac app integration

- `[Done]` `VLLMAppleKit` Swift Package foundation
- `[Done]` FoundationとSwift Concurrency中心の公開API
- `[Done]` typed health、hardware、profile、chat model
- `[Done]` `async/await` request API
- `[Done]` `AsyncThrowingStream` streaming interface
- `[Done]` schema compatibility check
- `[Done]` localizable error message key
- `[Done]` Managed Local daemon launcher foundation
- `[Done]` Swift HTTP clientのBearer session認証
- `[Done]` Swift runtime event decodingと再購読interface
- `[Done]` ManagedRuntimeのtoken file連携
- `[Done]` POSIX Unix Domain Socket HTTP/SSE client
- `[Done]` UDS request/response sizeとheader size limit
- `[Done]` stream cancellation時のsocket shutdown
- `[Done]` daemon stdout/stderrの400 entry bounded log capture
- `[Done]` log message単位の8KiB limit
- `[Done]` readiness timeoutとprocess exit監視
- `[Done]` failure時のみの上限付きrestart policy
- `[Done]` SIGTERM grace periodとSIGKILL fallback
- `[Done]` crash recovery後のUDS再接続
- `[Done]` app bundle / Application Support resource resolver
- `[Done]` private Application Support directoryとsession token生成
- `[Done]` Darwin UDS path長を回避する短いsocket resource配置
- `[Done]` Unix Domain Socket対応transport
- `[Done]` runtime state/failure event stream foundation
- `[Done]` 英語、日本語、簡体字中国語のlocalization catalog
- `[Done]` 最小SwiftUI Mac chat sample
- `[Done]` sample transcript 200件、prompt 32Ki文字、response 128Ki文字の上限
- `[Done]` sampleのbounded conversation contextとstream cancellation
- `[Done]` Xcode app target、Swift package、3言語resource、検証付きdaemon bundle build phase sample
- `[Later]` Objective-C adapter
- `[Later]` notarizationとApp Sandbox統合sample

### Schema and testing

- `[Done]` runtime snapshot JSON Schema v1
- `[Done]` health response JSON Schema v1
- `[Done]` runtime event JSON Schema v1
- `[Done]` context境界値test
- `[Done]` scheduler memory limit test
- `[Done]` atomic profile persistence test
- `[Done]` localhost API integration test
- `[Done]` Swift model decoding test
- `[Done]` backend commandとsecurity validation test
- `[Done]` OpenAI non-streaming proxy integration test
- `[Done]` SSE streaming proxy integration test
- `[Done]` managed backend process readiness/shutdown test
- `[Done]` model metadataとKV memory計算test
- `[Done]` session authenticationとprivate token file test
- `[Done]` bounded EventBusとgap recovery test
- `[Done]` authenticated HTTP event stream test
- `[Done]` private UDS lifecycle integration test
- `[Done]` UDS path length境界test
- `[Done]` Python 80 test passing
- `[Done]` Swift bounded log buffer test
- `[Done]` Swift → Python UDS authentication integration test
- `[Done]` Swift managed daemon crash/restart/reconnect integration test
- `[Done]` Swift 8 test passing
- `[Done]` Swift resource resolver permission/path/fallback test
- `[Done]` SwiftUI Mac sample build passing
- `[Done]` JSON Schemaによるlive health/runtime/SSE event response validation
- `[Done]` 未対応schema keywordを拒否するdependency-free schema validator
- `[Done]` concurrent request saturation、503 early rejection、slot recovery test
- `[Done]` bounded latency/error metricsによるmemory soak runner
- `[Done]` RSS growth thresholdとmachine-readable exit status
- `[Done]` non-streaming/streaming mixed応答検証とbackend PID生存性判定
- `[Done]` 1800秒、PID監視、RSS上限を必須化する30分認定mode
- `[Done]` compatible backendの起動、readiness、認定、shutdownを束ねる実model qualification CLI
- `[Done]` greedy、seed付きsampling、SSE完了を束ねる昇格probe
- `[Done]` sampling再現性とstream/non-stream digest一致のfail-closed gate
- `[Done]` 実modelで30分以上のlong-running memory stability test
- `[Done]` daemon SIGKILL後のstale UDS replacementとtoken/auth recovery test
- `[Done]` daemon relaunch後のSIGTERM cleanupと残留process確認
- `[Later]` real-model correctness regression suite

## Phase 1 Completion Criteria

Phase 1を完了とする条件：

- `[Done]` `vllm-apple serve <model>` だけで実際の対応modelを起動できることを検証する
- `[Done]` OpenAI互換のnon-streaming/streaming chatが実modelで成功する
- `[Done]` Swift ManagedRuntimeからGemma 2 2B IT 4-bitのMLX load、UDS streaming chat、shutdown実model E2E
- `[Done]` MLX backendをトップレベルserveから選択、既存backend portの誤接続拒否、Gemma 2 2B IT 4-bitでnon-stream／SSE実機スモーク、Mac sampleのモデル付き起動設定
- `[Done]` modelに応じた安全なcontextが自動設定される
- `[Done]` memory pressure時に新規workloadを抑制し、daemonが異常終了しない
- `[Done]` backend errorが構造化され、Swift側で復旧可能性を判定できる
- `[Done]` Python、Swift、end-to-end testが継続的に成功する

## Model Optimization Compiler — Companion Track

推論の安定性を守るため、model変換処理を`vllm-appled`へ直接載せない。共有coreを利用する
`vllm-apple-optimize` workerを別processとして実行し、Mac UIは別app targetとして提供する。
original modelは常にread-onlyとし、生成物はimmutable artifactとして保存する。

```text
VLLMAppleKit / Control API
        ├── vllm-appled             stable inference
        └── vllm-apple-optimize     isolated optimization worker
                    ↓
           immutable model artifact
```

### O0 — Contracts and safe dry-run planner

- `[Done]` `OptimizationPlan`、objective、quality budget、resource budget model
- `[Done]` calibration dataset manifestとdataset fingerprint schema
- `[Done]` source hash、license、transform履歴を持つartifact manifest schema
- `[Done]` hardware/model metadataから候補を返す副作用なしdry-run planner
- `[Done]` required disk、peak memory、output sizeの保守的事前見積もり
- `[Done]` 明示実行・bounded I/O profiler実測値によるestimated duration
- `[Done]` original model pathへのwriteを拒否するpath safety policy
- `[Done]` optimizer state/event schemaとbounded progress event
- `[Done]` structured optimizer error/recoverability taxonomyとCLI JSON error
- `[Done]` plan、manifest、path traversal、disk/memory境界値test
- `[Done]` optimizer plan、profile、event、artifact、error、adapter、worker、checkpoint、MLX invocation/report、perplexity、deterministic generation、quality gate Schema v1
- `[Done]` `vllm-apple-optimize plan` CLI entry point

### O1 — Representation optimization

- `[Done]` versioned backend adapter interfaceとbounded registry
- `[Done]` 外部packageをimportしないMLX dependency/model capability detection
- `[Done]` `vllm-apple-optimize capabilities` CLI
- `[Done]` FP16/BF16 → INT8/INT4 quantization candidate generation
- `[Done]` MLX safetensors用versioned exporter adapter
- `[Later]` GGUF等へのversioned exporter adapter
- `[Later]` KV cache precision、context、batch configuration search
- `[Done]` isolated subprocess worker、bounded pipe drain、process-group cancel
- `[Done]` private sibling workspaceと失敗・cancel時のcleanup
- `[Done]` regular file、file/byte/depth上限のstreaming検証、`fsync`、atomic promotion
- `[Done]` versioned terminal worker resultとbounded stage event
- `[Done]` source/output/execution/budgetへbindingしたpersistent checkpoint manifest
- `[Done]` private permission、64 KiB上限、strict load、atomic replaceを持つCheckpointStore
- `[Done]` prepared/failed/cancelledのrestartとconverted/completedのresume decision protocol
- `[Done]` kernel解放型`flock`と同一process guardによるcross-process checkpoint lease
- `[Done]` workerへのcheckpoint stage遷移、explicit resume、attempt更新の統合
- `[Done]` converted workspaceからcommandを再実行しないvalidation/promotion resume
- `[Done]` promotion済み・checkpoint未更新状態のartifact再検証とreconciliation
- `[Done]` MLX/MLX-LM 0.26.x–0.31.xとApple Siliconのversion/platform compatibility gate
- `[Done]` shell、upload、remote code trustを使わない固定argument export invocation
- `[Done]` side-effect-free dry-runと明示的`--execute`/`--resume`を持つexport CLI
- `[Done]` 8 MiB bufferによるmodel snapshot全fileのstreaming SHA-256 binding
- `[Done]` 最大16 MiBのsafetensors headerからweight非loadでdtypeを判定
- `[Done]` virtual environment launcherを保持したworker dependency同一性
- `[Done]` backend child outputのprivate workspace rootへの安全な正規化
- `[Done]` GPT-2 FP32 safetensors → MLX 4 bit export、再load、1 token生成smoke
- `[Done]` output tree hash、size、file数、peak RSS、latencyをcheckpointとartifact manifestへ記録
- `[Done]` private、atomic、idempotent sidecar manifestとversioned export report
- `[Done]` baseline/quantizedを別processで順次実行するbounded local perplexity runner
- `[Done]` 同一dataset fingerprint、slice、token数を要求するquality regression gate
- `[Done]` 英語、日本語、简体中文のdomain/language slice評価
- `[Done]` GPT-2 4 bit reject / 8 bit approveの実model quality smoke
- `[Done]` bounded token ID、fingerprint、期待条件scoreによるdeterministic generation gate
- `[Done]` GPT-2 8 bitの6 prompt生成比較と中国語general差分によるreject

### O2 — Calibration and evaluation

- `[Done]` local-only perplexity runnerとPIIを外部送信しないprivacy boundary
- `[Later]` activation全量を保持しないonline statistics / disk streaming capture
- `[Later]` layer、head、neuron importance report
- `[Done]` 英語、日本語、简体中文のsmoke evaluation datasetとslice report
- `[Done]` prompt/生成文をreportへ残さないlocal-only deterministic generation runner
- `[Done]` `contains`/`prefix`を明示する英語、日本語、简体中文の期待応答dataset
- `[Done]` code、math、retrieval smokeをdomain/languageで選択するevaluation suite
- `[Done]` 用途filterをdataset fingerprintへbindingし、異なる選択条件の比較を拒否
- `[Done]` 実際のdomainから未評価能力を算出するgeneration gate
- `[Done]` 明示的chat template適用とprompt形式のreport/fingerprint binding
- `[Done]` template適用後のsample入力token数とprompt/model context上限の事前検証
- `[Done]` baseline/candidate間のprompt形式、token budget、入力token数一致の強制
- `[Done]` Gemma 2 2B IT BF16 / MLX 8 bitのtask score実測と100% token一致gate
- `[Done]` 1K、4K、16K以上の段階評価schemaとfail-fast coordinator
- `[Done]` state bytes、load peak、steady-state RSSの分離report contract
- `[Done]` tokenizer準拠retrieval datasetと実backend measurement adapter
- `[Done]` `/tokenize`によるbounded長さ調整とSSE境界をまたぐretrieval検証
- `[Done]` perplexity、latency、RSSのbaseline比較
- `[Done]` quality gate通過を必須とし、task score、artifact size、throughput、peak RSSで決定的に順位付けするbounded candidate比較
- `[Done]` 未評価能力とquality regressionを明示するrelease gate

### O3 — Weight and structural optimization

- `[Later]` outlier-aware quantization、weight clustering、low-rank approximation
- `[Later]` structured / unstructured pruning experiment adapter
- `[Later]` attention head、MLP、layer functional similarity analysis
- `[Later]` layer bypass、head merge、layer merge candidate generation
- `[Done]` quality budget超過時のcandidate自動reject。dataset fingerprintとslice集合の完全一致を前提にperplexity相対劣化をslice別budgetと比較し、`quality_approved=false`のcandidateをrankなし・`quality_gate_failed`で自動除外。合格候補0件ではartifactを選択しない
- `[Later]` optional LoRA/SFT repair adapterとrepair前後の再評価

### O4 — Mac companion app

- `[Later]` `VLLMAppleOptimizer` Mac app target
- `[Later]` model、用途、品質／速度／memory優先度の設定UI
- `[Later]` disk/memory見積もりと明示的な実行confirmation
- `[Later]` progress、pause、resume、cancel、failure recovery UI
- `[Later]` original / optimized responseとbenchmark比較
- `[Later]` artifact、provenance、license、未評価能力report
- `[Later]` 英語、日本語、简体中文localization
- `[Later]` App Sandbox、notarization、大容量file access sample

## Phase 2 — Apple Runtime Planner

- `[Done]` versioned AppleExecutionPlan schema、decision reason、deterministic dry-run
- `[Done]` StateMemorySpec（weights、KV、recurrent、prefix、window、expert、scratch）
- `[Done]` AppleChipProfileとbackend capability contract foundation
- `[Done]` 実hardware/backend capability detectionとAppleChipProfile永続化
- `[Done]` prefill/decode別bounded profile schemaと集計器（TTFT、TPOT、throughput、peak memory）
- `[Done]` 実backend streamのtoken timing、usage、backend RSS instrumentation
- `[Done]` bounded SSE phase-profile CLIとusage欠落時のfail-closed検証
- `[Done]` plannerと既存context、scheduler、elastic memory policyのatomic safe-point接続
- `[Done]` active/pending execution plan observabilityとphase別batch admission gate
- `[Done]` CPU GEMM/GEMV micro benchmark。profile-bound FP32 CPU capabilityで64×64×64 GEMMと256×256 GEMVを各7 sample実測し、安定output digest、bounded shape/batch、work-item throughput、private atomic reportを確認。M4でmedian 7,035,792 ns／37.23M work-items/s（GEMM）、1,846,583 ns／35.24M work-items/s（GEMV）。証跡: [CPU GEMM](evaluation/cpu-gemm-fp32-m4-2026-09-21.json)、[CPU GEMV](evaluation/cpu-gemv-fp32-m4-2026-09-21.json)
- `[Later]` GPU GEMM/GEMV micro benchmark
- `[Later]` Unified Memory bandwidth測定
- `[Later]` Metal launch latency測定
- `[Later]` Attention throughput測定
- `[Later]` quantized matmul benchmark
- `[Done]` model、shape、batch、context別kernel profile。model metadataからGQA heads、head dimension、KV dtype、context tier、block数・working setを固定するversioned Paged Attention profile、profile-bound shape benchmark、batchを含むdevice benchmark identity、private/atomic strict loaderとCLIを実装済み
- `[Done]` automatic phase batch sizing。memory pressure、thermal、power mode、CPU backend制約からprefillを1/2/4、decodeを1へ決定し、plan identityとdecision reasonへ固定してscheduler admissionで超過を拒否する
- `[Later]` adaptive state allocationとage/pressure別precision
- `[Done]` continuous memory pressure monitoring。macOS libdispatch memory-pressure sourceでnormal/warning/critical変化をevent-driven取得し、同一状態をcoalesce。daemon起動をブロックしない隔離threadからadmission、scheduler、safe-point elastic cacheとruntime eventへ反映し、source不可時は`vm_stat`系telemetryへfail-softする
- `[Done]` thermal/power状態をversioned plan identityとdecision reasonへ固定し、prefill batchを保守的にclampするscheduling foundation
- `[Done]` 15秒bounded thermal/power monitor、同一状態coalesce、current hardware snapshotとruntime change eventへの反映
- `[Done]` Swift SDKのtyped operating-state event decodeと未知のcurrent値に対するfail-soft fallback
- `[Done]` versioned BackendEngine交換契約。vLLM-Metal、Native MLX、Native Metal、Core ML draft、CPUを共通enumで扱い、version、architecture、precision、phase、operator、isolation、ready/lifecycle、request deadline/cancel safe point、retryable fallback attemptをfail-closed registryへ統合
- `[Later]` 各production backend processをBackendEngine registryへ直接登録するcomposition rootと段階的切替
- `[Later]` CPU/Core ML draft + GPU verifyのheterogeneous speculative execution
- `[Done]` bounded kernel self-test/performance probe contractとprofile単位quarantine registry
- `[Done]` hardware、OS、toolchain、MLX、backend versionを束ねるenvironment fingerprint
- `[Done]` 隔離subprocessによるNative MLX smoke probe adapter
- `[Done]` vector add、16x16 matmul、bounded KV copyのoperator別MLX probe suite
- `[Done]` operator単位の独立quarantineとbounded registry構築
- `[Done]` probe必須のfail-closed operator dispatcher contractとscheduler接続
- `[Done]` Swift/Metal隔離subprocessによるNative Metal compile/dispatch/readback probe
- `[Done]` probe専用private temporary module cacheとnative failure quarantine
- `[Done]` MLX/Metal結果を統合するRuntimeProbeCoordinatorとregistry構築
- `[Done]` active reservation終了時のscheduler safe-point dispatcher適用
- `[Done]` MLXをimportしないtoolchain、MLX、backend version discovery
- `[Done]` production daemon起動時のRuntimeProbeCoordinator自動実行とbounded event
- `[Done]` environment fingerprint別private/atomic/strict probe cache
- `[Done]` cached quarantine再利用と一致時のnative startup probe省略
- `[Done]` 既定7日expiry、未来時刻拒否、suite version変更時のcache失効
- `[Done]` probe cache JSON Schemaとversion不一致を再probeするno-migration policy
- `[Done]` 非連続block table、page gather、context切り詰めを含むPaged Attention probe
- `[Done]` compressed latentからkey/value projectionを行うbounded MLA probe
- `[Done]` MLX Paged Attentionの14、256、1024 token decode shape tier
- `[Done]` Native Metal Paged Attentionのcompile、dispatch、readback probe
- `[Done]` model metadata由来のbounded/versioned Paged Attention shape profile
- `[Done]` shape別identityと64 MiB hard limitを持つNative Metal profile consumer
- `[Done]` 実Mac Metal deviceでの128-token shape correctness/performance smoke
- `[Done]` shape benchmarkのprivate/atomic保存とstrict identity-bound loader
- `[Done]` Qwen相当32Q/8KV head、dimension 128、1K contextの実Mac Metal計測
- `[Done]` shape benchmark CLIとApplication Support fingerprint別既定保存先
- `[Done]` score/softmax/output 3-stage Metal Paged Attentionとscore/output並列化
- `[Done]` Qwen相当1K shapeで旧single-thread比約132倍の実Mac改善確認
- `[Done]` 256-thread bounded scratchによるsoftmax max/sum reduction
- `[Done]` Qwen相当1K/4K shapeの実Mac correctnessと約1.73/2.32 ms中央値
- `[Done]` 32/64/128/256 thread候補のcorrectness-gated shape autotuning
- `[Done]` Qwen相当1K shapeの実Mac winner測定
- `[Done]` 複数sample中央値と2% deterministic tie-break
- `[Done]` winner/全候補のmodel、hardware、environment別private/atomic永続化
- `[Done]` tuning reportのprivate file、identity、winner再計算strict loader
- `[Done]` tuning CLI、JSON出力、Application Support既定保存
- `[Done]` daemon tuning pending/active stateとreservation-bound tuning ID
- `[Done]` 最後のactive request完了後だけwinnerを適用するscheduler safe point
- `[Done]` request-bound tuning contextのmanaged backend invocation境界への伝播
- `[Done]` OpenAI JSONを変更しないbounded/versioned tuning header contract
- `[Done]` streaming完了・切断までのreservation保持と確実なsafe-point解放
- `[Done]` model/hardware/environment完全一致reportのdaemon起動時自動探索・install
- `[Done]` private directory、最大64候補、破損report隔離、最新版決定的選択
- `[Done]` 明示report指定と自動tuning無効化のCLI/daemon contract
- `[Done]` backend tuning context strict parserとversioned JSON Schema
- `[Done]` ContextVarによるasync request分離とmalformed時の安全なfallback
- `[Done]` shape完全一致winner lookupとPaged Attention kernel invocation bridge
- `[Done]` dependency-free ASGI middlewareとaccepted/rejected/hit/miss metrics
- `[Done]` managed backend commandへのASGI middleware自動登録
- `[Done]` bounded `serve --help` capability gateと未対応versionの安全な無効化
- `[Done]` frontend multiprocessing無効化によるrequest/kernel同一process contract
- `[Done]` backend accepted tuning ID応答とcontrol plane ack/missing/mismatch metrics
- `[Done]` kernel hookがshape winnerを消費した場合だけ返すapplied acknowledgement
- `[Done]` native v2 topology / Python hook / C++ ABIのbounded source inspector
- `[Done]` integration inspection JSON SchemaとCLI終了コードcontract
- `[Done]` vLLM-Metal `813e738d`実sourceのnative v2検出と安全な非互換判定
- `[Done]` NAX/tiled/per-token/split-reduce別native v2 shape/config model
- `[Done]` upstream eligibility準拠のbounded candidate generation
- `[Done]` correctness/digest-gated中央値と2% deterministic tie-break
- `[Done]` hardware/source-bound profile schemaとprivate atomic persistence
- `[Done]` candidate eligibility、中央値、winner、profile ID strict loader
- `[Done]` native v2実kernel measurement adapterとversioned C++ helper ABI
- `[Done]` native extension capability-gated benchmark helper bridge
- `[Done]` native measurement symbol capability handshakeとbenchmark前fail-fast gate
- `[Done]` model metadataからのbounded decode/prefill実device profile生成CLI
- `[Done]` vLLM-Metal `813e738d`向けnative measurement patchと実Mac profile生成
- `[Done]` native v2 profileのbounded自動探索とlazy Primitive内request-local production dispatch適用
- `[Done]` patched vLLM-Metal serverでのGemma 2 2B BF16 end-to-end profile hit検証
- `[Done]` production shape captureからのbounded自動profile生成とprefill coverage
- `[Done]` exclusive maintenance leaseとsingle-flight idle tuning coordinator
- `[Done]` daemonでのobservation/helper発見とprofile適用時backend recycle
- `[Done]` observation更新監視とidle debounceによる同一shape再計測防止
- `[Done]` runtime snapshot/eventとSwift SDK/Mac appへのnative v2 tuning状態公開
- `[Done]` authenticated enable/disable/retry control endpointとMac app操作UI
- `[Done]` native v2 tuning preferenceのprivate永続化とdaemon再起動時復元
- `[Done]` profile適用後readiness失敗時のlast-known-good rollbackとquarantine
- `[Done]` quarantine診断のbounded snapshotとMac app表示
- `[Done]` quarantine retention policyと再計測合格後だけのexplicit restore gate
- `[Done]` explicit restore後のmanaged backend safe-point適用とreadiness再確認
- `[Later]` Mac個体別runtime autotuner（batch、tile、KV block、prefill chunk、kernel）
- `[Done]` OS、toolchain、MLX version変更時のprofile失効と安全な再benchmark
- `[Done]` state/workspace統合budgetとMoE expert residency
- `[Done]` Qwen3.8-Flash-Next bounded metadata inspectionとcapability gate

## Compatibility Track — Qwen3.8-Flash-Next / Qwen4 Preview Architecture

Qwen3.8-Flash-Nextは、Gated DeltaNet、Qwen Sparse Attention（QSA）、MoE、Gated Residual、
N-gram Embedding、MTP、Vision Encoderを組み合わせた`qwen4_exp`系hybrid architectureである。
標準Transformerとして近似せず、backendが未対応の場合はload前に明示的に拒否する。

本trackは直近のplanner safe-point統合と実model安定性を妨げない時期に開始する。公式vLLM
recipeは現時点で専用imageとCUDA／ROCm構成を前提とするため、通常vLLM対応をそのまま
vLLM-Metal対応とは見なさない。

- `[Done]` `qwen4_exp` / `qwen4_exp_text` configのbounded metadata inspection
- `[Done]` layer別Gated DeltaNet recurrent state計算
- `[Done]` QSA block/indexer stateとsparse retrieval budgetの`StateMemorySpec`拡張
- `[Done]` 512 expert、10 routed + 1 sharedのworking-set/residency計算
- `[Done]` 51B N-gram Embeddingを独立residency classとして計画
- `[Done]` Gated Residual scratch stateとMTP追加weight/state計算
- `[Done]` text-only、Vision、MTP、native 262K、YaRN 1Mを別capabilityとして判定
- `[Done]` vLLM-Metal architecture capability gateと構造化error
- `[Done]` Native MLX architecture capability gate
- `[Done]` quantized artifact実サイズによるMac適合判定とhard memory ceiling
- `[Done]` qualification reportへのTTFT、TPOT、tokens/sec、peak RSS profile統合
- `[Done]` self-hosted Apple Silicon qualification workflowのNative MLX backend対応
- `[Done]` MLX qualification process起動前のUnified Memory hard ceiling gate
- `[Done]` phase profile/memory fitのSwift typed decodeとMac app三言語表示
- `[Done]` self-hosted認定でphase/memory evidenceを必須化するSwift CI gate
- `[Done]` 英語・日本語・简体中文の本文非保存semantic smokeとSwift evidence gate
- `[Done]` incremental hash完全一致と16 MiB SSE上限によるconstant-memory quality判定
- `[Done]` direct vLLM-Metal qualificationへのarchitecture/memory load前gate統合
- `[Done]` Homebrew vLLM-Metal candidateの起動前preflightでMetal未選択を拒否する実機確認
- `[Done]` vLLM-Metal明示architecture feature probeとdaemon/qualification昇格契約
- `[Done]` 公式Qwen3.8-Flash-Next 48層configでのmetadata/state回帰固定
- `[Done]` weight取得前の`--model-metadata` backend capability preflight
- `[Done]` MLXをimportせず既存Qwen4構成部品を監査するstatic readiness CLI
- `[Done]` Transformers 5.16.1 Qwen4-Exp準拠のGated Residual/QSA依存なしCPU参照fixture
- `[Done]` 128 byte固定MLX fixtureとCPU oracleのbounded numerical comparison
- `[Done]` 公式1,658 tensor／131 shardのtext・MTP・Vision weight mapping schema
- `[Done]` weight非ロードのbounded safetensors index検査とpath traversal拒否
- `[Done]` Qwen4 GDN/QSA/PLE chunk-invariant cache-state契約とconfig fingerprint binding
- `[Done]` prefill／segmented prefill／token decodeの純CPU semantic cache fixture
- `[Done]` Qwen専用workflowのweight/cache静的証跡とpromotion bundle binding
- `[Done]` GDN/QSA/GR/MoE/PLE/Vision/MTP全tensorのcomponent分類
- `[Done]` source/destination各1 shard上限のconstant-memory MLX conversion plan
- `[Done]` conversion plan ID・config・index digestのpromotion bundle binding
- `[Done]` 8 MiB固定bufferとshard単位atomic置換によるidentity-preserving staging
- `[Done]` SHA-256 binding済みprivate checkpointからの安全な中断再開と改変拒否
- `[Done]` 完了stage全shardのdigest再検証とunexpected file拒否
- `[Done]` verified stageからのmode-aware component/shard adapter contract生成
- `[Done]` weight data非読込のbounded safetensors header・dtype・shape・offset検証
- `[Done]` index/header完全一致、重複key、overlap、gap、shard越境のfail-closed拒否
- `[Done]` manifest digest再検証付きread-only tensor catalog／bounded `pread` reader
- `[Done]` 1 open shard・最大8 MiB chunk・requested mode別tensor access gate
- `[Done]` destination array・stream chunk・scratchのatomic tensor memory admission
- `[Done]` component別上限、並行load overcommit拒否、例外時reservation自動解放
- `[Done]` resident／on-demand expert／partitioned PLE／optional mode別load plan
- `[Done]` MoE active expert比率と最大PLE partitionによるresident working-set算定
- `[Done]` packed MoE expert axisとconfig topologyの完全一致gate
- `[Done]` expert axis-0 bounded slice readerとslice単位memory reservation
- `[Done]` backend非依存Qwen tensor conversion ABI v1とstrict request/response binding
- `[Done]` bounded file-backed worker output、timeout、helper ownership/permission gate
- `[Done]` worker側stage contract/load plan再構築とrequested mode binding
- `[Done]` reservation lease内converter protocol、全chunk消費・shape保持gate
- `[Done]` 16 MiB上限のone-shot MLX correctness converter entrypoint
- `[Done]` BF16/F16/F32 decode・dtype変換・eval・値非保存digest evidence
- `[Done]` conversion reservationからresident destinationへのatomic縮小
- `[Done]` backend非依存resident tensor storeとexplicit unload lifecycle
- `[Done]` cleanup失敗resourceのreservation保持quarantineとrelease retry
- `[Done]` runtime ABI v1 load/unload/status/quarantine-retry/shutdown command schema
- `[Done]` session binding、contiguous sequence、bounded idempotent replay cache
- `[Done]` operation別strict responseと本文非保存status contract
- `[Done]` 16 KiB length-prefixed private Unix socket transport
- `[Done]` macOS `LOCAL_PEERCRED`／Linux `SO_PEERCRED` current-user peer gate
- `[Done]` socket device/inode binding済みsafe cleanupと接続command上限
- `[Done]` verified reader・admission・resident store・service・transport worker composition
- `[Done]` private session credentialのatomic no-clobber publishとinode-bound cleanup
- `[Done]` multimodal Qwenのrequested mode別text/Vision/MTP capability gate
- `[Done]` text-only memory fitからMTP runtime working setを除外するmode-aware budget
- `[Done]` download前artifact admission（Unified Memory、disk staging、構造化判定）
- `[Done]` self-hosted qualification workflowのbackend起動前artifact admission evidence
- `[Done]` artifact admission evidenceのSwift typed decodeとbounded CI再検証
- `[Done]` admission evidenceとqualification modelの識別子binding
- `[Done]` admissionとqualification memory-fitのartifact/resident byte binding
- `[Done]` large-memory runner、exact artifact/resident bytes、text-only、30分、Swift証跡再計算を固定したQwen専用qualification workflow
- `[Done]` Qwen認定前後のmodel tree streaming SHA-256再検証（constant-memory、private manifest、report非公開）
- `[Done]` Qwen認定での任意CMS provenance mode（trusted CA・signer identity・load前後署名再検証）
- `[Next]` 大容量Apple Siliconでtext-only smoke、TTFT、TPOT、RSS、品質gate
- `[Later]` worker compositionへ実MLX resident backendを注入するproduction entrypoint
- `[Later]` production MLX adapterのcorrectness合格後にNative Metal kernelを比較検討

参照：
[Qwen model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)、
[vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next)。

## Phase 3 — Kernel Optimization

状態：`[Done]`。kernel実装、capability gate、実機microbenchmark、production dispatch、
fallback、fusion、stress、および大容量model向け再現可能qualification契約まで完了。
`large-memory` runner上でのQwen3.8-Flash-Next実weight認定結果は、外部運用証跡であり
このrepository／現在の32 GB環境だけでは生成せず、成功reportが得られるまでmodel昇格は行わない。

- `[Done]` MLX correctness smoke baselineとoperator dispatcher contract
- `[Done]` 小型matmulとKV copyのMLX correctness/performance baseline probe
- `[Done]` sequence 8/32、head dimension 8のscaled dot-product attention probe
- `[Done]` causal prefill、single-token decode、4 query/2 KV head GQA probe
- `[Done]` Paged Attention page gather/decodeとMLA latent projection probe
- `[Done]` MLX Paged Attentionの14、256、1024 token decode tier
- `[Done]` Native Metal Paged Attentionのbounded correctness/performance probe
- `[Done]` model-backed shape profile schema、GQA検証、KV working-set算出
- `[Done]` bounded Native Metal shape profile consumer
- `[Done]` 実Mac Metal deviceでのshape別probe smoke
- `[Done]` model、hardware、environmentに結び付くshape benchmark永続化
- `[Done]` Qwen相当1K model shapeの実Mac correctness/performance計測
- `[Done]` shape benchmark CLIとApplication Support fingerprint別既定保存先
- `[Done]` score/softmax/output 3-stage Metal kernelとbounded score buffer
- `[Done]` Qwen相当1K shapeで約1.93 msの実Mac中央値
- `[Done]` constant-memory threadgroup softmax max/sum reduction
- `[Done]` Qwen相当1K/4K shapeの実Mac長context測定
- `[Done]` bounded shape別thread幅autotunerと実Mac winner選択
- `[Done]` tuning中央値、2% tie-break、winnerのfingerprint別永続化
- `[Done]` tuning report strict loaderとwinner policy再検証
- `[Done]` tuning CLIとMac app向けversioned JSON出力
- `[Done]` daemon winner適用safe pointとactive request構成固定
- `[Done]` request-bound tuning contextのmanaged backend invocation境界への伝播
- `[Done]` compatible tuning reportのdaemon起動時bounded自動探索・install
- `[Done]` backend parser、request-local lookup、kernel invocation bridge
- `[Done]` managed vLLM serverへのcapability-gated ASGI middleware登録
- `[Done]` current native v2 source topology qualificationとfalse-positive防止
- `[Done]` vLLM-Metal native v2 kernel-family別autotuning contract
- `[Done]` native v2実kernel measurement adapterとC++ dispatch configuration ABI
- `[Done]` native extension capability-gated benchmark helper bridge
- `[Done]` model metadataからのbounded decode/prefill実device profile生成CLI
- `[Done]` vLLM-Metal `813e738d`向けnative measurement patchと実Mac profile生成
- `[Done]` native v2 profileのbounded自動探索とlazy Primitive内request-local production dispatch適用
- `[Done]` patched vLLM-Metal serverでのGemma 2 2B BF16 end-to-end profile hit検証
- `[Done]` production shape captureからのbounded自動profile生成とprefill coverage
- `[Done]` exclusive maintenance leaseとsingle-flight idle tuning coordinator
- `[Done]` daemonでのobservation/helper発見とprofile適用時backend recycle
- `[Done]` observation更新監視とidle debounceによる同一shape再計測防止
- `[Done]` runtime snapshot/eventとSwift SDK/Mac appへのnative v2 tuning状態公開
- `[Done]` authenticated enable/disable/retry control endpointとMac app操作UI
- `[Done]` native v2 tuning preferenceのprivate永続化とdaemon再起動時復元
- `[Done]` profile適用後readiness失敗時のlast-known-good rollbackとquarantine
- `[Done]` quarantine診断のbounded snapshotとMac app表示
- `[Done]` quarantine retention policyと再計測合格後だけのexplicit restore gate
- `[Done]` explicit restore後のmanaged backend safe-point適用とreadiness再確認
- `[Done]` OS、toolchain、MLX version変更時のprofile失効と安全な再benchmark
- `[Done]` state/workspace統合budgetとMoE expert residency
- `[Done]` Qwen3.8-Flash-Next bounded metadata inspectionとcapability gate
- `[Done]` layer別Gated DeltaNet recurrent state計算
- `[Done]` QSA block/indexer stateとsparse retrieval budgetの`StateMemorySpec`拡張
- `[Done]` 512 expert、10 routed + 1 sharedのworking-set/residency計算
- `[Done]` 51B N-gram Embeddingを独立residency classとして計画
- `[Done]` Gated Residual scratch stateとMTP追加weight/state計算
- `[Done]` text-only、Vision、MTP、native 262K、YaRN 1Mを別capabilityとして判定
- `[Done]` quantized artifact実サイズによるMac適合判定とhard memory ceiling
- `[Done]` Native MLX architecture capability gate
- `[Done]` qualification reportへのbounded phase profile統合
- `[Done]` self-hosted qualification preflight/workflowのMLX切替
- `[Done]` MLX実model qualificationへのload前memory fit統合
- `[Done]` qualification phase metricsのSwift SDK/Mac app統合
- `[Done]` qualification artifactのphase/memory evidence CI検証
- `[Done]` 三言語bounded semantic smokeのqualification統合
- `[Done]` 三言語応答の本文非保存exact-match gate
- `[Done]` vLLM-Metal/MLX共通のqualification load前memory fit
- `[Done]` vLLM-Metalのbounded Qwen feature宣言probe
- `[Done]` 公式48層・36 GDN・12 QSA・Vision/MTP config照合
- `[Done]` 公式multimodal artifactのtext-only qualification mode
- `[Done]` requested modeを記録・再検証するSwift text-only evidence gate
- `[Done]` 大容量Apple Silicon Qwen text-only認定の再現可能なfail-closed契約。`large-memory` self-hosted runner、正確なartifact／resident bytes、load前Qwen4 weight map・conversion・cache検査、text-only、三言語品質、phase、30分memory、前後integrity、Swift再計算、promotion bundleを固定。実weightの合格は未取得のためmodel自体は未昇格
- `[Done]` kernel capability、self-test結果、quarantine理由のversioned registry
- `[Done]` vLLM-Metal Paged Attention capability/benchmark統合
- `[Done]` 計測で選択したnative Metal Paged Attention v2拡張とproduction dispatch
- `[Done]` MLX MLA latent projection correctness/performance probe
- `[Done]` MLX Q8/Q4 quantized matmul baselineと固定誤差・性能互換性gate。M4／MLX 0.32.1、1×64入力・16×64 weight・group 32・各3 sampleでdense MLX比Q4 1.016倍／Q8 1.045倍、固定誤差gate内で合格。証跡: [MLX Q4/Q8 probe](evaluation/mlx-q4-q8-matmul-probe-2026-09-20.json)
- `[Done]` Q4 dequantized GEMV＋SiLUを単一MLX lazy graphで実行するfusion候補。staged materialization比0.877倍、固定誤差1e-5内で合格
- `[Done]` RMSNorm＋RoPEを単一MLX lazy graphで実行するfusion候補。staged materialization比0.912倍、固定誤差1e-5内で合格
- `[Done]` top-2 MoE routing、gated Expert GEMM、weighted reductionのbatched MLX graph。token／expert逐次基準比0.754倍、固定誤差2e-4内で合格
- `[Done]` environment-bound capabilityが合格したpatternだけを最長一致・非重複で置換し、未計測／quarantine時は元graphを保存するbounded graph fusion pass。実機証跡: [MLX Phase 3 fusion probe](evaluation/mlx-phase3-fusion-probe-2026-09-20.json)
- `[Done]` kernel autotuning、fingerprint別profile cache、環境変更時失効
- `[Done]` probe承認済みMetal→MLX→CPU correctness fallback suite
- `[Done]` Metal toolchain／OS／MLX／backend fingerprint変更時のcache失効と、self-hosted Apple Silicon workflowでのruntime compile、correctness、性能退行probe
- `[Done]` bounded multi-model Metal command submission stress。論理2 model（vector add／Paged Attention）から各3回、最大同時2で実commandを投入し、6/6成功、digest不一致0、peak concurrency 2を確認。証跡: [multi-model Metal stress](evaluation/metal-multi-model-command-stress-2026-09-20.json)

## Phase 4 — Vision

- `[Done]` OpenAI互換の複数image input frontend。inline PNG／JPEGだけを受理し、magic byte、個別／合計byte数、画像数、message／part数をload前に検証してremote URLと偽装形式を拒否。Qwen3-VL managed chatも共通parserへ接続
- `[Done]` model-neutral image preprocessing pipeline。固定shapeのRGB変換、Resize、FP32 Normalize、Patchify、FP16出力を依存注入可能な契約で実装し、不正geometryを実行前に拒否
- `[Done]` image／model revision／preprocessing／encoder fingerprintに結合した容量・entry数制限付きthread-safe LRU vision encoder cache。oversize outputは保存せず、hit／miss／eviction／resident bytesを観測可能
- `[Done]` compatibility key（model revision、preprocessing、encoder、image shape）別のdeterministic multimodal batching。request内の複数画像を分割せず、request／image／patch／encoded byte上限を満たすstable batchを構築し、oversize requestと重複IDを実行前に拒否
- `[Done]` Resize → Normalize → Patchify → Projectionを単一MLX lazy graphで実行する融合候補。arm64／MLX 0.27.1、256×256 RGB→224×224、patch 16、projection 64、3 sampleで段階materialize比0.834倍、固定誤差1e-5内で合格。model固有interpolationのqualificationではない。証跡: [MLX Vision fusion probe](evaluation/mlx-vision-fusion-probe-2026-09-20.json)
- `[Done]` image latency、images/sec、memory/image benchmark。MLX allocator peakを用い、batch 1／2／4・各3 sampleで融合前処理＋projectionを測定。中央値3.568／3.966／4.790 ms、280.3／504.2／835.1 images/sec、最大2.228／2.204／2.183 MB/imageで全batchのdigestが安定。end-to-end model qualificationではない。証跡: [MLX Vision benchmark](evaluation/mlx-vision-benchmark-2026-09-20.json)

## Phase 5 — Audio

- `[Done]` fixed-capacity、preallocated、non-blocking SPSC audio ring buffer。interleaved multi-channel wraparound、overflow時の未読sample保護、underrun／overflow／resident frame telemetryを実装し、producer／consumer pathでは明示lockとI/Oを使用しない
- `[Done]` chunk境界を保持するbounded multi-channel linear resamplerと、全波形を保持しないstreaming log-band feature encoder。連続入力と分割入力の同一性、有限値、retained state上限を検証
- `[Done]` bounded streaming audio session state。chunk sequence、resampler位相、feature window、累積input／resampled frame、named recurrent stateをsession単位で保持し、欠落／重複chunk、終了後入力、duration／state byte超過を拒否。registryはsession数制限、明示close、idle reapでstateを消去
- `[Done]` admission-controlled REALTIME／INTERACTIVE／BACKGROUND priorityとdeadline scheduler。priority class内EDF、推定実行時間を含むdeadline admission、重複／capacity拒否、cancel、期限切れtaskの非実行、late completion／最大lateness telemetryを実装。audio callback外でlockとtask実行を管理
- `[Done]` backend-neutral streaming ASR integration contract。ordered PCM→resample→feature→deadline scheduler→ASR backend→bounded transcriptを接続し、empty intermediate chunk、deadline admission failure、期限切れ非実行、language／時間範囲／confidence／文字数／final markerを検証。実ASR modelのqualificationは未実施で昇格しない
- `[Done]` audio digest、encoder／feature fingerprint、sample rate、channel、sample rangeに結合したbounded thread-safe LRU audio encoder cache。oversize拒否、entry／byte eviction、hit／miss／resident bytes／rejection telemetryを実装
- `[Done]` backend-neutral ASR→dialogue→speech synthesis foundation。final transcriptだけを発話へ進め、dialogue／speech型、言語一致、PCM S16LE frame alignment、byte／duration上限、elapsed timeを検証し、end-to-end latencyとreal-time factorを算出。実dialogue／TTS modelは未認定
- `[Done]` dropout、latency、real-time factor benchmark。16 kHz mono、20 ms chunk、10秒／500 chunkのstreaming resampler＋log-band feature＋session stateを実測し、dropout 0、failure 0、p50 0.304 ms、p95 0.337 ms、max 0.760 ms、RTF 0.0154で合格。ASR／TTS modelとCore Audio device latencyは対象外。証跡: [Audio streaming benchmark](evaluation/audio-streaming-benchmark-2026-09-20.json)

## Phase 6 — Video

- `[Done]` 3候補のbounded初期profile catalogと共通load前artifact/Unified Memory admission
- `[Done]` bounded FFmpeg VideoToolbox hardware decoder integration。regular local file、input byte、H.264／HEVC／VP9／AV1 codec、resolution、frame数、decoded byte、timeoutをload前に制限し、明示`-hwaccel videotoolbox`でBGRA frameを取得。64×64 H.264、10 fps、10 frameの実機smokeに合格。証跡: [VideoToolbox decoder smoke](evaluation/videotoolbox-decoder-smoke-2026-09-20.json)
- `[Done]` VideoToolbox raw BGRAをcaptured stdoutとframe別heap copyを介さず匿名mmapへ直接書き、read-only `memoryview`でframe参照するGPU staging互換copy削減path。64×64 H.264、10 frameの実機smokeに合格。証跡: [mapped staging smoke](evaluation/videotoolbox-mapped-staging-smoke-2026-09-20.json)
- `[Done]` AVAssetReaderのVideoToolbox対応CVPixelBufferを`alwaysCopiesSampleData=false`で取得し、`CVMetalTextureCacheCreateTextureFromImage`によりBGRA8 Metal textureへ直接bindするnative zero-copy path。hardware decode対応をcodecごとに検証し、64×64 H.264の10/10 frameでbinding failure 0を実証。証跡: [VideoToolbox Metal zero-copy smoke](evaluation/videotoolbox-metal-zero-copy-smoke-2026-09-20.json)
- `[Done]` presentation timestamp順reorder、bounded queue、最大reorder幅、late-frame drop、duplicate／capacity rejection、cancel、present／drop／最大lateness telemetryを持つframe scheduler。`drain_late`は古いframeを推論前に除外して現在frameまで進める
- `[Done]` scene-aware deterministic temporal sampler。先頭／末尾、keyframe、scene-change scoreを優先し、残り枠を時間軸上のfarthest-point samplingで補完。最大frame数、最小時間間隔、重複ID、最大input数を検証し、入力順に依存しないselection reportを生成
- `[Done]` video digest、時間範囲、transform／model fingerprintへ結合したframe／patch／embedding／scene共通cache。global LRUに加えてtier別byte budgetを持ち、raw frameがembeddingを追い出す干渉を防止。oversize拒否、tier clear、hit／miss／eviction／resident telemetryを実装
- `[Done]` backend-neutral video VLM integration。temporal sampling→frame単位embedding cache lookup→missだけをbatch encode→元の時間順へ復元→language backendを接続し、encoder count／順序／ID／digest／byte数と応答文字数を検証。特定video VLM modelのqualificationは未実施で昇格しない
- `[Done]` bounded streaming video upload／spool foundation。private 0600 artifactへordered chunkを逐次保存し、全videoをmemoryへ保持せずsequence、chunk数、個別／合計byte、final SHA-256を検証。finalize前artifact非公開、明示close／digest mismatch／idle reapでunlink
- `[Done]` persistent FFmpeg／VideoToolbox workerへordered compressed chunkを逐次投入し、stdout／stderrを並行drainしてdecoded BGRA frameをPTS付きでframe schedulerへ即時送るbounded incremental path。input byte／frame／stderr／shutdown timeoutを制限。64×64 H.264 MPEG-TSを9 chunkで投入し、10/10 frame、scheduler rejection 0、late drop 0を実証。証跡: [incremental stream smoke](evaluation/videotoolbox-incremental-stream-smoke-2026-09-20.json)
- `[Done]` frames/sec、seconds-of-video/sec、memory/minute benchmark。320×180 H.264、30 fps、10秒／300 frameをVideoToolbox＋anonymous mmapで実測し、906.3 frames/sec、30.21 video-sec/sec、retained buffer 69.12 MB、414.72 MB/video-minute、peak RSS 35.11 MBで合格。証跡: [VideoToolbox throughput benchmark](evaluation/videotoolbox-throughput-benchmark-2026-09-20.json)
- `[Done]` M4/32GB向け動画生成qualification profile。Wan 2.2 TI2V-5BをTier Aの640×384・33 frame・20 step・batch 1・量子化必須profileとし、候補modalityからsource-free modeを決定するrunner契約を実装。動画候補は`text-to-video`、画像候補は`text-to-image`を既定とし、非対応modeをworker起動前に拒否する。384高は実MLX-Gen smokeで確認したWanの32-pixel境界
- `[Done]` Wan 2.2 TI2V-5Bのlocal-only Diffusers T2V worker adapter。MPS必須、`local_files_only`、batch 1、640×384・33 frame・20 step request、VAE tiling、CPU-seeded generation、bounded progress telemetry、frame数／shape検証、private MP4のstreaming SHA-256と即時削除を実装。I2Vと実model qualificationは未実施
- `[Done]` Wan 2.2 TI2V-5Bの量子化artifact load前readiness。Diffusers sourceの`WanPipeline`、local artifactの`WanPipeline` identity、Diffusers形式、denoiser／text encoder／VAE完全性、transformer metadataの4/8-bit宣言をweight import・load・Metal allocationなしで検証する専用CLIを実装
- `[Done]` Wan 2.2 TI2V-5B workerのmodule residency contract。現行Diffusers 0.34.0の`text_encoder->transformer->vae`順を固定検証し、`enable_model_cpu_offload(device="mps")`で一度に一moduleだけをMPSへ移す。契約欠落時は全module常駐へfallbackせず生成前に拒否
- `[Done]` Wan 2.2 TI2V-5B正式qualification CLI。artifact readiness、実component容量、指定resident見積り、現在hardware、load前admission、local-only T2V worker、sample間memory recovery、反復実行、private reportを単一fail-closed経路へ統合
- `[Done]` race-safe model integrity manifestを再利用した生成qualification artifact binding。Wan正式CLIは全regular fileのpath／size／SHA-256からroot digestを計算してreport provenanceへ保存し、同容量・同量子化表記の別artifactへのreport replayを拒否。既存v1 reportはdigestなしのlegacy provenanceとして読取互換を維持
- `[Done]` M4/32GB動画試験modelとして`AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit` revision `6875952a110b6bdbcfc00d72b1d89a8e02ab0fc3`を選定。mixed Q8/BF16、MLX-Gen形式、18,189,391,581 bytes（16.94 GiB）、Apache-2.0、非gated。公式Diffusers非量子化版とGGUF版は初回候補から除外。証跡: [Wan video model selection](evaluation/wan-video-model-selection-2026-09-20.json)
- `[Done]` 選定したWan mixed Q8/BF16 artifactを`models/wan2.2-ti2v-5b-diffusers-8bit`へ配置し、18,189,391,581 bytes、22 files、Q8、MLX-Gen、`WanPipeline`、Apache-2.0、全componentをload前検証。MLX-Gen 0.33.1 capabilityでT2V／first-frame I2Vを確認し、Diffusers workerへの誤接続は形式gateで拒否
- `[Done]` Wan向けMLX-Gen isolated T2V worker。in-process CLIで640×384・33 frame・20 step、`--low-ram`、inactive denoiser release、bounded JSON progress、MLX allocator peak telemetry、private MP4 digest／削除をqualification contractへ接続
- `[Done]` MLX-Gen Wan専用load前readinessと正式qualification CLI。MLX-Gen 0.33.1以上、console entrypoint、MLX-Gen形式、Wan pipeline／base identity、mixed Q8/BF16、component完全性をbackend import／weight load／Metal allocationなしで検証し、artifact digest binding付き反復T2V runnerへ接続。配置artifactでreadiness合格
- `[Done]` M4/32GB最小profileの単発実model smoke。mixed Q8/BF16、640×384、33 frame、20 step、16 fps、batch 1、low-RAMでload、20/20 denoise、33/33 decode、health validation、MP4保存に成功。出力213,484 bytes、2.0625秒、memory pressure normal、最大thermal fair。生成物はdigest／metadata取得後に削除。peak memoryと反復安定性は未計測。証跡: [Wan MLX-Gen minimal generation smoke](evaluation/wan-mlx-gen-minimal-generation-smoke-2026-09-20.json)
- `[Done]` MLX-Gen動画workerの実測peak保持とbounded failure diagnostics。生成中の上限超過をbackend内`SystemExit`へ隠さずsample telemetryとして収集し、評価器でfail判定する。MLX-Genがstderrへ出す構造化failureまたは直前512-byte短文をbounded adapterで回収する。low-RAM resident見積りと共通8% emergency reserveの二重余裕も解消
- `[Done]` M4/32GB、MLX-Gen 0.33.1、mixed Q8/BF16 artifactで640×384・33 frame・20 step・batch 1の2-sample正式qualificationに合格。2件ともshape検証、MP4 digest、memory pressure normal、thermal fair、private output即時削除を確認。peak RSSは11,298,210,948／11,298,208,452 bytes（最大10.52 GiB）、median wall 722.27秒、minimum 0.0438 frames/sec。異なるseedのdigestは相違し、prompt／動画はreportへ保存しない。証跡: [Wan MLX-Gen 2-sample qualification](evaluation/wan-mlx-gen-640x384-2sample-qualification-2026-09-20.json)
- `[Done]` 同一最小profileの4-sample stability gateに合格。4/4件が640×384・33 frameで完了し、生成失敗0、issues 0、全件memory pressure normal／thermal fair、sample間memory recovery、private cleanupを確認。peak RSSは11,298,210,948〜11,355,004,900 bytesでdrift 56,793,952 bytes（最大値比0.50%）、median wall 860.66秒、minimum 0.0367 frames/sec。全seedでdigestが相違し、prompt／動画は非保存。証跡: [Wan MLX-Gen 4-sample stability](evaluation/wan-mlx-gen-640x384-4sample-stability-2026-09-20.json)
- `[Done]` 動画frame-count promotion契約。CLIの`--frames`と`--baseline-report`を追加し、4-sample以上・全normal・同一artifact provenanceの合格reportを必須化。幅、高さ、steps、batch固定、最大2倍、`frames - 1`の4境界を満たす一軸変更だけをworker起動前に許可する
- `[Done]` 33→49 frameの一軸promotionを2-sample正式qualificationで実証。2/2件が640×384×49で完了し、issues 0、memory pressure normal、thermal fair、sample間recovery、private cleanup、異なるseedのdigest相違を確認。最大peak RSS 11,533,691,012 bytes（33-frame 4-sample最大比+178,686,112 bytes／+1.57%）、median wall 987.66秒（+126.99秒／+14.76%）。prompt／動画は非保存。証跡: [Wan MLX-Gen 49-frame qualification](evaluation/wan-mlx-gen-640x384-49frames-2sample-2026-09-20.json)
- `[Done]` promoted videoの同一frame数2→4 sample stability契約。49-frame 2-sample reportだけでなく、それを許可した33-frame 4-sample parent reportも同一provenanceで検証し、frame promotion chainとsample-count promotionの両方をload前に再検証する`--promotion-parent-report`をMLX-Gen video CLIへ追加
- `[Done]` Wan 49-frame profileの4-sample stability gate。Apple M4/32 GiB、640×384、49 frame、20 stepsで4/4成功、失敗0、全件memory pressure normal／thermal fair、最大peak RSS 11,533,691,012 bytes、RSS range 0、median wall 1,098,215 ms、4 digest distinct、prompt/output非保存、private cleanupを確認。証跡: [49-frame 4-sample stability](evaluation/wan-mlx-gen-640x384-49frames-4sample-stability-2026-09-21.json)
- `[Next]` 33-frame root＋49-frame 4-sampleのchained promotion contractで、他条件を固定した65-frame・2-sampleを一軸評価する。全件normalの場合だけ4-sample安定化へ進む
- `[Later]` HunyuanVideo 1.5 8.3Bを候補とする480p、step-distilled、SSTA、model offload検証
- `[Later]` Wan 2.2 A14B量子化版をstretch候補とするT2V/I2V別artifact、dual-expert residency、CPU/SSD offload検証
- `[Done]` video diffusion pipelineのDiT/expert、text encoder、3D VAE別artifact admissionとconservative resident-memory hard ceiling
- `[Done]` privacy-preserving動画生成qualification report schemaとdeterministic evaluator（first-output/wall latency、peak RSS、memory pressure、thermal state、frames/sec、output metadata、plan fingerprint）
- `[Done]` backend-neutralなbounded telemetry event contractとconstant-memory sample collector
- `[Done]` shellを介さないbounded JSONL subprocess telemetry adapterとtimeout時process-group停止
- `[Done]` workspace-bound one-shot worker request、prompt digest binding、0600 atomic保存、consume後unlink
- `[Done]` Diffusers sourceのbounded AST scanによる6候補pipeline class readiness gate（backend import/model/Metal allocationなし）
- `[Later]` MLX、Diffusers、ComfyUI固有workerからqualification sampleを取得するadapter
- `[Done]` 最小profile合格後だけ解像度、frame数、steps、連続生成を一軸ずつ増やす段階的memory-stability gate。Wanで33-frame 2/4-sample合格後にだけ49-frameを許可し、同形状2→4 sampleはparent promotion chainも再検証する
- `[Done]` 2段目以降のvideo frame promotion contract。直前の4-sample安定profileと初期4-sample rootをcandidate/shape/all-normal条件で再検証し、両plan digestをchain identityへ結合する。これにより49-frame安定化後の65-frame候補を初期証跡なしでは起動できない
- `[Done]` model license、量子化方式、変換元digest、workflow provenanceを記録し、weightと生成動画を保存・uploadしないprivacy gate。Wan実機reportでartifact root digest、backend/version、mixed Q8/BF16、licenseを固定し、prompt/MP4非保存とprivate cleanupを確認

## Phase 7 — Generative Media

- `[Done]` 3候補のbounded初期profile catalogと共通load前artifact/Unified Memory admission
- `[Done]` image generation workload。local-only Diffusers MPS worker、bounded telemetry、private output digest/delete、artifact/provenance gateをQwen-Image-2.1 INT8の実生成で確認
- `[Done]` M4/32GB向け画像生成qualification profile。512×512、batch 1、20 steps、独立2 sampleをApple M4/32 GiBで実測合格
- `[Later]` FLUX.2 [klein] 9B Baseを優先候補とする量子化、text encoder分離、VAE tiling、逐次module residency検証
- `[Done]` MLX-Gen互換Z-Image Turbo 4-bitを優先候補とするbackend readiness、512×512・9 steps実機qualification
- `[Later]` Qwen-Image-2512を候補とするMPS/MLXまたは対応backendの量子化、offload、peak Unified Memory検証
- `[Done]` 配置済みQwen-Image-2.1 revision `b3179ad355be050328e483a9dfdd9e60cd62adfa`を画像生成候補catalog、Diffusers静的pipeline readiness、isolated worker routingへ追加。loadなし検査でDiffusers形式、`QwenImage21Pipeline`、33,131,616,240 bytes、27 files、BF16非量子化、transformer 14.23 GB、text encoder 17.53 GB、VAE等1.35 GBを確認。公式Hub APIとの照合でlocal revision一致、非gated、BF16 7,115,124,736 parametersを確認。Qwen Research Licenseは非商用限定。証跡: [Qwen-Image-2.1 readiness](evaluation/qwen-image-2.1-artifact-readiness-2026-09-20.json)
- `[Done]` Qwen-Image-2.1隔離runtimeを`.venv-qwen-image-21`へ構築。Python 3.12.9、Torch 2.14.0、Transformers 5.17.0、Diffusers 0.41.0.dev0 commit `80c7ed262aeffbeb43ef13ae04baeb9b84515a69`を固定し、`QwenImage21Pipeline`のload-free source readinessとMPS buildを確認。再現用requirementsと証跡: [runtime requirements](../requirements/qwen-image-2.1-runtime.txt)、[runtime readiness](evaluation/qwen-image-2.1-runtime-readiness-2026-09-21.json)
- `[Done]` Qwen-Image-2.1 workerの逐次module residency contract。固定Diffusers sourceの`text_encoder->transformer->vae`宣言を照合し、`enable_model_cpu_offload(device="mps")`を必須化。契約不一致またはAPI欠落時は全pipeline `.to("mps")`へfallbackせず生成前に拒否し、VAE tilingとbounded callback telemetryを維持
- `[Done]` Qwen-Image-2.1専用bounded Diffusers image qualification CLI。512×512・batch 1・20 steps・2 sampleを初期値とする。Diffusers model CPU offloadは非active moduleのweightもCPU側へ保持し、Apple Siliconでは同じUnified Memoryを消費するため、総artifact weight floor + 1 GiB allocator margin + image working setをresident bytesとして自動見積りする。pipeline/source readiness、load-free artifact identity、dynamic Unified Memory ceiling、artifact root digestを生成前に照合し、既存workerのprivate PNG digest・即時削除へ接続
- `[Done]` Qwen-Image-2.1のload-free量子化residency planner。denoiser/text encoderだけをINT8またはINT4へ変換し、VAE/その他をBF16に維持するcomponent別projectionを生成する。INT8理論値はweight 17,249,254,204 bytes、allocator/image余白込み18,327,190,332 bytesでphysical safe ceiling 31,610,959,299 bytes内。ただしscale/metadata overheadとruntime kernelは未実証のため生成可能とは判定しない。証跡: [INT8 residency plan](evaluation/qwen-image-2.1-int8-residency-plan-2026-09-21.json)
- `[Done]` 隔離runtimeへTorchAO 0.18.0を固定し、Diffusers `PipelineQuantizationConfig` / `TorchAoConfig`、TorchAO `Int8WeightOnlyConfig`、小型LinearのINT8変換とCPU forwardをload-free probeで確認。変換準備とMPS実行準備を分離し、今回のprobeではMPS build=true、availability=falseのため`conversion_ready=true`、`mps_runtime_ready=false`を記録。証跡: [TorchAO readiness](evaluation/qwen-image-2.1-torchao-readiness-2026-09-21.json)
- `[Done]` Qwen-Image-2.1 INT8 conversion plan。source/output分離、既存output拒否、symlink/file-count境界、最大safetensors shard、TorchAO readiness、動的memory ceiling、15% staging込みdisk admissionをweight load前に統合。配置済みartifactは最大shard 9,968,332,504 bytes、変換peak推定28,295,522,836 bytes、必要available memory 31,044,301,905 bytes、必要disk 19,836,642,335 bytes。現状はmemory gateだけで安全停止しoutputは未作成。証跡: [INT8 conversion plan](evaluation/qwen-image-2.1-int8-conversion-plan-2026-09-21.json)
- `[Done]` Qwen-Image-2.1 INT8 atomic conversion workerと変換後artifact inspector。正式outputと同一filesystemに一意なstagingを作り、Diffusers pipeline-level TorchAO `Int8WeightOnlyConfig`をtransformer/text encoderへ適用してsafetensors保存する。`QwenImage21Pipeline` identity、TorchAO INT8 weight-only metadata、両componentの量子化、artifact root digestを再検査し、全条件合格時だけatomic renameする。変換例外・片側だけの量子化・metadata不一致ではstagingを除去して正式outputを残さない
- `[Done]` conversion plan合格時だけ隔離runtimeのatomic workerを起動するbounded CLI orchestration。stdout/stderr各64 KiB上限、24時間以内timeout、shellなしargv、独立process group、timeout/oversize時TERM→KILLを実装。実artifactでの確認はdynamic ceiling 15,364,437,443 bytesに対し変換peak 28,295,522,836 bytesだったため`started=false`となり、weight loadもoutput作成も行わなかった
- `[Done]` transformerとtext encoderを別child process／同一atomic stagingで順番に変換し、各process終了時にOSへmemoryを返すcomponent-streaming INT8 converter。safetensors headerから最大tensor working setを検証することで変換peakを11,085,606,043 bytes、必要available memoryを13,834,385,112 bytesへ精密化。unquantized VAE/processor/schedulerだけをコピーし、component別TorchAO保存後に共通inspectorとdigest gateを通して18,516,268,404-byte／26-file artifactへatomic昇格。root digest `dcacfc334ed0821c18a1ff079e37cae9d99de0966c821c508a145134e85f38cf`
- `[Done]` Qwen-Image-2.1 TorchAO INT8正式2-sample qualification。Apple M4/32 GiB、512×512、20 stepsで2/2生成成功、memory pressure全件normal、thermal fair、最大peak RSS 2,599,387,136 bytes、median wall 216,912 ms。prompt/outputを保存せずprivate directory cleanupを確認。証跡: [INT8 2-sample qualification](evaluation/qwen-image-2.1-torchao-int8-2sample-qualification-2026-09-21.json)
- `[Done]` Qwen-Image-2.1 INT8 4-sample stability gate。Apple M4/32 GiB、512×512、20 stepsで4/4成功、memory pressure全件normal、thermal fair、最大peak RSS 2,600,943,616 bytes、RSS range 2,228,224 bytes（最小値比約0.086%）、median wall 257,631 ms。4 output digestはすべて異なり、prompt/output非保存とprivate cleanupを確認。証跡: [INT8 4-sample stability](evaluation/qwen-image-2.1-torchao-int8-4sample-stability-2026-09-21.json)
- `[Done]` Qwen-Image-2.1 INT8の一軸promotion。Apple M4/32 GiB、768×768、20 steps、独立2 sampleで2/2成功、最大peak RSS 2,599,305,216 bytes、median wall 494,727 ms、thermal fair、異なる2 output digest、prompt/output非保存とprivate cleanupを確認。第2 sampleの終了時memory pressureはwarningだったがqualificationの失敗条件には抵触せず、issuesは空。証跡: [INT8 768×768 2-sample promotion](evaluation/qwen-image-2.1-torchao-int8-768-2sample-promotion-2026-09-21.json)
- `[Done]` 768×768・20 stepsの2-sample baselineをmemory pressure normalの回復状態から再取得。2/2成功、最大peak RSS 2,582,462,464 bytesだったが、再び第2 sample終了時にwarningを再現。warningを含むbaselineからの4-sample昇格はload前に安全停止するため、768×768を安定profileへは昇格せず512×512を認定上限に維持する。証跡: [INT8 768×768 recovery attempt](evaluation/qwen-image-2.1-torchao-int8-768-2sample-recovery-2026-09-21.json)
- `[Done]` Qwen-Image-2.1 INT8の768×768 warning切り分けのうち、workerによる`pipeline`破棄・`torch.mps.empty_cache()`・独立process終了と、runnerによる次sample開始前のnormal連続観測は実装済みであることを確認。その状態でも第2 sample中のwarningが2回再現したため、単純なcache解放漏れ・開始前回復不足は原因候補から除外
- `[Done]` 生成workerのeffective resident telemetryにPyTorch MPSの`current_allocated_memory` / `driver_allocated_memory`を追加。OS process peakとMLX allocator peakに加え、MPS driver allocationの最大値をhard-ceiling判定へ反映し、probe不可時だけfail-softする
- `[Done]` Qwen-Image-2.1 INT8の明示的text-encoder phase解放。model offload下でprompt embeddingを先に確定し、hook除去後にtext encoder/tokenizer参照を破棄、GC・MPS synchronize/cache解放後に`transformer->vae` offload chainを再構成する。解放前後を既存bounded telemetryで観測し、API欠落時はfail-closedする回帰テストを追加
- `[Next]` 上記staged releaseを768×768・20 stepsの独立2 sampleで再qualificationし、MPS allocator peak、全sampleのmemory pressure normal、生成結果、private cleanupを確認する。合格後だけ4-sample stabilityへ昇格する
- `[Later]` FLUX.2 [dev]をstretch候補とする4-bit級量子化、CPU/SSD offload、chunking検証（非量子化weightはM4/32GBでload前にreject）
- `[Done]` diffusion pipelineのmodel、text encoder、VAE別artifact admissionとconservative resident-memory hard ceiling
- `[Done]` privacy-preserving画像生成qualification report schemaとdeterministic evaluator（first-output/wall latency、peak RSS、memory pressure、thermal state、output metadata、plan fingerprint）
- `[Done]` backend-neutralなbounded telemetry event contractとconstant-memory sample collector
- `[Done]` shellを介さないbounded JSONL subprocess telemetry adapterとtimeout時process-group停止
- `[Done]` workspace-bound one-shot worker request、prompt digest binding、0600 atomic保存、consume後unlink
- `[Done]` Diffusers sourceのbounded AST scanによる6候補pipeline class readiness gate（backend import/model/Metal allocationなし）
- `[Done]` FLUX.2/Qwen Image向けDiffusers image worker execution core、pipeline identity gate、streaming output hash、qualification後削除
- `[Done]` local-only Diffusers MPS text-to-image runtime、BF16 compute、VAE tiling、step telemetry、one-shot executable
- `[Later]` Diffusers image-editとWan/HunyuanVideo video worker adapter
- `[Later]` MLX、ComfyUI固有workerからqualification sampleを取得するadapter
- `[Done]` 512×512合格後だけ768/1024と連続生成へ進む段階的memory-stability gate。Qwen-Image-2.1 INT8で512×512・2/4 sample合格後に768×768のみ一軸promotionし、warning再現により4 sample/1024昇格をload前停止
- `[Done]` model license、gated artifact、quantization provenanceを記録し、weightと生成画像を保存・uploadしないprivacy gate。Qwen-Image-2.1 reportにartifact root digest、TorchAO INT8、backend/version、licenseを固定し、prompt/PNG非保存とprivate cleanupを確認
- `[Later]` audio and music generation workload
- `[Done]` video generation workload。Wan 2.2 TI2V-5BのMLX-Gen local-only worker、bounded telemetry、T2V qualification、private MP4 digest/delete、33-frame 4-sample stability、49-frame promotionをApple M4/32 GiBで実証
- `[Later]` latent memory manager
- `[Later]` temporal/spatial attention state
- `[Later]` tile、frame chunk、temporal chunk scheduling

## Phase 8 — MoE and Large Models

- `[Later]` expert residency manager
- `[Later]` expert selection telemetry
- `[Later]` correctness-neutral expert predictor
- `[Later]` SSD expert tier
- `[Later]` hierarchical KV cache
- `[Later]` large Unified Memory optimization
- `[Later]` cold prefix、vision、video embedding cache

## Phase 9 — Multi-Mac

- `[Later]` Thunderbolt transport
- `[Later]` high-speed Ethernet transport
- `[Later]` pipeline parallel execution
- `[Later]` modality parallel execution
- `[Later]` distributed KV/state
- `[Later]` topology-aware partitioning
- `[Later]` node failure recovery

## Cross-Cutting Track — Portable Numeric Formats / Streaming Conversion

NVFP4など実行先が直接扱えない数値形式を、保存形式・runtime表現・演算／累積形式に分離する。
NVFP4 → INT8を最初の候補としつつ、FP16/BF16展開、既存MLX量子化表現、直接fused演算を比較する。
「INT8を格納できる」と「対象演算がINT8で高速に動く」は別capabilityとして判定する。
仕様は[設計書 §42.1](Design-Specifications.md#421-portable-numeric-format-layer)に定義する。

- `[Done]` versioned NumericFormatDescriptor／ConversionPlanの最小契約とNVFP4 1D・16要素blockのCPU参照decode／scale付きINT8変換（全16値・非負有限scale全127値の数値一致、小規模上限）
- `[Done]` CPU参照変換のsource/target digest binding（plan・payload・block/global scale）、出力整合性検証と元入力からの再検証
- `[Done]` immutable複数adapter planning registry、完全一致route選択、重複ID・曖昧候補の拒否（組み込みはNVFP4 CPU参照routeのみ）
- `[Done]` C-order論理shape・単一scale軸のTensorGeometry、行境界を保持するscale index計算、geometry-bound planning ID（1〜8次元、参照上限内）
- `[Done]` geometry-aware NVFP4→INT8 CPU参照変換・decode、単一scale軸の境界処理、shape/axisを含む入出力digest検証
- `[Done]` scale付きINT8から既存MLX correctness converterへのF32参照bridge（shape保持・buffer予約検査・F32 overflow拒否、mock consumer検証）
- `[Done]` MLX 0.31.2実機bridge検証：E2M1全16コード、有限非負scale全127コード、2通りのscale軸、F32/F16/BF16の762ケースで出力digest一致（小規模・global scale=1）
- `[Done]` MLX実機のties-to-even・F16 subnormal/underflow・最大有限値検証、F16/BF16 overflowとNaN/Inf入力の拒否（選定境界ケース）
- `[Done]` opt-in NumericPrecisionPolicy（絶対/相対許容誤差・zero underflow許可）、backend実行前の参照判定と実行後digest照合、MLX実機762ケースの誤差ゼロpolicy検証
- `[Done]` version付きprecision policy/実行契約のstrict roundtrip・canonical ID、tensor digest/dtype/policy/bridge結合とMLX correctness呼び出し時の照合
- `[Done]` 変換evidenceへのprecision checked状態・contract/policy IDとbounded診断出力（tensor名・値・元digest・閾値は非保存、不完全evidence拒否）
- `[Done]` in-process resident storeのscale付きINT8 load境界、precision契約照合、source/F32 bridge scratch/destination一括admission、unload/quarantine lifecycle統合
- `[Done]` concrete MLX numeric resident backend：F32復元・policy参照bit/digest照合・F16/BF16/F32常駐resource・明示解放、実機end-to-end検証
- `[Done]` private bounded NVFP4 artifact（strict JSON・128 KiB・0600・owner・no-follow・digest/inode/size検証）、runtime `load_numeric`、MLX常駐/unloadまでのsocket実機end-to-end
- `[Done]` numeric artifactのone-shot claim/consume/quarantine lifecycle、64 KiB以下の安全なsource reader、producer／socket client／MLX worker CLI（生成・load・status・unload・shutdown）
- `[Done]` digest-bound bounded tile plan、最大2 bufferの明示lease・再利用時zeroize・cooperative cancel/cleanup、buffer/metadata/scratch/destination一括admission、runtime/CLI/MLX streaming load接続
- `[Done]` runtime socketのnon-consuming disconnect probeをstream safe pointへ伝播（監視threadなし、次command dataは消費しない）
- `[Done]` private 0600 packed/scale fileのowner・regular file・no-follow・size・SHA-256・inode/mtime再検証、unaligned tile対応NVFP4 incremental decode、全source materializationなしのbounded MLX常駐load
- `[Done]` manifest-last生成、3-file one-shot claim/consume/quarantine、runtime自動判別、CLI `--file-backed`によるpacked/scale file-backed provider直接接続
- `[Done]` request ID指定のout-of-band `cancel` control message、最大8接続の並行Unix socket受付、tile safe pointへのsignal合成、client/CLI cancel入口
- `[Done]` worker起動時の5分grace・4096 entry上限・identity再検証付きorphan companion quarantine
- `[Done]` cancel・consume・shutdownの実socket競合試験、shutdown時active connection即時回収、active/cancel/consume/quarantine/orphanのbounded診断（native INT8演算kernelは未実装）
- `[Done]` NVFP4 1D／2D（複数axis）16要素block scale、tensor global scale、low/high-first packed nibble順序をplan/digestへ結合するartifact adapterとCPU参照・file streaming実行（未知layoutはfail-closed）
- `[Later]` exporter固有のscale layout・padding・swizzleを明示識別・変換するNVFP4 artifact adapter
- `[Later]` MXFP4／MXFP6／MXFP8、FP8 E4M3／E5M2とvariant、FP16／BF16／FP32、signed/unsigned INT8／INT4／INT2の段階的対応
- `[Later]` NF4／codebook量子化、groupwise affine、zero-point、double quantization、mixed precision、outlier/residual・sparse表現の拡張adapter
- `[Later]` Safetensors／GGUF／MLX／Core ML artifactとGPTQ／AWQ／各exporterのmetadata・packing adapter（container、量子化recipe、演算形式を分離）
- `[Later]` weights／activations／KV・recurrent state／MoE expert／vision・audio・diffusion tensorを同一契約で扱うeligibility matrix
- `[Later]` CPU vectorized／MLX／Metalのdecode・repack・requantize・layout変換と、tile単位convert + GEMV/GEMM/attention融合
- `[Later]` asynchronous prefetch、backend completion barrier、file/compute overlapとUnified Memory bandwidth ceilingへの統合
- `[Later]` load時変換／初回利用時変換／反復利用cache／毎回fused変換を比較するcost modelとprefill/decode別route選択
- `[Later]` NVFP4 → scale付きINT8の表現保存経路と一般的な再量子化経路の比較、INT8演算・累積型・scale粒度の対応検証
- `[Later]` source/scale/layout/kernel/environment digestに結合した変換cache、bounded LRU、単一変換共有、quarantine・rollback
- `[Later]` chip／OS／toolchain／backend／operator／shape別capability probeとreference fallback、未知recipeは明示unsupported
- `[Later]` scalar誤差・operator誤差・モデル品質とend-to-end速度／peak memory／帯域／energyを組み合わせたpromotion gate
- `[Later]` CLI／Swift診断と英語・日本語・简体中文表示（source/runtime/compute形式、route、誤差budget、fallback理由）

範囲は広く定義するが、形式名だけで対応済みとしない。adapterごとにinspect/decode/convert/execute/qualifiedを
区別し、metadataと参照結果が確認できたvariantから順に有効化する。CPU/GPU/ANE schedulerは固定した
後半フェーズへ留めず、必要なcapability・計測・fallback契約が揃った項目から優先度を引き上げる。
この層の変換・同期costはresource ledgerへ取り込み、複数deviceの並列変換／演算を最適化する。

## Adaptive Track — CPU / GPU / ANE Heterogeneous Scheduling

AppleExecutionPlannerと既存のglobal schedulerを拡張し、CPU、GPU、ANEを個別backendではなく、
Unified Memoryとmemory bandwidthを共有する一つの実行系として管理する。ANEは公開Core ML APIで
実行可能な固定graph中心の処理だけを対象とし、dynamic LLM decodeを前提にしない。

- `[Done]` CPU、MLX GPU、Native Metal、Core ML/ANEのprofile-bound versioned capability・operator/phase/precision eligibility registry、sticky quarantine、probe ID binding、bounded fallback decision
- `[Done]` 既存kernel probe/cacheからoperator単位のdevice registryを構築するcomposition（MLX/Metal実測precisionをFP32へ限定、vector addをauxiliary phaseへ限定）
- `[Done]` optional dependencyなしのCPU vector add／8x8 matmul／KV copy bounded correctness・latency probeと共通composition統合
- `[Done]` 公開Core ML APIの`.cpuAndNeuralEngine` surface probe、bounded Swift subprocess、runtime report evidence（surface合格だけではexecution capabilityへ昇格しない）
- `[Done]` private requestとbounded outputを使うCore ML fixed-graph correctness・latency probe adapter、model tree SHA-256をoperator/probe identityへ結合、前後integrity再検証、auxiliary/FP32限定composition
- `[Done]` repository-owned決定論的`x * 2` Core ML fixture generator、coremltools 8.1／NumPy 1.26.4固定、coremlcompiler・tree integrity・3 sample ANE probe・生成物削除を行うself-hosted workflow
- `[Done]` M4ローカル実機でfixture生成／compile／`.cpuAndNeuralEngine` prediction 3回の数値一致（中央値約96.5 µs、fixture tree digest binding）
- `[Done]` probe合格済みmodel digest／capability ID／auxiliary phase／FP32 eligibilityを必須にするCore ML fixed-graph resource load・bounded execute・integrity再検証・明示unloadとscheduler直前dispatch gate
- `[Done]` Core ML load／predictionをSwift subprocessへ隔離するpersistent ANE worker adapterと、hardware／OS／model tree SHA-256／I/O identity別のbounded process cacheをfixed-graph backend lifecycleへ接続。resourceのload／unloadを跨いで同一workerを再利用し、active leaseをevictせず、idle workerだけをLRU回収し、容量超過・identity不一致・active closeをfail-closedにする
- `[Later]` Core ML compiler出力そのものを再利用する場合の、toolchain／Core ML versionを含むdisk artifact cacheと署名・失効policy
- `[Done]` CPU thread、GPU command queue、ANE in-flight task、Unified Memory、memory bandwidthを原子的に予約・解放する統合resource ledgerとruntime snapshot
- `[Done]` backend exchangeのversioned dispatch contract。architecture／precision／phase／operator／isolation、deadline／cancel safe point、retryable bounded fallback、逆順shutdownを共通化
- `[Later]` operator graphの依存関係とdevice間同期costを含むproduction composition rootへの共通dispatch contract接続
- `[Done]` probe済みcapabilityに限定したCPU/GPU/ANE共通bounded microbenchmark schemaとrunner（cold load、execution、変換、同期、throughput、peak memory、energyのunknown保持、output digest安定性）およびprivate atomic report・strict再計算loader
- `[Done]` deterministic CPU vector add／matmul／KV copy、MLX／Metal probe kernel、loaded Core ML fixed graphを共通schemaへ接続するnative measurement adapter（kernel時間とsubprocess込みend-to-end時間を分離）
- `[Done]` available・FP32 capabilityだけからCPU/MLX/Metal/Core ML代表shapeを構築するbounded deterministic suite、明示operation map、欠損operation拒否、suite ID
- `[Done]` 現在のMacでCPU vector add 256要素、8x8 matmul、KV copy 128要素を各3 sample実測し、安定output digestとsuite report生成を確認
- `[Done]` 現在のM4／macOS 26.6.2／MLX 0.27.1でCPU 3・MLX 6・Metal 2・Core ML/ANE 1 capabilityを各3 sample再qualificationし、12/12 correctness合格（ANE prediction中央値81,291 ns）
- `[Done]` self-hosted macOS ARM64でCPU 3／MLX 6／Metal 2／Core ML 1 operatorを各3 sample測定し、全probe必須gate、private atomic report、14日artifact retention、fixture/report削除を行うheterogeneous qualification workflow
- `[Done]` capability correctness probeとend-to-end promotion性能判定を分離し、subprocess起動costだけで正しいGPU kernelをbenchmark前に誤隔離しないqualification gate
- `[Done]` 同一profile／operator／phase／precision／shape／batchのreportだけを比較し、最低3 sample、output digest一致、peak memory非悪化、cold load償却込み5%以上のend-to-end latency改善を要求するdevice promotion gate
- `[Done]` accelerator固有operatorへ同一identityの決定論的CPU baselineを供給するbounded reference adapterとFP32 digest正規化
- `[Done]` M4上の4要素Core ML fixed graphをCPUと各3 sample比較し、出力一致を確認した上で起動cost非改善のためCPU維持（ANE誤昇格なし）
- `[Done]` M4実機でのMLX/Metal/Core ML代表shape qualification（CPU 3・MLX 6・Metal 2・Core ML/ANE 1の12/12 correctness合格。energyとpeak memoryは実測sourceがある場合だけ記録）
- `[Later]` prefill、decode、Vision/Audio encoder、sampling、draft/verify別のend-to-end performance profile
- `[Done]` promotion winnerのbenchmark report／capability／exact workload identityを結合するversioned device placement plan、probe profile一致検証、active/pending scheduler safe-point適用、reservation plan ID
- `[Done]` device placement planのprivate atomic persistence、全field／plan ID再計算、最大30日TTL、future／expired拒否、current破損時last-known-good fallback
- `[Done]` runtime snapshot／JSON Schemaへactive・pending plan、有効期限、最大64件の非機密placement metadataを追加
- `[Done]` runtime probe／dispatcher準備後のdaemon起動時placement current／last-known-good自動restore、profile専用path、applied／deferred／not-found／rejected rollback event
- `[Done]` 新plan昇格時に同一profile・未期限切れcurrentだけをlast-known-goodへ保存し、破損planによるrollback slot汚染を防ぐatomic promotion
- `[Done]` strict qualification report、CPU baseline、複数candidate、cold-load償却、改善率、memory gateからplacement planを生成しcurrent／last-known-goodへatomic promoteする管理CLI
- `[Done]` SIGHUP signal handlerからfile I/Oを分離したdaemon非同期reload、probe未準備拒否、active request中safe-point defer
- `[Done]` 認証付き`POST /v1/device-placement`のstrict reload／rollback管理API、profile専用file control、safe-point deferred受付、bounded event
- `[Done]` 管理API応答とplacement CLIの英語・日本語・简体中文diagnostics、stable message key
- `[Done]` runtime placement event／snapshotのSwift SDK typed model、旧client向け既定値、strict evidence検証と三言語Mac app表示
- `[Done]` 昇格済みANE routeのtimeout／数値不一致を固定codeへ変換し、probe済みGPUからCPUまで実行するbounded end-to-end fallback contract
- `[Done]` 1024幅×16層dense+ReLU代表encoderの決定論的generator／CPU reference／integrity-bound Core ML qualification、同一digest benchmarkによるANE placement昇格とruntime fallback
- `[Done]` Core ML modelをprocess内で保持するbounded persistent Swift worker、resource load／unload連動、timeout／worker failureのretryable fallback変換
- `[Done]` hardware／OS／model tree SHA-256／I/O identityに結合したbounded Core ML worker cache。同一identityはpersistent workerを共有し、active lease保護、idle-only LRU eviction、厳密なclose accountingを実装
- `[Later]` Core ML compiler artifactのdisk再利用を行う場合のtoolchain／Core ML version binding、署名、quarantine、失効
- `[Done]` M4実機でpersistent workerの連続5 prediction出力一致（load約235ms、end-to-end 0.84–1.41ms、kernel 16–84µs）
- `[Done]` Core ML/ANE in-flight、CPU thread、GPU command queue、Unified Memory、bandwidth slotを同一admissionで原子的に予約し、失敗時にmemory予約をrollbackするresource ledger
- `[Done]` fallbackごとのbackend resource原子的引き継ぎと、profile-bound・3 sample以上・出力一致・5%以上の並列改善を必須にするfail-closed contention evidence gate
- `[Done]` CPU／GPU／ANE逐次・並列medianを測定するbounded contention adapter、private strict profile、digest再計算、hardware一致した全件qualified profileのruntime起動時install
- `[Done]` M4実機5 sample contention qualification（output digest全一致、CPU+MLX 40.2%、CPU+ANE 22.8%、MLX+ANE 5.53%改善で3/3組を昇格）
- `[Done]` hardware profile ID別private既定path、daemon startup strict restore、not-found/rejected/applied event、runtimeのprofile ID・認定pair数診断
- `[Done]` contention runtime診断のstrict typed Swift SDK、旧client unavailable fallback、Mac app英語・日本語・简体中文表示（Swift 33 tests・Mac sample build合格）
- `[Done]` contention profileのvalid-current-only last-known-good promotion、破損時fallback、safe-point reload／rollback、認証付きstrict管理APIと三言語応答
- `[Done]` contention reload／rollbackのtyped Swift SDK、strict evidence検証、Mac app英語・日本語・简体中文操作UI
- `[Done]` probe承認済みfallbackとcontention認定がある場合だけqueue先頭をidle backendへ移すbounded work stealing（resource原子予約、容量不足時FIFO復元）
- `[Done]` memory pressure、thermal state、low-power modeに応じた新規admissionのconcurrency／batch／device割当の段階的縮退、safe-point回復
- `[Next]` Vision/Audio encoderとembedding/classifierから開始するANE routing、GPU LLM pipelineとの非同期連携
- `[Later]` CPUまたはANE draft + GPU verifyによるheterogeneous speculative execution
- `[Done]` contention全ペア合格時だけ原子的に一括予約するCPU／GPU／ANE bounded pipeline並列化
- `[Done]` hardware／OS／model／shape別autotuning profile。model-backed shape、hardware/source/environment fingerprint、private atomic profile、OS/toolchain/MLX変更時の失効、TTL、quarantine、明示的再計測restore、last-known-good rollbackをnative v2 tuningとdevice placementで実装
- `[Done]` backend別correctness比較、timeout／compile failure／numerical mismatch時のANE → GPU → CPU fallback。同一workloadのCPU reference digestと数値誤差を昇格時に検証し、昇格済みCore ML routeの固定retryable codeからprobe済みGPU、CPUへbounded resource引き継ぎでfallback
- `[Later]` TTFT、TPOT、tokens/sec、frames/sec、energy/request、peak Unified Memoryを用いたpromotion gate
- `[Done]` device assignment、queue wait、fallback、contention、thermal/power decisionの固定キー・上限付きruntime observabilityとstrict schema
- `[Done]` scheduling observabilityのtyped Swift SDK、旧runtime unavailable fallback、Mac app英語・日本語・简体中文diagnostics
- `[Done]` 安全上のthermal／memory縮退を維持した自動／省電力／最高性能policy選択の認証付きruntime管理API、typed Swift SDK、Mac app三言語操作UI（runtime内の選択）
- `[Done]` scheduling preferenceのprivate・上限付きatomic永続化、daemon再起動時の復元と不正設定時のautomatic fallback

## Cross-Cutting Work

### Reliability

- `[Done]` context計算のunderflow防止
- `[Done]` schedulerのhard memory ceiling
- `[Done]` profile書き込みのatomicity
- `[Done]` bounded HTTP concurrency
- `[Done]` backend processのreadiness、exit、timeout監視
- `[Done]` bounded backend log buffer
- `[Done]` bounded event historyとsubscriber ceiling
- `[Done]` structured runtime error taxonomyとrecoverability
- `[Done]` raw detailを公開しないfailure fingerprintとSwift typed decode
- `[Done]` private/atomic daemon crash diagnosticsとbounded log digest
- `[Done]` optional native telemetry登録をcontrol readinessから隔離
- `[Done]` Swift ManagedRuntime restart testを決定的なUDS fault-injection fixtureへ分離
- `[Done]` host memory pressureから隔離した決定的なmonitor transition回帰fixture
- `[Done]` 固定enum、最大32 rule、指定hit、one-shot／repeat、saturating不要のbounded counterによる決定論的fault injection基盤。backend executeのretryable／fatal／timeoutとstop回収へ接続し、非機密snapshotとfallback非回帰を実装
- `[Later]` profile persistence、scheduler admission、worker crash、client切断を横断するplatform-wide fault scenario matrix

### Security

- `[Done]` localhost-only default
- `[Done]` request size limit
- `[Done]` profile fileのprivate permission
- `[Done]` UDS permissionとsession authentication
- `[Done]` model tree全regular fileのstreaming SHA-256 manifest生成・起動前検証（symlink/special file拒否、変更検出、bounded走査）
- `[Done]` detached CMS trusted manifest署名、trusted CA chain、signer SHA-256 identityのmodel load前検証
- `[Done]` remote TLSとBearer API session token（非loopbackの明示opt-in、TLS identity・認証必須、秘密鍵安全性検査）
- `[Done]` optional mTLS client certificate identity gate。owner検証済みclient CA指定時は`CERT_REQUIRED`を設定し、Bearer tokenと併用する
- `[Done]` mTLS subject／SAN authorization mapping、serial revocation、atomic policy file差し替えによる無停止rotation。CA handshakeとBearer tokenへ追加で適用し、owner-only／no-symlink／bounded `O_NOFOLLOW` read、変更中・破損・消失時fail-closed、非機密count snapshotを実装

### Observability

- `[Done]` runtime state、memory、scheduler snapshot
- `[Done]` backend failureのruntime state反映
- `[Done]` reconnect可能なruntime state/failure SSE event
- `[Done]` request IDとstructured logging（HTTP/UDS応答、backend proxy、SSE、request-scoped eventを相関し、本文非保持・256件上限のメモリ内JSON recordを実装）
- `[Done]` tokens/sec、TTFT、TPOTのbounded phase profile
- `[Done]` Unified MemoryとKV usage（OS/framework/KVを分離し、source付きused/capacity/ratioをruntime snapshotへ公開）
- `[Done]` allocator内peakとOS backend resident peakのsigned差分metric
- `[Later]` operator別backend選択、fallback、quarantine telemetry
- `[Later]` GPU/CPU utilization、bandwidth、thermal、power
- `[Later]` Vision、Audio、Video固有metrics

### Packaging and release

- `[Done]` 日本語・英語・简体中文の製品Webサイトとレスポンシブなアプリ画面・キャラクター表示
- `[Done]` 三言語`hreflang`相互参照と英語`x-default`によるlocale discovery
- `[Done]` 言語metadata、操作contract、ローカルasset、動的状態文言を固定するWebサイト回帰テスト

- `[Done]` Python package metadataとCLI entry points
- `[Done]` Swift Package metadata
- `[Done]` supported Python、vLLM、vLLM-Metal、Transformers version matrix
- `[Done]` current platform module/classとCPU fallbackの起動前検出
- `[Done]` sampling・streaming昇格probeと30分qualification前段統合
- `[Done]` 未昇格versionを通常matrixから分離し、exact version照合・Metal platform・30分証跡を必須化するcandidate stack昇格workflow
- `[Done]` qualification reportへのbackend stack version結合とSwift側の昇格証跡必須gate
- `[Done]` Swift checkerでの期待candidate stack三version完全一致再検証
- `[Done]` vLLM candidate昇格modelの前後integrity・任意CMS provenance再検証
- `[Done]` 認定証跡を再検証しmodel IDをSHA-256化する決定論的promotion bundle
- `[Done]` promotion bundleのdetached CMS署名とCA chain・signer identity・元証跡一括検証
- `[Done]` Swift SDKのpromotion bundle typed decode・CryptoKit SHA-256・bundle ID再計算
- `[Done]` Mac sampleの認定directory自動検出と三言語promotion verified表示
- `[Done]` Mac directory pickerからのprivate staging・copy後再検証・bundle ID atomic import
- `[Done]` macOS Security frameworkによるdetached CMS・custom CA trust・signer SHA-256 native検証
- `[Done]` Mac importerの署名必須modeとhash-only／trusted-signature三言語状態分離
- `[Next]` vLLM 0.28.x / Transformers 5.15.x実環境での昇格試験
- `[Done]` Python 3.12既定値とdevelopment dependency lock
- `[Done]` MakefileによるPython／Swift一括check
- `[Done]` Python 3.10/3.12/3.13とSwift/macOS CI
- `[Done]` manual/self-hosted Apple Silicon実device qualification workflow
- `[Done]` Metal platform、memory pressure、GPU coreのqualification preflight
- `[Done]` qualification reportのSwift typed decodeを必須化するworkflow gate
- `[Done]` MLX LM OpenAI server adapterとgreedy repeat/stream equivalence qualification gate
- `[Done]` Gemma 2 2B IT MLX 4-bit実機qualification smoke（20秒、193/193成功、9.61 req/s、RSS増加0、clean shutdown）
- `[Done]` Gemma 2 2B IT MLX 4-bitの30分memory stability qualification（12,722/12,722成功、7.07 req/s、RSS増加0、clean shutdown）
- `[Done]` soak memory上限を終了時RSSではなく定常peak growthで判定
- `[Done]` backend-local MLX allocator/KV cache telemetry wrapper（4 KiB response、4,096 node traversal上限）
- `[Done]` MLX allocator、KV使用量、OS available memoryによるload後context再評価
- `[Done]` MLX tokenizer互換面と長context KV実測adapter（prompt本文/token ID非保持）
- `[Done]` Gemma 2 2B IT MLX 4-bitの128/1,024/4,096 token段階試験（retrieval 100%、KV実測約8.2 KiB/token）
- `[Done]` model有効context上限の自動/明示検出と生成予約を含む事前gate（`model_context_limit_exceeded`）
- `[Done]` MLX実測KV calibration readerとcontext recommendation反映（3段階/4K/identity/単調性gate、25%余裕、実測範囲cap）
- `[Done]` calibration reportのprivate/atomic保存とhardware fingerprint適合gate
- `[Done]` calibration artifactのApplication Support自動保存・bounded探索・最新適合report選択
- `[Done]` daemon起動時のbackend適合calibration自動適用とruntime snapshot provenance
- `[Done]` Swift SDKのcalibration provenance typed decodeとMac app多言語診断表示
- `[Done]` native measurement symbolによるGemma 2 2B IT実Mac profile生成（M4、1K/4K、全候補correctness合格）
- `[Done]` native v2 profileのbounded自動探索とlazy Primitive内request-local production dispatch適用
- `[Done]` patched vLLM-Metal serverでのGemma 2 2B BF16 end-to-end profile hit検証
- `[Done]` production shape captureからのbounded自動profile生成とprefill coverage
- `[Done]` exclusive maintenance leaseとsingle-flight idle tuning coordinator
- `[Done]` daemonでのobservation/helper発見とprofile適用時backend recycle
- `[Done]` observation更新監視とidle debounceによる同一shape再計測防止
- `[Done]` runtime snapshot/eventとSwift SDK/Mac appへのnative v2 tuning状態公開
- `[Done]` authenticated enable/disable/retry control endpointとMac app操作UI
- `[Done]` native v2 tuning preferenceのprivate永続化とdaemon再起動時復元
- `[Done]` profile適用後readiness失敗時のlast-known-good rollbackとquarantine
- `[Done]` quarantine診断のbounded snapshotとMac app表示
- `[Done]` quarantine retention policyと再計測合格後だけのexplicit restore gate
- `[Done]` explicit restore後のmanaged backend safe-point適用とreadiness再確認
- `[Done]` OS、toolchain、MLX version変更時のprofile失効と安全な再benchmark
- `[Done]` state/workspace統合budgetとMoE expert residency
- `[Done]` Qwen3.8-Flash-Next bounded metadata inspectionとcapability gate
- `[Done]` layer別Gated DeltaNet recurrent state計算
- `[Done]` QSA block/indexer stateとsparse retrieval budgetの`StateMemorySpec`拡張
- `[Done]` 512 expert、10 routed + 1 sharedのworking-set/residency計算
- `[Done]` 51B N-gram Embeddingを独立residency classとして計画
- `[Done]` Gated Residual scratch stateとMTP追加weight/state計算
- `[Done]` text-only、Vision、MTP、native 262K、YaRN 1Mを別capabilityとして判定
- `[Done]` quantized artifact実サイズによるMac適合判定とhard memory ceiling
- `[Done]` Native MLX architecture capability gate
- `[Done]` qualification reportへのbounded phase profile統合
- `[Done]` self-hosted qualification preflight/workflowのMLX切替
- `[Done]` MLX実model qualificationへのload前memory fit統合
- `[Done]` qualification phase metricsのSwift SDK/Mac app統合
- `[Done]` qualification artifactのphase/memory evidence CI検証
- `[Done]` 三言語bounded semantic smokeのqualification統合
- `[Done]` 三言語応答の本文非保存exact-match gate
- `[Done]` vLLM-Metal/MLX共通のqualification load前memory fit
- `[Done]` vLLM-Metalのbounded Qwen feature宣言probe
- `[Done]` 公式48層・36 GDN・12 QSA・Vision/MTP config照合
- `[Done]` 公式multimodal artifactのtext-only qualification mode
- `[Done]` requested modeを記録・再検証するSwift text-only evidence gate
- `[Next]` 大容量Apple SiliconでQwen text-only実model qualification
- `[Next]` 専用runner上でvLLM 0.28.x昇格workflowを実行
- `[Done]` locked PyInstallerによるApple Silicon standalone daemon生成
- `[Done]` standalone daemonを埋め込むunsigned Mac app release candidateとSHA-256検証CI
- `[Done]` ephemeral keychain、inside-out hardened runtime署名、公証、staple、Gatekeeper検証release workflow
- `[Done]` P12／password、P8 private key、Key ID／Issuer、一時keychain secretの非漏洩fail-fast検証
- `[Done]` macOS／Linux共通OpenSSL Base64 decodeによるrelease credential materialize
- `[Done]` notarized ZIP、内部実行file、Info.plist、公証結果、source commitを結ぶbounded release manifest
- `[Done]` GitHub artifact attestationとupload前manifest再検証
- `[Done]` notarization run artifactの再取得、四重evidence検証、tag/commit bindingによるdraft release昇格gate
- `[Done]` Swift SDKのbounded Mac release evidence verifierとstreaming archive SHA-256
- `[Done]` Mac appのrelease directory picker、三言語verified/failed表示
- `[Done]` model-aware qualification preflightのarchitecture feature・mode・Unified Memory hard ceiling統合
- `[Done]` Qwen／candidate stack／汎用Metal workflowのmodel load前fail-closed証跡
- `[Next]` protected `mac-release` environment上で実資格情報による初回notarized artifact生成
- `[Next]` 初回notarized artifactをexact tagへ結合しdraft release昇格を実行
- `[Later]` Ruff ruleの段階的拡張と既存style debt解消

## Recommended Immediate Sequence

直近の実装順序：

1. `[Done]` vLLM-Metal process adapterとversion compatibility check
2. `[Done]` OpenAI chat proxyおよびstreaming
3. `[Done]` standard Transformer model metadata inspectionとautomatic context設定
4. `[Done]` UDS、session authentication、bounded event stream
5. `[Done]` Swift UDS transport、ManagedRuntimeのcrash recoveryとlog capture
6. `[Done]` 最小SwiftUI Mac chat sample
7. `[Done]` concurrent load、bounded soak runner、daemon crash/relaunch test
8. `[Done]` 実modelのend-to-endと30分以上のmemory stability test
9. `[Done]` OptimizationPlan / ArtifactManifest schemaとsafe dry-run planner
10. `[Done]` optimizer duration profilerとstructured error taxonomy
11. `[Done]` representation optimization backend adapterとcapability detection
12. `[Done]` isolated conversion workerとatomic artifact lifecycle
13. `[Done]` persistent checkpoint manifestとresume decision protocol
14. `[Done]` cross-process leaseとworker resume integration
15. `[Done]` executable MLX exporterとversion compatibility gate
16. `[Done]` 小型実model export smoke、artifact provenance、resource計測
17. `[Done]` quantization baseline比較とperplexity quality regression gate
18. `[Done]` deterministic generation品質gate
19. `[Done]` 多言語期待応答と選択可能なcode、math、retrieval smoke suite
20. `[Done]` instruction model task scoreとchat template/token budget contract
21. `[Done]` bounded semantic anchor cache、RuntimeService、backend state adapter contract
22. `[Done]` AppleExecutionPlan schema、StateMemorySpec、deterministic dry-run
23. `[Done]` AppleChipProfile capability detectionとatomic persistence
24. `[Done]` prefill/decode別bounded profile schemaと集計器
25. `[Done]` 実backend stream instrumentationとprofile取得CLI
26. `[Done]` plannerとscheduler、elastic memory policyのatomic safe-point接続
27. `[Done]` 段階的long-context evaluation schemaとfail-fast memory coordinator
28. `[Done]` tokenizer準拠retrieval datasetと実backend long-context adapter
29. `[Done]` semantic cache elastic memory budgetとscheduler safe-point適用
30. `[Done]` kernel self-test、performance probe、backend quarantine foundation
31. `[Done]` Native Metal Paged Attentionと1K長context probe tier
32. `[Done]` model-backed Paged Attention shape profile
33. `[Done]` Native Metal shape profile consumerとallocation hard limit
34. `[Done]` 実Mac Metal deviceでの128-token shape probe
35. `[Done]` shape benchmarkのprivate/atomic/strict永続化
36. `[Done]` Qwen相当1K shapeの実Mac Metal計測
37. `[Done]` shape benchmark CLIとApplication Support既定保存先
38. `[Done]` 3-stage並列Metal Paged Attention kernel
39. `[Done]` Qwen相当1K shapeの旧single-thread比約132倍改善
40. `[Done]` constant-memory threadgroup softmax reduction
41. `[Done]` Qwen相当1K/4K shapeの実Mac測定
42. `[Done]` correctness-gated shape別thread幅autotuning
43. `[Done]` tuning中央値、2% tie-break、winnerのfingerprint別永続化
44. `[Done]` tuning report strict loaderとwinner再検証
45. `[Done]` tuning CLIとApplication Support既定保存
46. `[Done]` daemon winner適用safe pointとreservation-bound tuning ID
47. `[Done]` request-bound winner configurationのmanaged backend境界への伝播
48. `[Done]` compatible tuning reportのdaemon起動時自動install
49. `[Done]` backend request-local parserとkernel dispatch bridge
50. `[Done]` managed vLLM serverへのmiddleware登録とack telemetry
51. `[Done]` native v2 source/ABI inspectorと現行upstream非互換判定
52. `[Done]` native v2 kernel-family別profileとautotuning contract
53. `[Done]` native v2実kernel measurement adapterとC++ ABI
54. `[Done]` capability-gated vLLM-Metal benchmark helper bridge
55. `[Done]` bounded native v2実device profile生成CLI
56. `[Done]` model有効context上限の事前検出と8,192 token境界の明示的error分類
57. `[Done]` MLX実測KV係数をcontext recommendationへ安全側で反映
58. `[Done]` calibration reportのprivate/atomic保存とhardware fingerprint適合gate
59. `[Done]` calibration artifactのApplication Support自動探索と最新適合report選択
60. `[Done]` daemon起動時の適合calibration自動適用とruntime snapshotへのprovenance公開
61. `[Done]` Swift SDKでcalibration provenanceをtyped decodeしMac app診断へ表示
62. `[Done]` native measurement symbolの実Mac capability handshakeとmissing symbolの特定
63. `[Done]` vLLM-Metal `813e738d`向けnative measurement patch、実機build、Gemma 2 2B IT M4 profile生成
64. `[Done]` native v2 profileのbounded自動探索とlazy Primitive内request-local production dispatch適用
65. `[Done]` patched vLLM-Metal serverでのGemma 2 2B BF16 end-to-end profile hit検証
66. `[Done]` production shape captureからのbounded自動profile生成とprefill coverage
67. `[Done]` exclusive maintenance leaseとsingle-flight idle tuning coordinator
68. `[Done]` daemonでのobservation/helper発見とprofile適用時backend recycle
69. `[Done]` observation更新監視とidle debounceによる同一shape再計測防止
70. `[Done]` runtime snapshot/eventとSwift SDK/Mac appへのnative v2 tuning状態公開
71. `[Done]` authenticated enable/disable/retry control endpointとMac app操作UI
72. `[Done]` native v2 tuning preferenceのprivate永続化とdaemon再起動時復元
73. `[Done]` profile適用後readiness失敗時のlast-known-good rollbackとquarantine
74. `[Done]` quarantine診断のbounded snapshotとMac app表示
75. `[Done]` quarantine retention policyと再計測合格後だけのexplicit restore gate
76. `[Done]` explicit restore後のmanaged backend safe-point適用とreadiness再確認
77. `[Done]` OS、toolchain、MLX version変更時のprofile失効と安全な再benchmark
78. `[Done]` state/workspace統合budgetとMoE expert residency
79. `[Done]` Qwen3.8-Flash-Next bounded metadata inspectionとcapability gate
80. `[Done]` layer別Gated DeltaNet recurrent state計算
81. `[Done]` QSA block/indexer stateとsparse retrieval budgetの`StateMemorySpec`拡張
82. `[Done]` 512 expert、10 routed + 1 sharedのworking-set/residency計算
83. `[Done]` 51B N-gram Embeddingを独立residency classとして計画
84. `[Done]` Gated Residual scratch stateとMTP追加weight/state計算
85. `[Done]` text-only、Vision、MTP、native 262K、YaRN 1Mを別capabilityとして判定
86. `[Done]` quantized artifact実サイズによるMac適合判定とhard memory ceiling
87. `[Done]` Native MLX architecture capability gate
88. `[Done]` qualification reportへのTTFT、TPOT、tokens/sec、peak RSS profile統合
89. `[Done]` self-hosted Apple Silicon qualification workflowのNative MLX backend対応
90. `[Done]` MLX qualification process起動前のUnified Memory hard ceiling gate
91. `[Done]` phase profile/memory fitのSwift typed decodeとMac app三言語表示
92. `[Done]` self-hosted認定でphase/memory evidenceを必須化するSwift CI gate
93. `[Done]` 英語・日本語・简体中文の本文非保存semantic smokeとSwift evidence gate
94. `[Done]` incremental hash完全一致と16 MiB SSE上限によるconstant-memory quality判定
95. `[Done]` direct vLLM-Metal qualificationへのarchitecture/memory load前gate統合
96. `[Done]` vLLM-Metal明示architecture feature probeとdaemon/qualification昇格契約
97. `[Done]` 公式Qwen3.8-Flash-Next 48層configでのmetadata/state回帰固定
98. `[Done]` multimodal Qwenのrequested mode別text/Vision/MTP capability gate
99. `[Done]` requested mode別memory budgetとSwift text-only evidence gate
100. `[Done]` download前artifact admissionによる不適合な大容量model取得の防止
101. `[Done]` self-hosted認定へのartifact/resident対入力とload前fail-closed gate統合
102. `[Done]` Swift SDKでのartifact admission evidence再計算とCI必須gate
103. `[Done]` model-bound admission evidenceによる別候補report replay防止
104. `[Done]` 同名別量子化artifact間のadmission evidence replay防止
105. `[Next]` 大容量Apple SiliconでQwen text-only実model qualification
106. `[Later]` Mac companion app
107. `[Later]` M4/32GB画像生成qualification（FLUX.2 [klein] 9B Base、Qwen-Image-2512、量子化FLUX.2 [dev]）
108. `[Done]` M4/32GB動画生成qualification。Wan 2.2 TI2V-5B mixed Q8/BF16の640×384・33 frame・20 stepで2-sampleと4-sample stabilityに合格し、33→49 frame一軸promotionの2-sampleも合格。artifact digest binding、memory recovery、private cleanup、prompt／動画非保存を実機証跡で確認済み
109. `[Done]` 画像・動画6候補のbounded qualification plan schema、CLI、load前aggregate admission
110. `[Done]` denoiser、text encoder、VAE別容量証拠とaggregate完全一致によるload前fail-closed gate
111. `[Done]` 生成本文非保存の実測evidence evaluator、plan binding、private/atomic report保存
112. `[Done]` 最大4096 eventのconstant-memory生成telemetry collectorとstrict lifecycle検証
113. `[Done]` 16 KiB/event上限、24時間hard timeout、process-group cleanup付きJSONL backend境界
114. `[Done]` prompt非永続化の32 KiB one-shot生成worker request ABIとsymlink/race防御
115. `[Done]` Flux2、QwenImage、Wan、HunyuanVideo 1.5のDiffusers静的readiness CLI
116. `[Done]` Diffusers image qualification worker coreとworkspace-bound一時生成物のstreaming digest/delete
117. `[Done]` FLUX.2/Qwen Imageのlocal-only MPS Diffusers runtimeとisolated worker entrypoint
118. `[Done]` 実FLUX.2 [klein] checkpointで判明した`Flux2KleinPipeline` identityへのreadiness/runtime修正
119. `[Done]` MLX Diffusers変換/MFLUX artifactのbounded静的形式判定、component実容量集計、backend誤接続防止
120. `[Done]` MLX-Gen互換Z-Image Turbo 4-bit artifactの生成とM4/32GB最小profile qualification
120a. `[Done]` MLX-Gen 0.33.1のZ-Image Turbo capability検出、candidate-bound worker route、base-model固定、2-step未満のload前拒否
121. `[Later]` Qwen-Image-2512-4bitのMFLUX workerとoffload前提memory-stability qualification
122. `[Done]` MFLUX Z-Image/Qwen Image backend classとartifact形式を分離したloadなしreadiness gate
123. `[Done]` 配置済みMLX Diffusers変換artifactの量子化layer非互換を実機で特定し、別directoryへMLX-Gen 4-bit packageを生成する配置手順を確定
124. `[Done]` MFLUX Z-Image/Qwen Image local-only one-shot worker、private output digest/delete、memory ceiling telemetry接続
125. `[Done]` 配置済みFLUX.2 Klein 9B 4-bitのMLX-Gen形式、component実容量、非商用license provenance静的検査
126. `[Done]` MLX-Gen local-only FLUX.2 Klein worker接続
127. `[Done]` MLX-Gen 0.18.2+、console entrypoint、FLUX.2 Klein base identity、4-bit形式のload前readiness gate
128. `[Done]` MLX-Gen bounded JSON progressを用いたFLUX.2 Klein one-shot worker
129. `[Done]` MLX-Gen 0.33.1でFLUX.2 Klein 512×512実機最小profile qualification
130. `[Done]` FLUX.2 Klein routeで未対応の`--vae-tiling`を事前smokeで検出し、512 profileをautomatic VAE decodeへ修正
131. `[Done]` MLX-Gen JSON event捕捉とqualification telemetry出力のstream分離による再帰防止
132. `[Done]` M4/32GBで20-step完走、worker 2-step ABI完走、Peak MLX 7.76GB／Peak RSS 5.74GBの実測
133. `[Done]` 20-step worker evidenceのreport保存と連続2回memory-stability gate
134. `[Done]` private request生成、worker反復、streaming collector、atomic report保存を束ねるgenerative qualification runner
135. `[Done]` 連続sample間のpeak RSS差25%以内を要求するmemory-stability gate
136. `[Done]` worker hard ceilingへRSSとMLX allocator peakの最大値を反映するeffective-resident telemetry
137. `[Done]` FLUX.2 Klein Base 9B 4-bitの512×512・20-step連続2回実機合格（peak 7,761,058,338 bytes、memory pressure normal）
138. `[Done]` 連続qualification runnerの正式CLI化とhardware/backend/model provenance拡張
139. `[Done]` provenance付き正式CLIによるbaseline再発行とschema consumer回帰
140. `[Done]` generative reportのbounded race-safe strict loader、aggregate再計算、plan/provenance replay拒否
141. `[Done]` 現在のMac・MLX-Gen・local artifactに対するgenerative report strict verification CLI
142. `[Done]` 正式CLIのload-before-admission拒否にresident／hard ceiling／disk診断を追加
143. `[Done]` provenance付きFLUX.2 Klein baselineの現在Mac/backend/artifact照合合格
144. `[Done]` 512 baselineから一軸だけ変更する768解像度admissionと実機段階昇格（20-step独立2回、最大effective resident 10,886,404,598 bytes、memory pressure normal）
145. `[Done]` 合格済み同一provenance baselineのplan hashを後続planへ結合するresolution promotion gate
146. `[Done]` 768合格reportを起点に、解像度を固定した4回独立生成の一軸promotion gateと長時間memory-stability qualification（最大effective resident 10,886,420,556 bytes、peak spread 0.001%未満、4回目memory pressure warning）
147. `[Done]` bounded inter-sample memory pressure回復待ちと、全sample normalを必須にする次解像度promotion blocker
148. `[Done]` pressure recovery gate有効下での768×768・4回再qualificationとall-normal baseline取得（最大effective resident 10,886,404,598 bytes、全sample pressure normal）
149. `[Done]` all-normal 768 baselineと512 rootをchain digestで結合する1024解像度の二段階promotion admission
150. `[Done]` 二段階promotion admission通過後の1024×1024・20-step実機qualification（診断code有効の再試行でも1 sample目がruntime hard ceiling超過。M4/32GBの現profile上限を768に確定）
151. `[Done]` MLX-Gen形式のZ-Image Turbo 4-bitで512×512・9-stepを独立2回実行するmemory-stability qualification（最大effective resident 5,627,119,126 bytes、全sample pressure normal、thermal fair）
152. `[Done]` generative subprocessのstderrをdeadlockなしでdrainする4 KiB bounded tailと、秘密情報を含まないstructured worker failure code
153. `[Done]` 0.25 GB MLX cacheを独立candidate profileとしてplan hashへ結合し、通常baseline流用を拒否する512実機qualification（最大effective resident差65,154 bytes、0.00084%減に留まり1024昇格は見送り）
154. `[Done]` transformer compile解除とblockごとのMLX materialization/cache解放を独立profileへ結合した512実機qualification（最大effective residentは通常profile比76 bytes減に留まり昇格見送り）
155. `[Done]` fused SDPAの512-token attention query chunkingを独立profileへ結合した512実機qualification（最大effective residentはblockwiseと同値、通常比76 bytes減に留まり昇格見送り）
156. `[Done]` outer compileを維持したcombined QKV/MLP expansionの512-token sequence chunkingと独立512実機qualification（通常profileと生成hash・最大effective residentが完全一致し昇格見送り）
157. `[Done]` incremental block load・release barrier・compiled graph rebind・stable weight keyを必須にするweight residency feasibility gate（現MLX-Gen 0.33.1は前三契約がなくfail-close）
158. `[Later]` MLX-Gen側のblock streaming ABI実装後に行うweight residency profileと512 root再qualification
159. `[Done]` NSProcessInfo thermal stateとactive電源系統別power modeの検出、Python profile／Swift SDKの後方互換decode
160. `[Done]` thermal／powerをplan identityとdecision reasonへ固定し、scheduler safe-point適用可能なprefill batch縮退policyを実装
161. `[Done]` daemon lifecycleへ停止可能なthermal／power monitorを接続し、重複を除いたbounded state-change eventを公開
162. `[Done]` `runtime.operating_state`をSwift SDKのtyped payloadへ接続し、production decoder設定で回帰固定
163. `[Done]` RuntimeEventの標準／snake_case decoder互換とwire encode互換を復元し、Unixソケットで未知値後のevent継続受信を検証
164. `[Done]` thermal/power probe失敗・不正値をunknownへ縮退し、回復通知・handler再試行・停止中probe結果破棄を回帰検証
165. `[Done]` operating-stateイベントの並行更新順序とsnapshot整合性を保証し、発行失敗後の再試行で通知が失われないことを検証
166. `[Done]` RuntimeServiceの最新thermal/power・空きmemoryからexecution planをdry-run生成するpreview API（model spec必須、SoC/memory照合、active plan不変）
167. `[Done]` 認証付きGET /v1/execution-plan/previewとconfigured context上限の適用、対応vLLM-Metal daemonのchip profile接続
168. `[Done]` Swift SDKのHTTP／UnixソケットexecutionPlanPreview取得とtyped plan、dry-run・schema・memory上限・応答整合性の検証
169. `[Done]` preview未実装の独自Swift client向け既定実装と、HTTP認証route／Unixソケット取得の回帰テスト
170. `[Done]` execution-plan-preview-v1 JSON Schemaと成功／生成不可の排他的契約、live応答・認証拒否の回帰検証
171. `[Done]` Python planner生成の共通preview fixtureをSwift/Pythonで照合し、Swiftのplan ID長・backend・precision・decision reason検証をSchemaへ整合
172. `[Done]` ローカルdaemonのexecution-plan-preview CLI（private token file、1 MiB応答上限、redirect/proxyなし、JSONと終了code）
173. `[Done]` probe-bound Core ML fixed-graph resource lifecycle、bounded prediction、integrity再検証、明示unload、auxiliary dispatch gate
174. `[Done]` capability-bound CPU/GPU/ANE microbenchmark schema・runnerと、改変・profile混同を拒否するprivate atomic report
175. `[Done]` CPU／MLX／Metal／Core ML native measurement adapterとkernel／end-to-end時間の分離
176. `[Done]` capability由来のrepresentative device benchmark suiteとCPU 3-kernelローカル実測
177. `[Done]` CPU／MLX／Metal／Core ML全probe必須のself-hosted heterogeneous benchmark workflowとbounded report artifact
178. `[Done]` correctness・memory・cold-load償却・最小latency改善を要求するdevice promotion gate
179. `[Done]` benchmark-evidenced versioned device placement planとscheduler／RuntimeService safe-point適用
180. `[Done]` device placement planのstrict persistence・TTL・last-known-good fallbackとbounded runtime diagnostics
181. `[Done]` daemon startup placement restore・rollback eventとvalid-current-only last-known-good promotion
182. `[Done]` strict benchmark reportからplacement planを生成する管理CLIとSIGHUP safe-point reload
183. `[Done]` 認証付きplacement reload／rollback APIと英語・日本語・简体中文diagnostics
184. `[Done]` Swift SDKのtyped device placement snapshot／event、旧client互換とMac app三言語diagnostics
185. `[Done]` promoted ANE routeのtimeout／output mismatchをGPU→CPUへ縮退するbounded runtime fallback検証
186. `[Done]` MLX FP32 attentionのbounded数値比較と、capability correctness／promotion performanceを分離した実機qualification gate
187. `[Done]` 現在のM4でCPU／MLX／Metal／Core ML全12 capabilityの3-sample correctness再qualification
188. `[Done]` accelerator固有workload用bounded CPU referenceとCore ML FP32 digest正規化、実機非改善時CPU維持gate
189. `[Done]` 1024幅×16層代表encoderのM4実機qualification（CPU 2.052秒、Core ML end-to-end 302ms、ANE kernel 1.36ms、85.3%改善）とplacement適用／CPU fallback
190. `[Done]` persistent Core ML worker lifecycleとM4連続5 prediction実測（end-to-end最小0.84ms）、retryable scheduler fallback
191. `[Done]` CPU／GPU／ANE／Unified Memory／bandwidthの原子的resource ledger、runtime API契約、VLLM Metal配置のGPU resource accounting
192. `[Done]` numeric streaming runtimeのcancel・consume・shutdown実socket競合試験、connection即時回収、秘密情報を含まないorphan／lifecycle診断
193. `[Done]` fallback時のbackend間resource原子的引き継ぎと、profile-bound correctness／5%改善を要求するfail-closed contention evidence gate
194. `[Done]` CPU／GPU／ANE contention benchmark adapter、private strict profile、hardware-bound runtime起動時自動install
195. `[Done]` M4実機CPU／MLX GPU／Core ML ANE contention qualification（5 sample、digest全一致、3/3組が5%以上改善）
196. `[Done]` contention profileのprofile別private既定path、daemon startup fail-closed restore、bounded runtime diagnostics
197. `[Done]` contention profile runtime診断のstrict typed Swift SDK、旧client fallback、Mac app三言語表示、Swift 33 tests・sample build合格
198. `[Done]` contention profileのcurrent／last-known-good promotion、safe-point daemon reload／rollback、認証付き三言語管理API
199. `[Done]` contention reload／rollbackのtyped Swift SDK、strict evidence検証、Mac app三言語操作UI
200. `[Done]` contention全ペア合格時だけ原子的に一括予約するCPU／GPU／ANE bounded pipeline並列化
201. `[Done]` probe承認済みfallbackとcontention認定がある場合だけqueue先頭をidle backendへ移すbounded work stealing、容量不足時FIFO復元
202. `[Done]` memory pressure、thermal state、low-power modeに応じた新規admissionのconcurrency／batch／device割当の段階的縮退、safe-point回復
203. `[Done]` device assignment、queue wait、fallback、contention、thermal/power decisionの固定キー・上限付きruntime observabilityとstrict schema
204. `[Done]` scheduling observabilityのtyped Swift SDK、旧runtime unavailable fallback、Mac app三言語diagnostics
205. `[Done]` thermal／memory縮退を優先する自動／省電力／最高性能policy選択の認証付きruntime管理API、typed Swift SDK、Mac app三言語操作UI
206. `[Done]` scheduling preferenceのprivate・上限付きatomic永続化、daemon再起動時の復元と不正設定時のautomatic fallback
207. `[Done]` Vision/Audio encoderとembedding/classifierを用途別に識別し、probe合格済みCore ML capabilityへの完全一致を必須にするANE routing gate。bounded非同期ANE encoder → GPU LLM依存pipeline、stage間の出力順序保証、Unified Memory／device reservation解放、未実測のANE/GPU overlapを有効化しない逐次handoffを実装
208. `[Done]` Qwen3-VL vision ANE adapterのfail-closed静的admission。固定commit、Qwen3-VL architecture／processor、vision shape、全24 block・patch embedding・merger・3 deep-stack mergerのindexed tensor inventoryをartifact fingerprintへ結合。safetensors headerを実測し、配置済み2B 4-bitモデル内のvision tensor 315件はすべてBF16と確認。モデル全体のMLX affine INT4設定をvision weightのprecisionと誤認せず、Core ML artifactは未変換として維持
209. `[Done]` Qwen3-VL Core ML変換manifestとpromotion gate。source artifact fingerprint／固定revision／Core ML tree integrity／I/O shape／precision／converter versionを結合し、最大絶対誤差内のCPU／MLX reference一致、反復成功、ANE候補の非劣化を満たす場合だけ用途別DeviceCapabilityを発行。別sourceへのreplay、tree改変、数値不一致、性能劣化はfail-closed
210. `[Done]` Homebrew vLLM-Metal 0.29.0の実Qwen3-VL ABIに整合するANE → GPU LLM embedding bridge。`image_grid_thw`からspatial merge後token数を再計算し、main hidden statesと全deep-stack出力の件数・shape・hidden幅を検証してから、capability-bound非同期pipelineでGPU callbackへhandoff。不正出力はGPU実行前に拒否し全resourceを解放
211. `[Done]` Qwen3-VL vision safetensors全315件のheader・dtype・shapeをpayload非materializeで検証するCore ML変換計画。BF16→FP16容量、24 block、patch Conv3DのMLX `OTWCI`→Core ML `OICTW`転置、linear／norm／position embedding、merger／3 deep-stack mergerを分類し、固定grid profileとsource fingerprintへplan IDを結合
212. `[Done]` planに従うQwen3-VL vision BF16→FP16実変換worker。8 MiB上限のstreaming変換、patch Conv3DのMLX `OTWCI`→Core ML `OICTW`転置、tensorごとのSHA-256、source変更検知、出力容量上限、private directoryとfsync付きatomic publishを実装し、Homebrew vLLM-Metal環境で全tensor出力・数値転置・途中失敗cleanupをfixture検証
213. `[Done]` staged weight全件のsize／SHA-256／重複／余分なfile／plan replayをfail-closed検証し、固定gridごとのpixel input、main hidden states、全deep-stack output shapeと、patch embedding／position interpolation／RoPE／24層attention・MLP／merger semanticsをsource・plan・weight treeへ結合する再現可能なCore ML graph specification
214. `[Done]` 推論用Homebrew環境から分離したPython 3.12／Core ML Tools 8.1 converter環境と再利用可能なtoolchain probe。決定的FP16 MIL graphを`.mlpackage`へ生成し、Xcode 27 Core ML compiler 3600.25.1で`.mlmodelc`へcompile、arm64/macOS 27の`.cpuAndNeuralEngine`設定で実predictionが期待値と一致（0.750 ms）。証跡: [toolchain probe](evaluation/coreml-toolchain-probe-2026-09-19.json)。これはtoolchain認定でありQwen3-VL／ANE単独実行の認定ではない
215. `[Done]` Qwen3-VL-2B実weight 315件／813,914,112 bytesのBF16→FP16 stagingと、1×16×16固定gridのpatch projection＋bilinear position embedding partitionをCore ML Tools 8.1で`.mlpackage`／`.mlmodelc`化。全1,536 weight列を通る決定的sparse入力で262,144出力を照合し、最大scaled error 0.001484（固定上限0.002）、CPU+ANE設定の単発prediction 1.075 msで合格。証跡: [patch partition](evaluation/qwen3-vl-2b-patch-coreml-2026-09-19.json)。Core MLの実device割当やvision tower全体の認定には拡張しない
216. `[Done]` Qwen3-VL vision block 0の実norm2→1024→4096 MLP→GELU-tanh→4096→1024 projection→residual partitionをCore ML化。全1024入力featureを通るdense入力で262,144出力を照合し、最大scaled error 0.004894（固定上限0.01）、CPU+ANE設定の単発prediction 2.106 msで合格。証跡: [block 0 MLP](evaluation/qwen3-vl-2b-block0-mlp-coreml-2026-09-19.json)。MLP単体の構築可能性確認でありblock全体／promotion認定ではない
217. `[Done]` Qwen3-VL vision block 0の実norm1→QKV→固定2D RoPE→native scaled-dot-product attention→projection→residual partitionをCore ML化。16 heads×256 tokens×64 head dimensionの262,144出力をdense入力で照合し、最大scaled error 0.002930（固定上限0.02）、CPU+ANE設定の単発prediction 1.840 msで合格。証跡: [block 0 attention](evaluation/qwen3-vl-2b-block0-attention-coreml-2026-09-19.json)。attention単体の構築可能性確認でありblock全体／promotion認定ではない
218. `[Done]` 合格した固定RoPE attention residualとMLP residualを同一Core ML graphへ直列統合し、Qwen3-VL vision block 0を完成。dense入力の262,144出力で最大scaled error 0.01349（統合用固定上限0.03）、CPU+ANE設定の単発prediction 2.544 msで合格。証跡: [complete block 0](evaluation/qwen3-vl-2b-block0-complete-coreml-2026-09-19.json)。1 blockの認定であり24 block towerへは未拡張
219. `[Done]` block builderを任意の先頭連続層へ一般化し、block 0→1の2層を単一Core ML graphへ直列化。層数で緩和しない固定scaled error上限0.03に対して実測0.01936、CPU+ANE設定の単発prediction 3.984 msで合格。証跡: [two-block tower](evaluation/qwen3-vl-2b-tower-2blocks-coreml-2026-09-19.json)。2／24 blockの認定でありtower全体へは未拡張
220. `[Done]` 最初のdeep-stack採取点layer 5までblock 0〜5の6層を単一Core ML graphへ拡張。層数で緩和しない固定scaled error上限0.03に対して実測0.02554、CPU+ANE設定の単発prediction 14.480 msで合格。証跡: [six-block tower](evaluation/qwen3-vl-2b-tower-6blocks-coreml-2026-09-19.json)。6／24 blockのmain hidden出力のみの認定でありdeep-stack mergerは未接続
221. `[Done]` layer 5の中間hidden stateをdeep-stack merger 0へ接続し、6層main hidden `[256,1024]`とdeep-stack出力`[64,2048]`を同一Core ML graphで個別に照合。本流は最大scaled error 0.02554（固定上限0.03）・13.292 ms、deep-stackは0.01068（固定上限0.04）・11.843 msで合格。証跡: [six-block tower + deep-stack 0](evaluation/qwen3-vl-2b-tower-6blocks-deepstack0-coreml-2026-09-19.json)。CPU+ANE設定の個別prediction値であり、実device割当や同時二出力取得のend-to-end latencyは未認定
222. `[Done]` 同じ固定誤差gateを維持してlayer 11／deep-stack merger 1へ拡張。12層main hiddenは最大scaled error 0.02393（上限0.03）・24.960 ms、deep-stack 1は0.02759（上限0.04）・19.772 msで合格。証跡: [12-block tower + deep-stack 1](evaluation/qwen3-vl-2b-tower-12blocks-deepstack1-coreml-2026-09-19.json)
223. `[Done]` layer 17／deep-stack merger 2の18層一体graphはcompile・実行でき、deep-stack出力はscaled error 0.01563で合格した一方、main hiddenは0.03119で固定上限0.03を超えたため昇格を停止。診断証跡: [18-block diagnostic](evaluation/qwen3-vl-2b-tower-18blocks-deepstack2-diagnostic-2026-09-19.json)。上限を緩和せず、layer 11後のFP16 hiddenをdigest-bound入力にするlayer 12〜17分割graphを実装。本流は0.00830・15.257 ms、deep-stack 2は0.00879・11.419 msで合格し、超過原因を長い一体graphの累積誤差へ限定。証跡: [blocks 12–17 segment](evaluation/qwen3-vl-2b-tower-segment-12-17-deepstack2-coreml-2026-09-19.json)
224. `[Done]` 合格済みFP16 handoff方式でlayer 18〜23とfinal preshuffle-norm mergerを構築。24層main hiddenは最大scaled error 0.02051（上限0.03）・16.318 ms、final `[64,2048]`は0.01099（上限0.04）・12.194 msで合格。証跡: [blocks 18–23 + final merger](evaluation/qwen3-vl-2b-tower-segment-18-23-final-coreml-2026-09-19.json)。これにより24 block、3 deep-stack merger、final mergerの全計算区間が個別認定済み。ただしsegment入力はFP16 reference artifactであり実Core ML前段出力の連結認定ではない
225. `[Done]` 不足していたlayer 6〜11分割（本流scaled error 0.01172、deep-stack 1は0.02271）を追加し、4つの6層Core ML modelを同一Swift process内で`MLMultiArray`のまま直接handoff。1回のpipelineからdeep-stack 0／1／2とfinalを順序付きで取得し、observable出力のscaled errorは順に0.01068／0.02759／0.01563／0.01050（固定上限0.04）、4 prediction合計48.342 msで合格。未消費の最終tower hiddenは累積0.04736を診断値として保持し、出力契約の合格判定には使用しない。証跡: [segment pipeline](evaluation/qwen3-vl-2b-segment-pipeline-coreml-2026-09-19.json)
226. `[Done]` 4つのcompiled modelを一度だけloadするpersistent segment pipelineを5回反復し、各回のstage latency、observable出力SHA-256、誤差、process peak RSSをbounded収集。4出力は全反復でdigest完全一致、cold初回58.929 ms、warm 36.451〜37.914 ms、観測peak RSS最大220,151,808 bytesで合格。証跡: [persistent segment pipeline](evaluation/qwen3-vl-2b-segment-pipeline-persistent-2026-09-19.json)。5回のbounded qualificationであり長時間memory soakや実device割当の証明ではない
227. `[Done]` Core ML専用typed outputを既存Qwen3-VL embedding adapterへ追加し、64桁graph ID、grid、final、3 deep-stackのshape／件数／順序をGPU予約・実行前にfail-closed検証。検証済みbundleからHomebrew vLLM-Metal 0.29.0の実`Qwen3VLVisionEncodeResult`を構築するABI bridgeを実装し、実MLX FP16配列でfinal `[64,2048]`とdeep-stack 3本の保持を確認。証跡: [Homebrew encode result ABI](evaluation/qwen3-vl-homebrew-encode-result-abi-2026-09-19.json)
228. `[Done]` Core ML pipelineのfinal／deep-stack 3本を、0700 directory・0600 file・固定4 file・各262,144 bytes・SHA-256・graph IDに束縛したatomic FP16 transportとして実装。Homebrew vLLM-Metal 0.29.0 processで全payloadを検証し、NumPy read-only viewから実MLX FP16配列へmaterializeして`Qwen3VLVisionEncodeResult`へ接続。合計1,048,576 bytes、final／deep-stack 3本すべて`[64,2048]`を実環境で確認。証跡: [Core ML → Homebrew transport](evaluation/qwen3-vl-coreml-homebrew-transport-2026-09-20.json)
229. `[Done]` Homebrew環境の実Qwen3-VL processorで決定的224×224 RGB画像を`pixel_values [256,1536]`／`image_grid_thw [1,16,16]`へ変換し、FP16 patch projection＋position embeddingの実Core ML出力を4区間24 block towerへ同一processで直接handoff。finalとdeep-stack 3本はいずれも`[64,2048]`、非有限値0、tower 4区間50.364 msでexecution smoke合格。証跡: [real processor vision smoke](evaluation/qwen3-vl-real-processor-vision-coreml-smoke-2026-09-20.json)。実画像経路の実行可能性確認であり、同一入力のMLX数値referenceや言語出力品質は未認定
230. `[Done]` 同じprocessor pixel valuesをHomebrew MLX vision towerでも実行してCore MLと比較し、元のBF16経路ではpatch＋position直後scaled error 0.02734、deep-stack 0／1／2は0.09668／1.43054／1.06348、finalは1.60059となるためlanguage model接続を停止。診断証跡: [real-image MLX/Core ML comparison](evaluation/qwen3-vl-real-image-mlx-coreml-diagnostic-2026-09-20.json)。変換先と同じMLX-FP16基準ではpatchが0.0009766で上限0.002を合格し、patch実装とBF16→FP16 precision driftを分離。実画像由来入力でblock 0〜5を独立実行すると全単層がmain上限0.03内、先頭6層を明示FP32化するとmain 0.01080／deep-stack 0 0.00439で合格した。先頭1層だけのmixed precisionも同区間はmain 0.02490／deep-stack 0 0.02661で合格。固定上限を緩和せず、誤差源を単層構造ではなくsegment内演算precisionと境界transportへ限定。証跡: [precision isolation](evaluation/qwen3-vl-real-image-precision-isolation-2026-09-20.json)、[block/segment precision isolation](evaluation/qwen3-vl-real-image-block-precision-isolation-2026-09-20.json)
231. `[Done]` Core ML artifactへ入力／main出力precisionを結合し、pipelineが隣接segmentの型連鎖を実行前にfail-closed検証するprecision-aware transportを実装。最初のsegmentはFP16 patch入力→FP32 main、中央と最終segmentはFP32→FP32を同一process内の`MLMultiArray`で直接handoffし、deep-stack／finalだけFP16 ABIへ変換。同条件のMLX-FP32 transformer基準に対してdeep-stack 0／1／2は0.000969／0.001953／0.001953、finalは0.002441で全固定gateを合格し、2反復のobservable digestも一致。warm 4-segment合計171.535 ms、peak RSS 1,904,918,528 bytes。FP32入出力artifactの独立qualificationも実機合格。証跡: [FP32 segment transport](evaluation/qwen3-vl-real-image-fp32-segment-transport-2026-09-20.json)
232. `[Done]` 実装誤差とは分離した元MLX-BF16 sourceとのprecision driftはdeep-stack 0／1／2が0.05225／1.43250／1.07227、finalが1.29883。生成本文を保存せずbaseline／candidate hash、baseline replay、task正答を比較するHomebrew実言語モデルquality runnerを実装。自由記述16-token greedy smokeは日本語・简体中文が完全一致したが英語不一致のため厳格exact gateは2/3で停止。Core ML Tools 8.1のMILにはBF16 tensor typeがなくnative BF16候補は構築不能。そこで決定的な赤・緑・青224×224画像を実processorで各`[256,1536]`へ変換し、英語／日本語／简体中文のdominant-color taskをbaselineとcandidate双方に課すbounded semantic calibrationを実施。task正答9/9、baseline replay 9/9、生成完全一致8/9で合格。証跡: [BF16 language quality smoke](evaluation/qwen3-vl-coreml-bf16-quality-smoke-2026-09-20.json)、[semantic color calibration](evaluation/qwen3-vl-coreml-semantic-color-calibration-2026-09-20.json)。この合格は単純色識別taskに限定し一般的なVQA品質へ拡張しない
233. `[Done]` 一つのbounded process内で実画像processor→Core ML patch→FP32 main hiddenの4 segment／24層direct handoff→FP16 final／deep-stack→Homebrew MLX language modelを連続実行するend-to-end chat vision runnerを実装。private一時pixel／transportをfinallyで削除し、生成本文を保存せずhash・正答・stage latency・peak RSSだけを出力。赤224×224画像の英語／日本語／简体中文dominant-color smokeは3/3正答、BF16 baselineとの生成hashも3/3完全一致。processor 0.962 ms、Core ML 4 segment合計242.377 ms、candidate言語生成148.874〜158.525 ms、peak RSS 2,437,857,280 bytes、一時transport削除を確認。証跡: [end-to-end chat smoke](evaluation/qwen3-vl-coreml-end-to-end-chat-smoke-2026-09-20.json)
234. `[Done]` end-to-end runnerを最大8画像のbounded batchへ拡張し、patch＋4 segmentを一度だけloadするSwift process内で赤／緑／青を連続処理。requestごとの0700 transport directoryへ4出力を分離し、英語／日本語／简体中文はtask正答9/9、生成完全一致8/9（緑・简体中文もbaseline／candidateとも正答）。Core ML latencyはcold 241.498 ms、後続169.360／169.803 ms、peak RSSは2,437,152,768→2,442,772,480→2,444,443,648 bytesで後続増分5,619,712／1,671,168 bytes。正常終了cleanupとrequest 0後の故障注入による異常終了cleanupの双方で一時directory残存0を確認。証跡: [consecutive chat qualification](evaluation/qwen3-vl-coreml-consecutive-chat-2026-09-20.json)
235. `[Done]` end-to-end runnerをtask／label分離契約へ一般化し、色・形状・物体・OCRでtaskに存在しないlabelを実行前拒否。再現可能な224×224 fixture generatorで黒円・家・`APPLE`を生成し、同一processの3 request、英語／日本語／简体中文を実機評価。baseline／Core ML candidateともtask正答9/9、生成hash完全一致9/9、後続Core ML latency 172.440／174.509 ms、後続peak RSS増分4,145,152／3,588,096 bytes、出力分離・cleanupも合格。証跡: [general vision-language calibration](evaluation/qwen3-vl-coreml-general-vision-calibration-2026-09-20.json)。固定合成画像と1×16×16 gridの結果でありopen-domain VQA品質へは拡張しない
236. `[Done]` artifact由来の4 segment precision profileを実行reportへ結合し、色・形状・物体・OCR×3言語の18ケースを固定gateとして精度配置を縮小。各6層segmentの先頭1層だけFP32とする4/24層FP32候補は正答18/18・完全一致17/18、全24層FP16候補は正答18/18・完全一致18/18で合格。最小の全FP16候補はwarm vision平均37.757 msで全FP32参考値171.535 msから77.99%短縮し、6 request後のpeak RSSは2,450,718,720 bytes。証跡: [minimal precision calibration](evaluation/qwen3-vl-coreml-minimal-precision-calibration-2026-09-20.json)。既知のBF16数値driftがあるため、この選択は固定6画像・3言語taskにのみ昇格し、一般runtimeへは未昇格
237. `[Done]` 公開`MLComputePlan.deviceUsage`を再帰的に集計するfail-closed inspectorと、pixel file read／`MLMultiArray` copy／patch prediction／4 segment prediction／transport copy／atomic writeの個別計測を実装。同一入力・各5反復で、全FP32はCPU preferred 749演算・warm中央値170.099 ms・peak RSS 1,921,220,608 bytes、4層FP32はCPU 128／ANE 624演算・56.575 ms・496,074,752 bytes、全FP16はANE preferred 746演算・36.219 ms・235,307,008 bytes。全FP16は全FP32比でwarm latency 78.71%、peak RSS 87.75%削減。segment間は同一`MLMultiArray`直接handoffで明示copy 0、外部transportはcopy 0.135 ms＋atomic write 2.393 ms。証跡: [device/copy/performance qualification](evaluation/qwen3-vl-coreml-device-copy-performance-2026-09-20.json)。device値はcompiled operationのpreferred割当であり、非公開915演算やruntime hardware counterの実行先までは断定しない
238. `[Done]` task限定で合格した全FP16 profileを、1 workerあたり10 prediction後にclean exitする反復worker方式で30分qualification。1800.919秒、526 batch／5,260 requestで失敗0、4 observable出力のdigest不一致0、直近256 sampleのmedian latency 421.464 ms、worker peak RSSは初期235,110,400／最終251,068,416／最大252,788,736 bytes（初期比+15,958,016 bytes）、parent peak RSS 35,454,976 bytes、shutdown failure 0で合格。証跡: [30-minute FP16 soak](evaluation/qwen3-vl-coreml-fp16-30min-soak-2026-09-20.json)。batchごとにworkerを再生成する試験であり、単一persistent worker内の長時間memory安定性には拡張しない。一般runtime昇格はopen-domain品質gateまたはBF16相当precisionの実装まで保留
239. `[Done]` patch＋4 segmentを一度だけloadするQwen3-VL persistent Swift/Core ML workerを実装。private空directoryへの固定4出力、最大64件のbounded admission、直列prediction、応答timeout時のworker停止、crash検出後の次request前自動再起動、明示shutdown protocolを追加。実機3連続predictionは52.304／37.517／37.221 ms、RSS 213,843,968→214,384,640 bytes、digest全一致、予期しないrestart 0、graceful shutdown後process停止を確認。request間kill注入後はrestart 1回で同一digestへ回復し再shutdownも成功。証跡: [persistent worker lifecycle](evaluation/qwen3-vl-coreml-persistent-worker-lifecycle-2026-09-20.json)
240. `[Done]` persistent vision workerを5モデルload後に同一processのまま30分連続実行。1800.820秒／1,793 requestで失敗0、timeout 0、4出力digest不一致0、restart 0、直近256 sampleのmedian latency 62.071 ms。worker peak RSSは初期／最終／最大すべて216,449,024 bytesで成長0、parent peak RSS 34,062,336 bytes、graceful shutdownと一時transport cleanupも合格。証跡: [persistent 30-minute soak](evaluation/qwen3-vl-coreml-persistent-30min-soak-2026-09-20.json)
241. `[Done]` persistent workerのstage順digestを実payloadと再照合し、graph ID／固定grid／shape／dtype／bytesへ束縛した0600 manifestを排他的作成・fsyncするadapterを実装。順序不一致を実機でfail-closed検出してdeep-stack 0／1／2→final順へ修正後、Homebrew MLXでfinal／deep-stack 3本をFP16 `[64,2048]`としてmaterializeし、実`Qwen3VLVisionEncodeResult`を構築。probe済みcapability registry、ANE 216,449,024 bytes予約、MLX GPU 1,000,000,000 bytes予約を通る非同期scheduler routeでCore ML→MLX handoffを完走し、全resource使用量0へ復帰、restart 0、正常shutdownを確認。証跡: [persistent scheduler bridge](evaluation/qwen3-vl-persistent-scheduler-bridge-2026-09-20.json)
242. `[Done]` persistent scheduler routeの実vision resultをHomebrew Qwen3-VL language generationへ接続。MLXをexecutor threadで実行するとnative runtime終了時にSIGSEGVとなる実機障害を検出し、ANE／GPUの予約と依存順序を維持したcaller-thread実行経路を追加した。赤／緑／青、円、家、OCR `APPLE`の6 requestを順序どおり処理し、英語／日本語／简体中文18/18正答、従来MLX vision baselineとの生成hash 18/18完全一致、restart 0、全resource解放、正常shutdown、一時領域削除を確認。queue saturationはfail-fast、encoder失敗時はGPU callback未実行となる回帰テストも追加。証跡: [persistent scheduled chat quality](evaluation/qwen3-vl-persistent-scheduled-chat-quality-2026-09-20.json)
243. `[Done]` server統合の安全境界として、非stream chatにもrequest ID、有限deadline、socketのnon-consuming切断signalを持つ`InferenceRequestContext`を追加し、対応engineへkernel contextと共に伝播。persistent ANE→GPU caller-thread schedulerはANE前、ANE完了後／GPU前、GPU完了後のsafe pointでcancel／timeoutを検査し、GPU未実行または結果非公開で全予約を解放する。HTTP timeoutは408、client切断は応答を書き戻さずmetadata-only 499として記録する。daemon停止時はmanaged engineを冪等に一度だけcloseするmodel lifecycleも接続し、実HTTP timeout後のrequest slot回収を含む回帰テストを追加
244. `[Done]` native modelを専用processのmain threadで生成・実行・closeするbounded inference transport。最大64 pending、4 MiB request／16 MiB responseのJSON-lines IPC、親側fail-fast backpressure、子側bounded queue、active cancel／deadline safe point、固定error code、worker exit時pending解放、shutdown→terminate→kill回収を実装。fake native delegateの同一process順序、child main-thread ownership、queue saturation、active cancel後の回復、実`/v1/chat/completions`経路、main-thread cleanupを確認
245. `[Next]` inline PNG／JPEG、private request workspace、persistent Core ML encoder、scheduler予約内MLX生成、OpenAI互換responseを扱うQwen3-VL managed delegateを上記専用process factoryへ接続し、Homebrew MLX実機の複数request順序、backpressure、cancel／timeout、client切断cleanupをqualificationする。一般runtime昇格はopen-domain品質gateまたはBF16相当precisionの実装まで保留

この順序により、まず推論runtimeの実model安定性を確立し、その境界を壊さずにoptimizerを
別processとして追加する。構造pruningはquantization、calibration、評価gateの後に着手する。
