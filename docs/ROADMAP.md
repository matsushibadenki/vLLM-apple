# vLLM-Apple Runtime Roadmap

最終更新：2026-08-22

本ロードマップは、[Design-Specifications.md](Design-Specifications.md)を実装可能な単位へ分解し、現在のコードベースに対する進捗を示す。

## ステータス

- `[Done]` implemented in the current codebase
- `[Next]` high-priority unfinished work
- `[Later]` planned, but not the closest next step

`[Done]` は設計済みではなく、現在のコードベースに実装と検証が存在する項目だけに付与する。

## 現在地

Phase 1のcontrol plane、メモリ安全性基盤、Macアプリ向けSwift SDK foundation、
3言語対応の最小macOS SwiftUI chat sampleまで実装済み。

現在の最優先目標は、実modelを使ったend-to-end経路と長時間安定性を完成させ、
Model Optimization Compilerへ実測durationとstructured failure modelを追加することである。

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

## Phase 1 — Foundation and Mac Integration

### Hardware and memory

- `[Done]` Apple Silicon、architecture、OS、CPU数の検出
- `[Done]` Unified Memory総量の検出
- `[Done]` `vm_stat`を利用した現在のavailable memory検出
- `[Done]` memory pressureの基礎判定
- `[Done]` 検出APIが利用できない場合の保守的fallback
- `[Next]` GPU core数とSoC名の検出精度向上
- `[Next]` macOS memory pressure notificationの継続監視
- `[Later]` thermal stateとpower modeの検出

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
- `[Next]` MLA、state-space、hybrid model固有のstate memory計算
- `[Next]` 未cache Hugging Face model metadataの取得
- `[Next]` model load前後でのcontext再評価
- `[Later]` workload履歴とthermal状態を用いた動的context調整

### Runtime profile

- `[Done]` immutable runtime profile model
- `[Done]` versioned profile schema
- `[Done]` private permissionによるprofile保存
- `[Done]` `fsync`とatomic replaceによる破損防止
- `[Next]` profile load、validation、migration
- `[Next]` hardware/model別profile cache
- `[Later]` benchmark結果を含むprofile versioning

### Basic scheduler

- `[Done]` operator単位のbackend選択interface
- `[Done]` CPU、MLX GPU、Metalの基礎routing
- `[Done]` transient memoryのhard admission limit
- `[Done]` thread-safe reservationとrelease
- `[Done]` request priority model
- `[Next]` queueingとpriority arbitration
- `[Next]` cancellation時のreservation自動解放
- `[Next]` backend failure時のMetal → MLX → CPU fallback
- `[Later]` profiler実測値によるbackend選択

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
- `[Next]` 実modelを用いたvLLM-Metal互換性検証
- `[Next]` graceful request drainを伴うshutdown
- `[Done]` Unix Domain Socket HTTP transport
- `[Done]` UDS pathのowner/type検証と0600 permission
- `[Done]` constant-time Bearer session token認証
- `[Done]` atomic 0600 session token file
- `[Done]` 256 eventのbounded runtime event ring
- `[Done]` 最大8 clientのbounded event subscription
- `[Done]` 遅延subscriberへの`stream.gap`通知
- `[Done]` `Last-Event-ID`対応SSE runtime event stream
- `[Later]` remote modeのTLSとauthentication

### CLI

- `[Done]` `vllm-apple hardware`
- `[Done]` `vllm-apple context`
- `[Done]` `vllm-apple profile`
- `[Done]` `vllm-apple serve`
- `[Done]` `vllm-apple serve <model>` command path
- `[Done]` managed backend port、startup timeout、max model context options
- `[Done]` `doctor` command
- `[Done]` UDS、session token、session token file options
- `[Next]` automatic model inspectionとrecommended configuration表示
- `[Next]` structured startup progress
- `[Later]` daemon install、start、stop、status command

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
- `[Next]` Xcode app targetへのdaemon bundle組み込みsample
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
- `[Done]` Python 41 test passing
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
- `[Next]` 実modelで30分以上のlong-running memory stability test
- `[Done]` daemon SIGKILL後のstale UDS replacementとtoken/auth recovery test
- `[Done]` daemon relaunch後のSIGTERM cleanupと残留process確認
- `[Later]` real-model correctness regression suite

## Phase 1 Completion Criteria

Phase 1を完了とする条件：

- `[Next]` `vllm-apple serve <model>` だけで実際の対応modelを起動できることを検証する
- `[Next]` OpenAI互換のnon-streaming/streaming chatが実modelで成功する
- `[Next]` Swift sample appからdaemon起動、model load、streaming chat、shutdownが成功する
- `[Next]` modelに応じた安全なcontextが自動設定される
- `[Next]` memory pressure時に新規workloadを抑制し、daemonが異常終了しない
- `[Next]` backend errorが構造化され、Swift側で復旧可能性を判定できる
- `[Next]` Python、Swift、end-to-end testが継続的に成功する

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
- `[Next]` hardware profiler実測値によるestimated duration
- `[Done]` original model pathへのwriteを拒否するpath safety policy
- `[Done]` optimizer state/event schemaとbounded progress event
- `[Next]` structured optimizer error/recoverability taxonomy
- `[Done]` plan、manifest、path traversal、disk/memory境界値test
- `[Done]` `vllm-apple-optimize plan` CLI entry point

### O1 — Representation optimization

- `[Later]` backend adapter interfaceとcapability detection
- `[Later]` FP16/BF16 → INT8/INT4 quantization candidate generation
- `[Later]` MLX、将来のGGUF等へのversioned exporter adapter
- `[Later]` KV cache precision、context、batch configuration search
- `[Later]` stage checkpoint、cancel、resumeと一時fileのatomic promotion
- `[Later]` output hash、size、peak RSS、latencyをartifact manifestへ記録

### O2 — Calibration and evaluation

- `[Later]` local-only calibration runnerとPIIを外部送信しないprivacy boundary
- `[Later]` activation全量を保持しないonline statistics / disk streaming capture
- `[Later]` layer、head、neuron importance report
- `[Later]` 英語、日本語、简体中文のcalibration/evaluation manifest
- `[Later]` code、math、long-context等を用途ごとに選択するevaluation suite
- `[Later]` perplexity、task score、latency、RSS、artifact sizeのbaseline比較
- `[Later]` 未評価能力とquality regressionを明示するrelease gate

### O3 — Weight and structural optimization

- `[Later]` outlier-aware quantization、weight clustering、low-rank approximation
- `[Later]` structured / unstructured pruning experiment adapter
- `[Later]` attention head、MLP、layer functional similarity analysis
- `[Later]` layer bypass、head merge、layer merge candidate generation
- `[Later]` quality budget超過時のcandidate自動reject
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

- `[Later]` CPU GEMM/GEMV micro benchmark
- `[Later]` GPU GEMM/GEMV micro benchmark
- `[Later]` Unified Memory bandwidth測定
- `[Later]` Metal launch latency測定
- `[Later]` Attention throughput測定
- `[Later]` quantized matmul benchmark
- `[Later]` model、shape、batch、context別kernel profile
- `[Later]` automatic batch sizing
- `[Later]` adaptive KV allocation
- `[Later]` continuous memory pressure monitoring
- `[Later]` thermal-aware scheduling foundation

## Phase 3 — Kernel Optimization

- `[Later]` native Metal Paged Attention
- `[Later]` MLA kernel
- `[Later]` quantized GEMV/GEMM
- `[Later]` RMSNorm、RoPE、activation fusion
- `[Later]` MoE routingとExpert GEMM
- `[Later]` graph fusion pass
- `[Later]` kernel autotuningとcompiled kernel cache
- `[Later]` Metal failure時のMLX/CPU correctness fallback suite

## Phase 4 — Vision

- `[Later]` image input frontend
- `[Later]` image preprocessing pipeline
- `[Later]` vision encoder cache
- `[Later]` multimodal batching
- `[Later]` Resize → Normalize → Patchify → Projection fusion
- `[Later]` image latency、images/sec、memory/image benchmark

## Phase 5 — Audio

- `[Later]` audio ring buffer
- `[Later]` resamplerとfeature encoder
- `[Later]` streaming audio state
- `[Later]` REALTIME priorityとdeadline scheduler
- `[Later]` ASR integration
- `[Later]` audio encoder cache
- `[Later]` speech-to-speech foundation
- `[Later]` dropout、latency、real-time factor benchmark

## Phase 6 — Video

- `[Later]` hardware video decoder integration
- `[Later]` GPU-accessible bufferへのcopy削減path
- `[Later]` frame scheduler
- `[Later]` temporal sampler
- `[Later]` frame、patch、embedding、scene cache
- `[Later]` video VLM integration
- `[Later]` streaming video input
- `[Later]` frames/sec、seconds-of-video/sec、memory/minute benchmark

## Phase 7 — Generative Media

- `[Later]` image generation workload
- `[Later]` audio and music generation workload
- `[Later]` video generation workload
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

## Cross-Cutting Work

### Reliability

- `[Done]` context計算のunderflow防止
- `[Done]` schedulerのhard memory ceiling
- `[Done]` profile書き込みのatomicity
- `[Done]` bounded HTTP concurrency
- `[Done]` backend processのreadiness、exit、timeout監視
- `[Done]` bounded backend log buffer
- `[Done]` bounded event historyとsubscriber ceiling
- `[Next]` structured error taxonomyとrecoverability
- `[Next]` daemon crash diagnostics
- `[Later]` fault injection suite

### Security

- `[Done]` localhost-only default
- `[Done]` request size limit
- `[Done]` profile fileのprivate permission
- `[Done]` UDS permissionとsession authentication
- `[Next]` model file hash validation
- `[Later]` remote TLS、API key、client identity

### Observability

- `[Done]` runtime state、memory、scheduler snapshot
- `[Done]` backend failureのruntime state反映
- `[Done]` reconnect可能なruntime state/failure SSE event
- `[Next]` request IDとstructured logging
- `[Next]` tokens/sec、TTFT、TPOT
- `[Next]` Unified MemoryとKV usage
- `[Later]` GPU/CPU utilization、bandwidth、thermal、power
- `[Later]` Vision、Audio、Video固有metrics

### Packaging and release

- `[Done]` Python package metadataとCLI entry points
- `[Done]` Swift Package metadata
- `[Next]` supported Python、vLLM、vLLM-Metal version matrix
- `[Next]` reproducible development environment
- `[Next]` CI for Python and Swift
- `[Later]` signed daemon artifact
- `[Later]` notarized Mac app integration package

## Recommended Immediate Sequence

直近の実装順序：

1. `[Done]` vLLM-Metal process adapterとversion compatibility check
2. `[Done]` OpenAI chat proxyおよびstreaming
3. `[Done]` standard Transformer model metadata inspectionとautomatic context設定
4. `[Done]` UDS、session authentication、bounded event stream
5. `[Done]` Swift UDS transport、ManagedRuntimeのcrash recoveryとlog capture
6. `[Done]` 最小SwiftUI Mac chat sample
7. `[Done]` concurrent load、bounded soak runner、daemon crash/relaunch test
8. `[Next]` 実modelのend-to-endと30分以上のmemory stability test
9. `[Done]` OptimizationPlan / ArtifactManifest schemaとsafe dry-run planner
10. `[Next]` optimizer duration profilerとstructured error taxonomy
11. `[Later]` representation optimization workerとMac companion app

この順序により、まず推論runtimeの実model安定性を確立し、その境界を壊さずにoptimizerを
別processとして追加する。構造pruningはquantization、calibration、評価gateの後に着手する。
