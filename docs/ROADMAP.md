# vLLM-Apple Runtime Roadmap

最終更新：2026-08-21

本ロードマップは、[Design-Specifications.md](Design-Specifications.md)を実装可能な単位へ分解し、現在のコードベースに対する進捗を示す。

## ステータス

- `[Done]` implemented in the current codebase
- `[Next]` high-priority unfinished work
- `[Later]` planned, but not the closest next step

`[Done]` は設計済みではなく、現在のコードベースに実装と検証が存在する項目だけに付与する。

## 現在地

Phase 1のcontrol plane、メモリ安全性基盤、Macアプリ向けSwift SDK foundationまで実装済み。

現在の最優先目標は、vLLM-Metalを実推論backendとして接続し、Macアプリから安全にmodelを起動・streaming利用できる最小end-to-end経路を完成させることである。

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
- `[Next]` Hugging Face / MLX model metadataからKV bytes/tokenを自動算出
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
- `[Done]` inference未接続時の構造化503 response
- `[Next]` vLLM-Metal inference backend接続
- `[Next]` chat completionのtoken streaming
- `[Next]` graceful request drainを伴うshutdown
- `[Next]` Unix Domain Socket transport
- `[Next]` session tokenとsocket permission検証
- `[Later]` remote modeのTLSとauthentication

### CLI

- `[Done]` `vllm-apple hardware`
- `[Done]` `vllm-apple context`
- `[Done]` `vllm-apple profile`
- `[Done]` `vllm-apple serve`
- `[Next]` `vllm-apple serve <model>`
- `[Next]` automatic model inspectionとrecommended configuration表示
- `[Next]` structured startup progress
- `[Next]` `doctor` command
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
- `[Next]` daemon stdout/stderrのbounded log capture
- `[Next]` readiness timeout、crash recovery、restart policyの統合test
- `[Next]` app bundle / Application Support resource resolver
- `[Next]` Unix Domain Socket対応transport
- `[Next]` runtime progress event stream
- `[Next]` 英語、日本語、簡体字中国語のlocalization catalog
- `[Next]` 最小SwiftUI Mac chat sample
- `[Later]` Objective-C adapter
- `[Later]` notarizationとApp Sandbox統合sample

### Schema and testing

- `[Done]` runtime snapshot JSON Schema v1
- `[Done]` runtime event JSON Schema v1
- `[Done]` context境界値test
- `[Done]` scheduler memory limit test
- `[Done]` atomic profile persistence test
- `[Done]` localhost API integration test
- `[Done]` Swift model decoding test
- `[Done]` Python 10 test passing
- `[Done]` Swift 2 test passing
- `[Next]` JSON Schemaによる実response validation
- `[Next]` concurrent request load test
- `[Next]` long-running memory stability test
- `[Next]` daemon crash/restart integration test
- `[Later]` real-model correctness regression suite

## Phase 1 Completion Criteria

Phase 1を完了とする条件：

- `[Next]` `vllm-apple serve <model>` だけで対応modelを起動できる
- `[Next]` OpenAI互換のnon-streaming/streaming chatが実modelで成功する
- `[Next]` Swift sample appからdaemon起動、model load、streaming chat、shutdownが成功する
- `[Next]` modelに応じた安全なcontextが自動設定される
- `[Next]` memory pressure時に新規workloadを抑制し、daemonが異常終了しない
- `[Next]` backend errorが構造化され、Swift側で復旧可能性を判定できる
- `[Next]` Python、Swift、end-to-end testが継続的に成功する

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
- `[Next]` structured error taxonomyとrecoverability
- `[Next]` daemon crash diagnostics
- `[Later]` fault injection suite

### Security

- `[Done]` localhost-only default
- `[Done]` request size limit
- `[Done]` profile fileのprivate permission
- `[Next]` UDS permissionとsession authentication
- `[Next]` model file hash validation
- `[Later]` remote TLS、API key、client identity

### Observability

- `[Done]` runtime state、memory、scheduler snapshot
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

1. `[Next]` vLLM-Metal process adapterとversion compatibility check
2. `[Next]` model metadata inspectionとautomatic context設定
3. `[Next]` OpenAI chat proxyおよびstreaming
4. `[Next]` UDS、session authentication、bounded event stream
5. `[Next]` Swift ManagedRuntimeのcrash recoveryとlog capture
6. `[Next]` 最小SwiftUI Mac chat sample
7. `[Next]` concurrent load、memory pressure、long-running stability test

この順序により、実modelを使ったend-to-end経路を早期に完成させ、その後にMacアプリ配布品質と長時間安定性を高める。
