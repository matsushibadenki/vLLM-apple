# vLLM-Apple Runtime Roadmap

最終更新：2026-09-04

本ロードマップは、[Design-Specifications.md](Design-Specifications.md)を実装可能な単位へ分解し、現在のコードベースに対する進捗を示す。

## ステータス

- `[Done]` implemented in the current codebase
- `[Next]` high-priority unfinished work
- `[Later]` planned, but not the closest next step

`[Done]` は設計済みではなく、現在のコードベースに実装と検証が存在する項目だけに付与する。

## 現在地

Phase 1のcontrol plane、メモリ安全性基盤、AppleExecutionPlanner、StateMemorySpec、
prefill/decode別profile、Swift SDK、3言語macOS sample、Gemma実modelの30分安定性まで実装済み。

2026-08-30のstatus監査で、後続実装と実機記録が存在した古い`[Next]` 10件を`[Done]`へ更新した。
現在のactionableな`[Next]`表記は6件、重複参照をまとめた実作業は2件である。最優先は大容量Apple Siliconでの
Qwen3.8-Flash-Next text-only qualificationと、専用runnerでのvLLM 0.28.x昇格試験である。
設計判断は
[Architecture-Decision-Apple-Execution.md](Architecture-Decision-Apple-Execution.md)に固定する。

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
- `[Done]` MLA latent KV、state-space recurrent/conv state、hybrid layer構成固有のstate memory計算
- `[Done]` 未cache Hugging Face modelのweightを取得しないbounded metadata fetchとrevision binding
- `[Done]` model load前後でのcontext再評価
- `[Later]` workload履歴とthermal状態を用いた動的context調整

### Runtime profile

- `[Done]` immutable runtime profile model
- `[Done]` versioned profile schema
- `[Done]` private permissionによるprofile保存
- `[Done]` `fsync`とatomic replaceによる破損防止
- `[Done]` profile load、validation、migration
- `[Done]` hardware/model別profile cache
- `[Later]` benchmark結果を含むprofile versioning

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
- `[Later]` profiler実測値によるbackend選択

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
- `[Done]` 実modelを用いたvLLM-Metal互換性検証
- `[Done]` graceful request drainを伴うshutdown
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
- `[Done]` automatic model inspectionとrecommended configuration表示
- `[Done]` structured startup progress
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

- `[Done]` versioned AppleExecutionPlan schema、decision reason、deterministic dry-run
- `[Done]` StateMemorySpec（weights、KV、recurrent、prefix、window、expert、scratch）
- `[Done]` AppleChipProfileとbackend capability contract foundation
- `[Done]` 実hardware/backend capability detectionとAppleChipProfile永続化
- `[Done]` prefill/decode別bounded profile schemaと集計器（TTFT、TPOT、throughput、peak memory）
- `[Done]` 実backend streamのtoken timing、usage、backend RSS instrumentation
- `[Done]` bounded SSE phase-profile CLIとusage欠落時のfail-closed検証
- `[Done]` plannerと既存context、scheduler、elastic memory policyのatomic safe-point接続
- `[Done]` active/pending execution plan observabilityとphase別batch admission gate
- `[Later]` CPU GEMM/GEMV micro benchmark
- `[Later]` GPU GEMM/GEMV micro benchmark
- `[Later]` Unified Memory bandwidth測定
- `[Later]` Metal launch latency測定
- `[Later]` Attention throughput測定
- `[Later]` quantized matmul benchmark
- `[Later]` model、shape、batch、context別kernel profile
- `[Later]` automatic batch sizing
- `[Later]` adaptive state allocationとage/pressure別precision
- `[Later]` continuous memory pressure monitoring
- `[Later]` thermal-aware scheduling foundation
- `[Later]` BackendEngine交換契約（vLLM-Metal、Native MLX、Native Metal、Core ML draft、CPU）
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
- `[Next]` 大容量Apple SiliconでQwen text-only実model qualification
- `[Done]` kernel capability、self-test結果、quarantine理由のversioned registry
- `[Later]` vLLM-Metal Paged Attention capability/benchmark統合
- `[Later]` native Metal Paged Attention拡張は計測済みの不足が残る場合のみ
- `[Later]` MLA kernel
- `[Later]` MLX Q8/Q4 baselineと互換性gate
- `[Later]` fused dequantize + GEMV/GEMM + activation
- `[Later]` RMSNorm、RoPE、activation fusion
- `[Later]` MoE routingとExpert GEMM
- `[Later]` graph fusion pass
- `[Later]` kernel autotuningとcompiled kernel cache
- `[Later]` Metal failure時のMLX/CPU correctness fallback suite
- `[Later]` Metal toolchain/OS更新後のcompile smokeと性能退行検出
- `[Later]` multi-model Metal command submissionのserialization/concurrency stress test

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

- `[Done]` 3候補のbounded初期profile catalogと共通load前artifact/Unified Memory admission
- `[Later]` hardware video decoder integration
- `[Later]` GPU-accessible bufferへのcopy削減path
- `[Later]` frame scheduler
- `[Later]` temporal sampler
- `[Later]` frame、patch、embedding、scene cache
- `[Later]` video VLM integration
- `[Later]` streaming video input
- `[Later]` frames/sec、seconds-of-video/sec、memory/minute benchmark
- `[Later]` M4/32GB向け動画生成qualification profile（最初は低解像度、短尺、batch 1、bounded frames/steps）
- `[Later]` Wan 2.2 TI2V-5Bを優先候補とするT2V/I2V、high-compression VAE、量子化、逐次module residency検証
- `[Later]` HunyuanVideo 1.5 8.3Bを候補とする480p、step-distilled、SSTA、model offload検証
- `[Later]` Wan 2.2 A14B量子化版をstretch候補とするT2V/I2V別artifact、dual-expert residency、CPU/SSD offload検証
- `[Done]` video diffusion pipelineのDiT/expert、text encoder、3D VAE別artifact admissionとconservative resident-memory hard ceiling
- `[Done]` privacy-preserving動画生成qualification report schemaとdeterministic evaluator（first-output/wall latency、peak RSS、memory pressure、thermal state、frames/sec、output metadata、plan fingerprint）
- `[Done]` backend-neutralなbounded telemetry event contractとconstant-memory sample collector
- `[Done]` shellを介さないbounded JSONL subprocess telemetry adapterとtimeout時process-group停止
- `[Done]` workspace-bound one-shot worker request、prompt digest binding、0600 atomic保存、consume後unlink
- `[Done]` Diffusers sourceのbounded AST scanによる6候補pipeline class readiness gate（backend import/model/Metal allocationなし）
- `[Later]` MLX、Diffusers、ComfyUI固有workerからqualification sampleを取得するadapter
- `[Later]` 最小profile合格後だけ解像度、frame数、steps、連続生成を一軸ずつ増やす段階的memory-stability gate
- `[Later]` model license、量子化方式、変換元digest、workflow provenanceを記録し、CIではweightと生成動画を保存・uploadしないprivacy gate

## Phase 7 — Generative Media

- `[Done]` 3候補のbounded初期profile catalogと共通load前artifact/Unified Memory admission
- `[Later]` image generation workload
- `[Later]` M4/32GB向け画像生成qualification profile（最初は512×512、batch 1、単一画像、bounded steps）
- `[Later]` FLUX.2 [klein] 9B Baseを優先候補とする量子化、text encoder分離、VAE tiling、逐次module residency検証
- `[Next]` 配置済みZ-Image-Turbo-MLX-4bitを優先候補とするMLX backend readiness、512×512・9 steps実機qualification
- `[Later]` Qwen-Image-2512を候補とするMPS/MLXまたは対応backendの量子化、offload、peak Unified Memory検証
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
- `[Later]` 512×512合格後だけ768/1024と連続生成へ進む段階的memory-stability gate
- `[Later]` model license、gated artifact、quantization provenanceを記録し、CIではweightと生成画像を保存・uploadしないprivacy gate
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
- `[Done]` structured runtime error taxonomyとrecoverability
- `[Done]` raw detailを公開しないfailure fingerprintとSwift typed decode
- `[Done]` private/atomic daemon crash diagnosticsとbounded log digest
- `[Done]` optional native telemetry登録をcontrol readinessから隔離
- `[Done]` Swift ManagedRuntime restart testを決定的なUDS fault-injection fixtureへ分離
- `[Done]` host memory pressureから隔離した決定的なmonitor transition回帰fixture
- `[Later]` fault injection suite

### Security

- `[Done]` localhost-only default
- `[Done]` request size limit
- `[Done]` profile fileのprivate permission
- `[Done]` UDS permissionとsession authentication
- `[Done]` model tree全regular fileのstreaming SHA-256 manifest生成・起動前検証（symlink/special file拒否、変更検出、bounded走査）
- `[Done]` detached CMS trusted manifest署名、trusted CA chain、signer SHA-256 identityのmodel load前検証
- `[Later]` remote TLS、API key、client identity

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
108. `[Later]` M4/32GB動画生成qualification（Wan 2.2 TI2V-5B、HunyuanVideo 1.5 8.3B、量子化Wan 2.2 A14B）
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
120. `[Next]` Z-Image-Turbo-MLX-4bitの対応MLX workerとM4/32GB最小profile qualification
121. `[Later]` Qwen-Image-2512-4bitのMFLUX workerとoffload前提memory-stability qualification
122. `[Done]` MFLUX Z-Image/Qwen Image backend classとartifact形式を分離したloadなしreadiness gate
123. `[Next]` Z-Image artifactをMFLUX互換形式へ揃え、readinessを合格させる配置手順の確定
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
148. `[Next]` pressure recovery gate有効下での768×768・4回再qualificationとall-normal baseline取得

この順序により、まず推論runtimeの実model安定性を確立し、その境界を壊さずにoptimizerを
別processとして追加する。構造pruningはquantization、calibration、評価gateの後に着手する。
