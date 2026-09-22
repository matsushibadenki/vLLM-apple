# vLLM-Apple Runtime Roadmap

最終更新：2026-09-22

本ロードマップは、[Design-Specifications.md](Design-Specifications.md)を実装可能な単位へ分解し、現在のコードベースに対する進捗を示す。

## ステータス

- `[Done]` implemented in the current codebase
- `[Next]` high-priority unfinished work
- `[Later]` planned, but not the closest next step
- `[Pending]` on hold until required hardware, model artifact, runtime, upstream ABI, or signing credentials become available; not counted as locally actionable work

`[Done]` は設計済みではなく、現在のコードベースに実装と検証が存在する項目だけに付与する。

## 現在地

Phase 1のcontrol plane、メモリ安全性基盤、AppleExecutionPlanner、StateMemorySpec、
prefill/decode別profile、Swift SDK、3言語macOS sample、Gemma実modelの30分安定性まで実装済み。

2026-09-22のstatus監査で、contention profile管理、optimizer companion app、生成系backend、数値変換、
Qwen3-VL Core ML／Homebrew MLX統合までの後続実装と実機証跡を反映した。現行Macで安全に完了できる
既に`[Done]`となった実装項目は、大容量file bookmarkを含めて回帰試験済みである。現行Macで進められる実装・試験は`[Next]`、
追加のhardware、未配置model、専用runner、Developer ID／notary資格情報などが必要な実機qualificationは
`[Pending]`として保留する。保留項目は前提が揃うまでローカル完了数に含めない。HomebrewのvLLM-Metal 0.29.xは固定revisionの認定stackとは別candidate
として扱い、一般runtime昇格にはopen-domain品質またはBF16相当precisionの追加証拠を要求する。
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
- `[Done]` Qwen3-VL画像経路の30分安定性。全FP16 Core ML profileを1 worker 10 predictionの反復方式で5,260 request、persistent worker方式で1,793 request実行し、失敗・digest不一致・予期しないrestartはいずれも0。証跡: [反復worker 30分](evaluation/qwen3-vl-coreml-fp16-30min-soak-2026-09-20.json)、[persistent worker 30分](evaluation/qwen3-vl-coreml-persistent-30min-soak-2026-09-20.json)
- `[Done]` Homebrew vLLM-Metal同梱MLX Audio 0.5.4でKokoro-82M-6bitの実speech生成を確認。固定revision、Apache-2.0、weight SHA-256、24 kHz mono PCM S16LE、独立2 sample、異なるdigest、memory pressure normal、thermal nominal、peak RSS最大788,725,760 bytes、prompt/output非保存、private cleanupに合格。証跡: [Kokoro speech qualification](evaluation/kokoro-82m-6bit-mlx-audio-2sample-2026-09-22.json)
- `[Done]` MLX Audio 0.5.4でMiniMax Music3 affine 4-bitの実music生成を確認。固定revision、Community License、2 shard SHA-256、load前memory／pressure／thermal admission、44.1 kHz stereo PCM S16LE、5秒上限の独立2 sample、異なるdigest、memory pressure normal、thermal nominal、peak RSS最大9,789,440,000 bytes、prompt/output非保存、private cleanupに合格。証跡: [MiniMax Music3 qualification](evaluation/minimax-music3-4bit-mlx-audio-2sample-2026-09-22.json)
- `[Done]` Gemma 3 4B IT 4bitの固定revision取得・14ファイル照合、Homebrew 0.29.0でMLX-VLMロード確認。実画像要求はmultimodal encoder adapter未準備でEngineCore停止を再現し、不合格として記録: [attempt](evaluation/gemma-3-4b-it-vision-attempt-2026-09-19.json)。chat templateを備えるだけではbackend画像対応を保証しない。
- `[Done]` 配布版のQwen3-VL adapter登録とforward_readyを先に確認し、Qwen3-VL-2B-Instruct-4bit（revision `9c4f5209e57b31f4b9dfba735de3fb983739c9cc`、16ファイル照合済み）で画像smokeを実測。Homebrew 0.29.0、context 2048、memory fraction 0.25で3言語×赤青の6ケースすべて合格、正常shutdown。証跡: [vision smoke](evaluation/qwen3-vl-2b-vision-smoke-2026-09-19.json)。長時間認定・一般的画像理解能力の認定は含まない。
- `[Done]` `vision-smoke` CLI: 32x32赤・青PNGをOpenAI image_urlに埋め込み、3言語6ケースを画像付きSSEで検証。同一質問の画像差分を用い、画像・生成本文は非保存。PNG送信・画素・判定の回帰テストを追加。長時間vision認定とは別扱い。
- `[Done]` text-only qualification runnerによるvision誤認定を起動前に拒否（image-input probeが未実装のため）。
- `[Done]` ローカルGemma 3 4B PTの実weight headerでvision/projector tensor 439件を確認。Homebrew 0.29.0でMLX-VLMロード成功。`skip_vision=true`だけではweight欠落と判断できない。画像付きchatはtokenizerのchat template未定義によりHTTP 400となり、画像品質は未測定。
- `[Done]` 合格したQwen3-VL画像入力を30分qualificationへ接続。反復worker 5,260 requestとpersistent worker 1,793 requestで安定性を確認。固定task限定でありopen-domain VQA認定には拡張しない
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
- `[Done]` prefill、decode、sampling、Vision／Audio encoder、embedding、classifier、draft、verify別のoperator dispatch。共通`WorkloadPhase`／backend descriptorへphase vocabularyを追加し、bounded dependency graphの逆向きphase edgeをbackend実行前にfail-closed拒否する
- `[Done]` Qwen3-VL固定224×224 profileのVision encoderをpersistent Core ML/ANEへrouteし、probe-bound resource予約内でMLX GPU LLMへ順序付きhandoff。task限定実機qualificationとfallback／cleanupを完了
- `[Done]` 実audio encoderのCore ML/ANE-eligible routing。WhisperKitの固定revision Whisper tiny AudioEncoder（MIT、16,788,176 bytes、artifact SHA-256固定）を`.cpuAndNeuralEngine`で実行し、固定shape `[1,80,1,3000]→[1,384,1,1500]`、3/3有限値、入力別digest、6.49–14.22 ms、peak RSS 206,536,704 bytes、pressure normal、thermal nominalをApple M4で確認。音声／embedding本文は非保存。証跡: [Whisper tiny Core ML encoder](evaluation/whisper-tiny-coreml-audio-encoder-m4-2026-09-22.json)
- `[Done]` 汎用embeddingの実artifact Core ML routing。Apple公式MobileCLIP S0の固定revision（Apple-ASCL、image/text合計107,801,759 bytes、artifact SHA-256固定）を`.cpuAndNeuralEngine`で実行し、画像`256×256×3`とtext token `[1,77]`から各3入力で相異なる有限512次元embeddingを確認。image 1.23–1.83 ms、text 0.91–1.37 ms、peak RSS 317,358,080 bytes、pressure normal、thermal nominal、一時compiled artifact cleanupまでApple M4で合格。画像／token／embedding本文は非保存。証跡: [MobileCLIP S0 Core ML embedding](evaluation/mobileclip-s0-coreml-embedding-m4-2026-09-22.json)
- `[Done]` classifierの実artifact Core ML routing。Apple公式FastViT-T8 ImageNet-1K classifierの固定revision（Apple-ASCL、8,126,615 bytes、artifact SHA-256固定）を`.cpuAndNeuralEngine`で実行し、3 synthetic imageで1000要素の有限確率分布・確率和・入力別digest・top labelを検証。重複class nameによりdictionaryは999 unique keyであることを明示検証し、1.94–2.25 ms、peak RSS 217,677,824 bytes、pressure normal、thermal nominal、一時compiled artifact cleanupまでApple M4で合格。画像／確率本文は非保存。証跡: [FastViT-T8 Core ML classifier](evaluation/fastvit-t8-coreml-classifier-m4-2026-09-22.json)
- `[Pending]` 音声対応LLMとmodel固有projection artifact配置後、Whisper encoder出力をGPU LLMへ渡すmultimodal projection／quality qualification
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
- `[Done]` KV／recurrent／prefix／attention-window／MoE expert／workspaceを含む統合elastic memory再配分。同じbounded record、age／pressure policy、promotion済みprecision制約、pinned保護、backend transactionで一つのatomic planとしてreprecision／evictする
- `[Done]` prefill/decode別のCPU・Metal・Unified Memory bandwidth profile。hardware fingerprintとCPU prefill／decode・Metal実測reportの完全SHA-256を結合し、prefillは同shape GEMM、decodeは同shape GEMVを比較する。M4実測ではMetal speedup 35.27×／8.23×、shared bandwidth 37.79 GB/s、attention median 1,291,416 nsで両phaseをNative Metal候補とした。証跡: [M4 phase resource profile](evaluation/phase-resource-profile-m4-2026-09-21.json)
- `[Done]` MoE `(layer, expert)` working-set LRUとbackend residency adapter。entry／byte二重上限、active lease pin、決定論的LRU、load／release ownership、capacity不足拒否、pressure resizeのpending適用、hit／miss／eviction／rejection telemetryを実装し、Metal／MLX resource handleをopaqueに保持できる
- `[Done]` layer double-buffered prefetchと不足時のon-demand fallback。最大4,096 layerの順序・重複を検査し、専用1-thread／2-slotで次layer loadと現在layer consumeをoverlapする。prefetch例外は当該layerだけ同期loadへ一度fallbackし、成功・consume例外・cancel時にcurrent／orphan resourceを確実にreleaseする
- `[Done]` page-aligned fast-load optimizer artifactとkernel compatibility index。4–64 KiB page alignment、最大65,536 tensor／32 GiB、private atomic publish、source/output再hash、no-follow、重複・overlap拒否を実装。backend/version/environment/operator/format/layout/alignment/最大tensor bytesが一意一致するkernelだけを選択する

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
- `[Done]` Objective-C adapter。専用`VLLMAppleKitObjC` dynamic library product、`NSObject` health/chat result、completion-block型HTTP clientを追加し、model 256文字・prompt 32Ki文字・temperature 0〜2・max tokens 1〜1,048,576を同期検証する。Swift errorはstable `message_key`付き`NSError`へ変換し、Swift 6 concurrencyと全SDK testで実ビルド済み
- `[Done]` notarizationとApp Sandbox統合sample。bundled-daemon版とsandbox client版を別targetにし、sandbox版はoutbound network、user-selected read-only、app-scoped bookmarkだけを付与。Developer ID署名、hardened runtime、notary submit/wait、staple、ZIP／checksum／notary result／release manifest生成、OIDC provenance、exact-tag draft release gateをmanual workflowへ接続。実資格情報による初回artifact生成はPackagingの独立`[Pending]` gateとして維持
- `[Done]` App Sandbox client integration target。通常のbundled-daemon targetと分離した`VLLMAppleChatSandbox`を追加し、App Sandbox、outbound network client、user-selected read-only file、app-scoped bookmarkだけを付与。compile-time gateでdaemon resolve／起動を除外し、独立したloopback daemonへ接続する構成をApple M4上のXcode Debug buildで検証

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
- `[Done]` real-model correctness regression suite。既存のbounded generation evidenceを入力に、英語・日本語・简体中文を必須化し、同一candidate model hashの2〜16回反復、sample集合／dataset／prompt設定一致、baseline token agreement／expectation regression、run間output fingerprint再現性、worst-case latency／RSS regressionをfail-closed判定する。本文を再保存しないSchema v1／deterministic report IDと`vllm-apple-optimize correctness-regression` CLIを実装

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
- `[Done]` llama.cpp `convert_hf_to_gguf.py`用versioned exporter adapter。明示したowner-only converter pathと、呼び出し側が検証済みとしてexact allowlistした`bNNNN` buildだけをfail-closedで許可し、Safetensors FP16／BF16／FP32からGGUF F16／BF16／Q8_0への固定argument invocation、source snapshot SHA-256、保守的byte budget、isolated worker／checkpoint／atomic promotion接続、Schema v1を実装。別quantizerを要するQ4系は未検証のままexecutableとして公開しない
- `[Done]` KV cache precision、context、batch configuration search。FP32／FP16／BF16／INT8、context 128〜16M、batch 1〜256、KV block 8〜128、最大512候補を対象に、実state size以上のpeak memory、3 sample以上のdecode latency、最大絶対誤差／RMSE／cosine／output digestを必須化。capacity／latency／balanced objective別にquality・memory合格候補だけを決定選択する
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
- `[Done]` activation全量を保持しないonline statistics / disk streaming capture。Welford mergeでcount／mean／population variance／min／max／abs max／zero数だけをconstant memory集計し、1 update最大1,048,576値・総count上限・finite検査を行う。disk streamはfingerprintとaggregateだけをowner-only JSONLへ追記し、1 GiB hard ceiling、no-follow、0600、fsyncを適用する
- `[Done]` layer、head、neuron importance report。最大65,536 componentの集約activation絶対値とoutput sensitivityをsample数でweighted mergeし、積による決定論的score、全component内normalized score、bounded上位結果、SHA-256 report IDを生成する。raw activation／prompt／出力は保持しない
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

- `[Done]` outlier-aware quantization、weight clustering、low-rank approximation。threshold超過indexだけを保持するsorted sparse residual付きbounded affine量子化、最大65,536値／256 centroidの決定論的1D k-means、最大65,536要素／rank 64のpower-iteration＋deflation参照近似を実装し、MSE／最大誤差／Frobenius誤差を報告する
- `[Done]` structured / unstructured pruning experiment adapter。最大65,536 finite値を対象に、絶対値の小さいweightをstable順で除くunstructured maskと、row／column L2 normの小さいgroupを除くstructured maskを生成し、最低1要素／groupを維持してsquared errorを報告する
- `[Done]` attention head、MLP、layer functional similarity analysis。boundedな同長出力を一時比較し、cosine similarity、MSE、最大絶対誤差、sample数と内容digestに結合したcomparison IDだけを返す。非finite／shape不一致を拒否しraw出力をreportへ保持しない
- `[Done]` layer bypass、head merge、layer merge candidate generation。importance ceiling以下のlayerだけをbypass候補、同一layer内head／隣接layerのcosine similarity floor以上だけをmerge候補とし、最大4096候補、score・kind・IDによる決定順序、evidence-bound SHA-256 candidate IDを実装する。候補生成はartifactを変更しない
- `[Done]` quality budget超過時のcandidate自動reject。dataset fingerprintとslice集合の完全一致を前提にperplexity相対劣化をslice別budgetと比較し、`quality_approved=false`のcandidateをrankなし・`quality_gate_failed`で自動除外。合格候補0件ではartifactを選択しない
- `[Done]` optional LoRA/SFT repair adapterとrepair前後の再評価。method別にsample／epoch／learning-rate／LoRA rank／seedをhard boundしたrequest、source／output digestとmethodを結合したartifact契約、同一dataset・slice・token数のbefore／after perplexity gateを実装し、改善しないrepairを既定でrejectする

### O4 — Mac companion app

- `[Done]` `VLLMAppleOptimizer` Mac app target。macOS 13以降の独立Swift Packageとしてbuild／test可能にし、optimizer CLIとはversioned JSON planだけで接続する
- `[Done]` model、用途、品質／速度／memory優先度の設定UI。model/output directory、balanced／memory／speed／quality、license、memory／disk／duration hard limitを入力できる
- `[Done]` disk/memory見積もりと明示的な実行confirmation。dry-run planの候補別output／disk／peak memory／duration／blocking reasonを表示し、budget内かつadapter executableな候補だけを選択可能にする。変換は破壊的操作の明示確認後に限り`--execute`で開始する
- `[Done]` progress、pause、resume、cancel、failure recovery UI。owner-only bounded JSONL event transportによるprepare／convert／resume／validate／promoteのstage別progress、SIGUSR1／SIGUSR2による隔離converter process groupのcooperative pause／continue、signal-safe cancel、private checkpointからの明示resume、bounded error表示を実装
- `[Done]` original / optimized responseとbenchmark比較。同一のowner-owned bounded JSONLを使うperplexity slice比較に加え、最大32 sample・16K prompt・32 new tokenの決定的generationを両artifactへ逐次実行し、sample別token一致率／期待値score、合否、総elapsed、peak RSSを比較する。prompt／response本文はreportへ保存せずtoken-level比較に限定する
- `[Done]` artifact、provenance、license、未評価能力report。artifact ID、size、peak RSS、SHA-256、license、transform、tool version、perplexityのdomain／language slice、generationのsample／context上限、両quality gate合否と未評価能力を統合表示する。raw prompt／responseやweightは保持しない
- `[Done]` 英語、日本語、简体中文localization
- `[Done]` 大容量file access sample。model、output、perplexity、generationの選択を最大1 MiBのsecurity-scoped bookmarkとして保存し、再起動時のUIなし復元、stale bookmark更新、破損時のfail-closed消去、処理中だけのscope保持を実装。5 GiB疎model fileを読み込まずmetadata参照できることをSwift testで固定
- `[Done]` App Sandbox GUI sample targetと三言語UI build。sandbox targetはmodel directoryへの恒久権限やprocess executionを持たず、外部daemon接続とuser-selected read-only bookmarkに限定する
- `[Done]` Optimizer sandbox client向けversioned loopback transport基盤。HTTP loopback以外、userinfo／query／fragmentを拒否し、ephemeral session、cache／cookie／credential無効化、単一connection、1 MiB request／response上限、schema version／request UUID再照合を実装。transportはprocess探索・起動・signal操作を一切持たない
- `[Done]` 独立daemonのoptimizer plan API。loopback serverでだけ明示enableし、model／output allowlist rootを別々に最大16件、current-user所有のreal directoryへ固定。片側だけの設定、remote公開、symlink、allowlist外pathをload前拒否し、1 MiB versioned envelopeから実model metadataの副作用なしdry-run planを返す
- `[Done]` `VLLMAppleOptimizerSandbox` App Sandbox client target。外部optimizerを直接起動する現行targetとXcode targetを分離し、loopback endpoint、model/output picker、objective／budget／license、dry-run候補表示を英語・日本語・简体中文で実装。App Sandbox、network client、user-selected read-only、app-scoped bookmarkだけを付与し、process API非混入とApple M4 Debug buildを検証
- `[Pending]` 実Developer ID／notary資格情報による署名artifact生成。workflow、hardened runtime、submit／wait、staple、release manifestは実装済みであり、秘密情報をrepositoryへ保存せず初回artifactを検証する

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
- `[Done]` GPU GEMM/GEMV micro benchmark。Apple M4 native Metal、warm-up後7 sampleでFP32 64×64×64 GEMMはmedian 199,458 ns／1.314G work-items/s、256×256 GEMVは224,500 ns／291.9M work-items/s。全sampleを同一process／pipelineで実行しoutput digestを固定
- `[Done]` Unified Memory bandwidth測定。`storageModeShared`の32 MiB buffer間copy（read+write 64 MiB traffic）をApple M4で7 sample実測し、median 1,775,875 ns／37.79 GB/sを記録
- `[Done]` Metal launch latency測定。compile／pipeline生成を除外した同一queueのempty kernel command encode→commit→completionを7 sample実測し、end-to-end median 178,208 nsを記録
- `[Done]` Attention throughput測定。Apple M4 native Metalで1 query×128 KV token×64 head dimensionのscaled dot-product attentionを7 sample実測し、median 1,291,416 ns／811.96M算術work-items/sを記録
- `[Done]` quantized matmul benchmark。Apple M4 native MetalでINT8 64×64×64、INT32 accumulationを7 sample実測し、median 221,416 ns／1.184G work-items/sを記録。共通証跡: [M4 native hardware microbenchmarks](evaluation/native-hardware-microbenchmarks-m4-2026-09-21.json)
- `[Done]` model、shape、batch、context別kernel profile。model metadataからGQA heads、head dimension、KV dtype、context tier、block数・working setを固定するversioned Paged Attention profile、profile-bound shape benchmark、batchを含むdevice benchmark identity、private/atomic strict loaderとCLIを実装済み
- `[Done]` automatic phase batch sizing。memory pressure、thermal、power mode、CPU backend制約からprefillを1/2/4、decodeを1へ決定し、plan identityとdecision reasonへ固定してscheduler admissionで超過を拒否する
- `[Done]` adaptive state allocationの決定論的policy。KV／recurrent／prefix／attention-window／expert stateをbounded recordで表し、state age順、normal／warning／critical pressure、pinned状態、backendで明示promotion済みprecisionだけからretain／reprecision／evict planと解放byte数を生成する。未知pressureはnormalへfail-soft、未昇格precisionは選択しない
- `[Done]` backend-owned stateのatomic reprecision／rollback adapterをscheduler safe pointへ接続。backend snapshot→bounded plan→transaction begin／commitを同一safe pointで行い、commit例外時はrollbackする。active reservation中はpressureをpending化し、最後のreservation完了時にsemantic cache resize、adaptive state transaction、execution planを順に適用。単独adaptive構成、event、snapshot metricsも追加
- `[Done]` 実bufferを所有するKV／recurrent state backend adapter。FP32からFP16／symmetric INT8への変換を実データで行い、maximum absolute error、RMSE、cosine similarity、memory削減率を全て満たしたprecisionだけをpromotion対象にする。変換／evictionはstale planとbyte見積りを再検証してatomic commitし、rollback時は元bufferを維持する
- `[Done]` continuous memory pressure monitoring。macOS libdispatch memory-pressure sourceでnormal/warning/critical変化をevent-driven取得し、同一状態をcoalesce。daemon起動をブロックしない隔離threadからadmission、scheduler、safe-point elastic cacheとruntime eventへ反映し、source不可時は`vm_stat`系telemetryへfail-softする
- `[Done]` thermal/power状態をversioned plan identityとdecision reasonへ固定し、prefill batchを保守的にclampするscheduling foundation
- `[Done]` 15秒bounded thermal/power monitor、同一状態coalesce、current hardware snapshotとruntime change eventへの反映
- `[Done]` Swift SDKのtyped operating-state event decodeと未知のcurrent値に対するfail-soft fallback
- `[Done]` versioned BackendEngine交換契約。vLLM-Metal、Native MLX、Native Metal、Core ML draft、CPUを共通enumで扱い、version、architecture、precision、phase、operator、isolation、ready/lifecycle、request deadline/cancel safe point、retryable fallback attemptをfail-closed registryへ統合
- `[Done]` production backend processを`BackendEngineRegistry`へ直接登録するcomposition root。chat payloadをversioned requestへ保持し、factoryのatomic startup／失敗時reverse rollback、context-aware実行、retryable busy／execution failure、結果schema、model一覧、diagnostics、逆順shutdownを共通adapterへ実装。`BackendRegistryInferenceEngine`でRuntimeService ABIへ戻し、Qwen3-VL専用main-thread subprocess runnerを実際にregistry経由へ切替済み
- `[Done]` CPU／Core ML draft + GPU verifyのcorrectness-neutral実行基盤。CPU／ANE draftとvLLM-Metal／MLX／Native Metal verifyの型付き組合せ、3〜1024 sample・output一致・5%以上高速化を必須にするprofile gate、最大64 draft token／65,536 output token／4,096 round、Unified Memory／CPU／ANE／GPU resource予約、deadline／cancel safe point、GPU検証済みtokenだけのpublish、mismatch時のverifier token correction、全失敗時reservation解放を実装。production `BackendEngineRegistry`へ`DRAFT`／`VERIFY` phaseとして接続
- `[Done]` 合格したspeculative profileだけをowner-only／bounded／atomic JSONへ保存し、model hash・precision・backend・sample数・latency・output一致・再計算profile IDをstrict load時に再検証する永続化契約。不合格profileはfile生成前に拒否する
- `[Done]` 実model候補の負例qualification。固定revision Gemma 3 1B 4-bit draftとGemma 3 4B 4-bit verifierをMLX-LM 0.26.2で3言語各17 token実行し、token列3/3一致、pressure normal、thermal nominalを確認したが、native MLXはdraft／verifyともGPUで現在のCPU/Core ML draft契約外、かつmedian 469.78 ms→620.63 ms（32.1%低速化）のため保存・標準route注入を拒否。証跡: [Gemma 3 MLX speculative rejected](evaluation/gemma3-1b-4b-mlx-speculative-rejected-2026-09-22.json)
- `[Done]` CPU draft／GPU verifierを別MLX stream・別KV cacheで実行し、整数token IDだけを境界で渡すgreedy実model経路。Gemma 3 1B CPU＋4B GPUの3言語・各16 tokenで3/3出力一致、pressure normal、thermal nominal、最大MLX peak 3.13 GiB、process peak RSS 4,746,428,416 bytesを確認。各sampleのdraft受理7〜9 token、verifier補正7〜8回。ただしmedian baseline 493.43 msに対しspeculative 5,720.00 ms（約11.59倍）で5%改善gateは不合格。合格profile保存・標準route注入は行わない。証跡: [heterogeneous speculative RSS qualification](evaluation/gemma3-1b-cpu-4b-gpu-speculative-rss-2026-09-22.json)。旧[初回証跡](evaluation/gemma3-1b-cpu-4b-gpu-speculative-rejected-2026-09-22.json)も保持する
- `[Next]` 5%以上改善するmodel固有CPU/Core ML draft＋GPU verifierを3 sample以上でqualificationし、合格profileだけを標準decode routeへ注入する。Gemma 3 1B CPU＋4B GPUはcorrectnessに合格したが大幅に低速であり認定しない。より小さいdraft、十分大きいGPU verifier、またはCore ML/ANE draftの別構成が必要
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
- `[Done]` Mac個体別runtime autotuner（batch、tile、KV block、prefill chunk、kernel）。hardware／phase profileへ結合した最大256候補、batch 1〜256、tile 1〜4096、KV block 8/16/32/64/128、prefill chunk 16〜65,536のstrict構成を実装。同一output digest、3 sample以上、peak memory ceiling内の候補だけをprefill／decode weighted medianで順位付けし、同点はmemoryと構成順で決定する
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
- `[Pending]` 大容量Apple Siliconでtext-only smoke、TTFT、TPOT、RSS、品質gate
- `[Done]` worker compositionへ実MLX resident backendを注入するproduction entrypoint。`numeric-runtime-worker`がverified stage reader、mode-aware admission、numeric artifact reader、`Qwen4MLXNumericResidentBackend`、private UDS、session credential、command serviceを一体構築する。Apple M4／MLX 0.27.1でsocket経由artifact load、NVFP4 streaming、F16／BF16／F32実配列値、artifact consume、unload後reservation 0、正常shutdownを実機確認。証跡: [MLX resident production entrypoint](evaluation/qwen4-mlx-resident-production-entrypoint-m4-2026-09-22.json)
- `[Done]` production MLX adapter correctness後のNative Metal比較。実MLX resident backendはF16／BF16／F32値とlifecycleに合格。Native Metal NF4 fused GEMV／GEMM／attentionも参照一致したが、Apple M4で認定two/three-pass route比約0.85／0.89／0.20倍だったためproduction resident routeへは昇格せず、MLXを維持する決定を証跡化

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
- `[Done]` 33-frame root＋49-frame 4-sampleのchained promotion contractで65-frameへ一軸拡張し、2/2件が640×384×65で成功。全件memory pressure normal／thermal fair、issues 0、最大peak RSS 11,844,460,868 bytes、median wall 1,532,029 ms、2 digest distinct、prompt/output非保存、private cleanupを確認。証跡: [65-frame 2-sample qualification](evaluation/wan-mlx-gen-640x384-65frames-2sample-2026-09-21.json)
- `[Done]` 65-frame 4-sample stability gate。合格した65-frame 2-sample reportと初期33-frame 4-sample rootをload前に再検証し、Apple M4/32 GiB、MLX-Gen 0.33.1、mixed Q8/BF16、640×384×65、20 stepsで4/4成功。issues 0、全件memory pressure normal／thermal fair、最大peak RSS 11,844,460,868 bytes、RSS range 0、median wall 1,333,142 ms、4 digest distinct、prompt/output非保存、private cleanupを確認。証跡: [65-frame 4-sample stability](evaluation/wan-mlx-gen-640x384-65frames-4sample-stability-2026-09-21.json)
- `[Done]` HunyuanVideo 1.5 8.3BのDiffusers worker adapter。T2V／I2Vを`HunyuanVideo15Pipeline`／`HunyuanVideo15ImageToVideoPipeline`へmode別に固定し、load-free source/artifact readiness、private image digest binding、MPS model offload、VAE tiling、bounded telemetry、private MP4 digest/deleteを共通qualification CLIへ接続
- `[Pending]` HunyuanVideo 1.5 8.3B実artifact配置後の480p、step-distilled、SSTA、model offload実機qualification
- `[Done]` Wan 2.2 A14B量子化版のstretch worker adapter。T2V／I2Vをmode別Wan pipelineへ固定し、4/8-bit artifact readiness、ABI v2 private image binding、isolated MPS worker、共通qualification CLIへ接続
- `[Pending]` Wan 2.2 A14B量子化実artifact配置後のT2V/I2V別artifact、dual-expert residency、CPU/SSD offload実機検証
- `[Done]` video diffusion pipelineのDiT/expert、text encoder、3D VAE別artifact admissionとconservative resident-memory hard ceiling
- `[Done]` privacy-preserving動画生成qualification report schemaとdeterministic evaluator（first-output/wall latency、peak RSS、memory pressure、thermal state、frames/sec、output metadata、plan fingerprint）
- `[Done]` backend-neutralなbounded telemetry event contractとconstant-memory sample collector
- `[Done]` shellを介さないbounded JSONL subprocess telemetry adapterとtimeout時process-group停止
- `[Done]` workspace-bound one-shot worker request、prompt digest binding、0600 atomic保存、consume後unlink
- `[Done]` Diffusers sourceのbounded AST scanによる6候補pipeline class readiness gate（backend import/model/Metal allocationなし）
- `[Done]` MLX-Gen／MFLUX／Diffusers／ComfyUI固有workerから共通qualification sampleを取得するversioned adapter。Python backendは固定`python -m`、ComfyUIは明示的な外部telemetry workerだけを許可し、exact backend version allowlist、owner所有・非writable executable、workspace内real request、shellなしargv、JSONL v1 contract、24時間以内timeoutへfail-closed接続する。MLX-Gen／Diffusers／MFLUXは既存実workerへ接続済み、ComfyUIは同contract準拠workerを差し替え可能
- `[Done]` 最小profile合格後だけ解像度、frame数、steps、連続生成を一軸ずつ増やす段階的memory-stability gate。Wanで33-frame 2/4-sample合格後にだけ49-frameを許可し、同形状2→4 sampleはparent promotion chainも再検証する
- `[Done]` 2段目以降のvideo frame promotion contract。直前の4-sample安定profileと初期4-sample rootをcandidate/shape/all-normal条件で再検証し、両plan digestをchain identityへ結合する。これにより49-frame安定化後の65-frame候補を初期証跡なしでは起動できない
- `[Done]` model license、量子化方式、変換元digest、workflow provenanceを記録し、weightと生成動画を保存・uploadしないprivacy gate。Wan実機reportでartifact root digest、backend/version、mixed Q8/BF16、licenseを固定し、prompt/MP4非保存とprivate cleanupを確認

## Phase 7 — Generative Media

- `[Done]` 3候補のbounded初期profile catalogと共通load前artifact/Unified Memory admission
- `[Done]` image generation workload。local-only Diffusers MPS worker、bounded telemetry、private output digest/delete、artifact/provenance gateをQwen-Image-2.1 INT8の実生成で確認
- `[Done]` M4/32GB向け画像生成qualification profile。512×512、batch 1、20 steps、独立2 sampleをApple M4/32 GiBで実測合格
- `[Done]` FLUX.2 [klein] 9B Base 4-bitのMLX-Gen経路。artifact／license／backend readiness、逐次module residency、512×512・20-step 2-sample、768×768・2/4-sampleまで実機認定し、1024はhard-ceiling停止から安定上限を768へ固定。MLX-GenがVAE tilingを公開しないため、未実装optionへfallbackせずautomatic VAE decodeを認定profileに固定
- `[Done]` MLX-Gen互換Z-Image Turbo 4-bitを優先候補とするbackend readiness、512×512・9 steps実機qualification
- `[Next]` Qwen-Image-2512 4-bit MFLUXの現行Mac向けcomponent streaming/offload。28層text encoderの3言語実prompt、英語実embedding/maskのprivate別process handoffと量子化transformer固定層＋RoPE＋60 block、合成promptでの32px・実ノイズ2 step denoising→VAEまで`[Done]`。次は実prompt＋2 stepの一体実証、実用画像寸法・必要step数・peak Unified Memory・画像品質を段階的に検証する。配置artifactは25,907,123,451 bytes／21 files。従来の一括512×512・20 steps・2 sampleはresident見積り27,917,287,424 bytesでload前停止したため、同じ一括経路を再試行しない。証跡: [readiness/admission](evaluation/qwen-image-2512-mflux-readiness-admission-2026-09-22.json)、[32px two-step](evaluation/qwen-image-2512-mflux-streamed-2step-32px-synthetic-2026-09-22.json)
- `[Pending]` 上記component streamingで現行Macのload前admissionを通過できない場合の、大容量Apple SiliconによるQwen-Image-2512画像生成・memory-stability qualification
- `[Done]` Qwen-Image-2512のtext encoderを層単位で読むためのload-free staging inventory。7 safetensors shardの367 tensorとindexを照合し、28層は各466,115,840 bytes、共通weightは1,090,002,176 bytes、共通＋1層のpayload下限は1,556,118,016 bytesと算定。artifact全体のmetadataは4-bitだがtext encoderのtensorはBF16×338／F32×29で量子化weightは0件。deployable 21 files／25,907,123,451 bytesだけのdigestへ結合し、`.git/lfs`複製を除外する。これはweight loadなしの構造証跡であり、materialization overhead・実RSS・生成品質の合格ではない。証跡: [dtypeを含むtext encoder staging inventory](evaluation/qwen-image-2512-mflux-text-encoder-staging-dtype-2026-09-22.json)。旧[deployable inventory](evaluation/qwen-image-2512-mflux-text-encoder-staging-deployable-2026-09-22.json)と[Git LFSを含む過大計上report](evaluation/qwen-image-2512-mflux-text-encoder-staging-2026-09-22.json)は訂正履歴として保持
- `[Done]` MFLUX Qwen text encoderの選択的BF16/F32 tensor readerと第0層だけの実機forward smoke。safetensors headerのoffset／shape／dtype／byte上限を検証し、指定tensorの範囲だけをpreadしてMLXへ復元する。M4/32 GiBで第0層の13 tensorを読み、norm／q projectionはMFLUX標準mx.loadと完全一致、synthetic `[1,4,3584]`出力は有限値。peak MLX 504,735,132 bytes、process peak RSS 736,804,864 bytes、pressure normal／thermal nominal。これは単層実行証跡であり28層全体や画像生成の合格ではない。証跡: [Qwen-Image-2512 selective layer 0](evaluation/qwen-image-2512-mflux-layer0-selective-load-2026-09-22.json)
- `[Done]` Qwen-Image-2512 BF16 text encoderの28層逐次materializationとsynthetic full-tower smoke。共通weightを先にloadし、各層のweightをexact-offsetで1層ずつ読んでforward後にmaterialize／解放する。M4/32 GiBで独立process 2回とも28/28層完走、`[1,40]→[1,6,3584]`有限出力のdigest一致、各層pressure normal／thermal nominal、peak MLX 1,576,732,112 bytes、peak process RSS 1,916,436,480／2,035,269,632 bytes。text encoderのsynthetic smokeに限定し、prompt品質・transformer/VAE・画像生成は未認定。証跡: [initial full tower](evaluation/qwen-image-2512-mflux-28layer-streaming-smoke-2026-09-22.json)、[independent repeat](evaluation/qwen-image-2512-mflux-28layer-streaming-repeat-2026-09-22.json)
- `[Done]` Qwen-Image-2512の配置済みMFLUX tokenizer templateで英語・日本語・简体中文の固定実文章をtokenizeし、逐次28層encoderへ入力。M4/32 GiBで3/3有限embedding、shapeは`[1,13,3584]`／`[1,19,3584]`／`[1,12,3584]`、maskは各長さと一致、digestは3件相違、計84層実行、pressure全件normal、thermal fair、peak MLX 1,583,913,835 bytes、process peak RSS 1,321,123,840 bytes。prompt・embedding本文は非保存。これは実文章のencoder実行可能性であり、意味品質・private handoff・画像生成は未認定。証跡: [real-prompt streaming smoke](evaluation/qwen-image-2512-mflux-real-prompt-streaming-smoke-2026-09-22.json)
- `[Done]` Qwen-Image-2512専用のprivate一回消費embedding/mask ABI。BF16互換のMLX出力を値を失わないF32とI32 maskへ変換し、固定`[1,1..1058,3584]`／`[1,1..1058]`、16 MiB payload、candidate／plan／prompt／sample identity、owner-only directory/file、SHA-256、atomic publish、no-follow read、異常時cleanupを検証。M4/32 GiBの英語実promptで28層実行後、別processのconsumerが`[1,13,3584]`／`[1,13]`をdigest完全一致で受け取り、private残存0、pressure normal、thermal nominal。encoder peak MLX 1,580,437,823 bytes／process RSS 1,794,293,760 bytes、consumer peak RSS 48,513,024 bytes。これはtensor transportの認定でありtransformer/VAEや画像生成は未認定。証跡: [private handoff smoke](evaluation/qwen-image-2512-mflux-private-handoff-smoke-2026-09-22.json)
- `[Done]` Qwen-Image-2512量子化transformerのload-free inventory。6 shard／3625 tensor／60 blockをindexとsafetensors headerで照合し、BF16 2779／U32 846、各block payload 191,288,320 bytes、静的22,826,112 bytes、静的＋最大1 block 214,114,432 bytesと算定。846件のU32 packed weightすべてに対応するscales/biases、4-bitの形状比、および各blockの量子化weight存在を照合した。これはpayload下限とlayoutの証跡でありallocator、実RSS、60 block forwardの合格ではない。証跡: [transformer staging](evaluation/qwen-image-2512-mflux-transformer-staging-2026-09-22.json)
- `[Done]` 選択的U32/BF16 readerと4-bit MLX layer構築でtransformer第0・第59 blockの合成forwardを実機検証。現MFLUXの一部8-bit規則をそのまま使うとshape不一致で停止するため、検証済みlayoutに従い4-bitへ固定した。M4/32 GiBで画像`[1,4,3072]`／text`[1,13,3072]`から両有限出力。第0 blockのpeak MLX 193,871,058 bytes／RSS 395,132,928 bytes、第59 blockのpeak MLX 195,337,444 bytes／RSS 393,396,224 bytes、いずれもpressure normal。RoPEを省いた単独block smokeであり、実画像生成の認定ではない。証跡: [block 0](evaluation/qwen-image-2512-mflux-transformer-block0-smoke-2026-09-22.json)、[block 59](evaluation/qwen-image-2512-mflux-transformer-block59-smoke-2026-09-22.json)
- `[Done]` Qwen-Image-2512量子化transformerの合成60 block逐次materialization／forward／解放。M4/32 GiBで60/60 blockの両出力が有限、pressure全件normal／thermal fair、peak MLX 195,353,822 bytes、process peak RSS 498,073,600 bytes、約17.3秒。各block前にメモリ入場判定し、block単位でweightを選択的に読む。画像`[1,4,3072]`／text`[1,13,3072]`の合成状態を伝播させ、RoPE・実prompt・static入出力projection・timestep conditioningの実経路・VAE・画像生成は含まない。したがってfull transformerや生成品質の認定ではない。証跡: [60-block synthetic streaming](evaluation/qwen-image-2512-mflux-transformer-60block-synthetic-streaming-2026-09-22.json)
- `[Done]` 同じ60 block逐次経路にMFLUX標準Qwen RoPE（theta 10000、axes `[16,56,56]`、scaled、合成grid `[1,2,2]`、text 13 token）を接続。M4/32 GiBで60/60 blockが有限、pressure全件normal／thermal fair、peak MLX 197,361,374 bytes、process peak RSS 480,919,552 bytes、約17.9秒。これは合成小gridのRoPE適用であり、実latent寸法・実prompt・static projection・denoising・VAE・画像品質は未認定。証跡: [60-block RoPE synthetic streaming](evaluation/qwen-image-2512-mflux-transformer-60block-rope-synthetic-2026-09-22.json)
- `[Done]` Qwen-Image-2512量子化transformerの固定層を、選択的weight loadで60 block逐次経路へ接続。`img_in`／`txt_norm`／`txt_in`／`time_text_embed`／RoPE／60 block／`norm_out`／`proj_out`を合成image `[1,4,64]`、text `[1,13,3584]`、timestep 0.5で通し、有限出力`[1,4,64]`、60/60 block normal pressure／thermal fair、peak MLX 220,384,096 bytes、process peak RSS 430,620,672 bytesを確認。配置weightには標準MFLUX moduleと異なり`norm_out.linear.bias`があり、静的loaderで対応した。小grid・合成入力に限り、実prompt・実latent・denoising・VAE・画像生成は未認定。証跡: [static + 60-block synthetic](evaluation/qwen-image-2512-mflux-transformer-static-60block-synthetic-2026-09-22.json)
- `[Done]` 英語実promptをQwen text encoder 28層で処理し、F32 embedding／I32 maskをprivate一回消費handoffで別processへ渡して、量子化transformerの固定層・RoPE・60 blockを実行。M4/32 GiBで28/28 encoder層、60/60 transformer block、有限`[1,4,64]`出力、private残存0、pressure全件normal／thermal fair。transformer側peak MLX 220,187,488 bytes／process peak RSS 503,136,256 bytes。画像側は合成`[1,4,64]` latent、timestep 0.5と小gridであり、実latent・denoising・VAE・画像生成品質は未認定。証跡: [real-prompt to streamed transformer](evaluation/qwen-image-2512-mflux-real-prompt-transformer-synthetic-latent-2026-09-22.json)
- `[Done]` Qwen-Image-2512 MFLUX VAEのdecoder専用選択的loadと小latent復元。1 shard／192 BF16 tensorをheaderで照合し、decoder＋post-quant-convの108 tensor／146,591,206 bytesだけを読み、encoder 84 tensorはmaterializeしない。M4/32 GiBで合成packed `[1,4,64]`をunpack `[1,16,4,4]`後にdecodeし、有限`[1,3,1,32,32]`、peak MLX 439,150,838 bytes、process peak RSS 275,267,584 bytes、pressure normal／thermal fairを確認。transformer出力との接続、実latent、生成品質は未認定。証跡: [VAE decoder synthetic latent](evaluation/qwen-image-2512-mflux-vae-decoder-synthetic-2026-09-22.json)
- `[Done]` 合成text／image／timestepで量子化transformer固定層＋RoPE＋60 blockの`[1,4,64]`出力をVAE decoderへ直接渡す小grid一体forward。M4/32 GiBで60/60 block完走、VAEの有限`[1,3,1,32,32]`出力、peak MLX 467,962,770 bytes、process peak RSS 480,002,048 bytes、pressure normal／thermal fair。実promptを加えた同時実行はtext encoder途中のメモリ安全ゲートで停止したため未認定。ゲートは緩めず、外部メモリ使用量が落ち着いてから再確認する。実ノイズ・denoising・実用画像寸法・品質も未認定。証跡: [transformer→VAE synthetic](evaluation/qwen-image-2512-mflux-transformer-to-vae-synthetic-2026-09-22.json)
- `[Done]` Qwen-Image-2512の実ノイズ初期化、MFLUX FlowMatch Euler scheduler、量子化transformer固定層＋RoPE＋60 blockの2回逐次推論、scheduler latent更新、VAE decoder復元を32×32相当の合成promptで接続。M4/32 GiBで2/2 step・120/120 block、各更新latentと最終`[1,3,1,32,32]`が有限、pressure normal／thermal fair、peak MLX 468,062,634 bytes、process peak RSS 492,060,672 bytes。これはループの実行可能性証跡であり、32pxは画像品質を評価できず、実prompt・実用寸法・十分なstep数の生成認定ではない。証跡: [two-step synthetic-prompt denoising](evaluation/qwen-image-2512-mflux-streamed-2step-32px-synthetic-2026-09-22.json)
- `[Next]` 上記denoising経路を実用寸法へ拡大する。128×128・2 stepの初回試験は2/2 step・有限VAE出力まで到達したが、decode直後のmemory pressureがwarningで不合格（peak MLX 896,521,970 bytes、process peak RSS 539,246,592 bytes）。decoder前の不要weight解放とVAE encoderの非保持後、再試験はstep 0のメモリ安全ゲートで停止した。空きメモリが約7–9 GBの現状では反復せず、128pxは開始前10 GB、256pxは14 GB、512pxは20 GB以上を要求する。これらは暫定保守gateであり合格profileではない。失敗のraw証跡: [128px pressure warning](evaluation/qwen-image-2512-mflux-streamed-2step-128px-synthetic-2026-09-22.json)、[cleanup後のadmission stop](evaluation/qwen-image-2512-mflux-streamed-2step-128px-synthetic-cleanup-2026-09-22.json)
- `[Done]` 配置済みQwen-Image-2.1 revision `b3179ad355be050328e483a9dfdd9e60cd62adfa`を画像生成候補catalog、Diffusers静的pipeline readiness、isolated worker routingへ追加。loadなし検査でDiffusers形式、`QwenImage21Pipeline`、33,131,616,240 bytes、27 files、BF16非量子化、transformer 14.23 GB、text encoder 17.53 GB、VAE等1.35 GBを確認。公式Hub APIとの照合でlocal revision一致、非gated、BF16 7,115,124,736 parametersを確認。Qwen Research Licenseは非商用限定。証跡: [Qwen-Image-2.1 readiness](evaluation/qwen-image-2.1-artifact-readiness-2026-09-20.json)
- `[Done]` Qwen-Image-2.1隔離runtimeを`.venv-qwen-image-21`へ構築。Python 3.12.9、Torch 2.14.0、Transformers 5.17.0、Diffusers 0.41.0.dev0 commit `80c7ed262aeffbeb43ef13ae04baeb9b84515a69`を固定し、`QwenImage21Pipeline`のload-free source readinessとMPS buildを確認。再現用requirementsと証跡: [runtime requirements](../requirements/qwen-image-2.1-runtime.txt)、[runtime readiness](evaluation/qwen-image-2.1-runtime-readiness-2026-09-21.json)
- `[Done]` Qwen-Image-2.1 workerの段階的residency contract。text encoderはmodule単位sequential CPU offload、generation側はDiffusers block-level group offloadをtransformer/VAEへ適用し、MPSでは非同期streamを使わず1 block/groupでCPU↔MPS移送する。固定`text_encoder->transformer->vae`宣言、group-offload API、torch device構築の欠落時は全pipeline `.to("mps")`へfallbackせず生成前に拒否し、VAE tilingとbounded callback telemetryを維持
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
- `[Done]` 上記staged releaseの768×768・20 steps再qualificationを一度開始し、weight load前admissionがestimated resident 19,599,447,412 bytesに対する当時のhard ceiling 16,052,303,299 bytesを検出して安全停止。disk適合、memory不適合、weight load未開始であり、失敗後の無条件再実行は行わない。証跡: [768 staged load admission](evaluation/qwen-image-2.1-768-staged-load-admission-2026-09-21.json)
- `[Done]` Qwen-Image 2.1 component別staged loader。required componentに`None`を渡すpipeline loaderが全weightをloadする挙動を実機で検出したため使用せず、processor／text encoder／scheduler／VAE／transformerを各subdirectoryから固有classで直接loadする。text encode後にencoderを解放してからgeneration pipelineを手動構成し、各phaseでoffload契約を再検証、全component同時loadへ暗黙fallbackしない
- `[Done]` staged admissionへTorchAO materialization 1.5倍余裕を追加し、配置artifactの768 profile見積りを19,599,447,412→16,146,610,083 bytesへ更新。約25.8 GB空きのdirect-component実機試験でもtext encoder単体load中にworker hard ceiling超過を検出して安全停止し、process／private outputを回収。証跡: [direct component memory stop](evaluation/qwen-image-2.1-768-direct-component-memory-stop-2026-09-21.json)
- `[Done]` transformer／VAE側のblock streaming materialization。Diffusers 0.41.0.dev0のblock-level group offloadをMPS同期転送・1 block/groupで適用し、512×512 text-to-image 1-step smokeで従来26.67 GB generation peakを17,954,488,320 bytesへ削減して2/2完走。20-step image-editでは19,381,600,256 bytesで2/2完走し、実測peak＋1 GiBをstaged/image-edit admission floorへ反映。証跡: [group-offload smoke](evaluation/qwen-image-2.1-int8-group-offload-smoke-2026-09-22.json)、[image-edit qualification](evaluation/qwen-image-2.1-int8-image-edit-group-offload-2sample-2026-09-22.json)
- `[Done]` group offload適用後の768×768 text-to-imageを既存all-normal 512 baselineから20-step・2 sample再評価。2/2生成、異なるdigest、peak 17,954,488,320 bytes、median wall 393,331.26 ms、thermal fairで実行自体は合格したが、両sampleともpressure warningとなったためstable上限は512×512を維持。証跡: [group-offload 768 attempt](evaluation/qwen-image-2.1-int8-group-offload-768-2sample-2026-09-22.json)
- `[Done]` 768向けleaf-level offload負例。1-step・2 sample smokeは2/2完走したがpeak 17,954,488,320 bytesとpressure warningがblock-levelから改善せず、median wall 59,449.16 msへ増加したため採用せずblock-levelへ戻した。証跡: [leaf offload rejected](evaluation/qwen-image-2.1-int8-leaf-offload-768-rejected-2026-09-22.json)
- `[Done]` text encoder INT4候補をTorchAO 0.18.0でload-free実行可否判定。`Int4WeightOnlyConfig` APIは存在するが、BF16 LinearのCPU量子化が`mslk >= 1.0.0`不足で失敗し、CPU conversion／MPS runtimeの両gateをfalseとした。未検証artifactを生成せず、API・CPU変換・MPS forwardを分離したreadiness probeへ固定した。証跡: [TorchAO INT4 readiness](evaluation/qwen-image-2.1-torchao-int4-readiness-2026-09-22.json)
- `[Done]` INT4必須依存`mslk>=1.0.0`の固定runtime導入可否を`uv` dry-runで検証。利用可能版は`0.0.0`だけで依存解決不能だったため、環境を変更せず停止。証跡: [mslk resolution](evaluation/qwen-image-2.1-mslk-resolution-2026-09-22.json)
- `[Done]` owner-only sample一時領域、worker ABI v3の明示flag、成功／例外時cleanupを備えたprivate disk-backed group offload経路を実装。TorchAO INT8 artifactの512×512・1-step smokeではDiffusers 0.41.0.dev0がTorchAO subclass tensorをsafetensorsへserializeできないとしてhook設定時に明示拒否した。private file残存0を確認し、以後は同組合せをweight load前に拒否する。証跡: [disk offload rejected](evaluation/qwen-image-2.1-int8-disk-offload-rejected-2026-09-22.json)
- `[Done]` 独立process phase loader向けprivate prompt-embedding handoff契約。prompt本文を保存せず、許可tensor名、1〜5次元bounded shape、dtype、512 MiB上限、plan/prompt digest・sample・mode identity、owner-only file、payload SHA-256、atomic manifestを検証する。一回consume時は成功／改ざん／identity不一致のいずれでもpayloadとmanifestを削除する。固定runtimeの実Torch BF16 `[1,86,4096]` embeddingとINT64 `[1,86]` maskでsafetensors round-tripと残存0を確認
- `[Done]` text-encoder専用phase関数と独立process entry point。Qwen processor／encoderだけをlocal-only loadし、既存sequential CPU offload契約でprompt／image-edit conditionをencodeした後、通常CPU tensorへ固定してhooksとMPS cacheを解放し、private handoffへ渡す。既存one-shot private requestを消費し、workspace外handoffをweight load前に拒否する。manifest昇格失敗時の片側payload残存も回帰テストで防止。generation weightをloadしない境界とhook cleanupをunit testで検証。process間の起動順序・実機memory qualificationは未了
- `[Done]` text-to-image generation workerへprivate handoff消費を接続。`--phase-handoff`指定時はworkspace内の親directoryを検証し、plan/prompt/sample/mode digestをweight load前に照合、一回consumeしてからtext encoder weightを再ロードせずgeneration componentだけをloadする。image-editはcondition imageのprocess間リサイズ一致を保証するまで明示拒否し、既存single-process経路は維持
- `[Done]` text-encoder child→正常終了→memory pressure連続normal→generation childの順に実行するbounded二段runnerを実装。encoder timeout時はprocess group TERM→KILL、private request／handoffは成功・失敗時に回収する。Apple M4/32 GiBのINT8 artifactで512×512・1 step・2 sampleが2/2完走、generation child peak RSS 2,552,102,912 bytes、pressure normal、thermal nominal、private残存0を確認。ただし両出力digestは同じで1-step品質・多様性を認定しない。また現reportのpeakはgeneration childのみでencoder phase peakを含まないため、全phase memory qualificationとみなさない。証跡: [two-phase 1-step smoke](evaluation/qwen-image-2.1-int8-two-phase-1step-smoke-2026-09-22.json)
- `[Done]` 二段runnerへencoder childのbounded schema telemetryを追加し、encoder／generation両phaseのpeak RSS・worst pressure／thermal・end-to-end wall/first-outputをsample evidenceへ統合。Apple M4/32 GiB、Qwen-Image-2.1 TorchAO INT8、512×512・20 steps・2 sampleで2/2完走、異なるoutput digest、全phase peak 16,679,387,136 bytes、pressure全件normal、thermal fair、median end-to-end wall 194,783.01 ms、prompt/output非保存・private残存0を確認。1-step旧reportのgeneration-only peakとは比較しない。証跡: [two-phase 20-step 2-sample](evaluation/qwen-image-2.1-int8-two-phase-20step-2sample-2026-09-22.json)
- `[Done]` 二段512×512・20-step・4-sample stability gate。Apple M4/32 GiB、INT8 artifactで4/4完走、異なる4 output digest、全件pressure normal／thermal fair、全phase peak 16,679,387,136 bytesで一定、median end-to-end wall 218,248.85 ms。prompt/output非保存とprivate残存0を確認。証跡: [two-phase 4-sample stability](evaluation/qwen-image-2.1-int8-two-phase-20step-4sample-2026-09-22.json)
- `[Done]` 二段512×512・4-sample all-normal baselineから768×768・20-step・2-sampleを一軸promotion。Apple M4/32 GiBのINT8 artifactで2/2完走、異なるdigest、全件pressure normal／thermal fair、全phase peak 16,679,387,136 bytes、median end-to-end wall 423,561.76 ms、prompt/output非保存・private残存0。これは2-sample昇格gateの合格であり、4-sample安定profile認定ではない。証跡: [two-phase 768 2-sample](evaluation/qwen-image-2.1-int8-two-phase-768-2sample-2026-09-22.json)
- `[Done]` 二段768×768・20-stepの同形状2→4 sample stability gate。Apple M4/32 GiBのINT8 artifactで4/4完走、4つの異なるoutput digest、全件pressure normal／thermal fair、全phase peak 16,679,387,136 bytesで一定、median end-to-end wall 458,826.34 ms、prompt/output非保存・private残存0を確認。この実証条件に限りtext-to-imageのstable解像度上限を768×768へ昇格する。4件目は671,673.57 msと遅く、throughput安定とは主張しない。証跡: [two-phase 768 4-sample stability](evaluation/qwen-image-2.1-int8-two-phase-768-4sample-2026-09-22.json)
- `[Done]` Diffusers image CLIの非初期profile昇格に512初期rootと中間reportの二重bindingを実装。同形状2→4 sampleおよび次解像度の双方でall-normal・provenance・shapeを検証し、chain digestをplanへ固定する。root欠落はworker起動前に拒否する
- `[Done]` 昇格reportのprovenance不一致診断をfield名だけで提示する。異なるmodel artifactを誤指定した場合もdigest等の値・pathを漏らさず不一致原因を特定でき、誤ったbaselineをworker起動前に拒否する
- `[Done]` 768 stable profileから1024×1024への一軸promotionを512初期rootと768 4-sample安定reportのchain bindingで試行。現時点のM4/32 GiBは推定常駐19,028,230,144 bytesに対してdynamic hard ceiling 13,199,013,315 bytesのためload前admissionで安全停止し、生成は未実施。512×512 image-editはwarningを含むためこのtext-to-image baselineへ流用しない
- `[Pending]` 1024×1024の2→4 sample実機検証は、より大容量のApple Siliconまたは実証済み省メモリprofileがload前admissionを通過してから実施する
- `[Pending]` FLUX.2 [dev]をstretch候補とする実artifact配置後の4-bit級量子化、CPU/SSD offload、chunking検証（非量子化weightはM4/32GBでload前にreject）
- `[Done]` diffusion pipelineのmodel、text encoder、VAE別artifact admissionとconservative resident-memory hard ceiling
- `[Done]` privacy-preserving画像生成qualification report schemaとdeterministic evaluator（first-output/wall latency、peak RSS、memory pressure、thermal state、output metadata、plan fingerprint）
- `[Done]` backend-neutralなbounded telemetry event contractとconstant-memory sample collector
- `[Done]` shellを介さないbounded JSONL subprocess telemetry adapterとtimeout時process-group停止
- `[Done]` workspace-bound one-shot worker request、prompt digest binding、0600 atomic保存、consume後unlink
- `[Done]` Diffusers sourceのbounded AST scanによる6候補pipeline class readiness gate（backend import/model/Metal allocationなし）
- `[Done]` FLUX.2/Qwen Image向けDiffusers image worker execution core、pipeline identity gate、streaming output hash、qualification後削除
- `[Done]` local-only Diffusers MPS text-to-image runtime、BF16 compute、VAE tiling、step telemetry、one-shot executable
- `[Done]` Diffusers image-edit worker adapter。worker ABI v2へprivate PNG/JPEG入力のowner/mode/64 MiB上限、SHA-256・size binding、symlink/TOCTOU検査を追加し、ABI v1 text-to-imageを後方互換で維持。Qwen-Image-2.1／FLUX.2 pipelineへRGB sourceを渡すCLI・runner・isolated runtime経路を接続し、入力をreportやoutputへ保存しない
- `[Done]` Qwen-Image-2.1 INT8 image-editの初回load前admission。estimated resident 16,141,367,203 bytesに対し当時のdynamic hard ceiling 15,453,877,699 bytesでworker起動前に安全停止。証跡: [image-edit admission](evaluation/qwen-image-2.1-int8-image-edit-admission-2026-09-22.json)
- `[Done]` Qwen-Image-2.1 INT8 image-editのcondition contract修正。text encoderへcondition imageを含め、precomputed `image_pad_mask`をtext encoder解放後のgeneration pipelineへ明示結合する。condition vision tokensは出力512×512と独立に256pxへ制限し、同一リサイズ画像をvision contextとVAE conditionへ渡す。text encoderはmodule単位sequential CPU offloadで256px conditionを19,373,211,648 bytes、有限embedding `[1,86,4096]`、cleanup後MPS current 1,536 bytesまで実測
- `[Done]` Qwen-Image-2.1 INT8 image-editの512×512・20-step・2-sample実機qualification。block-level group offloadにより最大peakを従来26,672,431,104 bytesから19,381,600,256 bytesへ削減し、2/2生成、異なるoutput digest、median wall 219,401.91 ms、pressure warning、thermal fair、prompt/output非保存、private入力／出力／request cleanupを確認。warningを含むため上位resolution／4-sample promotion baselineには使用しない。証跡: [image-edit group offload](evaluation/qwen-image-2.1-int8-image-edit-group-offload-2sample-2026-09-22.json)。Wan／HunyuanVideo Diffusers worker adapterは実装済み
- `[Done]` MLX-Gen／MFLUX／ComfyUI worker adapter。backend family／version／command kindを型付きcapabilityとして公開し、未知version、symlink request、workspace外request、unsafe executableをworker起動前に拒否する
- `[Done]` 512×512合格後だけ768/1024と連続生成へ進む段階的memory-stability gate。Qwen-Image-2.1 INT8で512×512・2/4 sample合格後に768×768のみ一軸promotionし、warning再現により4 sample/1024昇格をload前停止
- `[Done]` model license、gated artifact、quantization provenanceを記録し、weightと生成画像を保存・uploadしないprivacy gate。Qwen-Image-2.1 reportにartifact root digest、TorchAO INT8、backend/version、licenseを固定し、prompt/PNG非保存とprivate cleanupを確認
- `[Done]` backend-neutralなaudio／music generation qualification契約。speech／music種別、最大30分・8 channel・192 kHzのbounded request、prompt SHA-256、private 0700 root／0600 PCM S16LE WAV、owner／no-follow／inode・size再検証、streaming output digest、duration／sample rate／channel一致、memory pressure／thermal判定、生成物削除を実装し、reportへprompt・音声bytes・pathを保存しない
- `[Done]` version固定MLX Audio speech worker。promptをargv／stdoutへ出さない0600 one-shot request、venv／model／Python path検証、shellなしprocess group、bounded stdout／stderr、timeout回収、0600 WAV、RSS／memory pressure／thermal telemetryを実装。Kokoro-82M-6bit固定revisionを配置し、独立2 sampleを2.977／2.947秒、5.650／5.325秒の音声、peak RSS 788,725,760／787,464,192 bytesで合格。生成物とrequestを全削除
- `[Done]` version固定MLX Audio music workerとMiniMax Music3 affine 4-bit実artifact qualification。Community License、2 shard SHA-256、14 GB available-memory gate、private one-shot requestを固定し、5秒上限・4 stepsの独立2 sampleを30.673／31.441秒、44.1 kHz stereo、peak RSS 8,674,295,808／9,789,440,000 bytesで合格。異なるoutput digest、normal pressure、nominal thermal、request／WAV cleanupを確認
- `[Done]` video generation workload。Wan 2.2 TI2V-5BのMLX-Gen local-only worker、bounded telemetry、T2V qualification、private MP4 digest/delete、33-frame 4-sample stability、49-frame promotionをApple M4/32 GiBで実証
- `[Done]` latent memory manager。Unified Memory arena上で1〜5次元・1/2/4 byte elementのlatent bufferを最大4096件再利用し、shape一致idle hit時は全領域zeroize、active leaseはpin、容量不足時はidle LRUだけを解放、明示trim／closeと非内容telemetryを実装
- `[Done]` temporal／spatial attention state。frame range、temporal overlap context、spatial tile bounds、state bytesをdigest-bound regionとして表し、同一tileの時間方向依存を明示する
- `[Done]` tile、frame／temporal chunk scheduling。最大65,536 task、最大64 concurrency、端tile／端frame処理、単一state admission、peak memoryによるconcurrency clamp、決定論的plan IDを実装

## Phase 8 — MoE and Large Models

- `[Done]` expert residency manager。`(layer, expert)`単位のentry／resident byte二重上限LRU、active lease pin、capacity拒否時の新規resource解放、pressure resizeのsafe-point遅延適用、逆順ownership cleanupを実装
- `[Done]` expert selection telemetry。最大65,536 sampleのringへlayer、選択expert、有限routing weight、latency、cache-hit expertだけを保持し、layer/expert別選択・hit数とmedian／p95／maximum latency、破棄件数を公開する
- `[Done]` correctness-neutral expert predictor。最大65,536 context／historyのfirst-order遷移頻度から決定論的prefetch hintだけを返し、未知contextは空候補、上限超過はFIFO evictionとする。最終実行expertは常にrouter実選択をそのまま採用し、予測が出力へ影響しない
- `[Done]` SSD expert tier。owner-only 0700 rootと0600 file、atomic replace／directory fsync、no-follow read、owner・mode・inode・size・mtime・SHA-256再検証、entry／byte二重上限のLRU、tamper eviction、明示remove／clear、内容非保持telemetryを実装
- `[Done]` hierarchical KV／state cache。private hot-memory／cold-file二階層、entry／byte二重LRU、atomic cold write、owner／mode／no-follow／size／digest再検証、cold-hit promotion、tamper eviction、明示remove／clear、非内容telemetryを実装
- `[Done]` large Unified Memory arena foundation。anonymous mmapによる最大1 TiBのlazy virtual arena、1–64 KiB alignment、最大65,536 zero-copy memoryview suballocation、first-fit free-range、release時zeroize、coalescing、fragmentation／peak／failure telemetry、free-page `madvise`、active allocation中close拒否を実装
- `[Done]` cold prefix、vision／video／audio embedding cache。KVと同じfingerprint-only keyと共有tier budgetで扱い、raw promptをkey／filename／telemetryへ保存しない

## Phase 9 — Multi-Mac

- `[Done]` bounded logical fabric schemaとdeterministic planner。最大64 node／2,016 link／1,024 stage／4,096 state shard、current health、memory capacity、modality、依存DAGを検証し、cycle・unknown dependency・capacity超過をload前拒否する
- `[Done]` Thunderbolt／Ethernet共通のqualified link contract。transport種別、双方向endpoint、実測bandwidth／latency、MTU、measurement ID、peer authenticationを必須化し、未認証・未計測linkを配置候補にしない。payload別transfer時間とmeasurement IDをplanへ固定する
- `[Done]` pipeline／modality-aware partition planner。依存出力の転送cost、resident memory、required modalityから決定論的配置を選び、cross-node transferをstage／node／transport／bytesへ結合する
- `[Done]` distributed KV/state replica sourceとnode failure recovery plan。各shardのSHA-256、size、replica nodeを検証し、healthy replica 0ではfail-closed。以前の配置で故障nodeがnon-checkpointable stageを所有した場合は再実行せず停止し、checkpointable stageだけをhealthy topologyへ再配置する
- `[Done]` Thunderbolt／Ethernet共通のauthenticated stream framing。64 MiB/frame、4 KiB header、magic／ABI version、厳密sequence、source/destination stage・node、transport、planned bytes、measurement ID、payload SHA-256を結合し、short read、replay、改ざん、別plan/link差し替え、endpoint不一致をfail-closed拒否する。stream自身は認証済みpeerだけを受理する
- `[Done]` Ethernet mTLS client／server接続境界。双方で`CERT_REQUIRED`を必須化し、clientはhostnameも検証。handshake後のDER peer certificateをSHA-256 pinへ再照合してからfabric streamを公開する。timeoutは0〜300秒、host／port／payload上限を接続前検証し、pin不一致時はsocketを即時closeする
- `[Done]` 実証明書によるEthernet loopback mTLS transport smoke。相互CA検証、hostname、双方certificate pin、1 MiB×8 frame、sequence／plan binding／payload digestを実socketで検証し、8/8受信、digest不一致0、12.420 ms、675.4 MB/sで合格。初回7/8 timeoutから最終送信half-close→receiver完了→closeへ修正し再合格。物理Ethernet性能認定には拡張しない。証跡: [合格](evaluation/multi-mac-ethernet-loopback-mtls-2026-09-22.json)、[初回close-order failure](evaluation/multi-mac-ethernet-loopback-mtls-failed-close-order-2026-09-22.json)
- `[Done]` bounded Multi-Mac execution coordinator。planとstage／placement／transferの完全一致、最大64並列・256 MiB result、dependency wave単位のmodality並列、pipeline順序、cross-node payload size／digest不変、stage output size、deadline／cooperative cancelを検査し、失敗waveの結果をreportへ公開しない。reportはstage/node/size/digest/latencyだけを保持する
- `[Pending]` 2台以上の実MacでThunderbolt／high-speed Ethernetの帯域・latency・MTUを計測し、peer certificateを固定したtransport qualificationを行う。現在のmTLS証跡は単一Mac loopbackであり物理link認定ではない
- `[Pending]` 上記qualified physical link上でpipeline／modality parallel、distributed KV/state replica、checkpointable node failure recoveryを実行し、単一node baselineとのcorrectness・latency・failure isolationを比較する。planner、framing、coordinatorの論理契約は実装済み

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
- `[Done]` exporter固有のvalue／scale layout・padding・swizzleを明示識別・変換するartifact adapter。contiguous row-major、column-major、row padding、tiled row-major、square power-of-two Morton swizzle、low/high-first nibbleをdescriptorとadapter digestへ結合し、全padding codeを検査してlogical contiguous表現へrepackする。exporter＋recipe完全一致以外はunsupported
- `[Done]` FP8 E4M3FN／E5M2、FP16／BF16／FP32、signed/unsigned INT8／INT4／INT2のbounded CPU参照対応。FP8 finite/subnormal/NaN/Inf境界とoverflowを検査し、2/4/8-bit groupwise affineはscale・zero-point・paddingを明示する
- `[Done]` OCP MX v1.0準拠MXFP4 E2M1、MXFP6 E2M3／E3M2、MXFP8 E4M3／E5M2 adapter。32要素block、E8M0共有scale、4/6/8-bit LSB-first contiguous packingを明示し、全element code、subnormal、signed zero、E4M3 NaN、E5M2 Inf/NaN、E8M0 NaN、unused padding、F32 overflowをCPU参照で検査する。共通symmetric INT8 requantizationへ接続し、format variantを個別`NumericFormat`としてrouting可能にした。vendor固有variantは標準OCP形式へ偽装せず別adapter IDを要求する
- `[Done]` NF4 codebook量子化とgroupwise affine zero-point参照codec。公式16値codebookのlow-nibble packing、末尾padding、有限値・65,536要素上限を検査する
- `[Done]` double quantization、mixed precision、outlier residual sparse表現のbounded CPU参照adapter。scale列のUINT8二重量子化、group最大値による2/4/8-bit mixed precision、threshold超過indexだけを保持するsorted sparse residualを実装し、有限値・65,536要素上限と復元整合性を検査する
- `[Done]` Safetensors／GGUF／MLX／Core ML artifactとGPTQ／AWQ／各exporterのstrict metadata adapter。container、量子化recipe、storage形式、compute形式、bits、group size、packing、exporterを分離したcanonical descriptorへ正規化し、GPTQ／AWQ、MLX affine、Core ML linear、GGUF Q4_0／Q4_K／Q8_0と非量子FP16／BF16／FP32を明示対応する
- `[Done]` weights／activations／KV・recurrent state／MoE expert／vision・audio・diffusion tensorを同一契約で扱うeligibility matrix。source／compute形式、role、backend、operator、要素数範囲、recipe、evidence、qualified状態をcanonical capability IDへ結合し、未知・未認定・曖昧recipeをfail-closed拒否する
- `[Done]` CPU vectorized／MLX／Metalのdecode・repack・requantize・layout変換と、NF4 decode + GEMV/GEMM/attention融合候補。optional NumPy CPUとMLX GPU candidateとして2/4/8-bit unpack／repack、FP8 E4M3FN/E5M2 decode、NF4 encode、byte／nibbleのrow-major／column-major／row-padding／tile-row-major／Morton layout変換を実装し、CPU側とMLX guarded candidateはgroupwise symmetric INT8 requantizeにも対応。全finite FP8 code、padding、全layout、4,097値・4 group sizeでscalar byte／digest完全一致を確認済み。M4の65,536値NF4 encodeはNumPyがscalar比11.44倍で合格。MLX 0.27.1 NF4はNumPy比0.77倍、requantizeはscalar比0.36倍かつfloat32境界差をexact host guardで補うため非昇格。5種byte layoutはdigest一致かつscalar比1.66〜1.93倍で候補昇格。Native MetalのNF4 fused GEMV/GEMM/attentionは参照一致したが、decode後実行比でそれぞれ約0.85倍（既存測定）、0.89倍、0.20倍のため非昇格とし、two/three-passを認定routeに維持。証跡: [NF4 CPU vectorized](evaluation/numeric-vectorized-nf4-m4-2026-09-22.json)、[NF4 MLX comparison](evaluation/numeric-mlx-nf4-m4-2026-09-22.json)、[MLX requant/layout](evaluation/numeric-mlx-requant-layout-m4-2026-09-22.json)、[NF4 Native Metal GEMV](evaluation/numeric-metal-nf4-m4-2026-09-22.json)、[NF4 Native Metal GEMM/attention](evaluation/numeric-metal-nf4-gemm-attention-m4-2026-09-22.json)
- `[Done]` asynchronous tile prefetch、backend completion barrier、read／convert／consume overlapとUnified Memory bandwidth ceilingへの統合。1/2 source buffer、最大65,536 tile・16 MiB/tile、順序付きconsume、cooperative cancel、実buffer peak、出力digestを持つpipelineとthread-safe bandwidth reservation ledgerを実装し、例外・cancel時もleaseを解放する
- `[Done]` load時変換／初回利用時変換／反復利用cache／毎回fused変換を比較するcost modelとphase別route選択。conversion、synchronization、反復compute、amortized uses、peak memory、output digestを比較し、memory ceiling内で出力一致する最小amortized latency routeだけを選択する
- `[Done]` NVFP4→scale付きINT8の表現保存経路と一般的なsymmetric INT8再量子化経路の比較。表現保存側は元block/global scaleで誤差ゼロ、再量子化側はtensor/group scaleのmaximum absolute error・RMSE・storageを算出し、signed INT8、INT32／FP32 accumulator、nvfp4-block／group／tensor scale粒度のbackend対応をfail-closed検証する
- `[Done]` source／scale／layout／kernel／environment digestに結合した変換cache。HMAC署名、private file、atomic manifest-last publish、64 entry／64 GiB hard ceilingのbounded LRU、同一identityのsingle-flight変換共有、出力再hash、tamper quarantine、明示revoke、変換失敗時rollbackを実装
- `[Done]` chip／OS／toolchain／backend version／operator／shape／format／tensor role／recipe別capability probeとCPU reference fallback。最大16 MiBのbounded probeでreference／candidate digestを比較し、不一致・例外をoperator identity単位でquarantineする。未probe／隔離時は明示CPU referenceへfallbackし、未知recipeは実行せずunsupportedを返す
- `[Done]` scalar誤差・operator一致・モデル品質とend-to-end速度／peak memory／throughput／energyを組み合わせたpromotion gate。maximum absolute error、RMSE、operator結果、品質退行上限を順にfail-closed評価し、同一workloadの最低3 sample、digest一致、memory／energy非悪化、5%以上のmedian latency改善を全て満たすrouteだけを昇格する
- `[Done]` CLI数値route診断と英語・日本語・简体中文表示。`numeric-route-diagnostic`がsource／runtime／compute形式、tensor role、route、誤差budget、fallback理由をversioned JSONで返し、非有限・負のbudgetをfail-closed拒否する
- `[Done]` Swift SDK／Mac appで同じ数値route診断schemaをtyped decodeし、英語・日本語・简体中文で表示する。全format／tensor role／route enum、有限非負error budget、schema／valid／message key／fallback上限をstrict検証し、64 KiB以下のowner regular fileをno-followで読むloaderを実装。Mac appは環境指定した証跡だけを読み、route chain、error budget、fallbackを三言語表示する

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
- `[Done]` Core ML compiler出力のprivate disk artifact cache。source digest、graph ID、toolchain、Core ML version、OS build、compute unitsをcache IDへ結合し、symlink／special file拒否、file数／容量上限、tree digest、32-byte以上のsecretによるHMAC-SHA256署名、atomic publish、厳格load、改ざん時quarantine、理由付き明示失効を実装
- `[Done]` CPU thread、GPU command queue、ANE in-flight task、Unified Memory、memory bandwidthを原子的に予約・解放する統合resource ledgerとruntime snapshot
- `[Done]` backend exchangeのversioned dispatch contract。architecture／precision／phase／operator／isolation、deadline／cancel safe point、retryable bounded fallback、逆順shutdownを共通化
- `[Done]` operator graphの依存関係とdevice間同期costをproduction composition rootへ接続。最大64 node／256 edge、欠損dependency・cycle・reserved payload拒否、決定論的topological order、依存結果の明示注入、request context safe point、合計sync cost hard budgetを経て各nodeを共通`BackendEngineRegistry`へdispatchする
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
- `[Done]` prefill、decode、Vision/Audio encoder、sampling、draft/verify別のbounded end-to-end performance profileと共通promotion gate。hardware/model/workload identity、最大256 sample、median/p95 latency、work-unit throughput、peak Unified Memory、全sample取得時だけenergy中央値、deterministic output digestを保持し、同一identity・3 sample以上・出力一致・memory/energy非悪化・5%以上のmedian改善を全て満たす場合だけ昇格する
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
- `[Done]` Core ML compiler artifact disk再利用のtoolchain／Core ML version binding、署名、quarantine、失効（上記共通cacheをpersistent ANE workerとQwen3-VL compiled artifactの双方が利用可能な独立契約として実装）
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
- `[Done]` Qwen3-VL Vision encoderのANE routingとMLX GPU LLM連携。fixed graph／precision／capability／resource identityを結合し、非同期submitとnative main-thread向けinline経路、実画像task限定quality、persistent worker soakまで実証
- `[Done]` Whisper tiny固定AudioEncoderのCore ML `.cpuAndNeuralEngine` routingと3-sample実機promotion evidence。artifact identity、固定I/O shape、有限値、入力別digest、latency、RSS、pressure、thermalを結合し、ANE単独実行を主張せずCPU+ANE許可profileとして認定
- `[Done]` MobileCLIP S0 image/text実artifactのCore ML `.cpuAndNeuralEngine` routingと3+3 sample実機promotion evidence。固定revision／license／source artifact digest、固定I/O、有限値、入力別digest、latency、RSS、pressure、thermal、compile cleanupを結合し、ANE単独実行やzero-shot分類品質は主張しない
- `[Done]` FastViT-T8実artifactのCore ML `.cpuAndNeuralEngine` classifier routingと3-sample実機promotion evidence。固定revision／license／source artifact digest、1000-class probability array、999 unique label dictionary、入力別digest、latency、RSS、pressure、thermal、compile cleanupを結合し、ANE単独実行や自然画像accuracyの再認定は主張しない
- `[Pending]` 音声対応LLMとmodel固有projection artifact配置後、Whisper encoderのGPU LLM projection／end-to-end audio quality gate
- `[Done]` CPUまたはANE draft + GPU verifyのbounded speculative executorとproduction backend registry adapter。未検証tokenをclientへ公開せず、GPU authoritative sequenceとの共通prefixだけをacceptし、最初の不一致はGPU tokenへ補正する
- `[Done]` speculative profileのowner-only atomic永続化、exact model／precision identity、再計算profile ID、qualified-only load gate
- `[Done]` Gemma 3 1B→4B native MLX実測候補は3/3同一出力だが32.1%低速化かつGPU draftのため不合格として証跡化し、profile保存／decode route昇格をfail-closed拒否
- `[Next]` CPU/Core ML draft artifactによるmodel固有draft／verify実測profile生成と、5%以上改善した合格profileだけの標準decode route昇格
- `[Done]` contention全ペア合格時だけ原子的に一括予約するCPU／GPU／ANE bounded pipeline並列化
- `[Done]` hardware／OS／model／shape別autotuning profile。model-backed shape、hardware/source/environment fingerprint、private atomic profile、OS/toolchain/MLX変更時の失効、TTL、quarantine、明示的再計測restore、last-known-good rollbackをnative v2 tuningとdevice placementで実装
- `[Done]` backend別correctness比較、timeout／compile failure／numerical mismatch時のANE → GPU → CPU fallback。同一workloadのCPU reference digestと数値誤差を昇格時に検証し、昇格済みCore ML routeの固定retryable codeからprobe済みGPU、CPUへbounded resource引き継ぎでfallback
- `[Done]` TTFT、TPOT、tokens/sec、frames/sec等をphase固有latency／work unitとして扱い、energy/requestとpeak Unified Memoryを組み合わせる共通promotion gate。energy未計測はunknownのまま保持し、存在するbaseline energyを悪化させる候補は拒否する
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
- `[Done]` profile persistence、scheduler admission、worker crash、client切断を横断するplatform-wide fault scenario matrix。atomic replace前中断→last-known-good、queue後reservation前拒否、reservation後worker crash、first chunk後disconnectを必須scenarioとし、service readiness、reservation／temporary file leak、prompt／output非保存を共通合格条件にするbounded reportを実装

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
- `[Done]` operator別backend選択／fallback telemetry。共通backend registryがrejected／failed／succeededをbackend・理由別にthread-safe集計し、256 key上限とoverflow countを持つversioned snapshotを公開する。quarantine状態は既存probe registryのoperator単位snapshotと組み合わせて診断する
- `[Done]` GPU/CPU utilization、Unified Memory bandwidth、thermal、powerを、内容を保持しない最大1024件のthread-safe ringへ記録する。利用不能なGPU／bandwidth／powerは0へ偽装せず`null`とし、median／p95／maximum、thermal状態別件数、破棄件数をversioned snapshotで公開する
- `[Done]` Vision、Audio、Video固有metricsを共通bounded schemaへ統合する。modality／operation、latency、入力・出力unit数、peak memory、成功可否だけを保持し、modality別sample／failure／latency median・p95・maximum／累積unit／最大peak memoryを集計する

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
- `[Pending]` vLLM 0.28.x / Transformers 5.15.x専用runner構築後の昇格試験。現workspaceの4 Python venvにはvLLM／vLLM-Metalがなく、Homebrew版vLLM-Metalは0.29.0、配置済みTransformersは5.16.1／5.17.0であるため、現stackを0.28.x認定証跡へ流用しない
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
- `[Pending]` 大容量Apple SiliconでQwen text-only実model qualification
- `[Pending]` 専用runner上でvLLM 0.28.x昇格workflowを実行
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
- `[Pending]` protected `mac-release` environment上で実資格情報による初回notarized artifact生成
- `[Pending]` 初回notarized artifactをexact tagへ結合しdraft release昇格を実行
- `[Done]` Ruff ruleをimport整列`I`へ段階拡張し、Python全対象の既存247件を機械修正して常設CI対象へ昇格

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
105. `[Pending]` 大容量Apple SiliconでQwen text-only実model qualification
106. `[Done]` Mac companion app。`samples/VLLMAppleOptimizer`で三言語UI、dry-run、明示確認付きexport、stage progress、pause／continue／cancel／resume、perplexity・generation比較、provenance／未評価能力表示を実装
107. `[Done]` M4/32GB画像生成基盤とFLUX.2 [klein] 9B Base 512×512実機qualification。Qwen-Image-2512と量子化FLUX.2 [dev]は個別のmemory gateとして未昇格
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
121. `[Next]` Qwen-Image-2512-4bitのcomponent streaming/offloadと現行Macでの段階的memory-stability qualification。worker、readiness、3言語実promptの28層text encoder、英語実promptのprivate別process handoffから量子化transformerへの接続、合成promptでの32px・実ノイズ2 step→VAEは`[Done]`。実prompt＋denoisingの一体実証、実用寸法・必要step数・画像品質が残る。大容量Apple Siliconを必要とする一括経路の再qualificationは別途`[Pending]`
122. `[Done]` MFLUX Z-Image/Qwen Image backend classとartifact形式を分離したloadなしreadiness gate
123. `[Done]` 配置済みMLX Diffusers変換artifactの量子化layer非互換を実機で特定し、別directoryへMLX-Gen 4-bit packageを生成する配置手順を確定
124. `[Done]` MFLUX Z-Image/Qwen Image local-only one-shot worker、private output digest/delete、memory ceiling telemetry接続。`mflux`独立distributionと`mlx-gen`同梱MFLUXをload-freeで識別し、配置済みQwen-Image-2512 4-bitへbounded qualification CLIを接続。Qwen image-editはABI v2入力を再検証後に0600 PNGへ複製し、生成終了・例外時とも即時削除する
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
158. `[Pending]` MLX-Gen側のblock streaming ABI実装後に行うweight residency profileと512 root再qualification
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
244. `[Done]` native modelを専用processのmain threadで生成・実行・closeするbounded inference transport。最大64 pending、4 MiB request／16 MiB responseのJSON-lines IPC、親側fail-fast backpressure、子側bounded queue、active cancel／deadline safe point、固定error code、worker exit時pending解放、shutdown→terminate→kill回収を実装。fake native delegateの同一process順序、child main-thread ownership、queue saturation、active cancel、実HTTP 408 timeout、client切断後のrequest slot／子safe-point cleanupと後続request回復、実`/v1/chat/completions`経路、main-thread closeを確認
245. `[Done]` inline PNG／JPEG、private request workspace、persistent Core ML encoder、scheduler予約内MLX生成、OpenAI互換responseを扱うQwen3-VL managed delegateを専用main-thread process factoryへ接続。strict config schema、5 compiled model、graph/model/revision identity、ANE capability、Unified Memory／CPU／GPU queue／bandwidth ledgerをload前に検証し、owner-thread生成・実行・close、HTTP chat、timeout前停止、private workspace cleanupを回帰試験で確認
246. `[Done]` 固定revisionからBF16 vision weight staging、patch＋4 segment build、個別数値qualification、atomic profile公開を行う再現可能builderを追加。再構築した5 compiled modelはgraph ID一致、patchと全segmentの固定誤差gateに合格。証跡: [runtime profile rebuild](evaluation/qwen3-vl-coreml-runtime-profile-rebuild-2026-09-22.json)
247. `[Done]` 上記process factoryをHomebrew vLLM-Metal 0.29.0／実MLXでqualification。venv entry pointのsymlinkを解決してsite-packagesを失う起動障害と、queue飽和を一般backend failureへ潰してHTTP応答なしになる障害を修正した。円・家・OCRの英語／日本語／简体中文9/9正答、3並行で2成功＋1 `engine_busy`、active timeout 408、client disconnect 499、cancel後の正常推論回復、main-thread ownership、restart 0、全resource解放、正常shutdown、private cleanupを確認。証跡: [Homebrew process factory qualification](evaluation/qwen3-vl-managed-process-factory-homebrew-2026-09-22.json)。一般runtime昇格はopen-domain品質gateまたはBF16相当precisionの実装まで保留

この順序により、まず推論runtimeの実model安定性を確立し、その境界を壊さずにoptimizerを
別processとして追加する。構造pruningはquantization、calibration、評価gateの後に着手する。
