# Apple Silicon向け次世代vLLM実行基盤 設計仕様書

## 1. プロジェクト概要

### 1.1 仮称

**vLLM-Apple Runtime**

正式名称は後から変更可能とし、本仕様では `vllm-apple` と呼ぶ。

本プロジェクトは、vLLMおよびvLLM-Metalを基盤として、Apple SiliconのCPU、GPU、Unified Memory、Metal、MLX、将来的にはNeural Engineを統合的に利用する、高性能かつモダリティ非依存のAI推論ランタイムを構築することを目的とする。

単なる「vLLMのMac版」あるいは「vLLM-Metalの高速版」にはしない。

最終的には、

* LLM
* Vision Language Model
* 画像生成モデル
* 動画理解モデル
* 動画生成モデル
* 音声認識モデル
* 音声生成モデル
* 音楽生成モデル
* Audio Language Model
* World Model
* Multimodal Agent
* 将来的な構造記憶・外部記憶型モデル

を同一の実行基盤上で扱える、

**Apple Silicon向け汎用AI Runtime OS**

を目指す。

現在のvLLM-MetalはApple Silicon上でMLXを主要計算backendとして使用するvLLM hardware pluginであり、Paged Attentionなども実装している。

一方で、2026年8月現在のnative multimodal対応は主として画像入力に限定され、動画入力はまだ対象外である。

したがって、本プロジェクトでは最初から「テキストLLM専用」という前提を置かない。

---

# 2. 基本設計思想

## 2.1 Model-centricではなくWorkload-centricにする

従来の推論runtimeは、

```text
Model
 ↓
Operator
 ↓
GPU
```

という構造を取ることが多い。

本プロジェクトでは、

```text
Workload
 ↓
Execution Graph
 ↓
Runtime Planner
 ↓
最適な演算器・メモリ・kernel
```

とする。

つまり、

「このモデルはGPUで動かす」

ではなく、

「今この瞬間のこの処理をどこで実行するのが最適か」

をruntime自身が判断する。

---

## 2.2 モダリティ非依存

LLMと映像AIを別runtimeとして作らない。

最下層ではすべてを、

```text
Tensor
Sequence
State
Memory
Stream
Event
```

として扱う。

例えば、

```text
Text token
Audio frame
Video frame
Image patch
Latent token
Spike/Event
```

を異なるデータ型として扱いつつ、同じscheduler上で処理可能にする。

---

## 2.3 Unified Memory First

Apple SiliconではCPUとGPUがUnified Memoryを共有する。

そのため、

```text
CPU RAM
 ↓ copy
GPU VRAM
```

というCUDA型の前提を設計中心に置かない。

基本モデルを、

```text
                   Unified Memory
                  /       |       \
                CPU      GPU      ANE
                 │        │        │
                 └──── Runtime ────┘
```

とする。

重要なのは、

**「データをどこへ移すか」より「そのデータを誰が処理するか」**

である。

---

# 3. システム全体構成

```text
┌────────────────────────────────────────────┐
│               Client Layer                 │
│                                            │
│ OpenAI API / Anthropic API / Native API    │
│ WebSocket / Streaming / Agent API          │
└────────────────────┬───────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────┐
│             Request Coordinator            │
│                                            │
│ Text / Image / Audio / Video / Agent       │
└────────────────────┬───────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────┐
│             Modality Frontend              │
│                                            │
│ Text Tokenizer                             │
│ Vision Encoder                             │
│ Audio Frontend                             │
│ Video Decoder                              │
│ Latent Encoder                             │
└────────────────────┬───────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────┐
│             Apple Runtime IR               │
│                                            │
│ Tensor / Stream / State / Event / Graph    │
└────────────────────┬───────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────┐
│          Adaptive Runtime Planner          │
│                                            │
│ Compute Planner                            │
│ Memory Planner                             │
│ KV/State Planner                           │
│ Stream Planner                             │
│ Kernel Planner                             │
│ Thermal Planner                            │
└───────────┬─────────────┬──────────────────┘
            │             │
      ┌─────▼─────┐ ┌────▼─────┐
      │ MLX       │ │ Native    │
      │ Backend   │ │ Metal     │
      └─────┬─────┘ └────┬─────┘
            │             │
      ┌─────▼─────────────▼─────┐
      │ Apple Silicon Hardware   │
      │                          │
      │ CPU / GPU / ANE          │
      │ Unified Memory / SSD     │
      └──────────────────────────┘
```

---

## 3.1 三層の責務境界

本システムを次の三層として扱う。現在実装済みのcontrol planeと、今後tokens/secを直接改善する
execution planeを混同しない。

| 層 | 主な責務 | 性能上の役割 |
|---|---|---|
| Control Plane | API、request scheduling、process isolation、memory safety、Swift SDK | 安定性、同時実行、公平性 |
| Execution Plane | prefill/decode、batching、attention、quantized compute、speculation | TTFT、TPOT、tokens/sec |
| Apple Runtime | Unified Memory、CPU/GPU/ANE、memory pressure、thermal、chip capability | 安全な資源上限と配置 |

三層は中央の`AppleExecutionPlanner`が結ぶ。control planeの改善だけを根拠に推論kernelの
高速化を主張せず、throughput改善はexecution planeを含む実測で判定する。

```text
Control Plane
      │ request / policy
      ▼
AppleExecutionPlanner ─── Apple Runtime profile
      │ versioned execution plan
      ▼
Execution Plane ───────── MLX / Metal / vLLM-Metal
```

---

# 4. ソフトウェア階層

## 4.1 vLLM Core Layer

可能な限り本家vLLMをforkしない。

vLLM側の、

```text
Request Scheduling
Continuous Batching
OpenAI-compatible API
Prefix Caching
Speculative Decoding
Model Registry
Distributed Execution
```

などを利用する。

vLLMは現在hardware plugin方式を採用しているため、Apple専用実装はできる限りpluginとして分離する。

理想は、

```text
vLLM upstream
      │
      ▼
vLLM hardware plugin API
      │
      ▼
vllm-apple
```

である。

## 4.2 BackendEngine契約

process isolationを維持したまま、実行backendを次の共通契約で交換可能にする。

```text
BackendEngine
 ├─ VLLMMetalBackend   # 初期の標準経路
 ├─ NativeMLXBackend   # MLX graphを直接制御する経路
 ├─ NativeMetalBackend # 実測で差が出るkernelのみ
 ├─ CoreMLDraftBackend # 固定shapeのdraft model候補
 └─ CPUBackend         # sampling、補助処理、correctness fallback
```

各backendはcapability、version、対応model architecture、precision、phase、fallback理由を
機械可読に返す。未対応機能を暗黙に別方式へ変更せず、plannerが明示的にfallbackを選ぶ。

実装済み`BackendEngineDescriptor`はbackend enum、version、model architecture、precision、phase、operator、
isolation方式をversion付きで固定する。`BackendEngineRegistry`はrequestの明示candidate順だけを評価し、未登録、
未ready、architecture／precision／phase／operator不一致を理由付きattemptとして拒否する。実行前後には共通の
`InferenceRequestContext`でdeadline／cancel safe pointを検査し、retryableと宣言されたbackend failureだけを次候補へ
fallbackする。non-retryable failureはCPUへ暗黙退避しない。shutdownは逆登録順で全engineへ伝播し、失敗をboundedに
集約する。

registryは各attemptをbackend、status（rejected／failed／succeeded）、reason別にthread-safe集計する。
telemetry keyは256件を上限とし、未知の動的failure reasonで無制限に増えないよう超過分をoverflow countへ集約する。
このversioned snapshotとoperator単位probe quarantine snapshotを併読することで、選択、fallback、隔離の理由を追跡できる。

`ProductionBackendComposition`はproduction chat engine factoryを上記registryへ直接登録する。全engineの起動が
完了した場合だけregistryを公開し、途中失敗時は開始済みengineを逆順回収する。chat payloadは
`BackendEngineRequest`内で保持し、`ManagedInferenceBackendEngine`がcontext-aware ABIだけを許可する。
busy／実行失敗はretryable、payload／結果schema不一致はnon-retryableとして分類する。
`BackendRegistryInferenceEngine`がmodel一覧、chat、diagnostics、冪等shutdownを`RuntimeService` ABIへ接続する。
Qwen3-VLのCore ML encoder＋MLX生成専用process qualification runnerはこのcomposition rootを通るため、
実backendとテスト用registryが別経路になることを防ぐ。

複数operatorのproduction compositionは`OperatorGraphDispatcher`を使う。graphは最大64 node／256 edgeに制限し、
欠損dependency、cycle、重複ID、予約済み`_dependencies` payloadを実行前に拒否する。決定論的topological orderで
各nodeの宣言済み依存結果だけを注入し、node間のprofile済みsynchronization nanoseconds合計がrequestのhard budgetを
超えない場合だけ共通`BackendEngineRegistry`へdispatchする。各node前には同じrequest deadline／cancel safe pointを検査する。
phase vocabularyはprefill、decode、sampling、Vision／Audio encoder、embedding、classifier、draft、verify、auxiliaryを
共通契約として持つ。encoder／embeddingからprefill、prefill／verifyからdecodeまたはsampling、draftからverifyという
許可済み依存だけを受け入れ、decodeからprefillのような逆向きedgeは一nodeも実行する前に拒否する。

end-to-end performance profileはprefill、decode、Vision encoder、Audio encoder、sampling、draft、verifyを
明示phaseとして区別する。hardware、model、backend、shape／batch／contextを含むworkload identityへ最大256 sampleを
束縛し、median／p95 latency、work-unit throughput、peak Unified Memory、全sampleで取得できた場合だけenergy/request、
deterministic output digestを集計する。promotionは同じhardware/model/phase/workload、双方3 sample以上、出力digest一致、
peak memory非悪化、計測済みenergy非悪化、median latency 5%以上改善を同時に要求する。

Core ML compiler出力を再利用する場合は`CoreMLArtifactCacheIdentity`へsource SHA-256、graph ID、
toolchain version、Core ML version、OS build、compute unitsをすべて結合する。cache root／entry／manifestは
private permissionとし、compiled treeのsymlink・special file、8,192 files超、32 GiB超を拒否する。
publishはprivate temporary treeへcopy・fsync後にatomic renameし、manifestとtree digestを32-byte以上の
secretによるHMAC-SHA256で署名する。load時はidentity、署名、全tree digestを再計算し、一つでも不一致なら
active entryから外してquarantineする。toolchain失効等は理由付きrevocation recordとともに同じquarantineへ移す。

---

# 5. Apple Runtime IR

このプロジェクトで最も重要な部分の一つとする。

MLX graphやvLLM graphを直接Apple専用schedulerへ渡すのではなく、その間に独自IRを置く。

## 5.1 IRノード

最低限、

```text
MatMul
GEMV
Attention
PagedAttention
MLA
Convolution
Normalization
Activation
Embedding
Routing
Sampling
FFT
STFT
Resample
ImageResize
VideoDecode
VideoEncode
StateRead
StateWrite
KVRead
KVWrite
MemoryTransfer
Synchronization
```

を表現できるようにする。

将来的には、

```text
SparseAttention
StateSpace
Mamba
Diffusion
FlowMatching
Wavelet
NeuralCodec
Spiking/Event
```

も追加可能とする。

---

# 6. Hardware Profiler

Macごとに性能特性が大きく異なるため、固定テーブルだけに依存しない。

初回起動時にmicro benchmarkを実施する。

測定項目は、

```text
CPU GEMM
CPU GEMV
GPU GEMM
GPU GEMV
Memory bandwidth
Metal launch latency
Unified Memory access latency
Attention throughput
Quantized matmul
FFT
Convolution
Image preprocessing
Memory pressure behavior
SSD sequential bandwidth
SSD random bandwidth
```

など。

結果を、

```text
~/Library/Application Support/vllm-apple/profiles/
```

等に保存する。

例：

```json
{
  "soc": "Apple M4",
  "memory_gb": 32,
  "gpu_cores": 10,
  "memory_bandwidth_measured": 112.4,
  "profile_version": 3
}
```

静的なSoC名称だけでなく、OS、Metal feature set、利用可能backend、対応precision、
実測bandwidthとlaunch latencyを含むversioned `AppleChipProfile`として保存する。

## 6.1 AppleExecutionPlanner

`AppleExecutionPlanner`を実行方針の唯一の決定点とする。

入力：

```text
ModelArchitectureProfile
StateMemorySpec
AppleChipProfile
WorkloadPhase (prefill / decode / auxiliary)
available memory / memory pressure / thermal state
measured kernel and backend profiles
```

出力するversioned `AppleExecutionPlan`：

```text
weight / state precision
context and state budget
prefill batch / decode batch
attention backend
speculation mode
CPU / GPU / ANE assignment
kernel selection
fallback chain
```

同じprofileとpolicyからは決定論的なplanを生成し、hard memory ceilingを超えない。
dry-runとdecision reasonを提供し、未知のcapabilityでは保守的なfallbackを選ぶ。

hardware snapshotはNSProcessInfoのthermal stateとactive power sourceに対応するpower modeを含む。
thermalは`nominal/fair/serious/critical/unknown`、powerは`automatic/low_power/high_power/unknown`へ
正規化し、取得不能時は`unknown`とする。保存済み旧profileと旧daemon responseには両fieldがないため、
Python loaderは`unknown`補完、Swift SDKはoptional decodeで後方互換を保つ。これらはtransient入力であり、
hardware/model identity fingerprintには結合しない。

plannerはthermal/power値をexecution planの入力としてplan identityとdecision reasonへ固定する。
memory pressureがwarning/critical、thermalがserious/critical、またはlow-power modeではprefill batchを1、
thermalがfairまたは状態がunknownなら2へ制限する。nominalかつautomatic/high-powerでのみ通常上限4を許可する。
適用は既存のscheduler safe pointを通すため、active requestの途中でbatch policyを変更しない。
daemonは15秒間隔のbounded monitorで状態を再取得し、同一値をcoalesceする。変更時はcurrent hardware
snapshotをatomicに置換して`runtime.operating_state` eventを発行する。monitorはshutdownで停止し、probe失敗を
control-plane failureへ昇格させない。動的plan再生成はmodel/chip profileを保持するruntimeでのみ後続実装する。
Swift SDKはstate-change eventをtyped current/previous値として公開する。未知のcurrent enum値ではtyped viewを
構築せずraw eventを保持し、protocol拡張によってevent stream全体が停止しないようにする。

---

# 7. Adaptive Compute Scheduler

## 7.1 CPU/GPU動的振り分け

モデル単位ではなくoperator単位で決定する。

例えば、

```text
large GEMM          → GPU
small GEMV          → CPU/GPU比較
sampling            → CPU
tokenization        → CPU
MoE routing         → CPU
attention           → GPU
FFT                  → CPU/GPU比較
image convolution   → GPU
video preprocessing → GPU
```

とする。

ただし固定ルールではなく実測値によって変更する。

## 7.2 PrefillとDecodeの分離

prefillとdecodeは同じbatch policyやkernel profileを共有しない。

```text
Prefill → compute-intensive / 大きいGEMM / GPU batch / TTFT最適化
Decode  → bandwidth-intensive / GEMV寄り / 小batch / TPOT最適化
```

各phaseに独立したmemory budget、batch上限、kernel selectionとmetricを持たせる。
CPU/GPU/ANEの同時利用は、共有memory競合を含むend-to-end実測で有利な場合だけ採用する。

KV／recurrent stateのprecision変更は`MeasuredStateBackendAdapter`がbackend所有bufferに対して行う。
FP32からFP16またはsymmetric INT8へ実際にencode/decodeし、maximum absolute error、RMSE、cosine similarity、
memory削減率の全gateに合格した組み合わせだけを`promoted_precisions`へ追加する。adaptive allocatorが生成したplanは
source precisionとtarget byte数を再検証してstagingし、commitまでは現行bufferを置換しない。stale plan、未昇格precision、
byte見積り不一致はfail-closedとし、rollbackでは元bufferをそのまま維持する。
同じadaptive planはKV、recurrent、prefix、attention window、MoE expert、workspaceを共通state kindとして扱う。
したがって個別cacheが競合して同じ空きmemoryを二重計上せず、age／pressure順のreprecision／evictionを一つの
backend transactionでcommitまたはrollbackできる。pinned stateは全kindで変更対象外とする。

---

# 8. Kernel Auto Tuner

モデルロード時に、

```text
model
quantization
shape
batch
context
SoC generation
```

からkernelを選択する。

例：

```text
M4
Qwen系
MXFP4
batch=1
decode
```

と、

```text
M4 Max
Qwen系
MXFP4
batch=32
prefill
```

では別kernelを使用可能とする。

vLLM-Metalでは、Unified paged varlen Metal kernelの導入によって旧版比で大幅なTTFT・throughput改善が報告されており、Apple Siliconでは専用kernelの最適化余地が非常に大きい。

---

# 9. Graph Fusion

以下のような連続処理を可能な限り融合する。

```text
RMSNorm
 ↓
Dequant
 ↓
MatMul
 ↓
Bias
 ↓
Activation
```

を、

```text
FusedKernel
```

へ変換する。

画像では、

```text
Resize
Normalize
Patchify
Projection
```

を融合可能にする。

音声では、

```text
PCM
 ↓
Resample
 ↓
Window
 ↓
FFT
 ↓
Mel
 ↓
Normalize
```

の一部を融合する。

映像では、

```text
Decode
 ↓
Resize
 ↓
Color conversion
 ↓
Normalize
 ↓
Vision Encoder
```

の中間コピー削減を最重要課題とする。

---

# 10. Unified Memory Manager

一般的なGPU runtimeのVRAM allocatorではなく、

**Unified AI Memory Manager**

を実装する。

管理対象：

```text
Model weights
KV cache
Recurrent state
Attention window state
Vision embeddings
Audio state
Video frame tensors
Diffusion latent
MoE experts
Prefix cache
Agent memory
Temporary workspace
```

## 10.1 StateMemorySpec

「LLM stateは常に標準KV cache」という前提を置かず、architectureごとの状態を
`StateMemorySpec`で表現する。

```text
weights
kv_cache
recurrent_state
prefix_state
attention_window
expert_working_set
scratch_workspace
```

standard MHA/GQA、SWA、MLA、state-space、Mamba/GDN、hybrid architectureごとに、
bytes/token、固定state、window依存state、quantization可否を計算する。Paged Attentionの
管理はまずvLLM-Metalの実装を利用し、計測可能な不足がbackend拡張で解消できない場合だけ
独自実装を検討する。

---

# 11. Memory Priority

各メモリブロックに、

```text
HOT
WARM
COLD
RECOMPUTABLE
PERSISTENT
STREAMING
```

を付与する。

例：

```text
直近KV              HOT
現在使用expert      HOT
過去KV              WARM
古いvision embedding COLD
Video frame          STREAMING
Intermediate latent  RECOMPUTABLE
Model weight         PERSISTENT
```

---

# 12. Adaptive KV Cache

vLLM-MetalにはPaged KV cacheが存在する。

本プロジェクトではさらに、

```text
L0 : hot KV
L1 : Q8 KV
L2 : Q4 KV
L3 : compressed Unified Memory
L4 : SSD
```

という階層型cacheへ拡張する。

重要度は、

```text
recency
attention frequency
prefix reuse probability
semantic relevance
agent ownership
```

から評価する。

長contextでは、backend capabilityと品質gateが許す場合に限り、次の段階的圧縮を選べる。

```text
recent state  → 高精度
older state   → INT8
cold state    → Q4またはsemantic compression
```

precision変更はattention結果の互換性、復元cost、perplexity/task scoreを評価し、
品質基準を満たさない場合は高精度stateへfallbackする。

---

# 13. Context自動設定

ユーザーが手動で、

```text
--max-model-len 65536
```

などを指定しなくてもよいようにする。

モデルロード時、

```text
モデルサイズ
利用可能Unified Memory
KV bytes/token
OS reserve
現在のmemory pressure
```

から安全な最大contextを計算する。

例：

```text
Detected: M4 / 32GB
Model: 14.3GB
Safe KV budget: 8.4GB

Recommended context:
32768  SAFE
65536  BALANCED
98304  AGGRESSIVE
```

とする。

ロード後はbackendが報告した実KV capacityと設定上限の小さい方をhard ceilingとする。直近最大64件の
admitted context workloadだけをbounded履歴として保持し、拒否requestは学習しない。thermal stateがfairの
場合はceilingの75〜87.5%、seriousは50〜75%、criticalは50%へ16-token block単位で縮退する。履歴需要へ
25%の余裕を加えた値を各範囲内で採用し、nominal復帰時はhard ceilingまで回復する。unknownは既存clientとの
互換性を保つため追加縮退せず、memory admissionの別gateで保守的に扱う。effective contextの変化は
`runtime.context_reevaluation`へ非機密snapshotとして発行する。

---

# 14. MoE Runtime

Apple Siliconにおける重要機能とする。

Unified Memoryの大容量性を活かして、

```text
Total parameters   120B
Active parameters   10B
```

のようなモデルを扱う。

## 14.1 Expert Predictor

過去のexpert選択から次のexpertを予測する。

```text
E4 → E17 → E31 → E17
```

から、

```text
P(E31)
P(E4)
P(E9)
```

を推測し、優先的に準備する。

これはcorrectnessには影響させず、外れた場合は通常処理へfallbackする。

---

# 15. Speculative Execution Manager

vLLM側のspeculative decodingを利用しつつ、Apple runtime側で方式を自動選択する。

vLLM-Metalでもspeculative decodingは現在設定可能である。

候補：

```text
Normal decode
Draft model
MTP
n-gram
Suffix prediction
EAGLE系
```

入力entropyやacceptance rateを測り、

```text
speculation ON
 ↓
accept率低下
 ↓
OFF
```

のように動的変更する。

将来の異種speculationでは、小型draftをCPUまたは対応時のCore ML/ANE、main modelのverifyを
GPUに配置する。ただしUnified Memory競合、同期cost、acceptance rateを含む実測で通常decodeを
上回る場合だけ有効化し、結果のcorrectnessを変えない。

---

# 16. 映像対応

ここは最初からアーキテクチャ上組み込む。

現在のvLLM-Metal native multimodalは画像中心で、動画はまだ対応範囲外である。

本プロジェクトでは動画を単に「画像の連続」として扱わない。

---

# 17. Video Pipeline

```text
Video file / Camera
        ↓
Video Decoder
        ↓
Frame Scheduler
        ↓
Temporal Sampler
        ↓
Vision Encoder
        ↓
Temporal Encoder
        ↓
Multimodal Model
```

---

# 18. 映像デコード

モデル推論前にCPUで動画を完全decodeしない。

可能な限り、

```text
Compressed video
 ↓
Hardware decode
 ↓
GPU accessible buffer
 ↓
Model preprocessing
```

とする。

将来的には、

```text
H.264
HEVC
ProRes
AV1
```

などをhardware decoding pathへ接続する。

重要なのは、中間RGBフレームの不要なコピーを避けること。

---

# 19. Temporal Scheduler

動画モデルでは全フレームを均等に処理しない。

例えば60fps映像に対して、

```text
motion low
→ 2 fps sampling

motion high
→ 12 fps

scene change
→ key frame強制抽出
```

など、内容依存samplingを可能にする。

---

# 20. Video Cache

映像用cacheはKV cacheとは分ける。

```text
Frame Cache
Patch Cache
Vision Embedding Cache
Temporal Feature Cache
Scene Cache
```

とする。

例えば同一映像に複数の質問を行う場合、

```text
Video decode
Vision encoding
```

を毎回繰り返さない。

---

# 21. 将来的な動画生成

Diffusion Transformer、Flow Matching、autoregressive videoなどを同じIRで扱えるようにする。

管理対象：

```text
Latent video tensor
Temporal attention KV
Spatial attention KV
Noise state
Condition embeddings
Reference image embeddings
```

特に大量のlatent tensorを扱うため、

```text
Tile
Temporal chunk
Frame chunk
```

をschedulerが分割できるようにする。

---

# 22. 音声対応

音声も将来追加ではなく、初期設計段階からstreaming modalityとして扱う。

---

# 23. Audio Runtime

```text
Microphone / Audio file
          ↓
      Audio Ring Buffer
          ↓
      Resampler
          ↓
     Feature Encoder
          ↓
 Audio / Speech Model
          ↓
        LLM
```

対応用途：

```text
Speech recognition
Speech-to-speech
Voice chat
Speaker recognition
Audio understanding
Music analysis
Music generation
Sound generation
```

---

# 24. Real-time Audio

音声についてはLLMとは異なり、deadlineを持たせる。

例：

```text
Audio deadline = 10 ms
LLM decode     = best effort
Video analysis = background
```

schedulerは、

```text
REALTIME
INTERACTIVE
NORMAL
BACKGROUND
```

のpriorityを持つ。

音切れを防ぐため、リアルタイムaudio threadでは、

```text
malloc
lock
blocking IO
model loading
```

を禁止する。

---

# 25. Streaming Audio State

音声は長時間入力を一つの巨大contextにしない。

```text
Audio frame
 ↓
Streaming encoder
 ↓
Persistent state
```

とする。

例えば、

```text
Conformer state
Mamba state
Audio KV
Speaker embedding
```

を保存する。

---

# 26. Neural Audio Codec対応

将来的なspeech-to-speechや音楽生成では、

```text
PCM
 ↓
Neural Codec
 ↓
Audio tokens
 ↓
Transformer
```

が重要になる。

したがってIRには、

```text
AudioToken
CodecState
StreamingLatent
```

を追加可能にする。

---

# 27. モダリティ統合

最終的には、

```text
Text
Image
Video
Audio
Sensor
```

を一つのrequestに含められるようにする。

例：

```json
{
  "input": [
    {"type": "text"},
    {"type": "video"},
    {"type": "audio"}
  ]
}
```

内部では、

```text
Text Stream
Video Stream
Audio Stream
     │
     ▼
Temporal Alignment Layer
     │
     ▼
Multimodal Runtime
```

とする。

---

# 28. 時間軸を第一級データとして扱う

映像・音声・センサーデータに共通する概念なので、

```text
timestamp
duration
sequence
sampling_rate
clock_domain
```

をIR自身に持たせる。

これによって、

```text
0.0s 映像
0.0s 音声
0.2s speech token
0.4s gesture
```

のような同期をruntimeが理解できる。

---

# 29. Agent Runtime

将来的にCoding Agent、音声Agent、視覚Agentを同時実行可能とする。

従来：

```text
Request
 ↓
LLM
 ↓
Response
```

ではなく、

```text
Agent
 ↓
LLM
 ↓
Tool
 ↓
Wait
 ↓
Vision
 ↓
LLM
 ↓
Audio output
```

を一つのtask graphとして扱う。

---

# 30. Agent Scheduler

Agent Aがtool実行待ちなら、

```text
Agent A → WAIT

Agent B → decode
Agent C → vision encode
Agent D → audio realtime
```

へ切り替える。

これによりGPU idle時間を減らす。

---

# 31. Thermal / Power Scheduler

MacBook Airのようなfanless Macでは非常に重要。

単純な最大性能だけを追求しない。

モード：

```text
Maximum Performance
Balanced
Silent
Battery
Sustained
```

`Sustained`では長時間のthermal throttlingを避けるように、

```text
GPU utilization
batch size
prefill chunk
background jobs
```

を調整する。

---

# 32. MLXとの関係

MLXを重要backendとして利用する。

MLXはlazy evaluationによって計算graphを構築し、実際の評価を後から行う設計になっている。

したがって、

```text
Apple Runtime IR
       ↓
Graph optimizer
       ↓
MLX graph
```

という経路を作る。

ただしMLXだけに依存しない。

---

# 33. Native Metal Backend

MLXまたはvLLM-Metalで埋められない計測済みのbottleneckに限り、専用Metal kernelを書く。
既存backendと同等のcorrectness、memory ceiling、fallbackを満たすことを導入条件とする。

候補：

```text
Paged Attention拡張（既存backendで解消できない場合のみ）
MLA
Quantized GEMV
Quantized GEMM
RMSNorm
RoPE
MoE routing
Expert GEMM
Vision patch projection
FFT後処理
Video tensor conversion
```

---

# 34. ANE / Neural Engine

初期版では必須にしない。

理由は、vLLM互換runtimeとして自由なdynamic graphを維持する方が優先だからである。

ただしbackend interfaceを、

```text
Backend
 ├─ CPU
 ├─ MLX GPU
 ├─ Metal
 └─ Neural Engine
```

としておき、将来的に対応可能にする。

ANEに向く処理だけを選択する。

候補：

```text
small encoder
vision preprocessing
audio encoder
embedding model
classifier
background model
```

など。

---

# 35. SSD Tier

Apple SiliconではSSDも実用的なmemory hierarchyとして扱う。

対象：

```text
Cold KV
Prefix cache
Vision embeddings
Video embeddings
MoE experts
Compiled kernel cache
Model shards
```

ただし頻繁なSSD書き込みは寿命とlatencyを考慮する。

---

# 36. Multi-Mac

vLLM-Metalでは現在Ray executorとpipeline parallelismによるmulti-Mac実行の基礎があり、Thunderbolt経由の2台Mac実行も検証されている。ただしまだ新しい領域である。

本プロジェクトでは、

```text
Mac A
CPU/GPU/Memory
    │
Thunderbolt / Ethernet
    │
Mac B
CPU/GPU/Memory
```

を一つのlogical compute fabricとして扱う方向を目指す。

最初の実装境界では、最大64 node、2,016 measured link、1,024 execution stage、4,096 distributed
state shardに制限する。nodeはmemory capacity、対応modality、healthを持ち、linkはThunderbolt／Ethernet、
実測bandwidth／latency、MTU、measurement ID、peer authenticationを持つ。未認証または未計測のlinkを
fallbackとして利用しない。

pipeline／modality stageはresident bytes、output bytes、required modality、依存DAG、checkpoint可能性を
宣言する。plannerはtopological orderで、memoryとmodalityを満たし、依存出力の実測transfer costが最小となる
healthy nodeへ決定論的に配置する。cross-node transferはsource/destination stage、node、transport、bytes、
estimated time、measurement IDをplan identityへ固定する。

distributed KV/stateはraw内容ではなくshard ID、SHA-256、size、replica nodeだけをplannerへ渡す。故障後に
healthy replicaが0なら処理を停止する。以前のplanで故障nodeがnon-checkpointable stageを所有していた場合も
再計画せず停止し、checkpointable stageだけを残存topologyへ再配置する。これらはtransportとexecutionの
安全契約であり、Thunderbolt／Ethernet上の実byte transfer認定とは別gateにする。

共通stream frameは最大64 MiB payload、4 KiB JSON headerとし、magic、ABI version、単調sequence、
source/destination stage・node、transport、planned bytes、measurement ID、payload SHA-256を保持する。senderと
receiverはplanの`FabricTransfer`と全fieldを照合し、short read、sequence replay、digest mismatch、別linkの
measurement replay、endpoint不一致を拒否する。framing層はpeer authenticationを代替せず、mTLSまたは
Thunderbolt peer identityを上位transportが検証済みの場合だけ構築可能とする。

Ethernet clientは`CERT_REQUIRED`、hostname verification、有効なhost／port、最大300秒timeoutを接続前に
検証する。TLS handshake後に得たDER peer certificateのSHA-256をplan外の明示pinと照合し、一致したsocketだけを
authenticated fabric streamへ渡す。certificate欠落／差し替え時はsocketを即時closeし、frameを送信しない。
server accept側も`CERT_REQUIRED`とclient certificate SHA-256 pinを必須にし、匿名clientやCA署名済みでも
想定nodeと異なる証明書をfabricへ昇格させない。
senderは最終frameの`sendall`後にwrite側をhalf-closeし、receiverの所定frame完了を待ってからsocket全体を
closeする。即時closeでTLS終端時に最終frameが失われることを実loopback試験で再現しているため、この終了順序を
transport lifecycle contractとする。

execution coordinatorはplanのstage、placement、reserved bytes、cross-node edgeが完全一致する場合だけ開始する。
依存が満たされたstageを1 waveとして最大64件並列実行し、modality間の独立性を利用する。pipeline stageは全依存
outputが揃うまで開始しない。cross-node payloadはtransfer adapter往復後にsizeとSHA-256が不変であること、各
node outputはplanのbyte数と一致することを必須とする。result保持は全stage合計256 MiB以下、deadline／cancelは
wave前、transfer前、node実行前後、publish前に検査し、失敗waveのpartial resultをreportへ公開しない。reportは
payloadを保持せずstage/node、size、digest、latency、peak result bytesだけを記録する。

---

# 37. Distributed Modality

単純なmodel parallelだけでなく、

```text
Mac A → Vision
Mac B → LLM
Mac C → Audio
```

というモダリティ分散も可能とする。

動画生成なら、

```text
Mac A → frames 0-31
Mac B → frames 32-63
```

ではなく、temporal dependencyを考慮したpartitioningを行う。

---

# 38. API

OpenAI互換APIを維持する。

さらに独自APIとして、

```text
/v1/runtime
/v1/hardware
/v1/profiles
/v1/video
/v1/audio
/v1/agents
```

等を追加する。

versioned control APIとruntime eventはJSON Schemaをrepositoryへcommitし、実際のHTTP/SSE responseを
integration testで検証する。schema validatorが未対応のkeywordを黙って無視することは禁止し、
schema追加時はvalidatorまたは標準validatorへの明示的な対応を必須とする。

---

# 39. Native Streaming API

映像・音声ではHTTP request/responseだけでは不十分なので、

```text
WebSocket
Unix Domain Socket
Shared Memory
Native IPC
```

を対応候補にする。

---

# 40. ディレクトリ構成

```text
vllm-apple/
│
├── pyproject.toml
├── README.md
├── LICENSE
│
├── vllm_apple/
│   │
│   ├── runtime/
│   │   ├── planner/
│   │   ├── scheduler/
│   │   ├── profiler/
│   │   ├── graph/
│   │   └── ir/
│   │
│   ├── memory/
│   │   ├── unified/
│   │   ├── kv/
│   │   ├── cache/
│   │   └── ssd/
│   │
│   ├── backends/
│   │   ├── mlx/
│   │   ├── metal/
│   │   ├── cpu/
│   │   └── ane/
│   │
│   ├── modalities/
│   │   ├── text/
│   │   ├── vision/
│   │   ├── video/
│   │   └── audio/
│   │
│   ├── moe/
│   │
│   ├── speculative/
│   │
│   ├── agent/
│   │
│   ├── distributed/
│   │
│   └── api/
│
├── sdk/
│   └── swift/
│       ├── Sources/VLLMAppleKit/
│       └── Tests/VLLMAppleKitTests/
│
├── schemas/
│   ├── api/
│   └── events/
│
├── native/
│   ├── metal/
│   ├── cpp/
│   └── rust/
│
├── benchmarks/
│   ├── text/
│   ├── vision/
│   ├── video/
│   ├── audio/
│   └── multimodal/
│
└── tests/
```

---

# 41. 言語

初期実装：

```text
Python
C++
Metal Shading Language
Swift（Macアプリ統合SDK）
```

を中心にする。

管理系、daemon、低レベルI/O、将来的なdesktop integrationにはRustも有力。

ただしvLLM compatibility layerはPythonを維持する。

---

# 42. Quantization

量子化形式と実際の計算形式を分離する。

```text
Storage format
      ↓
Runtime representation
      ↓
Compute representation
```

例：

```text
Disk      MXFP4
Memory    packed 4bit
Compute   fp16/vectorized
```

モデルファイル形式にkernel設計を縛らせない。

初期はMLXが提供するQ8/Q4表現とkernelを再利用する。profileで有効性を確認した後、

```text
dequantize → GEMM/GEMV → bias → activation
```

の中間materializationを減らすfused kernelを追加する。保存容量の削減とtokens/secの改善は
別metricとして測定する。

---

## 42.1 Portable Numeric Format Layer

2026-09-11追加要件。NVFP4 → INT8などの変換を個々のmodel loaderへ埋め込まず、Apple Runtime IRと
AppleExecutionPlannerが共有する数値形式互換層として設計する。以下の全体像には計画中の契約・kernelも含む。
2026-09-21時点では、bounded CPU参照変換、単一／複数scale軸geometry、low/high-first nibble順、
precision契約、MLX correctness bridge、
in-process resident storeとMLX F16/BF16/F32常駐backendまでを小規模tensorで実装・検証済みである。
private bounded artifactを使うlocal socket搬送、one-shot claim/consume/quarantine、producer/client/worker CLI、
最大2 bufferのin-process tile streaming、private packed/scale fileから全sourceを展開しないincremental decode、
manifest-lastの複数file artifact lifecycle/runtime搬送、MLX常駐と明示解放まで実装済みである。
request ID指定の別socket明示cancel、起動時orphan companion quarantineまで実装済みである。
native INT8演算、exporter固有scale swizzle、fused変換、他形式は未実装である。

### 抽象化の境界

`NumericFormatDescriptor`はversion、encoding/variant、bit width、signedness、logical shape、physical strides、
packing/nibble順序・endianness・padding、block/group shapeとaxis、scale型・shape・階層、zero-point、codebook、
scale方向（scale/inverse scale）、swizzle/layout、特殊値・丸め・飽和規則を記録する。scale等はtensor参照とdigestで
結合し、descriptorへweight全量を格納しない。回転・permutation・outlier/residual等の補助変換もrecipeに含める。

artifact container reader、量子化recipe interpreter、数値変換adapter、演算backendを分離する。
SafetensorsやGGUFというcontainer名、GPTQ/AWQというrecipe名を単一dtypeとして扱わない。
registryは`inspect → reference_decode → convert → execute → qualified`の対応段階を公開し、各variant、
operator、shape、weight/activation/state用途ごとの可否を示す。未知metadataを既知形式と推測しない。

`ConversionPlan`はsource/runtime/compute/accumulator/output descriptor、変換route、scale処理、
layout処理、tile/chunk、scratch上限、cache方針、reference、誤差budget、fallback、decision reasonをversion付きで
固定する。演算側は`ConversionAdapter`のeligibility・memory estimate・convert tile・synchronize・release契約を使う。
既存のStateMemorySpec、kernel probe、operator dispatcher、plan identityへ接続し、別のdevice選択器を作らない。

現在の小規模artifact搬送では、元packed payloadとscaleをcurrent-user private directoryの新規0600 fileへ
digest-boundで保存する。runtimeは安全な名前だけを受け、no-follow、owner、mode、size、inodeを検査した後、
load開始前にinboxからquarantineへclaimする。backend常駐が成功した場合だけ同じinodeをconsumeし、digest不一致、
内容不正、backend失敗、consume失敗は再実行用inboxへ戻さずquarantineに残す。protocolの再送cacheは同一sequenceの
重複loadを再実行しない。この仕組みはbounded correctness搬送であり、大規模weight streamingの代用ではない。

### NVFP4を最初の具体例にする

NVFP4ではE2M1値にblock scaleとglobal scaleを適用する。1Dの16要素blockと2Dの16×16 block、
scaleの配置・padding・swizzle等はexporterのvariantに依存するため、明示metadataから復元する。
参考：[NVIDIA Transformer Engine NVFP4仕様](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.15/user-guide/features/low_precision_training/nvfp4/nvfp4.html)。

現在のartifact adapterは、従来の単一axis plan IDを維持しながら`scale_axis`をuniqueな複数axisへ拡張し、
C-orderの各axisを16要素blockへ写像する。これにより1D blockと16×16の2D blockを同じ境界契約で扱う。
low-nibble-firstとhigh-nibble-firstは別adapter IDとして選択し、奇数要素時の未使用nibbleも順序別に検証する。
geometryとnibble順はplan/source/target digestへ結合され、inline artifactとfile-backed incremental decodeの双方で
roundtrip検証される。任意stride、scale padding、非C-order scale配置、swizzleはまだ推測せずunsupportedとする。

NVFP4 → INT8は単純castとして定義しない。まずpacked値とscaleを参照decodeし、次を候補として比較する。

- E2M1の有限値を2倍した整数をINT8 payloadへ写し、block/global scaleへ1/2を反映する表現保存経路。
  payloadの対応が正確でも、scale演算・累積・丸め・符号付きzeroまで同一とは限らない。block別scaleを扱える
  consumerが必要であり、通常のper-channel INT8 kernelへそのまま渡せるとはみなさない。
- 元の復元値をtargetのgroup/axisへ再量子化するINT8経路。scale粒度変更、clipping、丸めによる追加誤差を測定する。
- FP16/BF16のtile展開、既存MLX量子化kernel向けrepack、packed sourceを直接読むfused decode + 演算経路。

NVFP4由来の元々の量子化誤差と、変換で加わる誤差を別々に扱う。FP32参照値の保持は小規模fixtureに限定し、
本番modelは全量FP32/FP16展開を必須にしない。INT8 payloadは4-bit payload比で概ね2倍になるので、
scale・padding・scratch・cacheを含む実容量でadmission判定する。

### 対応範囲と高速化

FP4/FP6/FP8系、整数低bit系、浮動小数系、codebook系、block/group scaling、mixed precisionをdescriptorで
表現可能にする。weights、activations、KV/recurrent state、MoE expert、multimodal encoder、diffusionを対象にし、
動的activation scale更新やstate変換のタイミングも契約化する。training専用機構は推論artifactの解釈に必要な範囲を
優先し、training全体の互換を暗黙に保証しない。

変換はload時、初回利用時、tile/chunk逐次、演算内融合から選ぶ。prefill/decodeやexpert再利用率に応じて、
read + unpack + scale + requantize + layout + synchronization + computeの合計costを測定する。
CPU vectorization、MLX、Metal computeを順次実装し、成立するoperatorでは中間tensorの書戻しを省く。
double bufferとprefetchはbuffer所有権・completion barrierで管理し、使用中arrayを解放／書換えしない。
cancel、kernel failure、memory/thermal pressure時は新規prefetchを抑え、既存safe pointでrouteを切り替える。

CPU vectorized candidateは`numeric_vectorized.py`へscalar referenceと分離して置く。NumPyが無い環境では
明示的にunavailableとし、暗黙fallbackや必須依存追加を行わない。初期範囲は2/4/8-bit unpack／repack、
FP8 decode、NF4 encode、symmetric INT8 requantize、byte／nibbleの全対応layout gatherで、既存65,536 element上限、padding検査、finite検査、
ties-to-even、scale serialization、output digestをscalar経路と完全一致させる。候補は実hardwareで
最低3 sample・5%以上の改善を満たすまで標準routeへ昇格しない。

MLX candidateも同じpacked decode／repack、FP8、NF4、layout gather契約で実装するが、hostからMLX arrayへの
転送と結果readbackを含むend-to-end時間でCPU vectorized routeと比較する。M4／MLX 0.27.1の65,536値NF4では
出力digestは一致したものの、7 sample medianがMLX 12.457 ms、NumPy 9.645 msで5%改善を満たさないため、
MLX routeは非昇格としCPU vectorized routeを維持する。device実装が存在することを高速性の証拠にしない。

Native Metal candidateは全65,536 U4 codeのexact unpack検査と、256×256 NF4 weight／ones vectorの
CPU参照一致をkernel実行後に必須化する。同一command buffer内のNF4 decode→float GEMV二段経路と、
packed weightを直接読むfused GEMVを同一work itemで比較する。M4の7 sampleではtwo-pass median
232.459 µsに対しfused 306.916 µs（0.76倍）であり、中間書戻し削減だけでは5% gateを満たさないため
fused routeを非昇格とする。

現在の`NumericStreamingPlan`はsource SHA-256、総byte数、最大8 MiBのtile、1または2 buffer、alignmentを
canonical IDへ結合する。`NumericDoubleBufferStream`は各slotをgeneration付きleaseとして払い出し、未解放slotの
上書きと3個目のin-flight tileを拒否する。release/cancel/closeでは64 KiB以下の固定zero blockでslotを消去し、
cleanup自身がtile大の追加allocationを作らない。read-only viewはlease期間内だけ有効で、release後はzero化される。
memory admissionはtile buffer、metadata、変換scratch、destinationを同時予約し、成功後だけdestinationへ縮小する。
backend allocation evidenceはstream plan IDへ結合し、通常loadへのstreaming metadata混入も拒否する。
runtime transportは各tile safe pointでsocketをnon-consuming peekし、peer切断時だけ同じcancellation signalを
streamへ伝える。次commandが既に届いていても内容を消費せずcancelとは扱わない。
現段階のbridgeはcaller所有の全`ScaledInt8Tensor`とMLX用F32 sourceをまだ保持するため、全modelのpeak memory削減や
真のI/O overlapを主張しない。次段階でartifact readerからtileを直接供給し、socket cancellationをsafe pointへ接続する。

変換cacheは元artifactとscaleのdigest、descriptor、target/layout、kernel version、chip/OS/backendに結合する。
再量子化済みtensorをさらに繰り返し量子化せず、常にimmutable sourceから生成する。重複変換を共有し、
cache・scratch・prefetch枠はUnified Memoryの共通予算で制限する。長期保存はoptimizer artifactのprovenanceを継承する。

native対応は固定表だけに依存しない。格納可能型、load/store、変換命令、算術、dot/matmul、accumulator型、
operator/shape性能を個別probeする。Appleの公開APIに型が存在しても全chipで高速native演算できるとは判断しない。
参考：[Apple MTLTensorDataType](https://developer.apple.com/documentation/metal/mtltensordatatype)。
Core ML/ANEは公開APIで受理されるgraphと精度のみ対象とし、後半の異種schedulerへ変換・同期costを渡す。

### 完了条件

全bit pattern・zero・極値・scale異常・端数block・転置・swizzleをCPU参照で検証し、operatorの最大誤差／相対誤差／
RMSE、モデルのperplexityまたはtask品質を比較する。exact repackとlossy requantizationは別routeとして記録する。
cold load、warm reuse、prefill、decodeのTTFT/TPOT/throughput、変換時間、peak memory、帯域、energyを測定し、
native/既存MLX/floating fallbackとのend-to-end比較で採用を決める。高速化を未計測で保証しない。
最初の実装順はdescriptorとCPU参照 → NVFP4小規模変換 → 既存backend consumer → streaming/fused → 他形式拡張。

実装済み`NumericEligibilityMatrix`はsource／compute format、tensor role（weight、activation、KV、recurrent、
expert、vision、audio、diffusion）、backend、operator、要素数範囲、recipe ID、evidence ID、qualified状態を
canonical capability IDへ結合する。形式名の一致だけではeligibleにせず、完全一致する認定済みcapabilityが一件だけ
存在する場合に限り実行を許可する。未知recipe、未認定evidence、重複して曖昧なcapabilityはfail-closedである。

`NumericRouteProfile`はload convert、first-use convert、cached convert、fused every-useについて、conversion、
synchronization、compute latency、amortized uses、peak Unified Memory、output digestを記録する。phaseとcapability IDが
一致し、memory ceiling内かつ全候補のoutput digestが一致する場合だけ、amortized latencyが最小のrouteを決定論的に選ぶ。
これにより変換単体の速さではなく、read／convert／barrier／反復computeを含む総費用でrouteを選択する。

反復利用する変換結果は`NumericConversionCacheIdentity`へsource、scale、layout、kernel、environmentの各SHA-256を
結合する。cacheはprivate directory／file、HMAC署名、manifest-last atomic publish、load時の出力再hashを必須とする。
同一identityの並行requestはsingle-flightで一変換だけを実行し、待機requestはpublish済み結果を再検証して共有する。
64 entry／64 GiBのhard ceilingをaccess-time LRUで維持し、tamper、inventory failure、明示revoke、LRU evictionは
理由付きquarantineへ移す。converter失敗時はtemporary entryを削除し、active entryを公開しない。

数値routeの最終昇格は`evaluate_numeric_promotion`で行う。scalar maximum absolute error、RMSE、operator出力一致、
モデル品質scoreの退行上限を順に検査し、その後に同一hardware／model／phase／workloadのend-to-end profileを比較する。
双方3 sample以上、output digest一致、peak Unified Memory非悪化、計測済みenergy非悪化、median latencyの既定5%以上改善を
全て満たした場合だけpromotedとする。どの層で不合格になったかは固定reason codeで返す。

実行環境のcapabilityは`NumericProbeIdentity`へchip、OS build、toolchain、backend/version、operator、shape、
source／compute format、tensor role、recipe IDを結合する。`NumericCapabilityProbeRegistry`は最大16 MiBの固定probe入力で
CPU referenceとcandidateの出力digestを比較し、一致したidentityだけを実行可能にする。不一致またはcandidate例外は
identity単位でquarantineし、未probe／隔離済みidentityはCPU referenceへ理由付きfallbackする。recipe inventoryにない
形式はreference fallbackで対応済みに見せず、明示的なunsupportedとして拒否する。

CLIの`numeric-route-diagnostic`はsource／runtime／compute format、tensor role、選択route、maximum absolute error／RMSE
budget、fallback reasonをschema version付きJSONで返す。利用者向けmessageは英語、日本語、简体中文を同じmessage keyで
提供し、tensor値やpromptは含めない。非有限または負のbudget、制御文字を含むfallback reasonはfail-closedで拒否する。

`NumericTilePipeline`は最大65,536 tile、16 MiB/tile、1または2 source bufferに制限し、二buffer時は次tileのreadを
専用threadでprefetchしながら現在tileをconvert／consumeする。future completionを明示barrierとして順序を維持し、
tile indexと変換結果を連結したdigest、source／converted byte、実buffer peak、barrier数をreportする。
`NumericBandwidthLedger`はUnified Memory帯域予約をthread-safeに合算し、ceiling超過を処理開始前に拒否する。
変換例外、reader例外、cooperative cancelの全経路でprefetch executorとbandwidth leaseを回収する。

exporter固有packingは`NumericExporterLayoutAdapter`でvalue layoutとscale layoutを別々に宣言する。
対応layoutはcontiguous row-major、column-major、row padding、tiled row-major、square power-of-two Morton swizzleで、
low/high-first nibble順、padding code、shape、stride、tile shapeをadapter digestへ結合する。repack時はlogical要素外と
末尾未使用nibbleを含む全padding codeを検査し、logical contiguous順へ変換する。registryはexporter IDとrecipe IDの
完全一致だけを解決し、未知swizzleや曖昧なmetadataを推測しない。

NVFP4のINT8経路は`compare_nvfp4_int8_routes`で二種類を別routeとして比較する。表現保存経路はE2M1 codeを
scale付きsigned INT8へ正確に写し、元のblock／global scaleを保持するため追加誤差をゼロとして記録する。
一般経路はFP32参照値からtensorまたはgroup単位のsymmetric INT8へ再量子化し、maximum absolute error、RMSE、
payload＋scale storageを算出する。`Int8ExecutionCapability`はsignedness、INT32／FP32 accumulator、
nvfp4-block／group／tensor scale粒度を宣言し、演算kernelがscale ABIを持たないrouteをcompatibleにしない。

CPU参照codecはFP8 E4M3FNとE5M2をfinite／subnormal境界まで明示decodeし、NaN／Inf payloadとfinite range超過を
拒否する。encoderは全finite codebookから決定論的nearest値を選び、signed zeroを保持する。NF4は固定16値codebookを
low-nibble-firstでpackし、末尾paddingを検査する。groupwise affineはsigned／unsigned INT8・INT4・INT2について
groupごとのscaleとstored-domain zero-pointを保持する。これらはCPU reference能力であり、accelerator実行能力は
environment-bound capability probeに合格するまで付与しない。

OCP MX v1.0 adapterはMXFP4 E2M1、MXFP6 E2M3／E3M2、MXFP8 E4M3／E5M2を別formatとして扱う。
各blockは32要素とE8M0共有scaleを持ち、reference artifact packingは4／6／8-bit LSB-first contiguousとして
明示する。E8M0 `0xff`、E4M3 NaN、E5M2 Inf／NaN、nonzero padding、F32 range超過を実行不能として拒否し、
全codeのnormal／subnormal／signed zeroをCPU参照decodeする。decode後は共通groupwise symmetric INT8へ
requantizeできるが、Apple backendのnative能力や高速化は別probeなしに付与しない。vendor固有scale、saturation、
packing variantはOCP標準へ推測変換せず、独立adapter identityと全bit-pattern testを要求する。

scaleのdouble quantizationは有限scale列のmin/maxからUINT8 code、offset、stepを生成する。mixed precisionは
group最大絶対値と明示thresholdから2／4／8-bitをgroup単位で選び、各groupの開始位置とaffine tensorを保持する。
outlier residual表現はthresholdへclipしたbounded affine baseに対し、threshold超過indexだけを昇順・重複なしで保持し、
元値との差をfinite residualとして加算する。全経路を65,536要素以下のCPU referenceに限定し、accelerator capabilityを
暗黙に付与しない。

artifact metadataは`NumericArtifactDescriptor`でcontainer、quantization recipe、storage format、compute format、
bits、group size、packing、exporterを分離する。strict adapterはSafetensors上のGPTQ／AWQ、MLX affine、Core ML linear、
GGUF Q4_0／Q4_K／Q8_0、および非量子FP16／BF16／FP32をcanonical descriptor IDへ正規化する。
containerとrecipeの不正な組合せ、余分／欠損field、未知GGUF type、INT系compute dtypeはfail-closedで拒否し、
container名だけで実行能力を推測しない。

optimizer出力は`FastLoadArtifact`として4–64 KiBの選択page sizeへ各tensor先頭をalignする。source fileの
no-follow／regular file／範囲／SHA-256をcopy中に検証し、private temporary directoryへdataをfsyncした後、
manifest-lastでatomic publishする。loaderはprivate ownership、manifest上限、artifact ID、tensor digest、alignment、
重複name、offset overlap、最大65,536 tensor／32 GiBを再検証する。`KernelCompatibilityIndex`はbackend/version、
environment digest、operator、numeric format、layout、minimum alignment、maximum tensor bytesが一意一致するkernelだけを返す。

MoE expert residencyは`ExpertResidencyManager`が`(layer, expert)` keyで管理する。backend固有のMetal／MLX handleは
opaque `ExpertResource`としてload／release ownershipを明確にし、entry数とresident byteの二重上限内で決定論的LRUを
行う。active lease中のexpertはevictせず、全候補がpinnedなら新規load結果を解放してrequestを拒否する。
pressure resizeを即時適用できない場合はtarget boundsをpending化し、lease解放時に再適用する。診断にはhit、miss、
eviction、rejection、active lease、pending boundsだけを保持し、weight内容は保存しない。

expert選択は`ExpertSelectionTelemetry`の最大65,536 sampleのringへ、layer、重複のない最大64 expert、有限非負の
routing weight、選択latency、実際にresidency cacheへhitしたexpertだけを記録する。snapshotはlayer/expert別の
選択回数とcache hit、総選択数、median／p95／maximum latency、上限超過の破棄数を返す。token、activation、
routing入力、任意labelは保存せず、predictorの学習入力へ暗黙転用しない。

`CorrectnessNeutralExpertPredictor`は明示的に渡された実選択列だけからfirst-order遷移頻度を学習し、最大
65,536 context／historyの範囲で次expertのprefetch hintを頻度順・key順に返す。未知contextは空候補とし、
context上限超過はFIFOで破棄する。hintはresidency準備だけに使用し、`resolve`はrouterの実選択を変更・追加・
削除せずそのまま返すため、予測誤りは速度にだけ影響し数値結果を変えない。

非active expertのSSD tierは`ExpertSSDStore`で最大65,536 entry／4 TiB、単一expert最大1 TiBをhard ceilingとする。
rootはowner-only 0700、各expert fileは0600とし、temporary fileへのwrite・fsync後にatomic replaceしdirectoryも
fsyncする。readは`O_NOFOLLOW`で開き、owner、regular file、mode、inode、size、mtime、SHA-256を前後で再検証する。
entry数とbyte数の二重上限はLRUで回収し、tamper検出時はindexとfileをevictする。snapshotは容量、hit／miss、
eviction、integrity failureだけを返し、weight bytesやmodel入力は保持しない。

layer単位offloadは`LayerPrefetchCoordinator`が最大4,096 layerを順序付きで処理する。専用single-thread executorへ
次layerのloadを一件だけ先行投入し、現在layerのconsumeとoverlapすることでsource resourceを二slot以内に保つ。
prefetch例外時は同じlayerを同期loadするon-demand fallbackを一度だけ行い、失敗を別layerへ波及させない。
成功、consume例外、cooperative cancelの全経路で現在resourceと完了済みorphan futureをbackendへreleaseする。

`HierarchicalStateCache`はKV、prefix、vision／video／audio embeddingを同じfingerprint-only keyでhot memoryと
private cold fileの二階層へ格納する。各tierはentry数／byte数の独立LRU上限を持ち、hot evictionは容量内ならcoldへspillし、
cold hitはhot容量内ならpromotionする。cold writeはprivate temporary file、fsync、atomic replaceを用い、read時はowner、
mode、regular file、no-follow、size、SHA-256を再検証する。tamper entryはactive indexから削除する。
key、filename、telemetryにはraw promptやembedding内容を保存せず、kindとSHA-256 fingerprintだけを使用する。

大容量Unified Memoryは`UnifiedMemoryArena`のanonymous mmapを最大1 TiBのlazy virtual capacityとして確保し、
1–64 KiB alignmentのfirst-fit suballocationを最大65,536件まで提供する。各leaseはcopyを伴わないmemoryviewを返し、
release時にviewを無効化して範囲をzeroizeし、隣接free rangeをcoalesceする。snapshotはallocated／free／largest range、
fragmentation、peak、failure数を返す。free pageはplatformが対応する場合だけ`madvise`で回収し、active lease中のarena closeを拒否する。

video生成は`AttentionStateRegion`へframe start/count、temporal overlap context、spatial tile bounds、state bytesを固定する。
`GenerativeChunkPlan`は最大65,536 task、最大64 concurrencyでtemporal chunkとspatial tileの直積を生成し、端frame／端tileを
切り詰める。同一spatial tileの次temporal chunkは直前taskへ明示依存し、attention stateの順序を維持する。
単一stateがmemory ceilingを超えるplanは拒否し、複数taskの推定peakが超える場合はconcurrencyを安全側へclampする。

diffusion／video latentは`LatentMemoryManager`が`UnifiedMemoryArena`上で管理する。1〜5次元、1/2/4 byte element、
単一256 GiB以下のshapeだけを受け付け、最大4,096 bufferのidle LRUとactive lease pinを行う。同一shapeのidle
bufferは内容を全zeroizeしてから再利用し、arena容量不足時はidle bufferだけを順次解放する。全bufferがactiveなら
新規割当を拒否し、既存生成を破壊しない。trimはidle leaseを解放してfree page回収を試み、active lease中のcloseは
fail-closedとする。診断はbuffer数、active数、resident bytes、hit／miss／eviction／rejectionだけを返す。

platform fault qualificationは`PlatformFaultScenarioMatrix`でprofile persistence、scheduler admission、worker crash、
client disconnectの全pointを必須coverする。既定scenarioはatomic replace前中断からlast-known-good復元、queue後かつ
reservation前のcapacity拒否、reservation後worker crashから再起動、first stream chunk後disconnectからcancelを注入する。
各handlerはservice ready、active reservationゼロ、temporary fileゼロ、stored prompt/outputゼロを共通に証明し、
profile scenarioはlast-known-good復元も証明する。例外はmessageを保存せず型名だけのbounded failure reasonへ変換する。

---

# 43. モデルフォーマット

初期：

```text
MLX model
Hugging Face
Safetensors
```

を優先。

将来的に、

```text
GGUF
Core ML
ONNX
```

へのbridgeも検討する。

---

# 44. Observability

runtime内部を完全に可視化できるようにする。

表示：

```text
tokens/sec
TTFT
GPU utilization
CPU utilization
memory bandwidth
Unified Memory usage
KV usage
cache hit ratio
MoE expert activity
SSD cache
thermal pressure
power usage
```

映像：

```text
fps
decode latency
vision encoder latency
frame cache hit
```

音声：

```text
audio latency
buffer fill
RT deadline miss
```

実装済みの共通`WorkloadTelemetry`は、CPU/GPU utilization、Unified Memory bandwidth、power、
thermal stateを最大1024件のthread-safe ring bufferへ保持する。GPU、bandwidth、powerをOSから取得できない
場合は0へ置換せず`null`として欠測を明示し、median、p95、maximum、thermal状態別件数、上限超過による
破棄件数をschema version付きsnapshotで返す。Vision、Audio、Videoについてはmodality、bounded operation名、
latency、入力／出力unit数、peak memory、成功可否だけを同じ上限内へ記録し、modality別sample数、failure数、
latency分布、累積unit、最大peak memoryを集計する。prompt、画像、音声、frame、embedding、生成物pathは
このtelemetryへ保存しない。

---

# 45. Benchmark設計

単純なtokens/secだけを評価しない。

native hardware microbenchmarkは単一Swift／Metal processで全pipelineをcompileし、各operatorを一度warm-upした後に
7 sampleを取得する。対象はFP32 64×64×64 GEMM、FP32 256×256 GEMV、32 MiB shared-buffer copy、empty kernel、
1 query×128 KV×64 dimension attention、INT8 64×64×64＋INT32 accumulationである。各sampleはcommand encodeから
completionまでのend-to-end時間を測り、compile時間を除外する。copy bandwidthはread＋writeの64 MiB traffic、
演算throughputは明示work-item数から算出する。responseはoperator集合、sample数、正の時間、work／byte数をstrict検証し、
outputの非内容digestを持つprivate atomic reportだけを証跡化する。Apple M4実測値は
`native-hardware-microbenchmarks-m4-2026-09-21.json`を正本とする。

phase resource profileはCPU prefill GEMM／decode GEMV reportとnative Metal reportのfile全体SHA-256、CPU側hardware
fingerprintを結合する。prefillは同一64×64×64 FP32 GEMM、decodeは同一256×256 FP32 GEMVのmedianを比較し、
5%以上速い場合だけMetalを候補とする。両phaseへattention medianとshared Unified Memory bandwidthを付加し、
shapeやsample数が一致しないCPU evidence、欠損Metal operator、別hardwareを拒否する。M4 profileではprefill 35.27倍、
decode 8.23倍のMetal speedup、37.79 GB/sを記録し、両phaseをNative Metal候補とした。これは代表microbenchmarkの
配置候補であり、実model correctness gateなしにproduction routeへ昇格しない。

runtime autotunerはMac hardware fingerprintとphase profile IDへ結合し、batch、tile、KV block、prefill chunk、kernelの
最大256構成を比較する。各候補はprefill／decodeをそれぞれ3〜64 sample、peak memory、output digestとともに提出する。
baseline digest不一致またはmemory ceiling超過は順位付け前に除外し、残りをworkload指定のprefill／decode weight付き
median latency、peak memory、構成の辞書順で決定論的に選ぶ。全候補不合格時は既定構成を推測せずfail-closedとし、
winner、qualified／rejected数、hardware／profile identityからreport IDを生成する。

KV configuration searchはFP32／FP16／BF16／INT8、context 128〜16,777,216、batch 1〜256、KV block
8／16／32／64／128の最大512候補を扱う。各候補のpeak memoryは`bytes_per_token × context × batch`以上、
decode latencyは3〜64 sampleとし、maximum absolute error、RMSE、cosine similarity、baseline output digestを
quality gateとして検査する。memory ceilingとqualityを通過した候補だけをcapacity、latency、balanced objectiveに応じて
順位付けし、同点はlatency、capacity、memory、構成順で決定する。全候補不合格時はprecisionを暗黙に緩和しない。

LLM：

```text
TTFT
TPOT
tok/s
energy/token
memory/token
```

Vision：

```text
image latency
images/sec
memory/image
```

Video：

```text
frames/sec
seconds-of-video/sec
TTFT
memory/minute
```

Audio：

```text
real-time factor
latency
dropout
power/hour
```

Agent：

```text
task completion latency
tool-wait utilization
cache reuse
```

---

# 46. Reliability

専用kernelや自動最適化は必ずfallbackを持つ。

```text
Optimized Metal
      ↓ failure
MLX generic
      ↓ failure
CPU
```

correctnessを性能より優先する。

---

# 47. Security

サーバーはデフォルトで、

```text
127.0.0.1
```

のみlistenする。

外部公開は明示設定が必要。

TCP remote modeは`--allow-remote`だけでは有効にならない。非loopback addressへbindする場合は、TLS certificateと
current-user所有・owner-only・1 MiB以下のprivate key、およびBearer session tokenを同時に必須とする。TLS identityの
symlink、片側だけの指定、認証なしはmodel inspectionやbackend loadより前に拒否する。server contextはTLS 1.2以上に
制限し、既存のconstant-time Bearer検証をHTTPS requestにも共通適用する。loopbackとprivate UDSは既存既定値を維持し、
remote公開へ暗黙昇格しない。
`--tls-client-ca`を明示した場合は、owner検証済みCAをserver contextへ読み込み、TLS handshakeでclient certificateを
必須化する。mTLSはBearer tokenを置き換えず併用する。`--tls-client-policy`はclient CAとの併用を必須とし、
exact canonical subjectまたは型付きSAN（DNS／email／URI／IP Address）のallowlistに一致し、かつ大文字小文字を
正規化したserial denylistに該当しないidentityだけを許可する。policyはcurrent-user所有、group/world write禁止、
symlink禁止、64 KiB以下とし、`O_NOFOLLOW`で開いたdescriptorの前後`fstat`が一致した内容だけを採用する。
atomic replaceはrequest間に自動reloadされるためlistenerを停止せずrotationできる。変更後の破損・消失・unsafe fileは
last-known-goodへ戻さず全client certificateをfail-closedにする。公開snapshotはversionと各list件数だけを含む。
外部CRL／OCSP取得には依存せず、失効は運用者がatomic policyへ反映するserial denylistを権威情報とする。

モデルファイル、plugin、custom kernelにはhash検証を導入可能にする。

---

# 48. GUIとの分離

Runtime自体はheadless daemonとする。

```text
vllm-appled
```

を起動し、

```text
GUI
CLI
OpenCode
Codex-like agent
WebUI
```

が接続する。

GUIがクラッシュしても推論serverは維持できる。

---

## 48.1 Apple Platform Integration

Macアプリからruntimeを容易に利用できるように、runtime本体とアプリ固有UIの間へ安定した統合層を設ける。

標準構成は、SwiftアプリへPython runtimeを直接linkする方式ではなく、独立したheadless daemonとSwift SDKを組み合わせる方式とする。

```text
Swift / SwiftUI App
        │
        ▼
VLLMAppleKit（Swift Package）
        │
        ├─ Process lifecycle
        ├─ Typed request / response
        ├─ Async streaming
        ├─ Health / readiness
        └─ Error mapping
        │
        ▼
Unix Domain Socket / HTTP / WebSocket
        │
        ▼
vllm-appled
        │
        ▼
Apple Runtime
```

この構成により、UI processのクラッシュ、再起動、Swift concurrency、App Sandbox、runtime更新を推論engineから分離する。

### 48.1.1 統合モード

最低限、次のモードを提供する。

```text
Managed Local
  アプリが同梱daemonを起動・監視し、終了ポリシーを管理する。

Shared Local
  ユーザーまたは別アプリが起動したlocalhost daemonへ接続する。

Remote
  明示的に許可されたネットワーク上のruntimeへ接続する。
```

Phase 1では `Managed Local` と `Shared Local` を優先する。

### 48.1.2 Swift SDK

`VLLMAppleKit` をSwift Packageとして提供する。

公開APIは、可能な限りFoundationとSwift Concurrencyのみへ依存し、SwiftUIやAppKitには依存しない。

最低限、以下を提供する。

```swift
public protocol VLLMAppleRuntimeClient: Sendable {
    func hardware() async throws -> HardwareInfo
    func runtimeProfile() async throws -> RuntimeProfile
    func health() async throws -> HealthStatus
    func chat(_ request: ChatRequest) async throws -> ChatResponse
    func streamChat(_ request: ChatRequest) -> AsyncThrowingStream<ChatEvent, Error>
}
```

データ型は `Codable`、`Sendable` を基本とする。

callback専用APIではなく、`async/await` と `AsyncSequence` を第一選択とする。Objective-Cからの利用が必要になった場合は薄いadapterを追加し、core APIをObjective-C互換性へ縛らない。

Objective-C互換層は別target／dynamic library productの`VLLMAppleKitObjC`として提供し、Swift coreを変更しない。
`NSObject`派生のhealth／chat resultとcompletion-block型clientだけを公開する。modelは256文字、promptは32 Ki文字、
temperatureは0〜2、max tokensは1〜1,048,576へ制限し、不正入力はnetwork access前に拒否する。Swift errorは
任意detailを渡さずstable `message_key`付き`NSError`へ変換する。Objective-C adapterはnon-streaming health/chatを
対象とし、streamingとruntime eventはSwift Concurrency APIを正本とする。

### 48.1.3 通信方式

ローカル接続はUnix Domain Socketを第一候補とし、既存のOpenAI互換clientとの接続にはlocalhost HTTPを利用できるようにする。

```text
制御・短いrequest     Unix Domain Socket または HTTP
token streaming       WebSocket または streaming HTTP
将来の大容量media     Shared Memory / Native IPC
```

transportとrequest modelを分離し、同じSwift APIからtransportを切り替えられるようにする。

独自APIとeventにはversionを付ける。

```text
API version
Schema version
Runtime version
Minimum compatible client version
```

SDKとdaemonは起動時に互換性を確認し、非互換の場合は構造化errorを返す。

### 48.1.4 Daemon Lifecycle

Managed LocalモードではSDKまたは専用launcherが、次を管理する。

```text
実行ファイルの検出
署名・hashの検証
一意なsocketとportの割り当て
起動
readiness待機
異常終了の検出
診断logの収集
graceful shutdown
```

複数アプリや複数windowから同時に利用される可能性を考慮し、process ownership、client session、idle timeoutを明示的に管理する。

モデルロード中は単なるhealth successを返さず、最低限次の状態を区別する。

```text
STOPPED
STARTING
PROFILING
LOADING_MODEL
READY
DEGRADED
FAILED
STOPPING
```

### 48.1.5 配布とアプリバンドル

Macアプリへ組み込むartifactは、次を分離する。

```text
Swift SDK
Launcher
Runtime daemon
Native libraries / Metal libraries
Python environment
Model files
Hardware profile / kernel cache
```

モデルをアプリ本体へ必須同梱せず、初回取得、ユーザー選択、共有model directoryのいずれも選べるようにする。

runtime resourceの場所を固定絶対pathへ依存させず、app bundle、Application Support、開発用pathをresolver interfaceで切り替える。

配布物はApple Silicon向けに署名可能な構造とし、codesign、notarization、hardened runtimeを妨げる自己書き換えを避ける。生成されるprofile、log、kernel cacheは署名対象bundleの外へ保存する。

### 48.1.6 SandboxとSecurity

Mac App Sandboxを考慮し、SDKは必要となるentitlementと配置先を文書化する。

ローカル通信では、他processによるなりすましを防ぐため、socket permission、session token、client identityを検証可能にする。

Remoteモードは明示設定時のみ有効化し、認証とTLSを必須化できる設計とする。

SDKのerrorは機械判定可能なerror code、復旧可能性、ユーザー向けmessage keyを持たせる。ユーザー向け文言はアプリ側で英語、日本語、簡体字中国語へlocalizeできるよう、runtimeが固定文章だけを返す設計にしない。

### 48.1.7 App Integration Observability

アプリが独自の進捗UIや診断画面を構築できるよう、次のeventをstreamとして提供する。

```text
daemon state
hardware profiling progress
model download / validation progress
model loading progress
memory pressure warning
thermal state change
request queue state
token / media stream
recoverable and fatal error
```

daemonの強制終了でUnix Domain Socket entryが残った場合は、再起動時にownerとfile typeを検証し、
現在user所有のsocketだけを安全に置換する。session tokenはprivate fileから再利用し、通常終了では
socketを削除する。強制終了、再起動、認証復旧、通常終了後cleanupは実process integration testで
継続的に検証する。

event deliveryが遅いUIにruntime全体をblockさせないよう、buffering、coalescing、backpressure policyをevent種別ごとに定義する。

### 48.1.8 Integration Testing

Python側のAPI testに加えて、Swift SDKについて次を自動testする。

```text
Codable schema compatibility
daemon launch / readiness / shutdown
stream cancellation
daemon crash recovery
SDK・daemon version mismatch
socket permission
複数client接続
model loading中の状態遷移
```

sampleとして、SwiftUI製の最小Mac chat appを提供する。ただしsample appをruntime本体の依存先にはしない。

---

# 49. CLI

最終的には、

```bash
vllm-apple serve mlx-community/Qwen3.8-27B-mxfp4
```

だけで動かせるようにする。

常駐運用では、macOSのcurrent-user `LaunchAgent`を次の分離した操作で管理する。

```bash
vllm-apple daemon-install /path/to/model
vllm-apple daemon-start
vllm-apple daemon-status
vllm-apple daemon-stop
```

`daemon-install`はplistを0600で同一directoryへ一時生成し、fsync後にatomic replaceする。既存定義は
`--force`なしでは上書きしない。既定transportはowner-only Application Support配下のUDSとsession token fileを
必ず対にして使用する。start/stop/statusはplistのlabel、owner、regular-file、modeを再検証してから、shellを介さず
current-userの`gui/<uid>` domainへ`launchctl`を最大10秒で実行する。installとstartを分けることで、生成内容を確認せず
自動常駐させない。

起動例：

```text
Apple M4 detected
Memory: 32 GB
Model: Qwen 27B MXFP4

Profiling hardware...
Loading model...

Recommended configuration

Context      65536
KV cache      8.2 GB
Runtime       Metal + MLX
Mode          Interactive
Thermal       Balanced

Server running:
http://127.0.0.1:8000
```

native runtimeがprocess main thread ownershipを要求する場合は、control daemon内のowner threadではなく専用model
processを使用する。子processはfactory、models取得、全request実行、cleanupをmain threadだけで行い、stdin readerは
cancel flagとbounded queueの更新だけを担当する。IPCは改行区切りJSON、request 4 MiB、response 16 MiB、pending 64件を
hard limitとする。親はqueue上限で投入前に拒否し、request ID、有限deadline、client cancellationを子のsafe pointへ伝える。
任意例外本文は境界を越えず固定codeへ変換し、worker exitでは全pending waiterを解放する。停止時はshutdown、bounded wait、
terminate、killの順で回収する。streamingは途中出力後の安全なbackend fallback契約がないため、現段階では明示unsupportedとする。

---

# 50. 開発フェーズ

## Phase 1 — 基盤

状態：`[Done]`

vLLM-Metal互換pluginとして成立させる。

実装：

```text
[Done] hardware detection
[Done] memory detection
[Done] automatic context calculation
[Done] runtime profile
[Done] basic scheduler and hard memory admission limit
[Done] OpenAI API inference backend process connection and proxy
[Done] versioned local control API
[Done] daemon lifecycle and readiness foundation
[Done] vLLM-Metal environment doctor and managed process lifecycle
[Done] low-buffer SSE streaming proxy
[Done] Unix Domain Socket server and session authentication
[Done] bounded runtime event stream
[Done] Swift Unix Domain Socket transport
[Done] bounded Swift daemon logs and crash restart policy
[Done] Swift Package integration SDK foundation
[Done] Unix Domain Socket transport
[Done] runtime event streaming
[Done] minimal SwiftUI sample app
```

---

## Phase 2 — Apple Runtime Planner

状態：`[Done]`

実装：

```text
[Done] AppleExecutionPlan schema and deterministic dry-run
[Done] StateMemorySpec
[Done] AppleChipProfile and BackendEngine capability foundation
[Done] measured hardware/backend capability detection and profile persistence
[Done] bounded prefill/decode phase profile schema and aggregator
[Done] backend stream timing, usage, and RSS instrumentation
[Done] staged long-context evaluation schema and fail-fast memory coordinator
[Done] tokenizer-aligned retrieval dataset and live backend adapter
[Done] existing context/scheduler/elastic-memory atomic safe-point integration
[Done] CPU reference profiler and profile-bound accelerator benchmark contract
[Done] kernel self-test / benchmark / quarantine foundation
[Done] pressure・thermal・power・backend制約に基づくprefill/decode automatic batch sizingとscheduler admission
[Done] adaptive state allocation policy foundation。KV／recurrent／prefix／attention-window／expertを
bounded state recordとして扱い、明示的にpromotionされたprecisionだけをstate age順に選び、pressure別の
retain／reprecision／evict planと解放byte数を決定論的に生成する。pinned stateは変更しない
[Done] backend-owned stateのatomic reprecision／rollbackとscheduler safe-point適用。backendがstate recordを列挙し、
transaction begin／commit／rollbackを所有する。active reservation中はpending化し、最後のreservation完了時に
semantic cache resize、adaptive transaction、execution planの順で適用する。commit失敗はrollbackして旧stateを維持する
[Next] 実KV・recurrent state backendによる品質／memory promotion gate
[Done] event-driven memory pressure monitoring and safe-point propagation
```

---

## Parallel Track O — Model Optimization Compiler

状態：`[Next]` foundation、`[Later]` model変換と構造最適化

本trackは推論runtimeを変更する機能ではなく、open-weight modelからMacと用途に適した
新しいimmutable artifactを生成するcompanion systemとする。大量の一時memory、長時間処理、
失敗時の不完全fileを推論daemonから隔離するため、optimizerは必ず別processで実行する。

```text
Open Weight (read-only)
        ↓
OptimizationPlan + CalibrationManifest
        ↓
vllm-apple-optimize worker
        ↓
quantization / analysis / pruning / low-rank / repair
        ↓
evaluation and quality gate
        ↓
Immutable Optimized Artifact
```

予定directory：

```text
vllm_apple/optimizer/             planner、worker、adapter、evaluation
schemas/optimizer/               plan、event、artifact manifest
sdk/swift/.../Optimization*.swift
samples/VLLMAppleOptimizer/      Mac companion app
```

### O.1 安全境界

- original model directoryはread-onlyとして扱い、上書きを拒否する
- outputは一時directoryで生成し、検証、`fsync`、atomic promotion後に公開する
- source hash、license、tool/backend version、全transform、seedをmanifestへ記録する
- 実行前に必要disk、peak memory、workspaceを見積もり、hard admission limitを適用する
- optimizer crashは`vllm-appled`と実行中の推論へ影響させない
- activationとlogはbounded memoryまたはdisk streamingとし、全量をRAMに保持しない
- activation statisticsは`OnlineActivationStatistics`でWelford法を用い、個別値を保持せずcount、mean、
  population variance、minimum、maximum、absolute maximum、zero countだけをmergeする。1 updateは最大
  1,048,576 finite値、総countはsigned 64-bit以内とする。`ActivationStatisticsStream`はraw tensor名ではなく
  64文字SHA-256 fingerprintとaggregate snapshotだけをowner-only 0700 directory／0600 JSONLへ追記する。
  `O_NOFOLLOW`、owner／mode／regular-file検査、最大1 GiB、record単位fsyncを適用し、activation全量をdiskにも残さない
- layer、attention head、neuronのimportanceは、上記aggregateから得たmean absolute activationと明示的な
  output sensitivityをsample数でweighted mergeし、その積をscoreとする。最大65,536 component、最大4,096件の
  report出力に制限し、score降順・component key順で決定論的に並べる。全component scoreに対するnormalized score、
  truncation、SHA-256 report IDを返し、raw activation、prompt、model outputはreportへ含めない
- weight clusteringは最大65,536 finite値と最大256 centroidを対象に、quantile初期値、距離同値時の小さい
  centroid index優先、最大256 iterationの決定論的1D k-meansを行い、assignment、MSE、最大絶対誤差を返す。
  low-rank参照近似は最大65,536 matrix要素、rank 64、最大256 power iterationで成分を求めてdeflationし、
  left／right factorとFrobenius誤差を返す。outlier-aware経路は既存のbounded affine base＋sorted sparse residualを
  使用する。いずれもreference analysisであり、品質gateなしに元weightを置換しない
- pruning experimentは最大65,536 finite値に限定し、unstructuredでは絶対値・元index順、structuredでは
  row／column L2 norm・group index順で除去候補を決定する。fractionは0以上1未満とし、最低1 weight／groupを
  必ず残す。mask、zero置換した参照値、pruned count、squared errorだけを返し、元artifactを直接変更しない
- attention head、MLP、layerのfunctional similarityは、同一sample上の同長bounded出力を一時的に比較し、
  cosine similarity、MSE、最大絶対誤差、sample数、左右出力digestに結合したcomparison IDを返す。非finite値、
  shape不一致は拒否し、raw outputはreportへ保持しない
- structural candidate generatorはnormalized importanceが明示ceiling以下のlayerだけをbypass候補とし、
  cosine similarityが明示floor以上の同一layer内attention headだけをhead merge、隣接layerだけをlayer merge候補とする。
  最大4,096候補をscore降順、kind、candidate ID順で決定論的に返し、importance report IDまたはcomparison ID、
  component key、scoreからSHA-256 candidate IDを生成する。この段階ではartifactを変更せず、後続quality gateなしに
  candidateをpromotionしない
- optional repairは`RepairAdapter` protocolのLoRA／SFTだけを許可し、candidate model hash、dataset fingerprint、
  最大sample数、epoch、learning rate、LoRA rank、seedをbounded requestへ固定する。adapter出力はabsolute path、
  source／repaired model hash、methodを再照合し、同一dataset fingerprint・slice・sample／token countのbefore／after
  perplexityを既存quality gateで比較する。既定ではrelative regression 0以下、つまり悪化しないrepairだけをapproveし、
  artifact hashと再評価reportが一致しない場合はfail-closedとする
- profilerは明示実行とし、read/write sample量へhard upper boundを設ける。plan作成時に暗黙実行しない
- profiler実測値はhardware fingerprintが一致する場合だけ所要時間推定へ利用する
- failureはstable error code、localizable message key、recoverabilityを持つversioned JSONとする
- adapter interfaceとcapability reportはversion管理し、registry数にhard upper boundを設ける
- capability detectionは外部adapterをimport・実行せず、package metadataとmodel metadataだけを読む
- concrete exporterが隔離workerへ接続されるまでadapterをexecutableとして公開しない
- workerのstdout/stderrは常時drainし、保持量へhard upper boundを設ける
- artifact treeは全pathをRAMへ保持せず走査し、symlink、special file、上限超過を拒否する
- cancelとtimeoutはworker process group全体へ適用し、不完全workspaceを公開しない
- checkpointはplan、source、output、execution fingerprint、resource budgetへbindingする
- checkpoint fileはprivate directoryへsize上限付きでatomic保存し、外部入力としてstrictに検証する
- converted境界だけworkspace検証へresumeし、それ以前のpartial conversionは再利用しない
- 同一planの同時実行はkernel管理のfile lockで拒否し、process crash時はlockを自動解放する
- resume時は全bindingとworkspace safetyを再検証し、変換済みcommandを重複実行しない
- promotion/checkpoint間のcrash gapは公開artifactを再検証してcompletedへreconcileする
- exporterは検証済みdependency/platform/model matrixだけをexecutableとして公開し、未知versionを拒否する
- MLX変換はshellを介さない固定argument列とし、remote uploadとremote code trustを既定で禁止する
- GGUF変換は明示指定されたowner所有・group/world非writableのregular `convert_hf_to_gguf.py`と、
  呼び出し側が検証済みbuildとしてexact allowlistした`bNNNN`形式のllama.cpp versionを必須とし、未知buildを拒否する。
  Safetensors FP16／BF16／FP32からF16／BF16／Q8_0だけを
  Schema v1固定argument invocationとして公開する。source snapshot hash、output byte budget、checkpoint、isolated worker、
  private workspace検証、atomic promotionはMLXと共通化し、別quantizerを要するQ4系は専用pipelineが完成するまで拒否する
- 実行は副作用のないinvocation生成と明示的executeを分離する
- source bindingは全regular fileを固定長bufferでstreaming hashし、model全体をRAMへ保持しない
- configにdtypeがないsafetensorsはheader sizeをboundedに検証し、weightをloadせずdtypeを判定する
- virtual environmentのPython launcherはsymlink解決せず、capability検出とworker実行環境を一致させる
- backendが非既存output directoryを要求する場合はprivate childへ生成し、検証前にworkspace rootへ正規化する
- production numeric runtime workerは、verified Qwen stage reader、requested-mode-aware memory admission、private
  numeric artifact reader、実MLX resident backend、16 KiB length-prefixed UDS、same-user peer gate、inode-bound
  session credentialを単一composition rootで構築する。artifact load後はsource bundleをconsumeし、explicit unloadで
  MLX array参照とresident reservationを同時に解放する。実device qualificationはF16／BF16／F32の値、digest、
  streaming plan、unload後reserved bytes 0、orderly shutdownを確認するが、full model品質認定とは区別する
- artifact tree hash、byte/file数、elapsed time、peak child RSSをcheckpointとsidecar manifestへ保存する
- baselineとcandidateの評価は別processで順次実行し、両modelをUnified Memoryへ同時常駐させない
- evaluation JSONLはline/sample/token/domain数をboundedにし、全datasetやlogitsをRAMへ保持しない
- quality gateは同一dataset fingerprint、slice、token数を要求し、相対perplexity劣化で判定する
- deterministic generationは固定seedのgreedy decodingを使い、同一sampleのtoken一致率と期待条件scoreで判定する
- generation reportはpromptと生成文を保持せず、最大256 tokenのID列、fingerprint、期待条件scoreだけを保存する
- real-model correctness regressionは同一dataset／prompt contractとcandidate model hashに固定した2〜16回の
  generation reportを束ね、英語・日本語・简体中文のcoverage、baseline token agreement、expectation score、
  run間output fingerprint再現性、worst-case latency／RSS budgetをすべて満たす場合だけapproveする。集約reportは
  prompt、生成本文、token IDを複製せず、model／dataset hash、bounded測定値、判定、deterministic report IDだけを保存する
- Mac sampleはbundled-daemon版とApp Sandbox client版を別targetにする。sandbox版はcompile-time conditionで
  daemon resource解決とchild process起動を除外し、outbound network client、user-selected read-only file、
  app-scoped bookmarkだけをentitlementへ宣言して、独立管理されたloopback daemonへ接続する。sandboxを理由に
  broad filesystem entitlementや任意process executionへ拡張しない
- Optimizer sandbox transportはschema version 1、UUID request identity、operation、opaque bounded payloadだけを
  loopback HTTPへ送る。`http`かつ`localhost`／`127.0.0.1`／`::1`以外、userinfo、query、fragmentを拒否し、
  ephemeral URLSessionでcache、cookie、credential storageを無効化する。request／responseは各1 MiB、接続は1本に
  制限し、responseのschema versionとrequest UUIDが一致しない場合はUI stateへ反映しない。このtransportには
  executable探索、child process起動、signal、filesystem pathの自動展開を実装しない
- daemon側optimizer endpointは既定無効とし、loopback bindかつmodel root／output rootの双方を明示した場合だけ
  enableする。各rootは最大16件、current user所有、非symlinkの実directoryとし、output rootにはwrite権限も
  要求する。plan payloadは固定fieldだけを受理し、canonical model／output pathが各allowlist内にあることを
  model inspection前に確認する。plan操作はartifact directoryを作らず、実model metadataと現在hardwareから
  既存optimizer schema version 1のdry-run planだけを生成する。execute系operationは個別のcheckpoint／確認／
  cancellation contractが実装されるまで明示的に拒否する
- `samples/VLLMAppleOptimizerSandbox`は非sandbox optimizer appと別bundle ID／Xcode targetにし、同じversioned
  transport、plan schema、bookmark storeだけをsource共有する。UIはendpoint、model/output picker、objective、
  memory/disk/duration budget、license、dry-run候補だけを扱い、`Process`／`NSTask`／spawn実装をbuild sourceへ
  含めない。entitlementはApp Sandbox、outbound network client、user-selected read-only、app-scoped bookmarkに
  限定し、英語・日本語・简体中文のkey集合を同一に保つ
- heterogeneous speculative executionはCPUまたはCore MLをdraft、vLLM-Metal／MLX／Native Metalを
  authoritative verifierとする。3 sample以上のbitwise output一致と5%以上のend-to-end latency改善があるprofileだけを
  構築可能とし、draftとverifyはproduction `BackendEngineRegistry`の独立`DRAFT`／`VERIFY` requestとして扱う。
  各stageは共通resource ledgerで予約し、deadline／cancelをstage間とpublish直前に再検査する。clientへ渡せるのは
  GPUが返したsequenceとの共通prefixおよび最初のGPU correction tokenだけで、未検証draft tokenは一切公開しない
- datasetは期待条件の`contains`/`prefix`を明示し、短い正答の部分一致によるfalse positiveを避ける
- domain/language filterをdataset fingerprintへ含め、異なる用途選択のreport比較を拒否する
- 未評価能力は固定値ではなく、実際に評価したdomainから算出する
- instruction modelのchat template適用は明示的opt-inとし、prompt形式をfingerprintとreportへbindingする
- template適用後の実入力token数をsampleごとに記録し、prompt budgetとmodel contextを生成前に検証する
- baseline/candidateのprompt形式、token budget、sample token数が一致しない場合は比較を拒否する

### O.1.1 Edge-native semantic state reuse

FreeTokenのsemantic-aware caching、elastic memory、bandwidth-adaptive executionを参考にする。
ただしCUDA Graph、PCIe offload、NVIDIA quantization kernelへ依存せず、Apple Unified Memory上で
計測可能なbackend-neutral contractとして実装する。

- conversation turn、tool call、tool result、thinking blockの境界をsemantic anchorとして扱う
- anchorはraw promptや生成文を保持せず、session/prefix fingerprint、token位置、backend state handle、byte数だけを保持する
- context編集後は、現在も一致するprefix fingerprintのうち最深のanchorだけを再利用する
- entry数とstate byte数へhard upper boundを設け、LRU eviction対象をcallerへ返してbackend resourceを解放させる
- memory budget変更はscheduler safe pointだけで適用し、evictionは性能だけを変え推論結果を変えない
- CPU/Metal同時実行はUnified Memory帯域を競合し得るため、固定比率ではなく実測profileで判断する
- backend stateはopaque handleとaccounted byte数で所有し、daemonはstate本体へ触れない
- capture済みstateがcache admissionに失敗した場合もbackend releaseを必須とする
- restore不能なstale stateはcacheから破棄し、release失敗は上限付きretry queueへ移す
- runtime snapshotはcache容量、resident bytes、hit/miss、eviction、release failureを公開する
- semantic cacheの通常budgetを基準にWarningで1/2、Criticalで1/8へ縮小し、Normalで容量を復元する
- active scheduler reservationがある間はresizeをpendingとし、safe point到達後だけ適用する
- pressure、target budget、eviction数、applied/deferred状態をruntime eventへ公開する
- reportには評価対象外の能力を必須で列挙し、限定datasetの合格を汎用品質保証として扱わない
- cancel、checkpoint、resumeをstage境界で保証する
- calibration dataを既定でlocal外へ送信しない

### O.2 最適化段階

実装順序は次とする。

```text
O0  contracts、manifest、dry-run resource planner
O1  quantization、backend export、KV/context/batch search
O2  calibration、activation statistics、evaluation gate
O3  pruning、low-rank、clustering、structural analysis
O4  optional LoRA/SFT repair、artifact comparison、Mac UI
```

構造変更は最もriskが高いため、representation最適化と評価基盤より先に実装しない。

### O.3 品質gate

用途を限定した最適化であっても、観測していない能力の維持を保証してはならない。
artifact reportには、評価したdomain、language、context長と未評価領域を明示する。

最低限、英語、日本語、简体中文に加え、利用者が選択したcode、math、science、long-context等を
baselineと比較する。quality budgetを超えたcandidateは公開artifactへ昇格させない。

```text
quality(candidate, domain) >= baseline(domain) - allowed_regression(domain)
```

### O.4 Mac companion app

UIは推論chat sampleと分離し、model選択、用途、quality/speed/memory優先度、resource見積もり、
進捗、pause/resume/cancel、original/optimized比較、provenanceとlicense reportを提供する。
英語、日本語、简体中文へ対応する。Swift SDKはoptimizer workerのprocess実装へ依存せず、
versioned plan/event/artifact modelとtransport interfaceだけを公開する。

`samples/VLLMAppleOptimizer`は独立したmacOS 13 SwiftUI targetとする。最初の実装境界では
model/output directory、objective、license、memory/disk/duration上限を入力し、`plan` subcommandの
schema version 1 JSONを1 MiB以内で厳格decodeする。候補ごとのoutput size、required disk、peak memory、
duration、budget適合、実行可否、blocking reason、plan warningを表示する。CLIはshellを介さず通常file、
非symlink、実行可能な絶対pathだけを120秒上限で起動し、stdout/stderrを各1 MiBに制限する。
model/output/perplexity/generationの選択は種別ごとに最大1 MiBのsecurity-scoped bookmarkとして保存する。
再起動時はUIを出さずに復元し、stale bookmarkは再生成、復号不能・過大な値は即時消去する。復元URLの
security-scoped accessは処理中だけ保持し、恒久的なaccess tokenやfile内容を永続化しない。5 GiBの疎model
fileを全量readせずmetadata参照できる回帰testを置き、大容量file accessがRAM容量へ比例しないことを固定する。
dry-run planの表示までは副作用を持たず、
変換実行には別の明示確認、checkpoint、resume、quality gateを要求する。

実行UIはbudget内かつadapterが実行可能と判定した候補だけを選べる。破壊的操作の確認後、固定引数の
`export --execute`を起動し、owner-only checkpoint root、duration timeout、immutable outputを適用する。
SIGINT/SIGTERMはCLIの`CancellationToken`へ変換し、隔離worker process groupを停止してcancelled checkpointを
保存するため、GUI終了操作でconverterを孤児化しない。利用者は同一planと引数に限り`--resume`を明示できる。
完了JSONはworker/artifact hash一致、file count、絶対manifest pathを検証してからartifact ID、size、peak RSS、
license、transform/tool version、品質評価の有無を表示する。

artifact生成後の比較UIは、利用者が選択したowner-owned 16 MiB以下のJSONLをoriginalとoptimizedへ
同一の256 sample、1 sample 512 token、総131,072 token上限で逐次適用する。各評価reportはplan IDと
random UUIDに分離した0700 evaluation directoryへ0600 immutable fileとして保存し、dataset fingerprint、
slice集合、sample/token数の一致をCLI quality gateで強制する。UIはslice別perplexity regressionと合否、
`generation_quality`、`long_context`、`code`、`mathematics`、`safety_alignment`等の未評価能力を表示し、
不合格を成功表示へ変換しない。

response比較は別のgeneration JSONLを用い、original／optimizedに同一chat template、最大32 sample、
16,384 prompt token、32 new tokenを適用する。UIはsample ID、domain、language、token agreement、
期待値score合否と、両modelの総elapsed／peak RSSを並べる。privacy境界としてpromptと生成本文は
reportやUI stateへ保存せず、bounded token IDs、SHA-256 fingerprint、aggregate benchmarkだけを扱う。

進捗はCLIの`--event-output`で指定したowner-only 0600 JSONL journalを介す。journal親directoryは
current user所有かつ0700、既存file／symlinkを拒否し、1 MiBでhard stopする。各recordは既存の
`optimizer-event-v1` contractを使い、GUIはschema version、plan ID、有限な0〜1 progressを検証した
最終完全行だけを250 ms間隔で読み、prepare／convert／resume／validate／promoteを表示する。
pause／continueはGUIからCLIへSIGUSR1／SIGUSR2を送り、CLIのsignal handlerはthread-safe `PauseToken`だけを
変更する。worker loopが隔離converter process groupへSIGSTOP／SIGCONTを送り、pause中もcancelを監視する。
したがってCLI本体だけを停止してconverterを走らせ続ける状態を作らない。wall-clock timeoutはpause時間も含む。

---

## Phase 3 — Kernel Optimization

状態：`[Done]`

実装：

```text
vLLM-Metal Attention capability and benchmark integration
MLX Q8/Q4 baseline
fused dequantize + quantized GEMV/GEMM + activation
native Metal only for measured backend gaps
MoE kernel
fusion
kernel autotuning
```

実装完了境界は、MLX correctness baseline、Paged Attention／MLA、vLLM-Metal native v2
production dispatch、fingerprint別autotuning、Q4／Q8、Q4＋SiLU、RMSNorm＋RoPE、
top-2 MoE Expert GEMM、capability-gated graph fusion、Metal→MLX→CPU fallback、
toolchain更新時の再probe、multi-model command stressまでとする。大容量Qwen実weightは
repository外の`large-memory` self-hosted runnerを必要とするため、load前admission、30分安定性、
三言語品質、phase profile、前後integrity、Swift evidence再計算を含むworkflowを完了条件とし、
実report未取得のmodelを昇格しない。

---

## Phase 4 — Vision

状態：`[Next]`

実装：

```text
image input
vision encoder cache
multimodal batching
image preprocessing fusion
```

最初の実装境界として、OpenAI互換inline PNG／JPEGのbounded frontend、固定shapeの
Resize／Normalize／Patchify前処理、model revisionと前処理・encoder fingerprintに結合した
容量制限付きLRU encoder cacheを完了した。さらにmodel revision、前処理、encoder、shapeが
互換なrequestだけをまとめ、request内画像を分割せず4種類のresource上限を守るmultimodal
batch plannerを実装した。Resize、Normalize、Patchify、Projectionを単一MLX lazy graphにする
融合候補は、固定shapeの実機microbenchmarkで段階materialize経路とのcorrectnessと性能gateを
通過した。さらにbatch 1／2／4について、latency、images/sec、MLX allocator peak memory/imageを
同一のbounded runnerで計測し、全sampleのdigest安定性を確認した。Phase 4の完了境界は、この
共通入力・前処理・cache・batch・融合・operator benchmarkまでとする。特定VLMのend-to-end
昇格は別途model qualificationを必要とし、operator結果だけでは昇格しない。

---

## Phase 5 — Audio

状態：`[Next]`

実装：

```text
streaming audio
ASR
audio encoder
audio ring buffer
real-time scheduler
speech-to-speech foundation
```

最初の実装境界として、callback側で明示lock、blocking I/O、buffer拡張を行わない
preallocated SPSC ring bufferを実装した。overflowでは未読音声を上書きせず、新規frameを拒否して
telemetryへ記録する。後段にはchunk境界で位相を維持するmulti-channel linear resamplerと、
全波形を保持しないstreaming log-band feature encoderを用意した。chunk sequence、resampler
位相、feature window、累積frame、named recurrent stateをbounded sessionへ結合し、session数制限、
明示close、idle reap時のstate消去まで実装した。さらにREALTIME／INTERACTIVE／BACKGROUNDの
priority class内EDF、推定実行時間によるdeadline admission、cancel、期限切れtask非実行、
lateness telemetryを持つbounded schedulerをcallback外へ配置した。ordered PCMからresample、feature、
deadline scheduling、backend-neutral ASR worker、bounded transcriptまでを接続し、結果のlanguage、
時間範囲、confidence、文字数、final markerを検証する。audio encoder出力はaudio digest、
encoder／feature fingerprint、sample rate、channel、sample範囲へ結合したbounded LRUで再利用し、
oversize outputを保存しない。final ASR transcriptだけをdialogue／speech synthesis backendへ渡す
speech-to-speech foundationを追加し、言語一致、PCM frame alignment、byte／duration上限と
real-time factorを検証する。16 kHz mono、20 ms chunk、10秒のstreaming preprocessing／state
benchmarkでdropout、failure、p50／p95／max latency、RTFを実測した。Phase 5の完了境界は、
実モデルに依存しないaudio input、state、scheduling、ASR／speech contractsとローカル前処理の
実測までとする。実ASR／dialogue／TTS modelおよびCore Audio device latencyは未認定であり、
利用可能なmodelとdeviceを用いる別qualificationなしには昇格しない。

固定graph audio encoderの最初の実機profileには、固定revisionのWhisper tiny AudioEncoderを用いる。
Core ML configurationは`.cpuAndNeuralEngine`とし、実行器がANEだけを使用したとは主張しない。
`[1,80,1,3000]` FP16 log-melから`[1,384,1,1500]` FP16 embeddingへの3 sampleをApple M4で実行し、
全576,000要素の有限値と入力別digest、latency、peak RSS、memory pressure、thermalを検証する。
model revision、MIT license、全compiled artifactのpath／content digestを証跡へ結合し、audio／embedding本文は保存しない。
GPU LLMとのmodel固有projection、ASR品質、汎用embedding／classifierはこのencoder単体認定とは分離する。

汎用embeddingの最初の実artifact profileにはApple公式MobileCLIP S0のimage/text Core ML packageを用いる。
固定revision `3e0a7bfb9fe83da8a3efaa3fd8f7df24214bb947`、Apple-ASCL、source package合計
107,801,759 bytesとpath/content SHA-256を証跡へ結合し、qualificationごとにprivate rootへcompileして終了時に削除する。
Core ML configurationは`.cpuAndNeuralEngine`とし、ANE単独実行とは表現しない。画像`256×256×3`と
Int32 text token `[1,77]`からFloat32 `[1,512]`への各3入力について、全要素の有限性、入力別digest、
L2 norm、latency、peak RSS、memory pressure、thermalを検証する。Apple M4実測はimage 1.23–1.83 ms、
text 0.91–1.37 ms、peak RSS 317,358,080 bytes、pressure normal、thermal nominalで合格した。
synthetic image/tokenによるexecution qualificationであり、ANE実device割当、zero-shot分類精度、classifier、
または特定アプリケーションのsemantic quality認定へ拡張しない。画像、token、embedding本文は保存しない。

classifierの最初の実artifact profileにはApple公式FastViT-T8 ImageNet-1K Core ML packageを用いる。
固定revision `b42811edaa9f7ca6d79eeada204227704484c881`、Apple-ASCL、8,126,615 bytesと
path/content SHA-256を証跡へ結合し、private rootでcompileして終了時に削除する。入力は256×256 BGRA、
出力は1000要素のFP16 probability array、top class label、label probability dictionaryである。
元class labelに同一文字列`maillot`が2件あるためdictionaryは999 unique keyとなる。この既知の表現差を
1000-class欠損と誤判定せず、1000要素配列の有限性・確率和、999 unique key、入力別digest、top label、
latency、peak RSS、pressure、thermalを検証する。Apple M4実測は3入力1.94–2.25 ms、peak RSS
217,677,824 bytes、pressure normal、thermal nominalで合格した。これはsynthetic inputによる実行契約の
認定であり、ANE単独割当やImageNet／自然画像accuracyの再認定には拡張しない。画像と確率本文は保存しない。

---

## Phase 6 — Video

状態：`[Next]`

実装：

```text
hardware video decoding
frame scheduler
temporal cache
video VLM
streaming video
```

最初の実装境界として、FFmpegのVideoToolbox hardware accelerationを明示するbounded decoderを
実装した。local regular file、input byte、codec、resolution、frame数、decoded byte、timeoutを
load前に制限し、短いH.264 fixtureの実機decodeを確認した。さらにraw BGRAをcaptured stdoutや
frame別heap copyを介さず匿名mmapへ直接decodeし、read-only memoryviewとして渡すGPU staging
互換経路を実装した。さらにAVAssetReaderが返すVideoToolbox対応CVPixelBufferを
`alwaysCopiesSampleData=false`で保持し、CVMetalTextureCacheからBGRA8 Metal textureへ直接bindする
native bridgeを追加した。H.264 fixtureの10/10 frameでhardware decode対応、texture binding、failure 0を
確認した。presentation timestamp順reorder、bounded queue、最大reorder幅、late drop、cancel、
backpressure telemetryを持つframe schedulerも実装した。temporal samplerは先頭／末尾、keyframe、
scene-changeを優先し、残り枠を時間軸上のfarthest-point samplingで決定する。最大frame数と最小間隔を
守り、入力順に依存しない。frame、patch、embedding、scene artifactはvideo digest、時間範囲、
transform／model fingerprintへ結合した共通cacheに保存し、global LRUとtier別byte budgetの両方で
raw frameによるembedding evictionを防ぐ。temporal sampling、frame単位embedding cache、missだけの
batch encode、時間順復元、language backendを接続するvideo VLM integration contractも実装した。
encoderのcount、順序、ID、digest、byte数と応答をfail-closedで検証する。streaming inputは
ordered chunkをprivate 0600 artifactへ逐次spoolし、chunk／total budget、sequence、final SHA-256を
検証してからdecodeへ公開する。全videoをmemoryへ保持せず、close、digest mismatch、idle reapで
artifactをunlinkする。さらにpersistent FFmpeg／VideoToolbox workerへcompressed chunkを逐次投入し、
stdout／stderrを並行drainしてdecoded frameをPTS付きでschedulerへ送るbounded incremental pathを
追加した。短いMPEG-TS fixtureで10/10 frame、scheduler rejection／late drop 0を確認した。この経路の
Python scheduler payloadは現在CPU-owned BGRAであり、native zero-copy bridgeとのworker統合は別途必要。
320×180、30 fps、10秒のVideoToolbox＋mapped staging実測でframes/sec、video seconds/sec、
retained buffer bytes/video-minute、peak RSSを記録した。特定video VLM modelは未認定である。
M4/32GB向け動画生成qualificationのbounded profileとrunner mode契約までを実装し、次は
Wan 2.2 TI2V-5Bの実model qualificationへ進む。local-only worker
adapterはMPS、batch 1、T2Vだけを許可し、VAE tiling、bounded telemetry、出力shape／frame数、
private MP4のdigest計算後削除を強制する。I2Vは入力画像をrequest ABIへ安全に追加するまで許可しない。
量子化artifact readinessはDiffusers source上の`WanPipeline`、artifactのpipeline identityと必須component、
transformer metadataの4/8-bit宣言を、backend import、weight load、Metal allocationなしで検証する。
workerはDiffusers 0.34.0の`text_encoder->transformer->vae` residency sequenceを照合し、
`enable_model_cpu_offload(device="mps")`でmodule単位にMPSへ移す。契約不一致時は全常駐へfallbackしない。
正式qualification CLIはreadiness、artifact実容量、resident見積り、現在hardwareのload前admissionを
通過した場合だけlocal-only workerを複数回起動する。sample間はmemory pressureのnormal連続観測を待ち、
promptと生成物を保存せず、digestとshape、latency、peak resident、pressure、thermalだけをreportへ残す。
認定前にmodel treeの全regular fileをrace-safeにhashし、path、size、file SHA-256から導出したroot digestを
provenanceへ結合する。同じartifact byte数と量子化名を持つ別weightへのreport replayは拒否する。
digest field追加前のv1 reportはlegacy evidenceとして読取可能だが、新しいWan正式認定は必ずdigestを持つ。

最初の実modelは`AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit`の固定revision
`6875952a110b6bdbcfc00d72b1d89a8e02ab0fc3`とする。これはmixed Q8/BF16のMLX-Gen packageであり、
配布容量は18,189,391,581 bytes、Apache-2.0、非gatedである。名前に`diffusers`を含むが
Diffusers `from_pretrained`形式ではないため、既存Diffusers workerへ誤接続してはならない。
MLX-Gen動画readinessとisolated workerを先に実装し、その後に固定revisionだけを取得する。

選定artifactの配置後検査では18,189,391,581 bytes、22 files、Q8、必須component完備を確認した。
MLX-Gen 0.33.1はこのlocal pathをWan T2V／first-frame I2Vとして認識する。isolated T2V workerは
`--low-ram`とinactive denoiser releaseを必須にし、in-process実行によってRSSだけでなくMLX allocator
peakもhard ceiling判定へ含める。backend JSON progressはbounded sinkへ隔離し、生成MP4はdigest後に削除する。
正式MLX-Gen動画CLIはversion 0.33.1以上、console entrypoint、Wan pipeline／base identity、mixed Q8/BF16、
component完全性をload前に検証し、合格したartifactだけを反復T2V runnerへ渡す。配置artifactはこのgateに合格した。
実modelの単発smokeでは640×384、33 frame、20 stepをM4/32GBで完走し、memory pressure normal、
最大thermal fairを確認した。Wanはwidth／heightに32-pixel境界を要求するため、旧640×360 profileは
640×384へ修正した。low-RAM resident見積りは最大同時常駐phase、1 GiB allocator余裕、decoded videoで
構成し、共通artifact admissionが別途確保する総Unified Memoryの8% emergency reserveを重複加算しない。
生成中は上限超過でもtelemetryを完了まで保持し、評価器が実測peakでfail判定する。backendが失敗を
`SystemExit`へ変換する場合は、stderrの構造化failureまたは直前512-byte短文だけを診断へ取り込む。
正式2-sample認定では両sampleが640×384・33 frameで完了し、memory pressure normal、thermal fair、
最大peak RSS 11,298,210,948 bytes、median wall 722.27秒、minimum 0.0438 frames/secを記録した。
生成物とpromptは保存せず、MP4 digest取得後のprivate output削除も確認した。続く同一profileの
4-sample stability gateも4/4件で合格し、全件memory pressure normal／thermal fair、最大peak RSS
11,355,004,900 bytes、peak drift 56,793,952 bytes（最大値比0.50%）、median wall 860.66秒を記録した。
解像度、steps、batchを固定し、Wanの時間軸境界を満たす49 frameへ一軸だけを拡大した
2-sample qualificationは2/2件で合格した。最大peak RSSは11,533,691,012 bytesで33-frame
4-sample最大値比+1.57%、median wallは987.66秒で+14.76%だった。全件memory pressure normal、
thermal fair、shape 640×384×49、private cleanupを確認した。次は49-frame profileを4-sampleへ昇格する。
続く49-frame 4-sample stabilityも4/4件で合格した。全件memory pressure normal／thermal fair、最大peak RSS
11,533,691,012 bytes、4 sampleのRSS range 0、median wall 1,098,215 ms、minimum 0.04197 frames/sec、
4 output digest distinct、prompt/output非保存、private cleanupを確認した。これにより次の一軸候補を65 frameとし、
33-frame rootと49-frame stable reportを結合するchained promotionでのみ2-sample評価を許可する。
この契約による65-frame 2-sample qualificationも2/2件で合格した。両件とも640×384×65、
memory pressure normal、thermal fairで、最大peak RSS 11,844,460,868 bytes、median wall 1,532,029 ms、
異なるoutput digest、prompt/output非保存、private cleanupを確認した。65-frame 2-sample reportと初期33-frame
4-sample rootを再検証した同一profileの4-sample stabilityも4/4件で合格した。全件640×384×65、20 steps、
memory pressure normal、thermal fairで、issues 0、最大peak RSS 11,844,460,868 bytes、RSS range 0、
median wall 1,333,142 ms、4 output digest distinct、prompt/output非保存、private cleanupを確認した。
frame-count promotionは4-sample以上かつ全sampleのmemory pressureがnormalである同一artifact reportを
baselineとして要求する。候補初期profileと同じ幅、高さ、steps、batchを維持し、frame数は直前値の
2倍以下かつ`frames - 1`が4の倍数でなければload前に拒否する。
昇格済みframe数を2 sampleから4 sampleへ安定性昇格する場合は、同形状の2-sample reportに加えて
`--promotion-parent-report`で初期frame数の4-sample reportを必須とする。両reportのartifact provenance、
candidate、shape、all-normal pressureを再検証し、先にframe promotion chain、次にsample-count promotionを
復元できた場合だけworkerを起動する。
さらに安定化済み49-frameから65-frame等へ進む場合は、直前の4-sample reportと初期33-frame
4-sample root reportを同時に要求する。両方のcandidate、幅、高さ、all-normal pressure、sample数、frame境界を
検証し、両plan digestからpromotion chain IDを生成する。直前frameの2倍超過、`frames - 1`が4の倍数でない値、
初期rootの差替えはweight load前に拒否する。

初期の動画生成qualification候補は、MacBook Air M4 / 32GBでload前memory admissionを通過する
構成に限定する。

```text
Tier A: Wan 2.2 TI2V-5B
        T2V/I2V、high-compression VAE、量子化、逐次module residencyを優先

Tier B: HunyuanVideo 1.5 8.3B
        480p、step-distilled、SSTA、model offloadを優先

Tier C: Wan 2.2 A14B quantized
        stretch候補。T2V/I2V artifactを区別し、dual expertを同時常駐させず、
        CPU/SSD offloadを併用する。非量子化artifactはM4/32GBでload前に拒否
```

最初の認定profileは低解像度、短尺、batch 1、bounded frames/stepsとする。DiTまたはexpert、
text encoder、3D VAEのartifact bytesとresident bytesを個別に見積もる。合格後は解像度、
frame数、steps、連続生成のうち一軸だけを増やす。reportにはfirst-frame/wall latency、
peak RSS、memory pressure、thermal state、frames/sec、output metadata、backend/model fingerprint、
量子化方式、変換元digest、licenseを含める。CIはweightおよび生成動画をartifactとして保存しない。

runnerは候補modalityから入力artifactを必要としない既定modeを決定する。videoは`text-to-video`、
imageは`text-to-image`とし、明示modeは候補catalogに含まれる場合だけ許可する。これにより
`image-to-video`を入力画像なしで誤起動せず、非対応modeはworker生成前にfail-closedとする。

実測backendはcontrol processへ直接importせず、shellを介さないsubprocessとして隔離する。共通JSONL
telemetryは1 event 16 KiB、1 sample 4096 eventを既定上限とし、collectorはevent履歴を保持しない。
timeoutまたはprotocol違反ではworker process groupを停止する。prompt、生成内容、生成物pathはtelemetryへ
含めず、completed eventに出力shape、frame数、content SHA-256だけを含める。

framework固有境界はversioned worker adapterで統一する。MLX-Gen、MFLUX、Diffusersは固定Python module、
ComfyUIは明示指定した外部telemetry workerとし、exact backend version allowlist、owner所有でgroup/world
非writableな実行file、workspace内non-symlink requestを起動前に検査する。venv Python symlinkはlinkと解決先の
所有者、解決先のregular/executable/permissionを再検査する。いずれも同じJSONL v1、timeout、privacy collectorへ接続する。

Diffusers対応は、frameworkをimportする前に配布sourceをbounded AST scanし、候補ごとのpipeline classを
照合する。T2V/I2Vの両modeを宣言する候補は両pipeline classを要求する。静的readiness通過はsource上の
API存在だけを示し、MPSでのcorrectness、memory fit、実生成成功を示すqualificationとしては扱わない。

Diffusers image worker coreはcandidate IDとpipeline classを固定対応させる。qualification生成物はworkspace内の
一時outputだけに限定し、bounded streaming SHA-256後にinodeを再確認して削除する。reportへ渡すのはshape、
digest、telemetryだけとし、画像bytesとpathは渡さない。workspace外fileをworkerが返した場合は拒否するが、
権限範囲外のfileをcleanup名目で削除しない。

Qwen3-VLのmanaged Core ML→MLX経路は、固定revisionからBF16 vision weightをstagingし、patchと4つの6層
segmentを構築・個別数値qualificationしてから、5 artifactを同一graph IDへ結合してatomic公開する。
subprocess起動時はvenvのPython symlinkと解決先をowner／mode／regular／executableで検証するが、実行時のpathは
symlinkのまま保持してvenv `sys.prefix`を失わない。process queueの飽和はbackend故障やfallbackとして扱わず、
`InferenceEngineBusy`をHTTP境界まで保持して503 `engine_busy`を返す。実機promotionは順序付き複数requestに加え、
並行backpressure、active timeout、client disconnect、cancel後の回復、resource zero、clean shutdownを必須とする。

最初の実runtimeはDiffusers text-to-imageとし、private request消費後にだけframeworkを遅延importする。
local-files-only、MPS availability、candidate/pipeline identity、memory hard ceilingをmodel load/generation境界で
検証する。step callbackを共通telemetryへ変換し、runtime終了時にpipeline参照とMPS cacheを解放する。

---

## Phase 7 — Generative Media

状態：`[Next]`（image／video、audio／music qualification foundation、Kokoro speechおよび
MiniMax Music3 4-bitの実機認定は完了。残るimage系の実機gateは個別に継続）

実装：

```text
image generation
audio generation
music generation
video generation
latent memory management
```

audio／music generationはbackend-neutralなqualification境界を先に実装する。requestはspeech／music種別、
prompt digest、seed、8–192 kHz、1–8 channel、最大1,800秒へ制限し、backendへ渡すprompt自体をreportへ
保存しない。backendは呼出元が用意したowner-only 0700 private root直下へPCM S16LE WAVを1件だけ生成する。
qualificationは`O_NOFOLLOW`、owner、0600、regular file、inode／size不変、最大size、sample rate、channel、
durationを検証しながらcontentをstreaming SHA-256し、終了時にregular owner fileだけを削除する。reportは
digest、音声metadata、wall time、peak RSS、memory pressure、thermal state、backend identityだけを保持し、
prompt、音声bytes、生成物pathを保持しない。実speech／music modelの対応表明は、この共通境界にversion固定workerを
接続し、複数sampleの品質とresource安定性を実機で確認した後に限定する。

speechの最初の認定profileは固定revisionのKokoro-82M-6bit、MLX Audio 0.5.4、24 kHz mono PCM S16LEとする。
worker requestは0600 one-shot JSONで渡し、promptをargv／stdoutへ含めない。Homebrew側のoptional G2P dependencyは
private Python overlayへ分離し、Homebrew管理領域のpackageを置換しない。2 sample実機qualificationでは異なるoutput
digest、memory pressure normal、thermal nominal、最大peak RSS 788,283,392 bytes、request／WAV cleanupを確認した。
musicは同じprivate subprocess境界の別schemaで固定revisionのMiniMax Music3 affine 4-bitを認定する。
Community License、2 shardのSHA-256、14 GB available-memory gateを記録し、44.1 kHz stereo PCM S16LE、
5秒上限・4 stepsの独立2 sampleで異なるdigest、memory pressure normal、thermal nominal、最大peak RSS
9,789,440,000 bytes、request／WAV cleanupを実機確認した。speechのG2P overlayとmusic modelを混在させず、
各workerはprompt／lyricsをargv、stdout、reportへ保存しない。

初期の画像生成qualification候補は、MacBook Air M4 / 32GBでload前memory admissionを通過する
構成に限定する。

```text
Tier A: FLUX.2 [klein] 9B Base
        量子化、text encoder分離、VAE tiling、逐次module residencyを優先

Tier B: Qwen-Image-2512
        対応MPS/MLX backend、量子化、offloadの組み合わせを検証

Tier C: FLUX.2 [dev]
        stretch候補。4-bit級量子化、CPU/SSD offload、chunkingを必須とし、
        非量子化artifactはM4/32GBでload前に拒否
```

配置済み`Qwen/Qwen-Image-2.1`は別候補`qwen-image-2.1`として扱う。固定revisionは
`b3179ad355be050328e483a9dfdd9e60cd62adfa`、Diffusers形式の`QwenImage21Pipeline`で、
論理artifact容量33,131,616,240 bytes、BF16非量子化である。Qwen Research Licenseにより
非商用利用に限定される。公式Hub APIとの照合でlocal／remote SHA一致、非gated、Diffusers、
BF16 7,115,124,736 parametersを確認した。system runtimeの旧Diffusersではpipeline classが不足するため
readinessはfail-closedとし、M4/32GBでの試験は対応runtimeを隔離して
text encoder／transformer／VAEの逐次residencyを用意してから行う。
隔離runtimeはPython 3.12.9、Torch 2.14.0、Transformers 5.17.0、Diffusers
0.41.0.dev0 commit `80c7ed262aeffbeb43ef13ae04baeb9b84515a69`へ固定し、load-free source scanで
`QwenImage21Pipeline`を確認した。全pipelineをMPSへ移す動作は使用せず、text encoder、transformer、
VAEの順に単一moduleだけをresidentにする契約を実装した。
workerは固定Diffusers sourceの`model_cpu_offload_seq`が`text_encoder->transformer->vae`と完全一致する
場合だけ`enable_model_cpu_offload(device="mps")`を使用する。契約不一致やAPI欠落時に全pipelineを
MPSへ移すfallbackは禁止し、weight load後でもgeneration開始前にfail-closedとする。

専用qualification CLIの最初の認定profileは512×512、batch 1、20 steps、独立2 sampleとする。
resident estimateはtext encoder、transformer、VAEのartifact bytesを集計する。Diffusersのmodel CPU
offloadは非active moduleのweightをCPU RAMへ保持し、Apple SiliconではCPUとGPUが同じUnified Memoryを
共有するため、GPUの同時常駐が3 phaseの最大値でも総memory admissionには全artifact weightを下限として使う。
この下限へ1 GiB allocator marginとRGBA float working imageを加える。
この見積りをdynamic Unified Memory ceilingと比較し、pipeline identity、artifact root digestをmodel load前に
固定する。2026-09-21のpreflightは推定18,612,345,141 bytesに対してceiling 9,576,265,155 bytesだったため、
weightをloadせず安全停止した。追加検証によりBF16の安全側見積りは34,209,552,368 bytesとなり、
物理memory 34,359,738,368 bytesに加えて8% emergency reserveを確保できない。したがって通常のCPU offloadで
BF16正式試験は行わず、量子化artifactまたはCPUにも全weightを保持しないdisk-backed phase loaderを先に用意する。
その2-sample実測へ合格後だけ768/1024と連続生成へ進む。
量子化residency plannerはdenoiserとtext encoderをcomponent単位でINT8/INT4へ射影し、VAEとその他の
componentはBF16を維持する。配置済みartifactのINT8理論値はweight 17,249,254,204 bytes、1 GiB
allocator marginと512×512 RGBA float working image込みで18,327,190,332 bytesとなり、32 GiB機の
physical safe ceiling 31,610,959,299 bytes内である。ただしこれはscale/metadata overheadを含まない下限であり、
変換後artifactの実容量とMPS kernel supportを確認するまでは`eligible_for_generation=false`を維持する。
INT8 backend候補はTorchAOとし、隔離runtimeへ0.18.0を固定する。readiness probeはDiffusersのpipeline-level
quantization API、`Int8WeightOnlyConfig`、小型LinearのCPU変換/forward、MPS build/availability、MPS上の
小型INT8 forwardを個別に記録する。conversion readinessとMPS runtime readinessは同一視しない。
2026-09-21時点ではCPU変換/forwardは成功したがprobe時のMPS availabilityがfalseだったため、model weightを
loadせず`conversion_ready=true`、`mps_runtime_ready=false`とした。
変換planはoutputがsource外かつ未作成であること、source内にsymlinkがないことを確認し、最大safetensors
shardとINT8 steady residentの合計を変換peakとして扱う。配置済みartifactでは最大shard
9,968,332,504 bytes、変換peak 28,295,522,836 bytes、8% emergency reserve込みの必要available memory
31,044,301,905 bytesとなる。出力diskはINT8 projected weightへ15% staging marginを加えた
19,836,642,335 bytesを要求する。いずれかのgateを満たさない場合はweightをloadせずoutputも作成しない。
変換workerは正式outputと同じ親directoryへ一意なstaging directoryを作成し、pipeline-level TorchAO
quantization mappingでtransformerとtext encoderへ`Int8WeightOnlyConfig`を適用する。safe serialization後に
pipeline class、quantization method/bit/weight-only metadata、量子化済みcomponent集合、artifact root digestを
再検査する。すべて一致した場合だけdirectoryをatomic renameし、例外または不完全な量子化ではstagingを削除して
正式outputを残さない。
CLI orchestrationはconversion planがeligibleの場合だけ隔離runtime workerをshellなしで起動する。workerの
stdout/stderrは各64 KiBに制限し、timeoutまたは出力超過時は独立process groupへTERM、猶予後にKILLを送る。
admission不合格時は`started=false`を返し、subprocessを作成しない。2026-09-21の実artifact確認ではdynamic
ceiling 15,364,437,443 bytesが変換peak 28,295,522,836 bytesを下回ったため、この経路で安全停止した。
component-streaming converterではtransformerとtext encoderを別child processで順に変換し、各process終了時に
allocatorを含むmemoryをOSへ返す。stagingへはVAE、processor、scheduler等の非量子化componentだけを先に
コピーし、2 componentのINT8 safetensorsを順次追加する。safetensors headerのshape/dtype/data offsetをbounded
検査して最大tensor working setを求め、transformer phase peak 8,323,117,083 bytes、text encoder phase peak
11,085,606,043 bytes、8% emergency reserve込み必要available memory 13,834,385,112 bytesとした。このgateを
通過して18,516,268,404-byte／26-file artifactへatomic昇格し、両componentのTorchAO INT8 weight-only metadataと
root digest `dcacfc334ed0821c18a1ff079e37cae9d99de0966c821c508a145134e85f38cf`を確認した。
Apple M4/32 GiBで512×512、20 steps、独立2 sampleの正式qualificationを実施し、2/2成功、全sample
memory pressure normal、thermal fair、最大peak RSS 2,599,387,136 bytes、median wall 216,912 msだった。
生成画像とpromptはreportへ保存せず、private output directoryにfileが残らないことを確認した。
同一profileの4-sample stability gateでも4/4成功し、全sample memory pressure normal、thermal fair、最大peak
RSS 2,600,943,616 bytes、peak RSS range 2,228,224 bytes（最小値比約0.086%）、median wall 257,631 msだった。
4 output digestはすべて異なり、各sampleのseed分離を確認した。次のpromotionは他条件を固定した768×768とする。
768×768、20 steps、独立2 sampleの一軸promotionは2/2成功し、最大peak RSS
2,599,305,216 bytes、median wall 494,727 ms、thermal fair、異なる2 output digest、prompt/output非保存と
private cleanupを確認した。第1 sampleのmemory pressureはnormal、第2 sample終了時はwarningだった。
evaluatorは両sampleとも有効と判定しissuesも空だが、これを直ち1024×1024昇格の根拠にはせず、同一
768×768 profileの4-sample stability gateでpressureの再現性とRSS driftを確認する。promotion evaluatorは
memory pressureがall-normalでないbaselineを拒否するため、このreportをbaselineにした4-sample起動はweight load前に
`generative resolution promotion requires an all-normal baseline`で安全停止した。したがって、回復後の
all-normal 2-sample baselineを改めて必須とする。
回復時のmemory pressure normal、thermal nominalを確認した後の再試験も2/2生成成功し、最大peak RSSは
2,582,462,464 bytes、median wallは482,200 msだったが、再び第2 sample終了時にmemory pressure warningを
記録した。このため768×768は現時点でstable profileに昇格せず、512×512を認定上限とする。workerは生成後に
`pipeline`を破棄して`torch.mps.empty_cache()`を実行し、各sampleは独立processで終了する。runnerも
次sampleの前にmemory pressure normalを連続観測する。これらを有効にした回復試験でwarningが再現したため、
再試験はgate緩和や待機の追加ではなく、text encoder解放、transformer/VAE逐次offload境界の実測と
component residencyの実効memory削減を実装・検証した後に限る。
worker telemetryのeffective residentはOSのprocess peak、MLX allocator peakに加え、PyTorch MPSの
`current_allocated_memory()`と`driver_allocated_memory()`の大きい方を含める。これによりMPS側allocationが
RSSに現れない場合でも共通hard ceilingを適用する。未対応runtimeやprobe例外は他の実測値へfail-softする。
image-editはgenerative worker ABI v2で扱う。sourceはworkspace内のcurrent-user所有regular fileに限定し、
group/world permission、symlink、空file、64 MiB超過、PNG/JPEG以外をload前に拒否する。親processはpath、byte数、
SHA-256をrequestへ結合し、consume時とDiffusers runtimeで再検証する。runtimeはfile descriptorからbounded readした
byte列だけをPillowへ渡し、最大4096×4096 pixels、RGB変換後のimage objectだけをpipelineへ渡す。request ABI v1は
既存text-to-image向けに読取互換を維持するが、新規requestは常にABI v2とする。source path、prompt本文、生成画像は
qualification reportへ保存せず、実機認定は512×512・2 sampleから開始する。

同じABI v2入力契約をvideo I2Vにも使用する。Wan 2.2は`WanPipeline`／`WanImageToVideoPipeline`、
HunyuanVideo 1.5は`HunyuanVideo15Pipeline`／`HunyuanVideo15ImageToVideoPipeline`をmodeごとに固定し、
候補ID、mode、artifactのpipeline classが完全一致しなければweight load前に拒否する。T2Vでは入力画像を禁止し、
I2Vではdigest-bound private PNG/JPEGを必須にする。HunyuanVideoのadapter実装完了と、未配置artifactでの
480p実機qualification完了は区別する。

MFLUX readinessは独立`mflux` wheelだけでなく、`mlx-gen` distribution内へ同梱された`mflux/` sourceも
distribution file listからload-freeで検出し、versionを`mlx-gen-bundled-<version>`として区別する。
Qwen-Image-2512 image-editではABI v2 sourceをdescriptor readとdigestで再検証し、workerのprivate output rootへ
0600 PNGとして複製してからMFLUXへ渡す。元入力をbackendへ直接再openさせず、複製は成功・失敗の双方で削除する。
Qwen-Image-2512の配置MFLUX packageは全体metadataが4-bitでもtext encoderはBF16/F32である。
28層×466,115,840-byteのweightを全常駐させるとM4/32 GiBのload前admissionを通過しないため、
indexとsafetensors headerを照合した後、指定tensorのbyte範囲だけを読み1層ずつmaterializeする。
実機のsynthetic `[1,40]`入力では独立process 2回とも28/28層が完走し、同じ有限出力digest、
各層normal pressure／nominal thermal、peak MLX 1,576,732,112 bytesを記録した。
続いて配置済みtokenizer templateで英語・日本語・简体中文の実文章をencodeし、計84層の逐次実行で
有限embedding 3/3、shape／mask整合、相異なるdigest、normal memory pressureを確認した。
これはtext encoderの実文章入力に対する実行可能性確認であり、意味品質、embedding handoff、transformer/VAE、
画像生成の認定とは区別する。後段の分離processへ進むため、candidate／plan／prompt／sample identityと
payload SHA-256に結合したprivate一回消費embedding/mask ABIを実装した。英語実promptのF32 embeddingと
I32 maskを別processでdigest一致のまま受け取り、private cleanupとnormal pressureを確認した。
transformer/VAEへ接続する際もload前admission、出力非保存、終了時cleanupを必須とする。
量子化transformerは60 block／6 shardで、header-only inventoryから各block 191,288,320 bytes、
静的部分を含む1 block payload下限214,114,432 bytesを得た。全846件のU32 packed weightの
scales/biasesと4-bit形状比をload-freeで検証し、各60 blockに量子化weightが存在することを確認した。
選択的U32/BF16 readerで第0・第59 blockを4-bit MLX moduleへ復元し、RoPEを省く合成入力の
単独forwardで有限出力とnormal pressureを確認した。現行MFLUXの一部8-bit量子化規則を
適用するとshape不一致になるため、検証済みlayoutを使用する。全60 blockを合成状態
`image[1,4,3072]`／`text[1,13,3072]`で逐次materialize／forward／解放し、60/60有限出力、
normal pressure、peak MLX 195,353,822 bytes、process peak RSS 498,073,600 bytesを確認した。
続いてMFLUX標準のscaled Qwen RoPEを合成grid `[1,2,2]`と13 token textへ適用し、
60/60 blockで有限出力、normal pressure、peak MLX 197,361,374 bytes、process peak RSS
480,919,552 bytesを確認した。このsmokeは実latent寸法、実prompt、static入出力projection、
timestep conditioningの実経路、VAE、画像生成を含まないため、full transformerと画像品質は
引き続き未認定とする。固定層の選択的loadも追加し、`img_in`、`txt_norm`、`txt_in`、
`time_text_embed`、RoPE、60 block、`norm_out`、`proj_out`の順で合成forwardを完走した。
配置weightには`norm_out.linear.bias`があるため、標準MFLUXのbiasなしmoduleをそのまま
使用せず、静的loaderでbiasありの3072→6144線形層へ合わせる。合成image `[1,4,64]`、
text `[1,13,3584]`、timestep 0.5から有限`[1,4,64]`を得て、peak MLX 220,384,096 bytes、
process peak RSS 430,620,672 bytes、pressure normalを確認した。実prompt、実latent、
denoising、VAE、画像生成の認定とは区別する。英語の実promptを28層text encoderで処理し、
F32 embedding／I32 maskをprivate一回消費handoffで別processへ渡す接続も実証した。
consumerは実embeddingを`txt_norm`／`txt_in`からRoPE付き60 blockへ入力し、
有限`[1,4,64]`出力、private残存0、全block pressure normalを確認した。ただし画像側は
合成小latent／合成timestepであり、実画像生成の認定には使用しない。
VAEは配置済み1 shard／192 BF16 tensorのうちdecoderとpost-quant-convの108 tensor
（146,591,206 bytes）だけを選択的に読み、encoder weightをmaterializeしない。
合成packed latent `[1,4,64]`を`[1,16,4,4]`へunpackした32×32相当のdecodeで
有限`[1,3,1,32,32]`出力、pressure normalを確認した。transformer出力との接続と
生成画像の品質は別段階で検証する。
合成text／image／timestepのtransformer固定層＋RoPE＋60 blockの出力を、同一processの
VAE decoderに直接入力する32×32相当の一体forwardも完走し、有限`[1,3,1,32,32]`、
peak MLX 467,962,770 bytes、process peak RSS 480,002,048 bytes、pressure normalを
確認した。実promptを含む同時実行はtext encoderのメモリ安全ゲートで停止したため、
この一体forwardの実prompt対応とdenoising・画像品質は未認定とする。
FlowMatch Euler schedulerと実ノイズ初期化を追加し、合成prompt／32×32相当で
2 step・120 blockの逐次denoising、VAE復元までを実行した。各stepのlatentと最終
`[1,3,1,32,32]`は有限、peak MLX 468,062,634 bytes、process peak RSS 492,060,672 bytes、
pressure normalだった。32px・2 stepは生成品質の判定に使わず、実promptと実用寸法の
qualificationを別途必要とする。
128×128・2 stepの拡大では、2 stepとVAEの有限出力後にmemory pressureがwarningとなり
qualification不合格だった。decoder前のtransformer／中間tensor解放と、使用しないVAE encoder
parameterの非保持を追加したが、空きメモリが約7–9 GBの再試験はstep 0の入場判定で停止した。
decoder単体の小latent smokeは改善後も合格し、peak MLXは417,229,590 bytes、process RSSは
246,398,976 bytesだった。128px以上の開始前空きメモリは暫定的に128px 10 GB、256px 14 GB、
512px 20 GBを下限とし、合格を意味する値ではない。

Qwen-Image-2.1ではpromptをsequential CPU offload下で先にencodeし、embeddingを確定した時点で既存offload hookを外す。
その後text encoderとtokenizerのpipeline参照を破棄し、GC、MPS synchronize、cache解放を行ってから、残る
transformerとVAEだけのgeneration pipelineを再構成する。generation側にはDiffusers block-level group offloadを
MPS同期転送、1 block/groupで適用する。CUDA専用stream prefetchはMPSで有効化せず、APIまたは固定component
sequenceが欠落する場合は全model residencyへfallbackしない。解放の前後でbounded progress telemetryを採取し、
`encode_prompt`またはhook除去APIが固定Diffusers runtimeに存在しない場合は通常pipeline residencyへ暗黙fallbackせず
生成前に拒否する。この経路の768×768実機再qualificationがall-normalになるまでは512×512を認定上限とする。

Qwen-Image-2.1 image-editではcondition imageもtext encoderのvision contextへ含める。text-onlyで事前生成した
embeddingへ後から画像だけを渡すと`image_pad_mask`が欠落するため、conditionを含む`encode_prompt`の3戻り値を保持し、
text encoder解放後のpipeline instanceへmaskを明示結合する。condition encoderは出力解像度と分離した256px area profileとし、
同じリサイズ済み画像をvision contextとVAE conditionへ渡す。text encoderのresidencyはdirect-MPS 25.37 GB、
CPU-only 22.87 GB、module単位sequential-MPS 19.37 GB（256px condition）を比較し、最後だけを採用する。
従来のmodel-level offloadでは512px出力のgeneration phaseが26,672,431,104 bytesに達したが、block-level
group offload後はtext-to-image smokeで17,954,488,320 bytes、20-step image-editで19,381,600,256 bytesへ低下した。
それぞれ実測値＋1 GiBをstaged／image-edit admission floorとする。image-editは512×512・20 step・2 sampleを
2/2完走し、異なるdigest、median wall 219,401.91 ms、pressure warning、thermal fair、private cleanupを確認した。
warningを含むため、この証跡から768pxまたは4-sample stabilityへ自動昇格しない。
同じgroup-offload構成の768×768・20 step・2 sampleも2/2生成し、peak 17,954,488,320 bytes、
median wall 393,331.26 ms、thermal fair、異なるdigest、private cleanupを確認したが、両sampleでmemory
pressure warningとなった。従って768×768はexecution-compatibleだがstable profileへは昇格せず、同じ
profileの無条件再実行ではなくgroup sizeまたはVAE decode residencyを別途改善してから再評価する。
より細粒度なleaf-level offloadも768×768・1 step・2 sampleで評価したが、peak
17,954,488,320 bytesとpressure warningはblock-levelから変わらず、median wall 59,449.16 msへ増加した。
このprofileは採用せずblock-levelへ戻す。text encoder INT4はTorchAO 0.18.0の
`Int4WeightOnlyConfig` API、CPU変換、MPS forwardを独立gateとしてprobeする。2026-09-22の固定runtimeでは
APIは存在したがBF16 LinearのCPU量子化が`mslk >= 1.0.0`不足で停止したため、混合INT4 artifactは生成しない。
`mslk>=1.0.0`もresolver上は利用不能（取得可能なのは0.0.0のみ）である。owner-only一時領域とworker ABI v3で
disk-backed group offloadも実装したが、TorchAO INT8 artifactではDiffusersがTorchAO tensor subclassを
safetensorsへserializeできず明示拒否した。この組合せは以後weight load前に拒否する。次の候補は量子化tensorを
再serializeしない独立process phase loader、またはTorchAO MPS INT4対応版が利用可能になった時点でのreadiness再評価とする。
独立process間ではprompt本文を保存せず、通常BF16/FP16/FP32 embeddingとbool/int maskだけをsafetensorsへ保存する。
handoff manifestはplan/prompt digest、sample index、mode、tensor shape/dtype、payload SHA-256/sizeを結合し、owner-only・
512 MiB上限・atomic保存・一回consume後の確実な削除を必須とする。このhandoff境界は実装済みであり、次にencoder childと
generation childの起動順序、終了確認、telemetry集約へ接続する。
このstaged release実装後の再qualificationは、weight load前admissionでestimated resident
19,599,447,412 bytesが当時のdynamic hard ceiling 16,052,303,299 bytesを超えることを検出し、安全停止した。
後段でtext encoder参照を解放しても`from_pretrained`が全componentを同時にloadする初期peakは削減されない。
required componentへ`None`を渡すpipeline loaderは実機で全weightをloadしたため廃止し、processor、text encoder、
scheduler、VAE、transformerを各subdirectoryから固有classで直接loadするcomponent loaderへ置き換えた。prompt
embedding確定後にtext encoderのhook・参照・MPS cacheを解放してからgeneration pipelineを手動構成する。
TorchAO materialization用1.5倍余裕を含む768 profile見積りは16,146,610,083 bytesである。しかし約25.8 GB空きの
direct-component試験でもtext encoder単体load中にworker hard ceilingを超えたため、M4/32GBでの768 profile昇格を停止する。
次はtext encoder shard単位streaming materialization、またはより大容量Apple Siliconでload peakを実測する。
その後、text encoderとgenerationを別processに分ける二段実行でこの制約を解消した。M4/32 GiBのTorchAO INT8
text-to-imageでは512×512・20 step・4 sampleと768×768・20 step・4 sampleがいずれもall-normalで完走し、
後者の4 digestは相異なり、全phase peakは16,679,387,136 bytesだった。この二段profileに限り768×768を
stable上限とする。先述のwarningを含む単一process profileおよびimage-editへは認定を拡張しない。
1024×1024は512初期rootと768安定reportをchain digestで結ぶ昇格経路を実装したが、現時点では推定常駐
19,028,230,144 bytesがdynamic hard ceiling 13,199,013,315 bytesを超え、load前に停止した。
したがって1024の出力品質・安定性は未認定である。
reportにはwall latency、peak RSS、memory pressure、thermal state、backend/model fingerprint、
quantization provenance、licenseを含める。CIはweightおよび生成画像をartifactとして保存しない。

Z-Image TurboのMLX-Gen経路では、backendがZ-Image classを持つこととartifact互換性を別gateにする。
`mlx-diffusers-conversion`は量子化設定が存在してもMLX-Genのquantized layer metadataを保証しないため、
workerへ渡さない。`mlxgen prepare --model Tongyi-MAI/Z-Image-Turbo --quantize 4`で別directoryへ生成した
`mlx-gen` packageだけを候補とし、base-model、4-bit metadata、Z-Image Turbo console routeをload前に
照合する。native packageはDiffusersの`model_index.json`を持たないためpipeline classは必須にしないが、
値が存在する場合は`ZImagePipeline`との一致を要求する。generic routerにはbase-model identityを明示し、
非Turbo routeへの誤dispatchを防ぐ。
backend制約に合わせて2 step未満をload前に拒否し、正式な最小profileは512×512、9 steps、batch 1、
独立2 sampleとする。Apple M4/32 GiBでの基準実測は、artifact 5,902,985,857 bytes、最大effective
resident 5,627,119,126 bytes、全sample memory pressure normal、thermal fairで合格した。9 GiBの
事前見積りは現在のhard ceilingを超えるためload前に拒否され、基準planは8 GiB見積りを使用する。

1024解像度への昇格は、512 initial reportと、768の同一shapeを4 sample測定したall-normal
stability reportを必要とする。両reportのplan SHA-256を決定論的chain digestへ結合し、candidate、
artifact/backend/hardware provenance、steps、frames、batch sizeを固定したまま解像度だけを変更する。
chainの欠落やwarning/critical/unknown pressureを一件でも含むbaselineはmodel load前に拒否する。
1024実行では10 GiBのresident見積りを維持し、dynamic hard ceilingを下回る時だけloadを許可する。
2026-09-08の初回試行は見積り10,737,418,240 bytesに対してceiling 10,501,027,267 bytesだったため、
model load前に安全停止した。768実測peakが10,886,404,598 bytesであるため、admissionを通す目的で
見積りを引き下げず、available memoryとemergency marginが回復してから同一planを再実行する。
memory回復後にadmissionを通過した初回workerは約18分でstatus 1となり、2 sample目を開始せず停止した。
この診断欠落を防ぐため、subprocess adapterはstdout telemetryとstderrを同時にdrainし、stderrは4 KiB tailに
制限する。親processへ公開するのはworkerが出力した単一fieldのallowlisted error codeだけとし、任意のstderr、
prompt、model/output pathは転記しない。これによりpipe deadlockと秘密情報漏洩を避けながら、
memory、import、I/O、validation、router exit、runtime failureを分類する。
診断有効の再試行はhard ceiling 13,661,173,187 bytesでadmissionを通過したが、1 sample目の実行中に
`memory_error`で停止した。この経路で明示的に`MemoryError`を投げる箇所はworkerのruntime ceiling check
だけであるため、M4/32 GiBの現profileでは1024を不合格、all-normal 768を最大認定解像度とする。
MLX cache上限を変更して再試験する場合は、その値をplan hashへ追加し、512 rootから別の一軸promotion
chainとして取り直す。1024段だけのcache変更は比較条件を壊すため認めない。
最初の低cache profileは0.25 GBを`flux2-klein-9b-base-low-cache` candidate identityとrequired strategyへ
固定し、通常candidateのreportをbaselineに指定した場合はcandidate不一致で拒否する。512×512・20 steps・
独立2 sampleの実測は全pressure normal、thermal fairで合格したが、最大effective residentは
7,760,992,758 bytesで通常profileとの差が65,154 bytes（0.00084%）に留まった。このためcache cap単独を
1024へ昇格せず、次の候補はactive transformer attention/MLP chunkingまたはblock単位residencyとする。
次の`flux2-klein-9b-base-blockwise` profileは、install済みMLX-Gen packageを変更せず、isolated worker内だけで
outer compiled predictを解除してdouble-stream/single-stream transformer blockの出力を`mx.eval`によりmaterializeし、
各境界でallocator cacheを解放する。MLXはcompile変換内の`mx.eval`を禁止するため、predict解除もprofile条件に含める。
patchは例外時も元のclass methodへ戻し、通常workerへ状態を漏らさない。演算とweightは変更しないが、
lazy graph lifetimeという実行条件が変わるためcandidate IDとrequired strategyを別にし、通常・low-cache reportの
baseline流用およびlow-cacheとの同時指定を拒否する。512×512・20 steps・独立2 sampleの新しいrootが全pressure
normalで合格するまで昇格を閉じる。改善が不十分な場合に限りattention queryまたはMLP sequence chunkingを別profile
として検討し、同様に512 rootから取り直す。
2026-09-10の初回実行は、既存profileと同じ10 GiB resident見積りに対してdynamic hard ceilingが
3,748,804,035 bytesだったためmodel load前に拒否した。admission目的で見積りを下げず、Unified Memory回復後に
同一commandを再実行する。
memory回復後の512×512・20 steps・独立2 sampleはstrict verificationを通過し、最大effective residentは
7,761,057,836 bytesだった。通常profileとの差は76 bytes未満であり、low-cache profileより65,078 bytes大きい。
全sample pressure normal、thermal fair、private出力なしだが、1024へ昇格できる実質的改善ではないためnegative
qualificationとして完了する。次はfused SDPAのquery軸chunkingを別candidateで512 rootから評価し、それでも改善が
なければMLP sequence chunkingまたはweight block residencyを検討する。
512-token query chunkを固定した`flux2-klein-9b-base-attention-chunked`も512×512・20 steps・独立2 sampleと
strict verificationに合格したが、最大effective residentはblockwiseと同じ7,761,057,836 bytesで、通常profile比
76 bytes減に留まった。全pressure normal、thermal fair、private出力なしの条件は満たすが昇格しない。
chunkごとのmaterializeにはouter compile解除が必要なため、生成hashはblockwise profileと一致し、compiledな
通常・low-cache pairとは異なる。この差をbitwise correctness同一性とは扱わず、独立numeric execution profileとして
保持する。次の候補はouter compileを維持したままcombined QKV/MLP activationを分割するか、transformer weightを
より細粒度にstagingする設計とする。
outer compileを維持する`flux2-klein-9b-base-mlp-chunked`は、double-stream MLPの`linear_in`と
single-stream統合`to_qkv_mlp_proj`だけをinstance IDで識別し、512-token sequence chunkへ分割する。
module nestingとquantized weight keyは変更せず、非対象Linearは通常経路を通す。512×512・20 steps・独立2 sampleの
strict verificationは合格し、全pressure normal、thermal fair、private出力なしだった。生成hashは通常profileとseedごとに
完全一致した一方、最大effective residentも通常と同じ7,761,057,912 bytesであったため昇格しない。これにより512 peakの
主要因はcache、block間lazy graph、fused-SDPA query output、combined QKV/MLP projection workspaceのいずれでもないことが
実測で絞られた。次はcompiled graphが参照中のarrayを解放しない制約を守りつつ、weight keyを変えないblock単位residency
またはstreaming loadが成立するかを先に設計・検証する。
現MLX-Gen loaderはprepared shardを`mx.load`で全件mergeし、component全体を`model.update`した後、
`CompiledPredictCache`がweight arrayを定数として捕捉する。この状態をapplication側monkeypatchだけでblock unloadすると、
compiled graphのstale array参照または旧array保持を起こすため採用しない。readinessにはweightをloadしない
`weight_block_residency` feasibility gateを追加し、incremental block loader、block release barrier、compiled graph
rebind、stable weight keyの4契約をすべて必須とする。現integrationはstable keyだけを満たし、前三契約をexact blocker
としてfail-closeする。将来MLX-Gen側にstreaming ABIが追加された場合もversion推測で有効化せず、このgateを満たしてから
別candidate identityと512 rootを作る。

---

## Phase 8 — MoE / Large Model

状態：`[Done]` foundation（実model別の性能昇格は継続）

実装：

```text
expert residency
expert prediction
SSD expert tier
large Unified Memory optimization
```

---

## Phase 9 — Multi-Mac

状態：`[Next]`（bounded planner、authenticated framing、loopback mTLS、execution coordinatorは実装済み。
物理複数Mac qualificationが未完了）

```text
Thunderbolt
high-speed Ethernet
pipeline parallel
modality parallel
distributed KV/state
```

logical fabricは最大64 node、2,016 link、1,024 stage、4,096 state shardに制限し、計測済み・認証済みlinkだけを
配置候補にする。plannerはmemory、modality、依存DAG、転送costから決定論的配置を作り、checkpointable stageだけを
healthy replicaから再配置できる。transport frameはplan ID、stage／node、sequence、transport、計測ID、planned bytes、
payload SHA-256を結合する。Ethernet境界は相互TLS、hostname検証、双方certificate pinを要求する。

単一Mac loopbackでは実証明書による1 MiB×8 frameを8/8受信し、digest mismatch 0を確認した。coordinatorは
dependency waveごとの最大64並列、pipeline順序、modality並列、cross-node payload不変、最大256 MiB result、
deadline／cooperative cancellationを実装する。ただしこの証跡から物理Thunderbolt／Ethernet性能や複数Mac上の
分散実行を認定してはならない。次のpromotionは2台以上の実Macでlink計測、correctness、failure isolation、
単一node baseline比較を行う。

---

## Adaptive Track — CPU / GPU / ANE Heterogeneous Scheduling

CPU、GPU、ANEの同時利用は固定した後半フェーズに置かず、依存するcapability、計測、fallback契約が
揃った単位から導入する。`AppleExecutionPlanner`をdevice placementの
唯一の決定点とし、各backendが独自に別deviceへ処理を逃がすことは禁止する。すべてのassignmentは
versioned execution planへ記録し、active request中は変更せずscheduler safe pointでのみ切り替える。

実装順序：

1. `[Done]` CPU／MLX GPU／Native Metal／Core ML/ANEを同じprofile-bound契約で表し、operator、phase、
   precision、probe ID、sticky quarantineによりbounded fallbackを決定するregistryを追加する。
2. `[Done]` 既存kernel probe/cacheをoperator単位でdevice registryへ昇格し、実測していないprecisionや
   phaseを利用可能としないcompositionを追加する。
3. `[Done]` standard libraryだけでCPU vector add、8x8 matmul、KV copyのcorrectnessとlatencyを測定し、
   CPU fallbackもprofile-bound probe ID必須でdevice registryへ登録する。
4. `[Done]` 公開Core ML APIで`.cpuAndNeuralEngine`設定surfaceをbounded Swift subprocessから検査し、
   surface合格だけではexecution capabilityへ昇格しないavailability evidenceを追加する。
5. `[Done]` digest-bound compiled model、private input、bounded outputで固定graphを`.cpuAndNeuralEngine`実行し、
   数値一致・latency・実行前後のmodel tree integrityを検証するadapterを追加する。fixture digestをoperatorと
   probe identityへ含め、合格結果もauxiliary phase／FP32だけへ限定する。
6. `[Done]` repository-ownedの決定論的`x * 2` Core ML fixture generatorとself-hosted qualificationを追加し、
   M4実機で生成、compile、tree integrity、3 sample predictionの数値一致を確認する。
7. `[Done]` 合格済みfixed graphだけにresource handleを発行するCore ML backend lifecycleを実装する。
   model digest、capability ID、auxiliary phase、FP32 eligibilityをload時とdispatch直前に照合し、predictionを
   bounded subprocessへ隔離する。実行前後のmodel tree integrityを再検証し、明示unload後の再利用を拒否する。
8. `[Done]` CPU thread、GPU command queue、ANE task、Unified Memory、memory bandwidthを同じresource
   ledgerで原子的に予約する。operator admission時のovercommitを部分予約なしで拒否し、既存memory予約を
   rollbackする。完了・cancel・dispatch競合時に両予約を解放し、capacity／used／availableをruntime APIへ公開する。
9. `[Done]` operator、shape、batch、precision、phaseとprobe capability IDへ結合した共通microbenchmark schemaを追加する。
   cold load、演算、変換、device同期、end-to-end latency、throughput、peak memory、energyを別fieldで保持し、
   未計測値はゼロにせずunknownとする。bounded sample、output digest安定性、private atomic保存、derived値再計算を必須にする。
10. `[Done]` CPU、MLX、Metal、Core MLのnative measurement adapterを接続する。CPUはbounded deterministic kernel、
   MLX／Metalはprobeと同じ隔離kernel、Core MLはload済みdigest-bound resourceを使い、device kernel時間と
   subprocess・integrity検証込みend-to-end時間を分離する。
11. `[Done]` available・FP32 capabilityだけから代表shapeを生成するbounded deterministic suiteを追加する。
   backend/operatorごとのoperationは明示mapを必須とし、欠損時はfallbackせず拒否する。CPUはvector add、8x8
   matmul、KV copyを現在のMacで各3 sample実測し、安定output digestとsuite report生成を確認する。
   decode寄りのGEMVは専用operatorとして追加し、rows/columns、batch、総積和要素数の上限を固定する。
   Apple M4実機のFP32・7 sampleで64×64×64 GEMMはmedian 7,035,792 ns／37.23M work-items/s、
   256×256 GEMVはmedian 1,846,583 ns／35.24M work-items/sとなり、全sampleのoutput digest一致を確認した。
12. `[Done]` M4実機でMLX、Metal、Core ML代表shapeをqualificationし、CPU 3、MLX 6、Metal 2、
   Core ML 1 capabilityの12/12 correctness合格を確認した。device間同期、tensor変換、Core ML
   compile/load時間はend-to-end latencyへ含める。
   self-hosted macOS ARM64 workflowはCPU 3、MLX 6、Metal 2、Core ML 1 operatorのprobe合格をすべて必須とし、
   各3 sampleのprivate atomic reportだけを14日artifactとして保存する。fixtureとrunner上のreportは常に削除する。
13. `[Done]` 同一workload identityのreportだけを比較するpromotion gateを追加する。CPU baselineを必須とし、
   最低3 sample、output digest一致、既知peak memoryの非悪化、cold load償却込み5%以上のend-to-end latency改善を
   満たすbackendだけをeligibleにする。欠損energy／memoryをゼロとして扱わない。
14. `[Done]` promotion winnerをbenchmark report ID、capability ID、exact workload identityとともにversioned
   device placement planへ結合する。dispatcherのhardware／environment probe profile一致を必須とし、active request中は
   pendingに保持してscheduler safe pointでのみ適用する。reservationには使用したplacement plan IDを固定する。
15. `[Done]` placement planをprivate atomic fileへ保存し、全fieldとplan IDを再計算するstrict loader、最大30日TTL、
   future／expired拒否、current破損時のlast-known-good fallbackを追加する。runtime snapshotにはactive／pending ID、
   有効期限、最大64件のoperator／shape／backend／改善率だけを公開し、入力やtensor値は含めない。
16. `[Done]` runtime probeとdispatcher準備後にdaemonがprofile専用current／last-known-goodを自動restoreし、
   applied／deferred／not-found／rejectedとrollback sourceをbounded eventへ記録する。新plan promotionでは同一profileで
   未期限切れのcurrentだけをlast-known-goodへ保存し、破損planでrollback slotを上書きしない。
17. `[Done]` strict qualification reportを入力にCPU baselineと複数candidateを比較し、cold-load償却、改善率、
   peak memory policy、TTLを指定してcurrent／last-known-goodへatomic promoteする管理CLIを追加する。
   daemonはSIGHUPのsignal handlerからfile I/Oを専用threadへ分離し、probe準備後だけ同じsafe-point restore経路で
   再起動なしにreloadする。active request中はdeferredとする。
18. `[Done]` 認証付き`POST /v1/device-placement`へstrictなreload／rollback actionを追加する。daemonのprofile専用
   file controlを介してsafe-point適用し、active request中の有効なplanはaccepted/deferredとする。API応答とCLIは
   stable message keyおよび英語・日本語・简体中文diagnosticsを返す。
19. `[Done]` runtime placement event／snapshotをSwift SDKのtyped modelへ追加し、Mac appで三言語表示する。
   旧clientではdisabledへ縮退し、placement count、plan ID、shape、改善率の不整合はSDK境界で拒否する。
20. `[Done]` 昇格済みANE routeのtimeoutと出力不一致を秘密情報を含まない固定codeへ変換し、同一workloadで
   probe済みGPU、CPUの順に実行するbounded fallback contractを追加する。
   accelerator固有operatorは同じoperator／phase／precision／shapeを持つbounded CPU referenceを生成し、
   FP32出力を正規化したdigestで比較する。小さすぎて起動costを償却できないgraphはCPU配置を維持する。
   代表encoderは最大1024幅・16層の決定論的dense+ReLU graphとし、非有限値を避ける正規化weight、
   integrity-bound compiled model、bounded CPU referenceを共有する。M4実測では1024幅×16層をCore MLへ
   昇格し、scheduler適用後のANE失敗からCPU fallbackまでend-to-endで確認済みとする。
   Core ML resourceはmodelを一度だけloadするpersistent Swift workerを所有し、改行区切りのbounded JSONで
   predictionを直列化する。read timeout、不正応答、worker終了は固定retryable codeへ変換し、unload時は
   stdin close、bounded wait、terminate、killの順でprocessを必ず回収する。
   worker cacheはhardware fingerprint、OS version、model tree SHA-256、input／output名、input countを
   cache identityへ結合する。同一identityのleaseだけがpersistent workerを共有し、active leaseはevictしない。
   fixed-graph backendのresource loadはcache leaseを取得し、unloadはleaseだけを解放するため、resource lifetimeを
   跨いで同一workerを再利用できる。backend closeは全resourceを逆に解放してからcache processを回収する。
   上限到達時は最古のidle workerだけをcloseして置換し、全entryがactiveの場合、identity不一致、active leaseを
   残したcloseはfail-closedとする。これはprocess内のloaded-model cacheであり、compiler artifactをdisk共有する
   cacheはtoolchain／Core ML version、署名、quarantine、失効契約が揃うまで有効化しない。
   reliability qualificationでは固定enumのfault point／actionと最大32件のruleだけを許可する決定論的injectorを
   使用する。指定hit回数、one-shot／repeatを明示し、request ID、入力、model名、任意error文字列は保持しない。
   backend executeへ注入したretryable／timeoutは通常のbounded fallbackを通し、fatalは即時停止する。
   stop注入時も全backendの逆順回収を続行してfailure型だけを集約する。
21. `[Done]` Vision/Audio encoderと汎用embeddingなど固定graph化しやすいauxiliary workloadから
   Core ML routingを開始し、LLM prefill/decodeはGPU baselineを維持する。Qwen3-VL Vision、Whisper tiny
   AudioEncoder、MobileCLIP S0 image/text embeddingを固定artifactと`.cpuAndNeuralEngine`設定で実機認定した。
   FastViT-T8 classifierも固定artifactと同じ設定で実機認定した。background modelの実artifact profileは
   未認定のため別の計画として扱う。
22. `[Done]` 共有memory bandwidth競合の代表組み合わせを逐次・並列で測定するbounded adapterとprivate
   strict profileを追加し、profile-boundで3 sample以上、出力一致、逐次実行比5%以上の改善を満たす
   evidenceだけをhardware identity一致時にruntime起動時installする。fallback時のresource予約は
   backendごとに原子的に引き継ぎ、容量不足の候補を実行せず次のbounded fallbackへ進める。
   M4実機の5 sample qualificationではoutput digestがすべて一致し、CPU+MLX 40.2%、CPU+ANE 22.8%、
   MLX+ANE 5.53%の逐次比改善で3組すべてを昇格した。profile ID別private既定path、daemon起動時の
   strict restore、runtime profile ID／認定pair数診断、strict Swift decode、旧client fallback、Mac appの
   英語・日本語・简体中文表示、valid-current-only last-known-good promotion、safe-point reload／rollback、
   認証付き三言語管理API、strict evidence検証付きSwift SDK、Mac appの三言語reload／rollback操作まで
   `[Done]`とする。全device pairの認定とresourceの原子的一括予約を必須にする2〜3 stage bounded pipeline
   並列化も`[Done]`とする。probe承認済みfallbackとcontention認定がある場合だけ優先度queueの先頭を
   idle backendへ移し、予約失敗時に元のFIFO位置へ戻すbounded work stealingも`[Done]`とする。
   memory pressure、thermal state、low-power modeに応じた新規admissionの同時数・batch・critical時CPU配置の
   段階的縮退、稼働中requestを維持したsafe-point回復を`[Done]`とする。device assignment、queue wait、
   fallback、contention、thermal/power decisionの固定キー・上限付きruntime observabilityとstrict schemaを
   `[Done]`とする。request ID、入力、operator名、任意のerror文字列を記録しない。typed Swift SDKと
   Mac app英語・日本語・简体中文diagnosticsを`[Done]`とする。自動／省電力／最高性能policyの
   認証付き管理API、typed Swift SDK、Mac app三言語操作UIも`[Done]`とする。最高性能を選んでも
   thermal／memoryによる縮退は上書きせず、緩和はactive request終了後のsafe pointに限る。
   選択はowner-onlyのbounded JSONへatomic保存し、daemon起動時に復元する。権限不備、symlink、破損、
   未知値ではautomaticへfail-closedする。保存に失敗した管理APIは稼働中policyを変更しない。
   これらの再起動時復元も`[Done]`とする。Vision/Audio encoderとembeddingのCore ML routing、
   capability／correctness gate、GPU LLM pipelineとの非同期連携基盤、classifier実artifact認定は
   `[Done]`とする。Whisper encoderからGPU LLMへのmodel固有projection／品質gateを`[Next]`とする。
23. `[Done]` CPU/Core ML draft + GPU verifyをcorrectness-neutralに扱うexecutor、GPU検証済みtokenだけの公開、
   correction、resource reservation、3 sample以上・同一出力・5%以上改善のprofile gateを実装する。
   合格profileだけをowner-only／bounded／atomicに永続化し、model hash、precision、backend、latency、
   再計算profile IDをload時に検証する。不合格profileは永続化しない。実候補として固定revisionのGemma 3
   1B 4-bit draftと4B 4-bit verifierをMLX-LM 0.26.2で測定し、3言語のtoken列は3/3一致したが、
   median 469.78 msから620.63 msへ32.1%低速化し、native MLX経路はdraftもGPUであるため不採用とした。
   次は同構成の無条件再試行ではなく、CPU/Core ML用draft artifactまたは十分大きいGPU verifierとの
   別構成を測定し、5%以上改善したprofileだけを標準decode routeへ昇格する。
24. `[Done]` thermal、memory pressure、low-power modeを入力に、batch、concurrency、device assignmentを
   段階的に縮退・復元する。既存requestをcancelせず、新規admissionを制限し、緩和は次のsafe pointで適用する。
25. `[Later]` hardware、OS、Core ML、MLX、Metal、model、shapeに結び付いたprofileを保存し、期限切れ、
   quarantine、last-known-good rollbackを既存kernel profileと同じfail-closed policyで管理する。

昇格条件は、backend間のbounded numerical comparisonまたはtask固有quality gateが合格し、代表workloadで
TTFT、TPOT、throughput、energy/requestの少なくとも一つが改善し、peak Unified Memory、memory pressure、
thermal stateが悪化しないことである。compile failure、timeout、shape非対応、数値不一致では
ANE → GPU → CPUの明示fallbackを使う。ANE利用不可は通常状態として扱い、runtime readinessを失敗させない。

Mac appには自動、省電力、最高性能の三policyとdevice assignmentの診断を英語、日本語、简体中文で
表示する。operator入力、prompt、生成内容はtelemetryに保存せず、queue wait、実行時間、fallback理由、
resource pressureだけをbounded metricとして保持する。

portable numeric route診断はPython CLIと同じschema version 1をSwift SDKの`NumericRouteDiagnostic`でtyped
decodeする。source／runtime／compute format、tensor role、routeはclosed enumとし、finiteかつ非負のabsolute error／
RMSE budget、`valid=true`、固定message key、bounded fallback reasonを検証する。file loaderは64 KiB以下のowner-owned
regular fileだけを`O_NOFOLLOW`で読み、inode／sizeをopen前後で照合する。Mac appは
`VLLM_APPLE_NUMERIC_ROUTE_DIAGNOSTIC`で明示指定された証跡だけを読み、format chain、error budget、fallbackを
英語、日本語、简体中文のcatalogで表示する。CLI由来のmessage本文はUI文言として信用せず、local message keyを使う。

---

# 51. 最終アーキテクチャ

```text
                        AI Application
                              │
             ┌────────────────┼────────────────┐
             │                │                │
            Text            Video            Audio
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                     Multimodal Runtime
                              │
                    Apple Runtime IR
                              │
               ┌──────────────┼───────────────┐
               │              │               │
        Compute Planner Memory Planner Stream Planner
               │              │               │
               └──────────────┼───────────────┘
                              │
                   Adaptive Scheduler
                              │
       ┌──────────────────────┼──────────────────────┐
       │                      │                      │
      CPU                    GPU                    ANE
       │                      │                      │
       └──────────────────────┼──────────────────────┘
                              │
                       Unified Memory
                              │
                 ┌────────────┴───────────┐
                 │                        │
                RAM                      SSD
                 │                        │
        ┌────────┴─────────┐      Cold AI State
        │                  │
     Model               State
                         │
              ┌──────────┼───────────┐
              │          │           │
             KV       Video        Audio
                       Cache        State
```

---

# 52. プロジェクトとして最も重要な原則

このプロジェクトの本質は、

**Apple SiliconでvLLMを動かすことではない。**

vLLMは上位のserving architectureとして利用する。

本当に作るべきものは、

**Apple SiliconというSoC全体を一つのAIコンピューターとして利用するruntime**

である。

そのため、

```text
LLM
Vision
Video
Audio
Agent
```

を別々のruntimeへ分割するのではなく、

```text
             AI Workload
                  │
         Apple Runtime Planner
                  │
          Unified AI Memory
                  │
        Heterogeneous Compute
```

という統一モデルを採用する。

これによって将来、新しいAIモデルが、

```text
Transformer
Mamba
MoE
Diffusion
Flow Matching
World Model
Neural Codec
Spiking model
```

のどれになったとしても、runtime全体を書き直さずに対応できる構造を目指す。

最終目標は「Apple Silicon版vLLM」ではない。

**テキスト、画像、映像、音、Agentを含むApple Silicon専用AI Runtime OSを構築し、その最初のserving frontendとしてvLLMを利用する。**

この位置付けにしておけば、将来vLLM自体より優れたserving architectureが登場したとしても、Apple Runtime Planner、Unified Memory Manager、Metal kernels、Audio/Video pipelineなどの資産をそのまま残すことができる。
