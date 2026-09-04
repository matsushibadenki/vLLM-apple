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

---

# 45. Benchmark設計

単純なtokens/secだけを評価しない。

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

---

# 50. 開発フェーズ

## Phase 1 — 基盤

状態：`[Next]`

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

状態：`[Next]`

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
[Later] CPU/GPU profiler
[Later] kernel benchmark
[Later] automatic batch
[Later] adaptive state allocation
[Later] memory pressure monitoring
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
- 実行は副作用のないinvocation生成と明示的executeを分離する
- source bindingは全regular fileを固定長bufferでstreaming hashし、model全体をRAMへ保持しない
- configにdtypeがないsafetensorsはheader sizeをboundedに検証し、weightをloadせずdtypeを判定する
- virtual environmentのPython launcherはsymlink解決せず、capability検出とworker実行環境を一致させる
- backendが非既存output directoryを要求する場合はprivate childへ生成し、検証前にworkspace rootへ正規化する
- artifact tree hash、byte/file数、elapsed time、peak child RSSをcheckpointとsidecar manifestへ保存する
- baselineとcandidateの評価は別processで順次実行し、両modelをUnified Memoryへ同時常駐させない
- evaluation JSONLはline/sample/token/domain数をboundedにし、全datasetやlogitsをRAMへ保持しない
- quality gateは同一dataset fingerprint、slice、token数を要求し、相対perplexity劣化で判定する
- deterministic generationは固定seedのgreedy decodingを使い、同一sampleのtoken一致率と期待条件scoreで判定する
- generation reportはpromptと生成文を保持せず、最大256 tokenのID列、fingerprint、期待条件scoreだけを保存する
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

---

## Phase 3 — Kernel Optimization

状態：`[Later]`

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

---

## Phase 4 — Vision

状態：`[Later]`

実装：

```text
image input
vision encoder cache
multimodal batching
image preprocessing fusion
```

---

## Phase 5 — Audio

状態：`[Later]`

実装：

```text
streaming audio
ASR
audio encoder
audio ring buffer
real-time scheduler
speech-to-speech foundation
```

---

## Phase 6 — Video

状態：`[Later]`

実装：

```text
hardware video decoding
frame scheduler
temporal cache
video VLM
streaming video
```

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

実測backendはcontrol processへ直接importせず、shellを介さないsubprocessとして隔離する。共通JSONL
telemetryは1 event 16 KiB、1 sample 4096 eventを既定上限とし、collectorはevent履歴を保持しない。
timeoutまたはprotocol違反ではworker process groupを停止する。prompt、生成内容、生成物pathはtelemetryへ
含めず、completed eventに出力shape、frame数、content SHA-256だけを含める。

Diffusers対応は、frameworkをimportする前に配布sourceをbounded AST scanし、候補ごとのpipeline classを
照合する。T2V/I2Vの両modeを宣言する候補は両pipeline classを要求する。静的readiness通過はsource上の
API存在だけを示し、MPSでのcorrectness、memory fit、実生成成功を示すqualificationとしては扱わない。

Diffusers image worker coreはcandidate IDとpipeline classを固定対応させる。qualification生成物はworkspace内の
一時outputだけに限定し、bounded streaming SHA-256後にinodeを再確認して削除する。reportへ渡すのはshape、
digest、telemetryだけとし、画像bytesとpathは渡さない。workspace外fileをworkerが返した場合は拒否するが、
権限範囲外のfileをcleanup名目で削除しない。

最初の実runtimeはDiffusers text-to-imageとし、private request消費後にだけframeworkを遅延importする。
local-files-only、MPS availability、candidate/pipeline identity、memory hard ceilingをmodel load/generation境界で
検証する。step callbackを共通telemetryへ変換し、runtime終了時にpipeline参照とMPS cacheを解放する。

---

## Phase 7 — Generative Media

状態：`[Later]`

実装：

```text
image generation
audio generation
music generation
video generation
latent memory management
```

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

最初の認定profileは512×512、batch 1、単一画像、bounded stepsとし、model、text encoder、
VAEのartifact bytesとresident bytesを個別に見積もる。合格後だけ768/1024と連続生成へ進む。
reportにはwall latency、peak RSS、memory pressure、thermal state、backend/model fingerprint、
quantization provenance、licenseを含める。CIはweightおよび生成画像をartifactとして保存しない。

---

## Phase 8 — MoE / Large Model

状態：`[Later]`

実装：

```text
expert residency
expert prediction
SSD expert tier
large Unified Memory optimization
```

---

## Phase 9 — Multi-Mac

状態：`[Later]`

```text
Thunderbolt
high-speed Ethernet
pipeline parallel
modality parallel
distributed KV/state
```

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
