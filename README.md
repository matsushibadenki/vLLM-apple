# vLLM-Apple Runtime

Apple Silicon向けの、メモリ安定性を重視したAI runtime control planeです。

現在はPhase 1として、hardware/memory検出、安全なcontext計算、runtime profile、
メモリ予約scheduler、headless daemon、versioned local APIを実装しています。

## 安定性の基本方針

- context計算ではphysical memoryとcurrent available memoryの小さい方を採用
- OS reserve、workspace、緊急headroomを推論用budgetから除外
- request bodyを4MiBへ制限
- server threadを既定32本へ制限し、過負荷時は503で早期拒否
- load計測値は固定bucketと上限付きerror分類で保持し、計測自体のmemory増加を抑制
- profileは一時fileへ書き、`fsync`後にatomic replace

Macアプリ用Swift Packageは `sdk/swift` にあります。SwiftUIやAppKitへ依存せず、
Foundation、`async/await`、`AsyncThrowingStream` を公開interfaceにしています。

```bash
python3 -m vllm_apple hardware
python3 -m vllm_apple doctor
python3 -m vllm_apple context --model-memory-gb 8 --kv-bytes-per-token 524288
python3 -m vllm_apple serve
python3 -m vllm_apple serve mlx-community/your-model --max-model-len 8192
python3 -m vllm_apple serve --socket-path /tmp/vllm-apple.sock \
  --session-token-file /tmp/vllm-apple.token
```

サーバーは明示指定しない限り `127.0.0.1` のみにbindします。推論backendが未設定の
場合もhealth、hardware、runtime profile APIは利用できます。

modelを指定した場合は、別processのvLLM-Metal serverをloopback interface上で起動し、
OpenAI APIをcontrol daemon経由でproxyします。SSEはresponse全体を保持せず、小さいchunkで
転送します。`doctor` が互換環境を確認できない場合は起動を拒否します。異なる環境を明示的に
利用する場合は `--backend-executable /path/to/vllm` を指定してください。

local directoryまたはHugging Face cacheにmodelがある場合は、weight shardと`config.json`から
標準Transformer/GQAのKV memoryを計算し、BALANCED contextを自動適用します。安全にinspection
できないmodelは、`--max-model-len`が指定されていなければ4096 tokenへ制限します。

Macアプリとのlocal接続用にUnix Domain Socketを作成できます。session tokenをcommand lineへ
露出させないよう、`--session-token-file`の利用を推奨します。token fileとsocketは0600で作成され、
runtime state/failure eventは `/v1/events` からSSEで購読できます。

daemonが強制終了してUDS entryが残った場合、次回起動はownerとsocket typeを検証したうえで
stale entryを置換します。session token fileは再利用し、通常のSIGTERM shutdownではUDSを削除します。

health、runtime snapshot、runtime eventのv1 schemaは`schemas/`に固定し、live HTTP/SSE responseを
dependency-free validatorで検証します。validatorが未対応のschema keywordを検出した場合もtestを
失敗させるため、schema制約が黙って無視されることはありません。

`VLLMAppleKit`の`UnixSocketRuntimeClient`はPOSIX UDS上でhealth、profile、chat、SSEを
利用できます。`ManagedRuntime`へsocket pathとtoken fileを渡すとUDS clientを自動選択し、
bounded stdout/stderr log、readiness監視、failure時だけの上限付きrestart policyを提供します。

`RuntimeResourceResolver`はapp bundle内の`vllm-appled`を検出し、Application Support配下の
profile/log/tokenと、Darwinのpath長制限を満たす`/tmp`配下のUDSを解決します。directoryは0700、
session tokenは0600で作成されます。

英語、日本語、簡体字中国語に対応した最小macOS SwiftUI chat sampleは
`samples/VLLMAppleChat`にあります。bundleにdaemonがない場合は`127.0.0.1:8000`へ接続します。
開発中のdaemon executableを明示する場合は環境変数を使用できます。

```bash
cd samples/VLLMAppleChat
swift run
VLLM_APPLE_DAEMON_PATH=/path/to/vllm-appled swift run
```

## 同時負荷とメモリ安定性の検証

`vllm-apple-soak`は、履歴sampleを無制限に保持せず、固定12 latency bucketと最大17 error keyで
throughput、failure、RSS増加量をJSON出力します。既定ではloopback以外への接続を拒否します。

```bash
python3 -m vllm_apple.soak --duration 300 --warmup 5 --concurrency 8
python3 -m vllm_apple.soak --mode chat --model your-model \
  --duration 1800 --concurrency 4 --pid 12345 --max-rss-growth-mib 256 \
  --session-token-file /path/to/session.token
python3 -m vllm_apple.soak --mode chat-mixed --model your-model \
  --duration 1800 --concurrency 4 --pid BACKEND_PID --max-rss-growth-mib 256 \
  --require-30-minute-window --session-token-file /path/to/session.token
```

request failureまたはRSS上限超過ではexit code 1、設定・接続準備のerrorではexit code 2を返します。
実modelの長時間判定では、model load後のdaemon PIDを`--pid`へ指定してください。
`chat-mixed`はnon-streaming JSONとstreaming SSEを交互に実行し、応答構造と`[DONE]`までの完走を
検証します。30分認定modeは1800秒以上、PID監視、RSS増加上限のすべてを必須とし、backendが
途中終了した場合も失敗します。

backendの起動からshutdownまでを一度に検証する場合は`qualify-model`を使用します。起動前に
vLLM/vLLM-Metalの互換性を検査し、model load後のPIDを自動的にRSS監視へ接続します。

```bash
python3 -m vllm_apple qualify-model /path/to/model \
  --backend-executable /path/to/vllm \
  --max-model-len 16384 --duration 1800 \
  --concurrency 4 --max-rss-growth-mib 256
```

短時間の配線確認に限り`--allow-short-run --duration 60`を使用できます。この結果は30分認定には
なりません。

daemonのSIGINT/SIGTERM shutdownでは新規リクエストを`server_draining`で拒否し、既存のHTTP/UDS
リクエストを既定30秒まで待ってからbackendを停止します。猶予時間は
`serve --shutdown-grace-period SECONDS`で変更できます。

operator dispatcherを有効にしたschedulerでは、MLX/Metal acceleratorは現在のhardware、OS、
toolchain、backend versionに結び付いたkernel probeが成功した場合だけ選択されます。未probe、
correctness不一致、性能退行、native crashはfail-closedでquarantineされ、利用可能なMLXまたはCPUへ
fallbackします。MLX probeはnative crashからdaemonを守るため短命な隔離subprocessで実行します。
Native Metal probeも固定された小型shaderをSwift子プロセスでcompile、dispatch、readbackし、専用の
temporary module cacheだけを使用します。Metal device、toolchain、command queueの異常はdaemonを
終了させず、そのoperatorのquarantineとして扱います。

`RuntimeProbeCoordinator`はMLX/Metalの結果を同じenvironment fingerprintへ統合し、dispatcherを
scheduler safe pointで適用します。処理中requestがある場合はpendingとして保持し、最後の
reservation解放後にexecution planやelastic memory policyと同じ境界で切り替えます。

互換性検査を有効にした`serve <model>`では、Metal toolchain、MLX package、backend versionを
MLX本体をimportせずに検出し、起動時にprobe coordinatorを自動実行します。probe完了数と
quarantine数はbounded runtime eventへ通知されます。問題調査時に限り
`--skip-runtime-probes`で自動probeを無効化できます。

probe結果はhardware/environment fingerprint別のprivate cacheへatomic保存されます。完全一致する
cacheではnative probeを省略し、quarantineも再利用します。OS、toolchain、MLX、backend、probe
suite versionが変わると別cacheになり、安全に再測定されます。cacheは既定7日で期限切れとなり、
未来の作成時刻も拒否します。

MLX suiteはvector add、16x16 matmul、bounded KV copyに加え、sequence長8/32・head dimension 8の
scaled dot-product attentionを検証します。attentionの出力は小数点以下5桁へ正規化し、無害な
浮動小数差を許容しながら両shapeのどちらかが壊れればoperator全体をquarantineします。
同じcapabilityにはcausal prefill、single-token decode、4 query head/2 KV headのGQAも含まれ、
全シナリオが一致した場合だけ`attention`を有効化します。異なるsuite versionのcacheは移行せず、
correctnessを再測定します。

Paged Attention probeは非連続block tableでKVを再構成し、14、256、1024 tokenのdecode tierを
検証します。Native Metal版も固定された14 token shaderのcompile、dispatch、readbackを独立した
capabilityとして検証します。MLA probeは16 token×4次元のcompressed latentを別々の
key/value projectionで8次元へ展開し、attention出力まで比較します。probe suite version 4より
各operatorを独立してquarantineし、現行suite version 6ではbounded数値toleranceを含む新しい
probe contractで再測定します。

実モデルの`config.json`から、query/KV head、head dimension、context、block数、KV working-setだけを
抽出したbounded shape profileも生成できます。重みtensorはロードしません。

```bash
python3 -m vllm_apple kernel-shape-profile /path/to/model \
  --contexts 128,1024,4096,16384 --block-tokens 16
```

Native Metal adapterはこのprofileを最大4 shapeずつ消費し、各shapeを別のprobe identityとして
計測できます。代表bufferは64 MiBをhard limitとし、上限を超えるshapeは確保前に拒否します。
測定reportはmodel profile、hardware、environment fingerprintに結び付けてprivate/atomic保存でき、
読み戻し時には権限、サイズ、全identity、result由来IDを再検証します。
長shapeでは最大絶対誤差`1e-5`以内のbounded数値ベクトル比較を使い、それ以外のprobeは従来どおり
SHA-256 digestの完全一致を要求します。現行suite version 7ではshape kernelをscore、softmax、outputの
3段階へ分割し、scoreをcontext方向、outputをhead dimension方向へ並列dispatchします。softmaxは
256-threadの固定scratchでmax/sum reductionし、context長に依存するthreadgroup memoryを持ちません。
中間score bufferも64 MiBの総allocation上限に含まれます。

shape autotunerは32、64、128、256 threadの最大4候補を同じCPU基準へ照合し、correctnessを
通過した候補のMetal中央値だけを比較します。最速から2%以内は同等とみなし、thread総数が小さい
構成を決定的に選びます。winnerと全候補はmodel、hardware、environment fingerprint別のversioned
reportとしてprivateかつatomicに保存できます。誤った高速kernelがwinnerになることはありません。
読み戻し時はfile owner、権限、サイズ、全fingerprintを検査し、保存済み候補からwinnerを再計算して
2% tie-break policyと一致しないreportを拒否します。

```bash
python3 -m vllm_apple metal-shape-tune /path/to/model \
  --contexts 128,1024 --samples 3

python3 -m vllm_apple metal-shape-tune /path/to/model \
  --contexts 1024 --samples 3 --stdout
```

daemonへreportをinstallすると、active requestがある間はpendingとして保持されます。各reservationは
開始時の`tuning_id`を固定し、最後のactive request完了後のscheduler safe pointでのみwinnerが
切り替わるため、1 request内でthread構成が混在しません。

推論時はwinner tableをversionedかつ4 KiB以下の内部headerとしてlocal backendへ渡します。
OpenAI互換のrequest JSONは変更しません。非streamingでは応答完了まで、streamingではupstreamの
終了またはclient切断まで同じreservationを保持するため、実行途中の構成切替を防ぎます。context-aware
contractに未対応のengineは従来の呼び出しへ安全にfallbackします。

model指定でdaemonを起動すると、model shape、Apple hardware、OS/Metal/MLX/backend versionが
完全一致する保存済みreportをApplication Supportから自動探索します。private directory内の候補を
最大64件に制限し、破損・権限不正・identity不一致のfileを採用せず、検証済みreportのうち最新を
決定的にinstallします。明示report指定または自動導入の無効化も可能です。

```bash
python3 -m vllm_apple serve /path/to/model \
  --metal-tuning-report /path/to/tuning.json

python3 -m vllm_apple serve /path/to/model --disable-metal-tuning
```

vLLM-Metal側には依存追加なしのASGI adapterを組み込めます。contextは`ContextVar`でrequestごとに
隔離され、headerが壊れている場合やshapeが完全一致しない場合は`None`を渡してbackend既定構成へ
fallbackします。別requestのwinnerが混入することはありません。

```python
from vllm_apple import BackendKernelTuningAdapter, KernelTuningASGIMiddleware

tuning = BackendKernelTuningAdapter()
app.add_middleware(KernelTuningASGIMiddleware, adapter=tuning)

# Paged Attentionの実call siteで使用するbridge
result = tuning.invoke_paged_attention(invoke_kernel, shape, pages, query)
```

`invoke_kernel`には`(shape, configuration, ...)`が渡されます。`configuration`が存在する場合は
`score_width`、`softmax_width`、`output_width`を各Metal dispatchへ適用します。metricsでは
accepted/rejected contextとshape hit/missを取得できます。

managed backend起動時は、`vllm serve --help`を10秒・1 MiB上限で検査し、`--middleware`と
`--disable-frontend-multiprocessing`の両方が利用できる場合だけmiddlewareを自動登録します。
未対応backendでは推論起動を維持したままtuning integrationを無効化します。受理したbackendは
responseにtuning IDを返し、control planeはacknowledged、missing、mismatchを別々に計測します。
acknowledgementはheaderをparseした時点ではなく、shape一致winnerがkernel hookから取得された場合だけ
付与されます。

vLLM-Metal sourceと現在のtuning ABIの互換性は、sourceを実行せずに検査できます。

```bash
python3 -m vllm_apple vllm-metal-integration-inspect /path/to/vllm-metal
```

`doctor`はPythonに加えてvLLM、vLLM-Metal、Transformersのverified matrixを検査します。現在の
vLLM 0.28.0 / Transformers 5.15.0は、vLLM-Metal側の対応確認が完了するまで未検証として拒否します。
詳細は[VERSION-COMPATIBILITY.md](docs/VERSION-COMPATIBILITY.md)を参照してください。

現行native v2はNAX、tiled、per-token、split/reduceでthread構成とshared memory条件が異なるため、
3-stage winnerを暗黙適用しません。詳細は
[VLLM-Metal-Integration.md](docs/VLLM-Metal-Integration.md)を参照してください。

native v2 tuning profileはNAX、tiled、per-token、split-KVを別familyとして扱います。最大9 sampleの
中央値、correctnessとoutput digest、2% tie-breakを適用し、hardwareとupstream source fingerprintへ
固定します。profileは最大16 shape・512 KiBでprivate atomic保存され、読み戻し時にwinnerを含む
全policyを再検証します。

native measurement ABIを実装したvLLM-Metal sourceでは、model metadataからboundedなdecode/prefill
shapeを生成し、実device profileを保存できます。未対応native extensionでは安全に終了します。

```bash
python3 -m vllm_apple vllm-metal-v2-tune /path/to/model \
  --source-root /path/to/vllm-metal \
  --helper /path/to/vllm-apple-v2-measure \
  --contexts 1024,4096 --samples 3
```

実モデルshapeのMetal benchmarkは1コマンドで実行できます。既定ではhardware、environment、
model profileごとにApplication Support配下へprivate保存し、`--stdout`ではMacアプリが扱いやすい
versioned JSONを返します。

```bash
python3 -m vllm_apple metal-shape-benchmark /path/to/model \
  --contexts 128,1024 --block-tokens 16 --samples 1

python3 -m vllm_apple metal-shape-benchmark /path/to/model \
  --contexts 1024 --stdout
```

## Prefill / Decode profile

streaming backendがOpenAI互換のstream usageを返す場合、TTFTとTPOTを分離して計測できます。
promptや生成本文はprofileへ保存せず、固定サイズbucket、token数、対象PIDのpeak RSSだけを保持します。

```bash
python3 -m vllm_apple execution-profile --save
python3 -m vllm_apple phase-profile --model your-model --samples 5 \
  --pid BACKEND_PID --session-token-file /path/to/session.token
```

usageが返らないbackendではtoken数を推測せず、`usage_missing`として失敗します。remote URLは
既定で拒否され、明示的な`--allow-remote`が必要です。

## 段階的long-context評価

`long-context-evaluate`はvLLMの`/tokenize`を使ってretrieval promptを目標token長へ調整し、
1K、4K、16Kなどを短いcontextから順に評価します。各段階でretrieval成功率、TTFT、TPOT、
tokens/sec、model load peak、steady/request peak RSS、state bytesを分離して記録します。

```bash
python3 -m vllm_apple long-context-evaluate \
  --url http://127.0.0.1:8001 --model your-model \
  --stages 1024,4096,16384 --memory-ceiling-gb 24 \
  --state-bytes-per-token 131072 --pid BACKEND_PID
```

`/tokenize`と`/v1/chat/completions`へ直接接続するため、既定URLはbackend用のport 8001です。
stream usageの実token数が目標から外れる、retrieval keyが一致しない、またはmemory ceilingを超えると
その段階を失敗にし、より長い段階を実行しません。promptと生成本文はreportへ保存しません。

## Semantic anchor cache

agentic contextの再prefill削減に向け、`SemanticAnchorCache`はconversation turn、tool call、
tool result、thinking境界のbackend state metadataを保持します。raw promptとtoken列は保存せず、
session/prefix SHA-256、token位置、opaque state handle、accounted bytesだけを管理します。

cacheはentry数とstate bytesのhard upper boundを持つthread-safe LRUです。`put`、`resize`、
`clear`はevictionされたanchorを返すため、backend側は対応するKV/recurrent stateを確実に解放できます。
`deepest_reusable`はcontext編集後も一致するprefix境界のうち最深のanchorだけを返します。

これはFreeTokenのsemantic-aware cachingを参考にしたbackend-neutralな独立実装であり、
FreeToken、CUDA、NVIDIA runtimeへの依存はありません。

`SemanticStateCoordinator`はdaemonの`RuntimeService`とbackend-owned stateを接続します。
backendはcapture時にopaque handleとaccounted bytesを返し、restore/releaseを実装します。
stale handleは復元失敗時にcacheから除去され、release失敗は最大1024件のretry queueへ移されます。
runtime snapshotにはcapacity、resident bytes、capture、hit/miss、eviction、restore/release failureを
出力します。現行OpenAI HTTP proxyはKV handle APIを持たないため既定ではdisabledです。

`ElasticMemoryController`はsemantic cacheの通常容量を基準に、memory pressureがWarningなら1/2、
Criticalなら1/8へ縮小し、Normalへ戻ると容量を復元します。active scheduler reservationがある間は
変更をpendingにし、safe pointでだけ適用します。現在は明示的pressure入力まで実装済みで、
macOS memory pressure notificationとの継続接続が次の作業です。

## Model Optimization Compiler dry-run

O0の`vllm-apple-optimize plan`はmodelを変更せず、INT8/INT4候補のoutput size、required disk、
peak memoryを見積もります。original modelと重なるpath、既存artifact path、resource budget超過を
検出します。MLX adapterが対応する環境とmodelではcandidateを実行可能として提示し、それ以外は
理由付きで拒否します。

所要時間を見積もる場合のみ、明示的にbounded I/O profileを取得します。読み書きsampleは既定で
各64 MiBに制限され、一時fileは`fsync`後に削除されます。`plan`自体がbenchmarkを暗黙実行する
ことはありません。

O1のadapter capability検出は外部packageをimport・実行せず、package metadata、platform、modelの
format/dtypeを確認します。現在のMLX exporterはApple Silicon、safetensors、FP16/BF16/FP32、
MLX/MLX-LM 0.26.xから0.31.x、4/8 bit affine quantizationだけを実行可能とします。
未検証versionは推測で実行せず、structured errorを返します。

変換workerの基盤は別process、最大64 KiBのbounded stdout/stderr、process-group cancel、
private sibling workspaceを提供します。成功時だけregular file、file数、byte数、directory深度を
streaming検証し、全fileとdirectoryを`fsync`してからartifact directoryをatomic renameします。
失敗またはcancel時はoutputを公開しません。MLX exporterは固定argument列でworkerへ接続され、
shell、remote upload、remote code trustを利用しません。source fingerprintはweightを含む全regular
fileを8 MiBずつstreaming SHA-256して生成するため、model全体をmemoryへ読み込みません。
safetensorsのdtypeがconfigにない場合は、最大16 MiBに制限したheaderだけを読み、weightをloadせず
判定します。

persistent checkpoint v1はplan、source fingerprint、output、command/environment fingerprint、
output byte予算へbindingされます。checkpoint fileはplan IDをSHA-256 filenameへ変換し、private
directoryへ0600でatomic保存します。`converted`だけがworkspace検証から再開でき、それ以前の
stageは変換を最初から再実行します。resumeは誤操作を避けるため明示的に有効化します。

workerはcheckpoint fileごとのcross-process `flock`を保持します。同じplanの二重実行は拒否され、
processが異常終了した場合はkernelがlockを自動解放します。`resume=True`ではbindingを再検証し、
`converted` workspaceからcommandを再実行せずvalidation/promotionへ進みます。promotionとcompleted
checkpoint更新の間で停止した場合も、公開済みartifactを再検証して状態をreconcileします。
成功時はartifact tree hash、size、file数、elapsed milliseconds、peak child RSSを記録した0600の
sidecar manifestと、Mac appからdecodeしやすいversioned export report JSONを生成します。

```bash
python3 -m vllm_apple.optimizer.cli capabilities /path/to/model

python3 -m vllm_apple.optimizer.cli profile /path/to/model \
  --workspace /path/to/temporary-workspace \
  --sample-mib 64 > optimizer-profile.json

python3 -m vllm_apple.optimizer.cli plan /path/to/model \
  --output /path/to/immutable-artifact \
  --objective memory \
  --max-memory-gb 16 \
  --max-disk-gb 40 \
  --max-duration-seconds 3600 \
  --performance-profile optimizer-profile.json \
  --license apache-2.0

# Dry-run: invocation JSONを表示するだけでoutput/checkpointを作成しない
python3 -m vllm_apple.optimizer.cli export /path/to/model \
  --output /path/to/immutable-artifact \
  --checkpoint-root /path/to/existing-private-parent/checkpoints \
  --plan-id my-plan --bits 4 --group-size 64 --max-output-gb 20

# 明示実行。中断後の安全な再開には --resume も付ける
python3 -m vllm_apple.optimizer.cli export /path/to/model \
  --output /path/to/immutable-artifact \
  --checkpoint-root /path/to/existing-private-parent/checkpoints \
  --plan-id my-plan --bits 4 --group-size 64 --max-output-gb 20 --execute

# baselineとcandidateは別processで順番に評価し、同時にmemoryへ載せない
python3 -m vllm_apple.optimizer.cli evaluate /path/to/baseline \
  --dataset docs/evaluation/smoke-multilingual-v1.jsonl \
  --output /path/to/baseline-evaluation.json
python3 -m vllm_apple.optimizer.cli evaluate /path/to/candidate \
  --dataset docs/evaluation/smoke-multilingual-v1.jsonl \
  --output /path/to/candidate-evaluation.json
python3 -m vllm_apple.optimizer.cli quality-gate \
  --baseline /path/to/baseline-evaluation.json \
  --candidate /path/to/candidate-evaluation.json \
  --max-perplexity-regression 0.10 \
  --output /path/to/quality-gate.json

# 固定seed・greedy生成を別processで比較する
python3 -m vllm_apple.optimizer.cli generate-evaluate /path/to/baseline \
  --dataset docs/evaluation/generation-smoke-multilingual-v1.jsonl \
  --max-samples 6 --max-new-tokens 16 \
  --output /path/to/baseline-generation.json
python3 -m vllm_apple.optimizer.cli generate-evaluate /path/to/candidate \
  --dataset docs/evaluation/generation-smoke-multilingual-v1.jsonl \
  --max-samples 6 --max-new-tokens 16 \
  --output /path/to/candidate-generation.json
python3 -m vllm_apple.optimizer.cli generation-quality-gate \
  --baseline /path/to/baseline-generation.json \
  --candidate /path/to/candidate-generation.json \
  --min-token-agreement 0.70 --max-expectation-regression 0 \
  --output /path/to/generation-quality-gate.json

# 用途と言語は繰り返し指定でき、選択条件もfingerprintへ含まれる
python3 -m vllm_apple.optimizer.cli generate-evaluate /path/to/model \
  --dataset docs/evaluation/task-suite-multilingual-v1.jsonl \
  --chat-template --max-prompt-tokens 4096 \
  --domain code --domain mathematics --language ja \
  --output /path/to/ja-code-math.json
```

durationは同一hardwareのprofile実測値がない限り推測せず`null`とし、warningへ理由を記録します。
CLI failureは`code`、localizableな`message_key`、`recoverability`を持つJSONとしてstderrへ返します。
perplexity runnerはJSONLを1行ずつ処理し、dataset 16 MiB、1行72 KiB、sample 4096、
sample当たり4096 tokens、合計100万tokensをhard upper boundとします。gateは同一dataset
fingerprint、slice、token数を要求し、未評価のgeneration、long-context、code、math、safetyを
reportへ明示します。
generation runnerは最大64 prompts、各256 tokensとし、生成文やpromptをreportへ保存しません。
比較に必要な上限付きtoken ID、SHA-256 fingerprint、期待文字列の一致scoreだけを保存します。
期待条件はdataset側で`contains`または`prefix`を明示します。短い数値正答などは`prefix`を使い、
`8`が`18`へ偶然含まれるような誤合格を防ぎます。
instruction modelでは`--chat-template`を明示し、tokenizer固有template適用後の入力token数を
各sampleへ記録します。prompt上限とmodel context上限は生成開始前に検証され、baselineとcandidateで
prompt形式、token budget、実入力token数が異なるreportは比較されません。

## 開発時の確認

再現可能な開発環境とCIの詳細は[DEVELOPMENT.md](docs/DEVELOPMENT.md)を参照してください。
runtime errorのrecoverabilityとprivate crash diagnosticは
[RUNTIME-FAILURES.md](docs/RUNTIME-FAILURES.md)を参照してください。
Unified Memory、allocator、KV cacheのsource-aware metricsは
[MEMORY-TELEMETRY.md](docs/MEMORY-TELEMETRY.md)を参照してください。

```bash
make bootstrap
make check
```
