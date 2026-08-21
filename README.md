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
```

request failureまたはRSS上限超過ではexit code 1、設定・接続準備のerrorではexit code 2を返します。
実modelの長時間判定では、model load後のdaemon PIDを`--pid`へ指定してください。

## 開発時の確認

```bash
python3 -m unittest discover -v
cd sdk/swift && swift test
cd ../../samples/VLLMAppleChat && swift build
```
