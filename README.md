# vLLM-Apple Runtime

Apple Silicon向けの、メモリ安定性を重視したAI runtime control planeです。

現在はPhase 1として、hardware/memory検出、安全なcontext計算、runtime profile、
メモリ予約scheduler、headless daemon、versioned local APIを実装しています。

## 安定性の基本方針

- context計算ではphysical memoryとcurrent available memoryの小さい方を採用
- OS reserve、workspace、緊急headroomを推論用budgetから除外
- request bodyを4MiBへ制限
- server threadを既定32本へ制限し、過負荷時は503で早期拒否
- profileは一時fileへ書き、`fsync`後にatomic replace

Macアプリ用Swift Packageは `sdk/swift` にあります。SwiftUIやAppKitへ依存せず、
Foundation、`async/await`、`AsyncThrowingStream` を公開interfaceにしています。

```bash
python3 -m vllm_apple hardware
python3 -m vllm_apple doctor
python3 -m vllm_apple context --model-memory-gb 8 --kv-bytes-per-token 524288
python3 -m vllm_apple serve
python3 -m vllm_apple serve mlx-community/your-model --max-model-len 8192
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

## 開発時の確認

```bash
python3 -m unittest discover -v
cd sdk/swift && swift test
```
