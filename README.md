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
python3 -m vllm_apple context --model-memory-gb 8 --kv-bytes-per-token 524288
python3 -m vllm_apple serve
```

サーバーは明示指定しない限り `127.0.0.1` のみにbindします。推論backendが未設定の
場合もhealth、hardware、runtime profile APIは利用できます。

## 開発時の確認

```bash
python3 -m unittest discover -v
cd sdk/swift && swift test
```
