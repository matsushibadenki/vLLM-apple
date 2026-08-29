# MLX real-model qualification

## 2026-08-29 — Gemma 2 2B IT MLX 4-bit

`models/gemma-2-2b-it-4bit`をMLX LM 0.26.2のOpenAI互換serverで起動し、
non-stream/stream mixed chatを4並列で30分間実行した。生成本文は保存せず、
promotion gateには固定長digestのみを残した。

| Metric | Result |
|---|---:|
| Model load | 2.765 seconds |
| Stability window | 1,800.486 seconds |
| Requests | 12,722 / 12,722 succeeded |
| Throughput | 7.066 requests/second |
| Mean latency | 566.015 ms |
| Maximum latency | 1,697.433 ms |
| RSS baseline | 2,118,205,440 bytes |
| RSS peak | 2,118,385,664 bytes |
| RSS peak growth | 180,224 bytes |
| RSS final | 1,827,405,824 bytes |
| Reported RSS growth | 0 bytes |
| Shutdown | clean |

MLX LM serverはrequest単位のseedを受け付けないため、promotion gateは`greedy_only`とした。
同一requestの反復、non-stream/stream digest一致、SSE `[DONE]`を検査し、すべて成功した。
この結果は4,096 token設定での短い8-token応答に対する安定性を示す。長context時のKV cache容量、
thermal throttling、sampled decodingの再現性は未評価であり、次の昇格条件として残す。

### Allocator telemetry follow-up

backend-local telemetry wrapperを使う10秒qualificationでは98/98 requestが成功し、
MLX allocatorとbounded KV cache走査、OS available memoryから96,013 tokens相当の安全側capacityを算出した。
設定値4,096 tokensに対する判定は`sufficient`で、context再評価は1回だけ実行された。
このcapacityは空きUnified Memoryから2 GiB以上をreserveした上限推定であり、model固有KV allocationの
長context実測値ではない。次は段階的なcontext probeで推定誤差を校正する。

### Long-context calibration

MLX wrapperへbounded `/tokenize`互換面を追加し、生成本文とtoken IDを保存せずに、
各段階のretrieval品質、TTFT、decode速度、RSS、実KV allocationを測定した。
MLX LM 0.26.2のstream usageは共通prefix再利用時に未処理suffix長を返すため、
full-context長には同一promptを処理したtokenizer endpointの値を採用する。

| Target | Actual prompt | Retrieval | TTFT | Decode | KV bytes | KV bytes / actual token |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 111 | 100% | 239.741 ms | 56.831 tok/s | 1,056,768 | 9,520 |
| 1,024 | 983 | 100% | 1,964.848 ms | 49.919 tok/s | 8,200,192 | 8,342 |
| 4,096 | 3,927 | 100% | 6,967.168 ms | 42.516 tok/s | 32,317,440 | 8,230 |

定常領域ではKV allocationは約8.2 KiB/tokenで、config由来の安全側推定106,496 bytes/tokenより
大幅に小さかった。8,192 targetはHTTP 200の後に生成tokenを返さず`first_token_missing`で停止した。
メモリ上限超過ではなくmodel有効context境界の可能性が高いため、成功扱いにはせず、次工程で
model metadataとchat template overheadを含む事前上限検出、および専用error codeへ分離する。

この境界はmodel metadataの`max_position_embeddings=8192`に対して、8,192 token入力目標と
32 tokenの生成予約を同時に収められないことが原因だった。long-context adapterはローカルmodelの
上限を自動検出し、外部serverでは`--model-max-context`で明示できる。backend requestを送る前に
`target + maximum_output_tokens`を検査し、超過時は`model_context_limit_exceeded`としてfail-fastする。

成功した3段階はbounded calibration reportとして`docs/evaluation`へ保存した。context推奨への採用時は
model identity一致、3件以上の成功、4,096 token target到達、token/KV量の単調増加、retrieval 100%を
検査する。実測最大9,521 bytes/tokenへ25%の余裕を加えた11,902 bytes/tokenを採用し、推奨contextは
実際に成功した3,927 tokensを越えない。このため、理論値106,496 bytes/tokenより現実的にしながら、
未観測contextへの外挿は行わない。

`long-context-evaluate --output <path>`はreportを一時fileへ書き、`fsync`後にatomic replaceする。
新規directoryは0700、reportは0600とし、既存directoryがcurrent user所有かつprivateでない場合は拒否する。
校正readerもsymlinkやshared fileを拒否し、model IDに加えてhardware fingerprintの完全一致を要求する。
`context`コマンドは現在のApple chip fingerprintを既定値として照合し、必要な場合だけ
`--calibration-hardware-fingerprint`で期待値を明示できる。

`long-context-evaluate --save`はApplication Support内をhardware fingerprintとmodel IDの
SHA-256短縮keyで分離し、timestampとevaluation IDを含む名前で保存する。`context --auto-calibration`は
同じ領域を最大256 entryまで新しい順に調べ、private file、schema、model、hardware、3-stage/4K、
単調性、retrieval gateをすべて通過した最新reportだけを採用する。新しいreportが破損している場合は
検証済みの古い候補へ戻り、適合候補がなければ理論値へ黙って切り替えず明示的に失敗する。

校正reportにはbackend identityも含め、保存directoryもbackend単位で分離する。現在のdaemonは
vLLM-Metal backendを管理するため、起動時の自動校正には`vllm_metal`で測定されたreportだけを採用する。
`mlx_lm`の本実測値をvLLM-Metalへ流用することはない。適合reportがない場合は従来の理論値を維持し、
不正候補がある場合は`invalid`、採用時は`applied`として、evaluation ID、校正bytes/token、
最大実測context、sample数、安全余裕率をruntime snapshotの`kv_calibration`へ公開する。

Swift SDKは`KVCalibrationProvenance`と`KVCalibrationStatus`としてruntime snapshotをtyped decodeする。
HTTP/UDS clientの双方から取得でき、Mac chat sampleは適用、無効、未検出、不正、未設定を診断欄へ表示する。
適用時は校正bytes/tokenと検証済み最大contextを示し、文言は英語、日本語、简体中文に対応する。
