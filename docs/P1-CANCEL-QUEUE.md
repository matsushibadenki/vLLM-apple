# P1 cancellation under response backpressure — 2026-10-28

## 日本語

🟢 [Done] Gemma 2互換経路の満杯応答キューに対するキャンセルを修正した。
以前は`cancel_request_state()`が共有registry lockを保持したまま`put(None)`で待機し、
consumerが読み取らない場合に他要求の登録・キャンセルも停止し得た。

producerはwriter lockを保持したまま空きを待たず、キャンセルは同じlockで
未配信応答を破棄し、終了通知を一つ置く。待機中のproducerをEventで起こし、
以後の書き込みを破棄する。modelの停止は従来どおり`context.stop()`へ委譲し、
GPU／KVの解放完了をこの通知だけで認定しない。通常時のFIFOとP1の1,024件上限は維持。

CPU回帰は、満杯の1,024件キューとproducer待機の同時条件、別要求のregistry操作、
二重キャンセル、遅れて到着するtokenの破棄、通常時のFIFOを検証する。
飽和は合成CPU試験であり、実モデルが同じ飽和条件に到達した証拠ではない。

検証：Ruff 0.17.2全件成功、compileall成功、全体回帰再実行1,591件成功（46 skip）。
初回全体回帰は既存`test_managed_engine`の満杯拒否試験が1件失敗した。
同ファイルの単独2件と全体再実行は成功。初回失敗を今回の修正で解決したとは扱わない。

最初の実HTTP試行はHomebrew環境のMLX 0.32.2を、P1の0.32.1固定gateが拒否した。
既定環境とguardは変更せず、配置済み`.venv-vllm-metal-031`を明示して別試行した。
これはGemma 2／MLX-LM互換経路の検証であり、vLLM-Metal 0.31.0の昇格ではない。

[固定版実HTTP report](evaluation/p1-cancel-queue-pinned-2026-10-28.json)では品質45/45、
短文の英語・日本語・简体中文各10/10品質・SLO合格。active cancelはHTTP 202後
2.559 msで終了を観測した。queued cancel、100 ms timeout、slow consumer、half-close、
HTTP接続上限、profile拒否、切断後の正答、identity不変、正常shutdownも成立した。
ただしwarmup 1/3がTTFT 26.571秒とE2E基準超過、長文1/12がTTFT 10.662秒で基準超過し、
**総合`passed=false`**。速度改善・SLO認定はしない。cold compileやhost負荷を根因と推定しない。
[版不一致の拒否report](evaluation/p1-cancel-queue-version-rejected-2026-10-28.json)も保持する。

🟠 [Next] 09:00 JST開始の固定identity長時間campaignで30分→合格時8時間を認定する。
短時間のHTTP成功やCPU回帰だけでSLO、性能向上、OOM、長時間安定性を認定しない。

## English

🟢 [Done] Saturated response queues can now be cancelled without waiting for a
consumer while holding the shared request registry lock. Cancellation discards
undelivered responses, wakes a blocked producer and preserves one terminal marker.
Normal FIFO and the existing P1 queue bound remain unchanged. Model stop and GPU/KV
reclamation retain their existing owner contracts. Saturation is tested with CPU
fixtures; it is not a real-model saturation or performance qualification.

Full regression retry: 1,591 tests, 46 skips; Ruff 0.17.2 and compileall pass.
The initial full run had one unrelated managed-engine queue-test failure; its
isolated retry and the full retry passed. Long-run qualification remains open.
Pinned real-model HTTP checks passed all 45 quality cases, cancellation, timeout,
slow-consumer recovery and clean shutdown. One warmup and one long-context request
missed SLOs; the overall result is false. No performance improvement is certified.

## 简体中文

🟢 [Done] 已修复响应队列已满时的取消等待：不再持有共享registry锁等待consumer。
取消时丢弃尚未发送的响应，唤醒等待中的producer，保留一个结束标记并拒绝后续写入。
正常FIFO、P1现有队列上限及model-owner停止／GPU和KV回收契约保持不变。
队列饱和由CPU合成测试验证，不代表真实模型饱和或性能认证。

完整回归重跑1,591项成功（46项skip），Ruff 0.17.2及compileall通过。
初次完整回归中，既有managed-engine队列测试失败1项；单独重试及完整重跑均通过。
🟠 [Next] 固定identity的30分钟／8小时长期认证仍未完成。
固定环境的真实HTTP测试45/45质量通过，取消、timeout、慢consumer、恢复及正常停止通过。
但warmup及长上下文各1次超过SLO，综合结果为false，不认证性能提升。

## Reproduction

```sh
.venv/bin/python -m unittest discover -s tests -p test_mlx_gemma2_compat.py
.venv/bin/python -m unittest discover -s tests
uv run --no-project --with ruff==0.17.2 ruff check vllm_apple tests
.venv/bin/python -m compileall -q vllm_apple tests
VLLM_APPLE_P1_EFFICIENCY=compact PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python scripts/qualify_gemma2_batch_mask.py --python .venv-vllm-metal-031/bin/python --model models/gemma-2-2b-it-4bit --output /tmp/p1-cancel-queue-pinned-2026-10-28.json --port 19221 --sustained-requests 30 --long-requests 12 --duration-seconds 0 --decode-concurrency 2 --prompt-concurrency 2 --prefill-step-size 512 --long-concurrency 1 --p1-profile
```
