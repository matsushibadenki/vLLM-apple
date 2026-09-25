# llama.cpp comparison — 2026-09-25

Reviewed revision: [`f805c57a2d0b7cc171e599303ce2040f6e1bfe15`](https://github.com/ggml-org/llama.cpp/tree/f805c57a2d0b7cc171e599303ce2040f6e1bfe15).

## 日本語

- [Done] llama.cppのキャンセル時キュー回収を参考に、`PriorityScheduleQueue`の内部エントリ蓄積を修正した。従来は待機request辞書からだけ削除し、consumerが動かない間は内部heapにキャンセル済みtokenが残り続けていた。
- [Done] 待機requestがなくなった時点でheapを空にする。待機requestが残る場合はheap長が`2 × 待機数 + 64`を超えたときに有効なエントリだけで再構築し、キャンセルごとの全件走査を避ける。既存のlock、priority、FIFO sequence、claimの容量予約を保持する。
- [Next] 実運用の長時間キャンセル負荷で待ち時間を計測する。今回の検証はモデルを使わない回帰試験であり、推論速度の改善は主張しない。
- [Later] prompt類似度によるslot選択とdecode失敗時のbatch縮小を評価する。これらはbackendが所有するKV状態やdecode結果の契約に依存するため、control planeへ単独では移植しない。

## English

- [Done] Applied llama.cpp's cancellation cleanup principle to the priority queue. Cancelled heap entries previously accumulated without limit when no consumer drained the queue, despite the configured request capacity.
- [Done] Empty pending queues now release the heap immediately. Otherwise, rebuilding above `2 × pending + 64` entries bounds retained metadata while amortizing cleanup. Priority, FIFO order, locking and claimed-request capacity remain intact.
- [Next] Measure latency under sustained operational cancellation load. These model-free regression tests do not establish inference throughput gains.
- [Later] Evaluate prompt-similarity slot selection and decode batch reduction only with explicit backend state and retry contracts.

## 简体中文

- [Done] 参考llama.cpp在取消任务时清理队列的设计，修复优先队列内部条目无限累积的问题。此前没有消费者时，取消的token会持续留在堆中，不受请求容量限制。
- [Done] 没有待处理请求时立即清空堆；否则在条目数超过`2 × 待处理数 + 64`时重建，限制元数据占用并分摊清理成本。保留优先级、FIFO顺序、锁及已领取请求的容量预留。
- [Next] 在实际长期取消负载下测量延迟。本次为不加载模型的回归测试，不代表推理吞吐量提升。
- [Later] 在明确后端状态及重试契约后，评估基于提示词相似度的slot选择和decode批大小缩减。

## Evidence

Upstream [`server_queue::cleanup_pending_task`](https://github.com/ggml-org/llama.cpp/blob/f805c57a2d0b7cc171e599303ce2040f6e1bfe15/tools/server/server-queue.cpp)
removes cancelled tasks from pending, deferred and unhandled queues while holding
the queue lock. vLLM-Apple uses a priority heap, so the implementation adapts this
principle using batched compaction rather than copying upstream's deque removal.
[`server-context.cpp`](https://github.com/ggml-org/llama.cpp/blob/f805c57a2d0b7cc171e599303ce2040f6e1bfe15/tools/server/server-context.cpp)
also contains the slot-similarity selection and decode batch reduction considered
for future backend work.

Reproduction: a capacity-2 queue, no consumer, and 10,000 enqueue/cancel pairs.
Both versions report zero queued requests; the original retains 10,000 heap
entries, while the modified version retains zero. A second regression keeps live
requests ahead of 2,000 cancelled background requests, checks bounded heap storage,
and verifies dispatch priority and FIFO order after compaction.

Validation: 29 tests passed across `test_scheduler_queue`, `test_scheduler`,
`test_scheduling_observability`, and `test_scheduling_preference`. Coverage includes
claim restoration, capacity accounting, cancelled claims, and the existing
cancellation-during-dispatch reservation cleanup test. Ruff and `git diff --check`
also passed. No backend dependency or public snapshot schema was changed.
