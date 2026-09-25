# vllm-mlx comparison — 2026-09-25

Reviewed upstream revision: [`37a16c76bb21c3bd94350db1e48ed71220aaa5d4`](https://github.com/waybarrios/vllm-mlx/tree/37a16c76bb21c3bd94350db1e48ed71220aaa5d4).

## 日本語

- [Done] `memory_cache.py`のshape／dtypeを優先する配列サイズ計測を参考に、MLX telemetryとsemantic snapshot admissionの共通関数を改善。メタデータが利用できる配列では`nbytes`に触れず、その他の配列は従来の計測方法を維持する。
- [Done] 入れ子の探索をiterator方式へ変更。配列の重複計上と循環を防ぎ、scalarや重複参照も探索上限に含める。上限に達した計測は未完了として扱い、snapshotの登録を拒否して解放する。
- [Next] 実MLXでのメタデータ計測と通常のキャッシュ構造を検証する。今回の標準venvにMLXはなく、GPU実測による性能改善は主張しない。
- [Later] upstreamのSSD promotion前のメモリ予約、continuous batching、prefix trieは別途評価する。現在のcontrol plane／model-owner境界や既存schedulerとの統合が必要なため、今回の変更には含めない。

## English

- [Done] Inspired by upstream metadata-first array accounting, the shared telemetry/snapshot traversal now prefers shape and dtype size, with an `nbytes` fallback for other array types.
- [Done] Iterator traversal bounds work and temporary storage even for wide containers, scalar metadata, aliases and cycles. Incomplete snapshots are rejected and released.
- [Next] Validate with real MLX caches; MLX is absent from the default environment. No GPU throughput improvement has been measured.
- [Later] Evaluate SSD promotion reservations, continuous batching and prefix tries against the existing scheduler and model-owner boundaries.

## 简体中文

- [Done] 参考上游优先读取元数据的设计，共用的遥测及快照计量函数优先使用shape和dtype大小，其他数组继续使用`nbytes`。
- [Done] 采用迭代器遍历，限制宽容器、标量及重复引用的处理量和临时内存；避免循环与重复计量。计量未完成时拒绝保存快照并释放资源。
- [Next] 使用真实MLX缓存验证。默认环境未安装MLX，尚未测量GPU吞吐量变化。
- [Later] 结合现有调度器与模型所有者边界，评估SSD提升前内存预留、连续批处理和前缀树。

## Evidence and limits

Reference implementations inspected: [memory accounting](https://github.com/waybarrios/vllm-mlx/blob/37a16c76bb21c3bd94350db1e48ed71220aaa5d4/vllm_mlx/memory_cache.py), [SSD cache](https://github.com/waybarrios/vllm-mlx/blob/37a16c76bb21c3bd94350db1e48ed71220aaa5d4/vllm_mlx/ssd_cache.py), [prefix cache](https://github.com/waybarrios/vllm-mlx/blob/37a16c76bb21c3bd94350db1e48ed71220aaa5d4/vllm_mlx/prefix_cache.py), and [vision cache](https://github.com/waybarrios/vllm-mlx/blob/37a16c76bb21c3bd94350db1e48ed71220aaa5d4/vllm_mlx/vision_embedding_cache.py).

Regression coverage includes metadata-only lazy arrays, packed/scales nested states,
scalar and empty arrays, fallback accounting, cycles, exact traversal limits, wide
containers, repeated references, and rejected snapshot cleanup. Accounting deduplicates
array object identities; distinct views sharing a storage allocation remain separate.

A Python 3.12 `tracemalloc` comparison against the pre-change function used a
preallocated list of 1,000,000 `None` values and the default 4,096-entry budget:

| Version | Peak temporary Python allocation | Result |
| --- | ---: | --- |
| Before | 8,000,248 bytes | `(0, True)` after scanning the whole list |
| After | 416 bytes | `(0, False)` at the traversal limit |

This synthetic check measures traversal allocations only, not model inference speed
or total process memory. The stricter budget can reject unusually large snapshots
that the previous traversal accepted.
