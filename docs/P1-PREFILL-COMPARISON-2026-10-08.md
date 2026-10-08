# Prefill comparison and diagnostic overhead — 2026-10-08

## 日本語

空のPython listを渡すprefill呼出は、元methodへそのまま転送し、clock・CPU計測・lock・統計更新を省略する。非空呼出と例外の伝播は保持する。対象はP1 profileのprefill計測のみ。空呼出を平均値に含めず、実prefillの統計を取得する。

合成比較は20,000空呼出×9回、旧計測と新経路を交互に実行。同一processの中央値はwall 19.432→2.024 ms、CPU 19.430→2.025 ms（約89.6%削減）。これは計測処理の負荷であり、推論速度や消費電力の改善率ではない。変更はKeep。

実モデルはM4、Gemma-2-2b-it-4bit、compact、decode/prompt concurrency 2。新workerで512→256、256→512、512→256を実行。各回の長文12件・短文30件、全252件で品質・SLO合格。全6回で正常停止、runtime/model identity不変、比較間identity一致を確認。warmupはこの252件に含めない。

| prefill tokens | 長文E2E：3回の平均値の中央値 | 長文平均値の範囲 | 短文E2E：3回の平均値の中央値 | allocator peak |
| --- | ---: | ---: | ---: | ---: |
| 512 | 704.313 ms | 615.055–1134.522 ms | 472.844 ms | 約2171.32 MiB |
| 256 | 643.218 ms | 625.440–654.100 ms | 533.458 ms | 約2090.33 MiB |

256のallocator peakは約81 MiB低いが、長文の範囲は重なり、短文中央値は悪化。既定512を維持し、256を標準へ昇格しない。RSS減少、TPOT p95、一般品質、ワット値、P3性能優位は未認定。P1の30分・8時間試験やr11根因修正を認定しない。次は非空prefill内の同期・cache解放とRSSの追加診断。長時間試験は09:00 JST開始。

## English

Keep the empty-prefill diagnostic fast path: it forwards the original call while avoiding clocks, CPU sampling, locks and counters for empty Python lists. Nonempty calls and exceptions retain their behavior. This applies only to P1 profile instrumentation. A synthetic alternating nine-trial comparison of 20,000 empty calls reduced median diagnostic wall time from 19.432 to 2.024 ms and CPU time from 19.430 to 2.025 ms, about 89.6%. This is not inference throughput or energy savings.

Six fresh M4/Gemma-2-2b-it-4bit workers compared compact prefill 512 and 256 at concurrency two. All 252 measured arithmetic requests passed quality and SLO; all workers shut down cleanly with unchanged and matching identities. The table shows medians of three run means, not per-request medians. Prefill 256 reduced allocator peak by about 81 MiB but did not consistently improve latency. Keep default 512; no performance or stability promotion. RSS savings, general quality, TPOT p95 and watts remain unverified. Diagnose nonempty prefill synchronization/cache release next; long campaigns start at 09:00 JST.

## 简体中文

保留空prefill计时快速路径：空Python list仍调用原method，但省略clock、CPU采样、lock和统计更新；非空调用及异常传播保持不变。仅用于P1 profile。20,000次空调用、9轮交替比较的wall中位数从19.432降至2.024 ms，CPU从19.430降至2.025 ms，约减少89.6%。这不是推理速度或功耗改善率。

M4/Gemma-2-2b-it-4bit、compact、并发2，以6个全新worker比较prefill 512及256。252个算术请求全部通过质量和SLO，正常停止及identity一致均通过。表格为3轮平均值的中位数，并非单请求中位数。256的allocator峰值约低81 MiB，但延迟没有一致改善，保留默认512，不晋升性能或稳定性资格。RSS、一般质量、TPOT p95、瓦数未验证。下一步调查非空prefill内同步及cache释放；长时间测试09:00 JST开始。

## Evidence

- [Raw reports and summary](evaluation/p1-prefill-comparison-m4-2026-10-08/summary.json)
- [Original synthetic baseline](evaluation/empty-prefill-timing-before-2026-10-08.json)
- [Alternating synthetic comparison](evaluation/empty-prefill-timing-after-2026-10-08.json)
- `scripts/compare_p1_prefill.py` preserves raw reports and hashes; rejects mixed source/model identity or incomplete/configuration-mismatched trials. It never promotes a candidate.

Validation / 検証 / 验证：対象8 tests、全回帰1478 tests（11 skip）、Ruff・diff check成功。[Receipt](evaluation/p1-prefill-validation-2026-10-08.json). Alternate venv: 1473 tests, 27 dependency skips.
