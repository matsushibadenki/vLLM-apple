# Kapitsa execution lifecycle ideas

2026-11-01。参照commit：`2d374d5e85bcc87708527102dee3ac7d40119dfe`（ローカルcheckout、変更なし）。[Apache 2.0 license](../../kapitsa/LICENSE)。設計を参考にPythonで独立実装し、Kapitsaのコード転載・ビルド・実行依存の追加は行わない。

## 日本語

🟢 [Done] [ExecutionSession](../../kapitsa/src/engine/execution_session.cpp)の「終了要求は一度、joinは再確認可能」という契約を、vLLM-appleのmodel-owner終了処理へ取り入れた。`ThreadAffineInferenceEngine.close()`は満杯の待機キューを排出し、受付との競合をlockで防ぐ。待機要求はcancelで完了させ、実行中のdelegateはownerが呼び出しから戻るまで保持する。停止待ちがtimeoutなら`False`を返し、再呼び出しで実際のthread終了を確認する。

起動timeout後にfactoryが戻った場合も、owner自身がdelegateをcloseする。metadata取得中の起動失敗も同じcleanup経路を通る。owner callbackからのcloseは停止要求だけを行い、自分自身をjoinしない。`RuntimeService.close()`も未完了を成功としてcacheせず、同時の終了呼び出しは`False`を返す。正常終了後はcloseを繰り返さない。

| 状態 | 参考設計 | vLLM-appleへの判断 |
|---|---|---|
| 🟢 [Done] | one-shot停止と繰り返せるjoin | 満杯キュー、起動timeout、同時close、owner threadの解放を修正 |
| 🟢 [Done] | [Mac共通GPU lock](../../kapitsa/src/bench/system.mm) | P1 campaign、Gemma 2単独qualification、Metal trace入口へ共通flockを接続。親子runnerはfdを継承。対象外runnerへの展開はNext |
| 🔴 [Later] | [SSD復元待ちと再prefill費用の比較](../../kapitsa/src/engine/disk_wait.h) | 永続KV artifactと復元契約が整った段階で採用候補。未計測の速度を実測として扱わない |
| 🔴 [Later] | [cache namespaceとstate format](../../kapitsa/src/engine/engine_disk.cpp) | KV／recurrent／draftのlayout変更で永続cacheを無効化する。現backendのcacheを互換artifactとみなさない |
| 🔴 [Later] | [streamed expert cache](../../kapitsa/src/model/expert_store.h) | Strataで需要read優先とprefillによるcache汚染回避を比較候補にする。native Metal handshakeは別実装・実機検証が必要 |

終了処理の回帰はCPU fixtureで、満杯キュー、未完了closeの再試行、待機要求のcancel、起動timeout後の遅いfactory、metadata失敗、owner callback、同時service closeを検証する。従来のキュー試験にあった10 ms sleepをenqueue完了Eventと明示cancelへ置き換え、threadの開始速度への依存を除いた。HTTP統合も模擬encoderで検証する。GPU kernel、推論品質、速度、8時間の安定性はこの変更の認定範囲に含まない。戻らないnative callの強制終了は行わず、process隔離が必要な課題として残す。

## English

🟢 [Done] Applied Kapitsa's separation of one-shot stop requests and repeatable joins to the model-owner lifecycle. Shutdown rejects queued work without waiting for capacity, keeps active delegates on their owner thread, cleans up late startup and metadata failures, and reports success only after the worker exits. Service shutdown retries incomplete closes and caches completed shutdown only. Queue tests synchronize on admission rather than a fixed sleep. Validation covers CPU fixtures and mock-encoder HTTP integration, without certifying GPU behavior or inference speed.

🟢 [Done] P1 campaigns, standalone Gemma 2 qualification and Metal trace collection now share a cooperative GPU lock. 🟠 [Next] Extend it to other benchmark entrypoints. 🔴 [Later] Compare SSD restore waits against measured prefill cost, version persistent state namespaces, and evaluate demand-first expert reads with Strata. A native call that never returns still requires process isolation.

## 简体中文

🟢 [Done] 将Kapitsa的一次停止请求与可重复join契约应用到model-owner生命周期。停止时不等待队列空位，取消待处理请求；正在执行的delegate仍由owner thread释放。起动timeout后迟到的factory、metadata失败、owner callback及并发close均有回归测试。Service仅缓存已完成的停止结果，未完成时允许重试。队列测试使用入队Event代替固定sleep。验证范围为CPU fixture及模拟encoder的HTTP集成，不认证GPU行为或推理性能。

🟢 [Done] P1 campaign、Gemma 2单独认证及Metal trace已接入共用GPU协作锁。🟠 [Next] 扩展到其他benchmark入口。🔴 [Later] 比较SSD恢复等待与实测prefill成本，为持久状态namespace增加版本约束，并评估Strata中需求读取优先。无法返回的native调用仍需要process隔离。

## GPU benchmark exclusion

🟢 [Done] `BenchmarkGpuLock`を次の入口へ接続した。CLI引数の検証後、model読込・出力directory作成前に取得し、処理とcleanupが終わるまで保持する。

- `scripts/run_p1_stability.py`：30分と8時間の間も保持し、qualification子processへ`pass_fds`と環境でdescriptorを渡す。
- `scripts/qualify_gemma2_batch_mask.py`：単独実行でも取得。親から受け取ったdescriptorは同じinodeであることを検証して使用する。
- `experiments/p3_mlx/profile.py`：trace記録からexportまで保持。記録processは既存`run_trial`の所有process groupでtimeout／中断時も回収する。
- 2026-11-02追加：`compare_p1_prefill.py`、`compare_p1_cache_entries.py`、`compare_p1_efficiency.py`は全比較・全反復を同じlockで囲む。試行間の割り込みも拒否し、子qualificationには`run_trial`経由でfdを継承する。`campaign.json`へlock identityと実行順序、自動昇格なしを記録する。timeoutでは失敗receiptを残して以降の試行を止める。
- 2026-11-02追加：`probe_p1_efficiency_resources.py`も3候補の資源観測全体を囲み、観測reportへlock identityを記録する。

2026-11-02の全回帰は1,615件成功（46 skip、42.364秒）。3campaignの21試行をCPU fixtureで動かし、実child processでfd継承と同一inodeを確認した。候補順序固定、試行間の別lock拒否、timeout後の後続停止・失敗receipt・解放も確認。7入口の起動前拒否と`run_trial`の所有process group回収を回帰し、Ruff 0.17.2／compileall／diff check成功。fixtureのpassed値は推論結果ではない。

[固定版Gemma 2の実HTTP report](evaluation/p1-campaign-lock-smoke-2026-11-02-r1.json)は45/45品質・SLO合格（warmup3、長文12、短文30、短文は英語・日本語・简体中文各10）。active／queued cancel、100 ms timeout、slow consumer、half-close、profile拒否、HTTP接続上限、切断後の正常応答を確認し、runtime／model identity不変・正常shutdownも合格。active cancelはHTTP 202後0.750 ms、timeout応答完了は102.246 ms。共通lockのpath／inodeをreportに記録した。[試験前観測](evaluation/p1-campaign-host-2026-11-02-r1.json)はBattery／automatic、nominal thermal、メモリ圧迫normal。前後の独立反復比較ではなく、過去AC測定への速度差帰属、以前のSLO超過の根因確定、長時間認定には使わない。

実機再現コマンド（新しいoutput名を使う）：

```sh
VLLM_APPLE_P1_EFFICIENCY=compact PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python scripts/qualify_gemma2_batch_mask.py --python .venv-vllm-metal-031/bin/python --model models/gemma-2-2b-it-4bit --output /tmp/p1-campaign-lock-smoke-2026-11-02-r1.json --port 19221 --sustained-requests 30 --long-requests 12 --duration-seconds 0 --decode-concurrency 2 --prompt-concurrency 2 --prefill-step-size 512 --long-concurrency 1 --p1-profile
```

既定はKapitsaと同じ`/tmp/kapitsa-gpu.lock`で、`BENCH_GPU_LOCK`の明示指定と`~/`展開にも対応する。同じファイルに対するPOSIX flockを使うprocess間で協調する。占有時は待たず`BenchmarkGpuBusy`で拒否し、実GPU処理や結果ファイル作成に進まない。親子継承は`VLLM_APPLE_BENCH_GPU_LOCK_FD`と実fdの両方が必要で、boolean markerだけの迂回は認めない。失効fdや別inodeは拒否する。

解放はfdのcloseで行い、共有open file descriptionへの明示`LOCK_UN`は行わない。親が終了しても継承した子が存続していれば排他は続く。lock fileはunlinkしない。report／campaign stateへpath・inode・継承有無とscopeを記録する。GPU device自体の予約ではなく、他のlock pathや未接続runner、通常のアプリを排他しない。

🟢 [Done] ロック追加後の全回帰1,608件成功（46 skip、39.366秒）。ロック単独8件、P1 runner2件、qualification12件、process cleanup3件成功。Ruff 0.17.2は既定CI範囲と変更した3runnerを検査し成功、compileall／diff check成功。モデルをloadせず、process間排他とCPU／模擬HTTP回帰を検証した。

English: Seven measurement entrypoints share Kapitsa's default flock file and fail before starting work when occupied. Three comparison campaigns retain it across every candidate and repetition; timeout aborts later trials after owned-process-group cleanup. Nested runners explicitly inherit and validate a real descriptor; closing one participant does not unlock the others. Reports record lock provenance. This is cooperative runner exclusion, not reservation of the GPU device. Other entrypoints remain unfinished.

简体中文：7个计测入口共用Kapitsa默认flock文件，占用时在工作开始前拒绝。3个比较campaign在全部候选及重复之间持续持锁；timeout回收所属process group并终止后续试验。父子runner显式传递并验证真实descriptor，一个参与者退出不会解除其他参与者的锁。报告记录锁来源。这是runner协作排他，不是GPU设备预约；其他入口尚未接入。

## Reproduction

🟢 [Done] model-owner変更時の全回帰1,600件成功（46 skip、36.361秒）。終了処理の単独9件成功、Ruff 0.17.2全件成功、compileall／diff check成功。全回帰にはmock encoderによるHTTP統合4件も含む。GPU lock追加後の最新結果は上節参照。

English: Owner-lifecycle regression: 1,600 tests, 46 skips. After GPU lock integration: 1,608 tests, 46 skips; all remaining tests pass. Ruff 0.17.2, compileall and diff checks pass.

简体中文：owner生命周期回归1,600项（46项skip）。GPU lock集成后1,608项（46项skip），其余全部通过；Ruff 0.17.2、compileall及diff检查通过。

```sh
.venv/bin/python -m unittest discover -s tests -p test_managed_engine.py
.venv/bin/python -m unittest discover -s tests -p test_qwen3_vl_managed_engine.py
.venv/bin/python -m unittest discover -s tests
uv run --no-project --with ruff==0.17.2 ruff check vllm_apple tests
.venv/bin/python -m compileall -q vllm_apple tests
```

全回帰の実行環境にはloopback HTTPのbind権限が必要。sandbox内の単独HTTP試験では`PermissionError`を確認し、通信可能な環境で全回帰を実行する。
