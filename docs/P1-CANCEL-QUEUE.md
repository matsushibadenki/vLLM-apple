# P1 cancellation under response backpressure — 2026-10-28

## Process completion and resource reservations 2026 11 04

🟢 [Done] 追補：childの待機推論commandをcancelした場合、queue mutex内で未取得であることを確認して除去し、delegateを呼ばず終端`request_cancelled`を返す。active要求はここで解放しない。受付からのdeadlineをchildで保持し、実行開始時に期限切れを確認するため、キュー待ち後にtimeout予算をリセットしない。

親が完了確認として受け付けるのは、対応idのobject型resultか、既知の終端error codeだけとする。単なるcancel受領通知、不正result、未知error、stdout EOFでは予約を保持し、所属childの停止・実終了確認後に回収する。完了callbackからengineをcloseする場合はreaderの自己joinを避け、reader自身がstdoutを閉じる。遅延diagnosticsは従来どおり終端応答まで上限に含める。

CPU子process試験はactive delegateをfile barrierで停止したまま、queued cancel→終端確認→後続受付→既存上限で余剰拒否→active終了後のFIFO実行を検証する。HTTP試験では408後に不正応答を注入し、scheduler／入力byte予約がchildの実終了後に一度だけ回収されることを確認する。期限切れdelegateの未呼出、unknown terminal／不正result／未知error／EOFと、完了callbackからのcloseも検証する。初期証拠とlogは保持し、追補結果を別記録へ保存する。

English: Queued inference cancellation now returns a terminal acknowledgment before the active native call finishes; the queue removal and owner claim are mutually exclusive. Queue time consumes the original deadline. Only valid terminal responses release reservations; protocol failure or EOF triggers owned-child termination and confirmed exit first. Completion callbacks may close the engine without joining the reader itself. CPU subprocess/HTTP tests cover these contracts; real GPU and sustained-load qualification remain open.

简体中文：等待中的推理请求现在可在active native调用结束前返回取消终态，队列移除与owner取得command互斥。等待时间消耗原deadline。仅合法终态释放预算；protocol失败或EOF先停止所属child并确认退出，再回收。completion callback可关闭engine，避免reader自身join。CPU子进程／HTTP测试验证这些契约；真实GPU及持续负载认证仍未完成。

追補の最終検証：全回帰1,626 tests成功（46 skip、181.681秒、OpenSSL 3明示）。不正応答／HTTP回収／callback closeの3 edge testsもResourceWarningをerrorとして成功。指定Ruff 0.17.3、compileall、diff検査成功。[source hash・実行記録](evaluation/p1-process-ack-followup-2026-11-04.json)。以下の32 tests／Ruff 0.16.5の記載は、追補前の初期検証を保持したもの。

🟢 [Done] 子プロセス所有の推論でも、HTTP timeout／切断だけでは受付枠・スケジューラ予約・入力byte予算を返さない。親はcancel通知を送ってcallerへ即時に戻り、childがdelegateから戻って送るresult／errorを完了確認として扱う。応答前のstdout EOF／protocol失敗ではprocess終了を待ち、cancel送信やpipe切断だけを実処理終了と扱わない。異常終了は実際のprocess終了確認後に全待機通知を完了する。

pending登録の回収をcallerのfinallyからresponse readerへ移し、完了Futureを既存APIの解放callbackへ渡す。callbackはpending lockの外で実行する。期限切れdiagnosticsも応答まで受付上限に含め、送信前のJSON／サイズ拒否は直ちに枠を返す。所有childのclose後はstdin／stdoutも閉じる。

CPU子プロセスfixtureをfile barrierで停止し、cancel後もcapacity 1が満杯であること、child応答／異常終了後の一度だけの完了、後続要求、HTTP timeout 408／切断後の4,096 bytes予約とbody byte保持・回収を検証する。実GPU／KVメモリ量、推論速度、長時間SLOの認定ではない。応答しないchildの予約は終了確認まで保持し、既存closeの強制回収対象はそのengineが生成したchildだけとする。

English: 🟢 [Done] Process-owned inference retains admission, scheduler and input-byte reservations after HTTP timeout/disconnect until the child returns a terminal response or actually exits. Sending cancel or losing stdout alone does not certify completion. Pending completion callbacks run outside the registry lock; late diagnostics stay bounded, unsent oversized requests release immediately, and owned pipes close at shutdown. CPU subprocess and HTTP tests verify retained reservations, exactly-once release, child-exit recovery and subsequent requests. Real GPU memory, performance and long-run SLO qualification remain open.

简体中文：🟢 [Done] process所有的推论在HTTP timeout／断开后保留受付、scheduler及输入byte预算，直到child返回终态响应或实际退出。发送cancel或stdout断开不等于实际完成。completion callback在registry锁外执行；超时diagnostics继续计入上限，发送前拒绝的过大请求立即回收，停止后关闭所属pipe。CPU子进程及HTTP测试验证预算保持、仅一次释放、child异常退出恢复及后续请求。真实GPU内存、性能及长期SLO认证仍未完成。

再現：`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -W error::ResourceWarning -m unittest tests.test_process_inference_engine tests.test_inference_request tests.test_managed_engine`。32 tests成功。全回帰1,621 tests成功（46 skip、156.771秒、試験のみOpenSSL 3を明示）。初回はmacOS LibreSSLの`pkey -check`非対応により既存署名試験1件失敗し、単独再現・OpenSSL 3での成功を確認。実資格情報・システム設定は変更していない。Ruff 0.16.5、変更3ファイルの構文、diff検査成功。指定Ruff 0.17.3はoffline cacheに存在せず、その版の成功とは扱わない。全体回帰とlintの最終結果は[検証記録](evaluation/p1-process-completion-2026-11-04.json)を参照。

🟠 [Next] 実backendの継続cancel負荷と固定identityの30分→8時間認定。戻らないnative callの運用上の回収時間は既存shutdown timeoutと分けて評価する。

## Native completion and resource reservations 2026 11 02

🟢 [Done] thread-affine native engineのactive要求がtimeout／切断した場合、HTTP handlerは終了してもスケジューラ予約と入力byte予算は実処理の終了まで保持する。従来はcallerがcancel例外で戻るとHTTP側のfinallyが即時解放し、delegateがまだ実行中なのに資源の空きを過大に公開していた。

commandはownerが確定する`Future`の完了通知を持ち、cancel例外がその通知をAPIへ渡す。APIは完了callbackで予約を一度だけ返す。専用の待機threadは追加せず、callbackの登録前に処理が終わっていた場合も即時・一度だけ解放する。待機要求の回収・shutdownも完了通知を発行し、callbackはqueue／state lockの外で実行する。通常応答と、完了通知を持たない既存engineの例外は従来の終了時解放を維持する。

実HTTPのCPU模擬delegateで、timeout 408／client切断後に4,096 bytesのスケジューラ予約と入力bodyのbyte予算が残ること、ownerの処理終了後だけ解放されること、その後の正常HTTP要求が成功することを確認する。これはnative thread所有経路の契約であり、GPU／KVの実メモリ量や、process engineのcancel acknowledgmentを認定するものではない。戻らないnative callの予約は保持し、強制thread終了で回収しない。

English: 🟢 [Done] An abandoned thread-owned native request retains scheduler and input-byte reservations until actual delegate completion, even after its HTTP handler returns. A completion callback releases once without adding a waiting thread and handles completion-before-registration races. CPU HTTP tests cover timeout, disconnect, retained reservations and successful recovery. Process-owned cancellation acknowledgments and real GPU memory qualification remain unfinished.

简体中文：🟢 [Done] thread所有的native请求在timeout或断开后，即使HTTP handler已返回，仍保留scheduler及输入byte预算，直到delegate实际结束。通过completion callback一次性释放，不增加等待thread，并处理注册前已完成的情况。CPU HTTP测试覆盖timeout、断开、预算保持及后续恢复。process所有的取消确认及真实GPU内存认证仍未完成。

検証：全回帰1,617件成功（46 skip、45.722秒）、Ruff 0.17.2／compileall／diff check成功。`test_inference_request.py`の6 testsには実HTTPの2終了経路と後続回復、完了済み／未完了通知の登録順序を含む。今回のGPUモデル実行はない。

## Model owner waiting queue recovery 2026 11 02

🟢 [Done] `ThreadAffineInferenceEngine`で、キャンセル済み・切断済み・期限切れの待機要求をキューから回収する。以前はcallerがcancelを検出して戻ってもcommandがキューに残り、実行中のdelegateが終了するまで後続要求を満杯として拒否していた。

新規受付時とcallerのcancel検出時に、最大64件の待機キューをmutex内で検査する。不要なcommandだけを取り除いて完了通知を出し、正常要求のFIFOと既存capacityを維持する。新規受付側でも検査するため、cancelされたcallerのpolling threadが先に動く必要はない。ownerが既に取り出したcommandはこの回収対象に含めず、その実行・終了処理はownerへ残す。終了sentinelも保持する。

CPU回帰では、active処理を止めたままcapacity 1の待機要求をcancel→replacement受付→上限超過拒否→active処理の完了後にreplacement実行を確認する。期限切れ・切断・caller cancelの3経路、正常FIFO、終了sentinel、100回／400件の回収も検証する。回収用の例外にqueue frameのtracebackを保持しない。これはnative model-owner経路の受付回復であり、過去のMLX-LM実測SLO超過の原因解決や推論速度改善を認定するものではない。

English: 🟢 [Done] Cancelled, disconnected and expired waiting commands now release bounded model-owner queue capacity before the active delegate returns. Admission also purges stale waiters, so recovery does not depend on the cancelled caller polling first. Valid FIFO order, shutdown sentinels and active ownership remain intact. CPU tests cover capacity-one recovery and 400 cancelled waiters across 100 cycles; real-model SLO and throughput qualification remain open.

简体中文：🟢 [Done] 已取消、断开或过期的等待command可在active delegate返回前释放有界队列容量。新请求入队时也清理无效等待项，无需等待旧caller先完成polling。正常FIFO、结束sentinel及active所有权保持不变。CPU测试覆盖capacity 1的恢复及100轮／400项取消；真实模型SLO与吞吐认证仍未完成。

Reproduction: `.venv/bin/python -m unittest discover -s tests -p test_managed_engine.py`。

検証：model-owner関連13 tests成功。最終全回帰1,612件成功（46 skip、39.388秒）。Ruff 0.17.2、compileall、diff check成功。全回帰の模擬HTTP統合を含むが、今回新しい実モデル測定は行っていない。

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
