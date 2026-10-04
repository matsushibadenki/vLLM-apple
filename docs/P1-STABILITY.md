# P1 standard text candidate — 2026-10-04

## 日本語

P1は[Next]。完了条件は実際の8時間混合負荷・障害注入と資源回収の合格であり、短時間smokeやrunnerの実装だけでは完了扱いにしない。

[Done] 検証候補をM4／32 GiB、配置済みGemma 2 2B 4-bit、MLX 0.32.1／MLX-LM 0.32.0へ限定した。既存source hash限定のGemma 2 mask／cancel互換経路を使用し、`--p1-profile`で次の上限を明示適用する。標準daemon全体を認定したものではなく、限定backend candidateである。

| 項目 | 上限・契約 |
| --- | --- |
| Model owner | upstreamの単一ResponseGenerator worker。設定modelのみ、adapter／draftなし |
| Context | tokenizer後のprompt＋最大出力 ≤4,096 tokens。生成前にowner threadで拒否 |
| Output | 1–512 tokens。model／出力上限はenqueue前に検証 |
| GPU allocator | 8 GiB上限、allocator cache 256 MiB。OOMの自動retryなし |
| Prefix cache | 4 entries以下、256 MiB trim。論理容量と物理memoryを混同しない |
| Backend batch | decode／prompt concurrency各2以下、prefill step ≤512 |
| Workload | 短文c2、約2K-token長文c1。最大contextと最大並列度の同時保証なし |
| HTTP | accepted connections／request threads各最大16、listen backlog 16。header deadline 5秒、body最大8 MiB・絶対期限10秒、socket I/O timeout 30秒 |
| Output queues | profileではrequest response queue最大1,024 items。取消を冪等にして終了markerの無制限追加を防ぐ |

8 GiB GPU上限とbounded入力を用いる保守的なcandidateであり、任意shapeのKV／scratchを精密予測するadmissionではない。OS reserveの動的計測を用いるdaemon admissionや機種横断認定は別検証が必要。header timeoutはsocketを閉じ、容量超過はbest-effort HTTP 503を返す。過負荷時はcontrol requestも拒否され得るため、取消APIが常に受理される保証はない。

[Done] `/vllm-apple/resources`でHTTP current／peak／rejection、request registry、allocator active／cache／peak、thread数、open FD数を観測する。全体snapshotはnon-atomic。queue p95はenqueueから最初のscheduler dequeueまでの固定容量histogram上限で、取消済みrequestを含む。backend内の全待ち時間、GPU停止時間やHTTP待機時間とは区別する。

[Done] qualification runnerを強化した。warmup失敗なら長時間試験を開始しない。modelファイルとruntime sourceのhash、runner source hash、profile上限を保存し、終了時の変更は認定を拒否する。checkpointは原子的に保存し、例外時は失敗理由とshutdown結果を保存する。既存証跡は上書きしない。

再起動前に各worker epochのRSS／allocator／thread／FD／registryの傾向を保存し、8時間gateでは全epochの合格を要求する。最後のworkerだけが安定していても合格にしない。RSS後半は16 MiB/時以下・増加64 MiB以内、allocator active／cache後半増加各64 MiB以内、thread／FD後半増加各4以内、idle registry 0を要求する。境界付近のrestartでsample不足を作らないよう、8時間終了前300秒はcrash注入を追加しない。これらは安定性の検出基準であり、leakが完全にないことの証明ではない。

### 実測と進行中の試験

- 初回[失敗smoke](evaluation/p1-profile-90sec-m4-2026-10-04.json)：upstreamの`default_model` draft sentinelを誤って拒否した。候補を修正し、失敗を保存した。
- [開発smoke r2](evaluation/p1-profile-90sec-m4-2026-10-04-r2.json)：正常189/189件だが途中でruntime sourceを変更したため資格を与えない。短いworker epochの資源安定性も未判定。
- [固定source smoke r3](evaluation/p1-profile-90sec-m4-2026-10-04-r3.json)：正常189/189件が品質・SLO合格、active cancel 18/18、queued cancel 6/6、timeout 6/6、worker crash回復2/2、half-close正常完了、model／output／context上限の拒否、正常shutdown。elapsed約101秒で、30分／8時間認定ではない。
- 最終CPU回帰1,407 tests合格（11 skip）、変更PythonのRuff合格。実socket testでtrickle header、容量超過503、期限後の枠回収、後続正常応答を確認した。
- [前回の中断](evaluation/p1-stability-m4-2026-10-04/state.json)：正常408件成功のcheckpoint後、runnerとbackend PIDが消失した。終了原因・shutdown結果は不明。statusをinterrupted、passed=falseとして保存し、認定しない。
- [再実行r3の状態](evaluation/p1-stability-m4-2026-10-04-r3/state.json)：user launchd管理の独立job（親PID 1）で30分smokeから開始し、合格した場合だけ8時間へ進む。[launch receipt](evaluation/p1-stability-launch-m4-2026-10-04-r3.json)にjob identityとcommandを保存した。[30分checkpoint](evaluation/p1-stability-m4-2026-10-04-r3/30min.json)／[8時間checkpoint](evaluation/p1-stability-m4-2026-10-04-r3/8hour.json)は実行段階に応じて生成される。実行中は`passed=false`、終了後のreportとsource同一性を確認するまで資格を昇格しない。

```sh
PYTHONPATH=. .venv/bin/python scripts/launch_p1_stability.py \
  --python /opt/homebrew/opt/vllm-metal/libexec/bin/python \
  --model models/gemma-2-2b-it-4bit \
  --output-directory docs/evaluation/p1-stability-new-run \
  --receipt docs/evaluation/p1-stability-new-launch.json --port 19146
python3 scripts/p1_stability_status.py docs/evaluation/p1-stability-new-run
```

実行中はこのruntimeを編集せず、競合するGPU試験を起動しない。runnerはMacのidle sleepを試験時間だけ抑止し、完了・失敗時に解除する。ユーザー操作によるsleepを強制したり無効化したりする試験ではない。実sleep／wakeは別の管理された試験が必要で、未認定の[Next]として残す。`--require-sleep-wake`は実観測ゼロを合格にしない。未保有Macの試験は[Later][pending]。

launchctl submitの一時jobを実測すると失敗時の再起動があったため、r2を制御停止し、r3ではKeepAlive=falseを明示したplistをuser launchdへbootstrapする。[起動lifecycle観測](evaluation/p1-launch-lifecycle-m4-2026-10-04.json)では同じ異常終了の12秒観測で旧方式2回、修正方式1回だった。確認用jobは削除済み。常設LaunchAgents配置や自動restartはしない。

監視helperはrunner PIDだけでなくcommandを照合し、checkpointが300秒より古ければstaleとする。`status=running`だけで稼働中とは判定しない。dead runner／stale checkpoint拒否、bootstrap失敗の保存とKeepAlive=falseの契約を確認するCPU test 2件に合格した。job終了後はreceiptのdomainとlabelを確認して`launchctl bootout <domain>/<label>`で当該試験jobを片付ける。

[Next] 連続試験の最終report、全worker epoch、障害後回収を監査する。その結果を限定profileの認定へ束縛し、daemon実HTTP経路・dynamic admission・実sleep／wakeの残課題を検証してからP1を[Done]にする。

2026-10-04更新：r3の30分試験は1,802秒で正常3,312/3,312件完了・品質合格、SLO合格3,298件（14件未達）により総合不合格。cancel／timeout／slow consumer、RSS・allocator・thread・FD、identity不変性・正常shutdownは合格した。8時間stageは開始せず、終了したlaunchd jobは削除済み。SLO未達を解消してから再試験する。

[Done] 失敗診断を追加。benchmarkは品質／TTFT／E2E／stream終了欠落の未達理由、request index、言語、prompt／output／cached token数、request開始時刻（window開始からの相対秒）と遅延を最初の64失敗requestまで保存する。prompt／回答本文は保存しない。phase通信例外は従来のerror code集計へ残し、request診断のobserved数とは区別する。soakは最初の64失敗windowを保存し、総失敗window数を別計上する。後続成功でも失敗記録を消さない。保存上限を超えたサンプルを全失敗の代表値とみなさない。

[Next] r3には途中のwindowが残っていないため、既存14件の原因は未特定。新しい診断で再取得し、latency failureと品質・HTTP失敗を分離して調査する。閾値や品質条件を緩めて合格にしない。

追加後の[90秒混合負荷](evaluation/p1-failure-diagnostics-m4-2026-10-04.json)は204/204件の
品質・SLO成功、失敗window 0、identity不変と正常shutdownを確認した。未達条件は再現せず、
30分／8時間の認定には代用しない。診断・window保持の対象19 testsとRuff成功。

全回帰1432 tests成功（11 skip）後、[r4の30分→合格時8時間試験](evaluation/p1-stability-m4-2026-10-04-r4/state.json)
を独立launchd jobで開始した。[receipt](evaluation/p1-stability-launch-m4-2026-10-04-r4.json)
のlabelは`com.vllm-apple.p1.0e8c7e25f6dc49dab3e12eae7795b8fa`。
起動直後にPID commandとcheckpoint鮮度を照合しrunningを確認した。最終結果が出るまで
未認定として扱い、runtime編集・依存更新・追加GPU試験を避ける。
既存の定期監視もr4へ更新し、進行中は通知せず合否・対応が必要な変更のみ通知する。

## English

Failure diagnostics now retain up to 64 failed requests per benchmark and 64 failed
windows per soak, with uncapped failure counters. They preserve timing, language,
token counts and failure reasons without storing prompt or generated text. Later
successful windows cannot erase earlier failures. Existing r3 evidence lacks the
intermediate windows, so the cause of its 14 SLO misses remains unproven.

The new 90-second mixed workload passed 204/204 quality/SLO requests and clean shutdown.
After 1,432 regression tests passed (11 skipped), r4 was launched for 30 minutes,
followed by eight hours only if the first gate passes. Running is not qualification.

P1 remains [Next]. The bounded candidate pins Gemma 2 2B on M4/32 GiB to reviewed MLX versions and source hashes. It restricts model selection, prompt plus output length, batch size, GPU memory, cache, HTTP connections, header/body deadlines and response queues. Cancellation is idempotent. These limits do not certify arbitrary contexts or the complete daemon path.

The fixed-source smoke passed 189 normal quality/SLO requests, 18 active cancellations, six queued cancellations, six timeouts, two worker-crash recoveries, half-close and model/output/context rejection checks. This is approximately 101 seconds, not sustained certification. A separate sequence runs a 30-minute qualification and starts the eight-hour test only after success. Running checkpoints never grant qualification.

The first long run was interrupted after 408 successful requests; termination and cleanup were not recorded. It remains unqualified. The replacement runs as an independent user launchd job. Monitoring verifies the runner command and checkpoint freshness, rather than trusting a saved running marker.

Update: the replacement 30-minute run completed all 3,312 quality requests but only 3,298 within SLO. Resource and cleanup gates passed; the overall gate failed and the eight-hour stage did not start. Its finished launchd job was removed.

All worker epochs must pass RSS and allocator/thread/FD/registry stability checks; a healthy final worker cannot hide earlier growth. The report retains model/runtime hashes and rejects changes during measurement. Queue p95 covers ingress to first scheduler dequeue, including cancelled requests, rather than total generation wait. Snapshots are non-atomic. The process temporarily inhibits idle sleep; actual sleep/wake and daemon integration remain [Next]. Additional hardware remains [Later][pending]. Final CPU regression: 1,407 tests passed, 11 skipped.

## 简体中文

失败诊断现保存每个benchmark最先的64个失败request及每次soak最先的64个失败window，
另保留完整失败计数。记录时间、语言、token数和失败原因，不保存prompt或回答正文。
后续成功不会覆盖先前失败。旧r3缺少中间window，14个SLO失败的原因仍未确定。

新的90秒混合负载通过204/204个质量及SLO请求并正常退出。全回归1432项通过（11 skip）后，
已启动r4的30分钟测试，仅在合格后执行8小时测试。运行中不授予认证。

P1仍为[Next]。限定候选固定M4／32 GiB上的Gemma 2 2B及已审查MLX版本和source hash，限制模型选择、prompt＋output长度、batch、GPU内存、cache、HTTP连接、header／body期限和响应队列。取消具有幂等性。这些上限不代表任意context或完整daemon路径已通过认证。

固定source的smoke通过189个正常质量／SLO请求、18次active cancel、6次queued cancel、6次timeout、2次worker crash恢复、half-close及model／output／context拒绝测试。约101秒结果不是长期认证。连续测试先执行30分钟，通过后才启动8小时。运行中的checkpoint不会授予资格。

第一次长期测试在408个成功请求后中断，未记录终止原因及回收结果，因此不认证。重新测试使用独立的user launchd job；监视会核对runner command和checkpoint时间，不能仅凭保存的running状态认定仍在运行。

更新：重新执行的30分钟测试完成3,312个质量合格请求，但仅3,298个满足SLO。资源及回收gate通过，总gate不合格，因此未启动8小时stage；已移除结束的launchd job。

每个worker epoch都必须通过RSS、allocator、thread、FD和registry稳定性检查，不能只看最后一个worker。报告保存模型与runtime hash，并拒绝测量期间的变更。queue p95仅表示入队到首次scheduler dequeue，包含已取消请求，并非完整生成等待。snapshot不是原子的。测试期间暂时阻止idle sleep；实际sleep／wake、daemon集成仍为[Next]。其他未持有硬件为[Later][pending]。最终CPU回归1,407 tests通过、11 skip。
