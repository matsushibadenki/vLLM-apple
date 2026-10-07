# P1 standard text candidate — 2026-10-04

## 長時間試験の開始時刻：09:00 JST

ユーザー指定（2026-10-07）により、長時間試験の一連の実行は日本時間の朝9:00から開始する。
r10は[制御停止](evaluation/p1-stability-m4-2026-10-07-r10/controlled-stop.json)し、runner／worker停止と専用job回収を確認した。
保存されたfailedは今回の中断を含み、性能原因の証拠や合格にしない。元の記録は保持する。
[Next] 次回は2026-10-08 09:00 JST、新規runのcompact／prefill512で30分→全条件合格時8時間。
既存の月・木9時の定期処理を更新済み。起動が遅れた場合に夕方・夜へ追いかけて開始しない。
English: Long-test campaigns start at 09:00 JST. R10 was stopped on user request and remains unqualified; a new run is planned for October 8 at 09:00 JST.
简体中文：长期试验从日本时间09:00开始。r10按用户要求停止并保留未认证记录；新试验计划于10月8日09:00启动。

## 直近の候補：r10 compact（制御停止）

[Done] [r9最終監査](evaluation/p1-r9-final-audit-2026-10-07.json)：品質3009/3009、SLO3000/3009。
RSS・awake・identity・正常停止は合格だが、cache増加87,461,321 bytesで資源gate未達、不合格。
8時間未開始。終了したr9専用launchd jobを回収した。
[Done] [効率化候補の比較](P1-EFFICIENCY.md)を実施。全回帰1454 tests成功（11 skip）。
[Next] [r10](evaluation/p1-stability-m4-2026-10-07-r10/state.json)はcompact／prefill512、30分全条件合格時だけ8時間。
[receipt](evaluation/p1-stability-launch-m4-2026-10-07-r10.json)に候補設定とjob identityを保存。
次回試験の実行中はruntime編集・追加GPU負荷を避ける。下記running記録は履歴である。
English: R9 failed with nine SLO misses and excess cache growth. R10 explicitly tests compact; no automatic promotion.
简体中文：r9因9个SLO超限及cache增长未达标而失败。r10显式测试compact，不自动提升认证。

## P1 cache上限の修正：2026-10-07

[Done] P1のCLI prompt cache予算256 MiBが保存先LRUの`max_bytes`へ反映されず、
応答完了時のinsertが既定の無制限設定になる抜けを修正。既存の厳しい上限は維持し、
初期化時に既存entryもtrimする。upstream fileは変更しない。
[実LRU再現](evaluation/p1-prompt-cache-budget-2026-10-07.json)では300 MiB相当の
合成byte metadataが旧設定で保存され、修正後には除去された。tensor性能測定ではない。
[実機smoke](evaluation/p1-cache-budget-smoke-m4-2026-10-07.json)は約92秒、
品質・SLO 240/240、awake・identity・正常停止に合格。短時間資源gateはfalse。
Python回帰のsandbox内試行はsocket禁止で失敗。socket利用可能な環境で再実行し、
1452 tests成功（11 skip）、Ruff成功。
smoke中のCPU回帰実行もあるため、速度比較・性能改善の証拠には使わない。

[Next] [r9再試験](evaluation/p1-stability-m4-2026-10-07-r9/state.json)をprefill512で開始。
30分全条件合格時だけ8時間へ進む専用launchd jobで、KeepAlive=false、証拠上書きなし。
[receipt](evaluation/p1-stability-launch-m4-2026-10-07-r9.json)のlabelだけを終了後に回収する。
実行中はruntime編集・追加GPU負荷を避ける。

[Next] [r8原因監査](evaluation/p1-r8-cause-audit-2026-10-07.json)：15件のSLO超過。
aggregate資源差は短文／長文の異なるworkload endpointを比較している。
同workload診断はplateauだが、gateは維持し漏れなしと認定しない。
遅いscheduler stepとhost loadの上昇が一部で重なるが低loadでも失敗する。
cache上限の抜けは確認済み、遅延・aggregate未達全体の根因と解消は未認定。

English: [Done] Apply the 256 MiB P1 prompt-cache budget to the LRU store, including completion-time insertion, preserving stricter limits. A real-LRU synthetic byte-metadata reproduction proves the budget defect; it is not a performance benchmark. The 92-second GPU smoke passed quality/SLO 240/240 and clean shutdown, but resources remain unqualified. [Next] Verify long-run latency and resource gates; the complete causes of R8 failures remain unproven. Gates are unchanged.

简体中文：[Done] 将P1的256 MiB prompt cache预算应用到LRU存储及完成时插入，保留更严格的限制。实际LRU的合成byte metadata试验复现了上限漏洞，不代表性能测试。约92秒GPU smoke的质量／SLO为240/240并正常停止，但资源尚未认证。[Next] 复测长期延迟与资源gate；r8全部失败的根因仍未确认，不放宽认证标准。

## 最新結果：2026-10-07

[Done] [r8最終監査](evaluation/p1-r8-final-audit-2026-10-07.json)：30分終了、品質2958/2958、SLO2943/2958。
awake・identity・正常終了は合格。SLOとaggregate資源gateが未達のため不合格、8時間未開始。
同workloadの資源診断は2群ともplateauだが、認定gateを置き換えず、根因の証明にも使わない。
[Next] 15件のSLO超過とaggregate資源gateの原因確認・修正・再試験。
以下のrunning記録は起動時の履歴であり、現在の稼働状態を示さない。
[ローカルtext preview](LOCAL-TEXT-PREVIEW.md)の短時間検証成功はP1長時間認定と別の範囲。

English: R8 finished and failed: quality 2958/2958, SLO 2943/2958 and aggregate resource gate unmet.
Eight-hour testing did not start. Earlier running statements are historical. Local preview validation does not certify P1.

简体中文：r8已结束且未通过：质量2958/2958，SLO2943/2958，aggregate资源gate未达标。
未开始8小时试验。下文running记录为启动时历史，本地preview验证不等于P1长期认证。

## 日本語

[Done] [r7 host-load監査](evaluation/p1-r7-host-load-audit-2026-10-06.json)：失敗window付近で
10 logical CPUに対し1分load average 34〜43を観測。一方load約7でも失敗があり、
全失敗の原因や外部process競合と断定しない。load averageはCPU使用率ではなくwindow境界の観測。

[Done] scheduler stepへ`time.thread_time_ns`によるcalling thread CPU時間を追加。
全stepのbounded histogramと、既存の直近64 slow stepへthread_cpu_msを記録する。
host wall時間・250ms閾値・SLO・認定gateは維持し、戻り値／例外／入力tensorは変更しない。
CPU時間は別thread／GPU処理を含まない。wallとの差はGPU時間ではなく、待ちやdescheduleなどを
含み得るため、kernel時間や原因特定として扱わない。追加clock読込の測定overheadは未測定。
[Next] 長時間の失敗stepでCPUとwallを突き合わせ、原因確認後に実装を修正する。

2026-10-06：[r7最終監査](evaluation/p1-r7-final-audit-2026-10-06.json)。
prefill256の30分は品質3162/3162、SLO3153/3162で不合格。8時間未開始。
awake・identity・shutdownは合格、RSS plateauは合格だがallocator active差169.8MBで資源gate未達。
短文と長文のTTFT超過が残っており、256を512より優れているとは認定しない。
終了した専用launchd labelは既に不在、runnerの終了を確認した。

[Done] 資源sampleへworkload hashを保存し、最新worker epoch内で同じ負荷のsampleを
比較する`resource_by_workload`診断を追加。保持されたsampleだけが対象であり、
欠測・labelなし・各負荷のsample不足は明示する。従来のaggregate gateは変更しない。
短文／長文の状態差と継続的な増加を調査するための診断で、memory leakや改善を証明しない。
r7の旧sampleにはworkload labelがないため、負荷別結果を推測で遡及付与しない。
[Next] 長文／短文遅延と資源gate未達の原因を、同一負荷の新しい実測で確認する。

[Done] 長時間launcher／runnerに`--prefill-step-size {128,256,512}`を接続し、
30分・8時間の両段階で指定値を固定する。既定512と認定閾値は維持。
全回帰1447 tests成功（11 skip）、Ruff成功。
[prefill256実機smoke](evaluation/p1-prefill256-m4-2026-10-05.json)は90秒指定／98.85秒、
品質・SLO 204/204、awake・identity・正常終了合格。初期長文12件のTTFT最大3.56秒。
短時間のall_epoch_resources_passedはfalseで、RSS収束・512比の性能優位は未認定。
[Next] [r7の状態](evaluation/p1-stability-m4-2026-10-05-r7/state.json)：prefill256候補で
30分→全条件合格時8時間の専用launchd jobを開始。command・checkpoint鮮度でrunning確認済み。
[receipt](evaluation/p1-stability-launch-m4-2026-10-05-r7.json)にparameterとjob identityを保存し、
既存定期監視もr7へ更新。実行中はruntime編集・追加GPU負荷を避け、資格は昇格しない。

[Done] RSS認定で時刻の重複・逆行・NaN／Infinity・欠測・負のRSSを拒否する。
従来は同じ時刻のsampleでも回帰分母ゼロを傾きゼロとしてplateauにできたため、
現在epochの全sampleを計算前に検証し、invalid_epoch_samplesとして未認定にする。
RSSの傾き／成長上限やSLOは変更していない。全回帰1446 tests成功（11 skip）、Ruff成功。
[既存実測への再適用](evaluation/p1-rss-sample-validation-2026-10-05.json)でr5／r6の
RSS判定と数値は完全に同じ。r6のRSS plateau未達・長文TTFT失敗は未解消で[Next]。
この修正はCPU上の認定処理であり、GPU性能や実安定性の改善を主張しない。

2026-10-05 r6終了：[最終監査](evaluation/p1-r6-final-audit-2026-10-05.json)。
品質3276/3276、SLO3275/3276、長文prefix missのTTFT 11.97秒で不合格、8時間未開始。
642観測でAC／automatic不変、suspend gap 0、awake条件は合格。
RSS直近成長58,507,264 bytes／傾き277,495,446 bytes毎時でplateau未達。
全期間RSSは15.6MB減少しており、直近傾きだけからmemory leakとは断定しない。
thread／FD／allocatorのresource plateau・identity・shutdownは合格、専用jobは回収済み。
[Next] sleep混入なしの長文TTFT超過とRSS収束を調査する。認定や閾値は変更しない。

2026-10-05 r5終了：[最終awake監査](evaluation/p1-r5-final-awake-audit-2026-10-05.json)。
active計測1801秒、品質3669/3669、SLO3667/3669、資源・identity・正常shutdownは合格、
8時間は未開始。失敗2件は2073-token prefix missのTTFT 10.36／12.47秒。
[途中OS電源ログ監査](evaluation/p1-r5-suspend-audit-2026-10-05.json)でSleep／DarkWakeとの
時間的重なりを確認。11回のsuspend gapとBattery→AC変化があり、固定awake条件の比較ではない。
GPU／kernelの性能不足やsleep単独原因とは断定せず、過去r4の短文失敗の原因にも転用しない。

[Done] P1 profileの通常試験へawake条件gateを接続した。各window前後のpower_sourceと
power_modeを全期間追跡し、一時的な変化が後で元に戻っても拒否する。
suspend gap観測、power条件変化、unknown／欠測はawake認定を拒否する。
理由はsuspend_gap_observed／power_conditions_changed／operating_conditions_unverifiedとして保存。
観測はwindow境界だけであり、その間の全電源イベントを証明するものではない。
sleep／wake専用試験を明示した場合はawake条件を要求せず、別scopeの検証として残す。
このgateはP1 runnerの認定判断であり、モデル演算・SLO閾値・OS電源設定を変更しない。
全回帰1436 tests成功（11 skip）。P1の30分／8時間認定は依然[Next]。

[awake gate実機smoke](evaluation/p1-awake-gate-m4-2026-10-05.json)：90秒指定の混合負荷で
品質・SLO 204/204、AC／automaticの40観測に変化・unknownなし、suspend gap 0、
awake gate・identity不変・正常shutdownが合格。30分認定ではない。
[r6](evaluation/p1-stability-m4-2026-10-05-r6/state.json)を30分→合格時8時間で起動し、
PID commandとcheckpoint鮮度によるrunning確認済み。既存定期監視もr6へ更新。
試験中は未認定で、runtime編集・追加GPU負荷を避ける。

2026-10-05：[r4監査](evaluation/p1-r4-slo-audit-2026-10-05.json)で品質3327/3327、
SLO3319/3327、失敗window 4、失敗request 8件を全て保存できた。
8件は全て短文のTTFT >5秒で、品質／E2E上限は合格。ほぼ全prefix hitでも発生した。
queue全期間最大は1.629秒なので、ingress→最初のdequeueだけでは遅延全体を説明できない。
これはrequestごとのqueue分解ではなく、原因はまだ未特定。資源・identity・shutdownは合格。
8時間は未開始。r4のlaunchd labelは既に存在せず、残存jobは確認されなかった。

[Done] P1 profile限定のscheduler step診断を追加。MLX-LM generate.pyの既知hashを確認して
`BatchGenerator.next`のhost wall時間を記録する。250ms以上のstepの総数と直近64件の
開始unix ns・経過ms・例外有無を保存し、入力・出力tensor／本文を取得しない。
GPU同期・CPU処理を分離したkernel計測ではない。collectorは各windowの前後に
queue／allocator／registry／HTTP資源とthermal／power／memory推定を取得し、失敗windowへ
保存する。snapshotはnon-atomic、測定の前後であり因果関係を証明しない。
観測が追加されたため、旧runとのcycle数・goodputの単純比較で改善を主張しない。

[実scheduler診断smoke](evaluation/p1-step-diagnostics-m4-2026-10-05.json)で90秒混合負荷
240/240件の品質・SLO成功、step 3187回、window前後の環境snapshot・identity不変・正常終了を
確認した。step p95はhistogram上限25ms、最大868.536ms。短時間ではTTFT超過未再現。
これは計測の動作確認であり、r4の8件の原因解消・性能優位・長時間安定性を認定しない。

全回帰1434 tests成功（11 skip）、Ruff成功後に[r5](evaluation/p1-stability-m4-2026-10-05-r5/state.json)
を起動し、PID commandとcheckpoint鮮度でrunningを確認した。30分→合格時8時間。
当時の定期監視はr5を対象とした。r5は終了し、専用jobは削除済み。
専用jobのreceiptは[p1-stability-launch-m4-2026-10-05-r5.json](evaluation/p1-stability-launch-m4-2026-10-05-r5.json)。

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

[Done] R7 boundary load reached 34–43 on ten logical CPUs near some failures, but failures
also occurred around load seven. Load average is not CPU utilization or proof of contention.
Scheduler diagnostics now record calling-thread CPU time separately from host wall time, using
bounded histograms and the existing latest-64 slow-step records. Other threads and GPU work
are excluded from the CPU clock; the difference is not GPU kernel time. Return/exception
behavior and qualification thresholds remain unchanged. Instrumentation overhead is unmeasured.
[Next] Correlate long-run failed steps before choosing a fix.

R7 failed: quality 3,162/3,162, SLO 3,153/3,162; eight-hour testing did not start.
Awake conditions, identity and shutdown passed. RSS plateau passed, but allocator active
change failed the resource gate. Prefill256 is not certified superior to 512.
[Done] Resource samples now carry workload hashes; diagnostic groups compare retained samples
within the latest worker epoch and the same workload. The aggregate qualification gate remains
unchanged. Older unlabelled samples do not gain inferred labels. Latency and resource causes remain [Next].

[Done] The long-run launcher accepts explicit prefill sizes 128/256/512 and preserves the
selected value through both stages; default 512 and qualification thresholds are unchanged.
Full regression: 1,447 tests, 11 skipped. The 256 smoke passed quality/SLO for 204/204 requests,
awake conditions, identity and graceful shutdown. Its short resource plateau did not pass;
RSS convergence and superiority over 512 remain unqualified. [Next] R7 is running 30 minutes
with prefill256 and proceeds to eight hours only if all gates pass. Monitoring targets R7;
no runtime edits or competing GPU tests while active.

[Done] RSS qualification now rejects duplicate/backward timestamps, nonfinite or missing
samples and negative RSS before regression. Previously zero time variance could appear as
zero slope and qualify a plateau. Limits remain unchanged. Reprocessing r5/r6 leaves their
RSS values and decisions unchanged; r6 stability and latency failures remain [Next].
This is validation of measurement evidence, not a runtime performance improvement.

R6 finished: quality 3,276/3,276, SLO 3,275/3,276; one long-prefix TTFT was 11.97 seconds.
Awake conditions passed, but recent RSS plateau failed. Net RSS decreased, so this does not
prove a leak. Identity and shutdown passed; the job was removed. Eight-hour testing did not start.

R5 finished with 3,669/3,669 quality passes and 3,667 SLO passes; resources, identity
and clean shutdown passed. Eight-hour testing did not start. Two long-prefix misses
overlapped OS sleep/darkwake activity, with eleven suspend gaps and Battery-to-AC
change, so this is not a controlled awake comparison or proof of a kernel defect.
The new P1 awake gate rejects suspend, changing power conditions and unknown observations.
Transient changes remain rejected even after returning to the initial state. Explicit
sleep/wake testing is separate. No SLO thresholds or OS power settings were changed.
The 90-second actual smoke passed quality and SLO for 204/204 requests; all 40 power
observations stayed AC/automatic, with no suspend gaps. Identity and shutdown passed.
The r6 30-minute trial is running and proceeds to eight hours only if it passes.
Neither the short smoke nor a running checkpoint certifies P1. Full regression: 1,436 tests, 11 skipped.

The r4 audit retained all eight failed requests across four windows: all were short
requests exceeding the five-second TTFT limit. Quality passed 3,327/3,327, SLO passed
3,319/3,327; resources, identity and shutdown passed, so eight-hour testing did not start.
The cumulative ingress-to-first-dequeue maximum was 1.629 seconds; this is insufficient
to explain the entire latency but is not a per-request decomposition or root-cause proof.
New source-pinned profiling measures `BatchGenerator.next` host-wall duration, retains
the latest 64 slow steps, and captures pre/post window environment and resource snapshots.
It does not measure isolated GPU kernels or certify performance improvement.

The 90-second instrumented smoke passed 240/240 quality/SLO requests and clean shutdown,
recording 3,187 scheduler steps and environment snapshots. The TTFT failure did not recur;
this validates observation, not a fix or sustained qualification.

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

[Done] r7部分失败附近，10个logical CPU的一分钟load为34〜43，但load约7时也失败。
load不是CPU使用率，也不证明竞争process。scheduler诊断新增calling thread CPU时间，
使用bounded histogram并记录既有最近64个slow step；CPU clock不包含其他thread及GPU工作。
wall与CPU差值不是GPU kernel时间。返回值／异常及认证阈值不变，新增计测开销未测量。
[Next] 对照长时间失败step的CPU与wall后确定修正。

r7失败：质量3162/3162，SLO3153/3162，未启动8小时。awake、identity及退出通过，
RSS plateau通过，但allocator active变化未通过资源gate。不认证256优于512。
[Done] 资源sample记录workload hash，诊断按最新worker epoch及相同负荷比较保留的sample，
原有aggregate认证gate不变。不会推测旧sample的label。延迟及资源问题根因仍为[Next]。

[Done] 长时间launcher可明确指定prefill128／256／512，并在两个阶段保持相同值，默认512及
认证阈值不变。全回归1447项通过，11项跳过。256短测的质量／SLO为204/204，awake条件、
identity及正常退出通过；短时间资源plateau未通过，RSS收敛及相比512的优势仍未认证。
[Next] r7以prefill256进行30分钟试验，仅全部gate通过后进入8小时。监控已转向r7，
运行期间避免runtime修改或竞争GPU试验。

[Done] RSS认证在回归计算前拒绝重复／倒退时间、非有限值、缺失值及负RSS。
此前时间方差为零可能被视为零斜率并通过plateau。未放宽阈值。重新处理r5／r6后
RSS数值及判定完全一致，r6的稳定性及延迟失败仍为[Next]，不宣称GPU性能改善。

r6结束：质量3276/3276，SLO3275/3276，长prefix TTFT为11.97秒。awake条件通过，
但近期RSS plateau未通过；总体RSS下降，不能据此断定泄漏。identity及正常退出通过，
专用job已清理，未启动8小时试验。

r5结束：质量3669/3669通过，SLO3667项通过，资源、identity和正常退出通过，未启动8小时试验。
2个长prefix miss与OS Sleep／DarkWake时间重叠，存在11次suspend gap及Battery→AC变化。
因此不是受控awake比较，也不能证明kernel缺陷。新P1 awake gate拒绝suspend、
电源条件变化及未知观测，即使之后恢复原状态也不会清除失败。显式sleep／wake试验另行验证。
未修改SLO阈值或OS电源设置。
90秒实机smoke的质量与SLO均为204/204，40次观测均为AC／automatic，
无suspend gap，identity及正常退出通过。r6的30分钟试验已启动，仅通过后进入8小时。
短期smoke或running状态不代表P1认证。全回归1436项通过，11项跳过。

r4保留了4个window的全部8个失败request，均为短输入的TTFT超过5秒。
质量3327/3327通过，SLO3319/3327；资源、identity及退出通过，未启动8小时试验。
整个运行的首次dequeue前最大等待为1.629秒，不足以解释完整延迟，但不能替代逐request分解或原因证明。
新诊断限定source hash，记录BatchGenerator.next的host wall时间、最近64个慢step，
及window前后的环境与资源。它不是独立GPU kernel计时，也不认证性能改善。

90秒诊断smoke通过240/240个质量及SLO请求并正常退出，记录3187个scheduler step及环境snapshot。
未复现TTFT失败，此结果仅验证观测功能，不代表问题已修复或长期认证完成。

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

2026-10-06 diagnostic validation / 診断検証 / 诊断验证：
[実機30秒指定smoke](evaluation/p1-workload-resources-m4-2026-10-06.json)は品質・SLO87/87、
正常終了合格。workload label欠落0、短文7 sample、長文1 sampleを分離。
長文はsample不足でplateau=false、診断自体のqualification=falseを維持する。
長時間・資源収束認定ではない。全回帰1448 tests成功（11 skipped）、Ruff・diff check成功。

English: The 30-second diagnostic smoke passed quality/SLO for 87/87 and shut down cleanly.
All retained samples were labelled; seven short-workload and one long-workload sample were
separated. The long group fails plateau for insufficient samples; diagnostics do not qualify
stability. Full regression: 1,448 tests, 11 skipped.

简体中文：30秒诊断smoke的质量／SLO为87/87，正常退出。label缺失0，分开7个短负荷sample
与1个长负荷sample。长负荷因样本不足而plateau=false，诊断不认证稳定性。
全回归1448项通过，11项跳过。

2026-10-06 CPU/wall validation / 診断検証 / 诊断验证：
[実機smoke](evaluation/p1-step-cpu-m4-2026-10-06.json)は48/48品質・SLO、正常終了成功。
614 stepでwall平均21.978ms、calling-thread CPU平均8.266msを別々に記録。
slow step例はwall841.954ms／CPU33.745ms。差の内訳は未特定でGPU時間とは扱わない。
全回帰1449 tests成功（11 skipped）、Ruff・diff check成功。
[Next] [r8](evaluation/p1-stability-m4-2026-10-06-r8/state.json)をprefill512で
30分→全条件合格時8時間として開始。command・checkpoint鮮度でrunningを確認し、定期監視更新。
実行中は資格保留、runtime編集／追加GPU負荷を避ける。

English: The actual smoke passed 48/48 quality/SLO with clean shutdown. Across 614 steps,
wall and calling-thread CPU means were 21.978ms and 8.266ms; one slow step was
841.954ms wall versus 33.745ms CPU. The difference remains unattributed. Full regression:
1,449 tests, 11 skipped. R8 is running with prefill512, proceeding from 30 minutes to eight
hours only on full success; qualification remains withheld and monitoring targets R8.

简体中文：实机smoke的质量／SLO为48/48，正常退出。614个step的wall与calling-thread CPU
均值分别为21.978ms及8.266ms；一个slow step为wall841.954ms／CPU33.745ms，差值原因未知。
全回归1449项通过，11项跳过。r8以prefill512进行30分钟试验，仅全部通过后进入8小时，
运行中不提升资格，监控已转向r8。
