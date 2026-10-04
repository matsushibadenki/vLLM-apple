# P0 implementation history through 2026-10-03

Preserved before the 2026-10-04 baseline audit. Status labels below are historical; see ROADMAP.md and P0-AUDIT.md for current scope.

## P0 — 比較可能な基準と実行経路の監査 [Next]

- [Done] [M4複数trickle body試験](evaluation/wrapper-multi-trickle-body-m4-2026-10-03.json)で待機枠1・使用枠2の状態を確認した。100 bytes宣言のbodyを2本同時に200 ms間隔で送信してもdeadlineは延びず、双方約10.002秒で408、追加要求503、最終使用枠0・生成開始0・準備失敗2・拒否1。後続3/3品質合格。全HTTP threadの上限や長時間soakは未認定。[Next] request header段階も含むHTTP処理の資源上限と長時間負荷を検証する。

- [Done] body読取りの最終chunk後にも絶対deadlineを検証し、期限超過の完全bodyを受理しないよう修正した。timeout引数の有限正数検証も追加。[M4遅いbody試験](evaluation/wrapper-slow-body-m4-2026-10-03.json)で未完bodyは約10.002秒後に408、待機枠0での追加要求は503、最終使用枠0・生成開始0・準備失敗1・拒否1を確認し、後続3/3品質合格。CPU回帰も並行実行したため性能比較には使わない。[Next] trickle送信・複数slow client・長時間soakとHTTP全体のthread上限を検証する。

- [Done] wrapper admissionをbody準備完了後のFIFOへ変更した。先頭の取消／timeoutはticketを除去し後続へ通知する。順序保持と先頭取消のtestを200回反復合格。[M4回帰](evaluation/wrapper-fifo-queue-cancel-m4-2026-10-03.json)で待機切断1・生成開始は先行1のみ・最終枠0・後続3/3品質合格を確認した。接続時刻順・body読取り中のFIFOではなく、実backend token schedulerの認定でもない。[Next] 長時間負荷と遅いbody送信時のadmissionを検証する。

- [Done] 全回帰でQwen4 cancel／shutdown socket競合テストのthread終了待ちが失敗したため、test cleanupをclient shutdown→join→server closeへ修正した。別threadのblocking readをcloseだけで起こす前提を除去し、該当テスト200回反復合格。Qwen4 runtime実装は変更していない。

- [Done] wrapperのstream生成中BrokenPipe／ConnectionResetを正常な切断として扱い、`active_disconnects`を待機取消と分けて公開・保存する。MLX-LM 0.32.0の`ctx.stop()` finally経路をlocal sourceで確認。[M4実model試験](evaluation/wrapper-active-cancel-m4-2026-10-03.json)でcontent受信後に切断、active_disconnects 1・adapter使用枠0を約51.5 msで観測し、後続3/3品質合格。GPU停止latency、非stream切断、半切断と長時間soakは未認定。

- [Done] wrapper admissionの使用枠・active・生成開始・取消・timeout・拒否・body準備失敗counterをlock付きsnapshotとして公開し、benchmarkは非負整数を検証して保存する。[M4実model待機切断試験](evaluation/wrapper-queue-cancel-m4-2026-10-03.json)は先行生成中に2枠使用を確認して待機clientを切断、取消1・生成開始1のみ・最終使用枠0を確認した。先行streamは200／DONE、後続3/3品質合格。生成開始後cancel・長時間soak・half-close・FIFOは未認定。

- [Done] wrapper生成admissionで容量枠確保後にbodyを最大8 MiB・絶対deadline 10秒で先読みし、worker待機後は元のbackendへ再生する。重複Content-Length／Transfer-Encoding、不正size、部分EOF、timeoutを拒否し、元socket timeoutと入力streamを復元する。実socketテストで未読body後の切断を生成前に検出し枠回収。[M4正常経路](evaluation/text-benchmark-body-admission-m4-2026-10-03.json)はclient c2で6/6品質／SLO合格、telemetry 19/19成功。CPU回帰同時実行のため性能比較には使わない。[Next] 実modelの長いqueue切断・生成中cancel・half-closeを確認する。

- [Done] wrapper admissionへ50 ms以下の待機pollと生成開始直前のcancel callback確認を追加した。検出後は生成せず枠を解放し、wrapperはEOF／socket errorをbest-effortで確認して切断済み応答を書かない。callback cancel・枠回収・後続admissionと、未読byteを消費しないprobeをテストした。実modelの切断cancelは未検証で、未読bodyが残る切断は検出が遅れ、read-side EOFもcancelとして扱う。[Next] body読取りとcancel検出を一体化し、half-close、長時間queue、生成中cancelを実機検証する。

- [Done] wrapper生成admissionへ待機件数（既定8、0〜128）と待機時間（既定30秒、最大300秒）の上限を追加した。超過／timeoutは503＋Retry-After、未読bodyを残さないようconnectionを閉じ、例外・timeout後も枠を解放する。[M4待機枠0試験](evaluation/text-benchmark-admission-m4-2026-10-03.json)は同時6件中1件成功・5件503（server log確認）、後続3/3品質合格、telemetry 19/19成功。CPU回帰も同時実行したため性能比較には使わない。[Next] client切断の待機中検知、FIFO、公平性、長時間cancel負荷を検証する。

- [Done] M4／Gemma 2 2B／MLX-LM 0.32.0 wrapperでclient並列度2を検証した。[直列化前](evaluation/text-benchmark-c2-memory-m4-2026-10-03.json)はwarmup 3/3合格後、測定0/6完了・6件timeout、telemetry 139/139成功。生成POSTを既定で直列化し、[修正後](evaluation/text-benchmark-c2-serialized-memory-m4-2026-10-03.json)は同条件6/6品質／SLO合格、telemetry 16/16成功。`--allow-concurrent-generation`は実験用opt-inとした。backend batching認定・性能改善ではなく、HTTP adapterでの安全な順次処理である。[Next] 長時間cancel／queue負荷、client切断と複数model条件を検証する。

- [Done] wrapper容量telemetryとbenchmark reportへ`snapshot_consistency=non_atomic`を明示した。[M4生成＋同時telemetry試験](evaluation/text-benchmark-concurrent-memory-m4-2026-10-03.json)でwarmup 3/3・測定30/30品質／SLO合格、100 ms待機間隔のbounded probe 50/50取得成功。単一生成workerに取得要求を重ねた短時間試験であり、原子的snapshot、複数同時生成、取得による性能影響は未認定。最終試験中はCPU回帰テストも実行したため速度比較には使わない。[Next] 独立条件でtelemetry有無の比較と複数生成要求時の動作を確認する。

- [Done] [M4 wrapper tokenize／idle telemetry実測](evaluation/wrapper-tokenize-memory-smoke-m4-2026-10-03.json)で英語22・日本語20・简体中文19 tokenが実生成usageと一致し、3/3品質合格。不正modelは400、続く正常tokenizeは成功した。idle memory取得30回は平均9.672 ms／最大12.663 ms。これはHTTP取得の所要時間で、推論中の干渉・overhead差や性能改善は未認定。[Next] 並列生成中のsnapshot一貫性とtelemetry有無の比較、eviction公開を検証する。

- [Done] benchmarkへ応答usageのcached prompt token集計を追加した。欠測・bool・負数・prompt数超過は未取得とし、warmupを除外する。[M4実測](evaluation/text-benchmark-cache-usage-m4-2026-10-03.json)でwarmup 3/3・測定30/30品質／SLO合格、30件すべてから580/610 prompt token再利用を取得した。backend報告値であり、evictionとKV正確性・性能改善は未認定。
- [Done] 実wrapperのMLX-LM 0.32.0互換性を修正した。欠落server設定を補い、単一workerのResponseGeneratorとHTTP起動へtelemetry handlerを明示接続した。LRU容量を`backend_lru_accounting`として区別し、token数null・traversal完全性falseを維持する。[M4実測](evaluation/text-benchmark-wrapper-memory-m4-2026-10-03.json)はwarmup 3/3・測定30/30品質／SLO合格、前後のKV容量7,774,208 bytes取得。0.32.0以外の新APIは自動認定しない。[Next] 計測負荷、tokenize endpointの実機検証、並列時のsnapshot一貫性とeviction公開を確認する。

- [Done] benchmarkへ任意の`--collect-backend-memory`を追加し、測定窓外の前後2点でwrapper容量telemetryを取得する契約を実装した。1秒・64 KiB・redirect拒否、schemaと非負整数・traversal完全性を検証し、取得不能は理由付き欠測にする。KV hit／evictionはnullで容量から推測しない。[Next] 実wrapper経路での取得と負荷、hit／eviction counterの公開を検証する。

- [Done] [Magnitude参照評価](magnitude-review.md)に基づき、evidence-only runtime autotunerへ任意のphase別ばらつきgateとbaseline改善marginを追加した。全sampleとpolicyをreport IDへ束縛し、noisyな最速候補と僅差の設定切替を防ぐ。synthetic検証であり、実LLM性能・daemon自動適用は未認定。[Next] 独立確認測定とE2E回帰へ接続する。

依存：なし。次の開発サイクルの最優先。成果物はbenchmark suite、capability matrix、再現可能なbaseline report。

- [Next] 起動から最終tokenまでの経路を追い、backendごとにstreaming、usage、cancel、prefix reuse、continuous batching、chunked prefill、KV精度、structured outputの「未対応／adapterのみ／実機合格」を記録する。フラグの存在だけでは有効と判定しない。
- [Next] 既存phase probe・qualification・soakを共通reportへ接続する。直接backendとdaemon経由を同じ入力で比較し、queue、tokenize、prefill、decode、serialization／SSE、model loadを分離する。
- [Done] P0のstream通信計測：phase probeで最終生成contentと`[DONE]`到着を分離し、end-to-end／stream tailを固定容量histogramへ集計。未取得sampleは欠測として区別する。既存のdecode計算を維持し、HTTP・schema・qualification回帰49件で検証。[仕様・再現手順](PHASE-TRANSPORT-METRICS.md)。実モデルの性能比較は未実施。
- [Done] P0のbounded HTTP benchmark runner：三言語・固定workload hash、並列度1〜32、失敗を含む件数、品質とTTFT／E2E SLOを満たすgoodput、言語別集計、参考値表示付きp99を実装。関連54テスト合格。[仕様](TEXT-BENCHMARK.md)。[M4実測](evaluation/text-benchmark-m4-smoke-2026-09-26.json)のGemma 2 2B／MLX-LMで並列度1は30/30正答。並列度2は応答停止後に手動回収し、不完全な失敗runとして保存。性能優位・batching認定は付与しない。
- [Done] direct／proxy benchmarkの比較manifestを追加した。同じworkload、request数、concurrency、SLO、model artifact digest、backend build digestだけを比較し、失敗・品質不合格を分母に残す。goodput／throughput／p99比は保存するが、artifact identity未指定、品質失敗、metric欠測なら性能結論を明示的に保留する。既存reportはidentity未検証なので自動昇格しない。[仕様](TEXT-BENCHMARK.md)。
- [Done] M4／Gemma 2 2B／MLX-LM 0.32.0でidentity-bound direct／daemon短時間比較を実施した。最低30件gate導入前の[9件report](evaluation/text-route-comparison-m4-2026-09-28.json)は`blocked_insufficient_samples`へ降格。[30件比較](evaluation/text-route-comparison-30req-m4-2026-09-28.json)は両経路30/30正答・SLO合格、direct 17.452、daemon 17.657 goodput tokens/s、proxy/direct比1.011736、p99 histogram上限は双方500 msで`comparable`。ただしclosed-loop c1、cache非制御、単一runでp99も参考値のため、約1.2%差を性能改善またはproxy overheadゼロの証拠にしない。runtime変更で旧30分証跡はidentity mismatchとして拒否されたためdaemonは明示的なbackend check skipで起動し、qualificationはfalseである。
- [Done] direct／proxy比較の反復series gateを追加した。3〜21 pairを同じworkload／artifact／backend buildへ束縛し、direct-first／proxy-firstを両方含み件数差1以内、goodput比の最大相対幅5%以内を要求する。[M4 3×30件series](evaluation/text-route-comparison-series-3x30-m4-2026-09-28.json)は順序2:1、全pairで両経路30/30正答・SLO合格、proxy/direct goodput比1.011736／1.005702／1.006541、中央値1.006541、相対幅0.5995%で`comparable_stable_reference`。p99は各30件なので参考値、cache非制御、c1限定であり、0.65%差を性能優位として認定しない。
- [Done] 測定前のbounded warmupをbenchmarkへ統合した。warmup件数をworkload identityへ束縛し、試行・完了・品質・errorを測定値とは別に保存する。比較器は両routeの同数・全件成功を要求し、失敗を`blocked_warmup_failure`で可視化する。[M4 warmup 3＋測定30件比較](evaluation/text-route-comparison-warm3-30req-m4-2026-09-29.json)は両routeで3/3 warmup、30/30測定が品質・SLO合格し、direct 18.448、daemon 18.783 goodput tokens/s、比1.018188で`comparable`。単一direct-first pair、c1、backend管理cache、参考p99に限定し、性能改善には認定しない。
- [Done] warmup 3件を固定したdirect／daemon比較を順序交替で3 pair取得した。[M4 3×30件series](evaluation/text-route-comparison-series-warm3-3x30-m4-2026-09-30.json)は各routeのwarmup 9/9、測定90/90が品質・SLO合格し、順序はdirect-first 2／proxy-first 1。proxy/direct goodput比は1.018188／1.071009／0.956578、中央値1.018188、相対幅11.2387%で許容5%を超え、`blocked_variance`となった。warmupだけでは比較のばらつきを抑えられず、性能優位を認定しない。
- [Done] benchmarkへ`--collect-operating-context`を追加した。warmup前・測定前後の3点で温度状態、電源供給元、電源モード、UTC時刻を記録し、`--target-pid`指定時には対象processの経過秒数も保存する。probeをgoodput計測窓から除外し、取得不能をunknown／nullで維持する。[M4観測smoke](evaluation/benchmark-context-m4-2026-10-01.json)でnominal／Battery Power／automatic／観測process ageを取得した。推論性能、連続温度、cache hit率は未測定である。
- [Done] 比較器へ`--require-operating-context`を追加した。両routeの3点でnominal温度・同じ既知の電源供給元と電源モード、UTC timestampの順序、測定前の同一UTC日、worker ageの非減少と測定前の経過時間差5秒以内を要求する。欠測・条件変化は`blocked_operating_context`として理由を保存する。任意gateとして既存reportとの互換性を維持し、性能qualificationはfalseのままとする。
- [Done] operating context gateを有効にして同一UTC日の独立3 pairをM4で測定した。[3×30件series](evaluation/text-route-comparison-series-context-3x30-m4-2026-10-01.json)は各routeのwarmup 9/9、測定90/90が品質・SLO合格し、全pairでgate合格。各routeをfresh processで起動し、worker age 30秒以降にwarmup開始、測定前age 30〜31秒、全18観測でnominal／AC Power／automatic、順序2:1を確認した。goodput比1.006013／0.984572／1.170932、相対幅18.5246%で`blocked_variance`。観測条件一致だけでは安定した速度比較を保証できず、性能改善は未認定。[再現条件](evaluation/text-context-series-protocol-m4-2026-10-01.json)。
- [Done] benchmarkの測定窓外snapshotへ1／5／15分load average、logical CPU数、利用可能メモリ推定値・取得元・fallback表示を追加した。pressureは利用可能比率からの推定と明記し、OS判定やCPU使用率には認定しない。欠測はnull、比較gateは維持する。[仕様](TEXT-BENCHMARK.md)。[M4観測smoke](evaluation/benchmark-context-system-m4-2026-10-02.json)でload／10 logical CPU／メモリ推定を取得した。推論速度の改善は未測定。
- [Done] benchmark reportへ固定容量のTTFT／E2E histogramを追加した。bucket境界・overflow・sample数・未取得件数・mean／maxを保存し、品質／SLO不合格も含め、warmupを除外する。失敗と`[DONE]`欠測は遅延ゼロにしない。境界・overflow・全失敗・欠測・warmupの関連検証を実施。[仕様](TEXT-BENCHMARK.md)。実モデルでの分布取得と性能改善は未測定。
- [Done] [M4実モデル観測smoke](evaluation/text-benchmark-distribution-m4-2026-10-03.json)でGemma 2 2B／MLX-LM directのwarmup 3/3・測定30/30が品質／SLO合格し、TTFT／E2E histogram各30 sample・欠測0と3点のsystem観測を取得した。E2E平均225.022 ms／最大348.379 ms。単一路線・identity未指定の短時間検証であり、速度改善や変動原因は未認定。試験backendは終了後に停止した。
- [Next] 短いclosed-loop測定の変動原因を調べるため、追加したsystem観測とrequest latency分布、cache hit／evictionを揃えて取得する。CPU使用率とbackground process別負荷は未取得。MLX-LM標準HTTPでcache hit／evictionは現在欠測であり、取得契約を先に整える。測定時間を延ばす比較は取得した条件と失敗理由を揃えて行い、同じ短時間試験の反復だけで性能認定しない。
- [Next] モデル・データ・tokenizer・chat template・量子化方式／group size・KV dtype・sampling・依存versionを固定する。reportへrevision／hash、再現コマンド、power／thermal、失敗・除外理由を保存する。
- [Next] 実MLX cacheでmetadata計測の完全性と負荷を確認し、viewの共有storageを二重計上し得る限界を明記する。queueキャンセルの長時間負荷とp95待ち時間を測る。

### 標準比較matrix

| 軸 | 最初に比較する条件 |
| --- | --- |
| Hardware | 手元のMacを正確に記録。続いて16 GB級、24–36 GB級、64 GB以上、異なるSoC世代を追加。未保有機は未評価表示 |
| Models | 配置済み2–4Bを基準に、収容可能な7–8B、14B級、MoE／hybridを順次追加。architectureごとに対応確認 |
| Workload | chat、coding、tool-use、長文読解。英語・日本語・简体中文ごとに品質を分離 |
| Context | 入力128／1K／4K／16K tokens、出力128／512 tokens。32K以上はmodel上限・admission通過時のみ |
| Load | concurrency 1／2／4／8、短長混在、到着率指定、飽和試験。容量を超える組合せは理由付きskip |
| Cache | cold model load、warm model＋prefix miss、exact prefix hit、prefix編集、cache容量超過を別集計 |
| Environment | AC／battery／low-powerを別条件とし、warm-up後と熱平衡後を比較。各backendは順番を交替して実行 |

同じbit数でもGGUFとMLXの量子化は等価とみなさない。同一artifactでの純粋な速度比較と、同一品質基準を満たす異形式の比較を分ける。
比較可能な共通モデル集合とbackend固有対応範囲を両方公開し、成功した組合せだけに母集団を狭めない。

計測はclient側TTFT、token interval由来TPOT、request p50／p95／p99、E2E latency、生成tok/s、**SLO内の成功output tokens/s（goodput）**を主指標にする。
SSE chunk数をtoken数とみなさず、usageやbackend token timestampを用いる。欠測は推定値で埋めずunavailableとする。本文最終tokenと`[DONE]`到着も分けて記録する。
RSS・allocator・KV・OS pressure・swap差分は別系列で記録し、重複するmemory値を単純加算しない。energy/tokenは取得可能な場合だけ補助指標とする。

**完了条件：** 手元のMacで少なくとも2 backend・同一品質基準のモデルについて、独立した3 runを再現できる。各主要ケースは合計100以上の完了requestを目安とし、sample数とばらつきを公開する。p99は1,000未満なら参考値扱い。比較不能・未対応条件も残す。

