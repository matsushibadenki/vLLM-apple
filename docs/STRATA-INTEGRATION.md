# Strata integration — 2026-10-10

## 日本語

🟢 [Done] 第24工程（2026-10-13）：`mlx_server`へ明示Strata設定を接続。
root／expected model SHA／resident bytes／entries／file上限を全て指定した場合のみ導入。
対応はreviewed MLX-LM0.32.0の単一local worker。defaultモデルを起動前にロードして導入し、
providerのbatchabilityを無効化。異なるモデル、adapter、draft、並行生成指定を拒否する。
生成workerをstop／join後、MLX同期してruntimeをclose。生成worker構築失敗もcloseする。
`StrataServingModel`は単一系列forwardだけを許可し、width>1をprefill、width1をdecodeと統計する。
prompt末尾のsingletonもdecode扱いで、HTTP prompt境界の正確な統計ではない。
server関連21 testsとnative関連19 tests成功。distributed／draftの導入前拒否、同期→closeの順序も検証。実HTTP／実モデルの長時間認定は保留。
⭕️ [Pending] 実artifactでHTTP streaming／cancel／cache再利用E2E、長時間安定性、速度資格。


🟢 [Done] 第23工程（2026-10-13）：`StrataRuntime.from_model()`を追加。
標準Qwen3 MoE ModelからMoE層を自動検出し、manifestの全layer／expert coverage一致と
expected model identityを確認してから導入する。異なるarchitecture、空のMoE集合、欠落／余剰Expertを拒否。
2層の合成モデルでattention／norm／embeddingを含む全logits一致を確認。native関連19 tests成功。
この入口はロード済みmodelを受け取るため、初期source bankのstartupメモリを削減しない。
⭕️ [Pending] serverのbatch／phase管理と同期runtimeの安全な接続、実モデルE2E、速度認定。


🟢 [Done] 第22工程（2026-10-13）：標準Qwen3 MoE向けの同期・明示導入型Strata接続層を実装。
公開API `vllm_apple.StrataRuntime` は `{layer_id: moe_block}` とmanifest-bound backendを受け取り、
全層が一つのentry／byte予算、LRU／cost_frequency policy、prefill／decode telemetryを共有する。
`mode="sequential"`が既定、`grouped`／`row_batch`は明示指定。後者のbatch上限は既定8。
`set_phase()`、`resize()`、`snapshot()`、冪等`close()`とcontext managerを提供。
導入失敗は変更済み全層の元switchを復元し、成功時は元重みbankを保持しない。
closeはactive Leaseを拒否し、成功後はhookの再呼出しとphase変更を拒否する。
元モデルに戻すにはreloadする。呼出し・phase変更・resize・closeの並行実行は非対応。
all-resident gatherは既存の個別opt-in APIとして維持し、この共有動的予算へ混在させない。
予測器／SSD tierの補助部品は実装済みだが、非認定の自動prefetchや既定採用は有効にしない。

利用順序：offline exporterで対象層のimmutable expert filesを生成し、expected model SHAを指定して
`MLXFileExpertBackend.from_manifest()`で開く。モデルload後・要求開始前にruntimeを構築し、
各prefill／decode区間の前にphaseを設定し、推論終了後にcloseする。外部source bank参照も解放する。
このAPIはloaderへの明示接続点であり、既存serverの自動切替・未対応architectureの認定ではない。
複数層のnative数値一致、共有予算、縮小fallback、元bank解放、導入rollback、終了拒否を検証。
native18 tests／周辺41 tests、ruff成功。
⭕️ [Pending] 低負荷全方式比較・実weight／GPU E2E・実OOM・自動server導入／既定採用。


🟢 [Done] 第21工程（2026-10-13）：[exact payload予算の多様prefillメモリ検証](evaluation/strata-diverse-prefill-exact-payload-memory-analysis-2026-10-13.json)。
CPU／Metal、1-row／8-row、各2 fresh workersで5入力のcold／warmを測定（8 trials／80要求）。
予算2,359,296 bytes、保持／予約ピークは予算内、active Leaseとclose後の保持は0。
最大絶対誤差2.235e-7。warm evalはrepeat128→16、mixed128→28、alternating128→128、
reversed32→32、random128→120。mixed warmのMLX activeはCPU4,195,352→4,314,136 bytes、
Metal4,195,352–4,203,544→4,535,320 bytes。batch scratchの増加を確認した。
全観測のRSSは入力・参照・module・allocator cacheを含むため、候補間の削減を認定しない。
速度workerを省く`--memory-only`と厳密予算`--exact-payload-budget`をprobeへ追加。
合成入力の機能／メモリ検証であり、実モデル・system GPU・実OOM・速度資格ではない。
🟠 [Next] install metadata費用と要求固定費の切り分け。
⭕️ [Pending] 低負荷全方式比較は自動チェック継続。実weight／GPU E2E・既定採用。


🟢 [Done] 第20工程（2026-10-13）：group admissionをfile総量から保守的payload上限へ変更。
grouped hookのinstall時に各Expertの8-byte safetensors prefix／fileサイズからpayload上限を取得し、
immutable tupleで保持する。要求時はcached payload合計と現在のentry／byte予算を比較する。
不足時fallbackは維持。全fileのmetadataを読むstartup費用が増えるが、要求時の追加header readはない。
manifest identity／設定／checksumは従来どおり各loadで確認し、managerの事前予約・実tensor量の検証も維持。
ファイルはinstallから利用終了までimmutableが契約。source bankやrollback重みは保持しない。

[2 payloadぴったりの機能比較](evaluation/strata-payload-admission-analysis-cpu-2026-10-13.json)：
保持予算18,432 bytes、d64／4 Experts／top2／affine4 group32、8-token要求。
各3 fresh processでbaseline／1行grouped／8行batch、9 trial／576要求（4,608 tokens）誤差0。
候補の保持／incoming予約ピーク18,432 bytes、残存Lease／拒否0、close保持0。
各candidate64要求でExpert eval512→64、acquire1024→128、miss／eviction両者32。
以前のheader込み判定で逐次に退いたpayload予算でも、grouped／batchが有効になった。
第19工程の厳密payload対照は修正前の履歴であり、今回の認定へ流用しない。
profile on／host未資格の機能証拠であり、参考速度をspeedup認定しない。

native関連16 tests成功：payload予算でのbatch、1 payloadへのresize後の逐次fallback、
install時にfile欠落なら元block不変、install後のchecksum不一致で全Lease解放を確認。
既存dense／4／8-bit・Router／順序／source所有権回帰も通過。
新しいheader probeは追加metadataのみで、checksum検証や予算を省略しない。
activation scratch／RSS／GPU／実OOMをweight予算で保証するとは主張しない。
今回の再開preflightはCPU最大約1,012%で未達。低負荷の自動比較は継続し、source変更後は新規試行を要求する。
🟠 [Next] exact payload予算で多様prefillのRSS／MLXピークを再取得し、startup費用と要求固定費を比較する。
⭕️ [Pending] 全方式の非競合資格、実weight／GPU E2E・実OOM・既定採用。


🟢 [Done] 第19工程（2026-10-12）：[多様prefillの比較とメモリ記録](evaluation/strata-diverse-prefill-analysis-cpu-metal-final-2026-10-12.json)。
新規probeはd512／intermediate1024／4 Experts／affine4-bit group32の独立ファイルを使用。
連続、長さ混在（1／2／3／5／8／13）、交互、順序反転、seed固定ランダムの5種類。
入力はsin波／random vector、選択はauthoritative fixture。実文章・tokenizer・実Router品質試験ではない。
反転は32 tokens、他は128。native SwitchGLU referenceとfileは別processで生成し、workerは測定前に一時source bankを解放。
1-token groupedと最大8-token batchを比較。速度はCPU各3 fresh process、各case1 warmup＋4要求。
メモリはCPU／Metal各2 fresh process、cold managerとcache／常駐を保持した2要求、counter有効。
合計14 worker／120速度要求＋80メモリ要求、全case最大誤差2.2352e-7、保持／解放契約合格。
Expert payload1,179,648 bytes、保持予算2,360,838 bytes（2 file）、3 Expertは収まらない。
保持／incoming予約ピークは2,359,296 bytes、残存Lease0、close後保持0。

| 入力・token数 | Expert eval／要求：1行→8行 | 再load回数／warm要求（両者同じ） | CPU参考中央値：1行→8行 (ms) |
|---|---:|---:|---:|
| 連続・128 | 128→16 | 0 | 57.226→50.880 |
| 長さ混在・128 | 128→28 | 48 | 89.857→87.154 |
| 交互・128 | 128→128 | 256 | 220.474→225.028 |
| 順序反転・32 | 32→32 | 0 | 14.312→14.097 |
| ランダム・128 | 128→120 | 153 | 160.961→155.714 |

連続性が低い場合は演算発行削減が限定的。交互の遅さには保持容量を超える選択と再読み込みが伴う。
I/OとRouter host取り出しはbatch化で削減されていない。速度のrun変動約5.6–173%、host資格なし。
この表は速度優位の認定ではなく、小標本の参考値。安定試行だけの選別は行っていない。

warm要求のMLX activeピーク：長さ混在ではCPU4.001→4.114 MiB、Metal4.001–4.028→4.325 MiB。
連続ではCPU両者4.001 MiB、Metalは1行4.001–4.028／8行4.001–4.220 MiB。
activation／一時結果／allocatorの増加を含むprocess全体値であり、scratch単独の厳密分解ではない。
128-token CPU cacheは長さ混在1.027→1.465 MiB、連続1.027→1.219 MiB。
warm RSS sample範囲は全caseで約330.9–335.4 MiB。import、input/reference、cache等が支配し、
同じMLX active量でもRSSは一致しない。2 repeatでRSSの方式優位は認定しない。
RSSは要求1ms sampleの下限、HWMはprocess生涯値。各要求前にMLX peakをresetし前後で同期。
RSS samplerとcounterは速度workerへ入れていない。Metal値はprocess unified allocator統計、専用VRAMではない。

[厳密payload予算の対照](evaluation/strata-diverse-prefill-tight-budget-analysis-2026-10-12.json)も保存。
2 payloadぴったり（2,359,296 bytes）はgroupedのheader込み保守判定に不足し、両候補が逐次fallback。
連続128-tokenでもeval256→256、数値一致。file予算を含めた本比較と混ぜてbatch効果としない。
🟠 [Next] payloadに基づくgroup admissionの検討、実prefillでの連続性・scratch／RSS・低負荷速度比較。
⭕️ [Pending] 実モデル／GPU E2E・system GPU・実OOM・速度資格・既定採用。


🟢 [Done] 第18工程（2026-10-12）：`grouped=True, maximum_group_rows=8`の明示候補を追加。
同じ順序のExpert選択が連続するtokenだけを、最大指定行数（1–64）でまとめて演算する。
順序が異なれば別group。既定1行、groupedなしの複数行指定は拒否。entry／byte不足では従来の逐次fallback。
Router、top-k、各tokenの最終重み付けは既存blockに残し、Leaseを全batchの同期評価まで保持する。
weight予算は維持するがbatch化でactivation scratchは増える。scratch／RSS／実OOMの上限保証ではない。
phase telemetryのsample／selected_expertsは実行group単位となり、Router token数と混同しない。
要求全体の4,096行／64 MiB出力上限を維持。native関連15 tests成功、dense／4／8-bit数値一致、
逆順groupの分離、group行上限、容量fallback、例外時解放を検証。

[8-token合成機能比較](evaluation/strata-contiguous-row-analysis-cpu-2026-10-12.json)：
`--row-suite --tokens-per-request 8 --resident-bytes 20000 --cycles 8 --warmup-cycles 2 --runs 3 --profile`。
baseline／1行grouped／8行batchを順序rotation、各3 fresh process、9 trial／576要求（4,608 tokens）誤差0。
入力は要求内で同一token vectorを8回繰り返す、連続選択に有利なfixture。一般prefillの代表値とはしない。
各candidate64要求でExpert eval512→64、Expert build span1024→128、acquire1024→128。
Router to_hostは両者64、load／miss／evictionは両者32。保持／incoming予約ピーク18,432 bytes、
残存Lease／拒否0、close後保持0。host同期そのものやI/Oは減っていない。
profile on・host資格なしのため速度認定しない。参考中央値grouped606.917／batch232.459 μs、baseline69.583 μs。
この差を本番speedupとしない。既存1-token自動比較は既定1行のまま維持する。
🟠 [Next] 連続選択が少ないprefill、batch scratch／RSS、低負荷での候補比較を確認。
⭕️ [Pending] 実weight／GPU／E2E・速度資格と既定採用。


🟢 [Done] 第17工程（2026-10-12）：startupと要求区間をREADY／GO／DONE／ACKで分離。
startupはimport・reference生成・export・preload・warmupを含む。READY後に再び低負荷を待ち、
parentが要求monitorを開始してGOを送る。DONEを受けた時点でmonitorを停止し、その後にACKでcleanupを許可する。
startupは約1秒、要求は約100ms間隔のCPU記録。終了時の短い端数は直前sampleとまとめ、CPU deltaを捨てない。
要求windowにはparity確認とcontroller合図処理を含むが、各要求のlatencyにはparity・合図は含まれない。
startup／READY待機／要求の記録を分け、資格判定は要求windowに対して実施する。
startupの失敗が要求の資格へ自動的に混入しないこと、要求記録欠落の拒否、
READY待機未達時にGOを送らずworkerを終了することを検証。関連34 unit tests、ruff成功。
`--quiet-timeout`（3–180秒、既定30秒）で各gateの待機上限を指定できる。
全体CPUはbenchmark／observerを含み、他process上限・PID条件を維持した。閾値は緩めていない。

[分離した部分常駐試行](evaluation/strata-phased-partial-analysis-cpu-2026-10-12.json)：
baseline1 trial／4,096要求の誤差0。startup観測約2.50秒、CPU140–222%；
要求window約0.293秒、CPU161–218%で両windowに条件未達がある。
次のlaunch前gateでjournal_bench等の高負荷を確認しcampaign停止。
[全常駐](evaluation/strata-phased-resident-analysis-cpu-2026-10-12.json)はlaunch前gate未達でtrial0。
[120秒待機の部分常駐再試行](evaluation/strata-phased-partial-retry-analysis-cpu-2026-10-12.json)もtrial0。
116 sampleの全体CPU約196–959%、中央値234%、他のPython process等の負荷が残った。
要求区間にも同時負荷があり、startup分離だけで全方式の非競合比較が成立したとはしない。
raw試行・gate未達の履歴を保存し、速度／既定採用資格はfalse。runtimeの既定経路は変更していない。
🟠 [Next] 要求中の負荷条件が成立するhost時間帯で、分離runnerによる全方式比較を揃える。
⭕️ [Pending] 全方式の非競合資格・実weightピーク／system GPU／実OOM／E2E認定。


🟢 [Done] 第16工程（2026-10-12）：CPU低負荷gate付き比較を再実行。
[全常駐](evaluation/strata-quiet-resident-analysis-cpu-2026-10-12.json)は14 trial／57,344要求、
[部分常駐](evaluation/strata-quiet-partial-analysis-cpu-2026-10-12.json)は16 trial／16,384要求、計73,728要求の誤差0。
すべてprofile off、64要求warmup、独立process・順序rotation。
全常駐は各4,096要求、4 repeatを計画したが最後のgate未達で14 trialで終了。
中央値の中央値はbaseline 47.667、逐次LRU143.459、grouped LRU121.125、gather49.209 μs。
逐次／gatherのrun変動は0.70／1.56%、grouped13.31%、baseline189.33%。
全常駐campaignは未完、実行中host条件にも未達があり、速度認定しない。

部分常駐は20,000-byte予算、全Expert36,864 bytes、各1,024要求×4 repeatを完了。

| 条件 | run中央値の中央値 (μs) | run総時間変動 | host条件通過trial |
|---|---:|---:|---:|
| baseline（全weight保持の参考値） | 45.781 | 1.74% | 2/4 |
| 逐次LRU | 217.969 | 1.90% | 4/4 |
| grouped LRU | 189.250 | 0.33% | 3/4 |
| grouped cost_frequency | 199.135 | 2.44% | 2/4 |

全方式でrun変動≤5%に到達。逐次LRU単体はhost条件も全試行通過した。
ただし他方式のCPU上限／PID変動条件が未達で、campaign全体の非競合比較資格はfalse。
良好なtrialだけを選び直して速度優位を認定しない。baselineは同じ保持予算を満たさない。
候補の保持／incoming予約ピーク18,432 bytes、拒否・残存Lease0、close後保持0。
各1,024要求のmiss／evictionは逐次640／640、grouped両方式512／512、再読み込み20%減を再確認。
source hashとartifact契約を検証。runtimeの既定設定は変更していない。
🟠 [Next] 全方式で実行中host条件を満たす比較を揃え、source初期化と要求区間の負荷を切り分ける。
⭕️ [Pending] 実互換weightのload／推論RSS・MLXピーク、system GPU・実OOM・E2E認定。


🟢 [Done] 第15工程（2026-10-11）：比較runnerに`--quiet-host`を追加。
1秒間隔でシステム全体CPU時間と取得可能なprocess CPU時間を計測し、
全体≤200%・他process最大≤75%・PID増減なしを3 sample連続で満たしてからworkerを開始する。
実行中も別threadで負荷を記録し、解析はhost条件とrun変動≤5%の両方を要求する。
全体CPUにはbenchmark自身も含む。取得できないprocessの負荷も全体CPUで検出する。
CPU sampleは瞬間的負荷・disk／GPU競合を完全には検出しない。
空／未完campaignやhost記録なしを非競合比較の成功としない。関連30 unit tests、ruff成功。

[低負荷比較の再試行記録](evaluation/strata-quiet-resident-retry-analysis-cpu-2026-10-11.json)：
30秒確認を2 campaignで実行したが条件未達、worker開始0、速度値は取得していない。
Rust／Swiftコンパイル等の高負荷を確認。他の作業を停止せず、speed資格／既定採用をfalseのまま維持する。
⭕️ [Pending] 同時負荷を避けた速度比較は、現hostで低負荷条件が満たされるまで未完。

[読み込みメモリ検証](evaluation/strata-load-memory-analysis-cpu-metal-2026-10-11.json)：
Apple M4／32 GiB、d1024・intermediate2048・4 Experts・affine4-bit／group32の合成fixture。
1 Expert payload 4.5 MiB、全Expert18 MiB。source生成／exportは別processで完了し、
workerは元のmodel bankを保持しない。CPU／Metal default device、保持capacity1／2、
従来load後退避／事前退避を各2 fresh processで比較（最終16 trial／96 acquire）。
checksum・load・weight evalを含み、推論kernel・Router・activationは含まない。
各load前に同期・MLX peak reset、後に同期。RSSは要求1ms間隔のsample最大とprocess生涯HWMを併記。

| device | 保持Expert数 | MLX activeピーク：従来→事前退避 (MiB) | sample RSSピーク：従来→事前退避 (MiB、2 run範囲) |
|---|---:|---:|---:|
| CPU | 1 | 9→4.5 | 69.44–70.02 → 64.66–65.41 |
| CPU | 2 | 13.5→9 | 72.86–74.22 → 69.22–69.62 |
| Metal default | 1 | 9→4.5 | 68.81–71.02 → 64.25–64.34 |
| Metal default | 2 | 13.5→9 | 73.36–75.12 → 69.97–70.69 |

MLX activeピークは全試行で旧Expert＋incomingの重なり1 Expert分（4.5 MiB）減。
close後は保持tensor／active／Lease残存0。ただしallocator cacheは残り、検証用clear後にcache0。
RSSはimport等とcacheも含むので、保持予算と一致しない。RSS sampleピークは短いtransientを逃し得る下限。
HWMはprocess生涯値でありload区間だけのピークではない。
Apple Siliconはunified memoryであり、RSSとMLX値を加算しない。
Metal defaultの値はMLX process allocator統計であり、専用VRAM／system全体GPUピークとは主張しない。
本番runtimeへclear_cacheを追加していない。計測は速度資格・実weight E2E・実OOM保証ではない。
🟠 [Next] 非競合hostの速度比較と実互換weightのload／推論を含むRSS・MLXピーク計測。
⭕️ [Pending] 実OOM、system全体GPUメモリ、長時間E2E認定。


🟢 [Done] 第14工程（2026-10-11）：MLX backendに読み込み前のpayload上限照会を追加。
safetensorsの8-byte prefixからheader長を確認し、file size−prefix−headerを予約量とする。
managerは予約量を保持予算に収めるために未使用Expertを先に退避し、pinned／容量不足をload前に拒否する。
読み込み後も実tensor量が予約以下か検証し、超過resourceは解放して拒否する。
対応backendで保持重み＋incoming payloadの論理ピーク`peak_reserved_bytes`を記録。
照会を持たない既存backendは従来のload後退避を維持する。
ファイルはoffline生成後immutableが契約。header prefix readはfile_stat spanに含まれる。

[実行記録](evaluation/strata-preload-reservation-cpu-2026-10-11.json)：
20,000 bytes予算のgrouped LRU／cost、10,000 bytes予算の逐次fallback各64要求、計192要求の誤差0。
保持＋incoming予約ピークはそれぞれ18,432／18,432／9,216 bytes、残存Lease0、close後保持0。
grouped Expert eval各64回、1 Expert fallbackは128回。
24 unit testsと14 native関連tests、ruff成功。pinned／単体容量不足のload前拒否、
予約違反resourceの解放、模擬MemoryError後の再試行、厳密tensor予算での回帰を検証。
ロード失敗前に退避したcacheは元に戻さないが、managerは再試行可能。

予約で旧Expert＋新Expertの論理重み超過を避ける。loader scratch、checksum buffer、
中間activation、allocator cache、OS RSS、GPU memoryは上限に含まれず、実OOM安全認定ではない。
読み込み前に実ファイル全体のMLX allocation量を制限できるとは主張しない。
確認時はRustコンパイル等の高負荷処理が残っており、追加の速度比較は認定目的で実行しなかった。
🟠 [Next] 非競合hostで比較を再取得し、実weightでload時RSS／GPUピークを測定する。
⭕️ [Pending] 実OOM／allocatorピーク・GPU／E2E認定。OSのMemoryErrorを意図的に発生させる検証は未実施。


🟢 [Done] 第13工程（2026-10-11）：比較runnerにwarmup、反復数、明示byte予算、部分常駐suiteを追加。
保持tensorの生涯ピークを記録し、Lease残存・予算超過を検出する。
[解析runner](../scripts/analyze_strata_comparison.py)はrun総時間の(max−min)/medianを算出し、
5%以下を安定性条件とする。数値一致・保持契約と速度判定を分離し、合成結果で既定採用しない。
関連19 unit tests、native関連13 tests、ruff成功。

[全常駐比較](evaluation/strata-stable-resident-analysis-cpu-2026-10-11.json)：
各方式6 fresh process、32要求warmup後に各1,024要求、24 trial／24,576要求すべて一致。
計測offでrun中央値の中央値はbaseline 70.969、逐次LRU 190.948、grouped LRU 178.490、
gather 56.427 μs。全常駐候補は要求中miss／evictionとも0。
総時間のrun変動は139–352%で、安定計測には未到達。
確認時には別のSwiftコンパイル等が高負荷で動作しており、host競合が疑われる。
この速度差を優位性や改善率として認定しない。

[部分常駐比較](evaluation/strata-partial-residency-analysis-cpu-2026-10-11.json)：
全Expert 36,864 bytesに対して保持予算20,000 bytes、entry上限4（byte制約を検証）。
各方式6 fresh process、32要求warmup後に各512要求、24 trial／12,288要求すべて一致。
候補の保持ピーク18,432 bytes、Lease残存0、close後保持0、拒否0。
512要求のmiss／eviction増分は逐次LRU 320／320、grouped LRU・cost_frequencyとも256／256。
groupedは再読み込み20%減だが、cost方式の優位はこのfixtureでは認められない。
run中央値の中央値はbaseline 110.344、逐次LRU 498.562、grouped LRU 377.542、
grouped cost 402.938 μs。run変動87–139%で速度認定なし。
baselineは全Expert保持の参考値であり、同じメモリ制約を満たす候補ではない。

[1 Expert予算fallback](evaluation/strata-single-expert-fallback-cpu-2026-10-11.json)：
10,000 bytes予算、top-k=2、grouped指定でも逐次経路にfallback。
64要求数値誤差0、Expert eval128回、保持ピーク9,216 bytes、要求中miss／eviction88／88、
Lease残存0・close後保持0を確認。分解計測onのため速度比較には使用しない。
保持上限は管理下tensorのみ。現在は読み込み後に退避するため、一時allocationやRSSのOOMを保証しない。
🟠 [Next] 他の高負荷処理と競合しない環境で同条件比較を再取得する。
⭕️ [Pending] 実weight・GPU／RSS、読み込み時ピークと実OOM、chat E2E／長時間認定。


🟢 [Done] 第12工程（2026-10-11）：全Expert常駐の`install_qwen3_moe_gather`候補を実装。
Router indicesをhostへ取り出さず、MLX gather_mm／gather_qmmへ直接渡す。
全Expertをmanifest検証してbank化し、元のSwitchGLU参照を置き換える。
要求中はlazy graphを返し、最終callerのmx.evalへ評価をまとめる。
任意indicesの内容検証はhost同期を避けるため行わず、既存blockのRouter出力のみを契約とする。
全Expertが予算内に収まらなければinstallを拒否し、元blockを維持する。
startup予算は元source tensor＋file bytes×2で保守的に判定するが、allocator overheadは保証外。
動的常駐・offloadの代替ではなくall-resident比較候補。closeは要求完了後のidle pointのみ。
native含む12 tests成功、dense／4／8-bitと40-token入力の数値一致・元source解放・予算拒否を確認。

[順序rotation比較](evaluation/strata-gather-analysis-cpu-2026-10-11.json)：
baseline／逐次／grouped／gather各3 fresh process、各64要求、12 trial／768要求すべて数値一致。
再利用中央値はbaseline 50.917 μs、逐次162.792 μs、grouped 138.125 μs、gather 55.958 μs。
gatherは64要求のRouter to_host 64→0、Expert build区間128→64、内部Expert eval128→0。
3 projectionのgather演算を1 tokenにつき1回ずつ発行し、ExpertごとのPython loopを省いた。
最終caller evalは残る。全常駐で要求中loadは全候補0。
gatherのrun総時間変動は約27.9%、baselineより遅く、P3資格／既定採用はfalse。
host同期とgraph構築を同時に変更しているため、個別の寄与率はこの比較だけでは確定しない。

判断：全Expertが収まる場合は既存MLX gather経路を基準にするのが適切。
独立file＋host Router＋Python Expert loopの参照経路が追加固定費を生むことを示す
有望な結果だが、Strata移植によるbaseline超えの証明ではない。
🟠 [Next] より安定した同条件計測と、メモリ不足時の部分常駐経路を切り分ける。
⭕️ [Pending] 実MoE weight・GPU／RSS／chat E2Eと長時間認定。

🟢 [Done] 第11工程：`grouped=True`の明示候補でtop-kの全Leaseを保持し、
未集約Expert出力を1回のmx.evalで確定してから解放する。既定は逐次実行。
entry／byte予算不足なら逐次fallback。byte判定はfile header込みの保守的な見積り。
source形式はdense／affine 4／8-bit、Routerと最終集約を維持する。
grouped時telemetry sampleはtop-k group単位、selected_experts数は維持する。
native含む11 tests成功、dense／4／8-bit数値一致、entry／byte上限fallback、失敗時解放を確認。

[同一sourceの比較](evaluation/strata-grouped-analysis-cpu-2026-10-10.json)：
逐次とgroupedの全Expert事前常駐CPU比較18 trial全件数値一致。
8要求のExpert eval回数は16→8。
再利用中央値（μs）はLRU 160.000→152.500、cost 183.291→185.125。
速度改善は一貫せず、run変動5%超、baselineより遅いため採用しない。
全Expert常駐でevictionがないので、このLRU/cost差を退避方式の優劣とは扱わない。
各campaignは別時刻の取得であり、interleaved取得・実モデル・GPU・RSS・P3資格は未検証。

🟠 [Next] Router host取り出しとPython／MLX graph発行の固定費を調査し、
常駐Expertをまとめたgather演算への接続候補を検討する。同期削減だけで高速化したとはしない。

🟢 [Done] 第10工程：任意有効化の固定13 phaseカウンターを追加し、
checksum read/hash、file stat、MLX load/weight eval、Router eval/to_host、Expert build/eval、
acquire/release、出力assembly/evalを分離した。標準設定は計測無効。
[分解結果](evaluation/strata-phase-analysis-cpu-2026-10-10.json)：profile、計測off、
全Expert事前常駐profileを各9 fresh process、合計27 trial取得し全件数値一致。
再利用時の3 trial中央値（μs）は次の通り。

| 条件 | baseline | LRU | cost_frequency |
|---|---:|---:|---:|
| 計測off | 64.959 | 226.333 | 177.250 |
| 分解計測on | 80.250 | 634.708 | 415.292 |
| 全Expert事前常駐・計測on | 142.542 | 323.166 | 247.625 |

計測on/offは別時刻の取得でhost条件と混ざるため、この差を計測overheadの純粋な値とはしない。
各runの総時間変動は5%を超え、厳密な寄与率や方式優位は認定しない。
事前常駐では全candidate trialで要求中miss増分とchecksum/load/weight-evalのcall数が0。
それでもbaselineを上回る遅延が残るため、I/Oだけが遅さの原因ではない。
8要求合計のphase中央値（LRU／cost、μs）は次の通り。

| phase | 分解計測on | 全Expert事前常駐 |
|---|---:|---:|
| acquire（読み込み内包） | 2507.84 / 1452.12 | 57.96 / 44.96 |
| checksum read | 295.45 / 215.92 | 0 / 0 |
| checksum hash | 39.71 / 27.08 | 0 / 0 |
| MLX load | 680.37 / 213.79 | 0 / 0 |
| weight eval | 796.67 / 373.96 | 0 / 0 |
| Expert eval | 1536.46 / 677.91 | 874.04 / 586.96 |
| Router eval | 452.58 / 197.04 | 313.79 / 187.79 |
| Router to_host | 396.33 / 154.79 | 253.08 / 157.46 |

中央値はphaseごとに算出し、加算してrequest総時間にしない。acquire内のload等は重複。
Router evalはgate/top-k実行を含み、純粋な同期待ちではない。MLX内部I/Oはload/evalと分離できず、
checksum readはOS cacheを含むhost read。物理SSD cold I/OやGPU counterは測定外。

現時点の原因判断：hash計算より読み込み・評価が大きく、I/Oを除いてもExpert単位の
逐次evalとRouter host取り出しの負担が残る。これは有力な構造要因であり寄与率の確定ではない。
🟠 [Next] top-k全体のLeaseを保持して評価境界をまとめる候補を、同じ数値suiteと計測で比較する。
小容量fallbackと大容量出力上限を維持し、改善が測定ノイズを超えるまで既定採用しない。
関連17 tests・native含む10 tests・Ruff成功。実Qwen品質／GPU／chat E2Eは未認定。

🟢 [Done] 第9工程：`scripts/compare_strata_experts.py`でbaseline／LRU／cost_frequencyを
各3 fresh process、順序をrotationして比較する合成CPU runnerを実装。
[測定記録](evaluation/strata-synthetic-qwen3-cpu-2026-10-10.json)は9 trialすべて数値一致。
初回の3 trial中央値はbaseline 0.194 ms／LRU 1.266 ms／cost 0.703 ms。
後続7呼び出しの中央値を各trialで取り、その3 trial中央値は0.135／0.336／0.213 ms。
hit/miss、常駐量、close後の0 byte、source SHAを保存する。
export・初期化は計測外、OS file cacheはresetしない。baselineは事前に評価済みsource重み、
候補は空の常駐cacheから開始する。絶対値は小さい合成blockのCPU時間で、モデルtokens/sではない。
cost方式もbaselineより遅いため採用せず、P3資格・自動適用はfalse。
量子化品質suite、RSS、GPU、chat E2E、電力、安定性は測定していない。

```sh
/path/to/mlx-python scripts/compare_strata_experts.py --output /tmp/new-strata-report.json
```

🟠 [Next] I/O・checksum・load・host Router同期を個別計測し、実互換weightを使った
メモリ適合性と性能のtradeoffを比較する。合成比較の改善率を実モデルへ外挿しない。

🟢 [Done] 第8工程：`install_qwen3_moe_residency`でMLX-LMの標準
Qwen3MoeSparseMoeBlockのswitch_mlpのみを明示置換する参照hookを実装。
gate、softmax、top-k、norm_topk_prob、最終集約は既存blockに任せる。
hookは未集約Expert出力を元の形状で返す。各Expertを逐次実行するためtop-k全体を
同時常駐させず、1 entry予算でも実行できる。Router結果のhost同期を伴い、速度向け設計ではない。
manifest全Expertの存在とsource量子化形式を確認してから置換する。
hookは元SwitchGLUを保持しない。別のcallerがsource参照を保持している場合や
allocator内部cacheについては解放を保証しない。baseline復帰は元モデル再読み込みで行う。
要求開始前にinstallし、phaseはsafe pointで明示指定する。自動phase推定やdaemon既定採用は行わない。
最大4096 token、未集約出力64 MiB上限。telemetryはExpert演算ごとのsample。

実MLXを含む12 tests成功。合成Qwen3ブロックのdense／4／8-bitとtop-k正規化
あり／なしの6構成で元blockとの誤差1e-5未満、gate identity維持、元SwitchGLU
weakref解放、1 entry動作、phase記録、重複Router拒否を確認。
これはQwen3の実装構造への接続検証でありQwen3.8/Qwen4Exp対応の証明ではない。

```python
hook = install_qwen3_moe_residency(
    sparse_moe_block, backend, executor, layer=layer_index, phase="prefill",
)
# Requests must be drained before changing execution phase.
hook.set_phase("decode")
```

🟠 [Next] 互換モデルの実weightでbaseline／LRU／cost_frequencyを比較し、
ファイルI/O・host同期・decode latencyを測定する。標準採用は既存P3判定の後。
⭕️ [Pending] 現在metadataのみのQwen3.8/Qwen4Exp実weightと専用architecture対応。

🟢 [Done] 第7工程：標準MLX-LM SwitchGLU/SwiGLUから独立Expertを書き出す
`export_switch_experts`を実装。denseまたは同一設定のaffine 4／8-bitに限定し、
Linear bias・独自activation・混在形式を拒否する。layer番号から全Expertを列挙し、
新規directoryへ保存して最後に完成manifestを公開する。既存directoryは拒否する。
失敗時は部分artifactを残し、完成manifestは作らない。途中結果はfrom_manifestで利用できない。
exportはoffline処理でありsource重みを保持する。runtimeで元SwitchGLUを残したまま
置換してもメモリ削減にならないため、runtime所有権の切り替えは別工程で扱う。
tensor／file／合計file容量を検査するが、file容量の判定は保存後で、source／allocator overheadは上限外。
実MLXを含む11 tests成功。合成dense／4／8-bit SwitchGLUからexport→manifest検証→
token別実行を行い、元SwitchGLUのRouter集約出力との差1e-5未満を確認した。
これは実APIでの合成構造検証であり対象Qwenの出力品質・速度の認定ではない。

```python
export_switch_experts(
    {layer_index: switch_glu}, new_directory, model_sha256=model_digest,
    maximum_expert_bytes=expert_limit, maximum_total_bytes=artifact_limit,
)
```

🟠 [Next] architectureを限定したモデル接続と、source重みのruntime所有権を移す手順。
⭕️ [Pending] Qwen対象weight取得後のE2E・品質・LRU比較。

🟢 [Done] 第6工程：`expert-manifest.json`の検証と`from_manifest`を実装。
schema_version=1、model_sha256、quantization_bits、group_size、expertsを要求する。
expertsはlayer・expert・size_bytes・sha256の配列。ファイル名は既存規則から生成し、
manifestは4 MiB／65,536 entry以内、重複field／entryを拒否する。
呼び出し側のexpected_model_sha256と一致させ、cache miss時の読み込み前にサイズと
SHA256を1 MiB単位で検証する。量子化設定はmanifestから取得する。
明示設定の旧constructorは合成試験用途として残す。実接続ではfrom_manifestを使う。
実MLX演算を含む10 tests・関連15 tests・Ruff成功。
チェックサムは変更検出であり署名や品質認定ではない。model_sha256の算出規則は
生成側と利用側で同一に固定する必要がある。trusted offline artifactが前提で、
検証とmx.loadの間の敵対的変更や既に常駐したentryの再検証は扱わない。

```python
backend = MLXFileExpertBackend.from_manifest(
    expert_root, expected_model_sha256=model_digest, maximum_file_bytes=file_limit,
)
```

🟠 [Next] 実モデルから独立Expertを生成するexportとarchitecture別Router接続。
⭕️ [Pending] 対象Qwenの実weightによる品質／E2E比較。

🟢 [Done] 第5工程：独立Expertファイルの4-bit／8-bit affine量子化を追加。
`quantization_bits`と`group_size`を明示し、各gate/up/downにweight(uint32)、
scales、biasesを要求する。量子化biasesはaffine補正値であり、Linear biasではない。
packed容量、scale形状、入出力次元を検査し、`mx.quantized_matmul`で重みを展開せず演算する。
resident bytesにはpacked weight・scales・biasesを含む。
実MLX CPUテスト6件成功。4／8-bit × group32／64の合成重みについて、
同じ量子化重みのdequantize参照との誤差1e-6未満と不一致groupの拒否を確認。
この一致は元の非量子化モデルに対する品質保証ではない。
group128はAPI候補として受理するが今回の数値検証範囲外。
関連17テスト中13件成功、native必須4件は実MLX環境で成功。Ruff成功。
🟠 [Next] bits／group／model identityを結ぶartifact manifestと実Router接続。
ファイル単体にはbits／groupの自己記述がないため、信頼できる生成手順と明示設定を前提とする。
速度、量子化による品質変化、実Qwen／GPU推論は未認定。

🟢 [Done] 第4工程：`execute_routed`でtoken別の複数Expert選択と重み付き集約を実装。
最大4096行、各行のtop-kを逐次実行し、全Router行を読み込み前に検証する。
選択順と未正規化の重みをそのまま使い、top-kの再計算はしない。
実MLX CPUテスト4件成功：異なるtoken選択、2 Expert集約、退避、数値誤差1e-6未満、
不正Routerの事前拒否、入力次元不一致時のLease解放を確認。関連CPUテストは12件成功、
native必須3件は別の実MLX環境で成功。Ruff成功。
本経路は参照用逐次実行であり、実モデルRouterへの自動接続や量子化、速度改善は未実装／未認定。
🟠 [Next] 量子化Expertの明示形式と数値比較。実モデル接続は対応architectureを限定する。

🟢 [Done] 第3工程：`MLXFileExpertBackend`を実装。Expertごとの独立safetensorsから
浮動小数点・biasなしSwiGLUのgate/up/downを読み込み、形状検査とmx.eval後に常駐管理へ渡す。
解放時はtensor参照を破棄する。allocatorがOSへ直ちにメモリを返すことは保証しない。
共有SwitchGLU配列のsliceをcacheしても元配列が残るため、その方式は採用しない。
各token groupで共通のRouter選択を保持して重み付き集約を行い、結果のmx.evalをLease解放前に完了する。
入力のRouter選択自体はbackend callerが渡す。モデルへの自動hookは未実装。

実MLX CPU配列で小さな独立Expertファイルを生成し、退避／再読み込み後の参照計算との
誤差1e-6未満・Lease解放・byte計上を検証した。これは合成重みの演算検証でありモデル品質や速度認定ではない。
sandboxではMetal device unavailableとなり、Metalにアクセスできる環境で成功した。
ローカルの`qwen3.8-flash-next-metadata`はconfigのみで、実モデルのExpert重みは未確認。
⭕️ [Pending] この対象モデルのE2E比較には実weightと量子化／architecture adapterが必要。
🟠 [Next] 複数Expert・量子化形式・実Routerの接続を個別に検証する。
独立ファイルは信頼できるoffline生成物を前提とし、読み込み前のfileサイズ上限は
MLX allocatorの厳密なpeak上限を保証しない。大容量sourceをそのまま読み込まない。

🟢 [Done] 第2工程：`ResidentExpertExecutor`でRouter選択と常駐管理を接続。
選択順・重みを保持し、全ExpertのLeaseを取得してからbackendの同期consumerを呼ぶ。
consumerはGPU結果をmaterializeしてから返す契約。lazy結果の完了は本接続部で保証しない。
途中の取得失敗・演算失敗では取得済みLeaseを解放し、成功時だけphase別telemetryへ記録。
cache hitは各acquireで返したLeaseから取得するため、共有カウンターの差分に依存しない。
関連22 tests・Ruff成功。テストbackendでの接続検証であり実MoE推論や高速化の証明ではない。
既存runtime内で実モデル用load_expert／release_expert実装は確認できていない。

🟢 [Done] 最初の工程として既存ExpertResidencyManagerへ明示選択の
`eviction_policy="cost_frequency"`を追加した。既定は`lru`。
保持スコアはアクセス回数 × 同期load_expert呼び出し時間 ÷ resident bytes ÷ 最終使用からの論理経過時間。
低いスコアから退避し、同点は最終使用時刻・ExpertKeyで決定する。
統計は常駐entry内だけに保持し、回数は100万で飽和する。退避後の履歴は保持しない。
Lease中のentryは候補から除外し、既存のbyte／entry上限と遅延resizeを共有する。
Router選択やモデルの重みには手を加えない。

load時間はホスト側呼び出し時間であり、MLXの遅延GPU実行やSSD missの実時間を
保証しない。backendは実際に利用可能なresourceを返す必要がある。
このスコアは比較候補で、最適性や速度改善の証明ではない。

🟢 [Done] 関連31テスト成功。高コストentryの保持、Lease保護、遅延予算変更、
未知policy拒否、既存LRUと投機実行・P3判定を検証した。
この工程では本番設定変更・GPU負荷・依存更新を行っていない。

🟠 [Next] 実MoE backendのRouter観測とExpert resource操作を確認し、
prefill／decode別の測定を収集する。LRUと同一モデル・精度・workloadで独立3回以上比較し、
既存P3判定へ接続する。品質、memory、TTFT／TPOT、P1／対象P2前提を満たすまで標準採用しない。

🔴 [Later] コストに基づく先読み制御、互換draft／Prompt Lookup、KV圧縮と融合演算。
⭕️ [Pending] 対応artifactと実機で検証できないMTP／ANE経路。

参照設計：https://github.com/Niko1221/Strata/blob/main/docs/HOW_IT_WORKS.md
直接のソース移植は行っていない。将来移植時は参照commitとlicenseを固定する。

## English

🟢 [Done] Stage24 (2026-10-13): explicit Strata flags integrated into mlx_server.
Root, expected model SHA, resident bytes/entries and file cap are required together.
Reviewed MLX-LM0.32.0, one local worker only. Preload/install before serving; disable provider batching,
reject model changes, adapters, drafts and concurrent generation. Join generation worker, synchronize MLX,
then close runtime; generator construction failure also closes it.
Single-sequence forwards label width>1 as prefill and width1 as decode, including singleton prompt chunks.
These labels are forward-shape telemetry, not exact HTTP prompt boundaries.
21 serving-related and19 native tests pass; distributed/draft admission and synchronize-before-close ordering are tested.
⭕️ [Pending] Real-artifact HTTP streaming/cancel/cache E2E, long-run stability and speed qualification.


🟢 [Done] Stage23 (2026-10-13): `StrataRuntime.from_model()` discovers all standard Qwen3 MoE layers,
checks exact manifest layer/expert coverage and expected model identity before installation.
Other architectures, missing/extra experts and empty MoE sets are rejected. Full logits from a synthetic
2-layer model match, including attention/norm/embeddings;19 native tests pass.
This accepts a loaded model and does not reduce startup source-bank memory.
⭕️ [Pending] Safe server batching/phase integration, real-model E2E and speed qualification.


🟢 [Done] Stage22 (2026-10-13): explicit synchronous Strata integration for standard Qwen3 MoE.
Public `vllm_apple.StrataRuntime` accepts `{layer_id: moe_block}` and a manifest-bound backend.
Layers share one entry/byte budget, LRU or cost_frequency policy and phase telemetry.
Modes are sequential (default), grouped and row_batch (explicit, default batch limit8).
Lifecycle: set_phase, resize, snapshot, idempotent close and context manager.
Failed installation restores all changed switches; successful installation retains no source bank.
Close rejects active leases and then disables hooks and phase changes. Reload to restore baseline.
Concurrent inference/lifecycle changes are unsupported. All-resident gather remains a separate opt-in API.
Predictor/SSD helpers exist; automatic speculative prefetch and default adoption remain disabled.

Export immutable experts offline, open from_manifest with expected model SHA, install before requests,
set prefill/decode before each phase and close after inference. Release external source-bank references.
This is an explicit loader integration point; server auto-install and other architectures are unqualified.
Multi-layer parity, shared bounds, shrink fallback, source release, rollback and shutdown pass:
18 native and41 supporting tests, plus ruff.
⭕️ [Pending] Quiet-host comparison, real-weight/GPU E2E/OOM and automatic server/default qualification.


🟢 [Done] Stage21 (2026-10-13): [diverse prefill under exact payload budgets](evaluation/strata-diverse-prefill-exact-payload-memory-analysis-2026-10-13.json).
Eight fresh CPU/Metal workers compare 1-row and 8-row groups across five fixtures, cold and warm (80 requests).
The 2,359,296-byte budget bounds retained/reserved payloads; leases and retention after close are zero.
Maximum error is 2.235e-7. Warm eval counts: repeat128→16, mixed128→28, alternating128→128,
reversed32→32, random128→120. Mixed warm MLX active peaks rise from 4,195,352 to 4,314,136 bytes
on CPU and from 4,195,352–4,203,544 to 4,535,320 bytes on Metal. Batch scratch increases.
RSS includes fixtures, references, modules and allocator cache; no RSS reduction is qualified.
The probe supports `--memory-only` and `--exact-payload-budget`. This is synthetic functional/memory evidence.
🟠 [Next] Separate install metadata costs and request overhead.
⭕️ [Pending] Quiet-host automation continues; speed, real-model/system-GPU/OOM/E2E and default qualification.


🟢 [Done] Stage20 (2026-10-13): grouped admission uses conservative payload bounds rather
than whole file sizes. Installation reads each expert prefix/size and caches an immutable tuple;
requests compare payload totals with current entry/byte limits. Sequential fallback remains.
Startup metadata work increases; no new per-request header I/O. Misses still verify manifest identity,
settings/checksums, reserve before allocation and check actual tensor bytes. Files must remain immutable.
[Exact-payload evidence](evaluation/strata-payload-admission-analysis-cpu-2026-10-13.json):18,432-byte
budget,d64/e4/top2/q4/g32,9 fresh trials/576 requests/4,608 tokens,zero error.
Retained/reserved peak18,432,no leases/rejections,zero retention after close. Per64 candidate requests,
eval512→64/acquire1024→128,32 misses/evictions in both modes. Header-only false fallback removed.
Stage19 tight-budget results remain historical. Profile-on/no host qualification: no certified speedup.
16 native-related tests pass, including exact-budget batching, resize fallback, missing-file install
leaving source intact and checksum-failure lease cleanup, plus existing dense/4/8-bit/ownership regressions.
Weight reservation does not bound scratch/RSS or certify real OOM. Resume preflight CPU max~1,012% failed;
automatic quiet comparisons continue with fresh evidence after this source change.
🟠 [Next] Diverse prefill peaks under exact payload budgets and startup/request costs.
⭕️ [Pending] Uncontended comparison, real-weight/GPU E2E,OOM and default qualification.


🟢 [Done] Stage19 (2026-10-12): [diverse prefill/memory probe](evaluation/strata-diverse-prefill-analysis-cpu-metal-final-2026-10-12.json).
Synthetic d512/i1024/e4 affine4/group32: repeat, mixed-length runs, alternating, reversed and seeded random selections.
32 reversed tokens,128 otherwise; sin/random vectors, authoritative ordered selections, not text/tokenizer/router quality.
Native SwitchGLU references generated separately; workers discard temporary source banks before measurement.
CPU timing:3 fresh workers per mode,1 warmup+4 requests per case. Memory:CPU/Metal2 fresh workers per mode,
2 requests (cold then retained/cache), counters enabled only in memory workers.
14 workers/120 timed+80 memory requests; max error2.2352e-7. Budget2,360,838 bytes allows two files but
not three payloads; retained/reserved peak2,359,296, no active leases, zero retention after close.
Eval counts per request: repeat128→16,mixed128→28,alternating128→128,reversed32→32,random128→120.
Warm reloads remain0/48/256/0/153 in both modes. CPU reference medians(ms):57.226→50.880,
89.857→87.154,220.474→225.028,14.312→14.097,160.961→155.714. Spread5.6–173%, no host qualification;
no speed advantage certified. Little continuity gives little batching benefit; I/O remains unchanged.
Warm mixed-run active peak increases CPU4.001→4.114 MiB,Metal4.001–4.028→4.325 MiB;
CPU mixed-run cache1.027→1.465 MiB. Warm sampled RSS overall330.9–335.4 MiB includes imports,
fixtures and cache; no RSS advantage established. Peaks are process-wide, not isolated activation scratch.
Requested1ms RSS samples may miss transients; HWM is lifetime-wide; MLX peak reset/sync per request.
Unified RSS/MLX measurements overlap, not dedicated VRAM. Speed/memory workers are separate.
[Tight-budget control](evaluation/strata-diverse-prefill-tight-budget-analysis-2026-10-12.json):
2 payloads exactly trigger conservative header-inclusive fallback, repeat eval256→256, parity passes.
🟠 [Next] Payload-based group admission, real prefill continuity, scratch/RSS and uncontended timing.
⭕️ [Pending] Real-model/GPU E2E, system GPU/OOM and speed/default qualification.


🟢 [Done] Stage18 (2026-10-12): opt-in `grouped=True, maximum_group_rows=8` batches only
contiguous tokens with identical ordered expert selections. Range1–64, default1; no regrouping of
reversed selections. Entry/byte shortages retain sequential fallback; leases survive batch evaluation.
The original block retains routing/top-k/per-token aggregation. Scratch increases and is outside the
weight budget. Telemetry samples/expert counts describe execution groups, not router token counts.
Native-related15 tests pass for dense/4/8-bit parity, order boundaries, caps, fallback and failure cleanup.
[Functional CPU suite](evaluation/strata-contiguous-row-analysis-cpu-2026-10-12.json):9 fresh trials,
576 requests/4,608 tokens, zero error. Repeated token vectors intentionally favor contiguous routing.
Per64 candidate requests: expert eval512→64, build spans/acquires1024→128; host extractions64 and
loads/misses/evictions32 unchanged. Retained/reserved peak18,432 bytes under20,000, no leases/rejections,
zero retention after close. Profile-on, no host qualification: no certified speedup or default adoption.
Reference medians grouped606.917/batch232.459/baseline69.583 us; not production performance.
The scheduled1-token comparisons keep the default1-row route.
🟠 [Next] Diverse prefill routes, scratch/RSS and low-load comparisons.
⭕️ [Pending] Real-weight/GPU/E2E and speed/default qualification.


🟢 [Done] Stage17 (2026-10-12): READY/GO/DONE/ACK separates startup, ready wait and requests.
Startup includes imports/reference construction/export/preload/warmup. After READY the parent waits for
quiet conditions, starts request monitoring and sends GO. DONE stops monitoring before ACK allows cleanup.
Approx1s startup and100ms request sampling; a short tail merges with the previous sample without losing CPU delta.
Request windows include parity/controller handoff, individual latencies exclude both. Qualification uses request
records independently of startup. Missing request records and failed ready gates fail closed.
`--quiet-timeout` allows3–180s, default30. CPU/PID thresholds unchanged; aggregate includes worker/observer.
34 related unit tests and ruff pass; runtime defaults unchanged.
[Partial](evaluation/strata-phased-partial-analysis-cpu-2026-10-12.json): one baseline trial/4,096 requests,
zero error; startup2.50s at140–222% CPU, request window0.293s at161–218%, both fail load conditions.
A subsequent launch gate detects journal_bench activity. [Resident](evaluation/strata-phased-resident-analysis-cpu-2026-10-12.json)
starts zero workers. [120s retry](evaluation/strata-phased-partial-retry-analysis-cpu-2026-10-12.json)
also starts zero workers;116 samples at196–959% aggregate CPU, median234%, other Python activity persists.
Raw trials/failures retained, no speed qualification or promotion. Request contention remains observable after startup separation.
🟠 [Next] Run all variants during host periods satisfying request-window gates.
⭕️ [Pending] Full uncontended comparison, real-weight/system-GPU/OOM/E2E qualification.


🟢 [Done] Stage16 (2026-10-12): repeated CPU comparisons with the host load gate.
[Resident](evaluation/strata-quiet-resident-analysis-cpu-2026-10-12.json):14 trials/57,344 requests;
[partial](evaluation/strata-quiet-partial-analysis-cpu-2026-10-12.json):16 trials/16,384 requests.
All73,728 requests have zero error; profile off,64 warmup requests, fresh workers and rotated order.
Resident medians: baseline47.667/sequential143.459/grouped121.125/gather49.209 us.
Sequential/gather spread0.70/1.56%; grouped13.31%,baseline189.33%. The last preflight gate stopped
this campaign before the planned16 trials, and host conditions fail in some trials: no qualification.
Partial budget20,000 bytes versus full bank36,864,1,024 requests per worker,4 workers per variant:
baseline45.781/sequential217.969/grouped LRU189.250/grouped cost199.135 us;
spreads1.74/1.90/0.33/2.44%. All meet timing stability, but host-pass counts are2/4,4/4,3/4,2/4.
Only sequential LRU passes both criteria in every trial; the complete uncontended comparison remains false.
Baseline retains all weights, not the same memory budget. Candidate retained/reserved peak18,432 bytes,
no rejections/leases, zero retention after close. Timed misses/evictions640 sequential versus512 grouped,
confirming20% fewer reloads. Source hashes/contracts checked; no runtime defaults changed.
🟠 [Next] Gather host-qualified results for every variant and separate startup/request load windows.
⭕️ [Pending] Real-weight load/inference RSS/MLX peaks, system GPU, real OOM and E2E qualification.


🟢 [Done] Stage 15 (2026-10-11): `--quiet-host` requires three consecutive approx-1s
samples with system CPU<=200%, visible background process CPU<=75% and no PID churn.
A controller thread records load during workers; analysis requires host checks and <=5% run-total spread.
Aggregate CPU includes benchmark load and inaccessible processes. CPU sampling does not certify disk/GPU isolation.
Empty/incomplete campaigns cannot qualify. 30 related unit tests and ruff pass.
[Two bounded quiet-host attempts](evaluation/strata-quiet-resident-retry-analysis-cpu-2026-10-11.json)
failed the 30-second gate while Rust/Swift compilation remained active: zero timing workers started.
⭕️ [Pending] Uncontended speed comparison remains unfinished under current host load.
[Memory evidence](evaluation/strata-load-memory-analysis-cpu-metal-2026-10-11.json): Apple M4, 32 GiB;
synthetic d1024/i2048/e4 affine4/group32, 4.5 MiB per expert. Fixture export in a separate process;
no original model bank in workers. CPU/Metal-default, capacities1/2, pre/post-load eviction,
two fresh processes each: final16 trials /96 acquires. Checksums/load/eval included; inference excluded.
Per-load sync/reset MLX peak/sync, requested1ms RSS sampling plus lifetime RSS HWM.
Active peaks consistently drop from9 to4.5 MiB (capacity1),13.5 to9 MiB (capacity2), on both devices.
Metal sampled RSS ranges: capacity1,68.81–71.02 to64.25–64.34 MiB; capacity2,73.36–75.12 to69.97–70.69 MiB.
After close: no weights/leases/active allocations, cache remains; test-only cache clear reduces cache to zero.
Sampled RSS peaks may miss short transients; HWM is lifetime-wide. MLX allocator statistics are not
system-wide/dedicated GPU memory, and unified RSS/MLX measurements overlap. No runtime cache-clear change.
🟠 [Next] Uncontended timing and real-weight load/inference peaks.
⭕️ [Pending] Real OOM, system-wide GPU memory and long-run E2E certification.


🟢 [Done] Stage 14 (2026-10-11): MLX payload admission now reserves file size minus the
8-byte prefix and header before loading. The manager evicts unpinned weights first, rejects
insufficient/pinned capacity before load, and rejects/releases a resource exceeding its reservation.
`peak_reserved_bytes` tracks retained weights plus incoming payload. Backends without this optional
contract keep the previous post-load eviction behavior. Files must remain immutable; prefix reads
are included in file_stat spans. Failed loads can leave an evicted cache, but retry remains possible.
[Functional evidence](evaluation/strata-preload-reservation-cpu-2026-10-11.json): 192 requests, zero error;
grouped LRU/cost at 20,000 bytes reserve at most 18,432 bytes, single-expert fallback at 10,000 reserves
9,216 bytes. No leases remain; close retains zero bytes. 24 unit / 14 native-related tests and ruff pass,
including pre-load rejection, reservation violation cleanup and simulated MemoryError recovery.
Reservations bound logical weight overlap, excluding loader scratch, checksum buffers, activations,
allocator caches, RSS and GPU memory. Real OS OOM is untested. Other high-load Rust compilation
was observed; these profile-on runs validate functionality, not speed.
🟠 [Next] Uncontended comparisons and real-weight load-time RSS/GPU peaks.
⭕️ [Pending] Real OOM/allocator and GPU/E2E qualification.


🟢 [Done] Stage 13 (2026-10-11): added warmup, repeated fresh workers, explicit byte budgets,
partial-residency suite and retained-tensor peak tracking. The analysis runner separates parity/retention
contracts from timing stability (run-total spread <=5%). 19 unit and 13 native-related tests pass; ruff passes.
[All-resident](evaluation/strata-stable-resident-analysis-cpu-2026-10-11.json): 24 trials / 24,576 timed requests,
32 warmup requests per worker; all parity passes, no timed candidate misses/evictions.
Median of run medians: baseline/sequential/grouped/gather 70.969/190.948/178.490/56.427 us.
Run-total spread 139–352%; concurrent high-load Swift compilation was observed. No speed qualification.
[Partial residency](evaluation/strata-partial-residency-analysis-cpu-2026-10-11.json): full bank 36,864 bytes,
20,000-byte retained budget, entry cap 4. 24 trials / 12,288 timed requests pass parity.
Candidate peak 18,432 bytes, no active leases/rejections, zero retained bytes after close.
Per 512 requests, misses/evictions are 320/320 sequential versus 256/256 grouped for both policies (20% fewer loads).
Median of run medians: baseline/sequential/grouped LRU/grouped cost 110.344/498.562/377.542/402.938 us;
spread 87–139%, no qualified speed advantage. Baseline retains all weights and is a reference, not a budget-compliant candidate.
[Single-expert fallback](evaluation/strata-single-expert-fallback-cpu-2026-10-11.json): 10,000-byte budget,
top-k=2 with grouped requested safely falls back to sequential, 64 requests with zero error, 128 expert evals,
peak 9,216 bytes, 88 timed misses/evictions, no leases and zero retention after close. Profile-on, not a speed comparison.
Loading happens before eviction: retained peaks do not bound transient allocations/RSS or certify real OOM safety.
🟠 [Next] Repeat under an uncontended host. ⭕️ [Pending] Real weights, GPU/RSS, load-time peaks/OOM and chat E2E certification.


🟢 [Done] All-resident device-index gather candidate removes host index materialization
and per-expert Python loops/internal evals. Twelve tests including native MLX pass
for dense/4/8-bit and 40-token parity, source release and budget rejection.
Twelve rotated fresh-process CPU trials (768 requests) pass parity: baseline/sequential/
grouped/gather reuse medians 50.917/162.792/138.125/55.958 us. Gather approaches baseline
but is slower and has 27.9% run spread, so no qualification/adoption. It requires all
experts resident; startup/allocator RSS, GPU and real models remain unverified.
🟠 [Next] Stable matched measurements and separate memory-constrained residency.

🟢 [Done] Opt-in grouped top-k pins all leases through one expert-result eval;
insufficient entry/byte budgets retain sequential fallback. Eleven native-related
tests pass including dense/4/8-bit parity and cleanup. Eighteen preloaded CPU
trials halve eval calls (16→8) but show inconsistent latency change:
LRU 160→152.5 us; cost 183.291→185.125 us. Spread exceeds 5%; no adoption.
These all-resident trials do not compare eviction policies. 🟠 [Next] Host routing
and Python/graph overhead, then bounded gather-based candidates.

🟢 [Done] Opt-in bounded host phase counters and 27 isolated synthetic trials separate
checksum I/O/hash, MLX load/eval, routing and expert execution. Preloading eliminates
all request-time miss/load/checksum calls but leaves candidates slower than baseline.
Per-expert synchronization and router host materialization remain substantial.
Run spread exceeds 5%; exact contributions and policy superiority remain unqualified.
Nested spans must not be summed; router eval includes gate/top-k, MLX load/eval may
include internal I/O. Separate-time controls do not isolate instrumentation overhead.
🟠 [Next] Evaluate grouped top-k completion with pinned leases and bounded fallback.

🟢 [Done] Nine fresh-process synthetic CPU trials compare baseline/LRU/cost policies
in rotated order; all pass parity. Median subsequent-call times: 0.135/0.336/0.213 ms.
Cost remains slower than baseline and is not adopted. Export/startup are excluded,
OS file cache is not reset, and no real-model/GPU/RSS/P3 qualification is granted.
🟠 [Next] Separate I/O/checksum/load/router-sync measurements and real-weight tradeoffs.

🟢 [Done] Explicit standard Qwen3 MoE switch hook preserves the original gate,
top-k normalization and aggregation. It returns unaggregated expert outputs and
retains no original switch bank. Install/change phase only at idle safe points;
baseline restoration reloads the source model. Twelve tests including native MLX
pass, with six synthetic dense/4/8-bit × normalized/unnormalized block comparisons
within 1e-5 and original-switch weakref release. This serial host-synchronized
reference path is not a speedup or Qwen4Exp qualification.
🟠 [Next] Real compatible weights, latency/I/O measurements and P3 comparison.

🟢 [Done] Offline standard SwitchGLU export supports uniform dense/affine 4/8-bit
experts in a new directory. Unsupported activations/bias/mixed formats are rejected;
the manifest is published last. Partial artifacts remain after failure without a
completion manifest. Eleven tests including native MLX pass; synthetic exports
match original SwitchGLU routed outputs within 1e-5. Source weights remain owned
by the offline caller; no runtime replacement or model performance is certified.
🟠 [Next] Architecture-specific hooks and runtime ownership transition.

🟢 [Done] A bounded manifest binds expected model SHA256, quantization and per-expert
size/checksum. Duplicate fields/entries and drift are rejected before loading on
cache misses. Ten tests including native MLX pass; 15 related tests pass.
Use `from_manifest` for integration. Trusted immutable offline artifacts are required;
checksums are not signatures, and concurrent mutation/cached-entry revalidation are
outside this contract. 🟠 [Next] Model export and architecture-specific routing hooks.

🟢 [Done] Explicit affine 4/8-bit expert support uses quantized matmul without dense
weight expansion. Packed weights/scales/affine biases are validated and counted.
Six native CPU tests pass; tested 4/8-bit × group32/64 agree with dequantized
quantized-weight references within 1e-6. Group128 is accepted but untested here.
This does not establish original-model quality or speed. 🟠 [Next] Bind bits/group/model
identity in an artifact manifest and connect real routing.

🟢 [Done] Per-token authoritative routing supports multiple experts and preserves
unnormalized weights and order. All rows are validated before loading; up to 4096
rows run sequentially. Four native MLX CPU tests pass, including numerical parity,
eviction and failure cleanup. This reference path does not certify throughput or
install real-model router hooks. 🟠 [Next] Explicit quantized-expert format and parity.

🟢 [Done] Independent-file MLX dense bias-free SwiGLU adapter validates tensors,
materializes weights/results and drops tensor references on eviction. Native CPU
array tests verify reference parity through eviction/reload; synthetic weights do
not certify model quality, GPU performance or OS memory reclamation. No model hook
is installed. Trusted offline files are required; file bounds do not cap allocator peaks.
⭕️ [Pending] Target-model E2E requires actual weights and architecture/quantization support.
🟠 [Next] Multi-expert, quantization and real-router integration validation.

🟢 [Done] A synchronous resident-expert execution bridge preserves router order and
weights, retains all leases until backend completion, cleans up failed acquisitions
and execution, and records bounded phase-specific hit telemetry. Twenty-two related
tests pass. The consumer must materialize GPU results before returning. Tested with
a test backend; real-model resource adapters and speedup remain unverified.

🟢 [Done] Opt-in cost/frequency eviction reuses the existing residency manager;
LRU remains the default. Scores combine capped resident access counts, synchronous
load time, bytes and logical age. Leased entries remain pinned; bounds and deferred
resize are shared. Metadata is discarded on eviction. Related tests: 31 passed.
Host load time does not prove GPU readiness or actual miss cost. No speedup is claimed.
🟠 [Next] Connect real MoE resource operations and phase measurements, compare against
LRU and use the existing P3 gate before adoption.
🔴 [Later] Prefetch, compatible speculation, KV compression and fusion.
⭕️ [Pending] MTP/ANE routes lacking testable artifacts or hardware.

## 简体中文

🟢 [Done] 第24阶段（2026-10-13）：mlx_server增加显式Strata参数。
root／expected model SHA／resident bytes／entries／file上限须同时指定。
仅支持reviewed MLX-LM0.32.0单local worker；服务前加载并安装，关闭provider batchability。
拒绝模型变更、adapter、draft及并发生成；生成worker停止并join后同步MLX并close runtime。
worker构建失败也close。仅允许单系列forward，width>1统计prefill，width1统计decode，
prompt末尾singleton也记为decode，因此并非精确HTTP prompt边界统计。
server相关21 tests及native相关19 tests通过；验证distributed／draft在安装前拒绝及同步后close顺序。
⭕️ [Pending] 真实artifact HTTP streaming／cancel／cache E2E、长期稳定性及速度认证。


🟢 [Done] 第23阶段（2026-10-13）：增加`StrataRuntime.from_model()`，自动发现标准Qwen3 MoE层，
安装前检查manifest全部layer／expert覆盖及expected model identity，拒绝其他architecture、缺失／多余Expert及空MoE集合。
包含attention／norm／embedding的两层合成模型全logits一致，native相关19 tests通过。
接口接收已加载model，不降低startup原权重bank内存。
⭕️ [Pending] server batch／phase安全接入、真实模型E2E及速度认证。


🟢 [Done] 第22阶段（2026-10-13）：标准Qwen3 MoE的显式同步Strata接入层完成。
公开`vllm_apple.StrataRuntime`接收`{layer_id: moe_block}`和manifest-bound backend。
所有层共享entry／byte预算、LRU／cost_frequency策略和prefill／decode统计。
默认sequential，可显式选择grouped／row_batch（默认batch上限8）。
提供set_phase、resize、snapshot、幂等close及context manager。
安装失败恢复所有已修改switch；成功后不保留原权重bank。
close拒绝active Lease，成功后拒绝hook再次执行及phase修改；恢复baseline需要reload。
不支持推理与生命周期操作并发。all-resident gather保留独立opt-in接口。
预测器／SSD辅助组件已有实现；自动预测prefetch及默认采用仍未启用。

先离线导出immutable专家文件，再使用expected model SHA打开from_manifest，要求开始前安装，
各prefill／decode区间前设置phase，结束后close，并释放外部source-bank引用。
这是显式loader接入点，未认证server自动切换或其他architecture。
多层数值一致、共享预算、缩小fallback、原bank释放、安装rollback及关闭后拒绝通过验证。
native18 tests、周边41 tests与ruff通过。
⭕️ [Pending] 低负载比较、真实weight／GPU E2E／OOM及自动server接入／默认采用认证。


🟢 [Done] 第21阶段（2026-10-13）：[严格payload预算下的多样prefill内存验证](evaluation/strata-diverse-prefill-exact-payload-memory-analysis-2026-10-13.json)。
CPU／Metal、1-row／8-row各两个独立worker，五种输入分别测量cold／warm，共8 trials／80请求。
预算2,359,296 bytes，保留及预约峰值均在预算内，Lease及close后保留均为0，最大误差2.235e-7。
warm eval：repeat128→16、mixed128→28、alternating128→128、reversed32→32、random128→120。
mixed warm MLX active峰值：CPU4,195,352→4,314,136 bytes，Metal4,195,352–4,203,544→4,535,320 bytes。
batch临时内存增加；RSS包含输入、参考、模块及allocator cache，不认定RSS降低。
probe增加`--memory-only`与`--exact-payload-budget`，本结果仅为合成功能／内存证据。
🟠 [Next] 分离install metadata成本与请求固定开销。
⭕️ [Pending] 低负载自动比较继续；真实模型／system GPU／OOM／E2E与默认采用认证。


🟢 [Done] 第20阶段（2026-10-13）：group admission改用保守payload上限而非file总量。
install读取各Expert prefix／size并缓存immutable tuple，请求按当前entry／byte预算判定，
不足仍逐次fallback。startup增加metadata读取，请求不额外读取header。load仍检查manifest／设置／checksum，
维持加载前预约和实际tensor量检查；文件使用期间必须immutable。
[恰好payload预算记录](evaluation/strata-payload-admission-analysis-cpu-2026-10-13.json)：
18,432-byte预算、d64／e4／top2／q4／g32，9个独立trial／576请求／4,608 tokens误差0。
保持／预约峰值18,432，无残留Lease／拒绝，close保持0。每64候选请求eval512→64、acquire1024→128，
两者miss／eviction32。消除仅因header引起的fallback。第19阶段严格预算结果保留为修正前历史，
profile on／无host资格，不认证速度提升。16个native相关tests通过，覆盖预算恰好时batch、
resize后fallback、缺file时不修改原block、checksum失败后Lease释放及dense／4／8-bit／所有权回归。
weight预算不保证scratch／RSS／实际OOM。再开preflight CPU最大约1,012%未达标，自动低负荷比较继续，
代码改变后要求新证据。
🟠 [Next] 严格payload预算下多样prefill峰值及startup／请求成本。
⭕️ [Pending] 无竞争比較、实weight／GPU E2E、OOM及默认采用认证。


🟢 [Done] 第19阶段（2026-10-12）：[多样prefill及内存记录](evaluation/strata-diverse-prefill-analysis-cpu-metal-final-2026-10-12.json)。
合成d512／i1024／e4／affine4／group32，连续、混合长度、交替、顺序反转、固定seed随机选择。
反转32 tokens，其余128，sin／random vector及authoritative选择，不属于文本／tokenizer／实际Router质量。
另一个process生成native SwitchGLU reference，worker在测量前释放临时source bank。
CPU速度每方式3个独立worker，每case1次warmup＋4次请求；内存CPU／Metal每方式2个worker，
每case cold及保留cache的2次请求，counter仅用于内存worker。
14 worker／120计时＋80内存请求最大误差2.2352e-7，预算2,360,838 bytes可容纳2个file，
不容纳3个payload，保持／预约峰值2,359,296，无Lease，close保持0。
每请求eval：连续128→16、混合128→28、交替128→128、反转32→32、随机128→120；
warm load两者仍0／48／256／0／153。CPU参考中位数(ms)依次57.226→50.880、89.857→87.154、
220.474→225.028、14.312→14.097、160.961→155.714。run变动5.6–173%，无host资格，不认证速度优势。
连续性少时收益有限，I/O未减少。混合长度warm active峰值CPU4.001→4.114 MiB、
Metal4.001–4.028→4.325 MiB；CPU cache1.027→1.465 MiB。
各case warm RSS sample约330.9–335.4 MiB，包含import／fixture／cache，不能判定RSS方式优势。
峰值是process全体而非独立scratch，1ms采样可能漏过短峰，HWM为生涯值，MLX每请求reset并同步。
速度／内存process分开，unified RSS及MLX值不可相加，不是专用VRAM。
[严格payload预算对照](evaluation/strata-diverse-prefill-tight-budget-analysis-2026-10-12.json)：
恰好2个payload因保守header判定退回逐次，连续eval256→256，数值一致，不混为batch收益。
🟠 [Next] 基于payload的group admission、实际prefill连续性、scratch／RSS及无竞争计时。
⭕️ [Pending] 实模型／GPU E2E、system GPU／OOM及速度／默认采用认证。


🟢 [Done] 第18阶段（2026-10-12）：明确候选`grouped=True, maximum_group_rows=8`，
仅合并连续且Expert顺序相同的token，上限1–64，默认1。逆序选择保持分组，entry／byte不足仍逐次fallback。
Lease保持至batch同步求值结束，Router／top-k／各token最终权重由原block处理。
activation scratch增加，不包含在weight预算内；telemetry计数为执行group，不能当作Router token数。
15个native相关tests通过dense／4／8-bit一致、顺序边界、行上限、容量fallback及异常释放。
[合成功能比较](evaluation/strata-contiguous-row-analysis-cpu-2026-10-12.json)：9个独立trial，
576请求／4,608 tokens误差0。输入重复8次相同token vector，特意有利于连续路由，非一般prefill代表。
每64候选请求Expert eval512→64、build span／acquire1024→128；host提取仍64，load／miss／eviction仍32。
20,000 bytes预算下保持／预约峰值18,432，无Lease／拒绝，close保持0。
profile on、无host资格，不认证速度或默认采用。参考中位数grouped606.917／batch232.459／baseline69.583 μs，
不作为生产性能。定时1-token比较仍维持默认1行。
🟠 [Next] 多样prefill路由、scratch／RSS及低负荷比较。
⭕️ [Pending] 实weight／GPU／E2E及速度／默认采用认证。


🟢 [Done] 第17阶段（2026-10-12）：READY／GO／DONE／ACK分开startup、准备后等待及请求。
startup含import／reference／export／preload／warmup。READY后再次等待低负荷，parent开始monitor后发送GO，
收到DONE即停止monitor，然后ACK允许cleanup。startup约1秒、请求约100ms采样，短尾段与前段合并，
不丢弃CPU delta。请求window含parity及合图通信，但单次latency不含；资格仅依据请求记录，
不被startup失败自动污染。缺少请求记录及READY等待失败均拒绝认证。
`--quiet-timeout`可设3–180秒（默认30），CPU／PID阈值不变，系统CPU含worker／observer。
34个相关unit tests及ruff通过，runtime默认设置未修改。
[部分常驻](evaluation/strata-phased-partial-analysis-cpu-2026-10-12.json)取得baseline1 trial／4,096请求误差0；
startup约2.50秒、CPU140–222%，请求约0.293秒、CPU161–218%，两段均有负荷未达标。
之后launch gate发现journal_bench等竞争负荷。
[全常驻](evaluation/strata-phased-resident-analysis-cpu-2026-10-12.json)启动0 worker；
[120秒重试](evaluation/strata-phased-partial-retry-analysis-cpu-2026-10-12.json)也为0，
116个sample系统CPU约196–959%、中位数234%，其他Python负荷仍持续。
保留raw试验及失败记录，不认证速度或晋升；分开startup后仍观察到请求期间竞争。
🟠 [Next] 在请求期间负荷达标的host时间段补齐所有方式。
⭕️ [Pending] 全方式无竞争资格及实weight／system GPU／OOM／E2E认证。


🟢 [Done] 第16阶段（2026-10-12）：重新执行CPU低负荷gate比较。
[全常驻](evaluation/strata-quiet-resident-analysis-cpu-2026-10-12.json)14 trial／57,344请求；
[部分常驻](evaluation/strata-quiet-partial-analysis-cpu-2026-10-12.json)16 trial／16,384请求，
共73,728次误差0。profile off、64次warmup、独立process及顺序rotation。
全常驻中位数：baseline47.667／逐次143.459／grouped121.125／gather49.209 μs。
逐次／gather变动0.70／1.56%，grouped13.31%，baseline189.33%；最后gate未通过，
计划16 trial仅完成14，部分运行期间host条件也未达标，不认证速度。
部分常驻20,000 bytes预算（全bank36,864）、每进程1,024次、每方式4个进程全部完成：
baseline45.781／逐次217.969／grouped LRU189.250／grouped cost199.135 μs，
变动1.74／1.90／0.33／2.44%，全部≤5%。host条件通过分别2/4、4/4、3/4、2/4。
仅逐次LRU每次均同时满足稳定性及host条件，整体无竞争比较资格仍为false。
baseline保留全weight，不满足相同预算。候选保持／预约峰值18,432 bytes，无拒绝／残留Lease，
close保持0。每1,024请求miss／eviction逐次640、grouped512，再次确认加载减少20%。
source hash及契约检查通过，未修改runtime默认设置。
🟠 [Next] 补齐所有方式的host达标比较，分开初始化及请求区间负荷。
⭕️ [Pending] 实weight加载／推理RSS及MLX峰值、system GPU、实际OOM及E2E认证。


🟢 [Done] 第15阶段（2026-10-11）：增加`--quiet-host`，约1秒采样，连续3次满足
系统CPU≤200%、可读取的其他process最大≤75%、PID无增减后才启动worker。运行期间也记录负荷，
解析要求host条件及run总时间变动≤5%。系统CPU含benchmark及无法单独读取的process；
不保证瞬间负荷或disk／GPU隔离。空／未完成campaign不认证。30个相关unit tests及ruff通过。
[两次低负荷尝试](evaluation/strata-quiet-resident-retry-analysis-cpu-2026-10-11.json)
各确认30秒仍未达标，Rust／Swift编译继续运行，未启动计时worker。
⭕️ [Pending] 当前host负荷下无竞争速度比较尚未完成。
[内存记录](evaluation/strata-load-memory-analysis-cpu-metal-2026-10-11.json)：Apple M4／32 GiB，
合成d1024／i2048／e4／affine4／group32，单Expert4.5 MiB。另一个process生成并export，
worker不保留原model bank。CPU／Metal default、capacity1／2、加载后／前退避各2个独立process，
最终16 trial／96 acquire，包含checksum／load／eval，不含推理。每次load前同步并reset MLX峰值，后同步；
RSS按要求1ms采样，并记录process生涯HWM。两种device下active峰值均由9降至4.5 MiB（capacity1），
13.5降至9 MiB（capacity2）。Metal sample RSS：capacity1由68.81–71.02降至64.25–64.34 MiB；
capacity2由73.36–75.12降至69.97–70.69 MiB。close后保持／active／Lease为0，cache仍保留，
测试clear后cache为0。RSS采样可能漏掉短暂峰值，HWM非独立load峰值。MLX统计是process allocator，
并非system全体GPU／专用VRAM；unified memory的RSS与MLX值不能相加。未修改runtime的cache策略。
🟠 [Next] 无竞争速度比较及实权重load／推理峰值。
⭕️ [Pending] 实际OOM、system全体GPU内存及长时间E2E认证。


🟢 [Done] 第14阶段（2026-10-11）：MLX加载前按file size减去8-byte prefix及header预约payload。
manager先退避未使用Expert，容量不足／pinned时在load前拒绝；实际tensor超出预约则释放并拒绝。
`peak_reserved_bytes`记录保持权重加incoming payload。无此可选接口的backend维持原加载后退避。
文件必须immutable，prefix读取计入file_stat。加载失败可能清空部分cache，但可重试。
[功能记录](evaluation/strata-preload-reservation-cpu-2026-10-11.json)：192次请求误差0；20,000 bytes预算的
LRU／cost grouped预约峰值18,432 bytes，10,000 bytes预算逐次fallback为9,216 bytes，
无残留Lease，close保持0。24 unit／14 native相关tests及ruff通过，包括load前拒绝、
预约超出后的释放与模拟MemoryError恢复。仅约束逻辑权重重叠，不包含loader临时分配、checksum buffer、
activation、allocator cache、RSS／GPU memory，未实施实际OS OOM试验。观察到Rust编译等竞争负荷，
本次profile-on结果仅验证功能，不认证速度。
🟠 [Next] 无竞争host重测及实权重加载期间RSS／GPU峰值。
⭕️ [Pending] 实际OOM／allocator及GPU／E2E认证。


🟢 [Done] 第13阶段（2026-10-11）：增加warmup、独立进程反复比较、明确byte预算、部分常驻suite及保持tensor峰值。
解析器将数值一致／保持约束与速度稳定性分开（run总时间变动≤5%）。19个unit及13个native相关测试、ruff通过。
[全常驻](evaluation/strata-stable-resident-analysis-cpu-2026-10-11.json)：24 trial／24,576次计时请求全部一致，
每个进程先warmup 32次，候选计时期间miss／eviction为0。
各run中位数的中位数：baseline／逐次／grouped／gather为70.969／190.948／178.490／56.427 μs。
run总时间变动139–352%，观察到同时运行的高负荷Swift编译，尚未取得稳定速度证据。
[部分常驻](evaluation/strata-partial-residency-analysis-cpu-2026-10-11.json)：全Expert 36,864 bytes，
保持预算20,000 bytes、entry上限4。24 trial／12,288次请求全部一致，候选保持峰值18,432 bytes，
无残留Lease／拒绝，close后保持量0。每512次请求miss／eviction由逐次LRU的320／320降至
两种grouped策略的256／256（加载次数减少20%），此fixture未显示cost策略优势。
中位数分别为baseline 110.344、逐次LRU 498.562、grouped LRU 377.542、grouped cost 402.938 μs；
run变动87–139%，不认证速度优势。baseline保持全部权重，仅作参考，不满足相同内存预算。
[单Expert预算fallback](evaluation/strata-single-expert-fallback-cpu-2026-10-11.json)：预算10,000 bytes、
top-k=2时grouped安全退回逐次路径，64次请求误差0，Expert eval 128次，峰值9,216 bytes，
计时期间miss／eviction 88／88，无残留Lease，close后保持0。此计测开启profile，不用于速度比较。
当前加载后才退避；保持峰值不包含临时分配／RSS，不保证实际OOM安全。
🟠 [Next] 在无其他高负荷竞争的环境重测。⭕️ [Pending] 实权重、GPU／RSS、加载峰值／OOM及chat E2E认证。


🟢 [Done] 全常驻device-index gather候选移除host indices取值及逐Expert Python loop／eval。
包含native的12项测试通过，dense/4/8-bit及40-token一致、source释放和预算拒绝已验证。
轮换顺序的12次CPU试验共768请求全部一致：baseline/逐次/grouped/gather中位数
50.917/162.792/138.125/55.958 us。gather接近baseline但仍较慢且波动27.9%，不认证或采用。
必须全Expert常驻，startup／allocator RSS、GPU及实际模型未验证。
🟠 [Next] 稳定同条件计测，另行验证内存受限常驻路径。

🟢 [Done] 可选top-k合并求值在完成前保持全部Lease，entry／byte不足时逐次fallback。
包含native的11项测试通过，dense/4/8-bit一致及失败回收验证。
18次预常驻CPU试验将eval调用16→8，但延迟变化不一致：LRU 160→152.5 us，
cost 183.291→185.125 us，波动超5%，不采用。全常驻试验不能比较退避策略。
🟠 [Next] host路由及Python／graph固定开销，之后考虑有界gather候选。

🟢 [Done] 可选有界host分阶段计测及27次独立合成试验，分开checksum I/O/hash、
MLX加载／求值、路由及Expert执行。预常驻使请求内miss/load/checksum调用全为0，
候选仍慢于baseline，逐Expert同步及Router host取值仍有明显开销。
run波动超过5%，不能认证精确贡献或策略优势。嵌套时间不可相加，Router eval包含
gate/top-k，MLX load/eval可能包含内部I/O，异时对照不能单独确定计测overhead。
🟠 [Next] 验证持有top-k Lease后合并求值的候选及有界fallback。

🟢 [Done] 轮换顺序、各3个独立进程的合成CPU比较共9次全部数值一致。
后续调用中位数baseline/LRU/cost为0.135/0.336/0.213 ms，cost仍慢于baseline，不采用。
不计export/startup、不清OS file cache，未认证实际模型/GPU/RSS/P3。
🟠 [Next] 分开测量I/O/checksum/load/router同步及实际权重tradeoff。

🟢 [Done] 标准Qwen3 MoE显式switch hook保留原gate、top-k正规化及集约，
返回未集约Expert结果且不保留原switch bank。只在空闲safe point安装及切换phase，
恢复baseline需重载模型。包含实际MLX的12项测试通过；dense/4/8-bit及正规化开关
共6种合成block比较误差低于1e-5，原switch weakref已释放。
逐次host同步参考路径不代表加速或Qwen4Exp认证。
🟠 [Next] 实际兼容权重、延迟／I/O测量及P3比较。

🟢 [Done] 标准SwitchGLU离线导出支持一致的dense／affine 4/8-bit格式，仅写入新目录。
拒绝不支持的activation、bias及混合格式，最后发布完成manifest。
失败保留部分文件但不发布manifest。包含实际MLX的11项测试通过，合成导出与原
SwitchGLU路由结果误差低于1e-5。离线caller仍持有source权重，未认证runtime替换或模型性能。
🟠 [Next] architecture专用连接及runtime权重所有权切换。

🟢 [Done] 有界manifest绑定model SHA256、量化配置及各Expert大小／校验值，
cache miss加载前拒绝重复项与内容变化。包含实际MLX的10项及相关15项测试通过。
接入时使用from_manifest，前提是可信且不变的offline文件；checksum不是签名，
并发修改及已常驻entry重验证不在本契约内。
🟠 [Next] 模型导出及architecture专用Router接入。

🟢 [Done] 加入显式affine 4/8-bit Expert支持，不展开dense权重，验证并计入packed
weight／scale／affine bias。实际CPU测试6项通过；4/8-bit × group32/64与同一量化权重
的展开参考误差低于1e-6。group128可配置但未在本次验证。
不代表原模型质量或速度认证。🟠 [Next] 用manifest绑定bits／group／model identity并接入实际路由。

🟢 [Done] 实现逐token多Expert路由，保留选择顺序及未归一化权重，加载前验证全部行。
最多4096行逐次执行。实际MLX CPU测试4项通过，包含数值一致、退避和失败回收。
这是参考路径，不代表吞吐认证，也未自动接入模型Router。
🟠 [Next] 明确量化Expert格式及数值比较。

🟢 [Done] 独立文件MLX浮点无bias SwiGLU适配器验证tensor并完成权重及结果求值，
退避时释放tensor引用。实际CPU数组验证了退避及重载后的数值一致；合成权重不代表
模型质量、GPU性能或OS内存回收认证。尚未安装模型hook，文件必须是可信offline产物。
⭕️ [Pending] 目标模型E2E需要实际权重及architecture／量化适配。
🟠 [Next] 多Expert、量化及实际Router接入验证。

🟢 [Done] 同步Expert执行连接保留Router顺序及权重，在backend完成前保持Lease，
失败时回收Lease，并记录有界的分阶段hit统计。相关22项测试通过。
consumer返回前必须完成GPU结果求值。仅验证测试backend，实际模型适配及加速尚未验证。

🟢 [Done] 在已有常驻管理器中加入可选cost_frequency策略，默认仍为LRU。
评分结合常驻访问次数、同步加载时间、内存大小和逻辑时间衰减；Lease保护、
容量上限及延迟resize保持共用。移除entry时丢弃统计。相关31项测试通过。
host加载时间不能证明GPU就绪或实际miss成本，尚未证明速度改善。
🟠 [Next] 接入实际MoE资源操作及分阶段测量，与LRU比较，通过已有P3判定后再采用。
🔴 [Later] 预取、兼容推测解码、KV压缩和融合。
⭕️ [Pending] 缺少可测试artifact或硬件的MTP／ANE路径。

## Explicit model integration example / 明示的なモデル接続例 / 显式模型接入示例

```python
from vllm_apple import StrataRuntime

# model: loaded standard mlx_lm.models.qwen3_moe.Model
# expert_root: previously exported immutable expert directory
with StrataRuntime.from_model(
    model, expert_root=expert_root, expected_model_sha256=model_sha256,
    maximum_file_bytes=maximum_expert_file_bytes,
    maximum_entries=resident_entries, maximum_bytes=resident_bytes,
    mode="grouped",
) as runtime:
    runtime.set_phase("prefill")
    logits = model(input_tokens, cache=cache)
    # Complete lazy non-expert work before lifecycle changes.
    import mlx.core as mx
    mx.eval(logits)
    runtime.set_phase("decode")
    logits = model(next_token, cache=cache)
    mx.eval(logits)
```

### Explicit local server / 明示設定server / 显式本地服务

```sh
python -m vllm_apple.mlx_server --model /path/to/qwen3-moe \
  --strata-expert-root /path/to/exported-experts \
  --strata-model-sha256 YOUR_EXPECTED_MODEL_SHA256 \
  --strata-resident-bytes 1073741824 --strata-resident-entries 8 \
  --strata-maximum-file-bytes 536870912 --strata-mode grouped
```

Budgets above are examples; choose values for your exported experts and available memory.
例の予算は対象Expertと利用可能メモリに合わせて変更してください。
示例预算须按导出的Expert及可用内存调整。
