# vLLM-Apple Runtime Roadmap

## 正常workloadのKV測定pointを固定

🟢 [Done] [warmup／長文／短文直後のmemory point](P1-NORMAL-MEMORY-POINTS.md)：drainとbenchmark起点PID一致、workload SHA接続、fault前のsnapshot。比較toolは古いfault後値を拒否。回帰で見つかったheader deadlineの期待切断をtestで扱い、expiry／slot回収基準は維持。
実機全3点を取得、正常停止・identity不変。長文LRU約211 MiB、短文約7.4 MiB、fault後約35.45 MiBでscope差を確認。
正常品質42/42だがSLO39/42、warmup SLO1/3で不合格。計測修正を速度・RSS・P1長時間資格へ広げない。
🟠 [Next] 全回帰の起動timeout／daemon／contention／generative process回収エラーを切り分ける。対象4 tests・Ruff成功だが、最終全回帰1484 tests（27 skip）は2 failures／4 errorsで未合格。固定pointで長文KV保持／eval／allocation待ちを診断。既定4 entry・256 MiB維持、長時間試験09:00 JST。
English: 🟢 [Done] Drained normal-workload memory points verified; SLO still failed. 🟠 [Next] Diagnose with fixed points; no performance promotion.
简体中文：🟢 [Done] 验证请求回收后的正常workload memory point；SLO仍失败。🟠 [Next] 固定point诊断，未晋升性能。

## KV反復の失敗を検出し採用を停止

🟢 [Done] [交互fresh-worker比較](P1-KV-ENTRY-REPEATS.md)：6 trial計画、同identity集計、非昇格、失敗で後続停止。今回2回目の3 entryがSLO39/42で失敗し、残り4回は未開始。
4 entryは品質／SLO42/42、warmup SLO2/3。3 entryは品質42/42、warmup SLO0/3。両方正常停止・identity一致・worker回収。既定4維持。対象7 tests、現venv全回帰1483 tests（27 skip）、offline Ruff成功。
最終KV sampleはfault試験後であり、以前の31.5%差を固定workloadの改善率として認定しない。速度・RSS・根因未認定。
🟠 [Next] 正常要求直後のKV測定点を固定し、eval／allocationを同条件で診断。長時間試験09:00 JST、定期更新は停止したまま。
English: 🟢 [Done] Alternating comparator stopped on 3-entry SLO failure; default four remains. 🟠 [Next] Fix memory sampling point before attribution.
简体中文：🟢 [Done] 交替比较在3-entry SLO失败后停止，保持默认4。🟠 [Next] 固定memory测量点再分析原因。

## KV entry候補を実測比較

🟢 [Done] [4／1／3 entryの比較](P1-KV-ENTRY-COMPARISON.md)：qualifierに明示entry数を追加。既定4、byte上限256 MiB維持。bounded runnerで3 fresh worker、長文・短文126/126品質／SLO、正常停止・identity一致。4 entry warmup SLO2/3は別記。
3 entryはLRU会計48.34→33.11 MiB、token reuse維持。ただし短文mean／maxとp95 bucket上限は悪化。1 entryは再利用率87.38→23.93%へ低下。どちらも既定へ昇格せず。全回帰1486 tests（11 skip）、Ruff成功。
🟠 [Next] 3／4 entryの順序を替えた独立反復でtail latency／RSS trendを比較。保持KV削減をRSS／速度改善と認定しない。
English: 🟢 [Done] Three real-model cache-entry trials and explicit comparison CLI. 🟠 [Next] Repeat three/four; keep default four.
简体中文：🟢 [Done] 3轮实际cache-entry试验及显式比较CLI。🟠 [Next] 重复3／4，保持默认4。

## 比較試験のtimeout回収を修正

🟢 [Done] [bounded trial cleanup](P1-TRIAL-CLEANUP.md)：専用process group、SIGINT猶予、残存group回収・runner reap。timeoutは別receiptへ保存し後続試験を停止、raw reportを保護。
実process対象5 tests、全回帰1486 tests（11 skip）、Ruff成功。GPU性能・電力・過去の64 MiB試験の資格に広げない。
🟠 [Next] このrunnerで短時間のKV／allocation候補を比較。長時間試験は09:00 JST、定期更新は停止したまま。
English: 🟢 [Done] Bounded cleanup and fail-closed timeout receipts. 🟠 [Next] Measured KV/allocation candidates; no performance promotion.
简体中文：🟢 [Done] 有界回收及timeout独立receipt。🟠 [Next] 实测KV／allocation候选；未晋升性能资格。

## 実KV保持量と候補の採否

🟢 [Done] [LRU保持量の実機確認](P1-KV-RETENTION.md)：P1 snapshotへ既存entry数／nbytes／上限を追加。tensor copy・同期なし。
256 MiB設定で最終4 entry／48.34375 MiB、長文・短文42/42品質／SLO、warmup品質3/3・SLO2/3、正常停止・identity不変。
64 MiB候補は180秒timeoutで未認定。raw checkpoint保持、runner／worker終了確認。候補コードを撤回し、既定256 MiB維持。全回帰1483 tests（11 skip）、Ruff成功。速度・RSS削減・timeout根因は未認定。
🟠 [Next] 活動中KVとLRU保持を分離し、allocation／eval待ちを同条件比較。長時間試験は09:00 JST。
English: 🟢 [Done] Retained KV accounting verified; reverted unvalidated 64 MiB candidate after timeout. 🟠 [Next] Profile active KV/allocation; default 256 MiB remains.
简体中文：🟢 [Done] 验证保持KV会计；64 MiB候选timeout后撤回。🟠 [Next] 分析活动KV／allocation，保持默认256 MiB。

## RSSとallocatorの差分を独立して比較

🟢 [Done] [memory window監査](P1-MEMORY-WINDOWS.md)：既存snapshotのRSS／allocator active／cache差分を独立して保存。欠測・PID変更を拒否し、追加HTTP／OS／GPU callなし。有界window collectorと既存reportのoffline監査を実装。全回帰1483 tests（11 skip）、Ruff成功。新fieldの実GPU検証は次の短時間試験で確認。
2試験の保持windowでRSSとallocatorの増減が一致しないことを確認。RSSからKV／allocatorの保持量を推定せず、差し引きで「その他memory」を作らない。根因・速度改善は未認定。
🟠 [Next] 同workloadでactive／cache／実KV保持量を揃えて比較し、効果が確認できる解放候補を一つ検証する。
English: 🟢 [Done] Independent RSS/allocator deltas and offline audit. 🟠 [Next] Compare actual retention; no causal or speed claim.
简体中文：🟢 [Done] 独立RSS／allocator差分及离线审核。🟠 [Next] 对比实际保持量；未认证根因或速度。

## 手動改善：2026-10-09

🟢 [Done] ユーザー指定により本スレッドの定期更新を終了。今回以降は手動依頼で実施し、自動再開しない。
🟢 [Done] [同workloadの活動差分](P1-ACTIVITY-WINDOWS.md)：既存前後snapshotを再利用し、PID変更／reset／欠測を拒否。最大64 window、追加HTTP／OS call／GPU同期なし。
実機30秒で10 window、102/102品質／SLO、初期45/45も合格。全回帰1482 tests（11 skip）、Ruff成功。正常停止・identity不変。page-in増分は全windowゼロ、短文RSS増加を観測。r11根因・速度・長時間資格は未認定。
🟠 [Next] 再allocation・保持領域とeval待ち／RSS増加の関係を計測。長時間試験は09:00 JST開始。
English: 🟢 [Done] Recurring updates stopped; manual work only. Bounded same-workload activity deltas verified: 102/102 plus 45 initial quality/SLO. 🟠 [Next] Profile allocation/retention; no speed or stability promotion.
简体中文：🟢 [Done] 定期更新已停止，仅手动推进。有界同workload差分通过：102/102及初始45件质量／SLO。🟠 [Next] 分析allocation／保持；未晋升速度或稳定性资格。

## r11最終監査：2026-10-08

🟢 [Done] [最終監査](evaluation/p1-r11-final-audit-2026-10-08.json)：30分の正常負荷3,009件は全件完了・品質合格、SLO合格2,992件。17件がTTFT基準超過し、うち4件はE2Eも超過。不合格で8時間は未開始。
active cancel 295/295、slow consumer 30/30、queued cancel 29/29、timeout 29/29。worker crash試験は0件で未検証。
awake・電源条件不変・runtime/model identity不変・正常停止は合格。単一worker epochで追加資源診断はallocator/thread/FD plateau=true、RSS plateau=false。
RSSの後半傾斜は約935.8 MB/hourで16 MiB/hour基準未達。全期間RSS減少や同workload診断だけで収束・漏れなしを認定しない。
RSS診断の未達と、今回30分判定を直接拒否したSLO失敗を区別する。遅延の根因は未確定。
receiptの専用labelだけ回収し、runner/worker停止を確認。自動再試行や証拠上書きなし。
🟠 [Next] queue/prefill/decode/同期とhost負荷を分離して遅延原因を調べる。長時間campaignは09:00 JST開始、定期処理は月・木09:00 JSTへ復元。
English: 🟢 [Done] R11 quality 3009/3009, SLO 2992/3009: 17 TTFT misses, including four E2E misses. Thirty-minute qualification failed; eight hours did not start. Fault counts, identity, awake conditions and clean shutdown were checked; RSS plateau remains unverified. Removed only the exact r11 job. 🟠 [Next] Diagnose latency causes without relaxing gates.
简体中文：🟢 [Done] r11质量3009/3009、SLO 2992/3009；17次TTFT超限，其中4次也超过E2E。30分钟认证失败，未进入8小时。核对fault次数、identity、awake及正常停止；RSS plateau未通过。仅回收r11专用job。🟠 [Next] 保持gate，调查延迟原因。

更新日：2026-10-09

## P1 RSS／scheduling診断の欠測を補う

🟢 [Done] [process activity診断](P1-PROCESS-ACTIVITY.md)：P1 resourcesへ累積fault／page-in／context switchとthread数を追加。Darwin公開ABI、欠測明示、標準経路は追加samplingなし。
実モデル45/45品質／SLO・identity不変・正常停止。OS取得1,200回中央値1.792 µs、subprocessゼロ。全回帰1480 tests（11 skip）、Ruff成功。
累積1点からr11根因・memory pressure・RSS安定性を認定しない。既存gateは維持。
🟠 [Next] 同PID／同workloadのcounter差分・eval遅延・RSS再増加を比較。長時間試験09:00 JST。
English: 🟢 [Done] Optional process activity counters verified; 45/45 quality/SLO and regression passed. 🟠 [Next] Correlate same-epoch deltas; no causal/stability promotion.
简体中文：🟢 [Done] 验证可选process活动counter，45/45质量／SLO及回归通过。🟠 [Next] 同epoch差分比较；未晋升因果／稳定性资格。

## P1 prefill内のeval／cache解放を分離

🟢 [Done] [既存operationの計測](P1-PREFILL-OPERATIONS.md)：隔離P1 profileでのみeval／clear_cacheを分離。元MLX module・GPU呼出数・同期は維持。
実モデル45/45品質／SLO、identity不変・正常停止。遅いprefill 4件738–831 msのほぼ全時間はeval内、clear_cache最大1.007 ms。今回のcache解放削減を見送る。
合成計測の追加wall約2.06 µs/call。全回帰1479 tests（11 skip）、Ruff成功。GPU時間・r11根因・速度・RSS改善は未認定。
🟠 [Next] eval待ち内のGPU実行／host scheduling／memory pressureとRSSを診断。長時間試験09:00 JST。
English: 🟢 [Done] Existing prefill eval/cache-release calls separated; 45/45 quality/SLO and regression passed. 🟠 [Next] Profile eval waits and RSS; no causal or speed certification.
简体中文：🟢 [Done] 分离现有prefill eval／cache释放；45/45质量／SLO及回归通过。🟠 [Next] 分析eval等待和RSS；未认证根因或速度。

## P1 prefill候補比較と計測負荷削減

🟢 [Done] [6回比較と空prefill改善](P1-PREFILL-COMPARISON-2026-10-08.md)：空呼出をそのまま転送し、不要なclock／CPU／lock／統計更新を削減。20,000呼出の合成比較中央値wall 19.432→2.024 ms、CPU 19.430→2.025 ms。推論速度・ワット値ではない。
実モデルprefill512／256各3回、長文・短文252/252品質／SLO、全worker正常停止・identity不変。256はallocator peak約81 MiB減だが速度は一貫して改善せず、既定512維持。対象8 tests、全回帰1478 tests（11 skip）、Ruff成功。P1長時間・P3性能未認定。
🟠 [Next] 非空prefill内の同期／cache解放とRSSを診断。長時間試験は09:00 JST開始。
English: 🟢 [Done] Empty-prefill diagnostic overhead reduced; six real-model trials passed 252/252 quality/SLO. Keep default 512; no speed/stability promotion. 🟠 [Next] Diagnose synchronization/cache release and RSS.
简体中文：🟢 [Done] 减少空prefill计时开销；6轮实际模型252/252质量／SLO通过。保留默认512，未晋升速度／稳定性资格。🟠 [Next] 调查同步／cache释放及RSS。

## P1 backend phase計測

🟢 [Done] [cache／prefill／decode計測](P1-BACKEND-PHASES.md)：4つのhost methodへbounded timingを追加。
実モデルではprefill最大1.825秒、cache最大4.249 ms、decode最大62.340 ms。
長文12/12・短文30/30品質／SLO、正常停止・identity不変合格。warmup SLO2/3。
全回帰1475 tests（11 skip）、Ruff成功。共有batch・入れ子・空prefillを含み、GPU時間やr11根因とは認定しない。
🟠 [Next] 非空prefill／同期／cache解放の分離、既存prefill候補と計測負荷の比較。長時間campaignは09:00 JST開始。
English: 🟢 [Done] Four bounded backend phase timings verified on real MLX. 🟠 [Next] Profile nonempty prefill/synchronization and compare candidates; speed/stability remain unqualified.
简体中文：🟢 [Done] 4个有界backend phase计时通过实际MLX验证。🟠 [Next] 非空prefill／同步分析及候选比较；速度及稳定性未认证。

## P1 tokenize分離とSLO判定の修正

🟢 [Done] [tokenize計測](P1-TOKENIZE-TIMING.md)：要求別に開始／終了／失敗を記録し、欠測を保持した区間集計を追加。
実モデルtokenize中央値0.453 ms、tokenize後の初回Response待ち中央値722.295 ms・最大18.57秒。
長文SLO8/12でもcollectorが合格を返す抜けを修正し、初期長文／短文の全件品質・SLO合格・失敗ゼロを必須化。
元reportは保持、新判定で長文失敗を拒否。対象19 tests、全回帰1473 tests（11 skip）、Ruff成功。
🟠 [Next] cache／prefill／decode／同期の分離と計測負荷比較。根因・速度・安定性は未認定、長時間試験は09:00 JST開始。
English: 🟢 [Done] Tokenization timing and strict initial-benchmark SLO gate. 🟠 [Next] Split remaining backend phases; no root-cause or speed certification.
简体中文：🟢 [Done] tokenize计时及严格的初始benchmark SLO判定。🟠 [Next] 分离其余backend phase；未认证根因或速度。

## P1要求別の計測

🟢 [Done] [要求別timing](P1-REQUEST-TIMING.md)：P1 profileでdequeue、context受領、初回Response準備／受領、SSE write、handler終了を最大64件の数値履歴へ記録。
本文／出力／request IDを保存せず、欠測は補完しない。実モデルsmoke長文12/12・短文30/30品質/SLO、warmup品質3/3・SLO2/3。
全stageを持つ48件の順序を確認、fault／回収・正常停止・identity不変成功。全回帰1468 tests（11 skip）、Ruff成功。
🟠 [Next] 短時間の同条件負荷で要求別時間から原因を分離。tokenize／prefill分離、collector identity接続と計測負荷の比較を続ける。長時間認定・速度改善は未達。
English: 🟢 [Done] Bounded per-request timing validated on real MLX. 🟠 [Next] Attribute latency and compare measurement overhead; warmup SLO/stability remain unqualified.
简体中文：🟢 [Done] 有界请求级计时通过实际MLX验证。🟠 [Next] 分离延迟及比较计测开销；warmup SLO及稳定性未认证。

## P1遅延の時刻照合

🟢 [Done] [r11遅延監査](P1-R11-LATENCY-AUDIT.md)：17件の失敗sampleを要求時間と遅いstepの時刻で照合。
15件はcache再利用がほぼ全入力の短文。15件で保持stepとの重複あり、約8.9秒の2件は重複なし。
bounded sample不在をidleにせず、時間重複を因果と判断しない。対象2 tests・Ruff成功。
🟠 [Next] request別queue／tokenize／初回出力／HTTP書込の計測で未説明時間を分離してから修正する。
English: 🟢 [Done] Offline timing audit covers 17 failures. 🟠 [Next] Per-request phase measurements; no causal attribution or speed claim.
简体中文：🟢 [Done] 离线时间审核覆盖17次失败。🟠 [Next] 请求级phase测量，尚不认证因果或速度。

## P1朝9時開始の再検証：r11

🟢 [Done] 2026-10-08朝9時台にcompact／prefill512の新規launchd jobを開始。
[receipt](evaluation/p1-stability-launch-m4-2026-10-08-r11.json)と
[state](evaluation/p1-stability-m4-2026-10-08-r11/state.json)を保存し、runner実command・checkpoint鮮度で稼働を確認。
🟠 [Next] 30分全条件合格時のみ8時間へ進み、終了後に品質・SLO・全worker epoch資源・awake・identity・shutdownを監査する。
running／checkpointは未認定。実行中はruntime編集・追加GPU負荷・CPU benchmark／回帰試験を避ける。
終了監査の定期確認は一時的に毎時とし、campaign終了後に元の月・木09:00 JSTへ戻す。
English: 🟢 [Done] Started fresh r11 in the 09:00 JST hour and verified runner command/checkpoint freshness. 🟠 [Next] Audit the final results; eight hours follow only after all 30-minute gates pass. Qualification remains withheld.
简体中文：🟢 [Done] 日本时间09:00时段启动新的r11并确认runner command及checkpoint新鲜度。🟠 [Next] 审核最终结果；30分钟全部gate通过后才进入8小时，目前未认证。

## 指示書の未着手項目：イベント履歴の全走査削減

🟢 [Done] [EventBus実測・改善](EVENT-HISTORY-OPTIMIZATION-2026-10-07.md)：1件ごとの履歴全走査を直接参照へ変更し、全再送をO(N²)からO(N)へ。
9回交互比較で標準256件の再送中央値0.569→0.138 ms（75.84%減）、10,000件599.766→5.347 ms（99.11%減）。
全規模p95／p99改善、最大遅延は全規模での改善を認定せず。登録中央値+0.042 µs、10,000件container size+2.6%の代償を記録。
順序・gap・heartbeat・保持上限を維持。全Python回帰1463 tests成功（11 skip）、Ruff成功。
🟠 [Next] P1は次回09:00 JST開始の長時間試験で再検証。履歴取得の改善を推論速度・電力・長時間安定性の認定に広げない。
🔴 [Later] supervisor polling、Swift main thread／SSEの実利用profile。未計測の変更は先に採用しない。
English: 🟢 [Done] Measured direct ring access; default replay median fell 75.84%. 🟠 [Next] P1 qualification at 09:00 JST. 🔴 [Later] Profile supervisor/UI paths; inference speed and energy remain unqualified.
简体中文：🟢 [Done] 实测直接ring访问，默认重放中位耗时减少75.84%。🟠 [Next] 日本时间09:00进行P1认证。🔴 [Later] supervisor／UI profile；推理速度及电能未认证。

## 指示書に基づく単一hotspot改善：RSS計測

🟢 [Done] [構造調査・profile・前後比較](OPTIMIZATION-REPORT-2026-10-07.md)：macOS RSS取得をsampleごとのps起動からOS APIへ変更。
4規模でRSS一致、1200 sampleのprocess起動1200→0、中央値約1.7–1.9 ms→0.001 ms。
品質・SLOの短時間実機確認45/45、正常停止・identity不変。全回帰1457 tests成功（11 skip）、Ruff成功。
GPU／モデルメモリは変更せず、推論速度・電力改善は未認定。
🟠 [Next] 朝9:00開始の次回P1長時間試験で新collectorを含むidentityを検証する。
English: 🟢 [Done] Measured and removed per-sample ps spawning on macOS. 🟠 [Next] P1 long-run verification at 09:00 JST; no inference-speed or energy claim.
简体中文：🟢 [Done] 测量并消除macOS每次RSS采样启动ps的开销。🟠 [Next] 日本时间09:00进行P1长期验证，不宣称推理速度或功率提升。

## 速度・メモリ・待機処理の効率化：2026-10-07

🟢 [Done] [効率化候補](P1-EFFICIENCY.md)：新要求受付のscheduler budget 500→50 ms、空idle queueの通知待ち、
allocator cache 256→64 MiBの明示選択を実装。既定baselineは維持。
各候補3回の実機比較で品質・SLO 648/648、故障系・正常停止成功。速度は範囲重複・変動が大きく未認定。
固定要求後の保持cache最大値255.32→63.77 MiB、idle read入口29→1回／3秒を観測。
active memoryはほぼ不変、RSSの減少は確認できず、ワット値は未測定。全回帰1454 tests成功（11 skip）。
🟠 [Next] r10 compact長時間試験はユーザーの朝9時指定に合わせ[制御停止](evaluation/p1-stability-m4-2026-10-07-r10/controlled-stop.json)。
次回は2026-10-08 09:00 JSTに新規runで開始する。一連の長時間試験は以降も日本時間9時開始。
r9はSLO3000/3009・資源gate未達で不合格、8時間未開始。最新試験稼働中はruntime編集・追加GPU負荷を避ける。
⭕️ [Pending] 管理者権限での電力samplerまたは外部電力計によるjoule/request・idle wattの検証。
English: 🟢 [Done] Explicit efficiency candidates and measured retained-cache/idle-read reductions. 🟠 [Next] R10 long-run qualification; speed remains unqualified. ⭕️ [Pending] Electrical-power measurement.
简体中文：🟢 [Done] 显式效率候选及保持cache／idle read减少的测量。🟠 [Next] r10长期认证；速度仍未认证。⭕️ [Pending] 实际功率测量。

## P1最優先修正：2026-10-07

🟢 [Done] prompt cache予算256 MiBを保存先LRUのbyte上限にも適用し、応答完了時の無制限insertを修正。
[実LRU再現](evaluation/p1-prompt-cache-budget-2026-10-07.json)と[実機smoke](evaluation/p1-cache-budget-smoke-m4-2026-10-07.json)を保存。
約92秒の品質・SLO 240/240、正常停止合格。短時間の資源gateはfalse、性能改善は未認定。
全Python回帰1452 tests成功（11 skip）、Ruff成功。
🟠 [Next] [r9](evaluation/p1-stability-m4-2026-10-07-r9/state.json)で同条件の30分→全条件合格時8時間を開始。
r8の遅延・aggregate資源未達の全原因は未確定。
認定gateを緩和せず、[原因監査](evaluation/p1-r8-cause-audit-2026-10-07.json)を参照する。
English: 🟢 [Done] Enforce the P1 LRU byte budget. 🟠 [Next] Long-run latency/resource requalification; no performance claim or relaxed gate.
简体中文：🟢 [Done] 修复P1 LRU byte上限。🟠 [Next] 长期延迟／资源复测；不宣称性能提升，不放宽gate。

## Local text preview 0.1.0 — 最小利用版

🟢 [Done] [ローカルtext preview](LOCAL-TEXT-PREVIEW.md)：M4／32 GiB・固定Gemma2 artifactの
事前確認、一つの起動コマンド、loopback text API、concurrency 1・資源上限、停止、wheelを実装。
新入口で未知依存／改変source／model／追加weight／未対象hardwareを拒否する。
[最終wheel検証](evaluation/local-text-installed-wheel-m4-2026-10-07-final.json)で三言語SSE、拒否条件、
資源上限、正常終了に合格。全Python回帰1451 tests成功（11 skip）、Swift SDK tests・Mac sample build成功。
[配布manifest](evaluation/local-text-preview-manifest-2026-10-07.json)と一時venvのアンインストール確認を保存。
この最小利用版の完了と、以下P0〜P4の本番・性能認定の完了は別のmilestoneである。
🟠 [Next] このM4で未達の長時間SLO、数値／性能、RAG一般品質は残す。時間を理由に⭕️ [Pending]へ移さない。
他hardware・署名資格情報・独立検証環境が必要な試験だけ⭕️ [Pending]にする。

English: The usable local text preview milestone is implemented; production/performance
qualification remains 🟠 [Next]. Only unavailable hardware/credentials/environments are ⭕️ [Pending].

简体中文：可用的本地text preview milestone已实现，生产／性能认证仍为🟠 [Next]。
仅缺少hardware／资格信息／独立环境的试验标为⭕️ [Pending]。

## 目標と優先順位

Apple Silicon Macで、品質を維持しながら最短の応答時間と高い持続スループットを実現し、長時間の推論でもメモリ不足・停止・状態混入を起こさないLLM実行環境を目指す。
「最も高速」は全モデル・全Macに対する無条件の宣言ではなく、公開した比較条件で、品質と安定性の基準を満たす実行経路の中から最速を選べることと定義する。
Intel Macは互換性の別枠とし、Apple Siliconの性能認定を適用しない。

開発順序は **比較基盤 → 標準LLM経路 → batching／KV再利用 → 実測hot path → 長時間認定と配布** とする。
安定性の回帰検証は全段階で行う。画像・音声・動画生成、独自形式、大規模分散の追加より、日常的なchat・coding・agent用途の改善を優先する。

2026-09-26の外部レビューを踏まえ、開発単位を「固定条件で比較 → 支配時間を特定 → 1か所改善 → E2E再測定」とする。P0の基準が得られた対象では、P2と並行してP3のhot path調査を開始できる。kernel追加や独自engineの構築自体を成果指標にせず、長context・複数request・Agent・限られたメモリでのgoodputと安定性を重視する。

既存の253項目を含む詳細履歴は、編集開始時の内容をそのまま[旧roadmap](ROADMAP-history-2026-09-25.md)へ保存した。未コミットの追記も保存対象に含む。
本書が今後の優先順位を定め、旧roadmapは証跡索引として参照する。旧書の完了表示は、一般用途の性能認定を意味しない。
[設計判断](Architecture-Decision-Apple-Execution.md)のcontrol／execution分離と計測優先方針を継続する。

## ステータスと完了条件

- 🟢 [Done] implemented in the current codebase — 現在のコードに実装がある。実機認定の範囲は別記する。
- 🟠 [Next] high-priority unfinished work — 次の開発サイクルで取り組む未完了作業。
- 🔴 [Later] planned, but not the closest next step — 依存作業の完了後に進める計画。
- ⭕️ [Pending] 現在の筐体・環境では検証できない作業。必要なhardware／artifact／資格情報と再開条件を添え、環境が整った時に着手候補へ戻す。

旧書の`⭕️ [Pending]`は履歴として保持する。本書では、優先順位待ちの`🔴 [Later]`と環境待ちの`⭕️ [Pending]`を区別する。現在のM4で実行可能な長時間試験や未実装項目は、時間がかかることだけを理由に`⭕️ [Pending]`へ移さない。
新規項目の完了にはコード／テスト、対象の実行経路、再現コマンド、認定範囲を必要とする。性能項目は実モデルの比較reportも必要とする。
以下の数値は**今後の受け入れ目標**であり、達成済みの測定値ではない。

### 全体監査：現在の筐体での完了と残作業

2026-10-07。実装済みの契約・runnerと実機認定を分ける。websiteは変更しない。

| 状態 | 範囲 | 確認結果／残作業 |
| --- | --- | --- |
| 🟢 [Done] | P0限定基準・経路監査 | M4／Gemma 2／c1／三言語算術、2 backend各3回。全モデル・全Macの認定ではない |
| 🟢 [Done] | P1資源上限・cancel／回復runner | bounded HTTP・入力・allocator・queue・cancelと短時間smoke。最新完了r9の30分は品質3009/3009、SLO3000/3009、RSS・awake・identity・shutdown合格だがSLOとcache資源gate未達 |
| 🟠 [Next] | P1長時間認定 | r9の9件のSLO超過とcache資源gate未達に対し、compactで朝9時から30分再試験→合格時8時間。このM4で可能なので⭕️ [Pending]にしない |
| 🟢 [Done] | P2実continuous batching／identity付きKV実験 | 実batch幅4と再prefill提出減少を確認。標準採用・数値／性能認定は未完了 |
| 🟠 [Next] | P2数値・p95／goodput gate | c4 SLOと全prompt基準とのlogit差を解消し再測定。このM4で試験可能 |
| 🟢 [Done] | P3選択・P4認定gateとrelease昇格接続 | 証拠不足・不合格はfallback／昇格拒否。合成fixtureのテストを実性能・24時間認定に代用しない |
| 🟠 [Next] | P3実計測／P4目的別profile・24時間認定 | 実collector接続、P1／P2前提の合格後に本筐体で実施。未知profileを自動採用しない |
| 🟢 [Done] | R0完全templateのtoken予算・固定三言語HTTP smoke | 実Gemma tokenizerで境界・丸ごと資料除外・基本prompt拒否。実HTTPは固定support codeと資料なしの6ケース成功・正常終了 |
| 🟠 [Next] | R0一般品質／L0固定LoRA | 資料不足・悪意ある資料・長文引用のsuiteとadapter互換性／memory／実MLXの検証。未実装を環境不足へ移さない |
| 🔴 [Later] | R1／L1／RL2・追加architecture | 検索・学習・複数adapter等は前提作業後。現在の筐体で可能な範囲は開発候補のまま |
| ⭕️ [Pending] | 別SoC／RAM・M5 NAX・分散・容量超過 | 対象実機・network・model artifactが必要。詳細は下記環境待ち表 |
| ⭕️ [Pending] | 実署名／notarization・独立clean-machine install | Developer ID／notary資格情報と独立検証環境が必要。既存workflowは実配布認定ではない |
| ⭕️ [Pending] | ANE互換draft | 互換Core ML draft artifactが必要。既存encoderの検証で代用しない |

English: Local failures and unfinished implementation remain 🟠 [Next]/🔴 [Later]. Only tests
requiring unavailable hardware, artifacts, credentials or independent environments are
⭕️ [Pending]. Completed tooling is not completed runtime qualification.

简体中文：本机试验失败和未实现项仍保留🟠 [Next]／🔴 [Later]。仅缺少目标硬件、artifact、
资格信息或独立环境的试验标为⭕️ [Pending]。工具已实现不等于实际运行认证完成。

## 現在地：再利用できる基盤と不足している証拠

| 状態 | 基盤・証拠 | 認定の限界と次の仕事 |
| --- | --- | --- |
| 🟢 [Done] | memory admission、thermal／pressure対応、予約付きscheduler、process隔離、profile／rollback基盤 | 制御機構の存在だけでは実LLMの速度向上を証明しない。主経路への適用をP0で監査する |
| 🟢 [Done] | [MLX server wrapper](../vllm_apple/mlx_server.py)：MLX-LM serverへの委譲、tokenize、allocator／cache計測 | wrapper独自のcontinuous batching実装ではない。backend versionごとの実効機能を測定する |
| 🟢 [Done] | [semantic cache](../vllm_apple/semantic_cache.py)、[state coordinator](../vllm_apple/semantic_state.py)、[MLX state adapter](../vllm_apple/mlx_semantic_state.py) | 契約・adapterと、標準HTTP経路で実KVが再利用されることを分けて検証する |
| 🟢 [Done] | [Gemma 2 2B 4-bit・30分report](evaluation/homebrew-029-text-30min-2026-09-19.json)：6,775/6,775成功 | Homebrew 0.29.0、context 1024、concurrency 1限定。KV容量再評価はunavailable。42.841 decode tok/s／平均TTFT 91.105 msは別の3-sample probe値 |
| 🟢 [Done] | [Qwen3-VL persistent worker・30分report](evaluation/qwen3-vl-coreml-persistent-30min-soak-2026-09-20.json) | 固定shape／固定taskの証拠。一般VQA、標準text経路、ANE単独実行の認定へ広げない |
| 🟢 [Done] | [MLX cache計測改善](vllm-mlx-review.md)、[キャンセルqueue回収](llama-cpp-review.md) | 現在の作業ツリーに実装あり。実MLX／継続キャンセル負荷での性能検証が残る |

現時点では、主要backend横断の同条件ランキング、代表モデル群の長時間SLO、全Mac世代での優位性を証明する比較資料は揃っていない。まずこの不足を埋める。

## モデル対応範囲を広げる計画

必要なarchitecture、共通operator、state契約、現行実装との差分は[LLMアーキテクチャ対応計画](LLM-ARCHITECTURE-SUPPORT.md)にまとめる。

- 🟢 [Done] A0初期診断：`inspect-architecture`とJSON schema、5系列の構造recipe・synthetic fixtureを追加。未知／未検証を明示し、実行認定は付与しない。詳細は上記対応計画を参照。
- 🟢 [Done] A0 recommendation統合：`inspect-model`をschema v2へ更新し、宣言一致と実機認定を分離。未知backendへの暗黙の能力付与も廃止。Gemma2／M4／MLX-LM 0.32.0の三言語smokeは3/3合格（短い算術task限定、標準backendへ未昇格）。
- 🟢 [Done] A0証跡gate：30分text qualificationのidentity bindingと、`inspect-model`／managed `serve`の任意検証を追加。7日期限・model/backend/runtime/hardware変更・設定上限超過を拒否する。
- 🟢 [Done] Homebrewが意図的に削除するRECORDへ対応。brew管理・分離venvを確認し、環境全体のbounded inventoryでbackend identityを検証する。
- 🟢 [Done] Homebrew MLX-LM 0.32.0／Gemma 2 2B／M4でidentity付き30分text試験に合格。6,491/6,491件成功、RSS peak増加15.9 MiB、三言語・stream一致・正常終了と`inspect-model`による証跡再検証を確認。[実測report](evaluation/architecture-gemma2-homebrew-bound-2026-09-26.json)。
- 🟢 [Done] 有効なidentity付き証跡を指定した通常`serve`で、MLX-LMのversion matrix範囲外だけを限定許可する。証跡なし・期限切れ・identity変更・他の互換性エラーは引き続き拒否する。
- 🟢 [Done] 変更後runtimeの[30分再認定](evaluation/architecture-gemma2-homebrew-serving-bound-2026-09-26.json)で5,995/5,995件成功。新しい証跡で`--skip-backend-check`なしの[通常serve実HTTP検証](evaluation/architecture-gemma2-homebrew-managed-serve-2026-09-26.json)も合格（三言語・greedy反復・stream一致・正常終了）。
- 🟠 [Next] 追加のDense／window／MoEモデルと、長文・並列負荷・cancel／recoveryを認定する。今回の通常serve確認は短いHTTP smokeであり、frontend全体の30分soakや性能優位の証明ではない。
- 🟠 [Next] P0のcapability matrixへ、モデル名だけでなくlayer構成・必須operator・weight形式・state layout・backend buildを登録し、unknownを対応済みと扱わない。
- 🟢 [Done] architecture registry v2で、model directory直下を最大4,096 entryに制限して走査し、weight形式、file数、総byte、filename／size由来のmetadata manifest digest、tokenizer関連file名を記録する。配置済みGemma 2は`safetensors` 1 file／1,470,988,882 bytesとして実機確認した。これは内容hash、完全性、load可否、backend互換性の証拠ではなく、`artifact_status`と全qualificationは`unverified`を維持する。
- 🟢 [Done] `inspect-architecture`へ任意のbackend build inventoryを接続した。環境全体のSHA-256、version、backendが明示したoperator、architecture側の不足operator、probe issueを分離し、静的宣言が揃っても実行qualificationは昇格させない。MLX-LM probeはMetalを初期化せずpackage metadataを読むためheadlessでもversionを取得できる。現行MLX-LM 0.32.0／build `fbe49ebc…e643`はoperator宣言なし・verified version matrix外なので`declared_incomplete`と実測した。既存のidentity付きGemma限定証跡とは別のfail-closed診断である。
- 🟢 [Done] tokenizer identityを同じartifact inventoryへ追加した。既知のtokenizer関連fileだけを合計256 MiBまでrace検出付きで読み、fileごとの内容SHA-256からmanifest digestを作る。同名・同sizeの差し替えも検出する。配置済みGemma 2は4 file／21,813,831 bytes／manifest `ae37b56a…8217`として確認した。tokenizer動作やchat template品質の認定には広げない。
- 🟠 [Next] Dense MHA／MQA／GQA、local/global混在、標準MoEを代表モデルで認定する。
- 🔴 [Later] MLA、KV共有、Gated DeltaNet／KDA、SSM、短いconvを個別state契約で広げ、その後に高度な疎・圧縮Attentionと再帰実行へ進む。

## 実行アーキテクチャ

```text
Swift SDK / CLI / OpenAI-compatible API
                  ↓
Control plane: admission / queue / lifecycle / telemetry / routing
                  ↓  bounded commands, request identity, cancellation
Backend-owned process: model / KV / batch scheduler / generation
                  ↓
MLX / vLLM-Metal / qualified optional backend → Metal GPU
                  ↘ qualified fixed-shape encoder → Core ML
```

- GPUをtext prefill／decodeの基準にする。CPUはtokenization・I/O・制御を担い、CPU／ANEへの演算移動は転送・同期・競合込みで改善した場合に限定する。
- modelとKVの所有者はbackend processに一本化する。control planeのpriority queueとbackendのtoken schedulerの責務を明記し、二重queueによる待ち時間を測る。
- 大きなtensorをHTTP／JSON／Swift境界で往復させない。Unified Memoryでもcopy・materialize・同期・page faultのコストは計測する。
- MLX-LM directとvLLM-Metalを優先比較し、llama.cpp Metal／GGUFとvllm-mlxを比較対象にする。比較への追加と製品backendとしての採用は別判断とする。
- 自動選択はmodel revision・precision・context・concurrency・SoC・メモリ・OS・backend buildに束縛する。実行中にbackendやKV形式を切り替えず、未認定条件は既知の安全経路へ戻す。

## P0 — 比較可能な基準と実行経路の監査 🟢 [Done]

- 🟢 [Done] [P0監査と再現手順](P0-AUDIT.md)を公開した。M4／32 GiB、同一Gemma 2 2B 4-bit artifactでMLX-LM／vLLM-Metalを各独立3 run、各回102件測定。各backendで306/306品質・SLO合格、英語・日本語・简体中文各102件合格。warmupは各回3件を別集計し、起動順を交替した。
- 🟢 [Done] [集約監査](evaluation/p0-2026-10-04/baseline-audit.json)へworkload／artifact／hardware／backend内build同一性、sample数、品質、SLO、重複run拒否とgoodputのばらつきを束縛した。p99は参考値。MLX中央値18.3411 tokens/s・相対幅1.76%、Metal中央値25.7337・相対幅6.19%。Metalは5%の安定性基準を超え、性能優位は未認定。
- 🟢 [Done] 配置model／tokenizer／template／quantization config、Python backend source、依存versionと再現コマンドを保存した。upstream revision、全native binary hash、生成KV dtypeは未取得と明記し、4-bit weightsからKV精度を推定しない。
- 🟢 [Done] backend起動→生成→SSEとdaemon／wrapper経路を監査し、streaming、usage、cancel、prefix reuse、batching、chunked prefill、KV精度、structured outputの実機合格／adapterのみ／未検証を記録した。未検証flagは能力認定へ昇格しない。
- 🟢 [Done] phase／qualification／soakの[共通証拠索引](evaluation/p0-2026-10-04/evidence-index.json)を追加した。旧direct／proxy比較と限定soakを元hash・scopeで結び、現runtimeへ資格を転用しない。client TTFT／TPOT／stream tailと内部queue／tokenize／prefill／decode／serialization／model loadを区別し、未取得内部計測をnullとして公開する。
- 🟢 [Done] 実MLX KVCache metadataで確保論理容量とactive stateの差、同一object alias、共有viewの二重計上限界を確認した。単独metadata取得負荷のみ測定し、推論改善は未認定。autotunerへ同じpolicy・hardwareと異なるprofile IDを要求する確認契約を追加した。独立実測とE2E改善、自動適用は未認定。

完了は上記限定scopeの比較基準と監査を指す。全比較matrixの合格、内部phase計測の完成、安定した速度優位、持続運用の認定を意味しない。実装の履歴と当時の未完了記述は[保存したP0履歴](ROADMAP-P0-history-2026-10-03.md)を参照。実運用、KV機能、最適化の残課題はP1–P3へ明示する。

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

**完了条件：** 手元のMacで少なくとも2 backend・同一品質基準のモデルについて、独立した3 runを再現できる。各主要ケースは合計100以上の完了requestを目安とし、sample数とばらつきを公開する。p99は1,000未満なら参考値扱い。比較不能・未対応条件も残す。上記M4／Gemma 2／c1／三言語算術scopeで達成。

## P1 — 標準text経路と持続的な安定性 🟠 [Next]

- 🟢 [Done] [r7 host-load監査](evaluation/p1-r7-host-load-audit-2026-10-06.json)は10 CPUに対しload34〜43の境界観測を確認、load約7の失敗もあり根因未特定。🟢 [Done] scheduler stepのcalling-thread CPU時間をwall時間から分離してbounded記録、SLO／認定閾値は維持。差値をGPU時間とは扱わず、計測overhead未測定。🟢 [Done] 実機smoke48/48品質・SLO・正常終了、614 stepでCPU／wall分離記録、全回帰1449 tests成功（11 skip）。🟠 [Next] [r8](evaluation/p1-stability-m4-2026-10-06-r8/state.json)をprefill512で30分→全条件合格時8時間として実行中。失敗のCPU／wall／環境を突き合わせ修正候補を選ぶ。

- 🟢 [Done] [r7最終監査](evaluation/p1-r7-final-audit-2026-10-06.json)：品質3162/3162、SLO3153/3162、awake・identity・正常終了合格、RSS plateau合格だがallocator資源gate未達、8時間未開始。prefill256の優位は未認定。🟢 [Done] workload hash付き資源sampleと同負荷別診断を追加、既存認定閾値は維持。実機短時間smoke87/87品質・SLO・正常終了、label欠落0、全回帰1448 tests成功（11 skip）。🟠 [Next] 新実測で状態差と増加、短文／長文TTFT超過の原因を調査。旧sampleへlabelを推測付与しない。

- 🟢 [Done] 長時間runnerでprefill128／256／512を明示指定し両段階へ固定伝達、既定512・閾値は維持。全回帰1447 tests成功（11 skip）。[256実機smoke](evaluation/p1-prefill256-m4-2026-10-05.json)は204/204品質・SLO・awake・identity・正常終了合格、短期RSS plateau未達。性能優位は未測定。🟠 [Next] [r7](evaluation/p1-stability-m4-2026-10-05-r7/state.json)で30分→全条件合格時8時間を測定中、認定保留、競合編集／GPU負荷を避ける。

- 🟢 [Done] RSS plateau認定にsample検証を追加し、時刻重複／逆行、NaN／Infinity、欠測、負RSSを拒否。分母ゼロを傾きゼロとして誤認定しない。閾値は維持。[既存r5／r6再監査](evaluation/p1-rss-sample-validation-2026-10-05.json)の数値と判定は不変。🟠 [Next] r6の長文TTFT及びRSS未収束の解消。

- 🟢 [Done] [r4診断監査](evaluation/p1-r4-slo-audit-2026-10-05.json)：品質3327/3327、SLO3319/3327。短文TTFT超過8件を全数記録、資源・identity・正常終了は合格、8時間未開始。queue累積最大1.629秒ではTTFT 5〜6.85秒を説明しきれず、原因は未特定。🟠 [Next] source固定のscheduler step host時間とwindow前後の環境・資源snapshotを追加して再測定。GPU kernel単体時間や性能改善として扱わない。

- 🟢 [Done] P1失敗診断：benchmarkに最初の64失敗requestのTTFT／E2E・言語・token数・未達理由、soakに最初の64失敗windowと総失敗window数を保存する。本文を保存せず、後続成功で失敗を消さない。旧r3の途中window欠落を補うための再測定基盤であり、既存14件の原因解明・SLO改善は未達。
- 🟢 [Done] 診断追加後の[90秒実機混合負荷](evaluation/p1-failure-diagnostics-m4-2026-10-04.json)は品質・SLO204/204、identity不変・正常shutdown。全回帰1432 tests成功（11 skip）。r4の30分再測定は上記TTFT失敗で不合格、8時間未開始。
- 🟢 [Done] [step診断smoke](evaluation/p1-step-diagnostics-m4-2026-10-05.json)：90秒240/240品質・SLO、scheduler step3187回、前後環境snapshotと正常終了を確認。短時間でTTFT超過は未再現。🟠 [Next] 同診断で長時間再測定し、原因を確認してから修正する。
- 🟢 [Done] [r5最終監査](evaluation/p1-r5-final-awake-audit-2026-10-05.json)：品質3669/3669、SLO3667/3669、資源・identity・正常終了は合格。11回のsuspend gapとBattery→AC変化を確認し、固定awake認定は拒否、8時間未開始。🟢 [Done] P1 runnerへsuspend／power変化／unknownを拒否するawake gateを接続し、全回帰1436 tests成功（11 skip）。🟢 [Done] awake gate実機90秒smokeは品質・SLO 204/204、40観測のAC／automatic不変、suspend gap 0、identity・shutdown合格。🟠 [Next] [r6](evaluation/p1-stability-m4-2026-10-05-r6/state.json)は終了：品質3276/3276、SLO3275/3276、awake条件合格だが長文TTFT 11.97秒とRSS plateau未達で不合格。8時間未開始。r4短文失敗とr5のsleep中長文失敗を同一原因とは扱わない。

- 🟢 [Done] [P1限定candidateと試験手順](P1-STABILITY.md)：HTTP全体の接続上限16、header絶対期限5秒、body上限とdeadlineをwrapper／Gemma互換経路へ適用。model固定・prompt＋output 4,096 tokens・output 512・GPU allocator 8 GiB／cache 256 MiBのopt-in profileを追加した。全worker epochのRSS／allocator／thread／FD／registryを検証し、queue待機p95の計測scopeを公開する。任意model／shapeやdaemon全体の認定ではない。
- 🟢 [Done] 固定sourceの[M4短時間smoke](evaluation/p1-profile-90sec-m4-2026-10-04-r3.json)で正常189/189品質・SLO合格、active cancel 18/18、queued cancel／timeout各6/6、worker crash回復2/2、half-close、profile拒否と正常shutdownを確認。全回帰1,407 tests合格（11 skip）。短時間の資源plateauと8時間安定性は未認定。
- 🟠 [Next] [最新の連続試験状態](evaluation/p1-stability-m4-2026-10-04-r3/state.json)：30分試験は品質3312/3312、SLO3298/3312で不合格。8時間は未開始、backend／runnerは終了しlaunchd jobも回収済み。14件のSLO失敗を調査・修正後に30分→合格時8時間へ進む。以前の408件後中断reportも未認定として保存した。PID command・checkpoint鮮度・全epoch資源・同一identityを確認し、実sleep／wakeとdaemon標準経路を含め、完了までP1を🟢 [Done]にしない。

- 🟠 [Next] P0監査の残課題：request headerを含むHTTP全体のthread／資源上限、長時間queue cancelとp95待ち時間、half-close、telemetry有無の独立比較を検証する。現在のM4で試験可能であり⭕️ [Pending]にはしない。旧30分soakの資格は現buildへ転用しない。

依存：P0で選んだ基準経路。成果物は認定text profileと、運用上の失敗から復帰できる標準server。

- 🟠 [Next] 起動時に依存ABI、実GPU選択、model／tokenizer identity、実KV capacityを確認する。取得不能は安全な上限と理由を提示し、未検証versionを自動昇格させない。
- 🟠 [Next] model-owner thread／processを固定し、load／generate／cancel／closeの所有権を一貫させる。既存main-thread process経路を再利用し、モデルをrequestごとにロードしない。
- 🟠 [Next] admissionにprompt＋最大出力のKV増分、prefill scratch、batch増分、allocator cache、OS reserveを反映する。最大contextと最大concurrencyを同時に保証しない。
- 🟠 [Next] client切断、active／queued cancel、timeout、遅いSSE consumer、worker crash、sleep／wake、shutdownを実backendで試験する。queue・IPC・出力bufferをboundedに保つ。
- 🟢 [Done] M4／Gemma 2 2B／MLX-LM 0.32.0のbatch mask不一致へ、version＋source hash限定のプロセス内互換修正を追加。[修正後HTTP試験](evaluation/text-benchmark-m4-gemma2-mask-fix-2026-09-26.json)は並列度1／2とも30/30正答。GPU上の8条件で未修正の逐次attentionと数値一致。Homebrew packageを変更せず、[専用起動経路](GEMMA2-BATCH-MASK-FIX.md)で明示適用する。
- 🟢 [Done] 修正したGemma 2経路の短時間M4回帰：[実測report](evaluation/gemma2-batch-mask-cancel-slow-m4-2026-09-27.json)で約2K prompt tokensの共通prefix編集12/12、並列度2の継続負荷100/100が品質・SLO合格。source hash限定の実験用cancel APIはactive contextへstopとrequest専用queue終了を通知し、HTTP 202後0.759 msでstream完了。1 KiB receive buffer・1秒読み取り停止の遅いconsumer、client切断後の正常応答、SIGINT正常終了も確認。
- 🟢 [Done] 修正したGemma 2経路に30分の短長混合・cancel反復runnerを追加し、M4で実行。[実測report](evaluation/gemma2-batch-mask-30min-m4-2026-09-27.json)は通常応答3162/3162正答、cancel 310/310、slow consumer 32/32、正常終了、RSS増加なしを確認した。一方で長prefixの102件がTTFT 10秒SLOを超え、SLO内3060/3162のため総合判定は不合格。安定性の証拠には使うが、30分SLO認定には使わない。
- 🟢 [Done] 長prefixの設定matrixと持続比較を実施。prefill step 512だけでは同時長prompt 2件のSLO未達を解消できなかったため、短文c2を維持し約2K-token長promptだけc1へ制限するprofileを追加。[30分再認定](evaluation/gemma2-batch-mask-long-c1-prefill512-30min-m4-2026-09-27.json)は3924/3924件が品質・SLO合格、cancel 384/384、slow consumer 39/39、危険thermal 0/385、回復・正常終了に合格。元profileの3162件・SLO超過102件から改善した。M4 Air／32 GiB、Gemma 2 2B 4-bit、短文c2・約2K長文c1限定の認定とする。
- 🟢 [Done] source-hash限定compat経路へqueued cancelと1〜600,000 msのrequest timeoutを追加。[短時間実HTTP report](evaluation/gemma2-queued-cancel-timeout-m4-2026-09-27.json)でqueued DELETE 202、実行前破棄、active blocker停止、100 ms timeout、後続回復、正常終了に合格。review済みcompat moduleを既存`BackendProcess`で管理する限定起動契約も追加し、[SIGKILL 3回のrestart report](evaluation/gemma2-worker-restart-qualified-m4-2026-09-27.json)で別PID・readiness・算術品質を毎回回復した。daemon自動watchdogやinflight replayの認定には広げない。
- 🟢 [Done] bounded poll・指数backoff・最大restart回数・lifecycle lock・状態snapshotを持つ`BackendSupervisor`を追加。[自動watchdog実測](evaluation/gemma2-worker-watchdog-qualified-m4-2026-09-27.json)は手動restartなしでSIGKILL 3/3回を検出し、backoff込み2.953〜3.710秒で別PID・readiness・算術品質を回復。restart failure 0、exhaustionなし、正常終了に合格した。standalone supervisorの認定であり、daemon統合ではない。
- 🟢 [Done] daemonのbackend起動・停止とnative v2 tuning／restore／quarantine rollbackを同じsupervisor transactionへ一本化。planned restart後にprocessが稼働していればwatchdogは二重restartしない。restart中の新規requestはupstream接続前に`503 backend_unavailable`を返し、接続後のworker消失も同じretryable errorへ変換する。proxy内でinflight requestを自動再送しないことを回帰テストで固定した。
- 🟢 [Done] watchdogのrestart成功・失敗・上限到達をdaemonへ通知し、失敗ごとのprivate crash diagnosticを保存する。上限到達時は`backend_exited`の構造化runtime failureを公開し、診断callbackの例外はwatchdogを停止させない。mock workerでrestart失敗2/2回とexhaustionを再現し、service状態と診断を検証した。[M4実機fault report](evaluation/gemma2-worker-watchdog-fault-qualified-m4-2026-09-28.json)ではSIGKILL 3/3回の待機中にclientが`503 backend_unavailable`を受け、3.096〜3.738秒で別PID・算術品質を回復した。
- 🟢 [Done] 8時間runnerへ明示的な28,800秒gate、各cycleの原子的checkpoint、反復queued／active cancel・timeout・slow consumer、最大512点のRSS時系列と後半線形傾向、sleep／wake観測gateを追加した。8時間RSS合格条件は後半16 MiB/時以下かつ増加64 MiB以内。[45秒M4 smoke](evaluation/gemma2-8hour-runner-smoke-m4-2026-09-28.json)は通常応答75/75、各fault 7/7、RSS増加3,637,248 bytes、後半推定9,417,299.695 bytes/時、正常終了に合格した。短時間値は8時間安定性の認定には使わない。
- 🟢 [Done] 同じsoak窓へ`BackendSupervisor`による定期間隔のworker SIGKILLを統合した。各注入でrestart中の`503 backend_unavailable`、別PID、readiness、算術品質を確認し、8時間gateでは注入0件または1件でも失敗なら不合格にする。RSS傾向は再起動を跨いで回帰せず、最新PID epochだけを評価する。[51秒M4 smoke](evaluation/gemma2-soak-worker-crash-smoke-m4-2026-09-28.json)は通常応答99/99、worker crash 2/2、2.983〜3.204秒の回復、正常終了に合格した。最新PIDのsampleは2点だけのためRSS plateauは未判定であり、短時間の安定性認定には使わない。
- 🟠 [Next] 認定profileで実際の8時間runを行う。蓋を閉じる等の実sleep／wakeを含むrunでは`--require-sleep-wake`を指定し、観測ゼロを合格にしない。各worker epochで十分なRSS sampleを集め、allocator安定域へ収束するか最終判定する。
- 🟠 [Next] cancelはbackendの安全な実行境界で処理する。解放完了まで予約を維持し、stream公開済みrequestの黙った再実行やtoken重複を禁止する。OOM retryは副作用と状態復元が証明できる経路だけに限定する。
- 🟢 [Done] watchdogとrestart backoff、clientへの503通知、上限到達時の構造化failureとprivate diagnosticを整備した。過負荷拒否とworker内部障害は別code／eventとして集計できる。
- 🟢 [Done] [Splash 1.1.0の設計レビュー](splash-review-2026-09-28.md)から、request単体上限とは別のaggregate入力byte admissionを独立実装した。default 16 MiB、設定範囲4 MiB〜1 GiBで、JSON read前からstream終了まで予約し全終了経路で解放する。超過は`503 request_bytes_exhausted`、単体超過は従来の413として区別し、`/v1/runtime.http`へcurrent／peak／limit／rejectionを公開する。Splash engine／DFlash2／kernelは対象modelと認定条件が異なるため導入・性能認定していない。

**完了条件：** 30分smokeを入口に、認定profileで8時間の混合負荷と障害注入を完走する。正常負荷では予期しないcrash／OOM／hang／状態混入0件、成功率99.9%以上を目標とする。意図したcancel／admission拒否は別に件数を出し、分母操作で成功率を上げない。
warm-up・cache充填後の同一負荷窓とidle回復時を比較し、RSS／allocator／queue／handleに継続的な増加がないことを確認する。注入後の予約・worker・一時file回収と後続正常応答も必須。

## P2 — Continuous batchingと実KV再利用 🟠 [Next]

- 🟢 [Done] [P2実験用backend](P2-BATCHING-KV.md)へnative continuous batching／chunked prefillとidentity付き実KV LRUを接続した。実schedulerでdecode幅4・複数sequence step 39回、prefix再入場でprefill提出545 tokens減少を確認。品質36/36合格だがc4 SLOは初回28/30・最終source22/30で未達。P1／標準経路は変更せずopt-inに留める。全CPU回帰1,411 tests合格（11 skip）。
- 🟢 [Done] 実Gemma 2／float16 KVで分割fresh参考とのlogit差0、branchコピー後のprefix保持、trim後missを確認。全prompt一括計算とは最大差0.0390625で既定の数値gate未達、argmax一致だけで認定しない。性能・数値qualificationはfalse、P2完了条件は未達として残す。

- 🟠 [Next] P0監査で未認定のprefix hit／eviction、KV数値正確性・dtype、chunked prefillと実batchingを検証する。adapter直列化とcached usageだけで機能の正確性や性能を認定しない。

依存：P0の計測、P1の所有権・cancel契約。成果物はinteractive／throughput profile。既存backend機能を先に利用し、不足だけを実装する。

- 🟠 [Next] backendのtoken単位continuous batchingを実HTTP経路に接続・確認する。単なる複数HTTP requestやcontrol plane queueをbatching達成と数えない。
- 🟠 [Next] chunked prefillで長い入力によるdecode停止を抑える。prefill token budget、active decode数、最大待ち時間を調整し、priority agingでbackground starvationを防ぐ。
- 🟠 [Next] 実際のKV／recurrent stateに結び付いたprefix reuseを有効化する。model／adapter／tokenizer／template／position／cache saltをidentityへ含め、exact token prefix一致のみを再利用する。
- 🟠 [Next] turn／tool境界anchor、copy-on-write、eviction／releaseをbackend所有下で検証する。共有prefixの書き換え、cancel、異なるsession間の状態混入を防ぐ。SWA／hybridはarchitecture固有の復元契約を要求する。
- 🟠 [Next] hit率に加えて再計算を省けたprompt tokens、TTFT、cache byte、eviction頻度を測る。metadata上のhitだけでは昇格しない。
- 🟠 [Next] chat／coding／agent／batchを比較workloadとして分ける。Agentは固定tool定義・system prompt、短いdecode、模擬tool待機、再入場を含め、再prefill token数、再入場TTFT、他requestのp95とstarvationを検証する。最初はbackend既存schedulerの設定profileで比較し、独自token schedulerの追加は不足の実測後に判断する。
- 🔴 [Later] prefix trie、SSD cache、prompt類似度slot選択。RAM内reuseを認定した後に、SSD read／write、昇格前予約、摩耗、privacyと実latencyを含めて評価する。

**完了条件：** prefixなしconcurrency 1のp95 TTFT／TPOT悪化を5%以内に抑え、代表concurrency 4でgoodput 20%以上改善を目標とする。prefix hitではprefill実行token数の減少とTTFT改善を確認する。未達なら標準有効化せず、改善するprofileだけを残す。

## P3 — 計測で選ぶkernel・量子化・speculative実行 🟠 [Next]

- 🟢 [Done] [P3共通E2E選択判定](P3-SELECTION.md)：kernel／量子化／speculative候補に独立3回、事前固定品質slice、memory、中央値5%改善、run範囲分離、TTFT／TPOT p95悪化5%以内を要求する。証拠不足ならbaseline保持。全trialとpolicyをreport hashへ結び付け、P1／P2前提と標準採用を別判定。CPU判定テストのみで高速化は未測定。
- 🟠 [Next] P0監査の未取得内部phase（queue／tokenize／prefill／decode／serialization／model load）をinstrumentationで分離し、CPU使用率・cache／eviction・energy取得と変動原因を調べる。autotuner確認契約とP3共通判定を実候補の独立確認測定とE2E回帰へ接続する。native build内容の固定も強化する。欠測をclient phase値で補わず、今回の速度差は性能認定しない。

依存：調査開始には対象経路のP0 baseline、標準採用にはP1の安定性と対象P2 workloadの回帰report。成果物は対象shapeに限定した高速化profileと安全なfallback。P2全体の完了前でも、baselineで支配時間が分かった経路の調査は並行できる。

- 🟠 [Next] P0 baseline取得後、GPU profilerで同期、CPU送信、dequantize、GEMV／GEMM、attention、KV copyの支配時間を特定する。decodeのmemory帯域律速とprefillのcompute律速を分けて検証し、律速を先に決め付けない。
- 🔴 [Later] upstream MLX／Metalの最適化を先に比較し、不足するhot pathに限りfused quantized GEMV／GEMM、RoPE／RMSNorm、paged／split-KV attentionを追加する。kernel単体の改善とE2E改善を別reportにする。
- 🔴 [Later] model／shape／SoC別にbounded autotuningを行う。compile・warm-up時間、p95、scratch使用量まで比較し、未測定shapeは既定kernelへ戻す。
- 🔴 [Later] weight 4／8-bit、KV量子化、mixed precisionを品質／速度／容量のPareto比較で選ぶ。perplexityだけでなく三言語coding、tool-use、long-context retrievalの劣化を検出する。
- 🔴 [Later] speculative decodeは小さい互換draft、GPU draft、利用可能な自己draft手法を候補として比較する。acceptance、verify cost、追加KV、全体latencyを測る。greedyの一致とsamplingの分布保存は別々に検証する。
- 🔴 [Later] ANEは固定shape encoderを優先する。draft導入には互換artifactとE2E改善の証拠を必要とし、CPU＋ANE＋GPUの同時利用自体を目標にしない。

既存Gemma CPU draft＋GPU verifierは[実測で約11.59倍低速](evaluation/gemma3-1b-cpu-4b-gpu-speculative-rss-2026-09-22.json)だったため既定有効化しない。より小さい互換draftやCore ML artifactが揃うまで、この構成の再試験を最優先作業にしない。

**完了条件：** 独立3 runでE2E中央値5%以上改善し、改善幅が測定ノイズを超える。対象profileのp95悪化5%以内、品質gate合格、memory budget内を満たす。未達の最適化は採用しない。
品質gateは評価前に固定し、deterministic correctness、task score、長文検索、構造化出力を分ける。量子化の許容差はsuiteごとに明記し、失敗sliceを総合平均で隠さない。

### Execution Plannerの拡張境界

- 🟠 [Next] 既存の[Execution Planner設計](Architecture-Decision-Apple-Execution.md)を基に、同一backend内のphase／operator設定とworkload別profileを実測へ結び付ける。model／state所有者はbackend processに維持し、upstreamのbatching、KV管理、kernelを先に比較する。
- 🔴 [Later] backendを跨ぐprefill／decode分割は、weight共有、KV／recurrent state形式、位置情報、同期、cancel／解放の契約を定義し、転送・変換・重複memory込みのE2E改善を証明できる場合だけ試す。DLPackなどの共有手段の存在だけでstate互換と判断しない。
- 🔴 [Later] Apple-native execution engineは既存backendで解決できない律速が反復して確認された場合の選択肢とする。NAX／ANEを含む候補は実device能力と認定済みkernelに基づき選び、SoC名だけで有効化しない。未保有hardwareは未評価とする。

English: Keep P0 first. Once a baseline exists, profile hot paths alongside P2;
promote optimizations only after stability and workload regression checks. Add
agent re-entry workloads and prefer existing backend policies. Cross-backend phase
splitting and a native engine remain conditional research, not committed replacements.

简体中文：P0仍为最高优先级。取得基线后，可与P2并行分析热点，但优化必须通过稳定性
及工作负载回归测试。增加Agent工具等待与再次进入推理的测试，优先使用现有后端策略。
跨后端阶段拆分和原生引擎仍是有条件的研究方向，不是已确定的替代方案。

## P4 — Mac別自動選択と配布認定 🟠 [Next]

依存：P1の安定性、P2／P3で合格したprofile。

- 🟢 [Done] [P4認定証拠gate](P4-CERTIFICATION.md)とrelease昇格の接続。全対応matrixセルのsource／署名artifact／identity、raw evidence hash、品質slice、P3性能判定、24時間soak、recovery／rollback／clean-machine／lock／license証拠を要求。欠落・失敗時は昇格を拒否する。profile activationのexact identity／thermal／memory fallbackと三言語診断契約も実装（runtime／SDK／UI接続は未完了）。認定済みセルはまだない。
- 🟠 [Next] 実collectorをこのgateへ接続し、P1／P2／P3の未達解消後に対象構成の独立性能認定・24時間soak・復旧／rollbackを実施する。workflowから証拠を取得できることは実機認定の完了を意味しない。

- 🔴 [Later] interactiveはp95 TTFT／TPOT、throughputはgoodput、省電力は取得可能なenergy/tokenを目的にする。未知hardwareではbounded calibrationを行い、発熱やmemory pressureを性能目的より優先する。
- 🔴 [Later] hardware／OS／model／backend fingerprint別の既知正常profileを配布し、変更時は再認定する。独立3回のqualificationは同じcandidateでも別process・別runで実施し、次release待ちを条件にしない。
- 🔴 [Later] 各サポート構成で24時間soakを行い、OS更新、backend更新、起動失敗、profile破損、rollbackをrelease gateへ組み込む。未保有Macには専用runnerが必要。
- 🔴 [Later] 英語・日本語・简体中文で起動、モデル互換性、メモリ不足、縮退理由、cancel／recoveryを同等に説明する。SDK／CLI／UIで状態とerror codeを統一する。
- 🔴 [Later] 再現可能な依存lock、SBOM／license、署名・notarization、clean machine install、アンインストールを検証する。署名workflowの存在と実署名artifactの検証を分ける。実行にはDeveloper ID／notary資格情報が必要。

**完了条件：** 対応matrixの各認定セルに品質・性能・24時間安定性reportとrollback手順がある。機種横断平均だけで「最速」と表示せず、対象条件と比較日を公開する。

## RAGとLoRAの段階的な完全対応

推論経路の安定化と並行して、**外部RAG接続 → 固定LoRA推論 → 内蔵検索 → LoRA学習 → 複数LoRA運用 → 統合認定**の順に進める。「完全対応」は公開したmodel／architecture／backend／量子化の対応matrix内で、取り込みから回答、学習から配信までのライフサイクルを完結できることとする。未知の全モデルへの無条件対応を意味しない。

### R0 — 外部検索との接続

- 🟢 [Done] RAG比較で点数改善と採用根拠を分離。後者は完全な前後identity、同一model artifact・既知依存version・evaluator source、正常shutdownを要求し、不足／変化の理由を保存。candidate prompt変更は許容、試験中変更は拒否。関連24 tests・Ruff成功。🟠 [Next] [現evaluator基準](evaluation/rag-identity-quality-m4-2026-10-05-r2.json)は厳密5/12・前後identity確認・正常終了合格。🟠 [Next] 基準から残る7ケースの品質改善を比較。

- 🟢 [Done] RAG collectorへlocal model全file／tokenizer及選択source 4件の試験前後hash、依存version比較を追加。関連23 tests・Ruff成功。artifact不足・変化・終了後hash失敗を拒否。[実機6ケース](evaluation/rag-identity-m4-2026-10-05.json)は品質・identity不変・正常終了合格。依存binary全体／ABI／試験中に戻された変更は未保証。🟠 [Next] identity付き12ケースbaselineと候補再評価。旧reportを遡及認定しない。

- 🟢 [Done] opt-in SIGUSR1 thread stack診断とcollectorの終了deadline後診断要求を実装。入力本文／locals／tensorを採取せず、signal記録と保存成功を区別。[実機6ケース](evaluation/rag-shutdown-diagnostics-m4-2026-10-05.json)・正常終了、CPU実processとmock timeout回帰成功。🟠 [Next] 終了遅延再現時のstack監査。今回未再現のため原因解消とは扱わない。

- 🟢 [Done] 新候補r5は回答不能が悪化し不採用。r6は6/12・既存5件維持だが独立再試験でshutdown deadline超過（kill/-9）のため不採用、r4 promptへ復元。🟢 [Done] HTTP collectorへgraceful／forced-killの独立記録を追加、deadline超過回帰を含む7 tests・Ruff成功。🟠 [Next] [再試験](evaluation/rag-quality-m4-2026-10-05-r6-repeat.json)の終了遅延と残る品質失敗を調査。

- 🟢 [Done] system roleなしの経路で資料後に元の質問・instructionを再提示し、完全template token予算へ含めた。[r4](evaluation/rag-quality-m4-2026-10-05-r4.json)／[独立再試験](evaluation/rag-quality-m4-2026-10-05-r4-repeat.json)とも5/12、baseline合格4件を維持し中国語長文が改善。全回帰1441 tests成功（11 skip）、Ruff成功。🟠 [Next] 残る7ケース、一般grounding・性能・変更後runtimeの認定。追加promptのcontextコストあり、旧identity証拠を転用しない。

- 🟢 [Done] RAG prompt候補2種を同一12ケースで実測（8/12・7/12）。回答不能の既存合格ケースが悪化したため不採用、runtime promptを復元。`compare_reports`で同一suite・完全な結果・正常終了と既存合格維持を検証し、総合点だけで採用しない。[比較監査](evaluation/rag-quality-candidate-comparison-2026-10-05.json)。🟠 [Next] 回帰なしの改善候補を検証。

- 🟢 [Done] 実HTTP collectorへ12ケースsuiteを接続し、[実モデルbaseline](evaluation/rag-quality-m4-2026-10-05.json)を保存。正常終了、厳密合格4/12。🟠 [Next] 回答形式5件、長文引用欠落2件、中国語の資料不足誤答1件を改善。一般grounding未認定、採点基準は維持。

- 🟢 [Done] `rag` CLIとPython APIで、検索済みチャンクから三言語の生成リクエストを構成する。資料のUTF-8 byte上限、丸ごとの除外、資料なし時のローカル回答不能、ループバックHTTP生成、出典ID／内容hashと参照IDの照合を実装。[利用手順](RAG-LORA.md)と`tests/test_rag.py`を参照。
- 🟢 [Done] 引用IDの存在確認と回答の事実性を分離する。`references_valid`でも`grounding_verified=false`とし、未知参照・引用なし・生成未完了を区別する。模擬HTTP試験は実モデルの品質認定に含めない。
- 🟢 [Done] 実tokenizerと完全chat templateによるcontext予算を実装。`rag --url --context-tokens`は同backend `/tokenize`で出力予約込み上限を確認し、低優先度資料を丸ごと除外、基本prompt超過・tokenize失敗なら生成拒否。system role非対応のGemmaは明示`--no-system-role`。三言語の[実tokenizer境界](evaluation/rag-real-token-budget-m4-2026-10-04-r2.json)と[実HTTP固定task](evaluation/rag-real-http-m4-2026-10-04-r2.json)6ケース・正常終了を確認。指定contextは固定model／server上限以下とし、一般grounding／性能認定へ広げない。
- 🟢 [Done] [三言語の合成品質suite](RAG-QUALITY-SUITE.md)：正しい資料・非空の情報不足資料・悪意ある追加資料・長い資料の12ケースと厳密な採点を実装。固定答案／引用／根拠資料保持／token予算／実生成を別判定し、欠測・誤答・余計な事実を拒否。CPU・既存模擬HTTP回帰20件成功。実モデル・一般grounding・性能は未認定。P1実行中のruntime編集・GPU負荷は追加していない。
- 🟠 [Next] 英語・日本語・简体中文の実モデル評価を追加。回答正確性、引用箇所との意味的整合性、資料不足時の回答不能、悪意ある資料への耐性を別指標で記録する。構成を固定したbaselineと比較し、runtime変更後のevidenceを再取得する。

### L0 — 一つの固定LoRAを安全に配信

- 🟠 [Next] 起動時に一つのimmutable adapterを読み込む経路を実装。base modelとadapterのhash、形式、rank、target modules、dtype、量子化との互換性を事前検証し、未対応の組み合わせは明示的に拒否する。
- 🟠 [Next] adapter分のメモリをadmissionへ計上し、model一覧／capability／qualificationにadapter identityを含める。adapterなし・ありの数値差と三言語品質を実MLXで検証する。単にload成功しただけでは認定しない。
- 🟠 [Next] 外部学習済みadapterと、別artifactとしてmergeしたモデルの導入手順を用意する。merge結果は独立modelとして再認定し、量子化前後の品質差とrollbackを確認する。

**R0／L0完了条件：** 対応matrixの少なくとも一構成で、出典付き回答と固定LoRA推論が実モデル試験に合格し、context超過・互換性不一致・メモリ不足を再現可能に処理できる。

### R1 — 取り込み・検索・回答を一体化

- 🔴 [Later] 文書取り込み、parser、chunking、安定したdocument／chunk IDと版管理、差分更新・削除を実装する。まずtext／Markdownから始め、PDF等は形式別の検証後に追加する。
- 🔴 [Later] embedding modelを生成modelと別capabilityとして管理し、`/v1/embeddings`、batching、次元・正規化・revision検証、メモリ競合の制御を実装する。embedding更新時には再indexを必須にする。
- 🔴 [Later] 永続vector index、keywordとのhybrid検索、reranker、重複排除、引用spanを実装する。検索recallと回答品質を別々に測定し、障害時の復旧・index migration・削除反映を検証する。
- 🔴 [Later] 検索前のtenant／ACL filterとrerank前の再検証、権限変更・資料削除時のcache無効化、監査を実装する。corpus revision、embedding revision、権限scopeをcache identityへ含め、別利用者の資料を混入させない。

### L1 — 学習と複数アダプター運用

- 🔴 [Later] 既存`RepairAdapter`の抽象契約を実MLX LoRA学習workerへ接続する。学習データ・seed・step・memory上限、checkpoint、cancel／resume、学習前後のperplexityと三言語task品質を記録する。QLoRAはbackend／architecture／量子化ごとの対応を検証する。
- 🔴 [Later] 学習と推論を別worker・budgetで管理し、GPU／unified memory競合を制御する。失敗時は既存配信を維持し、成果物の検証後に切り替え可能にする。
- 🔴 [Later] request単位のadapter指定、登録／削除／切り替え、GPU常駐上限とeviction、adapter別batch groupingを実装する。受付時にimmutable identityを固定し、処理中の差し替えを防止する。
- 🔴 [Later] KV／prefix cache keyにbase model、adapter hash、量子化、template等の実行identityを含める。異なるadapter間のKV共有を禁止し、同時実行・cancel・再起動で回答混入がないことを検証する。dynamic適用とmerge配信の品質・速度・メモリを比較する。

### RL2 — 完全対応の受け入れ条件

- 🔴 [Later] CLI／API／SDKで、文書登録・更新・削除→権限付き検索→引用付き回答、および学習→評価→登録→配信→rollbackを通して操作できるようにする。三言語で同じ状態とerror codeを説明する。
- 🔴 [Later] 対応matrix各セルで、LoRAとRAGを併用した品質、検索・rerank・prefill・decode別latency、p95 TTFT／TPOT、goodput、memoryを記録する。個別機能の合格を併用認定に代用しない。
- 🔴 [Later] 文書更新、adapter切り替え、同時利用、cancel、メモリ逼迫、worker障害を含む8時間試験とrelease前24時間試験を通す。権限・adapter間の情報混入ゼロ、削除反映、復旧・rollbackを必須条件とする。品質・性能閾値は認定前にsuiteへ固定する。

**English:** 🟢 [Done] External RAG request preparation, bounded local generation and reference-ID diagnostics are implemented; factual grounding and tokenizer limits are not certified. 🟠 [Next] Validate real-model RAG quality/context budgets and one immutable LoRA adapter. 🔴 [Later] Add ingestion, embeddings, hybrid retrieval, reranking, ACLs, actual LoRA/QLoRA training, multi-adapter serving with isolated KV caches, and combined 8/24-hour qualification. Full support is scoped to an explicit compatibility matrix.

**简体中文：** 🟢 [Done] 已实现外部RAG请求构建、有界本地生成和引用ID检查，尚未认证事实依据或token预算。🟠 [Next] 验证真实模型的RAG质量与上下文预算，并支持单个固定LoRA适配器。🔴 [Later] 完成文档导入、embedding、混合检索、重排、权限、LoRA／QLoRA实际训练、多适配器与KV隔离，以及组合场景的8／24小时认证。完整支持以明确的兼容矩阵为范围。

## 環境が整った時の作業候補 ⭕️ [Pending]

現在確認した筐体はMacBook Air／Apple M4／32 GiB。以下は本筐体だけでは認定できない条件として分離する。実装可能な共通runner・schema・fallbackは先に整備し、未保有機の性能値を推定で埋めない。

| 状態 | 作業候補 | 再開に必要な環境・確認内容 |
| --- | --- | --- |
| ⭕️ [Pending] | M5世代のNAX kernel認定とM4との比較 | 対象M5実機、対応OS／toolchain／backend。device能力検出、correctness、対象shapeのE2E・memory・fallbackを実測 |
| ⭕️ [Pending] | 16 GB級・64 GB以上・別SoCでの容量／thermal／24時間認定 | 各容量・SoCの実機runner。同一artifact、context、並列度、電源条件で比較し、容量超過を理由付きskip |
| ⭕️ [Pending] | M4 32 GiBのadmission予算を超えるモデル・context・batchの認定 | 必要なRAMを備えるMacと対象artifact。小さい量子化版が収まることを、元の構成の合格に代用しない |
| ⭕️ [Pending] | 複数Macの分散実行・通信込み性能 | 複数の対象Macと検証用network。通信・同期・障害時state回収を含めて評価 |
| ⭕️ [Pending] | ANE draft＋GPU verifierの実モデル比較 | 互換draftのCore ML artifactと変換・実行契約。M4のANEが利用可能でも互換artifactなしでは認定しない。既存encoder検証はこの保留に含めない |
| ⭕️ [Pending] | 署名・notarization済みreleaseとclean-machine install | Developer ID／notary資格情報と独立した検証環境。未署名のlocal buildとは別認定 |

English: `⭕️ [Pending]` means an unavailable test environment, with explicit resumption
requirements. Work runnable on this M4, including concurrency failures and long soaks,
remains `🟠 [Next]` or `🔴 [Later]`; it is not deferred merely because it takes time.

简体中文：`⭕️ [Pending]`表示缺少测试环境，必须注明恢复条件。当前M4可运行的工作，
包括并发故障排查和长时间测试，仍保留为`🟠 [Next]`或`🔴 [Later]`，不因耗时而搁置。

## 別トラックとして維持する作業 🔴 [Later]

| 作業 | 着手条件・必要な資源 |
| --- | --- |
| 一般VLM／Audio LLM | text経路の認定後。open-domain画像／音声品質、model固有projection、実artifactが必要 |
| 大容量MoE／hybrid／新architecture | 対応する実weightとメモリを持つMac。metadata対応を実推論成功と混同しない |
| 画像・音楽・動画生成 | LLM経路と別worker／budgetで維持。大容量artifact・専用quality suite・必要なMacを用意 |
| 独自数値形式・構造pruning | 既存量子化との比較で、品質維持とE2E上の必要性を示した後 |
| Multi-Mac | 2台以上の実Macと物理link測定。通信時間込みで単一Macを上回る用途があること。loopback試験は物理認定にしない |
| SSD／CPU offload | メモリ適合性を目的に評価し、最速経路とは別profileで公開 |

## 最初の実装順序

1. 🟢 [Done] P0のcapability matrixと比較manifestを作り、既存の実装・probe・実HTTP接続・未認定箇所を対応付けた。今回の基準scopeと欠測は[P0監査](P0-AUDIT.md)を参照する。
2. 🟠 [Next] 配置済み小型text modelで、MLX-LM directとvLLM-Metalの直接／daemon経由baselineを取る。
3. 🟠 [Next] cache計測とキャンセルqueue修正を実MLX・継続負荷で検証する。
4. 🟠 [Next] P1のcancel／memory／復旧を標準経路で確認し、8時間soakを通す。
5. 🟠 [Next] P2のbatchingとprefix reuseを一つずつ有効化し、同じsuiteで効果と回帰を比較する。
6. 🔴 [Later] P3のprofilerで支配時間を特定し、最も効果の大きい一箇所だけを最適化する。
7. 🔴 [Later] P4で機種を増やし、24時間認定と配布へ進む。

## English — execution summary

The goal is the fastest **qualified** route for each Apple Silicon Mac, model and workload, while preserving quality and sustained stability. This is a roadmap, not a claim of measured leadership.

- 🟢 [Done] Reuse the existing memory admission, scheduler, process isolation, profiling and rollback foundations. Existing Gemma and vision evidence covers limited configurations, not general certification. The original roadmap, including in-progress edits, is preserved in the linked history.
- 🟢 [Done] P0: [audit and reproducible baseline](P0-AUDIT.md), two backends × three runs × 102 requests; each backend completed 102 requests per language. Internal phases remain unavailable. Metal spread exceeds 5%; performance and general capabilities are not certified. Historical proxy/soak evidence retains its original scope.
- 🟠 [Next] P1: qualify the standard text route, cancellation, bounded queues, memory admission and failure recovery under an eight-hour mixed workload.
- 🟠 [Next] P2: validate backend-owned continuous batching, chunked prefill and actual KV reuse. Target 20% higher concurrency-four goodput with no more than 5% regression in single-request p95 TTFT/TPOT.
- 🟠 [Next] P3: the [common E2E selection gate](P3-SELECTION.md) is implemented; real collector integration and measured optimization remain unfinished. Require at least 5% reproducible E2E improvement beyond noise and verified stability before adoption.
- 🟠 [Next] P4: [release certification gate](P4-CERTIFICATION.md) is connected to draft promotion; no matrix cell is certified yet. Complete real 24-hour soaks, profile selection, rollback, signing and clean-machine installation. Additional hardware and credentials remain prerequisites.

All targets are prospective. Unsupported combinations, insufficient samples and failed candidates remain visible; no benchmark result is extrapolated to all Macs or models.

## 简体中文 — 执行摘要

目标是在保持质量和持续稳定性的前提下，为每台Apple Silicon Mac、每个模型和工作负载选择经过认证的最快路径。本路线图不代表已经取得性能领先。

- 🟢 [Done] 复用现有内存准入、调度器、进程隔离、性能记录和回滚基础。Gemma及视觉模型的证据仅适用于已测配置。包含未提交修改的旧路线图已完整保存。
- 🟢 [Done] P0：[路径审计和可复现基准](P0-AUDIT.md)，两个后端各3次、每次102个请求，每个后端各语言完成102个请求。内部计时仍缺测，Metal波动超过5%，不认证性能领先或通用能力。旧proxy／soak证据保留原适用范围。
- 🟠 [Next] P1：认证标准文本路径，完成取消、有界队列、内存准入、故障恢复及8小时混合负载测试。
- 🟠 [Next] P2：验证后端拥有的连续批处理、分块prefill和真实KV复用。目标为并发4时有效吞吐提升20%，单请求p95 TTFT／TPOT退化不超过5%。
- 🟠 [Next] P3：[共用E2E选择判定](P3-SELECTION.md)已实现；实际采集器对接及性能优化仍未完成。只有质量、稳定性通过且端到端提升至少5%、超过测量噪声时才采用。
- 🟠 [Next] P4：[发布认证gate](P4-CERTIFICATION.md)已接入草稿提升流程，目前没有已认证单元。仍需实际24小时测试、profile选择、回滚、签名和独立环境安装。额外硬件和资格信息仍是前提条件。

以上数字均为未来验收目标。保留不支持的组合、样本不足和失败候选，不把局部结果推广到所有Mac或模型。

## 一次資料と更新方針

2026-10-05：[公式更新確認と採用判断](upstream-review-2026-10-05.md)。P1 r5実行中はruntime／依存を維持。
Metal 0.30.0のMLX 0.32.1 ABI固定、MLX 0.32.3のbuffer／stream／GQA修正を分けて比較候補にした。
公開release／tagの存在だけで現projectの能力・性能・互換性を認定しない。

2026-09-26参照。[最新確認と採用判断](upstream-review-2026-09-26.md)。以下は比較候補の技術資料であり、このrepositoryでの性能認定の証拠ではない。実装時は参照commitとlicenseを固定する。

- [MLX-LM公式](https://github.com/ml-explore/mlx-lm)：Apple Silicon向け生成・量子化・streamingの比較基準。
- [vLLM-Metal公式](https://github.com/vllm-project/vllm-metal)：vLLMのApple Silicon plugin。使用buildの実効機能は個別に確認する。
- [llama.cpp公式](https://github.com/ggml-org/llama.cpp)：Metal／GGUF経路の比較対象。
- [vllm-mlx upstream](https://github.com/waybarrios/vllm-mlx)：continuous batchingとcache設計の比較対象。upstreamの性能値を本projectの実測として引用しない。

各開発サイクルで本書の状態・証跡・次の作業を更新する。詳細な実験履歴はevaluationと個別reviewへ置き、roadmapを実装履歴の追記だけで肥大化させない。

## 定期評価 / Recurring evaluation / 定期评估 — 2026-10-09

🟢 [Done] [短時間基準の再測定](optimization-2026-10-09.md)：144件算術品質合格、warmup SLOは2/3。コード変更なし、速度差を改善認定しない。🟠 [Next] 同条件反復でwarmup境界とprefill待ちを確認。

English: Unchanged baseline repeated; all 144 arithmetic checks passed, warmup SLO 2/3. No speedup attribution. Next: independent repeats and prefill wait diagnosis.

简体中文：重测未修改基线，144件算术质量全部通过，warmup SLO为2/3；不认证速度提升。下一步：同条件独立重复，定位prefill等待。
