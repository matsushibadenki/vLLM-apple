# Optimization report — RSS sampling / RSS計測 / RSS采样

## 日本語

[ソフトウェア最適化指示書](ソフトウェア最適化指示書.md)を基に、構造確認→baseline→profile→単一修正→再計測の順で実施した。
今回は**推論を観測する処理そのものの不要なprocess起動**を削減した。長時間試験は実施せず、朝9:00 JST開始の方針を維持する。

### Architecture確認と候補の選別

| 境界・構成 | 確認した経路 | 判断 |
| --- | --- | --- |
| frontend／backend | SwiftUIのMainActor AppModel→Swift async client→HTTP／Unix socket→Python API | UI frame時間は未測定。主観でUI最適化しない |
| control／execution | API→RuntimeService→scheduler／memory admission→別processのMLX／Metal | この変更でGPU kernelやモデル品質には手を加えない |
| main／event loop | bounded HTTP thread、event condition、backend supervisor、MLX queue | idle候補は既存実測あり。supervisor pollingは別の未計測候補 |
| CPU／GPU | MLX tokenizer・batch scheduler→GPU tensor計算、SSEで小さい応答を返す | GPU dispatch／同期の実時間は未測定。host wall時間で代用しない |
| cache／allocation | prompt LRU、allocator cache、semantic state、token-count single-flight | 前回cache削減実測は別scope。今回はallocator設定を変えない |
| I/O／serialization | JSON制御、SSE応答、profile／資格情報の原子的file保存 | fsyncは永続化の正しさに関わるため計測なしに削除しない。標準text経路にDB queryは見つからない |
| observability | phase_probeが各要求の前後と50 msごとにRSSを取得 | process起動が支配時間になっていることをprofileで確認し、今回の修正対象にした |
| 履歴／再計算 | EventBusの履歴走査、Swift SSE buffer処理 | 後続の[EventBus実測・改善](EVENT-HISTORY-OPTIMIZATION-2026-10-07.md)で履歴走査を解消。Swift処理はprofile候補として残す |

これは境界と主要経路の構造調査であり、全module・全UI操作の包括的profile完了ではない。

### 変更の報告

**Problem:** 並列benchmarkの各要求がRSS監視threadを持ち、50 msごとのsampleと要求前後に`/bin/ps`を起動していた。
監視そのものがCPU、process生成、pipe通信、文字列変換を発生させる。

**Root cause:** 欲しいのは一つのPIDのRSS値だが、外部processの起動・wait・pipe出力取得を毎回実行していた。

**Evidence:** [修正前profile](evaluation/rss-sampler-before-2026-10-07.json)は100回で約184 ms、うち約183 msがsubprocess.run。
[前後比較](evaluation/rss-sampler-comparison-final-2026-10-07.json)は8／64／256／512 MiBの実child allocationで各方式300回ずつ測定。
独立した要求ごとのsampling頻度、RSSのbyte単位、失敗扱いは変更していない。

**Changed files:** `vllm_apple/process_memory.py`、`vllm_apple/phase_probe.py`、`tests/test_process_memory.py`、
`scripts/benchmark_rss_sampling.py`、本報告とROADMAP／P1説明。

**Change:** macOSでは公開`proc_pidinfo(PROC_PIDTASKINFO)`から現在のRSSを読む。
API handleだけを再利用し、RSS値はcacheしない。呼出ごとに独立したbufferを使う。
非macOSまたはAPIのload不可時は従来のps方式へfallback。APIがpermission／process消滅／部分readを返した場合は、
ゼロや古い値に置き換えず`rss_unavailable`として扱う。

**Why it should improve performance:** 必要なOS問い合わせを残し、sampleごとのprocess生成・pipe・ps出力parseを取り除く。

**Before / After:** 同じ安定したchildのRSSは全20組で完全一致（最大差0 bytes）。以下はsampler単体のwarm測定。
Pythonは非debug build。各条件300 samplesのp99は参考値。測定中のsubprocess.runは起動件数計測のため同じwrapperを使用した。

| child allocation | median before→after | p95 before→after | p99 before→after |
| --- | --- | --- | --- |
| 8 MiB | 1.683271→0.000917 ms | 1.916291→0.001000 ms | 2.465292→0.002333 ms |
| 64 MiB | 1.824542→0.000958 ms | 2.802333→0.001084 ms | 3.717458→0.002292 ms |
| 256 MiB | 1.768126→0.000958 ms | 2.513000→0.001166 ms | 3.800625→0.002459 ms |
| 512 MiB | 1.850396→0.000917 ms | 2.621917→0.001083 ms | 3.312167→0.003166 ms |

median取得時間は約99.95%減少。最大時間は前後ともJSONへ保存している。
この数字を推論のE2E・token/sの改善率として表示しない。

**CPU impact:** 1200 readsで親CPU時間908.452→1.484 ms、計測用child CPU時間約1210.847→0 ms。
**GPU impact:** GPU処理変更なし。GPU時間・同期への効果は未測定。
**Memory impact:** RSSの値は同じ。96-byteの呼出bufferと一つのlibrary handleを使用。モデルのメモリ削減を意味しない。
**I/O impact:** sample用subprocess起動1200→0。各回のps用pipe／文字列parseも不要。OSの情報取得callは残る。
**Energy impact:** 不要なprocess処理は削減。joule／wattは未測定で、省電力率は認定しない。

**Correctness verification:** 4規模でRSS一致、終了済みPIDの拒否、100件の並列buffer分離、権限拒否・部分readで失敗、
fallbackのKiB→bytes変換とNone PIDを検証。[短時間実MLX HTTP試験](evaluation/rss-sampler-http-smoke-m4-2026-10-07.json)は
warmup 3/3、長文12/12、並列短文30/30の品質・SLOに合格し、cancel／fault確認、正常停止、identity不変も成功。

全Python回帰1457 tests成功（11 skip）、Ruff成功。最初のsandbox内socket試験は権限制限で失敗し、
localhostを利用できる環境で対象試験・全回帰を再実行して成功した。

**Regression risk:** Darwin ABIとprocessアクセス権に依存する。SDK headerのsize／field定義を確認し、read byte数不一致は拒否。
将来のAPI非互換を自動認定しない。他OSの実機試験は本Macでは未実施。DLLのload失敗時には旧経路を維持する。
sampling対象processは同時に変動し得るため、RSS値が一致する検証には入力待ちのchildを使った。

**Keep / Revert:** **Keep**。小さい変更でRSSの意味・頻度を維持し、計測負荷の大幅な減少と外部process起動ゼロを確認した。
過去のP1失敗の全原因が解消したとは扱わない。次回の9時開始試験は新collectorを含む新identityで記録する。

### 再現と出典

```sh
python3 scripts/benchmark_rss_sampling.py --output docs/evaluation/rss-sampler-new.json
python3 -m unittest tests.test_process_memory tests.test_phase_probe -q
```

出力pathは未作成のものを使う。macOS実測には対象PID参照とlocalhost試験が可能な権限が必要。
[Apple XNU公開header](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/sys/proc_info.h)と
配置SDKの`sys/proc_info.h`／`libproc.h`を2026-10-07に確認した。headerはAPSL 2.0、実装sourceの転載は行わず公開ABIへのbindingを実装。
参照URLのmainを検証済みversionの代用にせず、本Macの実APIでbyte数・RSS一致を検証した。

[Next] P1の9時開始再試験で品質・SLO・全worker epoch資源を再確認。旧collectorとの推論速度の優劣は別の比較が必要。
[Done] 後続の[EventBus計測](EVENT-HISTORY-OPTIMIZATION-2026-10-07.md)で履歴全走査を削減。RSS改善とは別の単一変更として比較・検証した。
[Later] supervisor polling、Swift main thread／SSE処理の実利用profile。
[pending] 管理者権限の電力samplerまたは外部計器によるjoule/request・idle watt。

## English

[Done] Following the supplied measurement-first guide, audited the main control/execution/UI boundaries and profiled RSS sampling.
The dominant sampler cost was launching `ps` for every request-boundary and 50 ms memory sample.
A small macOS binding now reads current RSS through `proc_pidinfo`; only the library handle is cached.
Non-macOS/API-load-unavailable cases retain the old fallback. Denied, partial and exited-PID reads fail rather than returning zero or stale data.

Four live child allocations (8–512 MiB), 300 reads per method per size, produced identical RSS in every paired check.
Median sampler latency fell from about 1.7–1.9 ms to 0.001 ms; p95/p99/max are in the linked report.
Across 1200 reads, sampler subprocesses fell from 1200 to zero. This is a measurement-overhead improvement, not a certified inference speedup.
GPU work/model memory are unchanged; electrical energy remains unmeasured. The short real-MLX smoke passed all 45 quality/SLO checks and clean shutdown.
Full regression passed 1,457 tests (11 skipped). Keep the change; continue P1 long-run qualification only in a campaign starting at 09:00 JST. ABI/permission failures remain explicit risks.

## 简体中文

[Done] 按指示书先调查主要控制／执行／UI边界，再profile RSS采样。主要成本是每次请求前后及每50 ms都启动ps。
macOS现在通过proc_pidinfo直接获取当前RSS，只复用library handle，不缓存RSS值。
非macOS或API无法加载时保留旧方式；权限拒绝、部分读取和PID退出均报错，不返回零或旧值。

4种实际child allocation（8–512 MiB），每种每方式300次读取，所有配对RSS完全一致。
采样中位延迟约1.7–1.9 ms降至0.001 ms，p95／p99／最大值记录于报告。
1200次读取的采样subprocess从1200降为0。这是测量开销优化，不是推理速度认证。
GPU计算和模型内存不变，电能未测量。短期实际MLX试验45/45质量及SLO通过并正常停止。
全回归1457项通过（11项跳过）。保留此改动；长期P1试验仍仅在日本时间09:00开始，继续保留ABI／权限失败的明确处理。
