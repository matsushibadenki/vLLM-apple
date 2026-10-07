# Event history replay optimization — 2026-10-07

## 日本語

[Done] 指示書の「必要な1件だけを処理する」「全体走査を避ける」を適用し、EventBusの履歴再送を改善した。
これはcontrol planeの履歴取得の測定であり、HTTP転送・Swift UI・推論速度・P1の長時間SLOの認定ではない。
標準保持件数は256のまま。10,000件は拡張設定の負荷試験である。

**Problem:** SSE購読者が保存済みのイベントを再送する際、同じ履歴を何度も読む。

**Root cause:** 1件取得するたびにdequeを先頭から探索するため、N件の再送はO(N²)。
探索中はpublisherと共有のCondition lockを保持する。競合による待ち時間そのものは今回未測定。

**Evidence:** [変更前](evaluation/event-history-before-2026-10-07.json)の計測は1,000件で約6.4 ms、
10,000件で約567 ms。profileでも履歴走査generatorが支配的だった。
[初回変更後](evaluation/event-history-after-2026-10-07.json)は再送を短縮したが、一部tail latencyが増加したため、
[同一processでの交互比較](evaluation/event-history-comparison-2026-10-07.json)を追加した。初回結果は削除していない。

比較はM4の非debug CPython 3.10.18で、前後を交互に9回、100／256／1,000／10,000件の4規模で実施。
P1 r9に記録された実際のresource payloadを使い、毎回のsequence・payload一致を検証した。
入力、実装、benchmarkのSHA-256とprofileをJSONに保存している。
変更前sourceはcommit `793f741f0dec28fa218153e0e57f462240b530ff` の `vllm_apple/events.py` と一致する。
microbenchmarkは専用・隔離された実行環境ではなく、最大値や短時間のtailには環境変動が残る。

**Changed files:** `vllm_apple/events.py`、`tests/test_events.py`、`scripts/benchmark_event_history.py`、本報告、ROADMAP、前回報告への相互参照。

**Change:** sequenceから直接参照できるlist ringへ変更。初期は空で、保持上限まで増やし、以後は古いslotを上書きする。
欠落通知、heartbeat、subscriber上限、payload、lock、yield時のlock解放を維持する。

**Why it should improve performance:** 1件取得はO(1)、全履歴の再送はO(N)。
追加index辞書、thread、polling、payloadコピーは導入しない。

**Before / After:** wall時間は全履歴再送の9回中央値、p95／p99／最大値は1件取得の値。
全履歴のwall／CPU計測には時刻記録と正しさ検証も含む。各規模の取得sample数は900／2,304／9,000／90,000。

| 件数 | 全再送中央値 ms 前→後 | 差分 ms（割合） | 1件 p95 µs 前→後 | 1件 p99 µs 前→後 | 1件 最大 µs 前→後 |
| --- | --- | --- | --- | --- | --- |
| 100 | 0.157041→0.060000 | -0.097041 (-61.79%) | 2.042→0.500 | 2.167→0.750 | 32.292→2.708 |
| 256 | 0.569167→0.137500 | -0.431667 (-75.84%) | 3.417→0.417 | 3.541→0.500 | 3.625→7.792 |
| 1,000 | 6.416959→0.528833 | -5.888126 (-91.76%) | 11.458→0.417 | 11.917→0.500 | 22.625→31.750 |
| 10,000 | 599.766167→5.347375 | -594.418792 (-99.11%) | 111.875→0.458 | 133.750→1.083 | 4707.459→621.083 |

全再送中央値とp95／p99は全規模で改善。一方、256／1,000件の単発最大時間は増えたため、最悪遅延の改善は認定しない。

**CPU impact:** 全再送CPU時間中央値は256件で0.571→0.137 ms、10,000件で599.343→5.339 ms。
登録側にはmoduloとlist更新が加わる。各規模9,000回の登録測定で、256件時の中央値は1.541→1.583 µs（+0.042 µs、約2.7%）、
p95は1.625→1.667 µs、p99は1.708→1.916 µs。登録側も一律高速化したとは扱わない。

**GPU impact:** GPU処理なし。GPU時間、token/s、推論E2Eへの効果は未測定。

**Memory impact:** 保持するイベント件数・payloadは同じ。container単体の浅いsizeは256件で2,736→2,200 bytes、
10,000件では82,992→85,176 bytes（+2,184 bytes、約2.6%）。listの余剰確保を含み、全process RSSや総allocation量ではない。
上書き後に古いpayloadへの参照が解放されることを検証した。モデルメモリ削減はない。

**I/O impact:** serialization、ネットワーク送信件数、永続化に変更なし。I/O実測は対象外。

**Energy impact:** CPU上の不要な走査を削減した。joule／wattは未測定であり、省電力率は認定しない。

**Correctness verification:** 対象8 tests成功。複数回のwrap、速い／遅い購読者、gap後の追加欠落、
空／未来cursorのheartbeatと通知、購読者停止中のpublish、保持上限、上書き解放を検証。
「1件取得に履歴全走査をしない」こともテストで固定した。
全Python回帰1,463 tests成功（11 skip）。既存の認証付きHTTP SSEとschema検証を含む。Ruff、diff whitespace check成功。

**Regression risk:** 連続sequenceとslot対応が正しさの前提。wrapと購読者ごとのcursorを回帰試験で覆う。
登録側の小さな追加処理と大規模時のcontainer size増加は上記の通り。
ネットワーク、UI入力遅延、同時多数購読時のlock競合は今後の実利用profileが必要。

**Keep / Revert:** **Keep**。標準256件でも再送中央値・p95／p99・CPUの減少を確認し、
保持件数を増やした場合の二乗増加を小さな実装変更で解消した。
登録側の+0.042 µsと10,000件containerの+2.6%を許容する。推論の速度向上やP1失敗の解消とは扱わない。

## 再現

```sh
git show 793f741f0dec28fa218153e0e57f462240b530ff:vllm_apple/events.py > /tmp/events-before.py
python3 scripts/benchmark_event_history.py --baseline-source /tmp/events-before.py --output /tmp/event-history-new.json
python3 -m unittest tests.test_events -v
```

baseline sourceは実行されるPythonコードなので、信頼できる自repoのsourceだけを渡す。出力は新しいpathを指定する。

## English

[Done] Replaced repeated EventBus history scans with direct sequence-indexed ring access, preserving ordering,
gap notifications, heartbeats and bounded retention. Nine alternating comparisons used a captured resource payload.
Default 256-event replay median fell from 0.569 to 0.138 ms (75.84%); 10,000-event replay fell from 599.766 to 5.347 ms (99.11%).
p95/p99 improved across all four sizes; single-event maxima did not improve at every size.
At 256 events, publish median increased by 0.042 µs (2.7%); at 10,000 events, shallow container size increased 2.6%.
Keep the small change with these tradeoffs. All 1,463 Python tests passed (11 skipped), including authenticated SSE/schema checks; Ruff passed.
These measurements concern history access, not network/UI latency, inference speed or electrical energy.
[Next] P1 requalification in the 09:00 JST campaign. [Later] Profile supervisor polling and Swift UI/SSE under actual use.
[pending] Electrical-energy measurement needs an authorized sampler or external meter.

## 简体中文

[Done] 将EventBus重复遍历历史改为按sequence直接访问ring，保留顺序、丢失通知、heartbeat和有界存储。
使用已记录的实际resource payload，前后交替比较9次。默认256条重放中位耗时0.569→0.138 ms（减少75.84%）；
10,000条599.766→5.347 ms（减少99.11%）。4种规模的p95／p99均改善，但单次最大延迟并非全部改善。
256条时发布中位耗时增加0.042 µs（2.7%）；10,000条时container浅层大小增加2.6%。接受这些代价，保留改动。
全部1,463项Python回归成功（跳过11项），包含认证SSE／schema检查；Ruff通过。
这些是历史访问测量，不代表网络／UI延迟、推理速度或电能改善。
[Next] 日本时间09:00开始的P1长期复测。[Later] 实际使用时的supervisor polling及Swift UI／SSE profile。
[pending] 实际电能测量需要授权sampler或外部仪表。
